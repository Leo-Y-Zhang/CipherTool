"""Local English-language scoring.

This is the single most important module in the toolkit: every automatic
solver works by proposing decryptions and asking "does that look like
English?". Everything here is computed on this machine from resources that
live in this source tree. Nothing is downloaded, and no text ever leaves the
process.

Where the model comes from
--------------------------
``data/corpus_*.txt`` holds roughly twenty-four thousand words of ordinary
English prose written for this project (narrative, dialogue, correspondence,
expository writing, history and everyday journalism). At first use we reduce
that prose to a stream of about one hundred and twenty thousand A-Z letters
and count how often each letter, pair, triple and quadruple occurs.

The model
---------
We score a candidate plaintext as the log probability of its letter sequence
under an *interpolated Markov chain of order three* -- each letter is
predicted from the three letters before it:

    score(text) = sum over i of  log10 P(x_i | x_{i-3}, x_{i-2}, x_{i-1})

Raw quadgram frequencies would be useless on a corpus this size: of the
26^4 = 456,976 possible quadgrams we only ever see a few tens of thousands,
so most real English quadgrams would score as "impossible". We therefore
smooth by backing off to shorter contexts, using add-K interpolation:

    P1(d)         = (C1(d) + 1) / (N + 26)
    P2(d | c)     = (C2(cd)   + K * P1(d))       / (C1(c)   + K)
    P3(d | bc)    = (C3(bcd)  + K * P2(d | c))   / (C2(bc)  + K)
    P4(d | abc)   = (C4(abcd) + K * P3(d | bc))  / (C3(abc) + K)

Read the P4 line as: "trust the quadgram count if we have seen its trigram
context often; otherwise fall back towards what the trigram model believes."
K is the number of imaginary observations we give the fallback, so a context
seen K times splits its belief evenly between the two orders. K = 6 was
chosen by measuring the model on held-out English (see ALGORITHMS.md); the
result is not sensitive to it.

A second, independent signal
----------------------------
Letter statistics alone cannot tell true English from a near-miss key that
happens to produce English-ish letter runs. So we also measure *word
coverage*: what fraction of the candidate's letters can be cut into words
from our lexicon, computed by dynamic programming over every possible
segmentation. The two signals are reported separately and both are required
before a candidate is labelled "strong" (see ``candidates.py``).

Performance
-----------
The order-3 table has 456,976 entries and takes a fraction of a second to
build, so it is built lazily on first use. ``analyse`` never touches it.
Solvers that score millions of candidates use :meth:`EnglishScorer.table`
and :meth:`EnglishScorer.encode` to work with flat integer indices instead
of strings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from .reference import COMMON_WORDS, IMPLAUSIBLE_DIGRAPHS

ALPHABET_SIZE = 26

#: Imaginary observations given to the lower-order model when interpolating.
SMOOTHING_K = 6.0

#: Longest word we will try to recognise during coverage segmentation.
MAX_WORD_LENGTH = 18

DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def corpus_files() -> list[Path]:
    """Every prose file that makes up our training corpus."""
    return sorted(DATA_DIR.glob("corpus_*.txt"))


def load_corpus_text() -> str:
    """Concatenate the corpus files into one string.

    Raises a clear error rather than silently scoring against nothing if the
    data files are missing from an installation.
    """
    files = corpus_files()
    if not files:
        raise FileNotFoundError(
            f"No corpus files found in {DATA_DIR}. The English scoring model "
            "cannot be built. Reinstall the package with its data files."
        )
    return "\n".join(path.read_text(encoding="utf-8") for path in files)


def _letters(text: str) -> str:
    return "".join(ch for ch in text.upper() if "A" <= ch <= "Z")


def _words(text: str) -> list[str]:
    """Split prose into uppercase alphabetic words (apostrophes dropped)."""
    out: list[str] = []
    current: list[str] = []
    for ch in text.upper():
        if "A" <= ch <= "Z":
            current.append(ch)
        elif ch == "'":
            continue  # DON'T -> DONT, which is how ciphertext will read
        else:
            if current:
                out.append("".join(current))
                current = []
    if current:
        out.append("".join(current))
    return out


# ---------------------------------------------------------------------------
# Score breakdown
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreBreakdown:
    """Transparent report of how a candidate was scored."""

    length: int
    ngram_total: float
    ngram_per_letter: float
    word_coverage: float
    longest_word: str
    words_found: int
    implausible_digraphs: int
    combined: float

    def describe(self) -> str:
        """One-line human-readable summary of the score components."""
        return (
            f"letters={self.length}  "
            f"ngram/letter={self.ngram_per_letter:.3f}  "
            f"word-coverage={self.word_coverage:.0%}  "
            f"words={self.words_found}  "
            f"longest={self.longest_word or '-'}  "
            f"odd-digraphs={self.implausible_digraphs}"
        )


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------


class EnglishScorer:
    """An order-3 interpolated letter model plus a word-coverage measure.

    Construct through :func:`default_scorer`, which caches one instance per
    process. Building the model twice is wasted work, not an error.
    """

    def __init__(self, corpus_text: str | None = None, k: float = SMOOTHING_K) -> None:
        self.k = k
        text = corpus_text if corpus_text is not None else load_corpus_text()
        self._letters = _letters(text)
        if len(self._letters) < 5000:
            raise ValueError(
                "Corpus is too small to build a usable language model "
                f"({len(self._letters)} letters). At least 5000 are needed."
            )

        self._count1 = [0] * ALPHABET_SIZE
        self._count2 = [0] * (ALPHABET_SIZE**2)
        self._count3 = [0] * (ALPHABET_SIZE**3)
        self._count4 = [0] * (ALPHABET_SIZE**4)
        self._tally()

        # Lower-order log tables are cheap; build them eagerly.
        self._log1 = self._build_order1()
        self._log2 = self._build_order2()
        self._log3 = self._build_order3()
        self._log4: list[float] | None = None  # built on demand

        self._lexicon = self._build_lexicon(text)
        self._lexicon_by_length: dict[int, set[str]] = {}
        for word in self._lexicon:
            self._lexicon_by_length.setdefault(len(word), set()).add(word)

    # -- corpus counting ---------------------------------------------------

    def _tally(self) -> None:
        """Count letters, pairs, triples and quadruples in one pass.

        ``index`` is carried forward so each new letter costs one multiply and
        one add rather than four string slices.
        """
        letters = self._letters
        c1, c2, c3, c4 = self._count1, self._count2, self._count3, self._count4
        values = [ord(ch) - 65 for ch in letters]

        for position, value in enumerate(values):
            c1[value] += 1
            if position >= 1:
                c2[values[position - 1] * 26 + value] += 1
            if position >= 2:
                c3[(values[position - 2] * 26 + values[position - 1]) * 26 + value] += 1
            if position >= 3:
                c4[
                    ((values[position - 3] * 26 + values[position - 2]) * 26
                     + values[position - 1]) * 26 + value
                ] += 1

    # -- model construction ------------------------------------------------

    def _build_order1(self) -> list[float]:
        """log10 P(d), Laplace-smoothed so no letter is impossible."""
        total = sum(self._count1)
        return [
            math.log10((self._count1[d] + 1) / (total + ALPHABET_SIZE))
            for d in range(ALPHABET_SIZE)
        ]

    def _build_order2(self) -> list[float]:
        """log10 P(d | c), interpolated with the order-1 model."""
        k = self.k
        prob1 = [10**value for value in self._log1]
        table = [0.0] * (ALPHABET_SIZE**2)
        for c in range(ALPHABET_SIZE):
            context_total = self._count1[c]
            denominator = context_total + k
            base = c * 26
            for d in range(ALPHABET_SIZE):
                numerator = self._count2[base + d] + k * prob1[d]
                table[base + d] = math.log10(numerator / denominator)
        return table

    def _build_order3(self) -> list[float]:
        """log10 P(d | bc), interpolated with the order-2 model."""
        k = self.k
        prob2 = [10**value for value in self._log2]
        table = [0.0] * (ALPHABET_SIZE**3)
        for bc in range(ALPHABET_SIZE**2):
            context_total = self._count2[bc]
            denominator = context_total + k
            c = bc % 26
            lower_base = c * 26
            base = bc * 26
            for d in range(ALPHABET_SIZE):
                numerator = self._count3[base + d] + k * prob2[lower_base + d]
                table[base + d] = math.log10(numerator / denominator)
        return table

    def _build_order4(self) -> list[float]:
        """log10 P(d | abc), interpolated with the order-3 model.

        This is the table the solvers actually use. It has 456,976 entries,
        which is why it is built only when something asks for it.
        """
        k = self.k
        prob3 = [10**value for value in self._log3]
        count3, count4 = self._count3, self._count4
        table = [0.0] * (ALPHABET_SIZE**4)
        log10 = math.log10
        for abc in range(ALPHABET_SIZE**3):
            denominator = count3[abc] + k
            lower_base = (abc % (26 * 26)) * 26  # the trigram context "bc?"
            base = abc * 26
            for d in range(ALPHABET_SIZE):
                numerator = count4[base + d] + k * prob3[lower_base + d]
                table[base + d] = log10(numerator / denominator)
        return table

    def table(self) -> list[float]:
        """The flat order-3 log-probability table, built on first request.

        Index a quadgram ``abcd`` (letters as 0..25) as
        ``((a * 26 + b) * 26 + c) * 26 + d``.
        """
        if self._log4 is None:
            self._log4 = self._build_order4()
        return self._log4

    # -- lexicon -----------------------------------------------------------

    def _build_lexicon(self, corpus_text: str) -> frozenset[str]:
        """Our word list: hand-typed common words plus our corpus vocabulary.

        Both sources are inside this repository, so the whole lexicon is
        auditable and nothing was downloaded.
        """
        words = set(COMMON_WORDS)
        for word in _words(corpus_text):
            if 1 <= len(word) <= MAX_WORD_LENGTH:
                words.add(word)
        # Single letters other than A and I are not English words and would
        # let coverage scoring accept anything at all.
        words = {w for w in words if len(w) > 1 or w in {"A", "I"}}
        return frozenset(words)

    @property
    def lexicon(self) -> frozenset[str]:
        """Every word the coverage measure and pattern matcher recognise."""
        return self._lexicon

    # -- scoring -----------------------------------------------------------

    def encode(self, letters: str) -> list[int]:
        """Turn an A-Z string into 0..25 integers for the hot-loop API."""
        return [ord(ch) - 65 for ch in letters]

    def score_values(self, values: Sequence[int]) -> float:
        """Score an already-encoded letter sequence.

        The first three letters have no full context, so they are scored with
        the longest context available. This matters for short texts and keeps
        scores comparable across candidates of the same length.
        """
        count = len(values)
        if count == 0:
            return 0.0
        log1, log2, log3 = self._log1, self._log2, self._log3
        total = log1[values[0]]
        if count > 1:
            total += log2[values[0] * 26 + values[1]]
        if count > 2:
            total += log3[(values[0] * 26 + values[1]) * 26 + values[2]]
        if count > 3:
            log4 = self.table()
            index = ((values[0] * 26 + values[1]) * 26 + values[2]) * 26
            for position in range(3, count):
                index += values[position]
                total += log4[index]
                # Slide the four-letter window: drop the oldest letter and
                # make room for the next one. 17576 = 26^3.
                index = (index % 17576) * 26
        return total

    def score(self, text: str) -> float:
        """Total log10 probability of *text* under the letter model.

        Non-letters are ignored. Higher (closer to zero) is more English-like.
        """
        return self.score_values(self.encode(_letters(text)))

    def normalised(self, text: str) -> float:
        """Score per letter, so texts of different lengths compare directly.

        Measured on held-out English (see tests/test_scoring.py): ordinary
        English lands near -0.89, a key with two letters wrong near -1.21,
        and anything genuinely wrong -- a bad shift, a scrambled alphabet,
        uniformly random letters -- near -2.75. The confidence thresholds in
        ``candidates.py`` sit in those gaps.
        """
        letters = _letters(text)
        if not letters:
            return float("-inf")
        return self.score_values(self.encode(letters)) / len(letters)

    # -- word coverage -----------------------------------------------------

    def word_coverage(self, text: str) -> float:
        """Fraction of letters explainable as a run of known words.

        Solved by dynamic programming: ``best[i]`` is the greatest number of
        letters in ``text[:i]`` that can be covered by non-overlapping known
        words. Working left to right, for each end position we try every word
        length and take the best. Unknown stretches are simply skipped, which
        is what lets a partly-correct decryption score partial credit.

        Two-letter words are only worth half a letter each, because a text of
        random letters accidentally contains a lot of AT, IN and SO.
        """
        letters = _letters(text)
        count = len(letters)
        if count == 0:
            return 0.0

        by_length = self._lexicon_by_length
        best = [0.0] * (count + 1)
        for end in range(1, count + 1):
            highest = best[end - 1]  # skip this letter, cover nothing
            top = min(MAX_WORD_LENGTH, end)
            for size in range(2, top + 1):
                bucket = by_length.get(size)
                if not bucket:
                    continue
                start = end - size
                if letters[start:end] in bucket:
                    weight = 1.0 if size > 2 else 0.5 * size
                    value = best[start] + (size if size > 2 else weight)
                    if value > highest:
                        highest = value
            best[end] = highest
        return min(1.0, best[count] / count)

    def find_words(self, text: str, minimum_length: int = 4) -> list[str]:
        """Known words of at least *minimum_length* appearing in *text*.

        Used as human-readable evidence in candidate reports: seeing
        ``['THROUGH', 'MESSAGE', 'CAPTAIN']`` is far more convincing than
        seeing a log-probability.
        """
        letters = _letters(text)
        by_length = self._lexicon_by_length
        found: list[str] = []
        seen: set[str] = set()
        for start in range(len(letters)):
            for size in range(MAX_WORD_LENGTH, minimum_length - 1, -1):
                bucket = by_length.get(size)
                if not bucket:
                    continue
                word = letters[start : start + size]
                if len(word) == size and word in bucket and word not in seen:
                    seen.add(word)
                    found.append(word)
                    break
        return found

    def count_implausible_digraphs(self, text: str) -> int:
        """Pairs that essentially never occur inside an English word."""
        letters = _letters(text)
        return sum(
            1
            for i in range(len(letters) - 1)
            if letters[i : i + 2] in IMPLAUSIBLE_DIGRAPHS
        )

    # -- combined report ---------------------------------------------------

    def breakdown(self, text: str) -> ScoreBreakdown:
        """Every component of the judgement, for display and for audit."""
        letters = _letters(text)
        total = self.score_values(self.encode(letters))
        per_letter = total / len(letters) if letters else float("-inf")
        coverage = self.word_coverage(letters)
        words = self.find_words(letters)
        longest = max(words, key=len) if words else ""
        odd = self.count_implausible_digraphs(letters)
        # The combined figure exists only to break ties in a single ranked
        # list. It is never presented as a probability.
        combined = total + 4.0 * coverage * len(letters) / 10.0 - 0.5 * odd
        return ScoreBreakdown(
            length=len(letters),
            ngram_total=total,
            ngram_per_letter=per_letter,
            word_coverage=coverage,
            longest_word=longest,
            words_found=len(words),
            implausible_digraphs=odd,
            combined=combined,
        )

    def describe_model(self) -> str:
        """Human-readable provenance, printed by ``cipher_tool model``."""
        distinct4 = sum(1 for value in self._count4 if value)
        return (
            "English scoring model\n"
            "---------------------\n"
            f"Source            : {len(corpus_files())} prose files in "
            f"{DATA_DIR}\n"
            f"Corpus letters    : {len(self._letters):,}\n"
            f"Corpus words      : {len(_words(load_corpus_text())):,}\n"
            f"Lexicon size      : {len(self._lexicon):,} distinct words\n"
            f"Model             : order-3 interpolated Markov chain "
            f"(add-K, K={self.k})\n"
            f"Quadgrams seen    : {distinct4:,} of {ALPHABET_SIZE**4:,} possible\n"
            "Provenance        : all prose written by the team for this "
            "project; word list typed by the team.\n"
            "Network use       : none. The model is built locally at runtime."
        )


# ---------------------------------------------------------------------------
# Process-wide access
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def default_scorer() -> EnglishScorer:
    """The shared scorer. Built once, on first use."""
    return EnglishScorer()


def score_text(text: str) -> float:
    """Convenience wrapper around the shared scorer."""
    return default_scorer().score(text)


def normalised_score(text: str) -> float:
    """Convenience wrapper: score per letter."""
    return default_scorer().normalised(text)


def annotate(diagnostics: dict, plaintext: str, scorer: EnglishScorer) -> dict:
    """Attach the standard evidence fields to a candidate's diagnostics.

    Every solver calls this so that candidate reports are consistent and
    ``Candidate.confidence`` always has the two signals it needs.
    """
    report = scorer.breakdown(plaintext)
    diagnostics["normalised_score"] = report.ngram_per_letter
    diagnostics["word_coverage"] = report.word_coverage
    words = scorer.find_words(plaintext, minimum_length=5)[:6]
    if words:
        diagnostics["words_seen"] = ", ".join(words)
    return diagnostics


def rank_by_score(
    items: Iterable[tuple[str, str]], scorer: EnglishScorer | None = None
) -> list[tuple[float, str, str]]:
    """Score ``(key, plaintext)`` pairs and return them best first."""
    engine = scorer or default_scorer()
    scored = [(engine.score(text), key, text) for key, text in items]
    scored.sort(key=lambda row: row[0], reverse=True)
    return scored
