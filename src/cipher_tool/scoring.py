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

from .reference import COMMON_WORDS, EXTENDED_WORDS, IMPLAUSIBLE_DIGRAPHS

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


#: Score per letter at which one window counts as English, for
#: :meth:`EnglishScorer.english_fraction`. Sits between the `promising` and
#: `strong` thresholds in candidates.py, and is defined here rather than
#: imported to keep scoring free of that dependency.
#:
#: MEASURED on the 2017 challenge 5A decrypted correctly: its prose windows
#: ran -0.99 to -1.37 and its enciphered tile windows -2.78 to -2.88, so
#: anything between about -1.5 and -2.5 separates them. -1.6 leaves room for
#: prose that is harder than that message's.
ENGLISH_WINDOW_NGRAM = -1.6

#: A lexicon word no longer than this, sitting inside text the lexicon cannot
#: explain, is a coincidence of letters rather than a word. NO is a word; the
#: NO in TECHNOLOGY is not.
SPURIOUS_WORD_LETTERS = 2

#: The same idea for a word with unexplained text on BOTH sides, where the
#: evidence is stronger and the limit can be looser. TECHNOLOGY splits as
#: TECH + NO + LOG + Y, and LOG is three letters, so a limit of two would
#: leave it stranded in the middle of a word nothing else recognised.
SANDWICHED_WORD_LETTERS = 3

#: An unexplained run no longer than this is a fragment of a word rather than
#: an unknown word in its own right, so a short word beside it probably
#: belongs to it. DGE is a fragment, so DO+DGE is DODGE; CHARLES is not, so
#: the DEAR before it stays a word of its own.
WORD_FRAGMENT_LETTERS = 4


def _absorb_spurious_words(
    flagged: list[tuple[str, bool]],
) -> list[tuple[str, bool]]:
    """Undo word breaks the lexicon's gaps invented.

    `segment` promises that a stretch it cannot account for comes back as one
    unbroken chunk. Left alone it does not keep that promise: the search will
    happily mine a short known word out of the middle of an unknown one,
    because two letters of NO cost far less as a word than as unknown text.
    Measured on a real solve, that turned CHARLES into ``CH A RLES``,
    TECHNOLOGY into ``TECH NO LOGY`` and DODGE into ``DO DGE``.

    Two rules, and the second is what stops the first going too far:

    1. A short known word with unexplained text on BOTH sides is absorbed
       into it. Nothing around it was recognised, so a word boundary there is
       a guess dressed as a finding.
    2. A short known word beside a SHORT unexplained run is absorbed into it.
       The length test is the whole safeguard: DGE is three letters and
       cannot be a word, so DO belongs to it, while CHARLES is seven and
       stands on its own, so the DEAR before it is left alone.
    """
    if not flagged:
        return flagged

    def join_runs(items: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
        out: list[tuple[str, bool]] = []
        for text, explained in items:
            if not explained and out and not out[-1][1]:
                out[-1] = (out[-1][0] + text, False)
            else:
                out.append((text, explained))
        return out

    # Both rules run together, to a fixed point, because each one creates
    # work for the other. TECHNOLOGY splits as TECH + NO + LOG + Y, and
    # neither rule fires on its own: NO has LOG (a real word) on its right so
    # it is not sandwiched, and LOG has NO on its left for the same reason.
    # Rule 2 absorbs NO into the fragment TECH, and only THEN is LOG
    # sandwiched between TECHNO and Y, which is what finally puts the word
    # back together.
    joined = join_runs(list(flagged))
    while True:
        changed = False

        for index in range(1, len(joined) - 1):
            text, explained = joined[index]
            if (explained and len(text) <= SANDWICHED_WORD_LETTERS
                    and not joined[index - 1][1] and not joined[index + 1][1]):
                joined[index] = (text, False)
                changed = True

        for index, (text, explained) in enumerate(joined):
            if not explained or len(text) > SPURIOUS_WORD_LETTERS:
                continue
            before = joined[index - 1] if index else None
            after = joined[index + 1] if index + 1 < len(joined) else None
            beside_fragment = (
                (before is not None and not before[1]
                 and len(before[0]) <= WORD_FRAGMENT_LETTERS)
                or (after is not None and not after[1]
                    and len(after[0]) <= WORD_FRAGMENT_LETTERS)
            )
            if beside_fragment:
                joined[index] = (text, False)
                changed = True

        if not changed:
            break
        joined = join_runs(joined)

    return joined


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
        counts: dict[str, int] = {}
        for word in _words(corpus_text):
            if 1 <= len(word) <= MAX_WORD_LENGTH:
                counts[word] = counts.get(word, 0) + 1
        # Hand-typed common words are, by construction, common. Give any that
        # our prose happened not to use a modest baseline so that word
        # segmentation still prefers them to rare corpus vocabulary.
        # Hand-typed words are, by construction, ordinary English. Give any
        # our prose happened not to use a baseline count so segmentation
        # still prefers them to rare corpus vocabulary. The core list gets
        # the higher floor because those really are the commonest words.
        for word in EXTENDED_WORDS:
            counts[word] = max(counts.get(word, 0), 2)
        for word in COMMON_WORDS:
            counts[word] = max(counts.get(word, 0), 6)
        # Single letters other than A and I are not English words and would
        # let coverage scoring accept anything at all.
        counts = {w: n for w, n in counts.items()
                  if len(w) > 1 or w in {"A", "I"}}

        total = sum(counts.values())
        # log10 P(word), used by `segment`. A word seen once in the corpus is
        # far less likely than THE, and this is what stops the split
        # preferring one long rare word to two short common ones.
        self._word_logp = {
            word: math.log10(count / total) for word, count in counts.items()
        }
        #: What an unexplained letter costs. Set well below the rarest word so
        #: that skipping text is always a last resort.
        self._unknown_letter_cost = math.log10(0.5 / total) * 2.0
        return frozenset(counts)

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

    def segment(self, text: str) -> list[str]:
        """Split a recovered plaintext back into words, as best we can.

        A decryption comes out as one unbroken run of letters, because the
        cipher destroyed the spaces. This puts them back by the same dynamic
        programming used for :meth:`word_coverage`, choosing the split that
        accounts for the most letters with known words.

        This is a READABILITY AID AND A GUESS. The letters are what the
        decryption actually produced; the spaces are this function's opinion.
        It gets ``ATONE`` versus ``AT ONE`` wrong sometimes, and a stretch it
        cannot account for is returned as one unbroken chunk rather than
        chopped into plausible-looking nonsense. Anything shown to a user
        must say which part is the decryption and which part is the guess.
        """
        letters = _letters(text)
        count = len(letters)
        if count == 0:
            return []

        logp = self._word_logp
        # best[i] is the best total log10 probability for the first i
        # characters, treating the plaintext as a run of independent words.
        #
        # Two earlier objectives were tried and are worth recording, because
        # both look reasonable and both produce nonsense:
        #
        #   * maximise covered letters -- indifferent between one seven
        #     letter word and seven one-letter ones, so NOTHINGSOFATAL came
        #     out as "NO THINGS OF AT A L";
        #   * maximise the square of each word's length -- now biased the
        #     other way, preferring one long rare word to two short common
        #     ones, so AREBUILT became "A REBUILT" and MEETMEAT "MEET MEAT".
        #
        # Word probability is the objective that actually models the
        # question. ARE and BUILT are both common, REBUILT is rare, so
        # "ARE BUILT" wins on the numbers rather than on a length rule.
        best = [0.0] * (count + 1)
        chosen = [0] * (count + 1)  # word length ending here, 0 = unexplained
        for end in range(1, count + 1):
            highest = best[end - 1] + self._unknown_letter_cost
            pick = 0
            for size in range(1, min(MAX_WORD_LENGTH, end) + 1):
                probability = logp.get(letters[end - size : end])
                if probability is not None:
                    value = best[end - size] + probability
                    if value > highest:
                        highest, pick = value, size
            best[end] = highest
            chosen[end] = pick

        # Walk back, gathering unexplained letters into single chunks so the
        # output never implies a word division we did not actually find.
        # Each piece carries whether the lexicon actually explained it: the
        # tidying below needs to know, and it cannot be recovered afterwards
        # because a two-letter chunk looks identical either way.
        flagged: list[tuple[str, bool]] = []
        unexplained: list[str] = []
        position = count
        while position > 0:
            size = chosen[position]
            if size == 0:
                unexplained.append(letters[position - 1])
                position -= 1
                continue
            if unexplained:
                flagged.append(("".join(reversed(unexplained)), False))
                unexplained = []
            flagged.append((letters[position - size : position], True))
            position -= size
        if unexplained:
            flagged.append(("".join(reversed(unexplained)), False))
        flagged.reverse()

        pieces = [text for text, _ in _absorb_spurious_words(flagged)]

        # A word our lexicon happens not to hold leaves its ending stranded:
        # TASKS becomes TASK + S because only TASK is listed. Printing
        # "TASK S" invents a word break that is certainly wrong, so a stray
        # one or two letters is glued back onto its neighbour. Longer
        # unexplained runs are left alone -- those are genuinely unrecognised
        # text and pretending otherwise would hide it.
        merged: list[str] = []
        for piece in pieces:
            stranded = len(piece) <= 2 and piece not in self._lexicon
            if stranded and merged:
                merged[-1] += piece
            else:
                merged.append(piece)
        return merged

    def segmented(self, text: str) -> str:
        """:meth:`segment` joined with spaces, for display."""
        return " ".join(self.segment(text))

    def english_fraction(self, text: str, window: int = 200) -> float:
        """How much of *text* reads as English, measured window by window.

        A mean over the whole message is the wrong instrument when part of
        the message is deliberately not prose, and competition messages are
        full of such parts -- embedded numbers, coordinates, keys, or in the
        case that prompted this, a steganographic frieze of black and white
        tiles enciphered along with the words.

        MEASURED on the 2017 challenge 5A, correctly decrypted: -0.99 per
        letter across the opening, -2.8 across the 1,500 letters of tiles,
        -1.37 at the end, and -2.070 for the message as a whole. Whole-text
        scoring called a perfect solve `weak`. Windowed, three quarters of
        the prose is plainly English and the answer is obvious.

        Returns the fraction of whole windows that read as English. Short
        texts get a single window, so this is never worse than the ordinary
        measure.
        """
        letters = _letters(text)
        if not letters:
            return 0.0
        size = min(window, len(letters))
        windows = [letters[start:start + size]
                   for start in range(0, len(letters) - size + 1, size)]
        if not windows:
            return 0.0
        english = sum(1 for piece in windows
                      if self.normalised(piece) >= ENGLISH_WINDOW_NGRAM)
        return english / len(windows)

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
    # Recorded for every candidate, because a message that is only partly
    # prose is otherwise indistinguishable from a failed decryption. Only
    # reported when it disagrees with the whole-message view, so ordinary
    # candidates are not cluttered with a number that says nothing new.
    portion = scorer.english_fraction(plaintext)
    if portion and report.ngram_per_letter < ENGLISH_WINDOW_NGRAM:
        diagnostics["english_fraction"] = round(portion, 3)
        diagnostics["partly_english"] = (
            f"{portion:.0%} of this message reads as English window by "
            "window, though the whole scores badly -- the rest may be "
            "numbers, coordinates or a key rather than prose"
        )
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
