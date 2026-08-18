"""Candidate plaintext management.

Every solver in this toolkit returns a *list of candidates*, never a single
"answer". A candidate carries the method that produced it, the key or
configuration, the English score, the plaintext, and whatever diagnostic
evidence the solver can offer.

The confidence label attached to a candidate is deliberately coarse and
deliberately pessimistic. It is derived from the scorer's per-letter
log-probability and dictionary coverage, and it never says "solved" -- the
strongest label available is ``strong``, and the printed report always tells
the operator to read the plaintext before believing it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

# ---------------------------------------------------------------------------
# Confidence thresholds.
#
# Thresholds on the *normalised* score (mean log10 probability per letter
# under the local order-3 model) and on dictionary word coverage.
#
# MEASURED, not guessed. Training the model on five of the six corpus files
# and testing on the sixth (held out, so the model has never seen it) over
# samples of 100 to 800 letters gave:
#
#     text                        ngram/letter        word coverage
#     ------------------------    ------------        -------------
#     real English                -0.89  (worst -1.03)    0.85
#     key with 2 letters wrong    -1.21  (worst -1.66)    0.65
#     wrong Caesar shift          -2.69                   0.05
#     scrambled monoalphabetic    -2.73                   0.07
#     uniformly random letters    -2.77                   0.06
#
# The thresholds below sit in the gaps. Note how far real English is from
# everything else, and how a nearly-correct key lands in between -- that
# middle band is exactly what "promising" is for.
#
# These are heuristics for sorting human attention, not statements of
# correctness. Re-run tests/test_scoring.py::test_calibration_separation if
# the corpus changes.
# ---------------------------------------------------------------------------
_STRONG_NGRAM = -1.15
_STRONG_COVERAGE = 0.72
_PROMISING_NGRAM = -1.80

#: How much of a message must read as English, window by window, before a
#: badly-scoring whole is treated as a partial solve rather than a failure.
#: Measured on the case that prompted it, roughly a third of the 2017
#: challenge 5A is prose and the rest is an enciphered tile frieze.
_PARTIAL_ENGLISH_FRACTION = 0.35
_PROMISING_COVERAGE = 0.35
_WEAK_NGRAM = -2.40

# ---------------------------------------------------------------------------
# Degeneracy
#
# Both signals above read the plaintext and neither counts how many DIFFERENT
# letters it uses, so a reading made of one short word repeated satisfies them
# completely. MEASURED on 626 letters:
#
#     text                ngram/letter    word coverage   distinct letters
#     ----------------    ------------    -------------   ----------------
#     ANDANDAND...           -0.637           1.000               3
#     THETHETHE...           -0.760           1.000               3
#     IDIDID...              -0.756           0.748               2
#     real English           -0.710           0.960              24
#
# ANDANDAND... outscores genuine English and every one of those is labelled
# `strong`. This is not hypothetical: a search with more key freedom than
# ciphertext collapses onto exactly this shape, and the scorer REWARDS the
# collapse rather than catching it.
#
# The separating signal is how much of the reading its two commonest letters
# take. Calibrated against the 48 published answers in the competition
# archive and 780 corpus samples from 40 to 3,000 letters: genuine text never
# exceeded 0.431 and never fell below 23 distinct letters; every collapse
# observed sat at 0.60 or above on two to seven letters.
#
# BOTH tests are length-gated, because short text is repetitive by nature and
# rejecting a correct answer is worse than anything this guard prevents.
# ATTACKATDAWN is a perfectly good plaintext whose two commonest letters are
# 58 per cent of it. Swept over 400 windows per length: genuine English last
# exceeds the share limit at 12 letters, is clear of it from 15, and from 40
# upward never passed 0.433 against a limit of 0.55. A fixed distinct-letter
# floor of 15 would likewise reject real 40-letter prose, which can use only
# 13, so that test waits until 200 letters where genuine text holds 19+.
_DEGENERATE_TOP_TWO_SHARE = 0.55
_DEGENERATE_SHARE_FROM = 40
_DEGENERATE_DISTINCT = 17
_DEGENERATE_DISTINCT_FROM = 200

CONFIDENCE_ORDER = ("strong", "promising", "weak", "unlikely")


def looks_degenerate(text: str) -> str | None:
    """Say why *text* is too repetitive to be a reading, or ``None``.

    A degenerate reading is not a worse answer than the plaintext; it is not
    an answer at all, so this overrides every other signal rather than
    competing with it.
    """
    total = len(text)
    if not total:
        return None
    counts = sorted(Counter(text).values(), reverse=True)
    share = sum(counts[:2]) / total
    if total >= _DEGENERATE_SHARE_FROM and share > _DEGENERATE_TOP_TWO_SHARE:
        return f"its two commonest letters are {share:.0%} of it"
    if total >= _DEGENERATE_DISTINCT_FROM and len(counts) < _DEGENERATE_DISTINCT:
        return f"it uses only {len(counts)} different letters in {total}"
    return None


@dataclass
class Candidate:
    """One proposed decryption.

    Attributes
    ----------
    method:
        Human-readable name of the technique, e.g. ``"Vigenere"``.
    key:
        Human-readable key or configuration, e.g. ``"key=LEMON"`` or
        ``"shift=3"``. Free text; solvers should make it copy-pasteable.
    score:
        Total English score. Higher is better. Comparable only between
        candidates over texts of the same length.
    plaintext:
        The candidate plaintext, letters only, uppercase.
    diagnostics:
        Free-form evidence the solver wants to expose (IC values, Kasiski
        factor counts, restart statistics, matrix determinants, ...).
    display:
        Optional pre-rendered plaintext preserving the original layout.
    """

    method: str
    key: str
    score: float
    plaintext: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    display: str | None = None

    # -- derived reporting -------------------------------------------------

    @property
    def normalised_score(self) -> float:
        """Score per letter, so candidates over different texts compare."""
        if not self.plaintext:
            return float("-inf")
        return self.score / len(self.plaintext)

    def confidence(self) -> str:
        """Coarse label for sorting human attention.

        Uses the per-letter n-gram score and, when the solver recorded it,
        dictionary coverage. Requiring *both* signals to agree stops a text
        that merely has English-looking letter statistics (a near-miss key,
        for example) from being labelled ``strong``.

        Both signals measure the *plaintext*, and neither knows how much
        freedom the search had to manufacture it. A solver that does know --
        because it can see it is working below the ciphertext length its
        attack needs -- says so by putting a ``confidence_cap`` in the
        diagnostics, and the label is weakened to it. Without that, a search
        with more key freedom than ciphertext produces a fluent sentence that
        satisfies both thresholds and is not the plaintext.
        """
        ngram = self.diagnostics.get("normalised_score", self.normalised_score)
        coverage = self.diagnostics.get("word_coverage")

        if coverage is None:
            # No coverage measurement available: fall back to n-gram only and
            # cap the label one notch below strong, because the n-gram model
            # alone cannot tell near-English from English.
            if ngram >= _PROMISING_NGRAM:
                label = "promising"
            elif ngram >= _WEAK_NGRAM:
                label = "weak"
            else:
                label = "unlikely"
        elif ngram >= _STRONG_NGRAM and coverage >= _STRONG_COVERAGE:
            label = "strong"
        elif ngram >= _PROMISING_NGRAM and coverage >= _PROMISING_COVERAGE:
            label = "promising"
        elif ngram >= _WEAK_NGRAM:
            label = "weak"
        else:
            label = "unlikely"

        # A message can be correctly decrypted and still score badly as a
        # whole, because part of it was never prose. MEASURED on the 2017
        # challenge 5A, decrypted perfectly: -2.070 per letter and 36 per
        # cent word coverage, because 1,500 of its 2,778 letters are a
        # steganographic frieze of black and white tiles enciphered along
        # with the words. Windowed, the prose is plainly English. Telling
        # somebody their correct answer failed is the worst outcome here.
        #
        # Raised no further than `promising`, deliberately: part of the
        # message really is not English, so `strong` would overstate it. And
        # only ever raised -- a reading that already scores well is untouched.
        if label in {"weak", "unlikely"}:
            portion = self.diagnostics.get("english_fraction")
            if portion is not None and portion >= _PARTIAL_ENGLISH_FRACTION:
                label = "promising"

        # Applied last, and after the partial-prose promotion above, because a
        # reading built from one repeated word scores as English on every
        # signal this class has -- including window-by-window prose, which it
        # passes perfectly. See `looks_degenerate` for the measurements.
        degenerate = looks_degenerate(self.plaintext)
        if degenerate is not None:
            label = max(label, "weak", key=CONFIDENCE_ORDER.index)

        cap = self.diagnostics.get("confidence_cap")
        if cap in CONFIDENCE_ORDER:
            # CONFIDENCE_ORDER runs strongest to weakest, so the larger index
            # is the weaker label and that is the one to keep.
            return max(label, cap, key=CONFIDENCE_ORDER.index)
        return label

    def preview(self, width: int = 76) -> str:
        """First *width* characters of the plaintext, ellipsised."""
        text = self.display or self.plaintext
        text = " ".join(text.split())
        if len(text) <= width:
            return text
        return text[: width - 3] + "..."


class CandidateSet:
    """An ordered collection of candidates, best first.

    Deduplicates on ``(method, plaintext)`` so that, for example, a hill
    climber finding the same solution from twenty restarts reports it once
    while still recording how many restarts agreed (useful evidence).
    """

    def __init__(
        self,
        candidates: Iterable[Candidate] = (),
        *,
        source_letters: str | None = None,
    ) -> None:
        #: The ciphertext these candidates came from, letters only, when the
        #: caller knows it. Used to spot candidates that did not actually
        #: decrypt anything -- see :meth:`identity_candidates`.
        self.source_letters = source_letters
        self._items: list[Candidate] = []
        self._index: dict[tuple[str, str], Candidate] = {}
        self.extend(candidates)

    #: How much of the longer text the shared prefix must cover before a
    #: near-match counts as the input being handed straight back. Deliberately
    #: a ratio rather than a letter count: a fixed slack that is right for a
    #: 450-letter message is far too loose for a 20-letter one.
    _IDENTITY_COVERAGE = 0.9

    def is_identity(self, candidate: Candidate) -> bool:
        """True if this candidate's "decryption" returned the input unchanged.

        Every cipher family here contains a key that does nothing: Caesar
        shift 0, affine a=1 b=0, Vigenere key AAA, Beaufort key A, Bifid
        period 1, the identity substitution alphabet, the identity Hill
        matrix. Handed plaintext, every solver finds its own identity key and
        scores it highly -- correctly, because the text really is English.
        What would be wrong is presenting that as a decryption, or counting
        several of them as agreeing.

        The comparison is deliberately NOT exact equality. Block ciphers work
        in fixed-size groups, so on a text whose length is not a multiple of
        the block size they hand back a padded or truncated copy. Hill 2x2 on
        451 letters returns 450, and an exact test called that a decryption --
        then ranked it FIRST, because one letter fewer at the same score per
        letter is a slightly better total, and so switched this guard off for
        every other candidate too. A solver dropping the odd letter has still
        decrypted nothing.

        Agreeing on every letter of the shared prefix is what makes this safe:
        a genuine decryption changes letters, and disagrees within the first
        few. The coverage ratio then stops a short accidental prefix match
        from being read as the whole message.
        """
        if self.source_letters is None:
            return False
        source = self.source_letters
        plaintext = candidate.plaintext
        if plaintext == source:
            return True
        shorter, longer = sorted((plaintext, source), key=len)
        if not shorter:
            return False
        if longer[:len(shorter)] != shorter:
            return False
        return len(shorter) >= self._IDENTITY_COVERAGE * len(longer)

    # -- container protocol ------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Candidate]:
        return iter(self.ranked())

    def __bool__(self) -> bool:
        return bool(self._items)

    # -- building ----------------------------------------------------------

    def add(self, candidate: Candidate) -> Candidate:
        """Add a candidate, merging duplicates.

        When the same (method, plaintext) pair arrives again we keep the
        higher-scoring record and increment an ``agreements`` counter, which
        solvers surface as evidence of convergence.
        """
        key = (candidate.method, candidate.plaintext)
        existing = self._index.get(key)
        if existing is None:
            candidate.diagnostics.setdefault("agreements", 1)
            self._items.append(candidate)
            self._index[key] = candidate
            return candidate

        existing.diagnostics["agreements"] = (
            existing.diagnostics.get("agreements", 1) + 1
        )
        if candidate.score > existing.score:
            existing.score = candidate.score
            existing.key = candidate.key
            agreements = existing.diagnostics["agreements"]
            existing.diagnostics.update(candidate.diagnostics)
            existing.diagnostics["agreements"] = agreements
        return existing

    def extend(self, candidates: Iterable[Candidate]) -> None:
        """Add every candidate in *candidates*, merging duplicates."""
        for candidate in candidates:
            self.add(candidate)

    # -- reading -----------------------------------------------------------

    def ranked(self) -> list[Candidate]:
        """All candidates, highest score first."""
        return sorted(self._items, key=lambda c: c.score, reverse=True)

    def top(self, count: int = 5) -> list[Candidate]:
        """The *count* highest-scoring candidates."""
        if count <= 0:
            return []
        return self.ranked()[:count]

    def best(self) -> Candidate | None:
        """The single highest-scoring candidate, or ``None`` if empty."""
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def score_gap(self) -> float | None:
        """Normalised-score margin between the best candidate and the best
        candidate that says something DIFFERENT.

        A large gap means the search found one clearly better answer. A tiny
        gap means several keys explain the ciphertext about equally well,
        which is exactly when a human must look at the text rather than the
        number.

        The comparison deliberately skips candidates with the same plaintext
        as the winner. Different methods routinely produce identical
        plaintext -- an affine key with a=1 IS a Caesar shift, and a keyword
        alphabet IS a substitution -- and treating that as a photo finish
        would warn about ambiguity at the exact moment the evidence is
        strongest. Agreement between methods is corroboration, not doubt:
        see :meth:`corroborations`.

        Returns ``None`` when every candidate agrees, because then there is
        no competing reading to measure against.
        """
        ranked = self.ranked()
        if not ranked:
            return None
        best = ranked[0]
        for candidate in ranked[1:]:
            if candidate.plaintext != best.plaintext:
                return best.normalised_score - candidate.normalised_score
        return None

    def corroborations(self) -> list[str]:
        """Every method that independently produced the winning plaintext.

        Two unrelated attacks arriving at the same text is strong evidence,
        and worth saying out loud rather than burying as a near-tie.

        Identity results are excluded, and this matters. Given text that was
        never encrypted, every solver finds its own do-nothing key -- Caesar
        shift 0, Vigenere key AAA, affine a=1 b=0 -- and all of them return
        the input unchanged. Counting those as agreement would announce that
        several independent attacks corroborate each other when in truth none
        of them decrypted anything. That is the most misleading thing this
        toolkit could say, so it does not say it.
        """
        ranked = self.ranked()
        if not ranked:
            return []
        best = ranked[0]
        if self.is_identity(best):
            return []
        methods: list[str] = []
        for candidate in ranked:
            if candidate.plaintext != best.plaintext:
                continue
            if self.is_identity(candidate):
                continue
            if candidate.method not in methods:
                methods.append(candidate.method)
        return methods

    def identity_candidates(self) -> list[Candidate]:
        """Candidates whose key did nothing, leaving the input unchanged."""
        return [c for c in self._items if self.is_identity(c)]

    def looks_unencrypted(self) -> bool:
        """True when the best candidate simply handed the input back.

        The honest reading is not "solved with key AAA" but "this text does
        not appear to be encrypted at all".
        """
        best = self.best()
        return best is not None and self.is_identity(best)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_UNCERTAINTY_NOTE = (
    "These are ranked guesses from a statistical search, not conclusions. "
    "Read the plaintext before trusting any of them."
)


def render_candidate(
    candidate: Candidate,
    index: int | None = None,
    *,
    full_text: bool = False,
    width: int = 76,
) -> str:
    """Format a single candidate as the plain block described in the spec."""
    lines: list[str] = []
    header = "Candidate" if index is None else f"Candidate {index}"
    lines.append(header)
    lines.append(f"  Method:      {candidate.method}")
    lines.append(f"  Key/config:  {candidate.key}")
    lines.append(
        f"  Score:       {candidate.score:.2f} "
        f"({candidate.normalised_score:.3f} per letter)"
    )
    lines.append(f"  Confidence:  {candidate.confidence()}  [heuristic]")

    diagnostics = {
        k: v
        for k, v in candidate.diagnostics.items()
        if k not in {"normalised_score"}
    }
    if diagnostics:
        lines.append("  Evidence:")
        for name in sorted(diagnostics):
            value = diagnostics[name]
            if isinstance(value, float):
                value = f"{value:.4f}"
            lines.append(f"    - {name}: {value}")

    if full_text:
        body = candidate.display or candidate.plaintext
        lines.append("  Plaintext:")
        for line in _wrap(body, width):
            lines.append(f"    {line}")
    else:
        lines.append(f"  Plaintext:   {candidate.preview(width)}")
    return "\n".join(lines)


def render_candidates(
    candidates: Sequence[Candidate],
    *,
    top: int = 5,
    full_text: bool = False,
    title: str | None = None,
    width: int = 76,
) -> str:
    """Format a ranked candidate list, always with the uncertainty note."""
    chosen = list(candidates)[: top if top > 0 else len(candidates)]
    blocks: list[str] = []
    if title:
        blocks.append(title)
        blocks.append("=" * len(title))
    if not chosen:
        blocks.append("No candidates were produced.")
        return "\n".join(blocks)

    for position, candidate in enumerate(chosen, start=1):
        blocks.append(render_candidate(
            candidate, position, full_text=full_text, width=width
        ))
        blocks.append("")

    blocks.append(f"NOTE: {_UNCERTAINTY_NOTE}")
    return "\n".join(blocks)


def _wrap(text: str, width: int) -> list[str]:
    """Hard-wrap *text* at *width*, preserving explicit line breaks."""
    if width <= 0:
        return [text]
    out: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            out.append("")
            continue
        start = 0
        while start < len(paragraph):
            out.append(paragraph[start : start + width])
            start += width
    return out
