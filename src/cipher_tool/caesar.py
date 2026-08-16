"""Caesar (additive shift) cipher: encryption, decryption and its attack.

The cipher
----------
Every letter moves *s* places along the alphabet, wrapping round at Z.
Numbering A..Z as 0..25:

    E(x) = (x + s) mod 26            D(y) = (y - s) mod 26

There are 26 keys and one of them does nothing, so 25 useful keys. That is a
key space small enough to read through by eye, which is why the Caesar cipher
is never used alone. It matters here because it is the *building block* of
larger ciphers: every column of a Vigenere ciphertext is a single Caesar
shift, and a keyword substitution alphabet is often a shifted one. The
substitution and Vigenere solvers in this toolkit crack one column at a time
by calling :func:`best_shift_by_chi_squared` below.

The attack, and why it uses two independent measures
----------------------------------------------------
Brute force is complete here -- all 26 decryptions are produced -- so the only
real question is how to judge them. We judge each one twice.

**1. The n-gram score** (``scoring.EnglishScorer``). An order-3 Markov model
over letters. It knows about *order*: THE and QXZ are both three letters, but
only one of them is English. This is the primary ranking.

**2. Chi-squared against English letter frequencies.**

    chi2 = sum over letters j of  (observed_j - expected_j)^2 / expected_j

where expected_j = N * f_j and f_j is the English frequency of letter j.
This measure ignores order completely and looks only at how often each letter
turns up. A shift is a *rigid rotation* of the frequency histogram, so exactly
one of the 26 rotations lines the tall E, T, A spikes of the ciphertext up
with the tall spikes English expects. Every other rotation drops a common
letter onto a slot where a rare letter was expected, and because the term is
divided by that small expected count, the penalty is enormous. That is why
frequency fitting works so well on a shift and so badly on a general
substitution, where no single rotation can fix anything.

Reporting both is the point. Chi-squared is blind to order, so on short or
unusual texts it can prefer the wrong shift; the n-gram model uses context and
is more reliable, but it is a black box to a reader. When both measures name
the same shift the answer is essentially certain; when they disagree the text
is short, or is not a Caesar shift at all, and a human should look. Every
candidate therefore carries ``chi_squared``, ``rank_by_chi2`` and
``measures_agree`` in its diagnostics.

An implementation note on the rotation
--------------------------------------
Chi-squared for all 26 shifts is computed from *one* pass of letter counting.
Under decryption shift s the number of plaintext letter j is the number of
ciphertext letter (j + s) mod 26, so the 26 candidate histograms are just 26
rotations of one count vector. That turns "26 decryptions of an n-letter
text" into "26 rotations of a 26-entry list", which is what makes this cheap
enough for the Vigenere solver to call it thousands of times.
"""

from __future__ import annotations

from typing import Any, Sequence

from .candidates import Candidate, CandidateSet
from .normalize import (
    ALPHABET,
    ALPHABET_SIZE,
    NormalizedText,
    from_numbers,
    letters_only,
    normalize,
    to_numbers,
)
from .reference import ENGLISH_LETTER_FREQUENCY
from .scoring import EnglishScorer, annotate, default_scorer

METHOD = "Caesar shift"


# ---------------------------------------------------------------------------
# Key and input validation
# ---------------------------------------------------------------------------


def _check_shift(shift: int) -> int:
    """Validate a shift and reduce it to the range 0..25.

    Booleans are rejected even though ``bool`` is a subclass of ``int``:
    ``encrypt(text, True)`` is far more likely to be a bug than a request for
    a shift of one.
    """
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise ValueError(
            "Caesar shift must be a whole number of alphabet positions, got "
            f"{shift!r} ({type(shift).__name__}). Pass an int such as 3, or "
            "-3 to shift backwards."
        )
    return shift % ALPHABET_SIZE


def _require_text(text: str) -> str:
    """Reject non-string input loudly rather than half-processing it."""
    if not isinstance(text, str):
        raise ValueError(
            f"text must be a string, got {type(text).__name__}. Read the file "
            "first, or pass NormalizedText.letters."
        )
    return text


# ---------------------------------------------------------------------------
# Encryption and decryption
# ---------------------------------------------------------------------------


def _apply_shift(letters: str, shift: int) -> str:
    """Add *shift* to every letter value. ``letters`` must be A-Z already."""
    return from_numbers(value + shift for value in to_numbers(letters))


def encrypt(text: str, shift: int) -> str:
    """Encrypt. Operates on letters only; returns letters only, uppercase.

    Non-letters in *text* are discarded, so spacing and punctuation of the
    input never leak into the ciphertext.
    """
    return _apply_shift(letters_only(_require_text(text)), _check_shift(shift))


def decrypt(text: str, shift: int) -> str:
    """Exact inverse of :func:`encrypt` for the same shift."""
    return _apply_shift(
        letters_only(_require_text(text)), -_check_shift(shift) % ALPHABET_SIZE
    )


def shifted_alphabet(shift: int) -> str:
    """The cipher alphabet A-Z would map onto under *shift*.

    Useful for reporting: ``shifted_alphabet(3)`` is ``"DEFG...ABC"``.
    """
    return _apply_shift(ALPHABET, _check_shift(shift))


def all_shifts(text: str) -> list[tuple[int, str]]:
    """Every possible reading of *text*, as ``(shift, plaintext)`` pairs.

    Entry *s* is ``decrypt(text, s)``: the plaintext you would get if the key
    were *s*. All 26 are produced, including s = 0 (the text unchanged), so
    that nothing is hidden from the operator. Note that the *set* of strings
    here is the same whichever direction you shift; only the labelling
    differs, and labelling them as decryptions is what a cryptanalyst wants.
    """
    letters = letters_only(_require_text(text))
    return [(shift, _apply_shift(letters, -shift % ALPHABET_SIZE))
            for shift in range(ALPHABET_SIZE)]


# ---------------------------------------------------------------------------
# The frequency-fitting attack
# ---------------------------------------------------------------------------


def chi_squared_by_shift(text: str) -> list[tuple[int, float]]:
    """Chi-squared distance from English for each of the 26 decryptions.

    Returned as ``(shift, chi2)`` pairs in shift order, lowest chi2 being the
    best fit. The value is scaled per letter, matching
    :func:`statistics.chi_squared_english`, so the two are directly
    comparable (``tests/test_caesar.py`` asserts they agree exactly).

    An empty text gives ``inf`` for every shift: with no letters the measure
    is undefined, and returning zeros would look like a perfect fit.
    """
    letters = letters_only(_require_text(text))
    total = len(letters)
    if total == 0:
        return [(shift, float("inf")) for shift in range(ALPHABET_SIZE)]

    counts = [0] * ALPHABET_SIZE
    for value in to_numbers(letters):
        counts[value] += 1

    # expected[j] is how often English would use letter j in a text this long.
    expected = [
        total * ENGLISH_LETTER_FREQUENCY[letter] / 100.0 for letter in ALPHABET
    ]

    results: list[tuple[int, float]] = []
    for shift in range(ALPHABET_SIZE):
        score = 0.0
        for j in range(ALPHABET_SIZE):
            # Decrypting by `shift` sends ciphertext letter (j + shift) to
            # plaintext letter j, so the candidate histogram is the ciphertext
            # histogram rotated by `shift`.
            difference = counts[(j + shift) % ALPHABET_SIZE] - expected[j]
            score += difference * difference / expected[j]
        results.append((shift, score / total))
    return results


def best_shift_by_chi_squared(text: str) -> int:
    """The shift whose decryption best fits English letter frequencies.

    This is the classic single-column attack, and it is deliberately cheap:
    the Vigenere and substitution solvers call it once per key position.

    It looks at letter frequencies only, so it is reliable on a few hundred
    letters and unreliable on a few dozen. Callers that can afford to should
    treat the result as a starting point and confirm it with the n-gram
    scorer -- which is exactly what :func:`solve` does.

    Ties, and the undefined case of empty input, resolve to the lowest shift.
    """
    return min(chi_squared_by_shift(text), key=lambda row: (row[1], row[0]))[0]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _ranks(values: Sequence[float], *, best_is_lowest: bool) -> list[int]:
    """1-based rank of each value; rank 1 is the best.

    Ties always break towards the lower index, whichever direction is better,
    so that a rank of 1 and the ``chi2_best_shift``/``ngram_best_shift``
    diagnostics can never contradict each other.
    """
    direction = 1.0 if best_is_lowest else -1.0
    order = sorted(
        range(len(values)),
        key=lambda index: (direction * values[index], index),
    )
    ranks = [0] * len(values)
    for position, index in enumerate(order, start=1):
        ranks[index] = position
    return ranks


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    time_budget: float | None = None,
    **options: Any,
) -> CandidateSet:
    """Try all 26 shifts and return every one of them, ranked.

    All 26 are returned, not the best few: the whole key space costs
    milliseconds, and showing only the winner would hide the single most
    useful piece of evidence a Caesar attack produces -- the *margin* between
    the best reading and the rest. Use ``CandidateSet.top(n)`` to trim for
    display, and ``CandidateSet.score_gap()`` to see the margin.

    Parameters
    ----------
    source:
        Ciphertext, or an already-normalised :class:`NormalizedText`.
    scorer:
        English scorer; ``None`` uses the shared :func:`default_scorer`.
    top:
        Accepted for a uniform solver interface and not used for filtering
        here, for the reason above.
    time_budget:
        Accepted for a uniform solver interface. Never triggers: 26
        decryptions of any realistic text take milliseconds, so cutting this
        search short could only lose evidence.
    **options:
        Anything else is ignored, and recorded in the diagnostics under
        ``options_ignored`` so that a mistyped option is visible rather than
        silently swallowed.

    Empty input gives an empty :class:`CandidateSet`.
    """
    engine = scorer if scorer is not None else default_scorer()
    if isinstance(source, str):
        normalized = normalize(source)
    elif isinstance(source, NormalizedText):
        normalized = source
    else:
        raise ValueError(
            "solve() needs ciphertext as a string or a NormalizedText, got "
            f"{type(source).__name__}."
        )

    letters = normalized.letters
    if not letters:
        return CandidateSet()

    ignored = ", ".join(sorted(options)) if options else ""

    plaintexts = [_apply_shift(letters, -shift % ALPHABET_SIZE)
                  for shift in range(ALPHABET_SIZE)]
    scores = [engine.score(plaintext) for plaintext in plaintexts]
    chi_squared = [value for _, value in chi_squared_by_shift(letters)]

    chi_ranks = _ranks(chi_squared, best_is_lowest=True)
    score_ranks = _ranks(scores, best_is_lowest=False)

    # The two measures are independent: one reads letter order, the other
    # reads letter counts. Agreement is strong evidence; disagreement is a
    # warning that the text is short or is not a shift cipher at all.
    chi_best = min(range(ALPHABET_SIZE), key=lambda s: (chi_squared[s], s))
    score_best = max(range(ALPHABET_SIZE), key=lambda s: (scores[s], -s))
    agree = chi_best == score_best

    candidates = CandidateSet()
    for shift in range(ALPHABET_SIZE):
        plaintext = plaintexts[shift]
        diagnostics: dict[str, Any] = {
            "shifts_tested": ALPHABET_SIZE,
            "chi_squared": chi_squared[shift],
            "rank_by_chi2": chi_ranks[shift],
            "rank_by_ngram": score_ranks[shift],
            "chi2_best_shift": chi_best,
            "ngram_best_shift": score_best,
            "measures_agree": agree,
            "cipher_alphabet": shifted_alphabet(shift),
        }
        if ignored:
            diagnostics["options_ignored"] = ignored
        annotate(diagnostics, plaintext, engine)
        candidates.add(
            Candidate(
                method=METHOD,
                key=f"shift={shift}",
                score=scores[shift],
                plaintext=plaintext,
                diagnostics=diagnostics,
                # A shift replaces letters one for one, so the plaintext is
                # exactly as long as the input letters and can be poured back
                # into the original layout.
                display=normalized.relayout(plaintext),
            )
        )
    return candidates
