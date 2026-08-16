"""Vigenere cipher: encryption, decryption and its full cryptanalysis.

The cipher
----------
A Vigenere cipher is a Caesar shift whose amount changes from letter to
letter according to a short repeating keyword. Writing plaintext letters as
numbers 0..25 and a key of length n as K_0..K_{n-1}:

    C_i = (P_i + K_{i mod n}) mod 26
    P_i = (C_i - K_{i mod n}) mod 26

A key of ``AAAA...`` adds zero every time, so it is the identity: the
"ciphertext" is the plaintext. That is worth remembering, because it is what a
buggy key-cleaning routine silently produces, and any solver that returns an
all-A key is telling you the text was never enciphered this way.

Why it resists naive frequency analysis
---------------------------------------
Under a single alphabet, E stays the commonest letter and the whole text can
be read off a frequency table. Under a key of length n, a plaintext E becomes
one of n different ciphertext letters depending on where it falls, so the
single-letter distribution is smeared out. The Index of Coincidence -- the
probability that two letters drawn at random from the text are equal -- drops
from the English value of about 0.0667 towards the flat-random 1/26 = 0.0385:

    IC(n) ~= (1/n) * ((N - n)/(N - 1)) * 0.0667
             + ((n - 1)/n) * (N/(N - 1)) * 0.0385

for a text of N letters. That formula is the reason a long key looks almost
like random letters, and it is also the reason the *key length* is the whole
game. The moment n is known, the cipher collapses: take every nth letter and
you have n independent Caesar shifts, each of which is trivial.

So the attack is in two stages: find n, then solve n Caesars.

Stage 1: three independent ways to find n
-----------------------------------------
Each is a different kind of evidence, and they fail in different ways, which
is exactly why this module reports all three instead of picking one.

1. **Kasiski examination** (:func:`kasiski_analysis`). If a repeated piece of
   plaintext happens to line up with the same part of the key, it enciphers to
   the same ciphertext. The gap between two such ciphertext repeats is then a
   multiple of n. Collect the gaps, factorise them, and n is the factor that
   keeps turning up. Kasiski is *structural* evidence -- it does not care about
   English letter frequencies at all -- but it needs the text to be long
   enough to contain lucky repeats, and short coincidental repeats add noise.

2. **Index of Coincidence by period** (:func:`ic_analysis`). Split the text
   into p columns, column i holding every letter enciphered by key position i.
   If p = n each column is a pure Caesar shift of English, and a Caesar shift
   does not change IC, so each column reads about 0.0667. If p is wrong the
   columns are mixtures and read near 0.0385. This is *statistical* evidence:
   it needs perhaps twenty letters per column before it means anything, so it
   goes blind on long periods over short texts.

3. **Column chi-squared fit** (:func:`column_fit_analysis`). For each period
   p, take each column and try all 26 possible key letters, scoring each with
   chi-squared against English letter frequencies; keep the best. A period
   where *every* column has some shift that fits English well is a far better
   bet than one where three columns fit and two do not, because a genuine key
   length must explain the whole text, not part of it. Raw chi-squared per
   letter grows as columns get shorter, which would quietly favour short
   periods, so each column is also scored as a *ratio* of its best shift to
   the average over all 26 shifts. For a real Caesar column one shift is far
   better than the rest and the ratio is small; for a mixed column every shift
   is equally bad and the ratio approaches 1. The ratio is what the ranking
   uses, and the worst column's ratio is reported separately so that "all
   columns fit" can be told apart from "most columns fit".

:func:`estimate_key_lengths` combines all three into one ranked table. Any
multiple of a good period is itself a good period -- if 4 works then 8, 12 and
16 must also work, since each of their columns is a subset of a column that
already fits -- so a multiple whose evidence is no better than its divisor's
is marked as demoted rather than presented as a separate discovery.

Stage 2: recovering the key
---------------------------
:func:`solve_key_for_length` splits the text into n columns and picks each
column's best Caesar shift by chi-squared. On a long text that is usually
right outright. On a short one, a column of 15 letters simply does not have
enough evidence and chi-squared picks a near-miss.

:func:`refine_key` fixes that. It walks the key one position at a time, tries
all 26 letters there, and keeps whatever maximises the n-gram score of the
*whole decrypted text*. This works because chi-squared judges a column using
only its own 26 letter counts, whereas the n-gram model judges the plaintext
using the letters either side of each position -- letters that live in
*different* columns. That cross-column information is invisible to the
per-column test, and it is what distinguishes a real key letter from one that
happens to give a plausible-looking frequency profile. It is a hill climb, so
it can stop at a local maximum, and the cost is only 26n decryptions per pass
rather than the 26^n of a full search.

For very short keys a full search is affordable, and :func:`brute_force_length`
does it: 26 keys at n=1, 676 at n=2, 17,576 at n=3.

Nothing in this module talks to the network, and no third-party solver was
consulted; the algorithms above are implemented here from their mathematics.
"""

from __future__ import annotations

import itertools
import time
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .candidates import Candidate, CandidateSet
from .normalize import (
    ALPHABET,
    ALPHABET_SIZE,
    NormalizedText,
    clean_key,
    columns,
    from_numbers,
    letters_only,
    normalize,
    to_numbers,
)
from .reference import ENGLISH_IC, ENGLISH_LETTER_FREQUENCY, RANDOM_IC
from .scoring import EnglishScorer, annotate, default_scorer
from .statistics import (
    Repeat,
    find_repeats,
    ic_by_period,
    index_of_coincidence,
    kasiski_factor_votes,
)

__all__ = [
    "MAXIMUM_KEY_LENGTH",
    "MAXIMUM_BRUTE_FORCE_LENGTH",
    "MINIMUM_COLUMN_FOR_IC",
    "MINIMUM_COLUMN_FOR_FIT",
    "encrypt",
    "decrypt",
    "decrypt_with_key",
    "tabula_recta",
    "render_tabula_recta",
    "column_shift_fits",
    "best_shift_for_column",
    "KasiskiCandidate",
    "KasiskiReport",
    "kasiski_analysis",
    "PeriodIC",
    "ic_analysis",
    "ColumnFit",
    "column_fit_analysis",
    "KeyLengthEvidence",
    "estimate_key_lengths",
    "describe_key_lengths",
    "solve_key_for_length",
    "Refinement",
    "refine_key",
    "BruteForceResult",
    "brute_force_length",
    "solve",
]

#: Default ceiling on the key lengths we are willing to consider. Competition
#: keywords are almost always words, and a twenty-letter English word is rare.
MAXIMUM_KEY_LENGTH = 20

#: Columns shorter than this make the Index of Coincidence meaningless. The
#: same threshold statistics.ic_by_period uses, repeated here so the reason is
#: visible at the point of use.
MINIMUM_COLUMN_FOR_IC = 20

#: Columns shorter than this make the chi-squared column fit unreliable. It is
#: lower than the IC threshold because the fit is measured as a ratio against
#: the same column's own average, which cancels most of the length effect.
MINIMUM_COLUMN_FOR_FIT = 8

#: Weights used to fold the three key-length signals into one number. They are
#: a presentation heuristic for ordering a shortlist, not a probability: the
#: solver tries several lengths regardless of how they scored.
KASISKI_WEIGHT = 0.30
IC_WEIGHT = 0.30
FIT_WEIGHT = 0.25
WORST_FIT_WEIGHT = 0.15

#: A multiple of a shorter period is demoted when the shorter period scores
#: within this much of it. Some slack is needed because the longer period's
#: columns are shorter and therefore noisier, so it can edge ahead by luck.
DEMOTION_TOLERANCE = 0.05

#: How many letters the brute-force search scores each key against. Ranking
#: 17,576 keys on the whole of a long ciphertext is wasted work: a couple of
#: hundred letters already separates English from noise by a wide margin, and
#: the handful of survivors are rescored on the full text afterwards.
BRUTE_FORCE_SAMPLE = 200

#: Refusal threshold for the exhaustive search. 26^4 = 456,976 keys is already
#: about a minute of pure Python here; 26^5 = 11,881,376 is hours, which is not
#: a search, it is a hang. Above this the statistical attack is the answer.
MAXIMUM_BRUTE_FORCE_LENGTH = 4

_QUADGRAM_CONTEXT = ALPHABET_SIZE**3  # 17576, the sliding-window modulus


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _letters_of(source: str | NormalizedText) -> str:
    """Letters-only view of either kind of input."""
    if isinstance(source, NormalizedText):
        return source.letters
    return letters_only(source)


def _key_values(key: str) -> list[int]:
    """Validate and encode a user-supplied key as shift amounts 0..25.

    Cleaning is deliberately identical to ciphertext cleaning, so ``"le mon"``
    and ``"LEMON!"`` are the same key. What is rejected is a key with no
    letters at all, which would otherwise silently become a zero-length key
    and crash somewhere far from the mistake.
    """
    if not isinstance(key, str):
        raise ValueError(
            "Vigenere key must be a string of letters, got "
            f"{type(key).__name__}"
        )
    cleaned = clean_key(key)
    if not cleaned:
        raise ValueError(
            f"Vigenere key {key!r} contains no letters A-Z. "
            "Supply a keyword such as LEMON."
        )
    return to_numbers(cleaned)


# ---------------------------------------------------------------------------
# Encryption and decryption
# ---------------------------------------------------------------------------


def encrypt(text: str, key: str) -> str:
    """Encrypt *text* with the repeating keyword *key*.

    Operates on letters only and returns letters only, uppercase: case,
    spacing, punctuation and five-letter grouping in the input are discarded,
    and the key advances only on letters. A key of ``"AAA"`` is the identity
    and returns the input's letters unchanged.

    Raises ``ValueError`` if the key contains no letters.
    """
    key_values = _key_values(key)
    length = len(key_values)
    values = to_numbers(letters_only(text))
    return from_numbers(
        value + key_values[index % length] for index, value in enumerate(values)
    )


def decrypt(text: str, key: str) -> str:
    """Exact inverse of :func:`encrypt` for the same key."""
    key_values = _key_values(key)
    length = len(key_values)
    values = to_numbers(letters_only(text))
    return from_numbers(
        value - key_values[index % length] for index, value in enumerate(values)
    )


def decrypt_with_key(text: str, key: str) -> str:
    """Decrypt with a known key. Alias of :func:`decrypt`, named for callers.

    The CLI and the auto pipeline read better saying ``decrypt_with_key`` when
    the key came from the operator rather than from a search.
    """
    return decrypt(text, key)


def tabula_recta() -> list[str]:
    """The 26x26 Vigenere square, one string per row.

    Row *k* is the alphabet rotated left by *k*, so that

        tabula_recta()[k][p] == chr(65 + (k + p) % 26)

    which is exactly ``encrypt(plaintext_letter, key_letter)``: find the row
    for your key letter, the column for your plaintext letter, and read off
    the ciphertext letter. Decryption runs the other way -- go along the key
    letter's row until you find the ciphertext letter, and the column heading
    is the plaintext.
    """
    return [ALPHABET[index:] + ALPHABET[:index] for index in range(ALPHABET_SIZE)]


def render_tabula_recta() -> str:
    """The Vigenere square with row and column labels, for documentation."""
    header = "    " + " ".join(ALPHABET)
    lines = [header, "    " + "-" * (2 * ALPHABET_SIZE - 1)]
    for index, row in enumerate(tabula_recta()):
        lines.append(f"{ALPHABET[index]} | " + " ".join(row))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Column fitting: one Caesar shift at a time
# ---------------------------------------------------------------------------


def column_shift_fits(column: str) -> list[float]:
    """Chi-squared per letter of *column* under each of the 26 key letters.

    ``result[k]`` is how badly the column reads as English once decrypted with
    key letter ``chr(65 + k)``. Lower is a better fit.

    The measurement is the same one :func:`statistics.chi_squared_english`
    makes, but it is computed from rotated *counts* rather than by building 26
    decrypted strings. Decrypting with key letter k sends ciphertext letter c
    to plaintext letter (c - k) mod 26, so the number of plaintext p's is just
    the number of ciphertext (p + k) mod 26's. Rotating a 26-entry count
    vector is 26 operations regardless of how long the column is, which turns
    a scan over the whole text into arithmetic on a fixed-size table.
    """
    total = len(column)
    if total == 0:
        return [float("inf")] * ALPHABET_SIZE

    counts = [0] * ALPHABET_SIZE
    for character in column:
        counts[ord(character) - 65] += 1

    expected = [
        total * ENGLISH_LETTER_FREQUENCY[letter] / 100.0 for letter in ALPHABET
    ]

    fits: list[float] = []
    for shift in range(ALPHABET_SIZE):
        score = 0.0
        for plain in range(ALPHABET_SIZE):
            difference = counts[(plain + shift) % ALPHABET_SIZE] - expected[plain]
            score += difference * difference / expected[plain]
        fits.append(score / total)
    return fits


def best_shift_for_column(column: str) -> tuple[int, float, float]:
    """Best key letter for one column, as ``(shift, its chi2, mean chi2)``.

    The mean over all 26 shifts is returned alongside the winner because on
    its own the winning chi-squared says nothing: a short column always
    produces a smallish best value simply because 26 tries on noisy data will
    turn up something. What matters is how much better the winner is than the
    field, and that comparison needs the field.
    """
    fits = column_shift_fits(column)
    best = min(range(ALPHABET_SIZE), key=lambda shift: fits[shift])
    average = sum(fits) / ALPHABET_SIZE
    return best, fits[best], average


# ---------------------------------------------------------------------------
# Method 1: Kasiski examination
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KasiskiCandidate:
    """One candidate key length from Kasiski examination, with its support."""

    length: int
    votes: int
    supporting: tuple[Repeat, ...]
    gaps: tuple[int, ...]
    demoted_of: int | None = None

    def describe(self) -> str:
        """One readable block: the candidate and the runs that argue for it."""
        head = f"  length {self.length:2d}: {self.votes:5d} weighted votes"
        if self.demoted_of is not None:
            head += f"  [multiple of {self.demoted_of}, demoted]"
        lines = [head]
        for repeat in self.supporting[:4]:
            usable = [
                gap for gap in repeat.distances if gap % self.length == 0 and gap > 1
            ]
            places = ", ".join(str(position) for position in repeat.positions[:6])
            gaps = ", ".join(str(gap) for gap in usable[:6])
            lines.append(
                f"      {repeat.text:<8} x{repeat.count} at [{places}]"
                f"  gaps [{gaps}] all divisible by {self.length}"
            )
        if not self.supporting:
            lines.append("      (no repeated run produced a gap it divides)")
        return "\n".join(lines)


@dataclass(frozen=True)
class KasiskiReport:
    """The outcome of a Kasiski examination, evidence included."""

    text_length: int
    votes: Counter
    candidates: tuple[KasiskiCandidate, ...]
    repeats_examined: int

    def best_length(self) -> int | None:
        """Highest-voted candidate that is not a demoted multiple."""
        for candidate in self.candidates:
            if candidate.demoted_of is None:
                return candidate.length
        return None

    def describe(self) -> str:
        """Readable report: ranked candidates, each with its repeated runs."""
        lines = [
            f"Kasiski examination of {self.text_length} letters "
            f"({self.repeats_examined} repeated runs of 3+ letters)",
        ]
        if not self.candidates:
            lines.append(
                "  No usable repeats. Either the text is short, the key is "
                "long, or this is not a repeating-key cipher."
            )
            return "\n".join(lines)
        for candidate in self.candidates:
            lines.append(candidate.describe())
        lines.append(
            "  Votes are divisors of the gaps between repeated runs, weighted "
            "by run length; a longer repeat is far less likely to be chance."
        )
        return "\n".join(lines)


def kasiski_analysis(
    source: str | NormalizedText,
    *,
    minimum_run: int = 3,
    maximum_run: int = 12,
    max_key_length: int = MAXIMUM_KEY_LENGTH,
    limit: int = 6,
) -> KasiskiReport:
    """Rank key lengths by Kasiski examination and show the runs behind each.

    The reasoning, once more, because the report is only useful if a human can
    check it: a repeating key of length n means key position i is reused every
    n letters. If the same plaintext run appears twice exactly n, or 2n, or 3n
    letters apart, the key is in the same phase both times and the ciphertext
    run repeats too. So the gap between identical ciphertext runs is a multiple
    of n, and n divides it. Every divisor of every gap gets a vote; n collects
    one from every repeat that lined up, while accidental repeats scatter
    their votes over unrelated divisors.

    Multiples of a stronger candidate are marked ``demoted_of`` rather than
    dropped: every gap that 4 divides, 2 divides as well, so 2 automatically
    inherits 4's evidence and vice versa is not true. Presenting them as two
    findings would be counting one piece of evidence twice.
    """
    if minimum_run < 2:
        raise ValueError("Kasiski needs runs of at least 2 letters")
    if maximum_run < minimum_run:
        raise ValueError(
            f"maximum_run ({maximum_run}) must be at least "
            f"minimum_run ({minimum_run})"
        )
    if max_key_length < 2:
        raise ValueError("max_key_length must be at least 2 for Kasiski")

    letters = _letters_of(source)
    votes = kasiski_factor_votes(
        letters,
        minimum_length=minimum_run,
        maximum_length=maximum_run,
        maximum_factor=max_key_length,
    )

    # Collect the repeats once so each candidate can be shown its own support.
    repeats: list[Repeat] = []
    for size in range(minimum_run, maximum_run + 1):
        found = find_repeats(letters, size)
        if not found:
            break
        repeats.extend(found)

    ranked = sorted(votes.items(), key=lambda row: (-row[1], row[0]))
    candidates: list[KasiskiCandidate] = []
    for length, count in ranked:
        demoted_of = None
        for kept in candidates:
            if (
                kept.demoted_of is None
                and length % kept.length == 0
                and kept.votes >= count
            ):
                demoted_of = kept.length
                break
        supporting = [
            repeat
            for repeat in repeats
            if any(gap > 1 and gap % length == 0 for gap in repeat.distances)
        ]
        supporting.sort(key=lambda repeat: (-len(repeat.text), -repeat.count))
        gaps = sorted(
            {
                gap
                for repeat in supporting
                for gap in repeat.distances
                if gap > 1 and gap % length == 0
            }
        )
        candidates.append(
            KasiskiCandidate(
                length=length,
                votes=count,
                supporting=tuple(supporting[:6]),
                gaps=tuple(gaps[:12]),
                demoted_of=demoted_of,
            )
        )
        if len(candidates) >= limit:
            break

    return KasiskiReport(
        text_length=len(letters),
        votes=votes,
        candidates=tuple(candidates),
        repeats_examined=len(repeats),
    )


# ---------------------------------------------------------------------------
# Method 2: Index of Coincidence by period
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeriodIC:
    """Mean column Index of Coincidence for one candidate period."""

    period: int
    mean_ic: float
    column_ics: tuple[float, ...]
    shortest_column: int

    @property
    def distance_from_english(self) -> float:
        """How far the mean column IC is from English. Smaller is better."""
        return abs(self.mean_ic - ENGLISH_IC)

    @property
    def distance_from_random(self) -> float:
        """How far above flat-random the mean column IC sits."""
        return self.mean_ic - RANDOM_IC

    def describe(self) -> str:
        """One readable line: the period, its IC, and both reference points."""
        return (
            f"  period {self.period:2d}: mean column IC={self.mean_ic:.4f} "
            f"({self.distance_from_english:+.4f} from English {ENGLISH_IC:.4f}, "
            f"{self.distance_from_random:+.4f} above random {RANDOM_IC:.4f}), "
            f"shortest column {self.shortest_column} letters"
        )


def ic_analysis(
    source: str | NormalizedText, max_key_length: int = MAXIMUM_KEY_LENGTH
) -> list[PeriodIC]:
    """Mean column IC for every period that has enough letters to measure.

    A Caesar shift is a relabelling of the alphabet, and relabelling cannot
    change how often two randomly drawn letters happen to match. So the IC of
    a column is the IC of the plaintext that produced it -- English, about
    0.0667 -- if and only if that column really was enciphered by one single
    key letter. Guess the wrong period and each column mixes several shifts,
    which flattens it towards 1/26 = 0.0385.

    Periods leaving fewer than :data:`MINIMUM_COLUMN_FOR_IC` letters per column
    are not reported at all, because IC on twelve letters is noise, and a
    printed noise value is how people talk themselves into a wrong key length.
    That means this method simply has nothing to say about long keys on short
    texts, which is honest rather than unhelpful: the other two methods do.
    """
    if max_key_length < 1:
        raise ValueError("max_key_length must be at least 1")

    letters = _letters_of(source)
    rows: list[PeriodIC] = []
    for period, mean_ic in ic_by_period(letters, max_key_length):
        parts = columns(letters, period)
        rows.append(
            PeriodIC(
                period=period,
                mean_ic=mean_ic,
                column_ics=tuple(index_of_coincidence(part) for part in parts),
                shortest_column=min(len(part) for part in parts),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Method 3: chi-squared column fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnFit:
    """How well every column of one candidate period fits a Caesar shift."""

    period: int
    key_guess: str
    shifts: tuple[int, ...]
    chi_squares: tuple[float, ...]
    average_chi_squares: tuple[float, ...]
    ratios: tuple[float, ...]
    shortest_column: int

    @property
    def total_chi(self) -> float:
        """Summed goodness of fit over the columns. Lower is better.

        Comparable only between periods with the same number of columns, which
        is why the ranking uses :attr:`mean_ratio` instead.
        """
        return sum(self.chi_squares)

    @property
    def mean_chi(self) -> float:
        """Mean best-shift chi-squared per letter across the columns."""
        return self.total_chi / len(self.chi_squares)

    @property
    def mean_ratio(self) -> float:
        """Mean of best-shift chi2 divided by that column's average chi2.

        Near 0 means every column has one shift that is dramatically better
        than the other 25 -- the signature of a real Caesar column. Near 1
        means no shift stands out, so the column is a mixture and the period
        is wrong.
        """
        return sum(self.ratios) / len(self.ratios)

    @property
    def worst_ratio(self) -> float:
        """The worst-fitting column's ratio.

        Reported separately on purpose. A key length has to explain the entire
        ciphertext: five columns that fit and one that does not is not a
        partial success, it is evidence against the period. Averaging hides
        that; the maximum does not.
        """
        return max(self.ratios)

    def describe(self) -> str:
        """One readable line: the fit figures and the key the shifts spell."""
        return (
            f"  period {self.period:2d}: mean best chi2/letter={self.mean_chi:.4f}"
            f"  total={self.total_chi:.3f}"
            f"  mean ratio={self.mean_ratio:.3f}"
            f"  worst column={self.worst_ratio:.3f}"
            f"  key guess {self.key_guess}"
        )


def column_fit_analysis(
    source: str | NormalizedText,
    max_key_length: int = MAXIMUM_KEY_LENGTH,
    *,
    minimum_column: int = MINIMUM_COLUMN_FOR_FIT,
) -> list[ColumnFit]:
    """Fit every column of every period to its best Caesar shift.

    This is the module's stand-in for the "twist" family of statistics, and it
    measures the same idea in a way that is easier to explain: if p is the key
    length then each of the p columns is one Caesar shift of English, so each
    column must have a shift under which its letter frequencies look English.

    Two numbers come out of each column: the chi-squared of its best shift,
    and that value divided by the average chi-squared over all 26 shifts. The
    ratio matters more than the raw value. Chi-squared per letter is inflated
    on short samples -- roughly 25/n even for a perfect model, since that is
    the expected value of the statistic on its degrees of freedom -- so raw
    values quietly favour short periods, which have longer columns. Dividing
    by the same column's own average cancels that: it asks not "how good is
    the best shift" but "how much better is it than the alternatives", and
    that question has the same answer whatever the column length.
    """
    if max_key_length < 1:
        raise ValueError("max_key_length must be at least 1")

    letters = _letters_of(source)
    rows: list[ColumnFit] = []
    for period in range(1, max_key_length + 1):
        if len(letters) // period < minimum_column:
            break
        parts = columns(letters, period)
        shifts: list[int] = []
        chi_squares: list[float] = []
        averages: list[float] = []
        ratios: list[float] = []
        for part in parts:
            shift, best, average = best_shift_for_column(part)
            shifts.append(shift)
            chi_squares.append(best)
            averages.append(average)
            ratios.append(best / average if average > 0 else 1.0)
        rows.append(
            ColumnFit(
                period=period,
                key_guess=from_numbers(shifts),
                shifts=tuple(shifts),
                chi_squares=tuple(chi_squares),
                average_chi_squares=tuple(averages),
                ratios=tuple(ratios),
                shortest_column=min(len(part) for part in parts),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# The combined key-length shortlist
# ---------------------------------------------------------------------------


@dataclass
class KeyLengthEvidence:
    """Everything three independent tests have to say about one key length.

    ``combined`` is a weighted blend of the three signals, each scaled to
    roughly 0..1, and it is only used to order the shortlist. A signal that
    could not be measured (IC on a short text, for instance) contributes zero
    rather than a guessed value, so a length resting on less evidence scores
    lower -- which is the conservative direction to be wrong in. Whatever was
    actually measured is listed in ``evidence_available``.
    """

    length: int
    kasiski_votes: int
    mean_ic: float | None
    ic_distance: float | None
    column_fit: float | None
    column_fit_total: float | None
    fit_ratio: float | None
    worst_fit_ratio: float | None
    combined: float
    key_guess: str = ""
    demoted_of: int | None = None
    rank: int = 0
    evidence_available: tuple[str, ...] = ()

    def describe(self) -> str:
        """One readable table row."""
        ic_part = (
            f"IC={self.mean_ic:.4f} ({self.ic_distance:+.4f})"
            if self.mean_ic is not None and self.ic_distance is not None
            else "IC=  --   (columns too short)"
        )
        fit_part = (
            f"chi2={self.column_fit:.3f} ratio={self.fit_ratio:.3f}"
            f"/{self.worst_fit_ratio:.3f}"
            if self.column_fit is not None
            else "chi2=  --   ratio=  --  "
        )
        note = ""
        if self.demoted_of is not None:
            note = f"  [multiple of {self.demoted_of}, demoted]"
        return (
            f"{self.rank:>3}. n={self.length:<3} score={self.combined:.3f}  "
            f"kasiski={self.kasiski_votes:<5} {ic_part}  {fit_part}  "
            f"key?={self.key_guess or '-'}{note}"
        )


def _combine_evidence(
    kasiski_score: float,
    ic_score: float | None,
    fit_score: float | None,
    worst_fit_score: float | None,
) -> float:
    """Weighted blend of the three signals; unmeasured signals count as zero."""
    total = KASISKI_WEIGHT * kasiski_score
    if ic_score is not None:
        total += IC_WEIGHT * ic_score
    if fit_score is not None:
        total += FIT_WEIGHT * fit_score
    if worst_fit_score is not None:
        total += WORST_FIT_WEIGHT * worst_fit_score
    return total


def estimate_key_lengths(
    source: str | NormalizedText,
    max_key_length: int = MAXIMUM_KEY_LENGTH,
) -> list[KeyLengthEvidence]:
    """Rank candidate key lengths using all three methods together.

    Returns every length that could be measured at all, best first, with
    multiples of a better-scoring divisor pushed to the end and flagged. An
    empty list means the input had no letters.

    The demotion rule is the important one. If the key really has length 4
    then splitting into 8 columns also works -- each of those 8 columns is
    half of a column that was already a single Caesar shift, so it is still a
    single Caesar shift. Every multiple of the true length therefore looks
    good, and a table that lists 4, 8, 12 and 16 as four separate findings is
    reporting one fact four times. So a length is demoted when some divisor of
    it scores within :data:`DEMOTION_TOLERANCE`. It stays in the list, because
    occasionally the shorter period is the accident (a keyword with a repeated
    half, say ``ABCABC``, genuinely has period 3 and the toolkit should say
    so), but it sorts below everything that is not a multiple.
    """
    if max_key_length < 1:
        raise ValueError("max_key_length must be at least 1")

    letters = _letters_of(source)
    if not letters:
        return []

    # A period longer than half the text leaves at most one letter per column,
    # where none of the three measurements mean anything.
    limit = max(1, min(max_key_length, len(letters) // 2))

    votes: Counter = Counter()
    if limit >= 2 and len(letters) >= 6:
        votes = kasiski_analysis(letters, max_key_length=limit).votes
    ic_rows = {row.period: row for row in ic_analysis(letters, limit)}
    fit_rows = {row.period: row for row in column_fit_analysis(letters, limit)}

    top_votes = max(votes.values()) if votes else 0
    ic_span = ENGLISH_IC - RANDOM_IC

    rows: list[KeyLengthEvidence] = []
    for length in range(1, limit + 1):
        ic_row = ic_rows.get(length)
        fit_row = fit_rows.get(length)
        available: list[str] = []

        kasiski_votes = int(votes.get(length, 0))
        kasiski_score = kasiski_votes / top_votes if top_votes else 0.0
        if top_votes:
            available.append("kasiski")

        ic_score: float | None = None
        if ic_row is not None:
            available.append("ic")
            # 0 at flat-random, 1 at English, clipped either side.
            ic_score = min(1.0, max(0.0, (ic_row.mean_ic - RANDOM_IC) / ic_span))

        fit_score: float | None = None
        worst_score: float | None = None
        if fit_row is not None:
            available.append("column-fit")
            fit_score = min(1.0, max(0.0, 1.0 - fit_row.mean_ratio))
            worst_score = min(1.0, max(0.0, 1.0 - fit_row.worst_ratio))

        rows.append(
            KeyLengthEvidence(
                length=length,
                kasiski_votes=kasiski_votes,
                mean_ic=ic_row.mean_ic if ic_row else None,
                ic_distance=ic_row.distance_from_english if ic_row else None,
                column_fit=fit_row.mean_chi if fit_row else None,
                column_fit_total=fit_row.total_chi if fit_row else None,
                fit_ratio=fit_row.mean_ratio if fit_row else None,
                worst_fit_ratio=fit_row.worst_ratio if fit_row else None,
                combined=_combine_evidence(
                    kasiski_score, ic_score, fit_score, worst_score
                ),
                key_guess=fit_row.key_guess if fit_row else "",
                evidence_available=tuple(available),
            )
        )

    by_length = {row.length: row for row in rows}
    for row in rows:
        for divisor in range(1, row.length):
            if row.length % divisor:
                continue
            shorter = by_length.get(divisor)
            if shorter is None:
                continue
            if shorter.combined >= row.combined - DEMOTION_TOLERANCE:
                row.demoted_of = divisor
                break

    rows.sort(
        key=lambda row: (row.demoted_of is not None, -row.combined, row.length)
    )
    for position, row in enumerate(rows, start=1):
        row.rank = position
    return rows


def describe_key_lengths(evidence: Sequence[KeyLengthEvidence]) -> str:
    """Render the key-length shortlist as a readable table with a legend."""
    if not evidence:
        return "No key-length evidence: the input contains no letters."
    lines = ["Candidate key lengths, best first", "=" * 33]
    for row in evidence:
        lines.append(row.describe())
    lines.append("")
    lines.append(
        "score    weighted blend of the three tests below (ordering aid only)"
    )
    lines.append(
        "kasiski  weighted votes from divisors of gaps between repeated runs"
    )
    lines.append(
        "IC       mean Index of Coincidence of the columns; English is "
        f"{ENGLISH_IC:.4f}, random {RANDOM_IC:.4f}"
    )
    lines.append(
        "chi2     mean chi-squared of each column's best Caesar shift; "
        "ratio is mean/worst column measured against"
    )
    lines.append(
        "         that column's own 26-shift average, so 0 means every column "
        "fits one shift far better than the rest"
    )
    lines.append(
        "key?     the key those best-fitting shifts spell out, before any "
        "whole-text refinement"
    )
    lines.append(
        "demoted  a multiple of a divisor that scored as well; the same "
        "evidence, not a second finding"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Key recovery
# ---------------------------------------------------------------------------


def _decrypt_values(values: Sequence[int], key_values: Sequence[int]) -> list[int]:
    """Decrypt already-encoded letters with an already-encoded key."""
    length = len(key_values)
    repeats = len(values) // length + 1
    stream = (list(key_values) * repeats)[: len(values)]
    return [
        (value - shift) % ALPHABET_SIZE for value, shift in zip(values, stream)
    ]


def solve_key_for_length(
    source: str | NormalizedText,
    length: int,
    scorer: EnglishScorer | None = None,
    *,
    refine: bool = False,
) -> str:
    """Recover the most likely key of exactly *length* letters.

    Splits the ciphertext into *length* columns and solves each independently
    as a Caesar shift, choosing the shift whose decrypted column has the
    smallest chi-squared distance from English letter frequencies. Column i is
    every letter at position i, i+length, i+2*length, ... so by construction
    all of them were enciphered by key letter i.

    This is fast and needs no language model, but it judges each key letter on
    one column's 26 letter counts alone. On short columns that is thin
    evidence. Pass ``refine=True`` (or call :func:`refine_key` yourself) to
    follow it with the whole-text pass that repairs near misses; *scorer* is
    consulted only in that case.
    """
    letters = _letters_of(source)
    if length < 1:
        raise ValueError(f"key length must be at least 1, got {length}")
    if not letters:
        raise ValueError("cannot recover a key from a text with no letters")
    if length > len(letters):
        raise ValueError(
            f"key length {length} exceeds the {len(letters)} letters available; "
            "every column would hold at most one letter"
        )

    shifts = [
        best_shift_for_column(column)[0] for column in columns(letters, length)
    ]
    key = from_numbers(shifts)
    if refine:
        return refine_key(letters, key, scorer).key
    return key


@dataclass(frozen=True)
class Refinement:
    """What the whole-text refinement pass did to a candidate key."""

    key: str
    original_key: str
    score_before: float
    score_after: float
    changed_positions: tuple[int, ...]
    rounds: int
    stopped_early: bool

    @property
    def changes(self) -> int:
        """How many key letters the pass altered."""
        return len(self.changed_positions)

    def describe(self) -> str:
        """One readable line: what changed, where, and what it bought."""
        if not self.changes:
            return (
                f"refinement: {self.original_key} unchanged after "
                f"{self.rounds} pass(es)"
            )
        return (
            f"refinement: {self.original_key} -> {self.key} "
            f"({self.changes} letter(s) at {list(self.changed_positions)}), "
            f"score {self.score_before:.1f} -> {self.score_after:.1f}"
        )


def refine_key(
    source: str | NormalizedText,
    key: str,
    scorer: EnglishScorer | None = None,
    *,
    max_rounds: int = 4,
    deadline: float | None = None,
) -> Refinement:
    """Improve a key one position at a time against the whole-text n-gram score.

    For each key position in turn, try all 26 letters there and keep whichever
    gives the highest-scoring decryption of the entire ciphertext. Repeat until
    a full pass changes nothing, or *max_rounds* passes have run.

    Why this catches what chi-squared misses: chi-squared asks whether one
    column's letters are distributed like English, using 26 counts drawn from
    perhaps twenty letters. Two shifts can easily produce similar-looking
    profiles at that sample size, and the wrong one wins by noise. The n-gram
    model asks a completely different question -- whether the decrypted text
    reads like English *in sequence* -- and the letters flanking each position
    belong to other columns, so it is using information the per-column test
    cannot see. One wrong key letter out of six leaves every sixth letter
    wrong, which wrecks the quadgram score even while the column's own
    frequency profile looks passable.

    It is a hill climb over one coordinate at a time, so it can stall in a
    local maximum: the returned :class:`Refinement` reports exactly what
    changed, and it is never presented as a proof of correctness.
    """
    engine = scorer or default_scorer()
    letters = _letters_of(source)
    cleaned = clean_key(key)
    if not cleaned:
        raise ValueError(
            f"cannot refine key {key!r}: it contains no letters A-Z"
        )
    if not letters:
        raise ValueError("cannot refine a key against a text with no letters")
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")

    values = to_numbers(letters)
    key_values = to_numbers(cleaned)

    def score_of(candidate_key: Sequence[int]) -> float:
        return engine.score_values(_decrypt_values(values, candidate_key))

    score_before = score_of(key_values)
    best_score = score_before
    stopped_early = False
    rounds = 0

    for _ in range(max_rounds):
        rounds += 1
        improved = False
        for position in range(len(key_values)):
            if deadline is not None and time.monotonic() >= deadline:
                stopped_early = True
                break
            current = key_values[position]
            best_letter = current
            local_best = best_score
            for trial_letter in range(ALPHABET_SIZE):
                if trial_letter == current:
                    continue
                key_values[position] = trial_letter
                trial_score = score_of(key_values)
                if trial_score > local_best:
                    local_best = trial_score
                    best_letter = trial_letter
            key_values[position] = best_letter
            if best_letter != current:
                improved = True
                best_score = local_best
        if stopped_early or not improved:
            break

    final_key = from_numbers(key_values)
    changed = tuple(
        position
        for position, (before, after) in enumerate(zip(cleaned, final_key))
        if before != after
    )
    return Refinement(
        key=final_key,
        original_key=cleaned,
        score_before=score_before,
        score_after=best_score,
        changed_positions=changed,
        rounds=rounds,
        stopped_early=stopped_early,
    )


# ---------------------------------------------------------------------------
# Exhaustive search for short keys
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BruteForceResult:
    """Outcome of an exhaustive search over every key of one length."""

    length: int
    keys_possible: int
    keys_tried: int
    sample_length: int
    best: tuple[tuple[str, float], ...]
    stopped_early: bool

    @property
    def exhaustive(self) -> bool:
        """True only if every possible key was actually scored."""
        return self.keys_tried >= self.keys_possible


def _quadgram_score(values: Sequence[int], table: Sequence[float]) -> float:
    """Order-3 log-probability of an encoded plaintext, ranking use only.

    Deliberately skips the first three letters, which have no full context.
    That drops at most three terms out of hundreds and costs a branch per
    candidate to include; the shortlist this produces is rescored with the
    full model before anything is reported, so the approximation never reaches
    a candidate's published score.

    The index arithmetic is the same sliding window ``scoring.score_values``
    uses: a quadgram (a,b,c,d) lives at ((a*26+b)*26+c)*26+d, and dropping the
    oldest letter to admit the next one is ``(index mod 26^3) * 26``.
    """
    count = len(values)
    if count < 4:
        return 0.0
    index = ((values[0] * 26 + values[1]) * 26 + values[2]) * 26
    total = 0.0
    for position in range(3, count):
        index += values[position]
        total += table[index]
        index = (index % _QUADGRAM_CONTEXT) * 26
    return total


def brute_force_length(
    source: str | NormalizedText,
    length: int,
    scorer: EnglishScorer | None = None,
    *,
    keep: int = 3,
    sample: int = BRUTE_FORCE_SAMPLE,
    deadline: float | None = None,
) -> BruteForceResult:
    """Score every one of the 26**length keys and keep the best few.

    Worth doing precisely because it cannot be fooled: the statistical attack
    can fail on a short ciphertext, where a three-letter key leaves columns of
    a dozen letters and chi-squared has nothing to work with, and exhaustive
    search does not care. 26 keys at length 1, 676 at 2 and 17,576 at 3 are all
    cheap. 26^4 = 456,976 is borderline -- around a minute of pure Python here
    -- and 26^5 = 11,881,376 is not feasible at all, which is why lengths above
    :data:`MAXIMUM_BRUTE_FORCE_LENGTH` are refused rather than attempted.
    """
    if length < 1:
        raise ValueError(f"brute force needs a key length of at least 1, got {length}")
    if length > MAXIMUM_BRUTE_FORCE_LENGTH:
        raise ValueError(
            f"refusing to brute force key length {length}: "
            f"26**{length} = {ALPHABET_SIZE ** length:,} keys would take hours. "
            f"The limit is {MAXIMUM_BRUTE_FORCE_LENGTH} "
            f"({ALPHABET_SIZE ** MAXIMUM_BRUTE_FORCE_LENGTH:,} keys). "
            "Use the statistical attack for longer keys."
        )
    if keep < 1:
        raise ValueError("keep must be at least 1")

    engine = scorer or default_scorer()
    letters = _letters_of(source)
    possible = ALPHABET_SIZE**length
    if not letters:
        return BruteForceResult(length, possible, 0, 0, (), False)

    window = letters if sample <= 0 else letters[:sample]
    values = to_numbers(window)
    table = engine.table()
    # Precomputed so the inner loop never evaluates i % length.
    key_positions = [index % length for index in range(len(values))]

    best: list[tuple[float, str]] = []
    threshold = float("-inf")
    tried = 0
    stopped_early = False

    for combination in itertools.product(range(ALPHABET_SIZE), repeat=length):
        if deadline is not None and tried % 512 == 0 and time.monotonic() >= deadline:
            stopped_early = True
            break
        tried += 1
        plain = [
            (value - combination[position]) % ALPHABET_SIZE
            for value, position in zip(values, key_positions)
        ]
        score = _quadgram_score(plain, table)
        if score > threshold or len(best) < keep:
            best.append((score, from_numbers(combination)))
            best.sort(key=lambda row: row[0], reverse=True)
            del best[keep:]
            threshold = best[-1][0]

    return BruteForceResult(
        length=length,
        keys_possible=possible,
        keys_tried=tried,
        sample_length=len(window),
        best=tuple((key, score) for score, key in best),
        stopped_early=stopped_early,
    )


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------


def _out_of_time(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _make_candidate(
    normalized: NormalizedText,
    key: str,
    diagnostics: dict,
    scorer: EnglishScorer,
) -> Candidate:
    """Decrypt with *key* and package the result with all its evidence."""
    plaintext = decrypt(normalized.letters, key)
    if set(key) == {"A"}:
        diagnostics["identity_key"] = (
            "the key is all A, so this decryption is the ciphertext unchanged "
            "-- the text is not a Vigenere of this length"
        )
    diagnostics["key_length"] = len(key)
    annotate(diagnostics, plaintext, scorer)
    return Candidate(
        method="Vigenere",
        key=f"key={key}",
        score=scorer.score(plaintext),
        plaintext=plaintext,
        diagnostics=diagnostics,
        # Same length as the input letters, so the original layout can be kept.
        display=normalized.relayout(plaintext),
    )


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    key: str | None = None,
    key_length: int | None = None,
    max_key_length: int = MAXIMUM_KEY_LENGTH,
    brute_force_up_to: int = 3,
    lengths_to_try: int = 4,
    refine: bool = True,
    time_budget: float | None = None,
) -> CandidateSet:
    """Attack a Vigenere ciphertext and return ranked candidates.

    Options
    -------
    key:
        A known key. Decrypts with it and returns that one candidate; no
        search is run at all.
    key_length:
        Force one key length instead of estimating it.
    max_key_length:
        Ceiling for the key-length estimate. Default 20.
    brute_force_up_to:
        Exhaustively try every key up to this length as well as the
        statistical attack. Default 3 (18,278 keys in total across lengths
        1 to 3). Set to 0 to skip it.
    lengths_to_try:
        How many of the shortlisted key lengths to attack. Default 4, so the
        answer never rests on the estimator having got its first choice right.
    refine:
        Run the whole-text refinement pass after chi-squared. Default True.
    time_budget:
        Seconds. Checked between key lengths and inside the exhaustive search;
        candidates carry ``time_budget_hit`` when a search was cut short. At
        least one key length is always attempted, so a tiny budget returns a
        real (if unpolished) answer rather than nothing.

    Every candidate's diagnostics carry the evidence for its key length, so a
    reader can see whether the shortlist was decisive or a coin toss.
    """
    engine = scorer or default_scorer()
    normalized = normalize(source) if isinstance(source, str) else source
    letters = normalized.letters
    results = CandidateSet()
    if not letters:
        return results

    if key is not None:
        cleaned = clean_key(key)
        _key_values(key)  # validation, with its explanatory error message
        diagnostics: dict = {
            "search": "none, key supplied by the operator",
            "supplied_key": cleaned,
        }
        results.add(_make_candidate(normalized, cleaned, diagnostics, engine))
        return results

    if max_key_length < 1:
        raise ValueError(f"max_key_length must be at least 1, got {max_key_length}")
    if brute_force_up_to < 0:
        raise ValueError(
            f"brute_force_up_to cannot be negative, got {brute_force_up_to}"
        )
    if brute_force_up_to > MAXIMUM_BRUTE_FORCE_LENGTH:
        raise ValueError(
            f"brute_force_up_to={brute_force_up_to} is refused: "
            f"26**{brute_force_up_to} = "
            f"{ALPHABET_SIZE ** brute_force_up_to:,} keys would take hours. "
            f"The limit is {MAXIMUM_BRUTE_FORCE_LENGTH}."
        )
    if key_length is not None:
        if key_length < 1:
            raise ValueError(f"key_length must be at least 1, got {key_length}")
        if key_length > len(letters):
            raise ValueError(
                f"key_length {key_length} exceeds the {len(letters)} letters "
                "available; there would be at most one letter per column"
            )

    deadline = None if time_budget is None else time.monotonic() + time_budget
    budget_hit = False

    evidence = estimate_key_lengths(normalized, max_key_length)
    evidence_by_length = {row.length: row for row in evidence}

    if key_length is not None:
        chosen = [key_length]
    else:
        chosen = [row.length for row in evidence][: max(1, lengths_to_try)]
    if not chosen:
        chosen = [1]

    for position, length in enumerate(chosen):
        # Always attempt the first length, however small the budget: returning
        # nothing at all is less useful than returning one unpolished answer.
        if position > 0 and _out_of_time(deadline):
            budget_hit = True
            break
        if length > len(letters):
            continue

        diagnostics = {"search": "chi-squared per column"}
        row = evidence_by_length.get(length)
        if row is not None:
            diagnostics["length_rank"] = row.rank
            diagnostics["length_score"] = row.combined
            diagnostics["kasiski_votes"] = row.kasiski_votes
            if row.mean_ic is not None:
                diagnostics["mean_column_ic"] = row.mean_ic
                diagnostics["ic_distance_from_english"] = row.ic_distance
            else:
                diagnostics["mean_column_ic"] = "not measured, columns too short"
            if row.column_fit is not None:
                diagnostics["column_chi_squared"] = row.column_fit
                diagnostics["column_fit_ratio"] = row.fit_ratio
                diagnostics["worst_column_fit_ratio"] = row.worst_fit_ratio
            diagnostics["evidence_available"] = ", ".join(row.evidence_available)
            if row.demoted_of is not None:
                diagnostics["demoted_of"] = row.demoted_of

        shortest_column = len(letters) // length
        if shortest_column < MINIMUM_COLUMN_FOR_FIT:
            diagnostics["warning"] = (
                f"columns hold only about {shortest_column} letters, which is "
                "too few for chi-squared to be trusted"
            )

        candidate_key = solve_key_for_length(normalized, length)
        diagnostics["chi_squared_key"] = candidate_key
        if refine:
            refinement = refine_key(
                normalized, candidate_key, engine, deadline=deadline
            )
            diagnostics["refinement"] = refinement.describe()
            diagnostics["refinement_changes"] = refinement.changes
            if refinement.stopped_early:
                budget_hit = True
            candidate_key = refinement.key
            diagnostics["search"] = "chi-squared per column, then whole-text refinement"

        results.add(_make_candidate(normalized, candidate_key, diagnostics, engine))

    for length in range(1, brute_force_up_to + 1):
        if length > len(letters):
            break
        if _out_of_time(deadline):
            budget_hit = True
            break
        found = brute_force_length(
            normalized, length, engine, keep=2, deadline=deadline
        )
        if found.stopped_early:
            budget_hit = True
        for candidate_key, sample_score in found.best:
            diagnostics = {
                "search": (
                    f"exhaustive over all {found.keys_possible:,} keys of "
                    f"length {length}"
                ),
                "keys_tried": found.keys_tried,
                "exhaustive": found.exhaustive,
                "ranking_sample_letters": found.sample_length,
                "ranking_sample_score": sample_score,
            }
            row = evidence_by_length.get(length)
            if row is not None:
                diagnostics["length_rank"] = row.rank
                diagnostics["kasiski_votes"] = row.kasiski_votes
            results.add(
                _make_candidate(normalized, candidate_key, diagnostics, engine)
            )

    if budget_hit:
        for candidate in results.ranked():
            candidate.diagnostics["time_budget_hit"] = True

    if top > 0:
        return CandidateSet(results.top(top))
    return results
