"""Ciphertext statistics and the heuristic cipher-family report.

Everything in this module is a *measurement*. The one place we draw
conclusions -- :func:`cipher_family_hypotheses` -- returns clearly labelled
heuristics with the evidence attached, and never says a cipher has been
identified.

The two workhorse measurements
------------------------------
**Index of Coincidence.** The probability that two letters picked at random
from the text (without replacement) are the same letter:

    IC = sum over letters of  n_i * (n_i - 1)  /  (N * (N - 1))

For a flat distribution over 26 letters IC = 1/26 = 0.0385. English prose is
lumpy, so IC is about 0.0667. The crucial property is that a *monoalphabetic*
cipher only relabels letters, so it leaves IC untouched; a *polyalphabetic*
cipher spreads each plaintext letter across several cipher letters and pulls
IC towards 0.0385. IC is therefore the first question to ask of any text.

**Chi-squared against English.** How far the letter distribution is from
English:

    chi2 = sum over letters of  (observed_i - expected_i)^2 / expected_i

A transposition cipher only reorders letters, so its letter frequencies are
*exactly* those of the plaintext and chi2 is small. A substitution cipher has
an English-like IC but a scrambled distribution, so chi2 is large. Comparing
IC and chi2 together separates the two families -- which is the single most
useful thing the analyse report does.

**Kasiski.** If a repeated plaintext string happens to line up with the same
part of a repeating key, it enciphers to the same ciphertext string. So the
distance between two identical ciphertext runs is likely to be a multiple of
the key length. Collect those distances, factorise them, and the factor that
turns up most often is the prime suspect for the key length.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .normalize import ALPHABET, NormalizedText, columns
from .reference import (
    COMMON_DIGRAPHS,
    COMMON_TRIGRAPHS,
    ENGLISH_IC,
    ENGLISH_LETTER_FREQUENCY,
    RANDOM_IC,
)

# ---------------------------------------------------------------------------
# Primitive measurements
# ---------------------------------------------------------------------------


def letter_counts(text: str) -> dict[str, int]:
    """Occurrences of each letter A-Z, including zeros."""
    counter = Counter(text)
    return {letter: counter.get(letter, 0) for letter in ALPHABET}


def letter_frequencies(text: str) -> dict[str, float]:
    """Percentage of the text made up by each letter."""
    total = len(text)
    if total == 0:
        return {letter: 0.0 for letter in ALPHABET}
    counts = letter_counts(text)
    return {letter: 100.0 * count / total for letter, count in counts.items()}


def index_of_coincidence(text: str) -> float:
    """Index of Coincidence of *text*.

    Returns 0.0 for texts shorter than two letters, where the measure is
    undefined rather than zero -- callers must not read that as evidence.
    """
    total = len(text)
    if total < 2:
        return 0.0
    counts = Counter(text)
    numerator = sum(count * (count - 1) for count in counts.values())
    return numerator / (total * (total - 1))


def normalised_ic(text: str) -> float:
    """IC scaled so that flat random text reads 1.0 and English reads ~1.73.

    Multiplying by 26 is the convention that makes the number independent of
    alphabet size, which some references use. Reported alongside raw IC so
    the operator can compare with whichever table they are reading.
    """
    return index_of_coincidence(text) * 26.0


def chi_squared_english(text: str) -> float:
    """Chi-squared distance between the text's letters and English.

    Small means "these letters are distributed like English". Scaled per
    letter so texts of different lengths are comparable.
    """
    total = len(text)
    if total == 0:
        return float("inf")
    counts = letter_counts(text)
    score = 0.0
    for letter in ALPHABET:
        expected = total * ENGLISH_LETTER_FREQUENCY[letter] / 100.0
        difference = counts[letter] - expected
        score += difference * difference / expected
    return score / total


def divisors(number: int) -> list[int]:
    """All positive divisors of *number*, ascending. ``divisors(0) == []``."""
    if number <= 0:
        return []
    found: set[int] = set()
    root = int(math.isqrt(number))
    for candidate in range(1, root + 1):
        if number % candidate == 0:
            found.add(candidate)
            found.add(number // candidate)
    return sorted(found)


def prime_factors(number: int) -> list[int]:
    """Prime factorisation of *number* with multiplicity, ascending."""
    if number <= 1:
        return []
    factors: list[int] = []
    remaining = number
    divisor = 2
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 1 if divisor == 2 else 2
    if remaining > 1:
        factors.append(remaining)
    return factors


# ---------------------------------------------------------------------------
# Repeated sequences and Kasiski
# ---------------------------------------------------------------------------


def repeated_ngrams(
    text: str, size: int, minimum_count: int = 2
) -> dict[str, list[int]]:
    """Every *size*-letter run occurring at least *minimum_count* times.

    Maps the run to the list of positions where it starts. Positions are into
    the letters-only text, which is what distance arithmetic needs.
    """
    if size <= 0 or size > len(text):
        return {}
    positions: dict[str, list[int]] = defaultdict(list)
    for start in range(len(text) - size + 1):
        positions[text[start : start + size]].append(start)
    return {
        gram: places
        for gram, places in positions.items()
        if len(places) >= minimum_count
    }


def repeat_distances(positions: Sequence[int]) -> list[int]:
    """Gaps between consecutive occurrences of one repeated run.

    Only consecutive gaps are used, not every pair. Every pairwise distance is
    a sum of consecutive gaps, so including them all would double-count the
    same evidence and inflate the factor tally.
    """
    return [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]


@dataclass(frozen=True)
class Repeat:
    """One repeated run of ciphertext and where it occurs."""

    text: str
    positions: tuple[int, ...]
    distances: tuple[int, ...]

    @property
    def count(self) -> int:
        """How many times this run occurs in the ciphertext."""
        return len(self.positions)


def find_repeats(text: str, size: int, minimum_count: int = 2) -> list[Repeat]:
    """All repeated runs of length *size*, most frequent first."""
    found = [
        Repeat(gram, tuple(places), tuple(repeat_distances(places)))
        for gram, places in repeated_ngrams(text, size, minimum_count).items()
    ]
    found.sort(key=lambda r: (-r.count, r.text))
    return found


def kasiski_factor_votes(
    text: str,
    minimum_length: int = 3,
    maximum_length: int = 12,
    maximum_factor: int = 30,
) -> Counter[int]:
    """Kasiski examination: how often each candidate key length divides a gap.

    For every repeated run of length ``minimum_length`` upwards, take the gaps
    between consecutive occurrences and award a vote to every divisor of that
    gap from 2 up to *maximum_factor*. A repeating key of length L makes gaps
    that are multiples of L, so L collects votes from every repeat that lined
    up; unrelated coincidences scatter their votes across many divisors.

    Longer repeats are much less likely to be coincidence, so a repeat of
    length n is weighted ``n - minimum_length + 1``. Without that weighting a
    swarm of accidental trigram repeats can drown out one decisive 7-gram.
    """
    votes: Counter[int] = Counter()
    for size in range(minimum_length, maximum_length + 1):
        repeats = find_repeats(text, size)
        if not repeats:
            # No repeats of this length means none of any greater length.
            break
        weight = size - minimum_length + 1
        for repeat in repeats:
            for distance in repeat.distances:
                if distance <= 1:
                    continue
                for factor in divisors(distance):
                    if 2 <= factor <= maximum_factor:
                        votes[factor] += weight
    return votes


def ic_by_period(text: str, maximum_period: int = 20) -> list[tuple[int, float]]:
    """Average IC of the columns produced by each candidate period.

    Splitting the ciphertext into *p* columns, where column *i* holds every
    letter enciphered by key position *i*, makes each column a plain Caesar
    shift if *p* really is the key length. Caesar shifts preserve IC, so the
    right period shows columns with English-like IC (~0.066) while wrong
    periods stay near random (~0.038).

    Periods that would leave columns shorter than 20 letters are skipped:
    IC is far too noisy to mean anything on samples that small, and reporting
    it anyway is how people talk themselves into wrong key lengths.
    """
    results: list[tuple[int, float]] = []
    for period in range(1, maximum_period + 1):
        if len(text) // period < 20:
            break
        parts = columns(text, period)
        average = sum(index_of_coincidence(part) for part in parts) / period
        results.append((period, average))
    return results


# ---------------------------------------------------------------------------
# The full analysis report
# ---------------------------------------------------------------------------


@dataclass
class Hypothesis:
    """A labelled guess about the cipher family, with its evidence.

    ``confidence`` is one of ``consider``/``possible``/``likely``. Nothing
    stronger exists on purpose: this report never identifies a cipher.
    """

    family: str
    confidence: str
    reason: str
    suggested_commands: tuple[str, ...] = ()


@dataclass
class TextStatistics:
    """Everything :func:`analyse` measured about one piece of ciphertext."""

    original: str
    letters: str
    length: int
    unique_letters: int
    missing_letters: tuple[str, ...]
    counts: Mapping[str, int]
    frequencies: Mapping[str, float]
    ic: float
    ic_normalised: float
    chi_squared: float
    repeats: Mapping[int, list[Repeat]]
    kasiski_votes: Counter
    ic_periods: list[tuple[int, float]]
    length_divisors: list[int]
    length_prime_factors: list[int]
    group_count: int
    uniform_group_length: int | None
    group_length_histogram: Mapping[int, int]
    doubled_letters: int
    digraph_hits: list[tuple[str, int]]
    trigraph_hits: list[tuple[str, int]]
    non_letter_characters: Mapping[str, int]
    hypotheses: list[Hypothesis] = field(default_factory=list)


def analyse(
    source: NormalizedText | str,
    *,
    maximum_repeat_length: int = 5,
    maximum_period: int = 20,
) -> TextStatistics:
    """Measure a ciphertext. Draws no conclusions beyond labelled heuristics."""
    from .normalize import normalize  # local import keeps the module standalone

    normalized = normalize(source) if isinstance(source, str) else source
    letters = normalized.letters
    length = len(letters)

    counts = letter_counts(letters)
    frequencies = letter_frequencies(letters)
    missing = tuple(letter for letter in ALPHABET if counts[letter] == 0)

    repeats = {
        size: find_repeats(letters, size)
        for size in range(2, maximum_repeat_length + 1)
    }

    group_lengths = Counter(len(group) for group in normalized.groups)

    non_letters = Counter(
        char for char in normalized.original if not char.isalpha() and not char.isspace()
    )

    digraph_hits = [
        (gram, letters.count(gram)) for gram in COMMON_DIGRAPHS if gram in letters
    ]
    digraph_hits.sort(key=lambda row: -row[1])
    trigraph_hits = [
        (gram, letters.count(gram)) for gram in COMMON_TRIGRAPHS if gram in letters
    ]
    trigraph_hits.sort(key=lambda row: -row[1])

    doubled = sum(1 for i in range(length - 1) if letters[i] == letters[i + 1])

    stats = TextStatistics(
        original=normalized.original,
        letters=letters,
        length=length,
        unique_letters=len(set(letters)),
        missing_letters=missing,
        counts=counts,
        frequencies=frequencies,
        ic=index_of_coincidence(letters),
        ic_normalised=normalised_ic(letters),
        chi_squared=chi_squared_english(letters),
        repeats=repeats,
        kasiski_votes=kasiski_factor_votes(letters),
        ic_periods=ic_by_period(letters, maximum_period),
        length_divisors=divisors(length),
        length_prime_factors=prime_factors(length),
        group_count=len(normalized.groups),
        uniform_group_length=normalized.uniform_group_length(),
        group_length_histogram=dict(group_lengths),
        doubled_letters=doubled,
        digraph_hits=digraph_hits[:10],
        trigraph_hits=trigraph_hits[:10],
        non_letter_characters=dict(non_letters),
    )
    stats.hypotheses = cipher_family_hypotheses(stats)
    return stats


# ---------------------------------------------------------------------------
# Heuristics -- explicitly NOT conclusions
# ---------------------------------------------------------------------------

#: Below this many letters, IC and chi-squared are too noisy to lean on.
RELIABLE_SAMPLE = 100


def cipher_family_hypotheses(stats: TextStatistics) -> list[Hypothesis]:
    """Suggest which families are worth trying, with the evidence for each.

    Reasoning, in the order a human would apply it:

    1. *Is the letter distribution English?* If chi-squared is small, the
       cipher did not change which letters appear, only where -- that is a
       transposition (or the text is already plaintext).
    2. *Is IC English-like but the distribution scrambled?* Then one fixed
       alphabet was used: Caesar, Atbash, affine, keyword or a general
       monoalphabetic substitution.
    3. *Is IC flattened towards random?* Several alphabets were used:
       Vigenere, Beaufort, autokey, or a digraphic cipher such as Playfair,
       Bifid or Hill.
    4. *Are there structural tells?* A missing J and an even length point at
       Playfair. Only digits point at Polybius. A length with a tidy
       rectangular factorisation invites a grid test.

    Every branch returns a hypothesis with its measurement attached so the
    operator can disagree with it.
    """
    hypotheses: list[Hypothesis] = []
    ic = stats.ic
    chi = stats.chi_squared
    length = stats.length

    if length == 0:
        return [
            Hypothesis(
                "none",
                "consider",
                "The input contains no alphabetic characters. "
                "Check the file, or try the `encodings` command if it is "
                "digits or symbols.",
                ("cipher_tool encodings <file>",),
            )
        ]

    small_sample = length < RELIABLE_SAMPLE
    caveat = (
        f" (only {length} letters -- statistics this short are unreliable)"
        if small_sample
        else ""
    )

    # -- 1. English letter distribution, i.e. letters merely moved ---------
    if chi < 0.12 and ic > 0.058:
        hypotheses.append(
            Hypothesis(
                "Transposition (rail fence, columnar, route/grid)",
                "likely" if not small_sample else "possible",
                f"Letter frequencies already match English (chi2/letter="
                f"{chi:.3f}) and IC={ic:.4f} is English-like. A cipher that "
                f"only rearranges letters leaves both unchanged{caveat}.",
                (
                    "cipher_tool transposition <file>",
                    "cipher_tool analyse <file>  # check the length factors",
                ),
            )
        )
        if chi < 0.05:
            hypotheses.append(
                Hypothesis(
                    "Plaintext or a very weak cipher",
                    "consider",
                    f"chi2/letter={chi:.3f} is extremely close to English. "
                    "Read the normalised text before running any solver.",
                    ("cipher_tool show <file>",),
                )
            )

    # -- 2. English-like IC, scrambled distribution ------------------------
    if ic > 0.058 and chi >= 0.12:
        hypotheses.append(
            Hypothesis(
                "Monoalphabetic substitution (Caesar, Atbash, affine, "
                "keyword, general substitution)",
                "likely" if not small_sample else "possible",
                f"IC={ic:.4f} is close to English ({ENGLISH_IC:.4f}), so one "
                f"fixed alphabet was used, but chi2/letter={chi:.3f} shows the "
                f"letters have been relabelled{caveat}.",
                (
                    "cipher_tool caesar <file>",
                    "cipher_tool affine <file>",
                    "cipher_tool substitution <file>",
                ),
            )
        )

    # -- 3. Flattened IC: several alphabets --------------------------------
    if ic < 0.058:
        # An IC at or below the flat-random value is NOT evidence for a
        # repeating key -- it is the absence of evidence for anything. Text
        # that is genuinely random sits here, and so does a polyalphabetic
        # cipher with a key long enough to flatten the statistics completely.
        # The two are indistinguishable by this measurement, so the report
        # must not prefer the interesting explanation over the dull one.
        essentially_flat = ic <= RANDOM_IC + 0.002

        if essentially_flat:
            strength = "possible"
            reason = (
                f"IC={ic:.4f} is indistinguishable from the flat-random "
                f"value ({RANDOM_IC:.4f}), so the letters are about as "
                "evenly spread as they can be. That is consistent with a "
                "repeating key, but equally with a key as long as the "
                "message, with a cipher this toolkit does not implement, or "
                f"with the text not being English underneath at all{caveat}."
            )
        else:
            strength = "likely" if ic < 0.050 and not small_sample else "possible"
            reason = (
                f"IC={ic:.4f} sits between random ({RANDOM_IC:.4f}) and "
                f"English ({ENGLISH_IC:.4f}), which is what mixing several "
                f"alphabets does{caveat}."
            )

        hypotheses.append(
            Hypothesis(
                "Polyalphabetic (Vigenere, Beaufort, autokey)",
                strength,
                reason,
                (
                    "cipher_tool vigenere <file>",
                    "cipher_tool beaufort <file>",
                    "cipher_tool autokey <file>",
                ),
            )
        )
        hypotheses.append(
            Hypothesis(
                "Digraphic/fractionating (Playfair, Bifid, Hill)",
                "possible",
                f"IC={ic:.4f} is also consistent with a cipher that encrypts "
                "letters in pairs or mixes their coordinates, which flattens "
                "single-letter statistics without a repeating key.",
                (
                    "cipher_tool playfair <file> --key <guess>",
                    "cipher_tool bifid <file> --key <guess>",
                    "cipher_tool hill <file>",
                ),
            )
        )
        if essentially_flat and not small_sample:
            hypotheses.append(
                Hypothesis(
                    "Possibly not an English letter cipher at all",
                    "consider",
                    f"Worth saying plainly: an IC of {ic:.4f} is about what "
                    "random letters give. If every solver comes back weak, the "
                    "likeliest explanations are that the message is not "
                    "English underneath, that it uses a cipher this toolkit "
                    "does not implement, or that the transcription is wrong. "
                    "A long search will not fix any of those.",
                    ("cipher_tool encodings <file>",
                     "cipher_tool show <file>  # check the transcription"),
                )
            )

    # -- 4. Structural tells -----------------------------------------------
    # Kasiski and the column-IC test are independent ways of measuring the
    # same thing. When they agree that is much stronger evidence than either
    # alone, so they are reported as ONE finding rather than two, and only
    # split apart when they actually disagree.
    shortlist = _kasiski_shortlist(stats.kasiski_votes)
    kasiski_top = shortlist[0][0] if shortlist else None

    ic_top: int | None = None
    ic_value = 0.0
    interesting_periods = [row for row in stats.ic_periods if row[0] > 1]
    if interesting_periods:
        candidate_period, candidate_ic = max(
            interesting_periods, key=lambda row: row[1]
        )
        if candidate_ic > 0.058:
            # Prefer the smallest period that also reads as English: if 7
            # works then 14 and 21 must too, and the shortest is the key.
            for period, value in interesting_periods:
                if value > 0.058 and candidate_period % period == 0:
                    candidate_period, candidate_ic = period, value
                    break
            ic_top, ic_value = candidate_period, candidate_ic

    kasiski_note = (
        "Kasiski examination: "
        + ", ".join(f"{k} scored {v}" for k, v in shortlist[:4])
        + ". Distances between repeated ciphertext runs tend to be multiples "
        "of the key length."
        if shortlist
        else ""
    )
    ic_note = (
        f"Splitting into {ic_top} columns gives mean column IC={ic_value:.4f}, "
        "English-like, which is what happens when each column is a single "
        "Caesar shift."
        if ic_top
        else ""
    )

    if kasiski_top and ic_top and kasiski_top == ic_top:
        hypotheses.append(
            Hypothesis(
                f"Repeating key of length {ic_top}",
                "likely" if not small_sample else "possible",
                f"Two independent tests agree. {kasiski_note} {ic_note}",
                (f"cipher_tool vigenere <file> --key-length {ic_top}",),
            )
        )
    else:
        if kasiski_top:
            hypotheses.append(
                Hypothesis(
                    f"Repeating key of length {kasiski_top} (or a multiple)",
                    "possible",
                    kasiski_note,
                    (f"cipher_tool vigenere <file> --key-length {kasiski_top}",),
                )
            )
        if ic_top:
            hypotheses.append(
                Hypothesis(
                    f"Repeating key of length {ic_top}",
                    "possible",
                    ic_note
                    + (
                        f" Note this DISAGREES with the Kasiski favourite "
                        f"({kasiski_top}); try both."
                        if kasiski_top
                        else ""
                    ),
                    (f"cipher_tool vigenere <file> --key-length {ic_top}",),
                )
            )

    if "J" not in stats.letters and length >= 40 and length % 2 == 0:
        hypotheses.append(
            Hypothesis(
                "Playfair",
                "possible",
                "The letter J never appears and the length is even. Playfair "
                "merges I/J into a 25-letter square and always produces an "
                "even number of letters.",
                ("cipher_tool playfair <file> --key <guess>",),
            )
        )

    if stats.unique_letters <= 25 and length >= 40 and stats.doubled_letters == 0:
        hypotheses.append(
            Hypothesis(
                "Playfair",
                "consider",
                "No doubled letters anywhere. Playfair splits doubled letters "
                "with a filler, so a genuine Playfair ciphertext usually has "
                "few or none.",
                ("cipher_tool playfair <file> --key <guess>",),
            )
        )

    if stats.unique_letters <= 6 and length >= 40:
        hypotheses.append(
            Hypothesis(
                "Fractionating cipher over a small alphabet (ADFGX-style, "
                "Polybius)",
                "likely",
                f"Only {stats.unique_letters} distinct symbols are used. That "
                "is the signature of a cipher that writes each letter as a "
                "pair of coordinates.",
                ("cipher_tool polybius <file> --decode",),
            )
        )

    digits = sum(
        count for char, count in stats.non_letter_characters.items() if char.isdigit()
    )
    if digits > stats.length:
        hypotheses.append(
            Hypothesis(
                "Numeric encoding or Polybius coordinates",
                "possible",
                f"The input holds {digits} digits and only {stats.length} "
                "letters. Try the encoding helpers and the Polybius decoder "
                "before any letter-based attack.",
                (
                    "cipher_tool encodings <file>",
                    "cipher_tool polybius <file> --decode",
                ),
            )
        )

    rectangles = [
        divisor
        for divisor in stats.length_divisors
        if 2 <= divisor <= 30 and 2 <= stats.length // divisor
    ]
    if rectangles and ic > 0.058 and chi < 0.20:
        hypotheses.append(
            Hypothesis(
                "Grid/route transposition",
                "consider",
                f"Length {stats.length} factorises as "
                + ", ".join(
                    f"{d}x{stats.length // d}" for d in rectangles[:6]
                )
                + ". Those are the grids a route or columnar cipher could "
                "have used.",
                ("cipher_tool transposition <file> --deep",),
            )
        )

    if not hypotheses:
        hypotheses.append(
            Hypothesis(
                "Inconclusive",
                "consider",
                f"IC={ic:.4f} and chi2/letter={chi:.3f} do not match any "
                "family cleanly. Run the full pipeline and read the "
                "candidates yourself.",
                ("cipher_tool auto <file> --normal",),
            )
        )

    order = {"likely": 0, "possible": 1, "consider": 2}
    hypotheses.sort(key=lambda h: order.get(h.confidence, 3))
    return hypotheses


def _kasiski_shortlist(votes: Counter, limit: int = 6) -> list[tuple[int, int]]:
    """Kasiski candidates, best first, with multiples of a winner demoted.

    If 3 is the real key length then 6, 9 and 12 also divide every gap that 3
    does, so they inherit its votes. Presenting them as separate findings is
    misleading, so a candidate is dropped when a smaller candidate it is a
    multiple of scored at least as well.
    """
    ranked = sorted(votes.items(), key=lambda row: (-row[1], row[0]))
    kept: list[tuple[int, int]] = []
    for factor, count in ranked:
        if any(factor % smaller == 0 and score >= count for smaller, score in kept):
            continue
        kept.append((factor, count))
        if len(kept) >= limit:
            break
    return kept


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_report(stats: TextStatistics, *, width: int = 76) -> str:
    """Format the full analyse report."""
    lines: list[str] = []

    def heading(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    heading("Input")
    lines.append(f"Alphabetic characters : {stats.length}")
    lines.append(f"Distinct letters used : {stats.unique_letters} of 26")
    if stats.missing_letters:
        lines.append(f"Letters never used    : {' '.join(stats.missing_letters)}")
    lines.append(f"Whitespace groups     : {stats.group_count}")
    if stats.uniform_group_length is not None:
        lines.append(
            f"Group length          : all groups are "
            f"{stats.uniform_group_length} characters -- almost certainly "
            "transcription formatting, NOT word boundaries"
        )
    else:
        histogram = ", ".join(
            f"{size}x{count}"
            for size, count in sorted(stats.group_length_histogram.items())
        )
        lines.append(f"Group lengths         : mixed ({histogram})")
    if stats.non_letter_characters:
        shown = ", ".join(
            f"{char!r}x{count}"
            for char, count in sorted(
                stats.non_letter_characters.items(), key=lambda r: -r[1]
            )[:10]
        )
        lines.append(f"Non-letter characters : {shown}")

    heading("Letter frequencies")
    ordered = sorted(stats.counts.items(), key=lambda row: (-row[1], row[0]))
    for letter, count in ordered:
        if count == 0:
            continue
        percent = stats.frequencies[letter]
        expected = ENGLISH_LETTER_FREQUENCY[letter]
        bar = "#" * min(40, int(round(percent * 2)))
        lines.append(
            f"  {letter}  {count:5d}  {percent:5.2f}%  "
            f"(English {expected:5.2f}%)  {bar}"
        )

    heading("Index of Coincidence")
    lines.append(f"  IC                  : {stats.ic:.5f}")
    lines.append(f"  IC x 26             : {stats.ic_normalised:.4f}")
    lines.append(f"  English reference   : {ENGLISH_IC:.5f}")
    lines.append(f"  Random reference    : {RANDOM_IC:.5f}")
    lines.append(f"  chi2 vs English     : {stats.chi_squared:.4f} per letter")
    lines.append(
        "  Reading: IC near English => one alphabet (or a transposition); "
        "IC near random => several alphabets."
    )
    lines.append(
        "           chi2 small => the letters themselves were not changed, "
        "only moved."
    )

    heading("Repeated sequences")
    any_repeats = False
    for size in sorted(stats.repeats):
        found = stats.repeats[size]
        if not found:
            continue
        any_repeats = True
        lines.append(f"  {size}-grams repeated: {len(found)}")
        for repeat in found[:8]:
            places = ", ".join(str(p) for p in repeat.positions[:8])
            gaps = ", ".join(str(d) for d in repeat.distances[:8])
            lines.append(
                f"    {repeat.text:<6} x{repeat.count}  at [{places}]"
                + (f"  gaps [{gaps}]" if gaps else "")
            )
        if len(found) > 8:
            lines.append(f"    ... and {len(found) - 8} more")
    if not any_repeats:
        lines.append("  None. That argues against a short repeating key.")

    heading("Kasiski examination (repeating-key evidence)")
    shortlist = _kasiski_shortlist(stats.kasiski_votes)
    if shortlist:
        lines.append("  Candidate key length : weighted votes from gap divisors")
        for factor, count in shortlist:
            bar = "#" * min(40, count // max(1, shortlist[0][1] // 30 or 1))
            lines.append(f"    {factor:3d} : {count:5d}  {bar}")
        lines.append(
            "  Votes are divisors of the gaps between repeated runs, weighted "
            "by run length. Multiples of a stronger candidate are hidden."
        )
    else:
        lines.append("  No usable repeats, so Kasiski has nothing to say.")

    heading("Index of Coincidence by period")
    if stats.ic_periods:
        lines.append("  Period : mean IC of the columns")
        best = max(stats.ic_periods, key=lambda row: row[1])[1]
        for period, value in stats.ic_periods:
            marker = "  <== highest" if value == best and period > 1 else ""
            bar = "#" * int(round((value - 0.030) * 800))
            lines.append(f"    {period:3d} : {value:.4f}  {bar}{marker}")
        lines.append(
            "  A period whose columns read as English (~0.067) is a candidate "
            "key length. Periods leaving fewer than 20 letters per column are "
            "not shown: the measurement would be meaningless."
        )
    else:
        lines.append("  Text too short to split into columns meaningfully.")

    heading("Length factors")
    lines.append(f"  Length            : {stats.length}")
    lines.append(
        f"  Prime factors     : "
        + (" x ".join(str(p) for p in stats.length_prime_factors) or "-")
    )
    lines.append(
        "  Divisors          : "
        + ", ".join(str(d) for d in stats.length_divisors)
    )
    grids = [
        f"{d}x{stats.length // d}"
        for d in stats.length_divisors
        if 2 <= d <= 40 and stats.length // d >= 2
    ]
    lines.append("  Possible grids    : " + (", ".join(grids[:14]) or "-"))
    lines.append(
        f"  Even length       : {'yes' if stats.length % 2 == 0 else 'no'} "
        "(Playfair and other digraph ciphers always give an even count)"
    )

    heading("Common English patterns present")
    lines.append(
        "  Digraphs : "
        + (
            ", ".join(f"{gram}x{count}" for gram, count in stats.digraph_hits)
            or "-"
        )
    )
    lines.append(
        "  Trigraphs: "
        + (
            ", ".join(f"{gram}x{count}" for gram, count in stats.trigraph_hits)
            or "-"
        )
    )
    lines.append(f"  Doubled letters: {stats.doubled_letters}")
    lines.append(
        "  Note: for an unsolved ciphertext these counts are coincidence, not "
        "meaning. They matter only when comparing candidate plaintexts."
    )

    heading("HEURISTIC cipher-family suggestions")
    lines.append(
        "  These are suggestions of what to TRY. They are not an "
        "identification, and the toolkit is often wrong about them."
    )
    for number, hypothesis in enumerate(stats.hypotheses, start=1):
        lines.append("")
        lines.append(f"  [{number}] {hypothesis.family}  ({hypothesis.confidence})")
        for line in _wrap(hypothesis.reason, width - 8):
            lines.append(f"      {line}")
        for command in hypothesis.suggested_commands:
            lines.append(f"      $ {command}")

    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    """Wrap on word boundaries, for prose in the report only."""
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def summarise(stats: TextStatistics) -> str:
    """One-line summary used by the auto pipeline."""
    return (
        f"{stats.length} letters, {stats.unique_letters} distinct, "
        f"IC={stats.ic:.4f}, chi2/letter={stats.chi_squared:.3f}"
    )
