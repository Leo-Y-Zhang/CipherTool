"""A digraph cipher whose two letters have their coordinates CROSSED.

The gap this closes
-------------------
``paired.py`` recognises a stream written in a paired alphabet -- it will tell
you, unaided, that a message is 3,322 playing cards alternating between a
36-card sub-deck and a 16-card sub-deck, and that the unit is two cards. Then
it stops, because it is a recogniser and says so. Nothing downstream could read
that shape, so the only honest output was a refusal, and the refusal is what
the paste screen printed.

This module is the missing solver.

The construction
----------------
A unit is two cells and carries one plaintext DIGRAPH. Index the first cell
``0..35`` as ``6 * p + q`` and the second ``0..15`` as ``4 * u + v``. Then

    first plaintext letter  = SQUARE_ONE[q][u]
    second plaintext letter = SQUARE_TWO[p][v]

so each cell carries one coordinate of each letter, and the two letters'
coordinates are interleaved across the two cells. That crossing is the whole
cipher, and it is also the way in.

Why it is solvable when a codebook is not
-----------------------------------------
36 x 16 = 576 units against ~1,661 units of message is under two observations
per cell. Read as 576 independent codebook entries the problem is
underdetermined, and a blind search over it produces confident garbage -- the
exact failure this toolkit exists to prevent.

But 576 = 24 x 24, and the crossing means the unknowns are not 576 cells but
two 24-letter squares: **48 unknowns, not 576**. 1,661 units is far more than
enough for 48. The construction is the way in, not more text.

The attack, in two stages, the first of which needs no key at all
----------------------------------------------------------------
1. RECOGNISE THE FRACTIONATION. Split each cell index into a high and a low
   coordinate and try each way of pairing one coordinate of the first cell
   with one of the second. Under the TRUE pairing each half of the plaintext
   is a plain monoalphabetic substitution, so its index of coincidence is
   English's 0.066; under any other pairing the classes are mixtures and the
   distribution is flatter. Measured on the 2025 material: 0.0669 for the true
   split against 0.0445 for the worst, over 192 candidate readings. No key is
   involved, so this stage is free and it cannot be fooled by search size.

2. CLIMB THE TWO SQUARES. With the split known, the odd and even plaintext
   positions are two independent 24-symbol substitutions read alternately.

Plain hill-climbing is NOT enough here and the measurement says so: 12 restarts
of 20,000 steps reached -1.78 per letter where the true key scores -0.9478,
because a swap inside one square is judged through the other square's noise.
Simulated annealing with a frequency-ranked start reaches the true key on the
first restart.

What this module does NOT do
----------------------------
It does not search cell orderings freely. The cell index has to be built from
an ORDER on each of the two symbol alphabets, and only a bounded family of
orders is tried (ascending, descending, and the two playing-card conventions
when the symbols look like card ranks). That is a real bound, stated here
rather than hidden: a message whose cells are ordered by some private
convention will not be recognised, and the module will refuse rather than
guess.
"""

from __future__ import annotations

import collections
import itertools
import math
import random
from dataclasses import dataclass, field

#: Fewer units than this and the index of coincidence is noise, so the
#: recognition stage cannot separate a true split from a lucky one.
MINIMUM_UNITS = 300

#: A split is only considered when both halves have a plausible number of
#: classes for an alphabet. 24 is the usual reduced alphabet (no J, no Z);
#: the band allows the neighbouring conventions without opening the search up
#: to arbitrary factorisations.
ALPHABET_RANGE = (20, 27)

#: Both halves must reach this index of coincidence. English is 0.066; a
#: mixture of classes falls away sharply, and the worst of the 192 readings of
#: the 2025 material scored 0.0445.
#:
#: MEASURED against message length, which is why the bar sits at 0.060 and not
#: lower: the true split scores 0.0669 at 1,661 units, 0.0641 at 600 and only
#: 0.0580 at 300. Below about 500 units the sampling noise swallows the signal
#: AND the recovery is garbage anyway (8% of letters right), so a bar in the
#: gap turns a length floor into a measurement rather than a magic number.
MINIMUM_IC = 0.060

#: Below this many units the attack was measured recovering a NEAR-MISS rather
#: than the key: 600 units gave 97.5% of letters right at 0.72 word coverage,
#: which is exactly the shape of reading that clears the `strong` thresholds
#: and is still wrong. 900 units and above recovered every letter. A reading
#: built from less than this is capped, not trusted.
CONFIDENT_UNITS = 900

#: A run that simply missed the key is capped here. MEASURED: recovered
#: readings sit at 0.81-0.83 word coverage, failed ones at 0.27-0.31, so the
#: band between is empty and the floor can sit in it.
RELIABLE_WORD_COVERAGE = 0.60

#: Annealing steps per restart. Fixed rather than timed ON PURPOSE: a
#: randomised search asserted to succeed inside a time budget is a flaky test
#: by construction, and this repository has already been bitten by one. A
#: step count is reproducible from the seed on any machine; a deadline is not.
#:
#: MEASURED on the 1,661-unit 2025 message, four seeds per setting: 8,000
#: steps recovered the key exactly 4 times out of 4 in 5.9s per restart,
#: 15,000 in 11.6s, 25,000 in 18.7s. 20,000 leaves room for a shorter message
#: without making the common case slow.
STEPS_PER_RESTART = 20000

#: Peak annealing temperature, in units of total log-probability. The schedule
#: cools to near zero and reheats twice per restart.
PEAK_TEMPERATURE = 40.0

#: English letters in descending frequency, for the frequency-ranked start.
BY_FREQUENCY = "ETAOINSHRDLCUMWFGYPBVKJXQZ"

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: Playing-card ranks, ace low and ace high. Recognised so that a deck laid
#: out in the ordinary way is indexed the ordinary way.
CARD_RANKS_ACE_LOW = "A23456789XJQK"
CARD_RANKS_ACE_HIGH = "23456789XJQKA"

METHOD = "Digraph cipher with crossed coordinates"


@dataclass(frozen=True)
class Split:
    """A recognised fractionation: how the two cells carry the two letters.

    Attributes
    ----------
    first_classes, second_classes:
        For each unit, the class that decides the first plaintext letter and
        the class that decides the second. Each is a list as long as the
        message in units.
    ic_first, ic_second:
        Index of coincidence of the two class streams. English is 0.066.
    shape:
        A short human-readable description of the indexing that produced this
        split, so the answer can be checked by hand.
    """

    first_classes: tuple[tuple[int, int], ...]
    second_classes: tuple[tuple[int, int], ...]
    ic_first: float
    ic_second: float
    shape: str
    first_extent: tuple[int, int] = (0, 0)
    second_extent: tuple[int, int] = (0, 0)

    @property
    def score(self) -> float:
        """The weaker of the two halves, which is what the split is judged on.

        Taking the minimum rather than the mean matters: a reading that gets
        one half right and mixes the other is not half a solution, it is a
        wrong split with a lucky half.
        """
        return min(self.ic_first, self.ic_second)


@dataclass
class Reading:
    """A recovered reading: the plaintext plus the two squares that made it.

    Attributes
    ----------
    plaintext:
        The decoded letters.
    score:
        Total log-probability under the English model, not per letter.
    square_one, square_two:
        Class-to-letter maps for the first and second letter of each digraph.
    split:
        The fractionation the reading was built on.
    """

    plaintext: str
    score: float
    square_one: dict = field(default_factory=dict)
    square_two: dict = field(default_factory=dict)
    split: Split | None = None
    extent_one: tuple[int, int] = (0, 0)
    extent_two: tuple[int, int] = (0, 0)

    def describe_key(self) -> str:
        """The key as a line of text, so a reader can check it by hand.

        A competition answer nobody can check by hand is worth very little, so
        both squares are printed in coordinate order rather than summarised.

        A cell the message never used is printed as ``.`` rather than skipped.
        Skipping it silently shortens the square and every letter after the
        hole then reads at the wrong coordinate -- a key that cannot be
        checked by hand, dressed up as one that can.
        """
        def render(square: dict, extent: tuple[int, int]) -> str:
            rows, columns = extent
            if not rows or not columns:
                return "".join(ALPHABET[square[key]] for key in sorted(square))
            return "".join(
                ALPHABET[square[(row, column)]] if (row, column) in square
                else "."
                for row in range(rows) for column in range(columns)
            )

        return (f"{self.split.shape if self.split else 'unknown split'}; "
                f"square1={render(self.square_one, self.extent_one)} "
                f"square2={render(self.square_two, self.extent_two)}")


def encipher(plaintext: str, square_one: str, square_two: str,
             cells_one, cells_two, *, width_one: int, width_two: int) -> str:
    """Encipher *plaintext* with two squares onto two alphabets of cells.

    Present so that the family can be exercised end to end rather than only
    on the one real message: a solver measured against nothing but its own
    single example is measured against nothing.

    ``square_one`` and ``square_two`` are the two reduced alphabets read row
    by row; ``cells_one`` and ``cells_two`` are the two cell alphabets, in
    index order. An odd final letter is dropped rather than padded, because
    padding invents a letter the message does not contain.
    """
    letters = "".join(c for c in plaintext.upper() if c.isalpha())
    height_one = len(square_one) // width_one
    height_two = len(square_two) // width_two
    # The crossing fixes the shapes against each other: the first cell
    # alphabet is (rows of square two) x (rows of square one) and the second
    # is (columns of square one) x (columns of square two).
    if len(cells_one) != height_two * height_one:
        raise ValueError("first cell alphabet is the wrong size")
    if len(cells_two) != width_one * width_two:
        raise ValueError("second cell alphabet is the wrong size")
    out = []
    for index in range(0, len(letters) - 1, 2):
        first, second = letters[index], letters[index + 1]
        if first not in square_one or second not in square_two:
            raise ValueError(f"{first!r}{second!r} is not in the squares")
        q, u = divmod(square_one.index(first), width_one)
        p, v = divmod(square_two.index(second), width_two)
        out.append(cells_one[p * height_one + q])
        out.append(cells_two[u * width_two + v])
    return "".join(out)


def decipher(ciphertext: str, square_one: str, square_two: str,
             cells_one, cells_two, *, width_one: int, width_two: int) -> str:
    """Invert :func:`encipher`. The inverse is written out, not assumed.

    A test built on an involution cannot detect a missing inversion, so the
    two directions are separate functions and the tests use squares that are
    not their own inverse.
    """
    height_one = len(square_one) // width_one
    index_one = {cell: i for i, cell in enumerate(cells_one)}
    index_two = {cell: i for i, cell in enumerate(cells_two)}
    cells = [ciphertext[i:i + 2] for i in range(0, len(ciphertext) - 1, 2)]
    out = []
    for i in range(0, len(cells) - 1, 2):
        first, second = cells[i], cells[i + 1]
        if first not in index_one or second not in index_two:
            raise ValueError(f"{first!r} or {second!r} is not a known cell")
        p, q = divmod(index_one[first], height_one)
        u, v = divmod(index_two[second], width_two)
        out.append(square_one[q * width_one + u])
        out.append(square_two[p * width_two + v])
    return "".join(out)


def index_of_coincidence(values) -> float:
    """Probability that two items drawn from *values* are equal.

    This is the statistic that separates a true fractionation from a false
    one without knowing any key: under the true split each half is a
    monoalphabetic substitution and keeps English's 0.066, while a false split
    merges classes and flattens the distribution.
    """
    counts: collections.Counter = collections.Counter(values)
    total = sum(counts.values())
    if total < 2:
        return 0.0
    return sum(n * (n - 1) for n in counts.values()) / (total * (total - 1))


def orderings(symbols) -> list[str]:
    """Candidate orders for one symbol alphabet, commonest conventions first.

    Deliberately a SHORT list. The cell index depends on the order chosen for
    each of the two symbol alphabets, and searching orders freely would mean
    factorial work and, worse, a search large enough to find a "split" in
    noise. Ascending and descending cover the ordinary cases; the two card
    conventions cover a deck, where neither ascending nor descending ASCII is
    the order anybody means.
    """
    symbols = sorted(set(symbols))
    found = ["".join(symbols), "".join(reversed(symbols))]
    for convention in (CARD_RANKS_ACE_LOW, CARD_RANKS_ACE_HIGH):
        if set(symbols) <= set(convention):
            candidate = "".join(c for c in convention if c in set(symbols))
            if candidate not in found:
                found.append(candidate)
    return found


def _factorisations(size: int) -> list[tuple[int, int]]:
    return [(a, size // a) for a in range(2, size) if size % a == 0]


def _cell_indices(cells, first_order: str, second_order: str,
                  rank_major: bool) -> dict[str, int] | None:
    """Global index for every cell under one ordering convention."""
    width = len(second_order)
    height = len(first_order)
    out = {}
    for cell in set(cells):
        if len(cell) != 2:
            return None
        head, tail = cell[0], cell[1]
        if head not in first_order or tail not in second_order:
            return None
        if rank_major:
            out[cell] = first_order.index(head) * width + second_order.index(tail)
        else:
            out[cell] = second_order.index(tail) * height + first_order.index(head)
    return out


def detect(cells) -> Split | None:
    """Find the fractionation of a paired-cell stream, or None.

    *cells* is the message as a flat list of two-symbol cells, alternating
    between the two sub-alphabets. Returns the best-scoring split when it
    clears :data:`MINIMUM_IC`, and None otherwise -- which must stay cheap,
    because this runs speculatively on anything that looks paired.
    """
    cells = list(cells)
    firsts, seconds = cells[0::2], cells[1::2]
    units = min(len(firsts), len(seconds))
    if units < MINIMUM_UNITS:
        return None
    set_one, set_two = set(firsts), set(seconds)
    if set_one & set_two:
        return None

    heads = {c[0] for c in cells if len(c) == 2}
    tails = {c[1] for c in cells if len(c) == 2}
    if len(heads) + len(tails) == 0:
        return None

    best: Split | None = None
    for first_order in orderings(heads):
        for second_order in orderings(tails):
            for rank_major in (False, True):
                table = _cell_indices(cells, first_order, second_order,
                                      rank_major)
                if table is None:
                    continue
                for rank_one, size_one in _rankings(set_one, table):
                    for rank_two, size_two in _rankings(set_two, table):
                        best = _search_splits(
                            firsts, seconds, units, rank_one, size_one,
                            rank_two, size_two, first_order, second_order,
                            rank_major, best)
    if best is None or best.score < MINIMUM_IC:
        return None
    return best


def _rankings(cells, table):
    """The two ways to number a sub-alphabet, both bounded and both tried.

    Ranking within the OBSERVED cells is right when every cell of the
    sub-alphabet turns up, which on a long message it does. It is wrong the
    moment one does not: every rank after the gap shifts by one and the
    structure disappears.

    Ranking by position within the sub-alphabet's own CONTIGUOUS RANGE
    survives a missing cell, and it matches how these sub-alphabets are built
    -- the 2025 message splits an ordinary 52-card deck into its first 36
    cards and its last 16. It assumes contiguity, which the other ranking does
    not, so both are offered rather than one being chosen in advance.
    """
    ordered = sorted(cells, key=lambda c: table[c])
    dense = ({c: i for i, c in enumerate(ordered)}, len(ordered))
    low, high = table[ordered[0]], table[ordered[-1]]
    spread = ({c: table[c] - low for c in cells}, high - low + 1)
    return [dense] if dense[1] == spread[1] else [dense, spread]


def _search_splits(firsts, seconds, units, rank_one, size_one, rank_two,
                   size_two, first_order, second_order, rank_major, best):
    """Try every factorisation and pairing for one numbering. Returns the best."""
    for a, b in _factorisations(size_one):
        parts_one = [(rank_one[x] // b, rank_one[x] % b)
                     for x in firsts[:units]]
        for c, d in _factorisations(size_two):
            parts_two = [(rank_two[y] // d, rank_two[y] % d)
                         for y in seconds[:units]]
            sizes = ((a, b), (c, d))
            for i, j in itertools.product((0, 1), repeat=2):
                classes = sizes[0][i] * sizes[1][j]
                other = sizes[0][1 - i] * sizes[1][1 - j]
                if not (ALPHABET_RANGE[0] <= classes <= ALPHABET_RANGE[1]):
                    continue
                if not (ALPHABET_RANGE[0] <= other <= ALPHABET_RANGE[1]):
                    continue
                left = tuple((p[i], q[j])
                             for p, q in zip(parts_one, parts_two))
                right = tuple((p[1 - i], q[1 - j])
                              for p, q in zip(parts_one, parts_two))
                split = Split(
                    first_classes=left, second_classes=right,
                    ic_first=index_of_coincidence(left),
                    ic_second=index_of_coincidence(right),
                    shape=(f"cells indexed {first_order}x{second_order}"
                           f"{' rank-major' if rank_major else ''}, "
                           f"first cell {a}x{b}, second cell {c}x{d}, "
                           f"letter one from "
                           f"({'hi' if i == 0 else 'lo'},"
                           f"{'hi' if j == 0 else 'lo'})"),
                    first_extent=(sizes[0][i], sizes[1][j]),
                    second_extent=(sizes[0][1 - i], sizes[1][1 - j]),
                )
                if best is None or split.score > best.score:
                    best = split
    return best


def _interleave(left, right, square_one, square_two, buffer):
    index = 0
    for lc, rc in zip(left, right):
        buffer[index] = square_one[lc]
        buffer[index + 1] = square_two[rc]
        index += 2
    return buffer


def _frequency_start(stream) -> dict:
    counts = collections.Counter(stream)
    order = [c for c, _ in counts.most_common()]
    return {cls: ALPHABET.index(BY_FREQUENCY[i]) for i, cls in enumerate(order)}


def _anneal(left, right, scorer, restarts: int, steps: int, rng) -> Reading:
    """Simulated annealing over the two squares at once."""
    classes_one = sorted(set(left))
    classes_two = sorted(set(right))
    buffer = [0] * (2 * len(left))
    best: Reading | None = None

    for attempt in range(restarts):
        square_one = _frequency_start(left)
        square_two = _frequency_start(right)
        if attempt:
            for square, keys in ((square_one, classes_one),
                                 (square_two, classes_two)):
                for _ in range(6):
                    x, y = rng.choice(keys), rng.choice(keys)
                    square[x], square[y] = square[y], square[x]
        current = scorer.score_values(
            _interleave(left, right, square_one, square_two, buffer))
        local = (current, dict(square_one), dict(square_two))

        cycle = max(steps // 3, 1)
        for step in range(steps):
            phase = (step % cycle) / cycle
            temperature = PEAK_TEMPERATURE * (1.0 - phase) ** 2 + 0.4

            use_first = rng.random() < 0.5
            square = square_one if use_first else square_two
            keys = classes_one if use_first else classes_two
            key = rng.choice(keys)
            if rng.random() < 0.12:
                # Bring in a letter the square is not using. Which letters the
                # reduced alphabet drops is not known in advance, so the search
                # has to be able to change its mind about it.
                unused = [v for v in range(26) if v not in square.values()]
                if not unused:
                    continue
                previous, square[key] = square[key], rng.choice(unused)
                undo = (None, key, previous)
            else:
                other = rng.choice(keys)
                if other == key:
                    continue
                square[key], square[other] = square[other], square[key]
                undo = (other, key, None)

            candidate = scorer.score_values(
                _interleave(left, right, square_one, square_two, buffer))
            delta = candidate - current
            if delta >= 0 or rng.random() < math.exp(delta / temperature):
                current = candidate
                if current > local[0]:
                    local = (current, dict(square_one), dict(square_two))
            elif undo[0] is None:
                square[undo[1]] = undo[2]
            else:
                square[undo[1]], square[undo[0]] = (square[undo[0]],
                                                    square[undo[1]])

        if best is None or local[0] > best.score:
            text = "".join(ALPHABET[i] for i in _interleave(
                left, right, local[1], local[2], buffer))
            best = Reading(plaintext=text, score=local[0],
                           square_one=local[1], square_two=local[2])
    assert best is not None
    return best


def recover(split: Split, *, scorer=None, restarts: int = 3,
            steps: int = STEPS_PER_RESTART, seed: int | None = None) -> Reading:
    """Climb the two squares for a recognised split and return the reading.

    The recogniser cannot say WHICH half carries the first letter of the
    digraph -- the two orderings tie exactly on the index of coincidence,
    because swapping them swaps two equally valid readings. So both are
    screened with one restart each and the remaining restarts go to the
    better, which is the same screen-then-refine shape the double columnar
    search uses and for the same reason: depth-first on the wrong shape spends
    the whole budget arriving nowhere.
    """
    from .scoring import default_scorer

    engine = scorer if scorer is not None else default_scorer()
    rng = random.Random(0 if seed is None else seed)

    orders = (
        (split.first_classes, split.second_classes,
         split.first_extent, split.second_extent),
        (split.second_classes, split.first_classes,
         split.second_extent, split.first_extent),
    )
    screened = [
        (_anneal(left, right, engine, 1, steps, rng), left, right, one, two)
        for left, right, one, two in orders
    ]
    screened.sort(key=lambda row: row[0].score, reverse=True)
    best, left, right, extent_one, extent_two = screened[0]
    if restarts > 1:
        deeper = _anneal(left, right, engine, restarts - 1, steps, rng)
        if deeper.score > best.score:
            best = deeper
    best.split = split
    best.extent_one = extent_one
    best.extent_two = extent_two
    return best


def solve(source, *, scorer=None, top: int = 5, seed: int | None = None,
          restarts: int = 3, time_budget: float | None = None, **options):
    """Read a paired-cell digraph cipher, or return nothing.

    Returns an EMPTY candidate set -- never a best guess -- when no
    fractionation clears its recognition bar. That is the common case and it
    has to stay cheap and silent, because a stage that always proposes
    something turns every message into this cipher.

    ``time_budget`` is accepted and ignored on purpose: the search is bounded
    by a step count, so it is reproducible from the seed, and a stage that
    RAISES on ``time_budget`` gets silently dropped from the pipeline. That
    has happened in this codebase before and cost a whole challenge.
    """
    from .candidates import Candidate, CandidateSet
    from .normalize import normalize
    from .scoring import annotate, default_scorer

    text = normalize(source) if isinstance(source, str) else source
    symbols = getattr(text, "symbols", "")
    letters = getattr(text, "letters", "")
    empty = CandidateSet(source_letters=letters)
    if len(symbols) < 4 or len(symbols) % 2:
        return empty

    cells = [symbols[i:i + 2] for i in range(0, len(symbols) - 1, 2)]
    split = detect(cells)
    if split is None:
        return empty

    engine = scorer if scorer is not None else default_scorer()
    reading = recover(split, scorer=engine, restarts=restarts, seed=seed)
    diagnostics = {
        "split_shape": split.shape,
        "split_ic_first": round(split.ic_first, 5),
        "split_ic_second": round(split.ic_second, 5),
        "units": len(split.first_classes),
        "search": (
            f"annealing, not exhaustive: {restarts} restart(s) of "
            f"{STEPS_PER_RESTART} moves over two squares"
        ),
    }
    units = len(split.first_classes)
    if units < CONFIDENT_UNITS:
        diagnostics["confidence_cap"] = "promising"
        diagnostics["thin_ciphertext"] = (
            f"{units} units, below the {CONFIDENT_UNITS} at which this attack "
            "was measured recovering every letter. At 600 units it recovered "
            "97.5% of them, which reads as English and is not the plaintext"
        )
    annotate(diagnostics, reading.plaintext, engine)

    # `annotate` is what measures word coverage, so this follows it. The cap
    # above only knows whether there was enough ciphertext to try; this one
    # knows whether the search actually arrived.
    coverage = diagnostics.get("word_coverage")
    if coverage is not None and coverage < RELIABLE_WORD_COVERAGE:
        diagnostics["confidence_cap"] = "weak"
        diagnostics["low_word_coverage"] = (
            f"{coverage:.2f} word coverage, below the "
            f"{RELIABLE_WORD_COVERAGE:.2f} that separated a recovered key "
            "from a missed one when this attack was measured"
        )

    found = CandidateSet(source_letters=letters)
    found.add(Candidate(
        method=METHOD,
        key=reading.describe_key(),
        score=engine.score(reading.plaintext),
        plaintext=reading.plaintext,
        diagnostics=diagnostics,
    ))
    return found
