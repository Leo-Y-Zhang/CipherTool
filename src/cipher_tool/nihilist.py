"""The Nihilist cipher: a Polybius square plus a repeating additive key.

The gap this closes
-------------------
Three files in this repository already NAME the Nihilist cipher as a
competition staple that arrives as digits. None of them could read one.
``polybius`` reads a stream of single digits; ``encodings`` recognises number
bases; neither can read *"97 26 57 58 105 68 57 69"*, because the tokens are
whitespace-separated, of VARIABLE WIDTH, and run past 99.

That is the fourth variant of a defect this project keeps rediscovering. The
first three were: a numeric ciphertext read as an empty paste, "no letters"
and "nothing pasted" printed identically, and a mixed letter/digit message
having its digits stripped in silence. This one is the same shape again --
the message is right there and the reader cannot see it -- and it costs
2023 challenge 9B, 1,494 tokens of real competition material.

How the cipher works
--------------------
Both the plaintext letter and the key letter are looked up in the same 5x5
Polybius square and written as a two-digit coordinate, row then column, each
digit 1-5. The ciphertext number is their SUM::

    ciphertext = (10 * row_p + col_p) + (10 * row_k + col_k)

There is no carry between the digits, so the sums are not arbitrary: the tens
digit runs 2..10 and the units digit runs 2..10 independently. Every value
ending in 1, and every value ending in 11, is IMPOSSIBLE.

Why it falls apart without a key
--------------------------------
That impossibility is the whole attack, and it is a CONSTRAINT rather than a
score, which is what makes it decisive rather than suggestive.

Split the message by a candidate period. Every value in one class shares the
same key coordinate ``k``, so ``value - k`` must be a valid coordinate for
EVERY value in the class. Sweep ``k`` over the 25 coordinates and keep the
ones that survive. A wrong period mixes key letters, the class spreads wider
than any single ``k`` can cover, and NO ``k`` survives -- the period is
excluded outright, not merely scored low.

MEASURED on 2023 9B, periods 1 to 14: every period except 7 and its multiple
14 left at least one class with zero feasible keys, and period 7 left exactly
ONE feasible key in each of its seven classes. The key is not searched for; it
is deduced.

Subtracting it leaves a plain monoalphabetic substitution over 25 cells, which
the substitution solver already handles, and that solve also recovers the
square -- so the answer can be checked by hand, square and key word and all.
"""

from __future__ import annotations

import collections
import itertools
import re
from dataclasses import dataclass

#: Fewer tokens than this and a class holds too few values to pin a key down,
#: so the constraint stops being a constraint.
MINIMUM_TOKENS = 40

#: Periods searched. Competition keys are keywords, so this is generous.
MAX_PERIOD = 20

#: When more than one key survives in several classes the combinations are
#: enumerated and scored. Past this many the module refuses rather than
#: picking one, because choosing among thousands of readings on a statistic
#: is how a search manufactures an answer. A real bound, stated not hidden.
MAX_COMBINATIONS = 20000

METHOD = "Nihilist (Polybius square plus additive key)"

_TOKEN = re.compile(r"\d+")


@dataclass(frozen=True)
class Recovery:
    """A recovered Nihilist key.

    Attributes
    ----------
    period:
        Length of the additive key.
    key_coordinates:
        The key's Polybius coordinates, one per position in the period.
    cells:
        The plaintext coordinates left after subtracting the key.
    """

    period: int
    key_coordinates: tuple[int, ...]
    cells: tuple[int, ...]

    def index_of_coincidence(self) -> float:
        """Index of coincidence of the recovered cells.

        Reported because it is the check that the subtraction produced a
        monoalphabetic stream rather than a mixture: English sits near 0.066
        and a wrong key falls towards 1/25.
        """
        counts = collections.Counter(self.cells)
        total = sum(counts.values())
        if total < 2:
            return 0.0
        return sum(n * (n - 1) for n in counts.values()) / (total * (total - 1))


def coordinates(size: int = 5) -> list[int]:
    """Every valid Polybius coordinate as a two-digit number, row then column."""
    return [10 * row + column
            for row in range(1, size + 1)
            for column in range(1, size + 1)]


def is_coordinate(value: int, size: int = 5) -> bool:
    """True when *value* is a valid row/column coordinate with no carry."""
    return 1 <= value // 10 <= size and 1 <= value % 10 <= size


def parse(text: str) -> list[int]:
    """Pull whitespace-separated numbers out of *text*, in order.

    Takes the RAW text rather than a normalised view on purpose. Normalising
    keeps the digits and throws the separators away, which turns
    ``"97 26 57"`` into ``"972657"`` -- and a Nihilist ciphertext's token
    boundaries are the only thing that makes it readable, because the tokens
    are not a fixed width.
    """
    return [int(match.group(0)) for match in _TOKEN.finditer(text)]


def feasible_keys(values, size: int = 5) -> list[int]:
    """Key coordinates under which every value in *values* decodes validly.

    This is the constraint the whole attack rests on, and it is worth being
    precise about why it is strong: it does not ask which key is LIKELIEST, it
    asks which keys are POSSIBLE. A statistic can be fooled by search size; a
    value ending in 1 cannot be a Nihilist sum whatever the key.
    """
    return [key for key in coordinates(size)
            if all(is_coordinate(value - key, size) for value in values)]


def encrypt(plaintext: str, square: str, key: str) -> list[int]:
    """Encipher with an explicit square and key, for testing against.

    *square* is the 25 letters of the Polybius square read row by row.
    """
    letters = [c for c in plaintext.upper() if c in square]
    if not key:
        raise ValueError("a Nihilist key cannot be empty")
    key_coords = []
    for character in key.upper():
        if character not in square:
            raise ValueError(f"{character!r} is not in the square")
        index = square.index(character)
        key_coords.append(10 * (index // 5 + 1) + (index % 5 + 1))
    out = []
    for position, character in enumerate(letters):
        index = square.index(character)
        coordinate = 10 * (index // 5 + 1) + (index % 5 + 1)
        out.append(coordinate + key_coords[position % len(key_coords)])
    return out


def decrypt(values, square: str, key: str) -> str:
    """Invert :func:`encrypt`. Written out rather than inferred."""
    key_coords = []
    for character in key.upper():
        index = square.index(character)
        key_coords.append(10 * (index // 5 + 1) + (index % 5 + 1))
    out = []
    for position, value in enumerate(values):
        coordinate = value - key_coords[position % len(key_coords)]
        if not is_coordinate(coordinate):
            raise ValueError(f"{value} does not decode at position {position}")
        row, column = divmod(coordinate, 10)
        out.append(square[(row - 1) * 5 + (column - 1)])
    return "".join(out)


def detect(values, *, max_period: int = MAX_PERIOD,
           size: int = 5) -> Recovery | None:
    """Deduce the period and the additive key, or return None.

    Prefers the SHORTEST period that works. A multiple of the true period
    always works too -- period 14 fires on a period-7 key -- and reporting the
    multiple would name a longer key than the message actually uses.
    """
    values = list(values)
    if len(values) < MINIMUM_TOKENS:
        return None
    if not all(any(is_coordinate(value - key, size)
                   for key in coordinates(size)) for value in values):
        return None

    for period in range(1, max_period + 1):
        if len(values) < period * 4:
            break
        classes = [values[start::period] for start in range(period)]
        options = [feasible_keys(group, size) for group in classes]
        if any(not group for group in options):
            continue
        total = 1
        for group in options:
            total *= len(group)
        if total > MAX_COMBINATIONS:
            continue
        best = None
        for combination in itertools.product(*options):
            cells = tuple(values[i] - combination[i % period]
                          for i in range(len(values)))
            recovery = Recovery(period=period, key_coordinates=combination,
                                cells=cells)
            score = recovery.index_of_coincidence()
            if best is None or score > best[0]:
                best = (score, recovery)
        if best is not None:
            return best[1]
    return None


def solve(source, *, scorer=None, top: int = 5, seed: int | None = None,
          max_period: int = MAX_PERIOD, time_budget: float | None = None,
          **options):
    """Read a Nihilist ciphertext, or return nothing.

    ``time_budget`` is accepted and passed to the substitution climb
    underneath. Refusing it outright is what once got a whole stage silently
    dropped from the pipeline, because every real run sets a clock and the
    tests did not.
    """
    from . import substitution
    from .candidates import Candidate, CandidateSet
    from .scoring import annotate, default_scorer

    raw = source if isinstance(source, str) else getattr(source, "text", "")
    empty = CandidateSet()
    values = parse(raw)
    if len(values) < MINIMUM_TOKENS:
        return empty
    # A stream of single digits is a Polybius message, not a Nihilist one, and
    # that solver already exists. Refusing here keeps the two apart instead of
    # offering two readings of the same digits.
    if max(values) < 11:
        return empty

    recovery = detect(values, max_period=max_period)
    if recovery is None:
        return empty

    order = sorted(set(recovery.cells))
    alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"[:len(order)]
    if len(order) > len(alphabet):
        return empty
    label = {cell: alphabet[i] for i, cell in enumerate(order)}
    relabelled = "".join(label[cell] for cell in recovery.cells)

    engine = scorer if scorer is not None else default_scorer()
    inner = substitution.solve(
        relabelled, scorer=engine, top=top, seed=seed,
        **({"time_budget": time_budget} if time_budget else {}))

    found = CandidateSet()
    for candidate in inner.ranked()[:top]:
        square = _square_from(recovery.cells, candidate.plaintext)
        key = "".join(square.get(coordinate, ".")
                      for coordinate in recovery.key_coordinates)
        diagnostics = dict(candidate.diagnostics)
        diagnostics.update({
            "nihilist_period": recovery.period,
            "key_coordinates": list(recovery.key_coordinates),
            "cell_index_of_coincidence": round(
                recovery.index_of_coincidence(), 5),
            "cells_seen": len(order),
        })
        annotate(diagnostics, candidate.plaintext, engine)
        found.add(Candidate(
            method=METHOD,
            key=(f"period {recovery.period}, key={key}, "
                 f"square={_render_square(square)}"),
            score=candidate.score,
            plaintext=candidate.plaintext,
            diagnostics=diagnostics,
        ))
    return found


def _square_from(cells, plaintext: str) -> dict:
    """The Polybius square implied by a cell stream and its reading."""
    square: dict = {}
    for cell, letter in zip(cells, plaintext):
        square.setdefault(cell, letter)
    return square


def _render_square(square: dict) -> str:
    """The 25 cells row by row, with an unseen cell shown as a dot.

    Never closed up. A square printed short reads at the wrong coordinates
    from the hole onward, which is a key that cannot be checked by hand
    wearing the costume of one that can.
    """
    return "".join(square.get(coordinate, ".")
                   for coordinate in coordinates(5))
