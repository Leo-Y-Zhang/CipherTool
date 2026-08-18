"""Recognise a paired-alphabet (fractionating) symbol stream.

What this module is for
-----------------------
Some ciphertexts are not written in letters at all. They are written in
*pairs* of symbols drawn from two disjoint alphabets -- a rank and a suit, a
row and a column, ADFGVX-style coordinates -- so that two symbols stand for
one cell. Reading such a stream a symbol at a time gives nonsense, and reading
only the letters in it gives worse than nonsense: a shorter, mutilated message
that still scores well enough for a substitution solver to answer confidently.

What this module is NOT
-----------------------
It is a **recogniser, not a solver**. It produces no plaintext, it is not an
``auto`` stage, and it never edits the stream. Everything it returns is
descriptive, and the description is written to be safe to print at a person
who has just been refused an answer.

The honesty rules it is built to keep
-------------------------------------
1. **One class of size one is a separator, not an alphabet.** ``A1B1C1D1``
   alternates perfectly and carries no structure at all. Selling that as a
   paired alphabet would be a confident wrong description, which is the exact
   failure this module exists to prevent.
2. **A single transcription slip is reported, never repaired.** Hand-copied
   competition material routinely loses one symbol; refusing on that would
   make the recogniser useless, and fixing it would mean inventing a symbol.
   So the break index is reported and the stream is handed back untouched.
3. **The claim is falsifiable.** When it does detect, it shuffles the same
   symbol multiset a fixed number of times and reports what fraction of the
   shuffles alternate too. Expected zero. That number is what separates "these
   symbols alternate" from "these symbols happen to be of two kinds".

Cost is linear in the stream at each level, plus ``control_trials`` linear
passes for the control. Nothing here is quadratic: the single-slip answer
comes out of one forward and one backward pass, not a search over cut points.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

#: Below this many symbols, two disjoint parity classes are chance rather than
#: evidence. Forty symbols is twenty cells, which is already thin.
MINIMUM_SYMBOLS = 40

#: The rank and suit letters a transcriber of a playing-card cipher uses. Both
#: T and X turn up for the ten, so both are allowed; a real deck still shows
#: only thirteen distinct ranks, which is what the naming rule checks.
RANKS = frozenset("23456789TXJQKA")
SUITS = frozenset("CDHS")

#: How many distinct ranks a stream must show before the deck is named. Below
#: this it is a paired alphabet that happens to use rank letters, and saying
#: "playing-card deck" would be a guess dressed as an observation.
MINIMUM_RANKS_TO_NAME_A_DECK = 10

#: Shuffles used for the control statistic. Two hundred passes over a
#: thousand-symbol stream is a few milliseconds and buys a falsifiable claim.
DEFAULT_CONTROL_TRIALS = 200


@dataclass(frozen=True)
class Alternation:
    """One level of strict alternation over a token stream.

    Attributes
    ----------
    first_class, second_class:
        The distinct tokens seen at even and at odd positions, sorted.
    tokens:
        Length of the stream at this level.
    units:
        ``tokens // 2`` -- the implied number of cells.
    distinct_units:
        How many distinct cells actually occur, which is usually far fewer
        than ``cells_available`` and is the more useful number. Counted only
        over the symbols BEFORE ``repairable_at``, because past a slip the
        pairing is off by one and manufactures cells that do not exist: over
        the whole of a real 52-card stream with one symbol missing that count
        came to 95, out of the 52 the alphabets allow, and the report said so
        in plain English to somebody with no way to check it.
    cells_available:
        ``len(first_class) * len(second_class)``.
    breaks:
        Indices where alternation could not be continued. Empty when clean.
    clean:
        True when there are no breaks at all.
    repairable_at:
        The single index at which one inserted or deleted token would restore
        alternation, or ``None``. Reported, never acted on.
    """

    first_class: tuple[str, ...]
    second_class: tuple[str, ...]
    tokens: int
    units: int
    distinct_units: int
    cells_available: int
    breaks: tuple[int, ...]
    clean: bool
    repairable_at: int | None


@dataclass(frozen=True)
class PairedReport:
    """What the recogniser concluded, and why.

    ``description`` is always safe to print, whether or not ``detected`` is
    true; when nothing was found it is empty and ``reason`` carries the
    explanation instead.
    """

    detected: bool
    reason: str
    levels: tuple[Alternation, ...]
    inventory_name: str | None
    shuffle_control: float | None
    description: str


def cells(stream: Sequence[str]) -> list[str]:
    """Pair the stream up two tokens at a time, dropping a lone tail token.

    The tail is dropped rather than padded because padding would invent a
    symbol. Callers that care report the leftover from ``tokens`` and
    ``units``.
    """
    return [
        "".join(stream[index:index + 2])
        for index in range(0, len(stream) - 1, 2)
    ]


def recognise(
    stream: str | Sequence[str],
    *,
    max_levels: int = 2,
    minimum: int = MINIMUM_SYMBOLS,
    control_trials: int = DEFAULT_CONTROL_TRIALS,
    seed: int = 0,
) -> PairedReport:
    """Decide whether *stream* is written in a paired alphabet.

    Parameters
    ----------
    max_levels:
        How many times to look for alternation. Level 0 runs over the symbols
        themselves; level 1 over the cells they make, which is what tells a
        one-cell-per-letter cipher apart from a digraphic or codebook one.
    minimum:
        Refuse below this many tokens at level 0.
    control_trials, seed:
        The falsifiability control -- see the module docstring. Set
        ``control_trials`` to zero to skip it.
    """
    tokens = list(stream)
    level, reason = _analyse(tokens, minimum)
    if level is None:
        return PairedReport(
            detected=False, reason=reason, levels=(), inventory_name=None,
            shuffle_control=None, description="",
        )

    levels = [level]
    # Only when level 0 is CLEAN. Past a slip the pairing is off by one, so
    # every cell after it is one this module invented, and level 1 is a claim
    # about the cells. MEASURED: on a two-level stream with a single symbol
    # deleted, the second level still "detects" -- over a cell stream that is
    # half fabricated.
    if max_levels > 1 and level.clean:
        deeper, _ = _analyse(cells(tokens), minimum // 2)
        if deeper is not None:
            levels.append(deeper)

    control = None
    if control_trials > 0:
        control = _shuffle_control(tokens, control_trials, seed)

    name = _name_inventory(level)
    report = PairedReport(
        detected=True, reason="", levels=tuple(levels), inventory_name=name,
        shuffle_control=control, description="",
    )
    return PairedReport(
        detected=True, reason="", levels=tuple(levels), inventory_name=name,
        shuffle_control=control, description=describe(report),
    )


def describe(report: PairedReport) -> str:
    """The printable block: what was found, in plain English.

    Written for somebody who has just been told their message could not be
    solved. It says what the structure is, what the message is made of, and
    what it is NOT -- because "1,251 letters" was the wrong reading that made
    this module necessary.
    """
    if not report.detected or not report.levels:
        return ""

    level = report.levels[0]
    lines: list[str] = []
    left, right = _rank_and_suit(level)
    leftover = " and one symbol left over" if level.tokens % 2 else ""

    if report.inventory_name == "playing-card deck" and left and right:
        lines.append(
            f"The symbols alternate strictly between two disjoint sets: "
            f"{len(left)} ranks ({', '.join(left)}) and {len(right)} suits "
            f"({', '.join(right)}). That is a {len(left) * len(right)}-card "
            f"{report.inventory_name}, and every PAIR of symbols is one card "
            f"-- {level.units} cards here{leftover}, not {level.tokens} "
            "letters."
        )
    else:
        first, second = level.first_class, level.second_class
        lines.append(
            f"The symbols alternate strictly between two disjoint sets of "
            f"{len(first)} and {len(second)}: a {len(first)} x {len(second)} "
            f"paired alphabet ({level.cells_available} cells). Every PAIR of "
            f"symbols is one cell -- {level.units} cells here{leftover}, not "
            f"{level.tokens} letters."
        )

    where = (
        ""
        if level.clean
        else f" in the {level.repairable_at} symbols before the break"
    )
    lines.append(
        f"{level.distinct_units} distinct cells occur{where}, out of "
        f"{level.cells_available} the two alphabets allow."
    )

    if not level.clean and level.repairable_at is not None:
        also_odd = (
            ", which is also why the count is odd"
            if level.tokens % 2 else ""
        )
        lines.append(
            f"Alternation breaks once, at symbol {level.repairable_at}; one "
            f"symbol looks missing or extra there{also_odd}. Nothing was "
            "changed -- repairing it would mean inventing a symbol."
        )

    if len(report.levels) > 1:
        deeper = report.levels[1]
        lines.append(
            f"The cells themselves alternate between two disjoint sets of "
            f"{len(deeper.first_class)} and {len(deeper.second_class)}, so "
            "the unit is TWO cells (four symbols) and each unit carries more "
            "than one letter -- a digraph or a codebook entry. Structure "
            "alone cannot say which."
        )

    if report.shuffle_control is not None:
        lines.append(
            f"Control: {report.shuffle_control:.1%} of random shuffles of "
            "these same symbols alternate this way, so the structure is in "
            "the order and not in the inventory."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _clean_prefix(tokens: Sequence[str]) -> int:
    """Length of the longest prefix whose two parity classes stay disjoint."""
    evens: set[str] = set()
    odds: set[str] = set()
    for index, token in enumerate(tokens):
        if index % 2 == 0:
            if token in odds:
                return index
            evens.add(token)
        else:
            if token in evens:
                return index
            odds.add(token)
    return len(tokens)


def _analyse(
    tokens: Sequence[str], minimum: int
) -> tuple[Alternation | None, str]:
    """Test one level. Returns the alternation, or None and the reason why not.

    The order of the checks is the order of the honesty rules in the module
    docstring: length first, then the separator case, then disjointness, and
    only then the single-slip tolerance.
    """
    count = len(tokens)
    if count < minimum:
        return None, (
            f"{count} symbols is too few; alternation needs at least "
            f"{minimum} before it is evidence rather than chance"
        )

    evens = {token for index, token in enumerate(tokens) if index % 2 == 0}
    odds = {token for index, token in enumerate(tokens) if index % 2}

    if len(evens) < 2 or len(odds) < 2:
        return None, (
            f"one of the two positions carries only {min(len(evens), len(odds))} "
            "distinct symbol, which is a separator between symbols rather "
            "than half of a paired alphabet"
        )

    shared = evens & odds
    if not shared:
        return _alternation(tokens, evens, odds, break_at=None), ""

    prefix = _clean_prefix(tokens)
    suffix = _clean_prefix(list(reversed(tokens)))
    if prefix + suffix < count - 1:
        return None, (
            f"the two positions share {len(shared)} symbol(s), and the fault "
            "is spread through the message rather than being one slip"
        )

    start = count - suffix
    tail = tokens[start:]
    tail_evens = {token for index, token in enumerate(tail) if index % 2 == 0}
    tail_odds = {token for index, token in enumerate(tail) if index % 2}
    head = tokens[:prefix]
    head_evens = {token for index, token in enumerate(head) if index % 2 == 0}
    head_odds = {token for index, token in enumerate(head) if index % 2}

    # A single inserted or deleted token flips the parity of everything after
    # it, so the tail's classes must line up with the head's under one of the
    # two pairings. If neither works, the two halves disagree about what the
    # alphabet IS, and that is not one slip however few symbols are shared.
    if not (tail_evens & head_odds) and not (tail_odds & head_evens):
        # Parity intact across the break: the tail is aligned with the head.
        first, second = head_evens | tail_evens, head_odds | tail_odds
    elif not (tail_evens & head_evens) and not (tail_odds & head_odds):
        # Parity flipped: one token was inserted or deleted at the break, so
        # the tail's even positions belong to the head's ODD class.
        first, second = head_evens | tail_odds, head_odds | tail_evens
    else:
        return None, (
            f"the two positions share {len(shared)} symbol(s), and the two "
            "halves of the message disagree about which alphabet is which"
        )
    return _alternation(tokens, first, second, break_at=prefix), ""


def _alternation(
    tokens: Sequence[str],
    first: set[str],
    second: set[str],
    *,
    break_at: int | None,
) -> Alternation:
    """Build the record for one detected level."""
    certain = tokens if break_at is None else tokens[:break_at]
    return Alternation(
        first_class=tuple(sorted(first)),
        second_class=tuple(sorted(second)),
        tokens=len(tokens),
        units=len(tokens) // 2,
        distinct_units=len(set(cells(certain))),
        cells_available=len(first) * len(second),
        breaks=() if break_at is None else (break_at,),
        clean=break_at is None,
        repairable_at=break_at,
    )


def _shuffle_control(
    tokens: Sequence[str], trials: int, seed: int
) -> float:
    """Fraction of random shuffles of the same multiset that also alternate.

    Expected 0.0. This is the number that makes the report falsifiable, and it
    is why the recogniser can say "the structure is in the order".
    """
    generator = random.Random(seed)
    shuffled = list(tokens)
    alternating = 0
    for _ in range(trials):
        generator.shuffle(shuffled)
        if _clean_prefix(shuffled) == len(shuffled):
            alternating += 1
    return alternating / trials


def _rank_and_suit(
    level: Alternation,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The (ranks, suits) pair when this level is a card deck, else two empty.

    Named separately from :func:`_name_inventory` because the description
    wants the two classes the right way round, and which of ``first_class``
    and ``second_class`` holds the ranks depends on whether the transcription
    starts on a rank or on a suit.
    """
    for ranks, suits in (
        (level.first_class, level.second_class),
        (level.second_class, level.first_class),
    ):
        if (
            set(suits) == SUITS
            and set(ranks) <= RANKS
            and len(ranks) >= MINIMUM_RANKS_TO_NAME_A_DECK
        ):
            return ranks, suits
    return (), ()


def _name_inventory(level: Alternation) -> str | None:
    """Name the alphabet when it is recognisable, otherwise name nothing.

    Only one inventory is named, and only on an exact match: four suits, and
    at least ten distinct ranks all drawn from the rank letters. Anything
    looser would let the report announce a deck of cards over a stream that
    merely happens to use the letter K.
    """
    ranks, _ = _rank_and_suit(level)
    return "playing-card deck" if ranks else None
