"""A Polybius square whose two coordinates are written as two BLOCKS.

The gap this closes
-------------------
An ordinary Polybius ciphertext interleaves the coordinates: row, column, row,
column. This one does not. It writes **every row coordinate first, and then
every column coordinate**, so the two halves of the message are the two rows
of the fractionation table read one after the other. Nothing in the toolkit
paired symbols that way, so 2024 challenge 8B -- 3,025 digits over five
symbols -- survived the Polybius square search, bifid at every period 2 to 30,
an ADFGX-style columnar at every width 2 to 12, and the whole pipeline at deep
effort, always at `weak`.

The setter's own hint on that page is *"Sometimes you need to look at
something from a different angle to understand it"*, which is a hint about
geometry rather than about keys.

Why it can be found without a key
---------------------------------
Score the pairing, not the letters. Take the share of the message held by its
ten commonest cell bigrams, or equivalently the index of coincidence of the
cells: a substitution renames cells and cannot move either, while a WRONG
pairing produces cells that are not letters at all and both collapse towards
uniform.

And the search is tiny, which is what makes the finding safe. In a true split
fractionation the two halves each hold one coordinate per letter, so they are
the same length: the split can only be at the middle, give or take a stray
symbol. There is no wide sweep to go fishing in.

MEASURED. On 2024 8B the split at 1,513 gives cell index of coincidence
**0.0663** and top-ten bigram share **0.1880**. The reference is 2023 8B, an
ordinary Polybius already solved and graded against its published decrypt:
0.0692 and 0.1844. They are the same numbers.

And the negative control, which is the part that matters: swept over ALL 2,600
split positions rather than just the middle, 2023 8B's best split scores 0.0436
against a median of 0.0409 -- the sweep finds nothing on a message that is not
seriated -- while 2024 8B's true split stands 0.0213 clear of the best of every
other split in the message. One position, no plateau, no second candidate.
"""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass

#: Each half must hold at least this many coordinates. Below it the index of
#: coincidence is noise: sweeping every split position on a message that is
#: NOT seriated threw up "peaks" of 0.0647 on 213 units, which is what noise
#: looks like when you let a search look at short windows.
MINIMUM_UNITS = 300

#: How far from the exact middle to look. A true split fractionation has two
#: halves of equal length by construction, so this only allows for a message
#: that gained or lost a symbol in transcription -- and hand-copied
#: competition material does that routinely. 2024 8B is 3,025 symbols with the
#: split at 1,513, one clear of the middle.
SLACK = 6

#: The cells must have the index of coincidence of English text. English over
#: a 25-cell square is about 0.069; a wrong pairing sits near 1/25 = 0.040,
#: and the observed median over every wrong split was 0.0409. The bar sits in
#: the gap.
MINIMUM_IC = 0.055

#: The cells must also beat the same construction on a SHUFFLE of the same
#: symbols by this much. The absolute bar alone would be a statement about the
#: alphabet size; this one is a statement about the ORDER, which is where the
#: cipher is.
MINIMUM_GAP = 0.012

#: Fixed so a rerun gives the same control, and a reported gap is reproducible
#: rather than a lucky shuffle.
CONTROL_SEED = 0

#: A seriated Polybius is written in the labels of a square -- five or six
#: symbols, sometimes eight. A 26-letter stream is not this cipher, and saying
#: so up front keeps the stage free on everything else.
MAX_SYMBOLS = 8

#: The cells must be a real alphabet, not a handful of values repeated.
#:
#: FOUND BY A CONTROL, not by reasoning. Handed an ordinary INTERLEAVED
#: Polybius message whose plaintext happened to repeat with a period dividing
#: the half-length, the detector paired every symbol with an identical one and
#: reported an index of coincidence of 0.204 -- the highest number in these
#: tests, from five distinct cells, on the wrong cipher entirely. A high index
#: of coincidence is also what COLLAPSE looks like, and nothing above
#: distinguishes the two. Real text over a 25-cell square uses 23 to 25 cells;
#: the reference message uses 23.
MINIMUM_CELLS = 18

#: Letters used to relabel cells before handing them to the substitution
#: solver. J is omitted because a 5x5 square omits one letter and J is the
#: usual choice; the label is arbitrary either way.
_CELL_LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXYZ"

METHOD = "Polybius with the two coordinates written as separate blocks"


@dataclass(frozen=True)
class Seriation:
    """A split that turns a symbol stream into letter-shaped cells.

    Attributes
    ----------
    split:
        Index where the second coordinate block starts.
    cells:
        The recovered cells, one per plaintext letter.
    index_of_coincidence:
        Of the cells. English over a 25-cell square is about 0.069.
    control:
        The same figure for the same construction over a shuffle of the same
        symbols, which is what the finding has to beat.
    """

    split: int
    cells: tuple[str, ...]
    index_of_coincidence: float
    control: float

    @property
    def gap(self) -> float:
        """How far the finding beats its own null control.

        The raw index of coincidence is not evidence by itself: it partly
        measures how many distinct cells there are. The margin over the same
        construction on shuffled symbols is the part that is about ORDER.
        """
        return self.index_of_coincidence - self.control


def index_of_coincidence(items) -> float:
    """Probability that two items drawn from *items* are equal."""
    counts = collections.Counter(items)
    total = sum(counts.values())
    if total < 2:
        return 0.0
    return sum(n * (n - 1) for n in counts.values()) / (total * (total - 1))


def pair_at(stream: str, split: int) -> list[str]:
    """Cells formed by pairing symbol *i* with symbol *split + i*."""
    count = min(split, len(stream) - split)
    return [stream[i] + stream[split + i] for i in range(count)]


def encipher(plaintext: str, square: str) -> str:
    """Encipher with a 5x5 square, first coordinates then second coordinates.

    Written out so the family can be exercised on more than the one real
    message: a solver measured against nothing but its own single example is
    measured against nothing.
    """
    size = 5
    firsts, seconds = [], []
    for character in plaintext.upper():
        if character not in square:
            continue
        row, column = divmod(square.index(character), size)
        firsts.append(str(row + 1))
        seconds.append(str(column + 1))
    return "".join(firsts) + "".join(seconds)


def decipher(ciphertext: str, square: str) -> str:
    """Invert :func:`encipher`. Written out rather than inferred."""
    size = 5
    stream = "".join(ciphertext.split())
    split = len(stream) // 2
    out = []
    for cell in pair_at(stream, split):
        row, column = int(cell[0]) - 1, int(cell[1]) - 1
        out.append(square[row * size + column])
    return "".join(out)


def detect(stream: str) -> Seriation | None:
    """Find the split, or return None. Cheap, and refuses far more than it finds.

    Only splits within :data:`SLACK` of the middle are tried, because a true
    split fractionation has two halves of equal length by construction. That
    is not a shortcut, it is the reason the finding can be trusted: there are
    thirteen candidates, not two thousand, so there is no room for the best of
    N coincidences.
    """
    stream = "".join(str(stream).split())
    alphabet = set(stream)
    if not 2 <= len(alphabet) <= MAX_SYMBOLS:
        return None
    if len(stream) < 2 * MINIMUM_UNITS:
        return None

    shuffled = list(stream)
    random.Random(CONTROL_SEED).shuffle(shuffled)
    shuffled = "".join(shuffled)

    middle = len(stream) // 2
    best: Seriation | None = None
    for split in range(middle - SLACK, middle + SLACK + 1):
        if min(split, len(stream) - split) < MINIMUM_UNITS:
            continue
        cells = pair_at(stream, split)
        found = Seriation(
            split=split,
            cells=tuple(cells),
            index_of_coincidence=index_of_coincidence(cells),
            control=index_of_coincidence(pair_at(shuffled, split)),
        )
        if best is None or found.index_of_coincidence > best.index_of_coincidence:
            best = found

    if best is None:
        return None
    if best.index_of_coincidence < MINIMUM_IC or best.gap < MINIMUM_GAP:
        return None
    if len(set(best.cells)) < MINIMUM_CELLS:
        # A high index of coincidence is also what a COLLAPSE looks like. See
        # MINIMUM_CELLS: this exact case was produced by a control and scored
        # higher than any real finding here.
        return None
    if len(set(best.cells)) > len(_CELL_LETTERS):
        # More cells than there are letters to name them with: a 6x6 square
        # carrying digits can do that, a letters-only plaintext cannot.
        return None
    return best


def solve(source, *, scorer=None, top: int = 5, seed: int | None = None,
          time_budget: float | None = None, **options):
    """Read a split-coordinate Polybius message, or return nothing.

    ``time_budget`` is accepted and handed to the substitution climb
    underneath. Refusing it outright is what once got a whole stage silently
    dropped from the pipeline, because every real run sets a clock and none of
    the tests did.
    """
    from . import substitution
    from .candidates import Candidate, CandidateSet
    from .scoring import annotate, default_scorer

    raw = source if isinstance(source, str) else getattr(source, "symbols", "")
    found = CandidateSet()
    seriation = detect(raw)
    if seriation is None:
        return found

    cells = sorted(set(seriation.cells))
    label = {cell: _CELL_LETTERS[i] for i, cell in enumerate(cells)}
    relabelled = "".join(label[cell] for cell in seriation.cells)

    engine = scorer if scorer is not None else default_scorer()
    inner = substitution.solve(
        relabelled, scorer=engine, top=top, seed=seed,
        **({"time_budget": time_budget} if time_budget else {}))

    for candidate in inner.ranked()[:top]:
        square = _square_from(seriation.cells, candidate.plaintext)
        diagnostics = dict(candidate.diagnostics)
        diagnostics.update({
            "split": seriation.split,
            "cell_index_of_coincidence": round(
                seriation.index_of_coincidence, 5),
            "cell_control": round(seriation.control, 5),
            "cell_gap": round(seriation.gap, 5),
            "cells_seen": len(cells),
        })
        annotate(diagnostics, candidate.plaintext, engine)
        found.add(Candidate(
            method=METHOD,
            key=(f"second coordinates start at symbol {seriation.split}; "
                 f"square={_render(square, seriation.cells)}"),
            score=candidate.score,
            plaintext=candidate.plaintext,
            diagnostics=diagnostics,
        ))
    return found


def _square_from(cells, plaintext: str) -> dict:
    """The square implied by a cell stream and its reading."""
    square: dict = {}
    for cell, letter in zip(cells, plaintext):
        square.setdefault(cell, letter)
    return square


def _render(square: dict, cells) -> str:
    """The whole grid in cell order, a cell the message never used as a dot.

    Never closed up. A square printed short reads at the wrong coordinates
    from the hole onward, which is a key that cannot be checked by hand
    wearing the costume of one that can. The grid is built from the symbol
    alphabet, not from the cells that happened to occur, so an unused cell
    still takes up its place.
    """
    rows = sorted({cell[0] for cell in cells})
    columns = sorted({cell[1] for cell in cells})
    return "".join(square.get(row + column, ".")
                   for row in rows for column in columns)
