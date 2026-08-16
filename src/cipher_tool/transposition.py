"""Route and grid transposition, and the transposition family dispatcher.

The cipher
----------
Write the plaintext into a rectangle along one path through the cells, then
read the rectangle out along a different path. Nothing else happens: no letter
is ever replaced. With a 3x4 grid, ATTACKATDAWN written in row by row is::

    A T T A
    C K A T
    D A W N

Read down the columns and the ciphertext is ACD TKA TAW ATN. Read along a
clockwise spiral from the top left and it is ATTA TN WAD C KA. Read the rows
alternately left to right and right to left (a *boustrophedon*, from the Greek
for "as the ox ploughs") and it is ATTA TAKC DAWN.

The route is the key, together with the shape of the rectangle and the path
used to write the text in. This module calls those the *read route*, the
*shape* and the *fill route*.

Routes as permutations
----------------------
Every route is nothing more than an ordering of the grid's cells, so a route
is stored as the function that produces that ordering. Writing the text in
along route F means "plaintext letter i goes in cell F[i]"; reading out along
route R means "ciphertext letter j is whatever sits in cell R[j]". Composing
the two, the cipher is the permutation

    ciphertext[j] = plaintext[ F^-1(R[j]) ]

which has two consequences worth stating plainly.

* Decrypting a message that was filled along F and read along R is the same
  operation as *encrypting* along the swapped pair (fill R, read F), because
  the inverse of ``R . F^-1`` is ``F . R^-1``. That is why :func:`solve`
  offers ``both_directions``: trying the swapped pair costs nothing extra and
  covers a sender who wrote along the fancy route and read along a plain one.
* Two different (fill, read) pairs can describe the same permutation, so the
  search will sometimes report one plaintext under several keys. The candidate
  set merges those and counts the agreements rather than hiding them.

Ragged rectangles
-----------------
Competition messages are rarely a perfect rectangle, so a grid usually has a
few empty cells. Where those empty cells sit is a *convention*, not a fact:
if the text was written in row by row the blanks are obviously the right-hand
end of the last row, but if it was written in along a spiral, a sender might
reasonably leave the blanks at the end of the spiral, at the bottom right of
the rectangle, or pad the message out with filler letters instead. Those give
different ciphertexts and nothing in the ciphertext says which was meant.

This module therefore refuses to guess. A ragged grid is only accepted with a
row-by-row fill, which is the one case where every convention agrees, and
:func:`solve` skips (and reports skipping) every other fill route on a ragged
shape. The read route may still be anything: once the blank cells are known,
reading past them is unambiguous.

Recognising the family
----------------------
A transposition changes *nothing* about the letter frequencies -- it moves
letters, it does not replace them -- so the chi-squared distance from English
measured by :func:`statistics.chi_squared_english` is exactly as small for the
ciphertext as for the plaintext, and the index of coincidence stays at the
English value of about 0.067. Unreadable text whose letter statistics are
flawless English is the signature of this whole family, and it is what
``cipher_tool analyse`` keys on before suggesting this module. Every candidate
below reports the ciphertext's chi-squared so that the evidence travels with
the answer.

The attack
----------
There is no clever statistic to exploit here, and no need for one: the key
space is small enough to enumerate. The shapes are constrained by arithmetic
(:func:`grid_shapes` takes them from ``statistics.divisors`` for an exact
rectangle, or from the ceiling division for a ragged one), the routes are a
fixed list of about fifteen, and the fills are the four plain ways of writing
text into a grid. A few thousand decryptions is under a second, so
:func:`solve` tries every (shape, fill, read) combination, scores each with the
English model and ranks them. The diagnostics name every grid dimension and
every route that was tested, because an exhaustive search is only exhaustive
over what it actually looked at.

:func:`solve_all` is the family dispatcher used by ``cipher_tool
transposition``: it runs the rail fence solver, the columnar solver and the
route solver above, merges their candidates into one ranked set, and shares a
time budget between them.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from . import columnar, rail_fence
from .candidates import Candidate, CandidateSet
from .normalize import NormalizedText, group_text, letters_only, normalize
from .scoring import EnglishScorer, annotate, default_scorer
from .statistics import chi_squared_english, divisors

#: A grid cell as ``(row, column)``, both zero-based.
Cell = tuple[int, int]

#: A grid is a list of rows of single characters; ``""`` marks an empty cell.
Grid = list[list[str]]

METHOD = "Route/grid transposition"

#: Shortest side :func:`grid_shapes` will propose. A grid one cell wide is not
#: an encryption: every route through it reads the text in order or backwards.
DEFAULT_MIN_SIDE = 2

#: Longest side :func:`grid_shapes` will propose by default. This caps the
#: text length the default search can cover at ``max_side ** 2`` letters, so
#: :func:`solve` widens it for longer texts and says so in the diagnostics.
DEFAULT_MAX_SIDE = 40

#: How many of the best-scoring combinations are turned into candidates.
DEFAULT_REFINE = 40


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Route:
    """One path through the cells of a rectangular grid.

    Attributes
    ----------
    name:
        The identifier used in keys and on the command line, e.g. ``"rows"``.
    description:
        One line of English for ``cipher_tool transposition --routes``.
    order:
        ``order(rows, cols)`` returns every cell of the grid exactly once, in
        the order this route visits them. This single function defines the
        route; :meth:`read` and :meth:`write` are bookkeeping on top of it.
    ragged_safe:
        True when using this route to *fill* a grid that has empty cells is
        unambiguous, which is only the case for a plain row-by-row fill. See
        the module docstring: for every other route, where the blanks sit is a
        convention rather than a fact, so this module declines to guess.
    """

    name: str
    description: str
    order: Callable[[int, int], list[Cell]]
    ragged_safe: bool = False

    def cells(self, rows: int, cols: int) -> list[Cell]:
        """Every cell of a ``rows`` x ``cols`` grid, in this route's order."""
        _check_side(rows, "rows")
        _check_side(cols, "cols")
        return self.order(rows, cols)

    def read(self, grid: Grid) -> str:
        """Read *grid* out along this route, skipping empty cells.

        The inverse of :meth:`write` *for the same route*: reading a grid
        along the route that filled it returns the text unchanged. The cipher
        uses two different routes, which is the whole point.
        """
        rows, cols = _grid_shape(grid)
        return "".join(
            grid[row][col]
            for row, col in self.cells(rows, cols)
            if grid[row][col]
        )

    def write(self, text: str, rows: int, cols: int) -> Grid:
        """Write *text* into a ``rows`` x ``cols`` grid along this route.

        Cells the text does not reach are left empty (``""``). Raises
        ``ValueError`` if the text does not fit, rather than truncating it.
        """
        if len(text) > rows * cols:
            raise ValueError(
                f"{len(text)} letters do not fit in a {rows}x{cols} grid "
                f"({rows * cols} cells). Choose a larger grid."
            )
        grid: Grid = [["" for _ in range(cols)] for _ in range(rows)]
        for index, (row, col) in enumerate(self.cells(rows, cols)):
            if index >= len(text):
                break
            grid[row][col] = text[index]
        return grid


# -- the individual routes --------------------------------------------------
#
# Each function below returns the cells of a rows x cols grid exactly once.
# They are deliberately written as plain enumerations rather than clever
# arithmetic: a route that visits a cell twice, or misses one, would silently
# corrupt a decryption, and an enumeration you can read line by line is the
# cheapest possible defence against that. tests/test_transposition.py checks
# every route against that property for a spread of shapes.


def _rows_order(rows: int, cols: int) -> list[Cell]:
    """Row by row, each row left to right (the ordinary reading order)."""
    return [(row, col) for row in range(rows) for col in range(cols)]


def _columns_order(rows: int, cols: int) -> list[Cell]:
    """Column by column, each column top to bottom."""
    return [(row, col) for col in range(cols) for row in range(rows)]


def _reverse_order(rows: int, cols: int) -> list[Cell]:
    """Ordinary reading order, backwards: the whole text reversed."""
    return _rows_order(rows, cols)[::-1]


def _boustrophedon_rows_order(rows: int, cols: int) -> list[Cell]:
    """Rows alternately left to right and right to left, starting leftwards.

    Named after the Greek for "as the ox turns while ploughing". It is a
    genuinely different permutation from plain rows, because every odd row is
    reversed within itself while the rows stay in order.
    """
    cells: list[Cell] = []
    for row in range(rows):
        span = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
        cells.extend((row, col) for col in span)
    return cells


def _boustrophedon_columns_order(rows: int, cols: int) -> list[Cell]:
    """Columns alternately downwards and upwards, starting downwards."""
    cells: list[Cell] = []
    for col in range(cols):
        span = range(rows) if col % 2 == 0 else range(rows - 1, -1, -1)
        cells.extend((row, col) for row in span)
    return cells


def _diagonals_order(rows: int, cols: int) -> list[Cell]:
    """Anti-diagonals, each read downwards, from the top-left corner outwards.

    The cells of one anti-diagonal all share the same value of ``row + col``,
    which runs from 0 at the top-left corner to ``rows + cols - 2`` at the
    bottom-right. Within a diagonal the rows that exist run from
    ``max(0, total - cols + 1)`` to ``min(total, rows - 1)``: the first bound
    is where the diagonal enters the grid from the right-hand edge, the second
    is where it leaves through the bottom.
    """
    cells: list[Cell] = []
    for total in range(rows + cols - 1):
        first = max(0, total - cols + 1)
        last = min(total, rows - 1)
        cells.extend((row, total - row) for row in range(first, last + 1))
    return cells


def _diagonals_alternating_order(rows: int, cols: int) -> list[Cell]:
    """The same anti-diagonals, read alternately down and up (a zigzag scan).

    Compared with :func:`_diagonals_order` this keeps the walk continuous:
    each diagonal starts where the previous one finished, so neighbouring
    plaintext letters more often stay neighbours. That makes it a slightly
    stronger route and a noticeably more common one.
    """
    cells: list[Cell] = []
    for total in range(rows + cols - 1):
        first = max(0, total - cols + 1)
        last = min(total, rows - 1)
        span = (
            range(first, last + 1)
            if total % 2 == 0
            else range(last, first - 1, -1)
        )
        cells.extend((row, total - row) for row in span)
    return cells


# Movement vectors, and the turn each spiral makes when it runs out of room.
_RIGHT: Cell = (0, 1)
_DOWN: Cell = (1, 0)
_LEFT: Cell = (0, -1)
_UP: Cell = (-1, 0)

_CLOCKWISE_TURN = {_RIGHT: _DOWN, _DOWN: _LEFT, _LEFT: _UP, _UP: _RIGHT}
_ANTICLOCKWISE_TURN = {_RIGHT: _UP, _UP: _LEFT, _LEFT: _DOWN, _DOWN: _RIGHT}

#: Starting direction for a spiral, by (corner, clockwise). Leaving a corner
#: in the clockwise sense means travelling along the edge that has the rest of
#: the grid on its right: rightwards from the top left, downwards from the top
#: right, and so on. The anticlockwise entries are the other edge at that
#: corner.
_SPIRAL_STARTS: dict[tuple[str, bool], Cell] = {
    ("top_left", True): _RIGHT,
    ("top_right", True): _DOWN,
    ("bottom_right", True): _LEFT,
    ("bottom_left", True): _UP,
    ("top_left", False): _DOWN,
    ("top_right", False): _LEFT,
    ("bottom_right", False): _UP,
    ("bottom_left", False): _RIGHT,
}


def _spiral_order(
    rows: int, cols: int, corner: str, clockwise: bool
) -> list[Cell]:
    """Spiral inwards from *corner*, turning in the given sense.

    The walk is the obvious one and needs no arithmetic: keep going in the
    current direction while the next cell exists and has not been used; when
    it does not, turn. A rectangle can never block the walk in more than one
    direction at a time until it is full, so a single turn always suffices --
    but the loop tries all four directions before giving up, because a
    silently truncated route would produce a confidently wrong plaintext, and
    the cost of the extra check is nothing.
    """
    start_row = 0 if corner.startswith("top") else rows - 1
    start_col = 0 if corner.endswith("left") else cols - 1
    turn = _CLOCKWISE_TURN if clockwise else _ANTICLOCKWISE_TURN
    direction = _SPIRAL_STARTS[(corner, clockwise)]

    total = rows * cols
    row, col = start_row, start_col
    seen: set[Cell] = set()
    cells: list[Cell] = []
    while len(cells) < total:
        cells.append((row, col))
        seen.add((row, col))
        if len(cells) == total:
            break
        for _ in range(4):
            next_row = row + direction[0]
            next_col = col + direction[1]
            if (
                0 <= next_row < rows
                and 0 <= next_col < cols
                and (next_row, next_col) not in seen
            ):
                row, col = next_row, next_col
                break
            direction = turn[direction]
        else:  # pragma: no cover - unreachable for a rectangle, kept as a guard
            raise ValueError(
                f"the spiral from {corner} became trapped in a {rows}x{cols} "
                "grid; this is a bug in the route, not bad input"
            )
    return cells


def _make_spiral(corner: str, clockwise: bool) -> Callable[[int, int], list[Cell]]:
    """Build the cell-order function for one spiral variant."""

    def order(rows: int, cols: int) -> list[Cell]:
        return _spiral_order(rows, cols, corner, clockwise)

    return order


def _build_routes() -> dict[str, Route]:
    """Assemble the route table. Called once, at import."""
    table: list[Route] = [
        Route(
            "rows",
            "row by row, each row left to right",
            _rows_order,
            ragged_safe=True,
        ),
        Route("columns", "column by column, each column downwards", _columns_order),
        Route("reverse", "the whole text backwards", _reverse_order),
        Route(
            "boustrophedon_rows",
            "rows alternately left to right and right to left",
            _boustrophedon_rows_order,
        ),
        Route(
            "boustrophedon_columns",
            "columns alternately downwards and upwards",
            _boustrophedon_columns_order,
        ),
        Route(
            "diagonals",
            "anti-diagonals, each read downwards",
            _diagonals_order,
        ),
        Route(
            "diagonals_alternating",
            "anti-diagonals read alternately down and up (zigzag scan)",
            _diagonals_alternating_order,
        ),
    ]
    for corner in ("top_left", "top_right", "bottom_right", "bottom_left"):
        pretty = corner.replace("_", " ")
        table.append(
            Route(
                f"spiral_cw_{corner}",
                f"clockwise spiral inwards from the {pretty} corner",
                _make_spiral(corner, True),
            )
        )
        table.append(
            Route(
                f"spiral_acw_{corner}",
                f"anticlockwise spiral inwards from the {pretty} corner",
                _make_spiral(corner, False),
            )
        )
    return {route.name: route for route in table}


#: Every route the toolkit knows, by name. The CLI lists these.
ROUTES: dict[str, Route] = _build_routes()

#: The route names, in the order :func:`describe_routes` prints them.
ROUTE_NAMES: tuple[str, ...] = tuple(ROUTES)

#: The routes tried as *fills* by default: the four plain ways of writing text
#: into a grid. Restricting the fills is what keeps the search small; the
#: ``both_directions`` option in :func:`solve` recovers the ciphers that wrote
#: along a fancy route and read along a plain one (see the module docstring).
FILL_ROUTES: tuple[str, ...] = (
    "rows",
    "columns",
    "boustrophedon_rows",
    "boustrophedon_columns",
)


def describe_routes() -> str:
    """A printable table of the available routes, for ``--routes``."""
    width = max(len(name) for name in ROUTE_NAMES)
    lines = [
        "Routes available for grid transposition",
        "=======================================",
        "",
        "A route is a path through the cells of the grid. The text is written",
        "in along one route (the fill) and read out along another.",
        "",
    ]
    for name in ROUTE_NAMES:
        route = ROUTES[name]
        notes = []
        if name in FILL_ROUTES:
            notes.append("used as a fill")
        if route.ragged_safe:
            notes.append("safe on a ragged grid")
        suffix = f"  [{'; '.join(notes)}]" if notes else ""
        lines.append(f"  {name.ljust(width)}  {route.description}{suffix}")
    lines.append("")
    lines.append(
        "Only a row-by-row fill is offered on a grid with empty cells: for "
        "every other"
    )
    lines.append(
        "route, where the blanks sit is a convention rather than a fact, so "
        "guessing"
    )
    lines.append("one would be inventing evidence.")
    return "\n".join(lines)


def _resolve_route(value: str | Route, what: str) -> Route:
    """Turn a route name into a :class:`Route`, or complain usefully."""
    if isinstance(value, Route):
        return value
    if not isinstance(value, str):
        raise ValueError(
            f"{what} must be a route name or a Route, got "
            f"{type(value).__name__} ({value!r})"
        )
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    route = ROUTES.get(key)
    if route is None:
        raise ValueError(
            f"unknown {what} {value!r}. Known routes: "
            + ", ".join(ROUTE_NAMES)
        )
    return route


# ---------------------------------------------------------------------------
# Grid shapes
# ---------------------------------------------------------------------------


def grid_shapes(
    length: int,
    min_side: int = DEFAULT_MIN_SIDE,
    max_side: int = DEFAULT_MAX_SIDE,
    *,
    allow_ragged: bool = False,
) -> list[tuple[int, int]]:
    """Every ``(rows, cols)`` rectangle a text of *length* letters could use.

    With ``allow_ragged=False`` this is exactly the factorisations of
    *length*: a rectangle holds ``rows * cols`` letters, so it fits the text
    exactly precisely when ``rows`` divides ``length``. The divisors come from
    :func:`statistics.divisors`, and both sides must lie between *min_side*
    and *max_side*, which is why a prime length returns nothing at all -- an
    honest answer, and a useful one, because it says the sender must have used
    a ragged grid or a different cipher.

    With ``allow_ragged=True`` the shapes whose last row is short are added as
    well. There is one of those per column count: given ``cols``, the only
    sensible row count is ``ceil(length / cols)``, since anything larger would
    leave a whole empty row. Competition messages are rarely a perfect
    rectangle, so these matter -- but note that a ragged grid is only ever
    used here with a row-by-row fill, because the position of the blank cells
    is otherwise a convention rather than a fact (see the module docstring).

    The result is sorted by ``(rows, cols)`` and contains no duplicates.
    """
    length = _check_int(length, "length", 0)
    min_side = _check_int(min_side, "min_side", 1)
    max_side = _check_int(max_side, "max_side", 1)
    if max_side < min_side:
        raise ValueError(
            f"max_side ({max_side}) must not be smaller than min_side "
            f"({min_side})"
        )
    if length < min_side * min_side:
        return []

    shapes: set[tuple[int, int]] = set()
    for rows in divisors(length):
        cols = length // rows
        if min_side <= rows <= max_side and min_side <= cols <= max_side:
            shapes.add((rows, cols))

    if allow_ragged:
        for cols in range(min_side, max_side + 1):
            rows = -(-length // cols)  # ceiling division, no floating point
            if min_side <= rows <= max_side:
                shapes.add((rows, cols))

    return sorted(shapes)


def is_ragged(length: int, rows: int, cols: int) -> bool:
    """True when a ``rows`` x ``cols`` grid has empty cells for *length* letters."""
    return rows * cols != length


# ---------------------------------------------------------------------------
# Encryption and decryption
# ---------------------------------------------------------------------------


def encrypt(
    text: str,
    route: str | Route,
    rows: int | None = None,
    cols: int | None = None,
    *,
    fill: str | Route = "rows",
) -> str:
    """Encrypt. Operates on letters only; returns letters only, uppercase.

    The text is written into a ``rows`` x ``cols`` grid along the *fill* route
    and read back out along *route*. Give either dimension and the other is
    derived by ceiling division, so ``encrypt(text, "columns", cols=6)`` means
    "six columns, as many rows as that needs".
    """
    letters = letters_only(text)
    read_route = _resolve_route(route, "route")
    fill_route = _resolve_route(fill, "fill")
    if not letters:
        return ""
    rows, cols = _resolve_shape(len(letters), rows, cols, fill_route)

    filled = fill_route.cells(rows, cols)[: len(letters)]
    at = {cell: index for index, cell in enumerate(filled)}
    # Walk the read route and pick up whatever letter is standing in each cell,
    # stepping over the cells the text never reached.
    return "".join(
        letters[at[cell]] for cell in read_route.cells(rows, cols) if cell in at
    )


def decrypt(
    text: str,
    route: str | Route,
    rows: int | None = None,
    cols: int | None = None,
    *,
    fill: str | Route = "rows",
) -> str:
    """Exact inverse of :func:`encrypt` for the same key.

    Undoing the cipher is a two-step replay, and the order matters:

    1. The *fill* route and the letter count say which cells hold a letter.
       This must come first, because on a ragged grid the read route walks
       over empty cells and the ciphertext has nothing for them.
    2. Walk the read route over exactly those cells, dropping the ciphertext
       letters back in one at a time, then read the grid out along the fill
       route -- which is the plaintext, by definition of how it was written.

    Skipping step 1 and cutting the ciphertext up blindly happens to work on
    an exact rectangle and is wrong on every ragged one, which is the classic
    way to get a confidently wrong answer out of a route cipher.
    """
    letters = letters_only(text)
    read_route = _resolve_route(route, "route")
    fill_route = _resolve_route(fill, "fill")
    if not letters:
        return ""
    rows, cols = _resolve_shape(len(letters), rows, cols, fill_route)

    filled = fill_route.cells(rows, cols)[: len(letters)]
    at = {cell: index for index, cell in enumerate(filled)}
    taken = [cell for cell in read_route.cells(rows, cols) if cell in at]

    plain = [""] * len(letters)
    for index, cell in enumerate(taken):
        plain[at[cell]] = letters[index]
    return "".join(plain)


def _resolve_shape(
    length: int, rows: int | None, cols: int | None, fill_route: Route
) -> tuple[int, int]:
    """Work out and validate the grid dimensions for *length* letters."""
    if rows is None and cols is None:
        raise ValueError(
            "give a row count or a column count for the grid; neither was "
            "supplied, and the shape of the rectangle is part of the key"
        )
    if rows is not None:
        rows = _check_int(rows, "rows", 1)
    if cols is not None:
        cols = _check_int(cols, "cols", 1)
    if rows is None:
        rows = -(-length // int(cols))
    if cols is None:
        cols = -(-length // int(rows))

    if rows * cols < length:
        raise ValueError(
            f"a {rows}x{cols} grid holds {rows * cols} letters, which is fewer "
            f"than the {length} letters of text"
        )
    if (rows - 1) * cols >= length:
        raise ValueError(
            f"a {rows}x{cols} grid is taller than {length} letters need: the "
            f"last row would be completely empty. Use rows="
            f"{-(-length // cols)}."
        )
    if is_ragged(length, rows, cols) and not fill_route.ragged_safe:
        raise ValueError(
            f"a {rows}x{cols} grid leaves {rows * cols - length} cells empty, "
            f"and the fill route {fill_route.name!r} does not say where those "
            "blanks belong. Only a row-by-row fill is unambiguous on a ragged "
            "grid; use fill='rows', or choose a shape that fits exactly."
        )
    return rows, cols


def _grid_shape(grid: Grid) -> tuple[int, int]:
    """Validate a grid and return its dimensions."""
    if not grid or not grid[0]:
        raise ValueError("the grid is empty")
    cols = len(grid[0])
    for index, row in enumerate(grid):
        if len(row) != cols:
            raise ValueError(
                f"the grid is not rectangular: row 0 has {cols} cells but row "
                f"{index} has {len(row)}"
            )
    return len(grid), cols


# ---------------------------------------------------------------------------
# The attack
# ---------------------------------------------------------------------------


def _search_pairs(
    fills: Sequence[str], reads: Sequence[str], both_directions: bool
) -> list[tuple[str, str]]:
    """The ordered (fill route, read route) pairs the search will try.

    A pair with the same route at both ends is the identity permutation and is
    dropped: it would return the ciphertext unchanged under every shape, which
    is not a decryption and only clutters the ranking.

    When *both_directions* is set, each pair is also tried swapped. The reason
    is in the module docstring: decrypting "filled along A, read along B" is
    the same permutation as encrypting "filled along B, read along A", so the
    swap is exactly how a small set of fill routes still covers a sender who
    wrote the text in along a spiral and read it out along the rows.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for fill_name in fills:
        for read_name in reads:
            options = [(fill_name, read_name)]
            if both_directions:
                options.append((read_name, fill_name))
            for pair in options:
                if pair[0] == pair[1] or pair in seen:
                    continue
                seen.add(pair)
                pairs.append(pair)
    return pairs


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    **options: Any,
) -> CandidateSet:
    """Attack the ciphertext and return RANKED CANDIDATES, never one answer.

    Every combination of grid shape, fill route and read route is decrypted
    and scored with the English model, and the best few are returned ranked.
    The search is exhaustive over the shapes and routes it was given, which is
    not the same as exhaustive over all route ciphers -- so the diagnostics of
    every candidate name the exact grid dimensions and the exact routes that
    were tried, and say how many combinations that came to.

    Options
    -------
    rows, cols:
        Pin the grid shape instead of searching. Either alone is enough.
    routes:
        Read routes to try, as a sequence of names. Default: all of
        :data:`ROUTE_NAMES`.
    fills:
        Fill routes to try. Default: :data:`FILL_ROUTES`.
    both_directions:
        Also try every pair swapped (default True). See :func:`_search_pairs`.
    allow_ragged:
        Include shapes with a short last row (default True). Those are only
        combined with a row-by-row fill; the skipped combinations are counted
        in the diagnostics rather than passed over silently.
    min_side, max_side:
        Bounds on the grid sides. ``max_side`` defaults to 40, widened to
        ``isqrt(length) + 1`` for texts too long for any 40-sided grid.
    refine:
        How many of the best-scoring combinations become candidates.
    time_budget:
        Seconds. The search stops cleanly between shapes and records
        ``time_budget_hit`` on every candidate.
    seed:
        Accepted and ignored: the search is exhaustive and deterministic. It
        is accepted so a caller can pass one option set to every solver.
    """
    engine = scorer if scorer is not None else default_scorer()
    normalized = normalize(source) if isinstance(source, str) else source
    letters = normalized.letters
    length = len(letters)

    pinned_rows = options.pop("rows", None)
    pinned_cols = options.pop("cols", None)
    route_names = _check_names(options.pop("routes", None), "routes", ROUTE_NAMES)
    fill_names = _check_names(options.pop("fills", None), "fills", FILL_ROUTES)
    both_directions = bool(options.pop("both_directions", True))
    allow_ragged = bool(options.pop("allow_ragged", True))
    min_side = _option_int(
        options.pop("min_side", None), "min_side", DEFAULT_MIN_SIDE, 2
    )
    default_max_side = max(DEFAULT_MAX_SIDE, math.isqrt(max(length, 1)) + 1)
    max_side = _option_int(
        options.pop("max_side", None), "max_side", default_max_side, 2
    )
    refine = _option_int(options.pop("refine", None), "refine", DEFAULT_REFINE, 1)
    time_budget = options.pop("time_budget", None)
    options.pop("seed", None)  # documented no-op; the search is deterministic
    if options:
        raise ValueError(
            "unknown option(s) for transposition.solve: "
            + ", ".join(sorted(str(name) for name in options))
        )
    if time_budget is not None and time_budget <= 0:
        raise ValueError(f"time_budget must be positive, got {time_budget}")

    results = CandidateSet()
    # A 2x2 grid is the smallest rectangle any route can scramble.
    if length < 4:
        return results

    if pinned_rows is not None or pinned_cols is not None:
        shapes = [_resolve_shape(length, pinned_rows, pinned_cols, ROUTES["rows"])]
    else:
        exact = grid_shapes(length, min_side, max_side, allow_ragged=False)
        # Exact rectangles first: a sender who could choose a shape that fits
        # is much more likely to have done so, and if a time budget cuts the
        # search short the work that got done should be the likely work.
        shapes = list(exact)
        if allow_ragged:
            ragged = grid_shapes(length, min_side, max_side, allow_ragged=True)
            shapes += [shape for shape in ragged if shape not in set(exact)]

    pairs = _search_pairs(fill_names, route_names, both_directions)
    needed = sorted({name for pair in pairs for name in pair})
    deadline = None if time_budget is None else time.monotonic() + time_budget
    budget_hit = False

    shortlist: list[tuple[float, int, int, int, str, str, str]] = []
    counter = 0
    tested = 0
    skipped_ragged = 0
    shapes_tested: list[str] = []

    for rows, cols in shapes:
        if deadline is not None and time.monotonic() > deadline:
            budget_hit = True
            break
        ragged = is_ragged(length, rows, cols)

        # The cell order of a route depends only on the shape, so build each
        # one once per shape rather than once per combination.
        orders = {name: ROUTES[name].cells(rows, cols) for name in needed}
        fill_maps: dict[str, dict[Cell, int]] = {}
        for fill_name, _ in pairs:
            if fill_name in fill_maps:
                continue
            if ragged and not ROUTES[fill_name].ragged_safe:
                continue
            filled = orders[fill_name][:length]
            fill_maps[fill_name] = {
                cell: index for index, cell in enumerate(filled)
            }

        used_here = 0
        for fill_name, read_name in pairs:
            at = fill_maps.get(fill_name)
            if at is None:
                # Ragged grid, ambiguous fill: refused, not guessed at.
                skipped_ragged += 1
                continue
            plain = [""] * length
            position = 0
            for cell in orders[read_name]:
                index = at.get(cell)
                if index is None:
                    continue
                plain[index] = letters[position]
                position += 1
            plaintext = "".join(plain)
            score = engine.score(plaintext)
            tested += 1
            used_here += 1
            counter += 1
            entry = (score, counter, rows, cols, fill_name, read_name, plaintext)
            if len(shortlist) < refine:
                heapq.heappush(shortlist, entry)
            elif entry > shortlist[0]:
                heapq.heappushpop(shortlist, entry)

        shapes_tested.append(
            f"{rows}x{cols}"
            + (f" ragged ({rows * cols - length} blank)" if ragged else " exact")
            + f", {used_here} routes"
        )

    chi = chi_squared_english(letters) if letters else float("inf")
    for score, _, rows, cols, fill_name, read_name, plaintext in shortlist:
        diagnostics: dict[str, Any] = {
            "grid": f"{rows} rows x {cols} cols",
            "fill_route": fill_name,
            "read_route": read_name,
            "ragged_cells": rows * cols - length,
            "ciphertext_chi_squared": chi,
        }
        annotate(diagnostics, plaintext, engine)
        results.add(
            Candidate(
                method=METHOD,
                key=f"grid={rows}x{cols} fill={fill_name} read={read_name}",
                score=score,
                plaintext=plaintext,
                diagnostics=diagnostics,
                # NOT normalized.relayout(): a transposition moves letters, so
                # plaintext letter i did not come from ciphertext position i,
                # and pouring it back into the original spacing would invent a
                # layout that means nothing.
                display=group_text(plaintext),
            )
        )

    grids_note = "; ".join(shapes_tested) if shapes_tested else "none"
    divisor_note = ",".join(
        str(value) for value in divisors(length) if 2 <= value <= 40
    )
    for candidate in results.ranked():
        candidate.diagnostics["grids_tested"] = grids_note
        candidate.diagnostics["routes_tested"] = ",".join(route_names)
        candidate.diagnostics["fills_tested"] = ",".join(fill_names)
        candidate.diagnostics["combinations_tested"] = tested
        candidate.diagnostics["length"] = length
        candidate.diagnostics["length_divisors_2_to_40"] = divisor_note or "none"
        if skipped_ragged:
            candidate.diagnostics["skipped_ambiguous_ragged_fills"] = skipped_ragged
        if budget_hit:
            candidate.diagnostics["time_budget_hit"] = True

    if top is not None and top > 0:
        return CandidateSet(results.top(top))
    return results


# ---------------------------------------------------------------------------
# The family dispatcher
# ---------------------------------------------------------------------------

#: Relative cost of each family, used to share out a time budget. The columnar
#: search is by far the most expensive (it searches permutations), the rail
#: fence by far the cheapest (a few hundred decryptions).
_FAMILY_WEIGHTS = {"rail_fence": 0.15, "columnar": 0.55, "routes": 0.30}


def solve_all(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    **options: Any,
) -> CandidateSet:
    """Run every transposition attack and return one merged ranked set.

    This is what ``cipher_tool transposition`` calls. It runs, in increasing
    order of cost:

    * :func:`rail_fence.solve` -- exhaustive over rail counts and offsets;
    * :func:`columnar.solve` -- column-pair statistics plus a permutation
      search;
    * :func:`solve` above -- every grid shape and route.

    Candidates keep the method that produced them, so the merged ranking says
    which family each answer came from. Nothing is dropped on the grounds that
    another family scored better: the three attacks disagree often enough on
    short texts that hiding the runners-up would be hiding the uncertainty.

    Options
    -------
    time_budget:
        Seconds for the whole dispatcher, shared out by
        :data:`_FAMILY_WEIGHTS`. A family that overruns its share eats into
        the next one's; a family that finds the clock already spent is skipped
        and says so.
    seed:
        Forwarded to the searches that use randomness (the columnar greedy
        restarts), so a run is reproducible.
    max_key_length:
        Longest columnar key to try.
    max_rails:
        Largest rail fence to try.
    Anything else is forwarded to :func:`solve`: ``routes``, ``fills``,
    ``rows``, ``cols``, ``allow_ragged``, ``both_directions``, ``min_side``,
    ``max_side``, ``refine``.
    """
    engine = scorer if scorer is not None else default_scorer()
    normalized = normalize(source) if isinstance(source, str) else source

    time_budget = options.pop("time_budget", None)
    if time_budget is not None and time_budget <= 0:
        raise ValueError(f"time_budget must be positive, got {time_budget}")
    seed = options.pop("seed", None)
    max_key_length = options.pop("max_key_length", None)
    max_rails = options.pop("max_rails", None)
    route_options = dict(options)  # everything else belongs to solve()

    # Each family gets at least a moment even if the budget is tiny; a solver
    # that is handed zero seconds cannot report anything at all, and "we ran
    # out of time" is more useful when it comes with what was managed.
    per_family = {
        name: (None if time_budget is None else max(0.05, time_budget * weight))
        for name, weight in _FAMILY_WEIGHTS.items()
    }
    deadline = None if time_budget is None else time.monotonic() + time_budget

    per_family_top = max(top, 3)
    merged = CandidateSet()
    ran: list[str] = []
    skipped: list[str] = []
    budget_hit = False

    def remaining(name: str) -> float | None:
        """Seconds this family may use, or ``None`` for no limit."""
        if deadline is None:
            return None
        left = deadline - time.monotonic()
        return min(per_family[name], max(0.0, left))

    rail_options: dict[str, Any] = {"top": per_family_top, "seed": seed}
    if max_rails is not None:
        rail_options["max_rails"] = max_rails
    columnar_options: dict[str, Any] = {"top": per_family_top, "seed": seed}
    if max_key_length is not None:
        columnar_options["max_key_length"] = max_key_length

    plan: list[tuple[str, str, Any, dict[str, Any]]] = [
        ("rail_fence", "rail fence", rail_fence.solve, rail_options),
        ("columnar", "columnar", columnar.solve, columnar_options),
        (
            "routes",
            "route/grid",
            solve,
            {"top": per_family_top, "seed": seed, **route_options},
        ),
    ]

    for name, label, runner, runner_options in plan:
        share = remaining(name)
        if share is not None and share <= 0.0:
            skipped.append(label)
            budget_hit = True
            continue
        call_options = dict(runner_options)
        if share is not None:
            call_options["time_budget"] = share
        found = runner(normalized, scorer=engine, **call_options)
        ran.append(label)
        merged.extend(found.ranked())
        if any(c.diagnostics.get("time_budget_hit") for c in found.ranked()):
            budget_hit = True

    for candidate in merged.ranked():
        candidate.diagnostics["families_run"] = ", ".join(ran) or "none"
        if skipped:
            candidate.diagnostics["families_skipped_no_time"] = ", ".join(skipped)
        if budget_hit:
            candidate.diagnostics["time_budget_hit"] = True

    if top is not None and top > 0:
        return CandidateSet(merged.top(top))
    return merged


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_int(value: Any, name: str, minimum: int) -> int:
    """Validate a whole-number argument of at least *minimum*.

    Floats are rejected rather than rounded: a grid of 3.5 rows is a mistake,
    not a request, and silently truncating it would produce a confidently
    wrong plaintext.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be an integer, got {type(value).__name__} ({value!r})"
        )
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _option_int(value: Any, name: str, default: int, minimum: int) -> int:
    """Read an integer solver option, treating ``None`` as "not supplied"."""
    if value is None:
        return default
    return _check_int(value, name, minimum)


def _check_side(value: Any, name: str) -> int:
    """A grid side must be a whole number of at least one."""
    return _check_int(value, name, 1)


def _check_names(
    value: Any, what: str, default: Sequence[str]
) -> tuple[str, ...]:
    """Validate a sequence of route names, or return the default."""
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        value = [part for part in value.replace(",", " ").split() if part]
    try:
        wanted = list(value)
    except TypeError as error:
        raise ValueError(
            f"{what} must be a sequence of route names, got {value!r}"
        ) from error
    if not wanted:
        raise ValueError(f"{what} must name at least one route")
    return tuple(_resolve_route(name, f"{what} entry").name for name in wanted)
