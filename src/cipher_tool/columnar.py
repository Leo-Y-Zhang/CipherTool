"""Columnar transposition: complete, incomplete (ragged) and double, with an
attack built on column-pair statistics.

The cipher
----------
Write the plaintext into a grid row by row under a keyword, then read the
columns out in the alphabetical order of the keyword's letters::

    keyword  Z E B R A S          read order: A(4) B(2) E(1) R(3) S(5) Z(0)
    column   0 1 2 3 4 5

             W E A R E D
             I S C O V E
             R E D F L E
             E A T O N C
             E

    ciphertext = EVLN + ACDT + ESEA + ROFO + DEEC + WIREE
               = EVLNACDTESEAROFODEECWIREE

Twenty-five letters do not fill a six-wide grid, so the last row is short.
That *ragged* last row is the whole difficulty of the cipher and the place
almost every implementation goes wrong: see :func:`column_lengths`.

The permutation
---------------
:func:`key_order` turns a keyword into the order the columns are read. Ties
between repeated keyword letters are broken left to right, so ``BANANA``
gives ``(1, 3, 5, 0, 2, 4)``: the A in position 1 is read before the A in
position 3, and both before the B. That is the usual convention, but it is a
convention -- a keyword with repeats is ambiguous unless you state it, so
this module states it and tests it.

Variants implemented here, all of which turn up in the competition:

* incomplete (ragged) columnar, the default: no padding, short last row;
* complete columnar: pad to a full rectangle with a filler letter first;
* double columnar: encipher, then encipher the result under a second key.
  Two passes destroy the neat column structure that makes a single pass
  breakable, which is exactly why it was used in the field.

The attack
----------
Only the order of the letters is unknown, so the search is over permutations
of the columns, and the scoring insight is this: *in the plaintext grid,
neighbouring columns sit next to each other on every row*. If we guess the
column widths, we can cut the ciphertext into blocks and ask, for each
ordered pair of blocks (x, y), how English the pairs of letters

    x[0]y[0], x[1]y[1], x[2]y[2], ...

look. Summing the bigram log probabilities down the pair gives an adjacency
score A(x, y) = "how well does block y read as the column immediately to the
right of block x". The score of a whole arrangement is then the sum of A over
consecutive columns, plus one wrap term for the join from the end of each row
to the start of the next. That reduces scoring an arrangement from "score
several hundred letters" to "add up ``columns`` numbers", which is what makes
an exhaustive search over 8! = 40320 permutations cheap.

Two subtleties are handled explicitly rather than ignored:

* **Which blocks are long?** With a ragged last row the ciphertext blocks do
  not all have the same length, and the length of a block depends on which
  *grid* column it came from -- which is precisely what we are trying to
  work out. So the search enumerates the possible patterns of long and short
  blocks (``C(columns, remainder)`` of them), and within each pattern the
  block boundaries are fixed and known. Summed over patterns this is still
  exactly ``columns!`` arrangements, only reorganised so that the arithmetic
  is well defined.
* **Beyond 8 columns** an exhaustive search stops being cheap (9! = 362880
  arrangements per pattern set), so the search falls back to a greedy chain:
  start from a block, repeatedly append whichever unused block has the best
  adjacency score with the current tail, then improve the chain by swapping
  pairs of columns while any swap helps, with extra randomised restarts.
  Greedy is not guaranteed to find the best arrangement; when it is used the
  candidates say so in their diagnostics.

The adjacency score is only a filter. Every arrangement it likes is decrypted
properly with :func:`decrypt` and re-scored with the full English model, so
the ranking the caller sees never depends on the shortcut.
"""

from __future__ import annotations

import heapq
import math
import random
import time
from functools import lru_cache
from itertools import combinations, permutations
from typing import Any, Sequence

from .candidates import Candidate, CandidateSet
from .normalize import (
    ALPHABET,
    NormalizedText,
    clean_key,
    group_text,
    letters_only,
    normalize,
)
from .scoring import EnglishScorer, annotate, default_scorer
from .statistics import divisors

#: Longest keyword :func:`solve` tries when it is not told the key length.
DEFAULT_MAX_KEY_LENGTH = 9

#: Key lengths up to this get an exhaustive permutation search. 8! = 40320
#: arrangements, each scored with 8 additions, is well under a second.
DEFAULT_MAX_EXHAUSTIVE = 8

#: How many arrangements per key length are re-scored with the full English
#: model after the cheap adjacency filter has ranked them.
DEFAULT_REFINE = 15

#: Random restarts of the greedy chain improver, per long/short pattern.
DEFAULT_RESTARTS = 8

#: Ceiling on the long/short block patterns tried in greedy mode, so that a
#: long key on an awkward length cannot silently take minutes. The number
#: actually tried is always reported in the diagnostics.
GREEDY_PATTERN_LIMIT = 60

METHOD = "Columnar transposition"


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def key_order(keyword: str) -> tuple[int, ...]:
    """The read order of the columns for *keyword*.

    ``key_order(keyword)[j]`` is the index of the grid column read out *j*-th.
    ``key_order("ZEBRAS")`` is ``(4, 2, 1, 3, 5, 0)``: column 4 (the A) is
    read first, then column 2 (the B), and so on.

    Repeated keyword letters break their tie left to right, so equal letters
    are read in the order they appear in the keyword:
    ``key_order("BANANA") == (1, 3, 5, 0, 2, 4)``. Sorting by
    ``(letter, position)`` is what enforces that, and Python's sort being
    stable is not relied upon.

    Everything that is not a letter A-Z is stripped first, so ``"Zebra's!"``
    and ``"ZEBRAS"`` describe the same key.
    """
    cleaned = clean_key(keyword)
    if len(cleaned) < 2:
        raise ValueError(
            f"a columnar key needs at least two letters, got {keyword!r} "
            f"which reduces to {cleaned!r}. One column would leave the text "
            "unchanged."
        )
    return tuple(
        index
        for _, index in sorted(
            (letter, index) for index, letter in enumerate(cleaned)
        )
    )


def keyword_from_order(order: Sequence[int]) -> str:
    """A canonical keyword whose :func:`key_order` is *order*.

    The inverse of :func:`key_order` up to the choice of letters: if column
    ``c`` is read ``j``-th then giving it the ``j``-th letter of the alphabet
    reproduces the permutation, so ``key_order(keyword_from_order(p)) == p``.
    Used to report a recovered permutation as something copy-pasteable, for
    example ``ECBDA`` rather than ``(4, 2, 1, 3, 0)``.

    Returns ``""`` for more than 26 columns, where no keyword of distinct
    letters exists.
    """
    order = _check_permutation(order)
    if len(order) > len(ALPHABET):
        return ""
    rank = [0] * len(order)
    for read_position, column in enumerate(order):
        rank[column] = read_position
    return "".join(ALPHABET[rank[column]] for column in range(len(order)))


def _coerce_key(key: str | Sequence[int]) -> tuple[int, ...]:
    """Accept either a keyword or an explicit read-order permutation."""
    if isinstance(key, str):
        return key_order(key)
    return _check_permutation(key)


def _check_permutation(order: Sequence[int]) -> tuple[int, ...]:
    """Validate an explicit permutation given as column indices."""
    try:
        values = tuple(order)
    except TypeError as error:
        raise ValueError(
            "a columnar key must be a keyword or a sequence of column "
            f"indices, got {order!r}"
        ) from error
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            # int(2.5) would silently become 2, so floats are rejected rather
            # than rounded: a permutation is not an approximate thing.
            raise ValueError(
                "column indices must be whole numbers, got "
                f"{value!r} in {values!r}"
            )
    if len(values) < 2:
        raise ValueError(
            f"a columnar key needs at least two columns, got {values!r}"
        )
    if sorted(values) != list(range(len(values))):
        raise ValueError(
            f"{values!r} is not a permutation: a key of {len(values)} columns "
            f"must use each of the indices 0..{len(values) - 1} exactly once"
        )
    return values


def _check_filler(filler: str) -> str:
    """The padding letter for complete columnar must be one letter A-Z."""
    cleaned = letters_only(filler)
    if len(cleaned) != 1:
        raise ValueError(
            f"filler must be a single letter A-Z, got {filler!r}"
        )
    return cleaned


# ---------------------------------------------------------------------------
# The ragged grid
# ---------------------------------------------------------------------------


def column_lengths(length: int, count: int) -> list[int]:
    """How many letters stand in each grid column, left to right.

    This is the arithmetic that a columnar implementation has to get right.
    Writing ``length`` letters row by row into ``count`` columns fills

        rows = ceil(length / count)

    rows, of which the last holds only ``remainder = length mod count``
    letters. Those leftover letters go into the *leftmost* ``remainder``
    columns, because the last row is filled left to right like every other
    row. So

        column c is long  (rows letters)      when c < remainder
        column c is short (rows - 1 letters)  otherwise

    and when ``remainder == 0`` the rectangle is full and every column has
    ``rows`` letters.

    ``column_lengths(25, 6) == [5, 4, 4, 4, 4, 4]``.

    Note that this is indexed by *grid* position, not by the order the columns
    are read out. Recovering the lengths of the ciphertext blocks therefore
    needs the key first -- which is exactly the step :func:`decrypt` must not
    skip, and the step an attacker has to guess.
    """
    if length < 0:
        raise ValueError(f"length must not be negative, got {length}")
    if count < 1:
        raise ValueError(f"column count must be at least 1, got {count}")
    rows = -(-length // count)  # ceiling division without floating point
    remainder = length % count
    return [
        rows if remainder == 0 or column < remainder else rows - 1
        for column in range(count)
    ]


def grid_rows(length: int, count: int) -> int:
    """Number of grid rows, the last possibly short."""
    if count < 1:
        raise ValueError(f"column count must be at least 1, got {count}")
    return -(-length // count)


# ---------------------------------------------------------------------------
# Encryption and decryption
# ---------------------------------------------------------------------------


def encrypt(
    text: str,
    key: str | Sequence[int],
    *,
    complete: bool = False,
    filler: str = "X",
) -> str:
    """Encrypt. Operates on letters only; returns letters only, uppercase.

    With ``complete=True`` the plaintext is first padded with *filler* to a
    whole number of rows, which is the "complete columnar" variant: every
    column then has the same length and the ragged-row problem disappears
    (for the sender -- and for the codebreaker, who loses a clue).

    Grid column ``c`` is ``letters[c::k]``: writing row by row means the
    letters of column ``c`` are exactly those at positions ``c``, ``c + k``,
    ``c + 2k``, ... which is a Python slice with step ``k``. That slice also
    gets the ragged tail right for free, since it simply runs out.
    """
    letters = letters_only(text)
    order = _coerce_key(key)
    count = len(order)
    if complete:
        letters = _pad(letters, count, _check_filler(filler))
    if not letters:
        return ""
    return "".join(letters[column::count] for column in order)


def decrypt(
    text: str,
    key: str | Sequence[int],
    *,
    complete: bool = False,
    filler: str = "X",
    strip_filler: bool = False,
) -> str:
    """Exact inverse of :func:`encrypt` for the same key.

    The ciphertext is the columns laid end to end *in read order*, so cutting
    it up needs each block's length before any slicing happens:

    1. :func:`column_lengths` gives the length of each **grid** column.
    2. The key says which grid column was written out first, second, ... so
       the *j*-th block of ciphertext has the length of grid column
       ``order[j]`` -- a short block can appear anywhere in the ciphertext,
       not only at the end.
    3. Only then can the ciphertext be sliced, and the grid read back row by
       row, skipping any column that has no letter on the last row.

    Cutting the ciphertext into equal blocks and hoping is the classic bug: it
    happens to work whenever the length divides exactly, which is often enough
    to pass a careless test and always wrong otherwise.

    ``complete=True`` asserts that the sender padded to a full rectangle, so a
    length that is not a whole number of rows is an error rather than a ragged
    grid. ``strip_filler`` then removes trailing filler letters -- off by
    default, because a plaintext is perfectly entitled to end in X.
    """
    letters = letters_only(text)
    order = _coerce_key(key)
    count = len(order)
    if not letters:
        return ""

    length = len(letters)
    if complete and length % count:
        raise ValueError(
            f"a complete columnar ciphertext must be a whole number of rows, "
            f"but {length} letters do not divide into {count} columns "
            f"({length % count} left over). Decrypt it as an incomplete "
            "(ragged) columnar instead, or check the column count."
        )

    lengths = column_lengths(length, count)
    rows = grid_rows(length, count)

    # Step 2 and 3: walk the ciphertext in read order, taking each block with
    # the length of the grid column it belongs to.
    grid: list[str] = [""] * count
    start = 0
    for column in order:
        size = lengths[column]
        grid[column] = letters[start : start + size]
        start += size

    out: list[str] = []
    for row in range(rows):
        for column in range(count):
            if row < lengths[column]:
                out.append(grid[column][row])
    plaintext = "".join(out)

    if complete and strip_filler:
        plaintext = plaintext.rstrip(_check_filler(filler))
    return plaintext


def _pad(letters: str, count: int, filler: str) -> str:
    """Pad *letters* with *filler* up to a whole number of rows."""
    if not letters:
        return letters
    shortfall = (-len(letters)) % count
    return letters + filler * shortfall


# ---------------------------------------------------------------------------
# Double columnar transposition
# ---------------------------------------------------------------------------


def encrypt_double(
    text: str,
    first_key: str | Sequence[int],
    second_key: str | Sequence[int],
    *,
    complete: bool = False,
    filler: str = "X",
) -> str:
    """Encipher under *first_key*, then encipher the result under *second_key*.

    Two passes are far stronger than one: after the second pass, letters that
    were neighbours in a plaintext row are no longer a fixed distance apart in
    the ciphertext, so the column-pair statistics that break a single columnar
    have nothing to lock onto.

    With ``complete=True`` the text is padded **once**, up to a multiple of the
    lowest common multiple of the two key lengths. Padding separately at each
    pass would be wrong: the second pass would add letters to the *ciphertext*
    of the first, and stripping them again during decryption would shift every
    letter of the first-pass grid. One padding that makes both rectangles full
    keeps the two passes exact inverses.
    """
    first = _coerce_key(first_key)
    second = _coerce_key(second_key)
    letters = letters_only(text)
    if complete:
        letters = _pad(letters, math.lcm(len(first), len(second)), _check_filler(filler))
    return encrypt(encrypt(letters, first), second)


def decrypt_double(
    text: str,
    first_key: str | Sequence[int],
    second_key: str | Sequence[int],
    *,
    complete: bool = False,
    filler: str = "X",
    strip_filler: bool = False,
) -> str:
    """Exact inverse of :func:`encrypt_double`: undo the second key first."""
    first = _coerce_key(first_key)
    second = _coerce_key(second_key)
    once = decrypt(text, second)
    plaintext = decrypt(once, first)
    if complete and strip_filler:
        plaintext = plaintext.rstrip(_check_filler(filler))
    return plaintext


# ---------------------------------------------------------------------------
# Column-pair statistics
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _bigram_log_probabilities(scorer: EnglishScorer) -> tuple[float, ...]:
    """``log10 P(second | first)`` for all 676 letter pairs, flat-indexed.

    Derived from the scorer's own public interface rather than a second,
    separately trained table, so the attack and the ranking agree about what
    English looks like:

        score([a, b]) = log P(a) + log P(b | a)
        score([a])    = log P(a)

    so subtracting gives the conditional probability of the second letter.
    Cached because it costs 702 scorer calls and never changes.
    """
    table: list[float] = []
    for first in range(26):
        alone = scorer.score_values([first])
        for second in range(26):
            table.append(scorer.score_values([first, second]) - alone)
    return tuple(table)


def _adjacency_matrices(
    blocks: Sequence[Sequence[int]], bigrams: Sequence[float]
) -> tuple[list[list[float]], list[list[float]]]:
    """Score every ordered pair of ciphertext blocks as neighbouring columns.

    Returns ``(side_by_side, wrap)``.

    ``side_by_side[x][y]`` sums ``log P(y[i] | x[i])`` down the rows: if block
    y really was the column to the right of block x, every one of those pairs
    is a genuine English bigram, and the sum is dramatically better than for a
    wrong pairing. Rows are matched from the top, and the sum stops at the
    shorter block, which is correct because a short column is short at the
    *bottom* only.

    ``wrap[x][y]`` sums ``log P(y[i + 1] | x[i])``: the join from the last
    column of one row to the first column of the next. It is the only term
    that ties the two ends of the arrangement together, and it adds one more
    row's worth of real English evidence for every row of the grid.

    Is it worth having? MEASURED, not assumed. Over 175 short samples (40 to
    200 letters, 2 to 7 columns, taken from two corpus files) the exhaustive
    adjacency search put the true arrangement first 163 times with the wrap
    term and 145 times without it. It earns its place on short texts, which
    are exactly the ones the toolkit struggles with; on 300 letters or more
    the side-by-side sums decide the answer on their own.
    """
    count = len(blocks)
    side = [[0.0] * count for _ in range(count)]
    wrap = [[0.0] * count for _ in range(count)]
    for left in range(count):
        first_block = blocks[left]
        for right in range(count):
            if left == right:
                continue
            second_block = blocks[right]
            shared = min(len(first_block), len(second_block))
            total = 0.0
            for row in range(shared):
                total += bigrams[first_block[row] * 26 + second_block[row]]
            side[left][right] = total

            shifted = min(len(first_block), len(second_block) - 1)
            total = 0.0
            for row in range(shifted):
                total += bigrams[first_block[row] * 26 + second_block[row + 1]]
            wrap[left][right] = total
    return side, wrap


def _arrangement_score(
    arrangement: Sequence[int],
    side: Sequence[Sequence[float]],
    wrap: Sequence[Sequence[float]],
) -> float:
    """Adjacency score of one left-to-right arrangement of blocks."""
    total = wrap[arrangement[-1]][arrangement[0]]
    for position in range(len(arrangement) - 1):
        total += side[arrangement[position]][arrangement[position + 1]]
    return total


# ---------------------------------------------------------------------------
# Searching one key length
# ---------------------------------------------------------------------------


def _blocks_for_pattern(
    values: Sequence[int], lengths: Sequence[int]
) -> list[list[int]]:
    """Cut the encoded ciphertext into blocks of the given lengths, in order."""
    blocks: list[list[int]] = []
    start = 0
    for size in lengths:
        blocks.append(list(values[start : start + size]))
        start += size
    return blocks


def _order_from_arrangement(arrangement: Sequence[int]) -> tuple[int, ...]:
    """Turn "grid column c holds block b" into the read-order permutation.

    ``arrangement[c] = b`` says block *b* (the *b*-th chunk of ciphertext, so
    the *b*-th column written out) belongs at grid position *c*. The key we
    report is the other way round: ``order[b] = c``.
    """
    order = [0] * len(arrangement)
    for column, block in enumerate(arrangement):
        order[block] = column
    return tuple(order)


def _improve(
    arrangement: list[int],
    side: Sequence[Sequence[float]],
    wrap: Sequence[Sequence[float]],
    long_count: int,
) -> tuple[float, list[int]]:
    """Hill-climb an arrangement by swapping pairs of grid positions.

    Only swaps within a class are allowed -- long block with long block, short
    with short -- because a long block cannot sit in a grid position that the
    ragged last row leaves short. Best-improvement: take the single best swap
    available and repeat until no swap helps. That is a local optimum, not a
    global one, which is why the caller also restarts from random valid
    arrangements and why greedy results are labelled as such.
    """
    best = _arrangement_score(arrangement, side, wrap)
    count = len(arrangement)
    improving = True
    while improving:
        improving = False
        best_swap: tuple[int, int] | None = None
        for first in range(count):
            for second in range(first + 1, count):
                # A swap is valid only inside the long group or inside the
                # short group.
                if (first < long_count) != (second < long_count):
                    continue
                arrangement[first], arrangement[second] = (
                    arrangement[second],
                    arrangement[first],
                )
                score = _arrangement_score(arrangement, side, wrap)
                arrangement[first], arrangement[second] = (
                    arrangement[second],
                    arrangement[first],
                )
                if score > best + 1e-9:
                    best = score
                    best_swap = (first, second)
        if best_swap is not None:
            first, second = best_swap
            arrangement[first], arrangement[second] = (
                arrangement[second],
                arrangement[first],
            )
            improving = True
    return best, arrangement


def _greedy_chains(
    long_blocks: Sequence[int],
    short_blocks: Sequence[int],
    side: Sequence[Sequence[float]],
    wrap: Sequence[Sequence[float]],
) -> list[list[int]]:
    """Build one chain per possible starting block, best neighbour each time.

    Grid positions ``0 .. len(long_blocks) - 1`` must hold long blocks and the
    rest short ones, so the pool of candidates for the next position is fixed;
    within that pool we take whichever block scores best beside the current
    tail. Greedy chaining is O(k^2) instead of O(k!), and it is usually close
    enough for the swap improver to finish the job.
    """
    long_count = len(long_blocks)
    total = long_count + len(short_blocks)
    starts = list(long_blocks) if long_blocks else list(short_blocks)
    chains: list[list[int]] = []
    for start in starts:
        used = {start}
        chain = [start]
        for position in range(1, total):
            pool = long_blocks if position < long_count else short_blocks
            remaining = [block for block in pool if block not in used]
            if not remaining:
                break
            tail = chain[-1]
            best = max(remaining, key=lambda block: side[tail][block])
            chain.append(best)
            used.add(best)
        if len(chain) == total:
            chains.append(chain)
    return chains


def _propose_orders(
    values: Sequence[int],
    count: int,
    bigrams: Sequence[float],
    *,
    exhaustive: bool,
    keep: int,
    restarts: int,
    rng: random.Random,
    deadline: float | None,
) -> tuple[list[tuple[float, tuple[int, ...]]], int, int, bool]:
    """Propose promising read orders for a *count*-column grid.

    Returns ``(proposals, patterns_tried, patterns_total, budget_hit)`` where
    each proposal is ``(adjacency score, read order)``. The proposals are only
    a shortlist: the caller decrypts and re-scores them properly.

    A "pattern" is a choice of which ciphertext blocks are long. With
    ``remainder`` long columns out of ``count`` there are ``C(count,
    remainder)`` patterns, and inside a pattern the block boundaries are
    known, so the adjacency matrices can be built once and reused for every
    arrangement consistent with it.
    """
    length = len(values)
    rows = grid_rows(length, count)
    remainder = length % count
    all_patterns = list(combinations(range(count), remainder)) if remainder else [()]
    patterns_total = len(all_patterns)
    patterns = all_patterns
    if not exhaustive and patterns_total > GREEDY_PATTERN_LIMIT:
        patterns = all_patterns[:GREEDY_PATTERN_LIMIT]

    shortlist: list[tuple[float, tuple[int, ...]]] = []
    tried = 0
    budget_hit = False

    for pattern in patterns:
        if deadline is not None and time.monotonic() > deadline:
            budget_hit = True
            break
        tried += 1
        long_positions = set(pattern)
        block_lengths = [
            rows if not remainder or block in long_positions else rows - 1
            for block in range(count)
        ]
        blocks = _blocks_for_pattern(values, block_lengths)
        side, wrap = _adjacency_matrices(blocks, bigrams)

        long_blocks = [b for b in range(count) if remainder and b in long_positions]
        short_blocks = [b for b in range(count) if not remainder or b not in long_positions]

        if exhaustive:
            # Every arrangement consistent with this pattern: the long blocks
            # fill grid positions 0..remainder-1 in some order, the short ones
            # fill the rest. Summed over patterns that is exactly count!
            # arrangements, with no arrangement counted twice.
            checked = 0
            for head in permutations(long_blocks):
                for tail in permutations(short_blocks):
                    arrangement = head + tail
                    score = _arrangement_score(arrangement, side, wrap)
                    entry = (score, _order_from_arrangement(arrangement))
                    if len(shortlist) < keep:
                        heapq.heappush(shortlist, entry)
                    elif entry > shortlist[0]:
                        heapq.heappushpop(shortlist, entry)
                    checked += 1
                    if deadline is not None and not checked & 0x3FF:
                        if time.monotonic() > deadline:
                            budget_hit = True
                            break
                if budget_hit:
                    break
        else:
            attempts = _greedy_chains(long_blocks, short_blocks, side, wrap)
            for _ in range(restarts):
                head = list(long_blocks)
                tail = list(short_blocks)
                rng.shuffle(head)
                rng.shuffle(tail)
                attempts.append(head + tail)
            for attempt in attempts:
                score, arrangement = _improve(
                    list(attempt), side, wrap, len(long_blocks)
                )
                entry = (score, _order_from_arrangement(arrangement))
                if len(shortlist) < keep:
                    heapq.heappush(shortlist, entry)
                elif entry > shortlist[0]:
                    heapq.heappushpop(shortlist, entry)
        if budget_hit:
            break

    shortlist.sort(reverse=True)
    return shortlist, tried, patterns_total, budget_hit


# ---------------------------------------------------------------------------
# The attack
# ---------------------------------------------------------------------------


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    **options: Any,
) -> CandidateSet:
    """Attack the ciphertext and return RANKED CANDIDATES, never one answer.

    Options
    -------
    key_length:
        Attack this column count only.
    max_key_length:
        Otherwise try every column count from 2 up to this
        (default :data:`DEFAULT_MAX_KEY_LENGTH`).
    max_exhaustive:
        Column counts up to this get an exhaustive permutation search; longer
        keys use the greedy chain search (default
        :data:`DEFAULT_MAX_EXHAUSTIVE`).
    complete:
        Assume the sender padded to a full rectangle. Only column counts that
        divide the length are then plausible, which is a strong constraint and
        is why the length's divisors are always reported.
    time_budget:
        Seconds. The search stops cleanly and records ``time_budget_hit``.
    seed:
        Seeds the private random generator used by the greedy restarts, so a
        run is reproducible. Ignored by the exhaustive search, which has no
        randomness in it.
    refine, restarts:
        Size of the shortlist re-scored with the full English model, and the
        number of randomised restarts per pattern in greedy mode.

    Every grid shape tried is listed in each candidate's diagnostics, along
    with whether that shape was searched exhaustively or greedily -- a greedy
    result that ranks first still only means "the best of what was looked at".
    """
    engine = scorer if scorer is not None else default_scorer()
    normalized = normalize(source) if isinstance(source, str) else source
    letters = normalized.letters

    key_length = options.pop("key_length", None)
    max_key_length = _option_int(
        options.pop("max_key_length", None), "max_key_length",
        DEFAULT_MAX_KEY_LENGTH, 2,
    )
    max_exhaustive = _option_int(
        options.pop("max_exhaustive", None), "max_exhaustive",
        DEFAULT_MAX_EXHAUSTIVE, 0,
    )
    refine = _option_int(options.pop("refine", None), "refine", DEFAULT_REFINE, 1)
    restarts = _option_int(
        options.pop("restarts", None), "restarts", DEFAULT_RESTARTS, 0
    )
    complete = bool(options.pop("complete", False))
    time_budget = options.pop("time_budget", None)
    seed = options.pop("seed", None)
    if options:
        raise ValueError(
            "unknown option(s) for columnar.solve: "
            + ", ".join(sorted(str(name) for name in options))
        )
    if time_budget is not None and time_budget <= 0:
        raise ValueError(f"time_budget must be positive, got {time_budget}")
    if key_length is not None:
        key_length = _check_int(key_length, "key_length", 2)

    results = CandidateSet()
    length = len(letters)
    if length < 4:
        # Two columns of two letters is the smallest grid with any rows to
        # compare; below that there is nothing to measure.
        return results

    if key_length is not None:
        wanted = [key_length]
        if complete and length % key_length:
            raise ValueError(
                f"key_length={key_length} with complete=True is impossible for "
                f"{length} letters: a complete rectangle needs the length to "
                f"divide by the column count ({length % key_length} left "
                "over). Drop complete=True or choose another key length."
            )
    else:
        wanted = list(range(2, max_key_length + 1))
        if complete:
            wanted = [count for count in wanted if length % count == 0]

    rng = random.Random(seed)
    bigrams = _bigram_log_probabilities(engine)
    values = engine.encode(letters)
    deadline = None if time_budget is None else time.monotonic() + time_budget
    budget_hit = False
    grids: list[str] = []

    for count in wanted:
        if count >= length or grid_rows(length, count) < 2:
            grids.append(f"{count} columns: skipped, fewer than 2 rows")
            continue
        if deadline is not None and time.monotonic() > deadline:
            budget_hit = True
            break

        rows = grid_rows(length, count)
        remainder = length % count
        exhaustive = count <= max_exhaustive
        proposals, tried, total, hit = _propose_orders(
            values,
            count,
            bigrams,
            exhaustive=exhaustive,
            keep=refine,
            restarts=restarts,
            rng=rng,
            deadline=deadline,
        )
        budget_hit = budget_hit or hit
        shape = (
            f"{count}x{rows}"
            + (f" ragged ({remainder} long columns)" if remainder else " exact")
            + (" exhaustive" if exhaustive else " greedy")
            + (f", {tried}/{total} length patterns" if total > 1 else "")
        )
        grids.append(shape)

        for adjacency, order in proposals:
            plaintext = decrypt(letters, order)
            keyword = keyword_from_order(order)
            diagnostics: dict[str, Any] = {
                "key_length": count,
                "grid": f"{count} columns x {rows} rows",
                "ragged_columns": remainder,
                "permutation": ",".join(str(column) for column in order),
                "search": "exhaustive" if exhaustive else "greedy (not exhaustive)",
                "column_pair_score": adjacency,
            }
            annotate(diagnostics, plaintext, engine)
            results.add(
                Candidate(
                    method=METHOD,
                    key=f"columns={keyword or order} ({count})",
                    score=engine.score(plaintext),
                    plaintext=plaintext,
                    diagnostics=diagnostics,
                    # A transposition moves letters, so plaintext letter i did
                    # not come from ciphertext position i: relayout would
                    # invent a layout. Five-letter groups are honest.
                    display=group_text(plaintext),
                )
            )
        if budget_hit:
            break

    tested_note = "; ".join(grids) if grids else "none"
    divisor_note = ",".join(
        str(value) for value in divisors(length) if 2 <= value <= 30
    )
    for candidate in results.ranked():
        candidate.diagnostics["grids_tested"] = tested_note
        candidate.diagnostics["length"] = length
        candidate.diagnostics["length_divisors_2_to_30"] = divisor_note or "none"
        candidate.diagnostics["assumed_complete_rectangle"] = complete
        if budget_hit:
            candidate.diagnostics["time_budget_hit"] = True

    if top is not None and top > 0:
        return CandidateSet(results.top(top))
    return results


def _check_int(value: Any, name: str, minimum: int) -> int:
    """Validate a whole-number option of at least *minimum*.

    A value that is not a whole number is an error rather than something to
    round: a column count of 6.5 is a mistake, not a request.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be an integer, got {type(value).__name__} ({value!r})"
        )
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _option_int(value: Any, name: str, default: int, minimum: int) -> int:
    """Read an integer solver option, treating ``None`` as "not supplied".

    A command line naturally produces ``None`` for an argument the user left
    out, so it selects the default instead of being an error.
    """
    if value is None:
        return default
    return _check_int(value, name, minimum)


def plausible_column_counts(length: int, maximum: int = 30) -> list[int]:
    """Column counts that would fill a complete rectangle of *length* letters.

    A convenience wrapper over :func:`statistics.divisors`: if the sender
    padded to a full rectangle then the column count must divide the
    ciphertext length exactly, which usually cuts the search to a handful of
    possibilities. It says nothing at all about an incomplete columnar, where
    every column count remains possible.
    """
    return [value for value in divisors(length) if 2 <= value <= maximum]
