"""The permutation cipher: one fixed shuffle, applied to every block.

The cipher
----------
Cut the plaintext into blocks of a fixed size and rearrange the letters
inside each block the same way every time::

    key BAEDC -> read order (1, 0, 4, 3, 2), block size 5

        plaintext   I N F I L | T R A T I | N G T H E
        read order  1 0 4 3 2 | 1 0 4 3 2 | 1 0 4 3 2
        ciphertext  N I L I F | R T I T A | G N E H T

Nothing but position changes, so the ciphertext has exactly the letter
frequencies of the plaintext -- the tell every transposition leaves.

Why it is not a columnar transposition
--------------------------------------
It looks like one and it is not, and the difference is the whole reason this
module exists. A columnar transposition writes the message into a grid and
reads out one WHOLE COLUMN at a time, so the letter at plaintext position 3
can end up thousands of places away. A permutation cipher never moves a
letter out of its own block, so displacement is bounded by the block size.

The toolkit could already break columnar, double columnar, rail fence and
every route through a grid, and none of them can express this. MEASURED on
the 2,142 letters of the 2018 National Cipher Challenge, challenge 6B: the
columnar solver reached ``weak`` and the route/grid solver reached ``weak``,
because no key either of them can write down describes the cipher. The
message is a period-5 permutation under the key BAEDC and it decrypts to
clean English -- "INFILTRATING THE DELIBERATIONS OF OUR ENEMIES IS A
PRINCIPAL GOAL...". A whole family was missing, not a harder case of one
that was present.

The attack
----------
The search is the same shape as the columnar attack, and it deliberately
reuses that code: take the letters at block offset ``a`` of every block as a
"stripe", and score putting stripe ``y`` immediately after stripe ``x`` by
summing ``log P(y[i] | x[i])`` down the blocks. If ``y`` really did follow
``x`` in the plaintext, every one of those pairs is a real English bigram
and the sum is far better than for any wrong pairing. That reduces the
problem from "decrypt and score the whole message" to "score n by n numbers
once, then rank permutations in n operations each", which is what makes an
exhaustive sweep affordable.

Two things are worth stating plainly because they are easy to get wrong:

* The ragged last block is dropped when the statistics are built, and only
  then. A partial block contributes at most a handful of pairs and would
  need its own special case in every matrix; dropping it costs nothing and
  keeps the arithmetic honest. It is still decrypted, because a reader wants
  the whole message.

* The identity permutation is never offered. It would "decrypt" any text to
  itself and would therefore be the top-scoring candidate for every piece of
  plain English ever pasted in, which is a confident answer to a question
  nobody asked.
"""

from __future__ import annotations

import itertools
import random
import time
from typing import Any, Sequence

from .candidates import Candidate, CandidateSet
from .columnar import (
    adjacency_matrices,
    arrangement_score,
    bigram_log_probabilities,
    key_order,
    keyword_from_order,
)
from .normalize import NormalizedText, group_text, letters_only, normalize
from .scoring import EnglishScorer, annotate, default_scorer

#: Largest block size :func:`solve` tries when it is not told one. Keys in
#: this competition are keywords, so a block is a word length; twelve covers
#: every one that has ever been set and leaves headroom.
DEFAULT_MAX_PERIOD = 12

#: Block sizes up to this are enumerated exhaustively -- every permutation,
#: no search heuristics, no chance of missing the answer. The cost is the
#: factorial, but paid against the n-by-n adjacency matrices rather than
#: against the message: 8! is 40,320 arrangements scored in eight additions
#: each. Above this the sweep falls back to restarts and hill climbing, and
#: the candidates say so.
DEFAULT_MAX_EXHAUSTIVE = 8

#: Restarts used per block size once the search is no longer exhaustive.
#: Deliberately many and short rather than few and long: measured on double
#: columnar, twelve restarts of thirty thousand steps found a key that six of
#: eighty thousand did not, at a fifth of the cost. Diversification beats
#: persistence when the landscape is full of local optima, and a permutation
#: landscape always is.
DEFAULT_RESTARTS = 24

#: How many arrangements per block size are decrypted in full and scored with
#: the real English model. The adjacency sum is a bigram approximation and
#: ranks the truth highly rather than always first, so a shortlist is checked
#: properly instead of trusting the proxy outright.
DEFAULT_REFINE = 8

#: A block of one letter is not a rearrangement of anything.
MIN_PERIOD = 2

METHOD = "Block permutation"


# ---------------------------------------------------------------------------
# The cipher
# ---------------------------------------------------------------------------


def _coerce_key(key: str | Sequence[int]) -> tuple[int, ...]:
    """Accept a keyword or an explicit read order, as columnar does."""
    if isinstance(key, str):
        return key_order(key)
    return _check_permutation(key)


def _check_permutation(order: Sequence[int]) -> tuple[int, ...]:
    """Validate an explicit read order given as block offsets."""
    try:
        values = tuple(order)
    except TypeError as error:
        raise ValueError(
            "a permutation key must be a keyword or a sequence of block "
            f"offsets, got {order!r}"
        ) from error
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "block offsets must be whole numbers, got "
                f"{value!r} in {values!r}"
            )
    if len(values) < MIN_PERIOD:
        raise ValueError(
            f"a permutation key needs at least {MIN_PERIOD} positions, got "
            f"{values!r}. A block of one letter is not a rearrangement."
        )
    if sorted(values) != list(range(len(values))):
        raise ValueError(
            f"{values!r} is not a permutation: a key of {len(values)} "
            f"positions must use each of 0..{len(values) - 1} exactly once"
        )
    return values


def _partial_order(order: Sequence[int], size: int) -> tuple[int, ...]:
    """The key restricted to a final block of *size* letters.

    Keeping the entries that still point at a real letter, in their original
    order, is the only reading that stays a permutation and the only one that
    agrees with the full-block case as ``size`` grows. It matters: the last
    two letters of the 2018 challenge 6B message are the "LE" of
    CONSTANTINOPLE, and leaving a short block alone spells it CONSTANTINOPEL.
    A cipher that is right for 2,140 letters and wrong for two is still
    wrong, and it is wrong in the place a reader looks last.
    """
    return tuple(value for value in order if value < size)


def encrypt(text: str, key: str | Sequence[int]) -> str:
    """Encrypt. Operates on letters only; returns letters only, uppercase.

    Block ``j`` of the output takes the letter that stood at offset
    ``order[j]`` of the input block.
    """
    order = _coerce_key(key)
    letters = letters_only(text)
    period = len(order)
    out: list[str] = []
    for start in range(0, len(letters), period):
        block = letters[start:start + period]
        pattern = (order if len(block) == period
                   else _partial_order(order, len(block)))
        out.append("".join(block[offset] for offset in pattern))
    return "".join(out)


def decrypt(text: str, key: str | Sequence[int]) -> str:
    """Exact inverse of :func:`encrypt` for the same key."""
    order = _coerce_key(key)
    letters = letters_only(text)
    period = len(order)
    out: list[str] = []
    for start in range(0, len(letters), period):
        block = letters[start:start + period]
        pattern = (order if len(block) == period
                   else _partial_order(order, len(block)))
        restored = [""] * len(block)
        for position, offset in enumerate(pattern):
            restored[offset] = block[position]
        out.append("".join(restored))
    return "".join(out)


# ---------------------------------------------------------------------------
# The attack
# ---------------------------------------------------------------------------


def _stripes(values: Sequence[int], period: int) -> list[list[int]]:
    """The letters at each block offset, one list per offset.

    The ragged final block is dropped, so every stripe has the same length
    and the adjacency sums compare like with like.
    """
    blocks = len(values) // period
    return [[values[block * period + offset] for block in range(blocks)]
            for offset in range(period)]


def _lift_and_reinsert(
    arrangement: list[int],
    side: Sequence[Sequence[float]],
    wrap: Sequence[Sequence[float]],
) -> tuple[float, list[int]]:
    """Hill-climb with both swaps and lift-and-reinsert moves.

    Swaps alone search a permutation badly, and the reason is worth keeping:
    an arrangement that is right except that one position sits a single place
    too early needs everything after it to shift along by one, and no swap of
    two positions does that. Lifting one position out and dropping it back
    somewhere else does it in a single move. Both neighbourhoods are searched
    to exhaustion, best-improvement, until neither helps.
    """
    best = arrangement_score(arrangement, side, wrap)
    count = len(arrangement)
    improving = True
    while improving:
        improving = False
        proposal: list[int] | None = None

        for first in range(count):
            for second in range(first + 1, count):
                arrangement[first], arrangement[second] = (
                    arrangement[second], arrangement[first])
                score = arrangement_score(arrangement, side, wrap)
                arrangement[first], arrangement[second] = (
                    arrangement[second], arrangement[first])
                if score > best + 1e-9:
                    best = score
                    proposal = list(arrangement)
                    proposal[first], proposal[second] = (
                        proposal[second], proposal[first])

        for taken in range(count):
            rest = arrangement[:taken] + arrangement[taken + 1:]
            value = arrangement[taken]
            for place in range(count):
                if place == taken:
                    continue
                trial = rest[:place] + [value] + rest[place:]
                score = arrangement_score(trial, side, wrap)
                if score > best + 1e-9:
                    best = score
                    proposal = trial

        if proposal is not None:
            arrangement[:] = proposal
            improving = True
    return best, arrangement


def _propose(
    stripes: Sequence[Sequence[int]],
    side: Sequence[Sequence[float]],
    wrap: Sequence[Sequence[float]],
    *,
    exhaustive: bool,
    keep: int,
    restarts: int,
    rng: random.Random,
) -> list[tuple[float, tuple[int, ...]]]:
    """The best few arrangements for one block size, best first."""
    period = len(stripes)
    identity = tuple(range(period))
    seen: dict[tuple[int, ...], float] = {}

    if exhaustive:
        for candidate in itertools.permutations(range(period)):
            if candidate == identity:
                continue
            seen[candidate] = arrangement_score(candidate, side, wrap)
    else:
        starts = [list(identity)]
        starts.extend(rng.sample(range(period), period)
                      for _ in range(restarts))
        for start in starts:
            score, settled = _lift_and_reinsert(list(start), side, wrap)
            key = tuple(settled)
            if key != identity and key not in seen:
                seen[key] = score

    ranked = sorted(seen.items(), key=lambda item: item[1], reverse=True)
    return [(score, order) for order, score in ranked[:keep]]


def _order_from_arrangement(arrangement: Sequence[int]) -> tuple[int, ...]:
    """Turn "plaintext position i came from block offset a" into the key.

    :func:`decrypt` reads its key the other way round -- ``order[j]`` is the
    plaintext offset that ciphertext position ``j`` restores to -- so the
    arrangement has to be inverted before it can be reported or used.
    """
    order = [0] * len(arrangement)
    for position, offset in enumerate(arrangement):
        order[offset] = position
    return tuple(order)


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
    period:
        Attack this block size only.
    max_period:
        Otherwise try every size from 2 up to this
        (default :data:`DEFAULT_MAX_PERIOD`).
    max_exhaustive:
        Sizes up to this enumerate every permutation
        (default :data:`DEFAULT_MAX_EXHAUSTIVE`).
    refine:
        Arrangements per size decrypted and scored properly.
    restarts:
        Restarts per size once the search is no longer exhaustive.
    seed:
        Makes the non-exhaustive sizes reproducible.
    time_budget:
        Seconds. The sweep stops cleanly between block sizes and every
        candidate records ``time_budget_hit``.
    """
    engine = scorer if scorer is not None else default_scorer()
    normalized = normalize(source) if isinstance(source, str) else source
    letters = normalized.letters

    period = options.pop("period", None)
    max_period = _option_int(options.pop("max_period", None), "max_period",
                             DEFAULT_MAX_PERIOD, MIN_PERIOD)
    max_exhaustive = _option_int(options.pop("max_exhaustive", None),
                                 "max_exhaustive", DEFAULT_MAX_EXHAUSTIVE, 2)
    refine = _option_int(options.pop("refine", None), "refine",
                         DEFAULT_REFINE, 1)
    restarts = _option_int(options.pop("restarts", None), "restarts",
                           DEFAULT_RESTARTS, 1)
    seed = options.pop("seed", None)
    time_budget = options.pop("time_budget", None)
    if options:
        raise ValueError(
            "unknown option(s) for permutation.solve: "
            + ", ".join(sorted(str(name) for name in options))
        )
    if time_budget is not None and time_budget <= 0:
        raise ValueError(f"time_budget must be positive, got {time_budget}")

    results = CandidateSet(source_letters=letters)
    if period is not None:
        periods = [_option_int(period, "period", MIN_PERIOD, MIN_PERIOD)]
    else:
        periods = list(range(MIN_PERIOD, max_period + 1))

    # Two whole blocks is the least that gives the adjacency sums anything to
    # work with; below that the answer would be a guess dressed as a search.
    periods = [size for size in periods if len(letters) >= 2 * size]
    if not periods:
        return results

    bigrams = bigram_log_probabilities(engine)
    values = engine.encode(letters)
    rng = random.Random(seed)
    deadline = None if time_budget is None else time.monotonic() + time_budget
    budget_hit = False
    tried: list[int] = []

    for size in periods:
        if deadline is not None and time.monotonic() >= deadline:
            budget_hit = True
            break
        tried.append(size)
        stripes = _stripes(values, size)
        side, wrap = adjacency_matrices(stripes, bigrams)
        exhaustive = size <= max_exhaustive
        for adjacency, arrangement in _propose(
            stripes, side, wrap, exhaustive=exhaustive, keep=refine,
            restarts=restarts, rng=rng,
        ):
            order = _order_from_arrangement(arrangement)
            plaintext = decrypt(letters, order)
            diagnostics: dict[str, Any] = {
                "period": size,
                "read_order": ",".join(str(value) for value in order),
                "blocks": len(letters) // size,
                "ragged_tail": len(letters) % size,
                "adjacency_score": round(adjacency, 3),
                "search": ("exhaustive over every permutation" if exhaustive
                           else f"{restarts} restarts, hill climbing"),
            }
            annotate(diagnostics, plaintext, engine)
            results.add(
                Candidate(
                    method=METHOD,
                    key=f"{keyword_from_order(order) or order} ({size})",
                    score=engine.score(plaintext),
                    plaintext=plaintext,
                    diagnostics=diagnostics,
                    # A transposition moves letters, so plaintext position i
                    # did not come from ciphertext position i and the original
                    # spacing would be an invention.
                    display=group_text(plaintext),
                )
            )

    summary = f"{tried[0]}-{tried[-1]}" if tried else "none"
    for candidate in results.ranked():
        candidate.diagnostics["periods_tried"] = summary
        if budget_hit:
            candidate.diagnostics["time_budget_hit"] = True

    if top is not None and top > 0:
        return CandidateSet(results.top(top), source_letters=letters)
    return results


def _option_int(value: Any, name: str, default: int, minimum: int) -> int:
    """Read an integer solver option, validating it.

    ``None`` means "not supplied", which is what a command line produces for
    an argument left out, so it selects the default. Anything else that is
    not a whole number is an error rather than something to round.
    """
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be an integer, got {type(value).__name__} ({value!r})"
        )
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value
