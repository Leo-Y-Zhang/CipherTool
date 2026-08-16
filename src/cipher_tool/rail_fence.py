"""Rail fence (zigzag) transposition, and an exhaustive attack on it.

The cipher
----------
Write the plaintext downwards across a set of imaginary rails, bouncing off
the top and the bottom rail as you go, then read the rails off one after
another::

    rails = 3, plaintext ATTACKATDAWN

        A . . . C . . . D . . .
        . T . A . K . T . A . N
        . . T . . . A . . . W .

    ciphertext = ACD + TAKTAN + TAW = ACDTAKTANTAW

Only the *positions* of the letters change, never their identities, so the
ciphertext has exactly the letter frequencies of the plaintext. That is the
tell a transposition always leaves: an English index of coincidence and a
tiny chi-squared against English (both measured by ``statistics.analyse``),
over text that reads as nonsense.

The zigzag as arithmetic
------------------------
Going down all the rails and back up again visits

    rails + (rails - 2) = 2 * rails - 2

positions before the pattern repeats: the top and bottom rails are visited
once per cycle, every rail in between twice. So the rail used at step ``t``
depends only on ``t`` modulo that cycle length::

    p = (t + offset) mod (2 * rails - 2)

    rail(t) = p                      if p < rails     (travelling down)
              (2 * rails - 2) - p    otherwise        (travelling back up)

The second line is a reflection about the bottom rail: cycle position
``rails`` is rail ``rails - 2``, position ``rails + 1`` is rail ``rails - 3``,
and so on up to position ``2 * rails - 3``, which is rail 1.

The offset variant
------------------
``offset`` starts the walk part of the way through the zigzag instead of at
the top rail, which is a genuine extra key: with three rails and offset 1 the
first letter lands on the middle rail already travelling downwards, so the
whole pattern of rail lengths changes and the ciphertext differs. Because the
zigzag is periodic with period ``2 * rails - 2``, two offsets that differ by a
whole cycle describe the identical walk, so an offset is reduced modulo the
cycle length. That is an identity of the cipher, not a guess about the user's
intent, and it is the only tidying this module does to a key.

The attack
----------
The key space is tiny. With at most 20 rail counts and at most ``2*20-2 = 38``
offsets each, fewer than 400 decryptions cover every rail fence a competition
could set, so :func:`solve` simply tries them all and ranks the results with
the English scorer. No statistics are needed to steer the search, which is why
a rail fence, once suspected, is always broken.
"""

from __future__ import annotations

import time
from typing import Any

from .candidates import Candidate, CandidateSet
from .normalize import NormalizedText, group_text, letters_only, normalize
from .scoring import EnglishScorer, annotate, default_scorer

#: Rail counts above this are not tried by :func:`solve`. A rail fence with
#: more rails than this leaves most rails holding one or two letters, which is
#: barely an encryption and is not used in practice.
DEFAULT_MAX_RAILS = 20

METHOD = "Rail fence"


# ---------------------------------------------------------------------------
# The zigzag itself
# ---------------------------------------------------------------------------


def cycle_length(rails: int) -> int:
    """Number of steps before the zigzag repeats: ``2 * rails - 2``.

    The top and bottom rails are visited once per cycle and each of the
    ``rails - 2`` middle rails twice, hence ``2 + 2 * (rails - 2)``.
    """
    _check_rails(rails)
    return 2 * rails - 2


def rail_sequence(length: int, rails: int, offset: int = 0) -> list[int]:
    """The rail index used by each of the first *length* letters.

    ``rail_sequence(6, 3)`` is ``[0, 1, 2, 1, 0, 1]``: down, down, bounce,
    up, bounce, down. This one function defines the cipher; encryption and
    decryption are both just bookkeeping on top of it.
    """
    _check_rails(rails)
    if length < 0:
        raise ValueError(f"length must not be negative, got {length}")
    cycle = 2 * rails - 2
    start = _normalise_offset(offset, rails)
    sequence: list[int] = []
    for step in range(length):
        position = (step + start) % cycle
        # Reflect the second half of the cycle back up the rails.
        sequence.append(position if position < rails else cycle - position)
    return sequence


def rail_counts(length: int, rails: int, offset: int = 0) -> list[int]:
    """How many letters land on each rail for a text of *length* letters.

    This is what makes decryption possible: the ciphertext is the rails
    concatenated, so we must know each rail's length before we can cut it up
    again. The counts are *not* simply "length / rails": the top and bottom
    rails are visited once per cycle and the middle rails twice, and a partial
    final cycle skews the tail. Counting the walk directly is exact, cheap and
    impossible to get subtly wrong, which is worth more than a closed form.
    """
    counts = [0] * rails
    for rail in rail_sequence(length, rails, offset):
        counts[rail] += 1
    return counts


# ---------------------------------------------------------------------------
# Encryption and decryption
# ---------------------------------------------------------------------------


def encrypt(text: str, rails: int, offset: int = 0) -> str:
    """Encrypt. Operates on letters only; returns letters only, uppercase.

    Every letter is dropped into the bucket for its rail, and the buckets are
    then concatenated from the top rail downwards.
    """
    letters = letters_only(text)
    _check_rails(rails)
    if not letters:
        return ""
    _check_rails_against_length(rails, len(letters))

    buckets: list[list[str]] = [[] for _ in range(rails)]
    for index, rail in enumerate(rail_sequence(len(letters), rails, offset)):
        buckets[rail].append(letters[index])
    return "".join("".join(bucket) for bucket in buckets)


def decrypt(text: str, rails: int, offset: int = 0) -> str:
    """Exact inverse of :func:`encrypt` for the same key.

    Decryption is the fiddly half, and it goes in three steps.

    1. Replay the zigzag for a text of this length to learn how many letters
       landed on each rail. The walk depends only on the length, the rail
       count and the offset, all of which we know without the plaintext.
    2. Cut the ciphertext into consecutive slices of exactly those lengths.
       Slice *i* is rail *i*, because encryption wrote the rails out in order.
    3. Walk the zigzag again, and at each step take the next unused letter
       from the rail that step belongs to. A cursor per rail keeps track.

    Step 1 is the step people skip: without the real per-rail counts the
    slices are wrong and the output is confidently wrong rather than obviously
    wrong, which is far more dangerous.
    """
    letters = letters_only(text)
    _check_rails(rails)
    if not letters:
        return ""
    _check_rails_against_length(rails, len(letters))

    sequence = rail_sequence(len(letters), rails, offset)
    counts = [0] * rails
    for rail in sequence:
        counts[rail] += 1

    slices: list[str] = []
    start = 0
    for rail in range(rails):
        slices.append(letters[start : start + counts[rail]])
        start += counts[rail]

    cursors = [0] * rails
    out: list[str] = []
    for rail in sequence:
        out.append(slices[rail][cursors[rail]])
        cursors[rail] += 1
    return "".join(out)


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

    Every rail count from 2 up to ``min(length - 1, max_rails)`` is tried with
    every offset in its zigzag cycle, and the decryptions are ranked by the
    English scorer. The search is exhaustive over the whole key space, so if
    the text really is a rail fence within those bounds the right key is in
    the set; the only question is whether it ranks first, which is why the
    caller is handed a ranked list rather than a verdict.

    Options
    -------
    max_rails:
        Largest rail count to try (default :data:`DEFAULT_MAX_RAILS`).
    time_budget:
        Seconds. The search stops cleanly when exceeded and every candidate
        records ``time_budget_hit``. Rarely needed: the full sweep is well
        under a second for competition-sized texts.
    seed:
        Accepted and ignored, so a caller can pass the same options to every
        solver. There is no randomness here: the search is exhaustive.
    """
    engine = scorer if scorer is not None else default_scorer()
    normalized = normalize(source) if isinstance(source, str) else source
    letters = normalized.letters

    max_rails = _option_int(
        options.pop("max_rails", None), "max_rails", DEFAULT_MAX_RAILS, 1
    )
    time_budget = options.pop("time_budget", None)
    options.pop("seed", None)  # documented no-op; the search is deterministic
    if options:
        raise ValueError(
            "unknown option(s) for rail_fence.solve: "
            + ", ".join(sorted(str(name) for name in options))
        )
    if time_budget is not None and time_budget <= 0:
        raise ValueError(f"time_budget must be positive, got {time_budget}")

    results = CandidateSet()
    length = len(letters)
    # A rail fence needs at least two rails and at least one more letter than
    # rails, so nothing below three letters can be a rail fence at all.
    if length < 3:
        return results

    highest = min(length - 1, max_rails)
    if highest < 2:
        return results

    deadline = None if time_budget is None else time.monotonic() + time_budget
    budget_hit = False
    tested = 0
    rails_tested: list[int] = []

    for rails in range(2, highest + 1):
        if deadline is not None and time.monotonic() > deadline:
            budget_hit = True
            break
        rails_tested.append(rails)
        for offset in range(cycle_length(rails)):
            plaintext = decrypt(letters, rails, offset)
            tested += 1
            diagnostics: dict[str, Any] = {
                "rails": rails,
                "offset": offset,
                "cycle_length": cycle_length(rails),
                "rail_lengths": ",".join(
                    str(count) for count in rail_counts(length, rails, offset)
                ),
            }
            annotate(diagnostics, plaintext, engine)
            results.add(
                Candidate(
                    method=METHOD,
                    key=f"rails={rails} offset={offset}",
                    score=engine.score(plaintext),
                    plaintext=plaintext,
                    diagnostics=diagnostics,
                    # NOT normalized.relayout(): a transposition moves letters,
                    # so plaintext letter i did not come from ciphertext
                    # position i and pouring it back into the original spacing
                    # would invent a layout that means nothing. Five-letter
                    # grouping is honest about that.
                    display=group_text(plaintext),
                )
            )

    summary = (
        f"{rails_tested[0]}-{rails_tested[-1]}" if rails_tested else "none"
    )
    for candidate in results.ranked():
        candidate.diagnostics["rails_tested"] = summary
        candidate.diagnostics["configurations_tested"] = tested
        if budget_hit:
            candidate.diagnostics["time_budget_hit"] = True

    if top is not None and top > 0:
        return CandidateSet(results.top(top))
    return results


# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------


def _check_rails(rails: int) -> None:
    """A rail count must be a whole number of at least two."""
    if isinstance(rails, bool) or not isinstance(rails, int):
        raise ValueError(
            f"rails must be an integer, got {type(rails).__name__} ({rails!r})"
        )
    if rails < 2:
        raise ValueError(
            f"rails must be at least 2, got {rails}. One rail would write the "
            "text out unchanged, which is not an encryption."
        )


def _check_rails_against_length(rails: int, length: int) -> None:
    """With ``rails >= length`` the zigzag never turns and nothing moves."""
    if rails >= length:
        raise ValueError(
            f"rails must be fewer than the {length} letters of text, got "
            f"{rails}. With that many rails every letter sits on its own rail "
            "and the ciphertext equals the plaintext."
        )


def _normalise_offset(offset: int, rails: int) -> int:
    """Validate an offset and reduce it modulo the zigzag cycle."""
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ValueError(
            f"offset must be an integer, got {type(offset).__name__} "
            f"({offset!r})"
        )
    if offset < 0:
        raise ValueError(
            f"offset must not be negative, got {offset}. Offsets count "
            "forwards through the zigzag from the top rail."
        )
    return offset % (2 * rails - 2)


def _option_int(value: Any, name: str, default: int, minimum: int) -> int:
    """Read an integer solver option, validating it.

    ``None`` means "not supplied", which a command line naturally produces for
    an argument the user left out, so it selects the default rather than being
    an error. Anything else that is not a whole number is an error: silently
    truncating 3.7 rails would be exactly the kind of quiet wrong answer this
    toolkit is not allowed to give.
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
