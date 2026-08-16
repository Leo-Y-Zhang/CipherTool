"""Bifid: fractionating a letter into two coordinates and scattering them.

Delastelle's Bifid (1901) is where the Polybius square of ``polybius.py``
stops being a curiosity and becomes a real cipher. The square turns each
letter into two coordinates; Bifid then *separates* those coordinates and
recombines them out of step, so that every ciphertext letter is built from
pieces of two different plaintext letters. That is the whole idea, and it is
why single-letter frequency analysis fails against it: the letter E no longer
maps to anything, because half of E has gone one way and half the other.

How encryption works
--------------------
Write the coordinates of each letter as a *column* underneath it, then read
the top row, then the bottom row, then re-pair the combined stream.

Worked example, using the standard square (I and J share a cell)::

        1  2  3  4  5
     1  A  B  C  D  E
     2  F  G  H  I  K
     3  L  M  N  O  P
     4  Q  R  S  T  U
     5  V  W  X  Y  Z

Take the plaintext ATTACK::

     plaintext   A  T  T  A  C  K
     row         1  4  4  1  1  2
     column      1  4  4  1  3  5

Read the row line first and the column line after it, as one stream::

     1 4 4 1 1 2   1 4 4 1 3 5

then cut that stream into pairs and read each pair as a coordinate::

     (1,4) (4,1) (1,2) (1,4) (4,1) (3,5)
       D     Q     B     D     Q     P

so ATTACK enciphers to DQBDQP. Notice what happened to the third ciphertext
letter B. Its pair is taken from positions 5 and 6 of the combined stream,
and for a six-letter message positions 1 to 6 are the whole *row* line. So
B's row coordinate is the row of the FIFTH plaintext letter (C), and B's
column coordinate is the row of the SIXTH (K). Neither of B's coordinates
has anything to do with the third plaintext letter, and neither comes from
the column line at all.

That is the general pattern for a block of even length: the first half of
the ciphertext is built entirely out of plaintext rows, and the second half
entirely out of plaintext columns. Each ciphertext letter still draws its
two coordinates from two different plaintext letters, which is exactly the
fractionation a monoalphabetic attack cannot follow.

Decryption reverses the read: write the coordinates of the ciphertext letters
in pairs, run them out as one stream of 2n numbers, cut the stream in half,
and the first half is the rows of the plaintext while the second half is its
columns.

The period
----------
Applied to a whole message, the fractionation spreads each letter's halves
arbitrarily far apart, which is strong but awkward to do by hand and, more
importantly, means a single transmission error destroys everything after it.
The practical variant -- and the one that turns up in competitions -- breaks
the message into blocks of a fixed *period* and fractionates each block
independently. ``period=5`` was Delastelle's own recommendation.

A short final block is not a special case: the arithmetic above only ever
refers to the length of the block it is working on, so a ragged tail of three
letters fractionates exactly like a block of three. There is no padding, and
the ciphertext is always the same length as the plaintext.

``period=1`` is worth knowing about: a block of one letter has stream
``[row, column]``, which re-pairs to ``(row, column)`` -- the same letter. So
period 1 is the identity, and when the solver reports it as the best answer
it is telling you the text was not fractionated at all.

The attack
----------
The period is small and the square is usually keyed by a word, so the search
space that matters is (period) x (candidate squares). :func:`solve` decrypts
under every period from 1 to ``max_period``, plus whole-message
fractionation, using the standard squares and any keyed square the operator
supplies through ``keywords``, and ranks the results by English score. Every
period tested is reported in the diagnostics.

Limitation, stated plainly: this solver does not *search* for an unknown
square. If the grid is a scrambled alphabet with no keyword behind it, the
periods will all score as noise, and the honest reading of that output is
"the square is not one of these", not "the text is not Bifid".
"""

from __future__ import annotations

import time
from typing import Iterable, Sequence

from .candidates import Candidate, CandidateSet
from .normalize import NormalizedText, normalize
from .polybius import LETTERS_AND_DIGITS, PolybiusSquare, clean_keyword
from .scoring import EnglishScorer, annotate, default_scorer

#: Periods searched by :func:`solve` unless told otherwise.
DEFAULT_MAX_PERIOD = 15


def _as_square(square: PolybiusSquare | str | None) -> PolybiusSquare:
    """Accept a square, a keyword for the standard square, or nothing."""
    if square is None:
        return PolybiusSquare.standard()
    if isinstance(square, PolybiusSquare):
        return square
    if isinstance(square, str):
        return PolybiusSquare.standard(square)
    raise ValueError(
        f"expected a PolybiusSquare, a keyword string or None, got "
        f"{type(square).__name__}"
    )


def _check_period(period: int | None) -> int:
    """Validate a period. ``None`` and ``0`` both mean the whole message."""
    if period is None:
        return 0
    if isinstance(period, bool) or not isinstance(period, int):
        raise ValueError(
            f"period must be a whole number of letters or None, got {period!r}"
        )
    if period < 0:
        raise ValueError(
            f"period must not be negative, got {period}; use 0 or None to "
            "fractionate the whole message as one block"
        )
    return period


def _blocks(length: int, period: int) -> list[tuple[int, int]]:
    """``(start, size)`` of each fractionation block.

    With ``period == 0`` the whole message is one block. Otherwise the blocks
    are consecutive and the last one is however long the message leaves it --
    ragged tails need no special handling because every block's arithmetic
    depends only on its own length.
    """
    if length <= 0:
        return []
    if period <= 0:
        return [(0, length)]
    return [
        (start, min(period, length - start))
        for start in range(0, length, period)
    ]


def encrypt(
    text: str,
    square: PolybiusSquare | str | None = None,
    period: int | None = None,
) -> str:
    """Encrypt *text* with Bifid. Operates on letters only, returns letters.

    *square* may be a :class:`PolybiusSquare`, a keyword for the standard
    keyed square, or ``None`` for the plain I/J square. *period* is the block
    length; ``None`` or ``0`` fractionates the whole message at once.
    """
    board = _as_square(square)
    size = _check_period(period)
    prepared = board.prepare(text)
    if not prepared:
        return ""

    out: list[str] = []
    for start, count in _blocks(len(prepared), size):
        pairs = [board.coordinates(char) for char in prepared[start : start + count]]
        # Rows first, then columns: the two halves of the block's stream.
        stream = [row for row, _ in pairs] + [column for _, column in pairs]
        out.append(
            "".join(
                board.letter(stream[index], stream[index + 1])
                for index in range(0, len(stream), 2)
            )
        )
    return "".join(out)


def decrypt(
    text: str,
    square: PolybiusSquare | str | None = None,
    period: int | None = None,
) -> str:
    """Exact inverse of :func:`encrypt` for the same square and period."""
    board = _as_square(square)
    size = _check_period(period)
    prepared = board.prepare(text)
    if not prepared:
        return ""

    out: list[str] = []
    for start, count in _blocks(len(prepared), size):
        stream: list[int] = []
        for char in prepared[start : start + count]:
            row, column = board.coordinates(char)
            stream.append(row)
            stream.append(column)
        # The first half of the stream is the plaintext rows, the second half
        # its columns -- undoing the "read the top line, then the bottom line"
        # of encryption.
        rows, columns = stream[:count], stream[count:]
        out.append(
            "".join(board.letter(rows[index], columns[index]) for index in range(count))
        )
    return "".join(out)


# ---------------------------------------------------------------------------
# Attack
# ---------------------------------------------------------------------------


def default_squares(keywords: Sequence[str] = ()) -> list[PolybiusSquare]:
    """The squares :func:`solve` tries: the two 5x5 conventions plus keyed ones.

    A keyword that cannot key a particular alphabet (a Q in a drop-Q square)
    is skipped for that shape only, not rejected outright.
    """
    squares: list[PolybiusSquare] = [
        PolybiusSquare.standard(),
        PolybiusSquare.without_q(),
    ]
    for word in keywords:
        for builder in (PolybiusSquare.standard, PolybiusSquare.without_q):
            try:
                squares.append(builder(word))
            except ValueError:
                continue
    return squares


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    keywords: Iterable[str] | None = None,
    squares: Sequence[PolybiusSquare] | None = None,
    max_period: int = DEFAULT_MAX_PERIOD,
    include_whole_message: bool = True,
    time_budget: float | None = None,
    **options: object,
) -> CandidateSet:
    """Try every period and every candidate square, and rank the decryptions.

    Options
    -------
    keywords:
        Keywords for keyed squares. Without them only the two standard 5x5
        arrangements are tried, which will not break a keyed message -- and
        the scores will say so.
    squares:
        Explicit squares to try instead of the defaults.
    max_period:
        Highest block length searched (default 15). Must be at least 1.
    include_whole_message:
        Also try fractionating the message as a single block (default on).
    time_budget:
        Seconds. The search stops cleanly when exceeded and every candidate
        records ``time_budget_hit``.

    ``top`` limits the returned set; pass ``top=0`` for everything.
    """
    engine = scorer if scorer is not None else default_scorer()
    if not isinstance(max_period, int) or isinstance(max_period, bool):
        raise ValueError(f"max_period must be an integer, got {max_period!r}")
    if max_period < 1:
        raise ValueError(
            f"max_period must be at least 1, got {max_period}; period 1 is the "
            "identity and is always worth testing"
        )
    ignored = ", ".join(sorted(str(name) for name in options)) if options else ""

    normalized = normalize(source) if isinstance(source, str) else source
    letters = normalized.letters
    candidates = CandidateSet()
    if not letters:
        return candidates

    words: list[str] = []
    for word in keywords or ():
        text = str(word)
        # Validate up front: a keyword with no usable symbol is an operator
        # error, not something to skip quietly.
        clean_keyword(text, LETTERS_AND_DIGITS)
        words.append(text)

    boards = list(squares) if squares is not None else default_squares(words)
    if not boards:
        raise ValueError("no squares to try: pass squares= or keywords=")

    periods = list(range(1, max_period + 1))
    if include_whole_message:
        periods.append(0)

    started = time.monotonic()
    budget_hit = False
    impossible: list[str] = []
    tested: list[int] = []

    for board in boards:
        # A square that cannot even represent the ciphertext cannot be the
        # square that produced it -- a drop-Q square never emits a Q. Ruling
        # it out is real evidence, so it is recorded rather than skipped.
        unusable = sorted(set(letters) - set(board.symbols) - set(board.merges))
        if unusable:
            impossible.append(
                f"{board.name} (cannot hold {', '.join(unusable)})"
            )
            continue
        for period in periods:
            # Checked only once some work is done, so that even an exhausted
            # budget returns one honest, flagged answer rather than nothing.
            if (
                tested
                and time_budget is not None
                and time.monotonic() - started > time_budget
            ):
                budget_hit = True
                break
            tested.append(period)
            plaintext = decrypt(letters, board, period)
            if not plaintext:
                continue
            label = "whole message" if period == 0 else str(period)
            diagnostics: dict[str, object] = {
                "period": label,
                "square": board.name,
                "grid": board.symbols,
                "blocks": len(_blocks(len(plaintext), period)),
            }
            if period == 1:
                diagnostics["note"] = (
                    "period 1 is the identity: this candidate is the input "
                    "unchanged, not a decryption"
                )
            if ignored:
                diagnostics["ignored_options"] = ignored
            annotate(diagnostics, plaintext, engine)
            candidates.add(
                Candidate(
                    method="Bifid",
                    key=f"period={label} square={board.name}",
                    score=engine.score(plaintext),
                    plaintext=plaintext,
                    diagnostics=diagnostics,
                    # Bifid preserves length, so the original layout can be
                    # reused for reading -- but only as layout: the letter in
                    # position i did not come from position i.
                    display=(
                        normalized.relayout(plaintext)
                        if len(plaintext) == len(letters)
                        else None
                    ),
                )
            )
        if budget_hit:
            break

    unique_periods = sorted(set(tested))
    summary = ", ".join(
        "whole message" if value == 0 else str(value) for value in unique_periods
    )
    for candidate in candidates:
        candidate.diagnostics["periods_tested"] = summary
        candidate.diagnostics["squares_tested"] = len(boards) - len(impossible)
        if impossible:
            candidate.diagnostics["squares_ruled_out"] = "; ".join(impossible)
        if budget_hit:
            candidate.diagnostics["time_budget_hit"] = True

    if top <= 0:
        return candidates
    return CandidateSet(candidates.top(top))
