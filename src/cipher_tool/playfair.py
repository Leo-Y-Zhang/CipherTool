"""Playfair: the digraphic substitution cipher, and a hill-climbing attack.

The cipher
----------
Playfair encrypts *pairs* of letters using a keyed 5x5 square. Twenty-five of
the twenty-six letters are written into the square, keyword first with repeats
removed and then the rest of the alphabet in order; the twenty-sixth letter is
left out, classically by merging I and J (some sources drop Q instead).

The plaintext is cut into digraphs, and each digraph is replaced by looking at
the rectangle its two letters make in the square:

* same ROW      -- take the letter to the right of each, wrapping round;
* same COLUMN   -- take the letter below each, wrapping round;
* otherwise     -- each letter is replaced by the one in its own row and the
                   other letter's column (the two opposite corners of the
                   rectangle the pair defines).

Doubled letters break the rules: both letters of ``LL`` are the same cell, so
"same row" and "same column" are both true at once and the two rules give
different answers. The cipher is simply undefined there, so a filler is pushed
between the pair, and an odd final letter is padded with the same filler.

Why it is stronger than a monoalphabetic cipher
-----------------------------------------------
The unit of encryption is the digraph, so the effective alphabet has 26 x 26 =
676 symbols rather than 26. Single-letter frequency analysis is therefore
useless: E is not enciphered as one fixed letter, it is enciphered differently
depending on what stands next to it. The index of coincidence of Playfair
ciphertext falls between English and random, because the flat-ish digraph
distribution washes out the single-letter peaks.

What Playfair cannot hide
-------------------------
* The ciphertext always has an EVEN number of letters.
* The omitted letter (classically J) never appears. This cuts both ways: a J
  in the ciphertext is proof that the square did *not* merge I/J, which is
  usable evidence rather than bad input, so :func:`solve` reads it and moves
  to the other standard omission instead of refusing to start.
* No digraph is ever a doubled letter: the rules always send two distinct
  letters to two distinct letters, and the preparation step guarantees no pair
  starts out doubled. So ``...LL...`` at an even offset is proof that a text is
  not Playfair ciphertext (or that the transcription slipped by one letter).
* No letter ever encrypts to itself: every rule moves to a different cell of
  the row, the column or the rectangle.
* Reversing a digraph reverses its ciphertext. If ``AB`` enciphers to ``XY``
  then ``BA`` enciphers to ``YX``, under all three rules.

:func:`validate_ciphertext` reports all of these, so the operator sees a clear
explanation instead of a traceback.

The attack
----------
There are 25! = 15,511,210,043,330,985,984,000,000 squares, so nothing is
brute-forced here. The interesting thing about this particular search space,
measured on this machine with this toolkit's own scorer, is that it is *not*
the gentle hill the phrase "hill climbing" suggests:

* Steepest ascent from a random square, trying all 300 letter swaps and taking
  every improvement, runs out of improvements after two or three sweeps at
  around -2.1 per letter. True English scores -0.84 per letter here and a
  random square scores -2.67, so the climb stops most of the way down.
* The gradient exists only near the answer, but there it is strong: take the
  true square, apply ten random transpositions to it, and the same steepest
  ascent puts it back *exactly*, every time.

So the search is not a climb, it is a hunt for the basin -- and once inside,
the climb is trivial. That is what simulated annealing is for. A move that
costs ``delta`` (a difference of log10 probabilities, so negative) is accepted
with probability

    P(accept) = exp(delta / T)

and ``T`` cools linearly to zero across the run: at high T the walk is nearly
random, at T = 0 it is strict hill climbing, and in between it can cross
valleys shallower than T. The temperature has to be on the scale of the score
differences it is judging, which grow with the length of the message, so it is
quoted per digraph and multiplied up.

Two further design choices, both measured rather than assumed:

* The annealing is scored with a *digraph* model, not the package's order-3
  model, and the polish afterwards uses the order-3 model. See
  :func:`_digraph_model` for the measurements behind that.
* The move set includes whole-row and whole-column swaps and mirrors, because
  a square that is correct "up to a row swap" is five transpositions from the
  truth with every intermediate square far worse. Single-letter swaps alone
  cannot cross that valley.

HONESTY ABOUT THIS ATTACK
-------------------------
Playfair hill climbing needs a lot of ciphertext. Below roughly 200 letters it
usually fails outright, and the top candidate it returns will be confident
nonsense. Around 300 letters it works often; by 500 it is reliable given a few
restarts -- one restart of the default length recovered a 520-letter message
about half the time, which is why the default is to run several. Every
candidate records the ciphertext length and an outlook string in its
diagnostics so the operator can see whether the attack ever stood a chance.

One further caution built into the diagnostics: cyclically rotating all rows,
or all columns, of a Playfair square leaves the cipher completely unchanged
(the rules only ever use "same row", "same column" and relative offsets modulo
five). Every square therefore has 25 equivalent forms, and a recovered square
is expected to be a rotation of the original rather than the original itself.
:func:`canonical_square` rotates a square into a standard position so that two
squares can be compared honestly.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Sequence

from .candidates import Candidate, CandidateSet
from .normalize import (
    ALPHABET,
    NormalizedText,
    clean_key,
    from_numbers,
    letters_only,
    normalize,
)
from .scoring import EnglishScorer, annotate, default_scorer, load_corpus_text

__all__ = [
    "PlayfairSquare",
    "playfair_square",
    "square_from_letters",
    "plain_square",
    "as_square",
    "rotate_square",
    "canonical_square",
    "prepare_digraphs",
    "prepare_text",
    "check_filler",
    "validate_ciphertext",
    "encrypt",
    "decrypt",
    "solve",
]

#: Side of the square, and how many letters it therefore holds.
SQUARE_SIZE = 5
SQUARE_LETTERS = SQUARE_SIZE * SQUARE_SIZE

#: Classical defaults: merge I/J, split doubles with X, and use Q when the
#: doubled letter is itself an X (the classic implementation bug is to use X
#: there as well, which produces another doubled pair and never terminates).
DEFAULT_OMITTED = "J"
DEFAULT_FILLER = "X"
DEFAULT_ALTERNATIVE_FILLER = "Q"

#: The only two omissions in general use, in the order a search should try
#: them. Nothing else is guessed at: a square omitting some other letter is a
#: real variant, but choosing one because the ciphertext happens to lack that
#: letter would be inventing evidence rather than reading it.
STANDARD_OMISSIONS = ("J", "Q")

#: When the omitted letter is I or J the two are *merged*: the omitted one is
#: rewritten as its partner. Any other omitted letter (Q is the usual choice)
#: is simply deleted from the plaintext, because there is nothing sensible to
#: rewrite it as. Both behaviours are documented on the functions that use
#: them; neither happens silently.
_FOLD_TARGETS = {"I": "J", "J": "I"}

#: Ciphertext shorter than this is not worth hill-climbing. Stated in the
#: docstring above and repeated in every candidate's diagnostics.
RELIABLE_LENGTH = 200

#: Search defaults, measured on 520-letter samples from the project corpus
#: (see ``_digraph_model`` and ``_search`` for what was measured and why).
#: One restart of 150,000 moves recovered the square on four seeds in eight
#: and takes about seven seconds, so five restarts fail about one run in
#: thirty. The search stops as soon as a restart produces something the
#: toolkit calls 'strong', so a solvable message usually costs far less.
DEFAULT_RESTARTS = 5
DEFAULT_ITERATIONS = 150000

#: The annealing temperature has to be on the scale of the score differences
#: it is comparing, and those grow with the length of the message, so the
#: default is quoted per digraph and multiplied up. Measured at 520 letters:
#: 0.030 solved 4 seeds in 6, 0.023 solved 4, 0.015 solved 2, and 0.038 solved
#: none -- the peak is broad on the low side and falls off a cliff above it.
DEFAULT_TEMPERATURE_PER_DIGRAPH = 0.030


# ---------------------------------------------------------------------------
# The square
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayfairSquare:
    """A keyed 5x5 Playfair square.

    Attributes
    ----------
    letters:
        The twenty-five square letters in row-major order, so ``letters[5r+c]``
        is the letter at row *r*, column *c* (both counted from zero).
    omitted:
        The single letter of the alphabet that is not in the square.
    folded_onto:
        The letter that :attr:`omitted` is rewritten as when preparing
        plaintext (``"I"`` for the classic I/J merge), or ``None`` when the
        omitted letter is deleted instead.
    """

    letters: str
    omitted: str
    folded_onto: str | None = None

    def __post_init__(self) -> None:
        if len(self.letters) != SQUARE_LETTERS:
            raise ValueError(
                f"A Playfair square needs exactly {SQUARE_LETTERS} letters, "
                f"got {len(self.letters)}: {self.letters!r}."
            )
        if any(not ("A" <= ch <= "Z") for ch in self.letters):
            raise ValueError(
                f"A Playfair square may only contain A-Z, got {self.letters!r}."
            )
        if len(set(self.letters)) != SQUARE_LETTERS:
            duplicates = sorted(
                {ch for ch in self.letters if self.letters.count(ch) > 1}
            )
            raise ValueError(
                "A Playfair square may not repeat a letter; these appear more "
                f"than once: {' '.join(duplicates)}."
            )
        if len(self.omitted) != 1 or not ("A" <= self.omitted <= "Z"):
            raise ValueError(
                f"The omitted letter must be a single letter A-Z, got "
                f"{self.omitted!r}."
            )
        if self.omitted in self.letters:
            raise ValueError(
                f"The omitted letter {self.omitted!r} must not appear in the "
                "square itself."
            )

    # -- geometry ----------------------------------------------------------

    def contains(self, letter: str) -> bool:
        """True if *letter* has a place in this square."""
        return len(letter) == 1 and letter.upper() in self.letters

    def position(self, letter: str) -> tuple[int, int]:
        """Return the ``(row, column)`` of *letter*, both counted from zero."""
        index = self.letters.find(letter.upper())
        if index < 0:
            raise ValueError(
                f"The letter {letter!r} is not in this square, which omits "
                f"{self.omitted!r}. Fold the text first (see fold()), or build "
                f"the square with a different omitted letter -- in Python, the "
                f"omit= library argument of playfair_square(); there is no "
                f"command-line flag for it."
            )
        return divmod(index, SQUARE_SIZE)

    def at(self, row: int, column: int) -> str:
        """The letter at ``(row, column)``. Indices are taken modulo five."""
        return self.letters[(row % SQUARE_SIZE) * SQUARE_SIZE + column % SQUARE_SIZE]

    def rows(self) -> tuple[str, ...]:
        """The five rows of the square, top to bottom."""
        return tuple(
            self.letters[index : index + SQUARE_SIZE]
            for index in range(0, SQUARE_LETTERS, SQUARE_SIZE)
        )

    # -- text preparation --------------------------------------------------

    def fold(self, text: str) -> str:
        """Rewrite *text* so every letter has a place in this square.

        With the classic I/J merge, J becomes I. With any other omitted letter
        (Q, usually) the letter is deleted, because there is no natural letter
        to rewrite it as; this is lossy and the docstrings of the callers say
        so. Non-letters are dropped, as everywhere else in the toolkit.
        """
        letters = letters_only(text)
        if self.folded_onto is None:
            return letters.replace(self.omitted, "")
        return letters.replace(self.omitted, self.folded_onto)

    # -- display -----------------------------------------------------------

    def grid(self) -> str:
        """The square as five lines of spaced letters, for human reading."""
        return "\n".join(" ".join(row) for row in self.rows())

    def compact(self) -> str:
        """The square as five space-separated groups, for one-line reports."""
        return " ".join(self.rows())

    def __str__(self) -> str:
        return self.grid()


def _check_omitted(omit: str) -> str:
    """Validate the ``omit=`` argument shared by most functions here."""
    cleaned = clean_key(omit)
    if len(cleaned) != 1:
        raise ValueError(
            f"The omitted letter must be exactly one letter A-Z, so that the "
            f"other 25 fill the square, got {omit!r}. In Python, pass "
            f"omit='J' for the usual I/J merge or omit='Q' to drop Q instead; "
            f"omit= is a library argument, not a command-line flag."
        )
    return cleaned


def _fold_target(omitted: str) -> str | None:
    """Where the omitted letter goes when preparing plaintext."""
    return _FOLD_TARGETS.get(omitted)


def square_from_letters(
    letters: str, *, omit: str = DEFAULT_OMITTED
) -> PlayfairSquare:
    """Build a square from all twenty-five of its letters, in reading order.

    This is the entry point for a square that is not described by a keyword --
    the hill climber uses it, and so does anyone transcribing a square from a
    printed grid.
    """
    omitted = _check_omitted(omit)
    cleaned = letters_only(letters)
    return PlayfairSquare(
        letters=cleaned, omitted=omitted, folded_onto=_fold_target(omitted)
    )


def playfair_square(keyword: str, *, omit: str = DEFAULT_OMITTED) -> PlayfairSquare:
    """Build the square keyed by *keyword*.

    The keyword's letters are written in first, each one only the first time it
    occurs, and the remaining letters of the alphabet follow in order. The
    omitted letter is folded before de-duplication, so a keyword of ``JAM``
    with the classic I/J merge starts the square with ``I A M``.

    Raises ``ValueError`` if the keyword contains no usable letters at all.
    """
    omitted = _check_omitted(omit)
    cleaned = clean_key(keyword)
    if not cleaned:
        raise ValueError(
            f"The Playfair keyword contains no letters A-Z (got {keyword!r}). "
            "Give a keyword such as 'MONARCHY', or call plain_square() for the "
            "unkeyed alphabet square, or square_from_letters() to type a "
            "square out in full."
        )

    target = _fold_target(omitted)
    folded = (
        cleaned.replace(omitted, "")
        if target is None
        else cleaned.replace(omitted, target)
    )
    if not folded:
        raise ValueError(
            f"Every letter of the keyword {keyword!r} is the omitted letter "
            f"{omitted!r}, so nothing is left to key the square with. Choose "
            "another keyword or another omit= letter."
        )

    order: list[str] = []
    used: set[str] = set()
    for char in folded:
        if char not in used:
            used.add(char)
            order.append(char)
    for char in ALPHABET:
        if char == omitted or char in used:
            continue
        used.add(char)
        order.append(char)

    return PlayfairSquare(
        letters="".join(order), omitted=omitted, folded_onto=target
    )


def plain_square(*, omit: str = DEFAULT_OMITTED) -> PlayfairSquare:
    """The unkeyed square: the alphabet in order, minus the omitted letter."""
    omitted = _check_omitted(omit)
    return PlayfairSquare(
        letters="".join(ch for ch in ALPHABET if ch != omitted),
        omitted=omitted,
        folded_onto=_fold_target(omitted),
    )


def as_square(
    key: str | PlayfairSquare, *, omit: str = DEFAULT_OMITTED
) -> PlayfairSquare:
    """Accept either a keyword or an already-built square.

    A :class:`PlayfairSquare` is returned unchanged, including its own omitted
    letter -- the ``omit=`` argument is then ignored, because the square
    already knows which letter it left out.
    """
    if isinstance(key, PlayfairSquare):
        return key
    if isinstance(key, str):
        return playfair_square(key, omit=omit)
    raise ValueError(
        "A Playfair key must be a keyword string or a PlayfairSquare, got "
        f"{type(key).__name__}."
    )


# ---------------------------------------------------------------------------
# Equivalent squares
# ---------------------------------------------------------------------------


def rotate_square(
    square: PlayfairSquare, *, rows: int = 0, columns: int = 0
) -> PlayfairSquare:
    """Cyclically rotate the square by *rows* rows and *columns* columns.

    The result enciphers *identically* to the input. The row rule ("move right,
    wrapping") and the column rule ("move down, wrapping") only use offsets
    modulo five, and the rectangle rule only asks which row and which column a
    letter is in, so rotating every row or every column by a constant changes
    no answer anywhere. This is why a solved square is normally a rotation of
    the square the sender actually used.
    """
    grid = square.rows()
    rotated_rows = [grid[(index + rows) % SQUARE_SIZE] for index in range(SQUARE_SIZE)]
    rotated = [
        "".join(row[(index + columns) % SQUARE_SIZE] for index in range(SQUARE_SIZE))
        for row in rotated_rows
    ]
    return PlayfairSquare(
        letters="".join(rotated),
        omitted=square.omitted,
        folded_onto=square.folded_onto,
    )


def canonical_square(
    square: PlayfairSquare, *, anchor: str = "A"
) -> PlayfairSquare:
    """Rotate *square* into the equivalent form with *anchor* at the top left.

    Because the twenty-five cyclic rotations of a square all encipher the same
    way, comparing two squares letter by letter is meaningless until they have
    been put in a standard position. This does that, and is the honest way to
    ask "did the solver find the right key?".
    """
    letter = clean_key(anchor)
    if len(letter) != 1:
        raise ValueError(
            f"The anchor must be a single letter A-Z, got {anchor!r}."
        )
    if not square.contains(letter):
        raise ValueError(
            f"The anchor {letter!r} is not in this square, which omits "
            f"{square.omitted!r}. Choose a letter the square contains."
        )
    row, column = square.position(letter)
    return rotate_square(square, rows=row, columns=column)


# ---------------------------------------------------------------------------
# Digraph preparation
# ---------------------------------------------------------------------------


def check_filler(
    value: str,
    *,
    square: PlayfairSquare | None = None,
    omit: str = DEFAULT_OMITTED,
    name: str = "filler",
) -> str:
    """Validate a filler letter and return it cleaned, or raise ``ValueError``.

    Public because every layer that *accepts* a filler has to reject a bad one
    at the point it is accepted. A filler only ever affects encryption -- it is
    inserted between doubled letters and used to pad an odd tail -- so a layer
    that takes a filler and then calls :func:`decrypt` or :func:`solve` is
    accepting an argument it cannot use. Validating here at least turns a
    silently discarded ``--filler XY`` into an error; whether such a layer
    should offer the option at all is its own decision.

    *square* is the square the filler will actually be enciphered in. Pass it
    whenever it is known: a filler has to be a letter the square contains, and
    that cannot be checked without it. Without one, the check falls back to the
    plain square for *omit*, which catches everything except a filler that is
    legal in the default alphabet but missing from some other square.

    *name* names the argument in the error message ("filler" or "alternative
    filler"), because being told which of the two is wrong is the whole point
    of reading the message.
    """
    grid = square if square is not None else plain_square(omit=omit)
    argument = name.split()[0]
    cleaned = clean_key(value)
    if len(cleaned) != 1:
        raise ValueError(
            f"The {name} must be a single letter A-Z, got {value!r}. In "
            f"Python, pass {argument}='X'; {argument}= is a library argument "
            f"of encrypt() and prepare_text(), not a command-line flag."
        )
    if not grid.contains(cleaned):
        replacement = next(ch for ch in "XZQKV" + ALPHABET if grid.contains(ch))
        raise ValueError(
            f"The {name} {cleaned!r} is not in the square, which omits "
            f"{grid.omitted!r}, so it could never be enciphered. Choose "
            f"another letter, for example {replacement!r}."
        )
    return cleaned


def prepare_digraphs(
    text: str,
    *,
    square: PlayfairSquare | None = None,
    filler: str = DEFAULT_FILLER,
    alternative: str = DEFAULT_ALTERNATIVE_FILLER,
    omit: str = DEFAULT_OMITTED,
) -> list[str]:
    """Cut *text* into the letter pairs Playfair will actually encipher.

    Three things happen, in this order:

    1. The text is reduced to letters and folded onto the square's alphabet
       (J becomes I under the classic merge; a dropped letter such as Q is
       deleted, which loses information -- see :meth:`PlayfairSquare.fold`).
    2. A doubled pair is split by pushing *filler* between the two letters, so
       ``LL`` becomes ``LX`` and the second ``L`` starts the next pair. When
       the doubled letter *is* the filler, *alternative* is used instead:
       ``XX`` becomes ``XQ``, not ``XX`` again. Getting that wrong is the
       classic Playfair implementation bug -- it either loops forever or emits
       a doubled pair the cipher cannot encipher.
    3. If a single letter is left over at the end it is padded with the filler,
       again swapping to *alternative* when the leftover letter is the filler.

    Raises ``ValueError`` if either filler is not a single letter of the
    square, or if the two fillers are the same letter (in which case rule 2
    would have nothing to fall back on).
    """
    grid = square if square is not None else plain_square(omit=omit)
    first_filler = check_filler(filler, square=grid)
    second_filler = check_filler(
        alternative, square=grid, name="alternative filler"
    )
    if first_filler == second_filler:
        raise ValueError(
            f"The filler and the alternative filler are both {first_filler!r}. "
            "They must differ, because a doubled filler is split using the "
            "alternative. In Python, pass filler='X' and alternative='Q'; both "
            "are library arguments, not command-line flags."
        )

    letters = grid.fold(text)
    pairs: list[str] = []
    index = 0
    total = len(letters)
    while index < total:
        head = letters[index]
        tail = letters[index + 1] if index + 1 < total else ""
        if tail == "" or tail == head:
            # Either a doubled pair or a lone final letter: pad, and consume
            # only the one letter so the partner starts the next pair.
            padding = second_filler if head == first_filler else first_filler
            pairs.append(head + padding)
            index += 1
        else:
            pairs.append(head + tail)
            index += 2
    return pairs


def prepare_text(
    text: str,
    *,
    square: PlayfairSquare | None = None,
    filler: str = DEFAULT_FILLER,
    alternative: str = DEFAULT_ALTERNATIVE_FILLER,
    omit: str = DEFAULT_OMITTED,
) -> str:
    """The prepared plaintext as one string.

    This, not the original text, is what a correct decryption returns: the
    fillers and the I/J merge are not reversible, so ``decrypt(encrypt(t))``
    equals ``prepare_text(t)``.
    """
    return "".join(
        prepare_digraphs(
            text,
            square=square,
            filler=filler,
            alternative=alternative,
            omit=omit,
        )
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _usable_omissions(letters: str, current: str) -> list[str]:
    """Standard omissions, other than *current*, that *letters* does not use.

    A square that omits X can never emit an X, so the ciphertext itself rules
    omissions out. This returns the ones it has not ruled out, in the order of
    :data:`STANDARD_OMISSIONS`.
    """
    seen = set(letters)
    return [
        letter
        for letter in STANDARD_OMISSIONS
        if letter != current and letter not in seen
    ]


def _fatal_problems(letters: str, square: PlayfairSquare) -> list[str]:
    """Problems that make decryption impossible rather than merely suspicious."""
    problems: list[str] = []

    if len(letters) % 2 == 1:
        problems.append(
            f"The ciphertext has an odd number of letters ({len(letters)}). "
            "Playfair enciphers pairs, so genuine Playfair ciphertext always "
            "has an even count: check the transcription for a dropped or an "
            "extra letter."
        )

    stray = sorted({ch for ch in letters if not square.contains(ch)})
    if stray:
        where = [
            str(index)
            for index, ch in enumerate(letters)
            if ch in stray
        ]
        shown = ", ".join(where[:8]) + (" ..." if len(where) > 8 else "")
        merge = (
            f"merges {square.folded_onto}/{square.omitted}"
            if square.folded_onto
            else f"drops {square.omitted}"
        )
        # Only suggest an omission the ciphertext could actually survive.
        # Advising omit='Q' for a text that also contains a Q sends the
        # operator round a loop and reads as though the tool has an answer.
        escape = _usable_omissions(letters, square.omitted)
        if escape:
            advice = (
                f"Either the square omits a different letter -- in Python, "
                f"omit={escape[0]!r}, a library argument rather than a "
                f"command-line flag, and solve() tries that automatically when "
                f"no key is supplied -- or the ciphertext is not Playfair."
            )
        else:
            ruled_out = " or ".join(repr(ch) for ch in STANDARD_OMISSIONS)
            advice = (
                f"No square omitting {ruled_out} can produce this text either, "
                f"because it uses all of those letters, so this is not "
                f"Playfair ciphertext under any standard square: check the "
                f"transcription, or try a different cipher."
            )
        problems.append(
            f"The ciphertext contains {' '.join(stray)}, which this square "
            f"cannot produce because it {merge} (positions {shown}). {advice}"
        )

    return problems


def validate_ciphertext(text: str, square: PlayfairSquare) -> list[str]:
    """Report every way *text* fails to look like Playfair ciphertext.

    Returns a list of complete sentences; an empty list means the text is well
    formed as far as this cipher's structure can tell. Written so a CLI can
    print the problems rather than crash, and so the operator is told what to
    do about each one.

    The first two problems below make decryption impossible and are what
    :func:`decrypt` refuses on. The third does not stop decryption, but a
    genuine Playfair ciphertext cannot contain it, so it is strong evidence
    that the text is either mis-transcribed or a different cipher entirely.
    """
    letters = letters_only(text)
    if not letters:
        return [
            "The input contains no letters A-Z, so there is nothing to "
            "decrypt. Check the file or the paste."
        ]

    problems = _fatal_problems(letters, square)

    # Doubled digraphs. Both Playfair rules send two distinct letters to two
    # distinct letters, and preparation guarantees no pair was doubled to start
    # with, so a doubled pair at an even offset cannot come out of the cipher.
    doubled = [
        index
        for index in range(0, len(letters) - 1, 2)
        if letters[index] == letters[index + 1]
    ]
    if doubled:
        shown = ", ".join(
            f"{letters[index]}{letters[index + 1]} at letter {index}"
            for index in doubled[:5]
        )
        problems.append(
            f"{len(doubled)} digraph(s) are a doubled letter ({shown}"
            + (" ..." if len(doubled) > 5 else "")
            + "). Playfair can never output a doubled pair, so the text is "
            "either mis-transcribed, offset by one letter, or not Playfair."
        )

    return problems


# ---------------------------------------------------------------------------
# Encryption and decryption
# ---------------------------------------------------------------------------


def _transform_pair(
    square: PlayfairSquare, first: str, second: str, step: int
) -> tuple[str, str]:
    """Apply the three Playfair rules to one pair.

    *step* is ``+1`` to encipher and ``-1`` to decipher: enciphering moves
    right along a row and down a column, so deciphering moves left and up. The
    rectangle rule takes no step, because swapping the two columns back is the
    same operation as swapping them in the first place -- it is its own
    inverse, which is why the same branch serves both directions.
    """
    row_first, column_first = square.position(first)
    row_second, column_second = square.position(second)

    if row_first == row_second:
        return (
            square.at(row_first, column_first + step),
            square.at(row_second, column_second + step),
        )
    if column_first == column_second:
        return (
            square.at(row_first + step, column_first),
            square.at(row_second + step, column_second),
        )
    return (
        square.at(row_first, column_second),
        square.at(row_second, column_first),
    )


def encrypt(
    text: str,
    key: str | PlayfairSquare,
    *,
    filler: str = DEFAULT_FILLER,
    alternative: str = DEFAULT_ALTERNATIVE_FILLER,
    omit: str = DEFAULT_OMITTED,
) -> str:
    """Encrypt. Operates on letters only; returns letters only, uppercase.

    The output is always an even number of letters, and is longer than the
    input's letter count whenever a filler had to be inserted. ``key`` is
    either a keyword or a :class:`PlayfairSquare`.
    """
    square = as_square(key, omit=omit)
    pairs = prepare_digraphs(
        text, square=square, filler=filler, alternative=alternative
    )
    out: list[str] = []
    for pair in pairs:
        head, tail = _transform_pair(square, pair[0], pair[1], +1)
        out.append(head)
        out.append(tail)
    return "".join(out)


def decrypt(
    text: str,
    key: str | PlayfairSquare,
    *,
    omit: str = DEFAULT_OMITTED,
) -> str:
    """Exact inverse of :func:`encrypt` for the same key.

    "Exact inverse" means it returns the *prepared* plaintext: the fillers
    inserted during encryption are still there and a merged J is still an I,
    because neither is recoverable. Compare against :func:`prepare_text`, not
    against the original text.

    Raises ``ValueError`` for ciphertext of odd length or containing the
    omitted letter, since neither can have come from this square.
    """
    square = as_square(key, omit=omit)
    letters = letters_only(text)
    if not letters:
        return ""

    problems = _fatal_problems(letters, square)
    if problems:
        raise ValueError(" ".join(problems))

    out: list[str] = []
    for index in range(0, len(letters), 2):
        head, tail = _transform_pair(
            square, letters[index], letters[index + 1], -1
        )
        out.append(head)
        out.append(tail)
    return "".join(out)


# ---------------------------------------------------------------------------
# The search: fitness functions
# ---------------------------------------------------------------------------


def _encode_pairs(letters: str) -> list[tuple[int, int]]:
    """Turn ciphertext into (first, second) pairs of alphabet indices 0..25."""
    values = [ord(ch) - 65 for ch in letters]
    return [
        (values[index], values[index + 1])
        for index in range(0, len(values) - 1, 2)
    ]


def _compress(pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Collapse the ciphertext to its distinct digraphs with multiplicities.

    This is the trick that makes the search affordable. Playfair maps digraphs
    to digraphs, so two occurrences of the same ciphertext digraph always
    decrypt to the same plaintext digraph, whatever the square. A digraph-level
    score therefore only needs one evaluation per DISTINCT digraph, weighted by
    how often it occurs: the cost of scoring a square is bounded by 600 (the
    number of digraphs Playfair can produce) no matter how long the message is.
    """
    counter = Counter(pairs)
    return [(first, second, count) for (first, second), count in counter.items()]


@lru_cache(maxsize=4)
def _digraph_model(omitted: str) -> tuple[float, ...]:
    """log10 P(xy) for all 676 letter pairs, counted from the local corpus.

    Why a second, much cruder language model when the package already has a
    carefully built order-3 one? Because this is the right fitness *during the
    search*, for two measured reasons (520-letter samples, project corpus,
    identical move set and schedule length, temperature scaled to each score's
    own range):

        fitness      moves     solved     evaluations per second
        ---------    -------   --------   ----------------------
        digraph      200,000   4 of 6     ~35,000
        order-3      200,000   0 of 6     ~5,600

    The speed comes from :func:`_compress`. The better hit rate comes from the
    shape of the landscape: a digraph score changes only where the square
    actually changed, whereas an order-3 score also swings on the three letters
    of context either side, which adds noise to exactly the small differences
    the annealer has to judge.

    The order-3 model still has the last word -- every annealed square is
    polished and every candidate is ranked with it -- so the crude model only
    ever chooses where to look, never what to believe.

    The corpus is folded onto the square's alphabet first, so that with the
    classic merge the model scores I where the plaintext will read I. Counts
    are over overlapping adjacent pairs; the decrypted digraphs sit at even
    offsets only, but the difference is far below the noise this fitness works
    at. Add-one smoothing keeps unseen pairs merely unlikely: an infinite
    penalty would tear holes in the landscape the search has to walk over.
    """
    letters = "".join(
        ch for ch in load_corpus_text().upper() if "A" <= ch <= "Z"
    )
    target = _FOLD_TARGETS.get(omitted)
    letters = (
        letters.replace(omitted, "")
        if target is None
        else letters.replace(omitted, target)
    )

    counts = [0] * 676
    values = [ord(ch) - 65 for ch in letters]
    previous = values[0]
    for value in values[1:]:
        counts[previous * 26 + value] += 1
        previous = value

    total = sum(counts)
    return tuple(
        math.log10((count + 1) / (total + 676)) for count in counts
    )


def _digraph_score(
    compressed: Sequence[tuple[int, int, int]],
    square: list[int],
    table: Sequence[float],
) -> float:
    """Score a square by the digraph statistics of what it decrypts to.

    The hot loop of the whole module. Decryption is inlined rather than
    delegated so that no intermediate plaintext is built: each distinct
    ciphertext digraph is turned into its plaintext digraph and looked up once.
    """
    position = [0] * 26
    for index, value in enumerate(square):
        position[value] = index

    total = 0.0
    for first, second, count in compressed:
        place_first = position[first]
        place_second = position[second]
        row_first, column_first = divmod(place_first, 5)
        row_second, column_second = divmod(place_second, 5)
        if row_first == row_second:
            head = square[row_first * 5 + (column_first - 1) % 5]
            tail = square[row_second * 5 + (column_second - 1) % 5]
        elif column_first == column_second:
            head = square[((row_first - 1) % 5) * 5 + column_first]
            tail = square[((row_second - 1) % 5) * 5 + column_second]
        else:
            head = square[place_first - column_first + column_second]
            tail = square[place_second - column_second + column_first]
        total += count * table[head * 26 + tail]
    return total


def _decrypt_values(
    pairs: Sequence[tuple[int, int]], square: list[int]
) -> list[int]:
    """Decrypt encoded pairs with an encoded square. The hot loop.

    *square* holds the twenty-five alphabet indices in row-major order. The
    inverse position table is rebuilt here rather than passed in: twenty-five
    assignments per call are nothing beside the per-digraph work, and having
    one source of truth removes a whole class of stale-index bug.
    """
    position = [0] * 26
    for index, value in enumerate(square):
        position[value] = index

    plain: list[int] = []
    append = plain.append
    for first, second in pairs:
        place_first = position[first]
        place_second = position[second]
        row_first, column_first = divmod(place_first, 5)
        row_second, column_second = divmod(place_second, 5)
        if row_first == row_second:
            # Same row: step LEFT to decrypt, wrapping.
            append(square[row_first * 5 + (column_first - 1) % 5])
            append(square[row_second * 5 + (column_second - 1) % 5])
        elif column_first == column_second:
            # Same column: step UP to decrypt, wrapping.
            append(square[((row_first - 1) % 5) * 5 + column_first])
            append(square[((row_second - 1) % 5) * 5 + column_second])
        else:
            # Rectangle: swap the columns. Self-inverse, so no direction.
            append(square[place_first - column_first + column_second])
            append(square[place_second - column_second + column_first])
    return plain


# ---------------------------------------------------------------------------
# The search: moves
# ---------------------------------------------------------------------------


def _swap_rows(square: list[int], first: int, second: int) -> list[int]:
    """Exchange two whole rows of the square."""
    trial = list(square)
    head = slice(first * SQUARE_SIZE, (first + 1) * SQUARE_SIZE)
    tail = slice(second * SQUARE_SIZE, (second + 1) * SQUARE_SIZE)
    trial[head], trial[tail] = square[tail], square[head]
    return trial


def _swap_columns(square: list[int], first: int, second: int) -> list[int]:
    """Exchange two whole columns of the square."""
    trial = list(square)
    for row in range(SQUARE_SIZE):
        left = row * SQUARE_SIZE + first
        right = row * SQUARE_SIZE + second
        trial[left], trial[right] = trial[right], trial[left]
    return trial


def _flip_rows(square: list[int]) -> list[int]:
    """Mirror the square top to bottom."""
    return [
        square[row * SQUARE_SIZE + column]
        for row in range(SQUARE_SIZE - 1, -1, -1)
        for column in range(SQUARE_SIZE)
    ]


def _flip_columns(square: list[int]) -> list[int]:
    """Mirror the square left to right."""
    return [
        square[row * SQUARE_SIZE + column]
        for row in range(SQUARE_SIZE)
        for column in range(SQUARE_SIZE - 1, -1, -1)
    ]


def _structural_moves(square: list[int]) -> Iterator[list[int]]:
    """Every whole-row, whole-column and mirror neighbour of *square*.

    Twenty-three squares: ten row swaps, ten column swaps, the two mirrors and
    the reversal. Cheap enough to try exhaustively in the polishing sweep.
    """
    for first in range(SQUARE_SIZE):
        for second in range(first + 1, SQUARE_SIZE):
            yield _swap_rows(square, first, second)
            yield _swap_columns(square, first, second)
    yield _flip_rows(square)
    yield _flip_columns(square)
    yield list(reversed(square))


def _modify(square: list[int], rng: random.Random) -> list[int]:
    """Return a random neighbour of *square*.

    The mix is the classical one. Ninety per cent of moves swap two letters,
    which is the fine-grained move that does the actual climbing. The rest are
    coarse moves -- swapping two rows or two columns, reversing the square,
    mirroring it top-to-bottom or left-to-right. Those exist because a square
    that is correct apart from an exchanged pair of rows is five letter-swaps
    from the truth, and every square in between scores far worse; without a
    move that crosses the valley in one step the search would sit in that local
    optimum until the temperature schedule ran out.

    (Reversing the whole square equals mirroring both ways, so the moves are
    not independent. That is harmless: they are proposal distributions, not a
    basis.)
    """
    roll = rng.random()

    if roll < 0.90:
        first = rng.randrange(SQUARE_LETTERS)
        second = rng.randrange(SQUARE_LETTERS - 1)
        if second >= first:
            second += 1  # draw uniformly from the 24 other cells
        trial = list(square)
        trial[first], trial[second] = trial[second], trial[first]
        return trial

    if roll < 0.96:
        first = rng.randrange(SQUARE_SIZE)
        second = rng.randrange(SQUARE_SIZE - 1)
        if second >= first:
            second += 1
        if roll < 0.93:
            return _swap_rows(square, first, second)
        return _swap_columns(square, first, second)

    if roll < 0.98:
        return _flip_rows(square)
    if roll < 0.99:
        return _flip_columns(square)
    return list(reversed(square))


# ---------------------------------------------------------------------------
# The search: annealing, then polishing
# ---------------------------------------------------------------------------


@dataclass
class _SearchResult:
    """One restart's best square and the bookkeeping that goes with it."""

    letters: str
    score: float
    digraph_score: float
    moves: int
    budget_hit: bool


def _polish(
    pairs: Sequence[tuple[int, int]],
    square: list[int],
    scorer: EnglishScorer,
) -> tuple[list[int], float, int]:
    """Steepest ascent from an annealed square, under the order-3 model.

    The annealer works with digraph statistics, which are blunt: they cannot
    tell two squares apart when the difference only shows up as an implausible
    four-letter run. So the last stretch is climbed under the model that will
    actually rank the candidate, trying every one of the 300 letter swaps and
    all 23 structural moves and taking any that improves, until nothing does.

    This costs about 1,300 order-3 evaluations per sweep, which is nothing
    beside the annealing that precedes it, and it guarantees the reported
    square is at least a local optimum of the model used to judge it.
    """
    score_values = scorer.score_values
    current = list(square)
    best = score_values(_decrypt_values(pairs, current))
    evaluated = 1

    improving = True
    while improving:
        improving = False
        for first in range(SQUARE_LETTERS):
            for second in range(first + 1, SQUARE_LETTERS):
                trial = list(current)
                trial[first], trial[second] = trial[second], trial[first]
                value = score_values(_decrypt_values(pairs, trial))
                evaluated += 1
                if value > best:
                    best, current, improving = value, trial, True
        for trial in _structural_moves(current):
            value = score_values(_decrypt_values(pairs, trial))
            evaluated += 1
            if value > best:
                best, current, improving = value, trial, True

    return current, best, evaluated


def _search(
    pairs: Sequence[tuple[int, int]],
    alphabet_values: list[int],
    scorer: EnglishScorer,
    *,
    omitted: str,
    restarts: int,
    iterations: int,
    temperature: float,
    rng: random.Random,
    deadline: float | None,
) -> Iterator[_SearchResult]:
    """Yield the best square of each of *restarts* independent annealing runs.

    A generator rather than a list so that the caller can stop as soon as one
    restart has produced something the project's own confidence thresholds
    call English. There is no point spending another twenty seconds proving
    what is already legible, and stopping is recorded in the diagnostics so
    the operator knows how many restarts actually ran.

    Acceptance is the Metropolis rule described in the module docstring:
    improvements are always taken, and a move that costs ``delta`` log10 units
    is taken with probability ``exp(delta / T)``. ``T`` falls linearly to zero
    across each run, so the walk starts nearly random and ends as a strict hill
    climb. The best square ever *visited* is kept separately from the current
    one, because annealing deliberately wanders away from good squares and a
    run often ends below its own high-water mark.

    The clock is only consulted every 1024 moves. Calling ``time.monotonic()``
    on every move would be a measurable fraction of the move itself, and the
    resulting overshoot is a few milliseconds.
    """
    table = _digraph_model(omitted)
    compressed = _compress(pairs)
    scorer.table()  # build the order-3 table once, before anything is timed
    exp = math.exp

    evaluated = 0
    budget_hit = False

    for restart in range(restarts):
        # The first restart always runs, even if the budget has already gone:
        # returning nothing at all would tell the operator less than returning
        # one bad square clearly labelled as cut short.
        if restart and deadline is not None and time.monotonic() >= deadline:
            budget_hit = True
            break

        square = list(alphabet_values)
        rng.shuffle(square)
        current = _digraph_score(compressed, square, table)
        evaluated += 1
        best_square = list(square)
        best_score = current

        for move in range(iterations):
            if deadline is not None and (move & 1023) == 0:
                if time.monotonic() >= deadline:
                    budget_hit = True
                    break
            temp = temperature * (1.0 - move / iterations)
            trial = _modify(square, rng)
            value = _digraph_score(compressed, trial, table)
            evaluated += 1
            delta = value - current
            if delta > 0.0 or (temp > 0.0 and rng.random() < exp(delta / temp)):
                square = trial
                current = value
                if value > best_score:
                    best_score = value
                    best_square = list(trial)

        polished, quadgram, used = _polish(pairs, best_square, scorer)
        evaluated += used
        yield _SearchResult(
            letters=from_numbers(polished),
            score=quadgram,
            digraph_score=best_score,
            moves=evaluated,
            budget_hit=budget_hit,
        )
        if budget_hit:
            break


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


def _trimmed(results: CandidateSet, top: int) -> CandidateSet:
    """Keep only the *top* highest-scoring candidates, if a limit was asked for.

    A non-positive *top* means "give me everything", which is what a caller
    comparing restarts against each other wants.
    """
    if top <= 0 or len(results) <= top:
        return results
    kept = CandidateSet()
    for candidate in results.top(top):
        kept.add(candidate)
    return kept


def _searchable_square(
    letters: str, preferred: PlayfairSquare
) -> tuple[PlayfairSquare, str | None]:
    """Pick the square alphabet the ciphertext could actually have come from.

    Which letter the square leaves out is not a setting the operator has to
    get right before the search will run -- when no key is supplied it is part
    of what is being searched, and the ciphertext states it outright. A J in
    the ciphertext is *proof* that the square did not merge I/J, which is
    genuine negative evidence, not an error in the input. Throwing it away as
    an exception forces the operator to guess the flag that would have worked;
    reading it and saying so leaves them better informed either way.

    So: if *preferred* can produce *letters*, it is returned unchanged. If it
    cannot, and exactly one standard omission is left standing, that square is
    returned together with a sentence recording the swap for the diagnostics.
    If the ciphertext rules out every standard square, *preferred* is returned
    unchanged and :func:`_fatal_problems` is left to explain why nothing fits
    -- because at that point there is no honest square to search with.

    Only ever used when no key was supplied. A supplied key is an assertion
    about the square, and quietly re-keying it under a different alphabet
    would answer a question the operator did not ask.
    """
    if preferred.omitted not in letters:
        return preferred, None

    escape = _usable_omissions(letters, preferred.omitted)
    if not escape:
        return preferred, None

    chosen = plain_square(omit=escape[0])
    was = (
        f"merged {preferred.folded_onto}/{preferred.omitted}"
        if preferred.folded_onto
        else f"dropped {preferred.omitted}"
    )
    now = (
        f"merges {chosen.folded_onto}/{chosen.omitted}"
        if chosen.folded_onto
        else f"drops {chosen.omitted}"
    )
    note = (
        f"the ciphertext contains {preferred.omitted}, so the square cannot "
        f"have {was}; the search used a square that {now} instead, where "
        f"{preferred.omitted} is a legal ciphertext letter. If that square is "
        f"wrong the text is not Playfair, and the plaintext below will read "
        f"as nonsense."
    )
    return chosen, note


def _outlook(length: int) -> str:
    """An honest one-line prognosis for a hill climb on this much ciphertext."""
    if length < 100:
        return (
            f"{length} letters: far too short. Playfair hill climbing needs "
            f"about {RELIABLE_LENGTH} letters; treat any result below that as "
            "noise, not as a solution."
        )
    if length < RELIABLE_LENGTH:
        return (
            f"{length} letters: below the roughly {RELIABLE_LENGTH} letters "
            "this attack needs. It usually fails here, and it fails "
            "confidently -- read the plaintext, do not trust the score."
        )
    if length < 300:
        return (
            f"{length} letters: marginal. Past about {RELIABLE_LENGTH} letters "
            "the attack starts to work, but expect several restarts to "
            "disagree with each other."
        )
    return (
        f"{length} letters: enough for this attack, which becomes reliable "
        "somewhere around 300-500 letters."
    )


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    key: str | PlayfairSquare | None = None,
    keys: Sequence[str | PlayfairSquare] | None = None,
    omit: str = DEFAULT_OMITTED,
    restarts: int = DEFAULT_RESTARTS,
    iterations: int = DEFAULT_ITERATIONS,
    temperature: float | None = None,
    seed: int | None = None,
    stop_when_strong: bool = True,
    time_budget: float | None = None,
) -> CandidateSet:
    """Attack Playfair ciphertext and return ranked candidates, never one answer.

    Two modes:

    * If ``key`` or ``keys`` is given, each one is simply used to decrypt and
      the results are scored and ranked. Nothing is searched.
    * Otherwise the 25-letter square is recovered by simulated annealing on
      digraph statistics followed by a steepest-ascent polish under the
      order-3 model, run ``restarts`` times from independent random squares.
      Every restart's best square becomes a candidate, so the operator can see
      whether the restarts agreed (the ``agreements`` diagnostic) or wandered
      off separately, which is the honest signal for whether the ciphertext
      was long enough.

    ``stop_when_strong`` (on by default) ends the search as soon as one
    restart produces a decryption that reaches the toolkit's own ``strong``
    thresholds -- both an English-like n-gram score and high dictionary
    coverage. Turn it off to make every restart run and see how many of them
    independently agree, which is the better evidence when the ciphertext is
    short. Either way the diagnostics record how many restarts ran.

    ``temperature`` defaults to ``0.030`` per digraph of ciphertext, because
    the scores being compared grow with the length of the message and the
    acceptance rule has to be on their scale. Raising it much above the
    default makes the search fail completely; the measurements are recorded
    beside ``DEFAULT_TEMPERATURE_PER_DIGRAPH``.

    Reproducibility: pass ``seed`` and the search is deterministic. Randomness
    comes from a private ``random.Random``, never the global module.

    Limits: hill climbing on Playfair needs roughly 200 letters of ciphertext
    to work at all and is only reliable from about 300-500. Below that the
    top candidate will still be returned, still scored, and still wrong; the
    ``outlook`` diagnostic on every candidate says so in words, and
    ``ciphertext_letters`` records what the attack had to work with.

    The omitted letter, when no key is supplied, is chosen from the ciphertext
    rather than taken on trust. ``omit`` says which square to *prefer*, but a
    ciphertext containing that letter proves the square did not omit it, so
    the search moves to the other standard omission and records what it did in
    the ``omitted_letter_changed`` diagnostic on every candidate. That is the
    ``cipher_tool playfair message.txt`` case: a J in the text is evidence
    about the square, not a mistake in the input. Pass a key to switch the
    behaviour off -- a supplied key is an assertion about the square, so a
    ciphertext that contradicts it raises instead.

    Raises ``ValueError`` if the ciphertext cannot have come from this cipher
    at all: odd length, a supplied key whose square could not have produced
    it, or -- with no key -- a text that uses every standard omitted letter and
    so cannot have come from any standard square. Empty input returns an empty
    :class:`CandidateSet`.
    """
    normalized = normalize(source) if isinstance(source, str) else source
    letters = normalized.letters
    engine = scorer if scorer is not None else default_scorer()
    results = CandidateSet()
    if not letters:
        return results

    supplied: list[str | PlayfairSquare] = []
    if key is not None:
        supplied.append(key)
    if keys:
        supplied.extend(keys)

    # The structural checks depend only on which letter the square omits, so
    # they can be made once, before any key is considered. With no key to go
    # on, the omitted letter is part of what is being searched, so a
    # ciphertext that rules out the requested omission re-points the search
    # rather than stopping it -- see _searchable_square.
    if supplied:
        reference = as_square(supplied[0], omit=omit)
        omission_note = None
    else:
        reference, omission_note = _searchable_square(
            letters, plain_square(omit=omit)
        )
    fatal = _fatal_problems(letters, reference)
    if fatal:
        raise ValueError(
            "This ciphertext cannot be Playfair with these settings. "
            + " ".join(fatal)
        )

    shared: dict[str, object] = {
        "ciphertext_letters": len(letters),
        "digraphs": len(letters) // 2,
        "omitted_letter": reference.omitted,
    }
    if omission_note is not None:
        shared["omitted_letter_changed"] = omission_note

    if supplied:
        for entry in supplied:
            square = as_square(entry, omit=omit)
            problems = _fatal_problems(letters, square)
            if problems:
                raise ValueError(
                    "This ciphertext cannot be Playfair with the supplied key. "
                    + " ".join(problems)
                )
            plaintext = decrypt(letters, square)
            diagnostics: dict[str, object] = dict(shared)
            diagnostics["attack"] = "supplied key (no search)"
            diagnostics["square"] = square.compact()
            annotate(diagnostics, plaintext, engine)
            label = entry if isinstance(entry, str) else square.letters
            results.add(
                Candidate(
                    method="Playfair",
                    key=f"key={label}" if isinstance(entry, str) else f"square={label}",
                    score=engine.score(plaintext),
                    plaintext=plaintext,
                    diagnostics=diagnostics,
                    display=normalized.relayout(plaintext),
                )
            )
        return _trimmed(results, top)

    if restarts == 0:
        # A caller that asks for no restarts has switched the search off (the
        # `auto` pipeline does exactly this at its lower effort levels). That
        # is a request, not a mistake, and the honest answer is no candidates.
        return results
    if restarts < 0 or iterations <= 0:
        raise ValueError(
            f"restarts must not be negative and iterations must be positive, "
            f"got restarts={restarts}, iterations={iterations}."
        )
    if temperature is not None and temperature < 0:
        raise ValueError(
            f"The annealing temperature must not be negative, got "
            f"{temperature}. Use temperature=0 for plain hill climbing, or "
            "leave it unset for the measured default."
        )

    rng = random.Random(seed)
    pairs = _encode_pairs(letters)
    alphabet_values = [
        ord(ch) - 65 for ch in ALPHABET if ch != reference.omitted
    ]
    deadline = None if time_budget is None else time.monotonic() + time_budget
    heat = (
        temperature
        if temperature is not None
        else DEFAULT_TEMPERATURE_PER_DIGRAPH * len(pairs)
    )

    # Each restart is turned into a scored candidate as it arrives, so that a
    # restart which has already produced legible English can end the search.
    attempts: list[tuple[_SearchResult, str, dict[str, object]]] = []
    evaluated = 0
    budget_hit = False
    stopped_early = False

    for result in _search(
        pairs,
        alphabet_values,
        engine,
        omitted=reference.omitted,
        restarts=restarts,
        iterations=iterations,
        temperature=heat,
        rng=rng,
        deadline=deadline,
    ):
        square = square_from_letters(result.letters, omit=reference.omitted)
        plaintext = decrypt(letters, square)
        evidence: dict[str, object] = {}
        annotate(evidence, plaintext, engine)
        attempts.append((result, plaintext, evidence))
        evaluated = result.moves
        budget_hit = budget_hit or result.budget_hit

        probe = Candidate(
            method="Playfair",
            key="",
            score=result.score,
            plaintext=plaintext,
            diagnostics=dict(evidence),
        )
        if stop_when_strong and probe.confidence() == "strong":
            stopped_early = True
            break

    outlook = _outlook(len(letters))
    for result, plaintext, evidence in attempts:
        square = square_from_letters(result.letters, omit=reference.omitted)
        diagnostics = dict(shared)
        diagnostics["attack"] = (
            f"digraph annealing then order-3 polish; {len(attempts)} of "
            f"{restarts} restarts run, {iterations} moves each, "
            f"T0={heat:.1f}"
        )
        diagnostics["outlook"] = outlook
        diagnostics["square"] = square.compact()
        diagnostics["square_canonical"] = canonical_square(square).compact()
        diagnostics["equivalent_squares"] = (
            "25 (all cyclic row/column rotations encipher identically, so the "
            "true square is probably a rotation of this one)"
        )
        diagnostics["moves_evaluated"] = evaluated
        diagnostics["digraph_score"] = result.digraph_score
        diagnostics["seed"] = seed
        if stopped_early:
            diagnostics["stopped_early"] = (
                "a restart reached the toolkit's 'strong' thresholds, so the "
                "remaining restarts were not run"
            )
        if budget_hit:
            diagnostics["time_budget_hit"] = True
        diagnostics.update(evidence)
        results.add(
            Candidate(
                method="Playfair",
                key=f"square={square.letters}",
                score=engine.score(plaintext),
                plaintext=plaintext,
                diagnostics=diagnostics,
                display=normalized.relayout(plaintext),
            )
        )

    return _trimmed(results, top)
