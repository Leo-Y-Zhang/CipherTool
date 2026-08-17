"""Polybius squares: rewriting each letter as a pair of coordinates.

The square
----------
A Polybius square is the alphabet written into a grid whose rows and columns
carry labels. The classical 5x5 square, with the numeric labels used since
antiquity, is::

        1  2  3  4  5
     1  A  B  C  D  E
     2  F  G  H  I  K
     3  L  M  N  O  P
     4  Q  R  S  T  U
     5  V  W  X  Y  Z

Every letter becomes the pair "row label, column label": A is 11, S is 43,
Z is 55. Twenty-six letters do not fit into twenty-five cells, so a
convention is needed. Two are in common use and both are supported here:

* merge I and J into one cell (the default, and what Playfair, Bifid and
  ADFGX normally assume);
* drop Q from the square altogether, which is why some texts write KW for QU.

MERGING IS LOSSY, AND THE LOSS IS NOT RECOVERABLE BY THE TOOL. Decoding cell
(2,4) of the square above always yields I, because the square no longer
records which of I and J went in. A decode reading "MAIL" might have been
"MAJL", and only a human reading the sentence can decide. The toolkit never
guesses: it prints I and leaves the judgement to the reader.

The labels themselves are free. "12345" is the default; "ADFGX" is the other
classic, chosen by the German army in 1918 because those five letters are
maximally unlike each other in Morse code, so a garbled transmission was
likely to stay readable. Anything else works too, provided the labels of one
axis are distinct.

Why the square matters
----------------------
On its own a Polybius square is a weak cipher. It is a monoalphabetic
substitution that happens to write its output as pairs of symbols, so once
the pairs are read as units the usual frequency analysis breaks it, and the
telltale of a small symbol alphabet (5 or 6 distinct characters) is visible
at a glance in the analyse report.

The square earns its keep as a *component*. Once a letter is two independent
coordinates, those coordinates can be moved around separately: that is what
Bifid does (see ``bifid.py``) and what the ADFGX field cipher did with a
columnar transposition. Splitting a letter into pieces and scattering the
pieces is what actually destroys letter statistics, and the square is the
tool that makes the pieces.

The attack in this module
-------------------------
Because a bare Polybius encoding is a fixed substitution, the whole secret is
the arrangement of the grid. :func:`solve` therefore does not search: it
decodes the text under each grid worth trying -- the standard alphabet, the
drop-Q alphabet, any keyed grid the operator suggests via ``keywords`` -- and
under the row/column transpose of each, then ranks the results by English
score. The label alphabet is read off the ciphertext itself.

Limitation, stated plainly: if the grid holds a genuinely unknown scrambled
alphabet, no amount of trying standard squares will find it. That case is a
25-symbol monoalphabetic substitution over the *pairs*, and the right tool is
the general substitution solver applied to the paired-up text, not this
module. ``label_permutations=True`` widens the search only over the order of
the labels (120 orders for a 5x5), which is a different and much smaller
question than the order of the alphabet inside the grid.
"""

from __future__ import annotations

import math
import time
from itertools import permutations
from typing import Final, Iterable, Mapping, Sequence

from .candidates import Candidate, CandidateSet
from .normalize import ALPHABET, NormalizedText, fold_to_ascii
from .scoring import EnglishScorer, annotate, default_scorer

# ---------------------------------------------------------------------------
# Alphabets and label sets
# ---------------------------------------------------------------------------

DIGITS: Final[str] = "0123456789"

#: The 25-letter alphabet used when I and J share a cell.
LETTERS_NO_J: Final[str] = "ABCDEFGHIKLMNOPQRSTUVWXYZ"

#: The 25-letter alphabet used when Q is dropped instead.
LETTERS_NO_Q: Final[str] = "ABCDEFGHIJKLMNOPRSTUVWXYZ"

#: All thirty-six symbols of the 6x6 square: no letter has to be sacrificed.
LETTERS_AND_DIGITS: Final[str] = ALPHABET + DIGITS

NUMERIC_LABELS_5: Final[str] = "12345"
NUMERIC_LABELS_6: Final[str] = "123456"
ADFGX_LABELS: Final[str] = "ADFGX"
ADFGVX_LABELS: Final[str] = "ADFGVX"

#: Label sets :func:`solve` will consider when they cover the ciphertext.
KNOWN_LABEL_SETS: Final[tuple[str, ...]] = (
    NUMERIC_LABELS_5,
    ADFGX_LABELS,
    NUMERIC_LABELS_6,
    ADFGVX_LABELS,
)


def _default_labels(size: int) -> str:
    """Labels used when the caller does not supply any."""
    if size == 5:
        return NUMERIC_LABELS_5
    if size == 6:
        return NUMERIC_LABELS_6
    if 2 <= size <= 9:
        return "123456789"[:size]
    raise ValueError(
        f"no default labels for a {size}x{size} square; pass row_labels "
        "explicitly"
    )


def _duplicates(text: str) -> list[str]:
    """Characters occurring more than once in *text*, in first-seen order."""
    seen: set[str] = set()
    repeated: list[str] = []
    for char in text:
        if char in seen and char not in repeated:
            repeated.append(char)
        seen.add(char)
    return repeated


def clean_keyword(
    keyword: str,
    alphabet: str,
    merges: Mapping[str, str] | None = None,
) -> str:
    """Normalise a squared-keyword: uppercase, folded, alphabet members only.

    Spaces and punctuation are dropped, accents are folded, and any character
    the *alphabet* covers only through a merge (J in an I/J square, say) is
    replaced by the letter it merges onto. Anything else that is a letter or a
    digit raises :class:`ValueError`, because silently dropping a Q from a
    keyword would build a square the operator did not ask for and could not
    reproduce by hand.
    """
    merge_map = dict(merges or {})
    out: list[str] = []
    for char in fold_to_ascii(str(keyword)).upper():
        if char in alphabet:
            out.append(char)
        elif char in merge_map:
            out.append(merge_map[char])
        elif char.isalnum():
            raise ValueError(
                f"keyword character {char!r} cannot be placed in this square: "
                f"its alphabet is {alphabet!r}. Rewrite the keyword or choose "
                "a square that contains that symbol."
            )
        # Everything else (space, hyphen, apostrophe) is formatting.
    cleaned = "".join(out)
    if not cleaned:
        raise ValueError(
            f"keyword {keyword!r} contains no symbol this square can use; "
            "a keyed square needs at least one usable letter."
        )
    return cleaned


def keyed_alphabet(
    keyword: str,
    alphabet: str,
    merges: Mapping[str, str] | None = None,
) -> str:
    """Build a keyed alphabet: keyword first, deduplicated, then the rest.

    ``keyed_alphabet("MONARCHY", LETTERS_NO_J)`` gives
    ``"MONARCHYBDEFGIKLPQSTUVWXZ"``.

    The point of a keyed square is that a human can rebuild it from one
    memorable word, so the rule has to be simple: write the keyword, skipping
    any letter already written, then continue with the unused letters of the
    alphabet in their normal order.
    """
    if not alphabet:
        raise ValueError("alphabet must not be empty")
    repeated = _duplicates(alphabet)
    if repeated:
        raise ValueError(
            f"alphabet {alphabet!r} repeats {''.join(repeated)!r}; every cell "
            "of a Polybius square must hold a different symbol"
        )
    cleaned = clean_keyword(keyword, alphabet, merges)
    seen: set[str] = set()
    ordered: list[str] = []
    for char in cleaned:
        if char not in seen:
            seen.add(char)
            ordered.append(char)
    for char in alphabet:
        if char not in seen:
            seen.add(char)
            ordered.append(char)
    return "".join(ordered)


# ---------------------------------------------------------------------------
# The square
# ---------------------------------------------------------------------------


class PolybiusSquare:
    """A labelled grid of symbols, and the encode/decode pair it defines.

    Coordinates handled in code are always **zero-based** ``(row, column)``
    indices; the printed labels are a presentation layer on top of them. So
    for the standard square ``coordinates("A") == (0, 0)`` while its printed
    form is ``"11"``. Keeping the arithmetic zero-based is what lets
    ``bifid.py`` do index arithmetic without constantly adding and
    subtracting one.
    """

    def __init__(
        self,
        symbols: str,
        *,
        row_labels: str | None = None,
        column_labels: str | None = None,
        merges: Mapping[str, str] | None = None,
        name: str | None = None,
    ) -> None:
        """Build a square from *symbols* written into the grid row by row.

        Parameters
        ----------
        symbols:
            The grid contents in reading order. Whitespace is ignored, so a
            square may be written out as five lines. The count must be a
            perfect square and no symbol may repeat.
        row_labels, column_labels:
            The printed labels. ``column_labels`` defaults to ``row_labels``,
            which defaults to "12345"/"123456" by size.
        merges:
            Symbols that are not in the grid but are accepted on input and
            silently folded onto a symbol that is, e.g. ``{"J": "I"}``. This
            is where the lossiness of a 25-cell square lives.
        name:
            Human-readable description used in candidate reports.
        """
        cleaned = "".join(str(symbols).split()).upper()
        size = math.isqrt(len(cleaned))
        if size < 2 or size * size != len(cleaned):
            raise ValueError(
                "a Polybius square needs a square number of symbols "
                f"(4, 9, 16, 25, 36, ...); got {len(cleaned)}: {cleaned!r}"
            )
        repeated = _duplicates(cleaned)
        if repeated:
            raise ValueError(
                f"symbol(s) {''.join(repeated)!r} appear more than once in the "
                "grid; a square must hold each symbol exactly once, otherwise "
                "decoding is ambiguous"
            )

        rows = row_labels if row_labels is not None else _default_labels(size)
        rows = "".join(str(rows).split()).upper()
        cols = column_labels if column_labels is not None else rows
        cols = "".join(str(cols).split()).upper()
        for axis, labels in (("row", rows), ("column", cols)):
            if len(labels) != size:
                raise ValueError(
                    f"a {size}x{size} square needs {size} {axis} labels, got "
                    f"{len(labels)}: {labels!r}"
                )
            duplicated = _duplicates(labels)
            if duplicated:
                raise ValueError(
                    f"{axis} labels {labels!r} repeat {''.join(duplicated)!r}; "
                    "labels must be distinct or coordinates are ambiguous"
                )

        merge_map: dict[str, str] = {}
        for source, target in dict(merges or {}).items():
            source = str(source).upper()
            target = str(target).upper()
            if len(source) != 1 or len(target) != 1:
                raise ValueError(
                    f"merges must map one symbol to one symbol, got "
                    f"{source!r} -> {target!r}"
                )
            if source in cleaned:
                raise ValueError(
                    f"{source!r} is already in the grid, so it cannot also be "
                    "merged onto another symbol"
                )
            if target not in cleaned:
                raise ValueError(
                    f"cannot merge {source!r} onto {target!r}: {target!r} is "
                    "not in the grid"
                )
            merge_map[source] = target

        self._symbols = cleaned
        self._size = size
        self._row_labels = rows
        self._column_labels = cols
        self._merges = merge_map
        self.name = name or f"{size}x{size} square"

        self._position = {
            symbol: (index // size, index % size)
            for index, symbol in enumerate(cleaned)
        }
        # Characters we consider to be message content rather than layout.
        # Letters always count; digits count only when the grid holds digits,
        # so a digit in the input to a 5x5 letter square is treated as
        # formatting (a line number, say) rather than as an error.
        content = set(ALPHABET) | set(cleaned) | set(merge_map)
        if any(char in DIGITS for char in cleaned):
            content |= set(DIGITS)
        self._content = content

    # -- description -------------------------------------------------------

    @property
    def size(self) -> int:
        """Side length of the grid."""
        return self._size

    @property
    def symbols(self) -> str:
        """Grid contents in reading order."""
        return self._symbols

    @property
    def rows(self) -> tuple[str, ...]:
        """The grid as a tuple of row strings."""
        size = self._size
        return tuple(
            self._symbols[index * size : (index + 1) * size]
            for index in range(size)
        )

    @property
    def row_labels(self) -> str:
        """The symbols naming the rows, e.g. "12345" or "ADFGX"."""
        return self._row_labels

    @property
    def column_labels(self) -> str:
        """The symbols naming the columns; usually the same as the rows."""
        return self._column_labels

    @property
    def merges(self) -> Mapping[str, str]:
        """Symbols accepted on input but folded onto another cell."""
        return dict(self._merges)

    @property
    def is_lossy(self) -> bool:
        """True when some input symbol cannot be recovered from a decode."""
        return bool(self._merges)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PolybiusSquare({self._symbols!r}, row_labels="
            f"{self._row_labels!r}, column_labels={self._column_labels!r})"
        )

    def render(self) -> str:
        """The grid as a printable block with its labels."""
        header = "   " + " ".join(self._column_labels)
        lines = [header]
        for index, row in enumerate(self.rows):
            lines.append(f" {self._row_labels[index]} " + " ".join(row))
        if self._merges:
            merged = ", ".join(
                f"{source}={target}" for source, target in sorted(self._merges.items())
            )
            lines.append(f" (merged on input: {merged}; a decode cannot undo this)")
        return "\n".join(lines)

    def transposed(self) -> PolybiusSquare:
        """The same square with rows and columns exchanged.

        Reading a square down the columns instead of along the rows is a
        genuine ambiguity when you do not know how the sender wrote it out,
        and it is exactly equivalent to filling the grid column by column, so
        one flag covers both conventions.
        """
        size = self._size
        flipped = "".join(
            self._symbols[column * size + row]
            for row in range(size)
            for column in range(size)
        )
        return PolybiusSquare(
            flipped,
            row_labels=self._column_labels,
            column_labels=self._row_labels,
            merges=self._merges,
            name=f"{self.name} (transposed)",
        )

    # -- coordinates -------------------------------------------------------

    def coordinates(self, letter: str) -> tuple[int, int]:
        """Zero-based ``(row, column)`` of *letter*, applying any merge."""
        if not isinstance(letter, str) or len(letter) != 1:
            raise ValueError(
                f"coordinates() takes exactly one character, got {letter!r}"
            )
        symbol = fold_to_ascii(letter).upper()
        symbol = self._merges.get(symbol, symbol)
        try:
            return self._position[symbol]
        except KeyError:
            raise ValueError(
                f"{letter!r} is not in this square ({self.name}); its symbols "
                f"are {self._symbols!r}"
            ) from None

    def letter(self, row: int, column: int) -> str:
        """The symbol at zero-based ``(row, column)``."""
        size = self._size
        for axis, value in (("row", row), ("column", column)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"{axis} index must be an integer, got {value!r}"
                )
            if not 0 <= value < size:
                raise ValueError(
                    f"{axis} index {value} is outside a {size}x{size} square "
                    f"(valid range 0..{size - 1})"
                )
        return self._symbols[row * size + column]

    def label_pair(self, letter: str) -> str:
        """The two printed labels of *letter*, e.g. ``"43"`` or ``"GG"``."""
        row, column = self.coordinates(letter)
        return self._row_labels[row] + self._column_labels[column]

    # -- text --------------------------------------------------------------

    def prepare(self, text: str) -> str:
        """The text exactly as this square will encipher it.

        Formatting (spaces, punctuation, and digits when the grid has none) is
        dropped, accents are folded, merged symbols are replaced by the cell
        they share. A letter the square genuinely cannot represent -- Q in a
        drop-Q square -- raises :class:`ValueError` rather than vanishing,
        because a silently shortened message is worse than a refused one.

        Call this to see the loss before it happens:
        ``standard().prepare("JAM") == "IAM"``.
        """
        out: list[str] = []
        for char in fold_to_ascii(str(text)).upper():
            if char in self._position:
                out.append(char)
            elif char in self._merges:
                out.append(self._merges[char])
            elif char in self._content:
                raise ValueError(
                    f"{char!r} cannot be represented in this square "
                    f"({self.name}); its symbols are {self._symbols!r}. "
                    "Rewrite that character (the usual convention for a "
                    "drop-Q square is to write QU as KW) or use a square "
                    "that contains it."
                )
        return "".join(out)

    def coordinate_stream(self, text: str) -> list[tuple[int, int]]:
        """Coordinates of every representable character of *text*, in order.

        This is the form ``bifid.py`` works in: the pairs are what the
        fractionating step takes apart.
        """
        return [self._position[char] for char in self.prepare(text)]

    def encode(self, text: str) -> str:
        """Encode *text* to label pairs: ``"ATTACK"`` -> ``"114444111325"``."""
        rows, cols = self._row_labels, self._column_labels
        return "".join(
            rows[row] + cols[column] for row, column in self.coordinate_stream(text)
        )

    def decode(self, text: str, *, strict: bool = True) -> str:
        """Decode a stream of label pairs back to symbols.

        Whitespace is always ignored, so five-figure groups and line breaks
        decode the same as an unbroken string. Any other character that is not
        a label raises :class:`ValueError` naming it and its position; pass
        ``strict=False`` to skip such characters instead, which is what
        :func:`solve` needs when it is guessing at the label set.

        An odd number of symbols raises: each source letter is two symbols, so
        an odd count means one was lost or one is spurious, and decoding it
        anyway would silently shift every following letter.
        """
        allowed = set(self._row_labels) | set(self._column_labels)
        kept: list[str] = []
        for index, char in enumerate(fold_to_ascii(str(text)).upper()):
            if char.isspace():
                continue
            if char in allowed:
                kept.append(char)
            elif strict:
                raise ValueError(
                    f"{char!r} at position {index} is not one of this square's "
                    f"labels (rows {self._row_labels!r}, columns "
                    f"{self._column_labels!r})"
                )
        if len(kept) % 2:
            raise ValueError(
                f"decoding needs an even number of symbols because each letter "
                f"is written as two, but {len(kept)} were found. One symbol is "
                "missing or one is spurious; check the transcription."
            )

        out: list[str] = []
        for index in range(0, len(kept), 2):
            row_label, column_label = kept[index], kept[index + 1]
            row = self._row_labels.find(row_label)
            if row < 0:
                raise ValueError(
                    f"{row_label!r} (symbol {index + 1}) is not a row label; "
                    f"the row labels are {self._row_labels!r}"
                )
            column = self._column_labels.find(column_label)
            if column < 0:
                raise ValueError(
                    f"{column_label!r} (symbol {index + 2}) is not a column "
                    f"label; the column labels are {self._column_labels!r}"
                )
            out.append(self._symbols[row * self._size + column])
        return "".join(out)

    # -- factories ---------------------------------------------------------

    @classmethod
    def standard(
        cls,
        keyword: str | None = None,
        *,
        row_labels: str = NUMERIC_LABELS_5,
        column_labels: str | None = None,
    ) -> PolybiusSquare:
        """The 5x5 square with I and J sharing a cell (the usual convention).

        With a *keyword* the alphabet is keyed first; a J in the keyword is
        folded to I, exactly as it would be in the message.
        """
        merges = {"J": "I"}
        if keyword is None:
            letters = LETTERS_NO_J
            name = "standard 5x5 (I/J merged)"
        else:
            letters = keyed_alphabet(keyword, LETTERS_NO_J, merges)
            cleaned = clean_keyword(keyword, LETTERS_NO_J, merges)
            name = f"keyed 5x5 (I/J merged) keyword={cleaned}"
        return cls(
            letters,
            row_labels=row_labels,
            column_labels=column_labels,
            merges=merges,
            name=name,
        )

    @classmethod
    def without_q(
        cls,
        keyword: str | None = None,
        *,
        merge_q_into: str | None = None,
        row_labels: str = NUMERIC_LABELS_5,
        column_labels: str | None = None,
    ) -> PolybiusSquare:
        """The other 5x5 convention: keep I and J apart, leave Q out.

        By default Q has nowhere to go, so encoding a Q raises rather than
        losing it. Pass ``merge_q_into="K"`` to adopt the "write KW for QU"
        habit automatically, which makes the square lossy in the same way the
        I/J square is.
        """
        merges = {"Q": merge_q_into.upper()} if merge_q_into else {}
        if keyword is None:
            letters = LETTERS_NO_Q
            name = "5x5 with Q dropped"
        else:
            letters = keyed_alphabet(keyword, LETTERS_NO_Q, merges)
            cleaned = clean_keyword(keyword, LETTERS_NO_Q, merges)
            name = f"keyed 5x5 with Q dropped keyword={cleaned}"
        return cls(
            letters,
            row_labels=row_labels,
            column_labels=column_labels,
            merges=merges,
            name=name,
        )

    @classmethod
    def six_by_six(
        cls,
        keyword: str | None = None,
        *,
        row_labels: str = NUMERIC_LABELS_6,
        column_labels: str | None = None,
    ) -> PolybiusSquare:
        """The 6x6 square over A-Z plus 0-9: thirty-six cells, no merging.

        Nothing is lost, which is why the 1918 ADFGVX cipher moved to it: a
        message full of map references and unit numbers needs its digits.
        """
        if keyword is None:
            symbols = LETTERS_AND_DIGITS
            name = "standard 6x6 (letters and digits)"
        else:
            symbols = keyed_alphabet(keyword, LETTERS_AND_DIGITS)
            cleaned = clean_keyword(keyword, LETTERS_AND_DIGITS)
            name = f"keyed 6x6 (letters and digits) keyword={cleaned}"
        return cls(
            symbols,
            row_labels=row_labels,
            column_labels=column_labels,
            name=name,
        )

    @classmethod
    def adfgx(cls, keyword: str | None = None) -> PolybiusSquare:
        """The 5x5 I/J square labelled ADFGX."""
        return cls.standard(keyword, row_labels=ADFGX_LABELS)

    @classmethod
    def adfgvx(cls, keyword: str | None = None) -> PolybiusSquare:
        """The 6x6 letters-and-digits square labelled ADFGVX."""
        return cls.six_by_six(keyword, row_labels=ADFGVX_LABELS)


# ---------------------------------------------------------------------------
# Module-level encrypt/decrypt in the toolkit's standard shape
# ---------------------------------------------------------------------------


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


def encrypt(text: str, square: PolybiusSquare | str | None = None) -> str:
    """Encrypt: each letter becomes its two labels. Returns the label stream."""
    return _as_square(square).encode(text)


def decrypt(text: str, square: PolybiusSquare | str | None = None) -> str:
    """Exact inverse of :func:`encrypt` for the same square."""
    return _as_square(square).decode(text)


# ---------------------------------------------------------------------------
# Attack
# ---------------------------------------------------------------------------


def _observed_symbols(stream: str) -> str:
    """The distinct symbols of *stream*, sorted."""
    return "".join(sorted(set(stream)))


def _label_sets_for(stream: str, requested: Sequence[str] | None) -> list[str]:
    """Label alphabets worth trying against this ciphertext.

    A Polybius stream uses exactly as many distinct symbols as the square has
    rows, so the ciphertext tells us the label set almost for free: if five
    distinct characters appear, those five characters are the labels. We also
    offer any well-known set that *covers* what we saw, because a short
    message can easily fail to use one of its labels.
    """
    if requested is not None:
        out: list[str] = []
        for labels in requested:
            cleaned = "".join(str(labels).split()).upper()
            if len(cleaned) < 2:
                raise ValueError(
                    f"label set {labels!r} needs at least two distinct symbols"
                )
            if _duplicates(cleaned):
                raise ValueError(f"label set {labels!r} repeats a symbol")
            if cleaned not in out:
                out.append(cleaned)
        return out

    observed = _observed_symbols(stream)
    found: list[str] = []
    if len(observed) in (5, 6):
        found.append(observed)
    for labels in KNOWN_LABEL_SETS:
        if set(observed) <= set(labels) and labels not in found:
            found.append(labels)
    return found


def _squares_for(
    size: int, labels: str, keywords: Sequence[str]
) -> tuple[list[PolybiusSquare], list[str]]:
    """Every square worth decoding with, plus notes on what could not be built.

    A keyword containing Q cannot key a drop-Q square, and a keyword with a
    digit cannot key a 5x5 square. Those combinations are skipped and reported
    rather than raising, because the same keyword list is offered to every
    square shape.
    """
    squares: list[PolybiusSquare] = []
    skipped: list[str] = []

    def attempt(builder, description: str) -> None:
        try:
            squares.append(builder())
        except ValueError as error:
            skipped.append(f"{description}: {error}")

    if size == 5:
        squares.append(PolybiusSquare.standard(row_labels=labels))
        squares.append(PolybiusSquare.without_q(row_labels=labels))
        for word in keywords:
            attempt(
                lambda w=word: PolybiusSquare.standard(w, row_labels=labels),
                f"keyed I/J square from {word!r}",
            )
            attempt(
                lambda w=word: PolybiusSquare.without_q(w, row_labels=labels),
                f"keyed drop-Q square from {word!r}",
            )
    elif size == 6:
        squares.append(PolybiusSquare.six_by_six(row_labels=labels))
        for word in keywords:
            attempt(
                lambda w=word: PolybiusSquare.six_by_six(w, row_labels=labels),
                f"keyed 6x6 square from {word!r}",
            )
    return squares, skipped


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    keywords: Iterable[str] | None = None,
    label_sets: Sequence[str] | None = None,
    try_transpose: bool = True,
    label_permutations: bool = False,
    time_budget: float | None = None,
    **options: object,
) -> CandidateSet:
    """Decode a Polybius stream under every square worth trying, and rank them.

    Unlike the letter ciphers, this solver reads the *original* text rather
    than its letters-only view: a numeric square encodes to digits, and
    ``normalize()`` deliberately throws digits away. When given a
    :class:`NormalizedText` we therefore work from ``.original``.

    Options
    -------
    keywords:
        Keywords to build keyed squares from. Without them only the two
        standard 5x5 arrangements and the standard 6x6 are tried.
    label_sets:
        Force the label alphabets instead of reading them off the text.
    try_transpose:
        Also decode with rows and columns exchanged (default on). This is a
        real ambiguity and costs one extra decode per square.
    label_permutations:
        Also try every ordering of the labels, applied to both axes at once
        (120 orders for a 5x5). Off by default; it multiplies the work and
        only helps when the sender wrote the labels out of order.
    time_budget:
        Seconds. The search stops cleanly when exceeded and records
        ``time_budget_hit``.

    ``top`` limits the returned set; pass ``top=0`` for everything.
    """
    engine = scorer if scorer is not None else default_scorer()
    if options:
        ignored = ", ".join(sorted(str(name) for name in options))
    else:
        ignored = ""

    raw = source.original if isinstance(source, NormalizedText) else str(source)
    folded = fold_to_ascii(raw).upper()

    words: list[str] = []
    for word in keywords or ():
        text = str(word)
        # Validate here so a typo in a keyword is an error, not a silent skip.
        # Any alphabet will do for the check: every square holds A-Z somewhere.
        clean_keyword(text, LETTERS_AND_DIGITS)
        words.append(text)

    candidates = CandidateSet()
    if label_sets is None:
        stream = "".join(char for char in folded if char.isalnum())
    else:
        allowed = {char for labels in label_sets for char in str(labels).upper()}
        stream = "".join(char for char in folded if char in allowed)
    if not stream:
        return candidates

    started = time.monotonic()
    budget_hit = False
    tried_labels: list[str] = []
    squares_tried = 0

    for labels in _label_sets_for(stream, label_sets):
        usable = "".join(char for char in stream if char in labels)
        if not usable:
            continue
        orders = (
            ["".join(order) for order in permutations(labels)]
            if label_permutations
            else [labels]
        )
        for order in orders:
            # Checked only once some work is done, so that even an exhausted
            # budget returns one honest, flagged answer rather than nothing.
            if (
                squares_tried
                and time_budget is not None
                and time.monotonic() - started > time_budget
            ):
                budget_hit = True
                break
            tried_labels.append(order)
            squares, skipped = _squares_for(len(order), order, words)
            for square in squares:
                shapes = [square]
                if try_transpose:
                    shapes.append(square.transposed())
                for shape in shapes:
                    squares_tried += 1
                    dropped = len(usable) % 2
                    trimmed = usable[: len(usable) - dropped]
                    plaintext = shape.decode(trimmed, strict=False)
                    if not plaintext:
                        continue
                    diagnostics: dict[str, object] = {
                        "labels": order,
                        "square": shape.name,
                        "grid": shape.symbols,
                        "symbols_read": len(trimmed),
                        "letters_out": len(plaintext),
                    }
                    if dropped:
                        diagnostics["odd_symbol_count"] = (
                            "the stream had an odd length; the final symbol was "
                            "ignored, so the transcription may be wrong"
                        )
                    if len(usable) != len(stream):
                        diagnostics["symbols_outside_label_set"] = (
                            len(stream) - len(usable)
                        )
                    if skipped:
                        diagnostics["squares_not_built"] = "; ".join(skipped)
                    if ignored:
                        diagnostics["ignored_options"] = ignored
                    annotate(diagnostics, plaintext, engine)
                    candidates.add(
                        Candidate(
                            method="Polybius",
                            key=f"square={shape.name} labels={order}",
                            score=engine.score(plaintext),
                            plaintext=plaintext,
                            diagnostics=diagnostics,
                            # Half as many letters come out as went in, so the
                            # original layout cannot be reused for display.
                            display=None,
                        )
                    )
        if budget_hit:
            break

    for candidate in candidates:
        candidate.diagnostics["label_sets_tried"] = len(set(tried_labels))
        candidate.diagnostics["squares_tried"] = squares_tried
        if budget_hit:
            candidate.diagnostics["time_budget_hit"] = True

    if top <= 0:
        return candidates
    return CandidateSet(candidates.top(top))


# ---------------------------------------------------------------------------
# Searching for an unknown square
# ---------------------------------------------------------------------------

#: Restarts for the substitution climb that recovers the cell alphabet.
DEFAULT_UNKNOWN_RESTARTS = 30

METHOD_UNKNOWN_SQUARE = "Polybius (square recovered by search)"

_SUBSTITUTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def solve_unknown_square(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    **options: object,
) -> CandidateSet:
    """Recover a keyed square nobody supplied. Ranked candidates only.

    :func:`solve` tries the squares it is handed, so a keyed message with no
    keyword available scored as noise -- measured, ``weak`` and wrong without
    the keyword, ``strong`` and right with it. A competition does not give
    you the keyword.

    **No search over squares is needed, which is the whole point.** A
    Polybius stream is a monoalphabetic substitution written two symbols at a
    time: map each distinct cell to a letter and it becomes an ordinary
    substitution cipher, which ``substitution.py`` already breaks well. That
    is the same joint the ADFGVX attack cuts at, and it costs about a second
    rather than the minutes a hill climb over 25! squares would.

    What comes back is the plaintext and the cell-to-letter mapping, which IS
    the square, read off by alignment. The row and column LABELS are not
    recovered and cannot be: they are a presentation layer, and every
    relabelling of the same grid produces the same plaintext.

    Options
    -------
    restarts:
        Restarts for the substitution climb.
    seed:
        Makes the run reproducible.
    """
    engine = scorer or default_scorer()
    text = source.original if isinstance(source, NormalizedText) else source
    restarts = int(options.pop("restarts", DEFAULT_UNKNOWN_RESTARTS))
    seed = options.pop("seed", None)
    if options:
        raise ValueError(
            "unknown option(s) for the Polybius square search: "
            f"{', '.join(sorted(str(name) for name in options))}"
        )

    results: CandidateSet = CandidateSet()
    symbols = [character for character in str(text) if not character.isspace()]
    if len(symbols) < 40 or len(symbols) % 2:
        # Two symbols per letter, so an odd count is not a whole message, and
        # below about twenty letters a substitution climb means nothing.
        return results

    pairs = ["".join(symbols[index:index + 2])
             for index in range(0, len(symbols), 2)]
    cells = sorted(set(pairs))
    if not 5 <= len(cells) <= len(_SUBSTITUTION_LETTERS):
        # Too few distinct cells to be a message; or more than there are
        # letters to map them onto, which a 6x6 square carrying digits can
        # produce but a letters-only plaintext cannot.
        return results

    alphabet = {cell: _SUBSTITUTION_LETTERS[index]
                for index, cell in enumerate(cells)}
    mapped = "".join(alphabet[pair] for pair in pairs)

    # Imported here rather than at module level: substitution.py is a heavier
    # module and nothing else in this one needs it.
    from . import substitution

    found = substitution.solve(mapped, scorer=engine, top=1,
                               restarts=restarts, seed=seed)
    best = found.best()
    if best is None:
        return results

    # The square, read off by alignment: cell i produced plaintext letter i.
    recovered: dict[str, str] = {}
    for pair, letter in zip(pairs, best.plaintext):
        recovered.setdefault(pair, letter)

    diagnostics: dict[str, object] = {
        "cells_used": len(cells),
        "square": " ".join(f"{cell}={letter}"
                           for cell, letter in sorted(recovered.items())),
        "search": (
            f"cell mapping, then a substitution climb of {restarts} restarts "
            "(not exhaustive)"
        ),
    }
    annotate(diagnostics, best.plaintext, engine)
    results.add(
        Candidate(
            method=METHOD_UNKNOWN_SQUARE,
            key=f"{len(cells)} cells recovered by substitution",
            score=best.score,
            plaintext=best.plaintext,
            diagnostics=diagnostics,
        )
    )

    if top is not None and top > 0:
        return CandidateSet(results.top(top))
    return results
