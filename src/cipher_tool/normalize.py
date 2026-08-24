"""Input normalisation for ciphertext.

Design rule that everything else in the toolkit depends on:

    The original text is never destroyed, and whitespace in the input is
    never treated as evidence about plaintext word boundaries.

Competition ciphertext is usually printed in five-letter groups. Those groups
are a transcription convenience, not information about the plaintext. Several
classical attacks would be badly misled by treating them as words, so this
module keeps two parallel views of the input:

* ``original``      -- exactly what the user gave us, byte for byte, less any
  byte-order mark (an encoding artefact, not part of the message).
* ``letters_only``  -- uppercase A-Z with everything else removed.

A position map ties the two together so a recovered plaintext can be re-laid
into the original layout for human reading.

There is a third view, added later and for a different reason: ``symbols``,
the uppercase A-Z *and* 0-9 stream, with an :class:`Inventory` counting what
each character of the input was. The letters-only view is a filter, and a
filter that cannot say what it removed lets the toolkit describe its own
leftovers as though they were the message. MEASURED on a real paste of 1,251
alphanumeric symbols, 891 of them letters: the screen said "Read 891 letters",
solved the wreckage as a monoalphabetic substitution and offered it as an
answer. Nothing on that screen was false; nothing on it was the truth either.

And a fourth, ``marks``, for the same reason one more time. A-Z and 0-9 are not
the only notations a message can be written in: 2024 challenge 9B is 12,935
characters of ``|`` ``/`` and ``\\``, which are none of them letters or digits,
so the symbol stream was empty and the paste screen announced "Read 0 symbols"
over a full page of ciphertext. The rule that resolves it is narrow, because
punctuation in an ordinary paste really is layout and reporting it would be
noise:

    Marks are layout while a symbol stream exists, and are the message when
    none does.

An inventory that was never measured is all zeroes, so it has no marks either
and takes exactly the path it always did.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Iterator

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALPHABET_SIZE = 26

#: Accented Latin letters are folded onto their base letter so that a paste
#: from a PDF does not silently drop characters. Anything that is still not an
#: ASCII letter after folding is treated as non-alphabetic.
#:
#: U+0300 to U+036F is the Unicode "combining diacritical marks" block. It is
#: written as escapes rather than literal characters so that every source file
#: in this project stays pure ASCII and is trivially auditable.
_COMBINING = re.compile("[\u0300-\u036f]")

#: U+FEFF is the byte-order mark. Notepad writes one at the start of every
#: file it saves as "UTF-8", and a copy-and-paste out of such a file can carry
#: one into the middle of a string. It is an encoding marker, not a character
#: of the message, so it is removed rather than carried around: left in, it
#: reaches the terminal and raises UnicodeEncodeError on a code page that
#: cannot represent it, which reads as a crash in the toolkit rather than as
#: an artefact of how the file was saved.
_BYTE_ORDER_MARK = "\ufeff"


def strip_bom(text: str) -> str:
    """Remove byte-order marks from anywhere in *text*.

    ``utf-8-sig`` removes a leading BOM when decoding a file, but a BOM can
    also arrive mid-string from a paste, or from concatenated files. This is
    the belt to that decoder's braces, and it is applied to every piece of
    text the toolkit takes in.
    """
    return text.replace(_BYTE_ORDER_MARK, "")


def fold_to_ascii(text: str) -> str:
    """Strip accents so that an accented paste does not lose letters.

    A capital E with an acute accent becomes a plain E, so text copied out of
    a PDF still yields the letters it appears to contain.

    Uses NFKD decomposition (letter + combining mark) and then discards the
    combining marks. Characters with no ASCII base are left untouched; they
    will simply be filtered out as non-alphabetic later.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return _COMBINING.sub("", decomposed)


def letters_only(text: str) -> str:
    """Return the uppercase A-Z letters of *text*, in order, nothing else."""
    folded = fold_to_ascii(text)
    return "".join(ch for ch in folded.upper() if "A" <= ch <= "Z")


def symbols_only(text: str) -> str:
    """Return the uppercase A-Z and 0-9 characters of *text*, in order.

    The letters-only view answers "what can a letter cipher read?". This one
    answers "what did the sender actually write?", and the gap between the two
    is the thing the toolkit used to throw away without saying so.
    """
    folded = fold_to_ascii(text)
    return "".join(
        ch for ch in folded.upper() if "A" <= ch <= "Z" or "0" <= ch <= "9"
    )


def clean_key(text: str) -> str:
    """Normalise a user-supplied alphabetic key the same way as ciphertext."""
    return letters_only(text)


@dataclass(frozen=True)
class Inventory:
    """What each character of an input was, counted by class.

    The defaults are all zero and that is load-bearing: an all-zero inventory
    means NOT MEASURED, never "measured and found empty". Anything that builds
    a :class:`NormalizedText` by hand gets one, so every predicate written
    over an inventory must read zeroes as "behave exactly as before".

    Attributes
    ----------
    letters:
        A-Z after accent folding.
    digits:
        0-9, ASCII, after the same folding.
    other:
        Non-space characters that are neither: punctuation and symbols.
    spaces:
        Whitespace. Layout, never content -- see the module docstring.
    """

    letters: int = 0
    digits: int = 0
    other: int = 0
    spaces: int = 0

    @property
    def symbols(self) -> int:
        """Letters plus digits: the stream a symbol cipher is written in."""
        return self.letters + self.digits

    @property
    def total(self) -> int:
        """Every character counted, including layout."""
        return self.letters + self.digits + self.other + self.spaces

    @property
    def digit_fraction(self) -> float:
        """Digits as a fraction of the SYMBOL stream, 0.0 when there is none.

        Deliberately not a fraction of the whole input: spaces and punctuation
        are transcription layout, and dividing by them would make the same
        message look less numeric merely for being printed in groups of five.
        """
        if not self.symbols:
            return 0.0
        return self.digits / self.symbols

    def describe(self) -> str:
        """One phrase naming the symbol stream, e.g. ``"1251 symbols: 891
        letters and 360 digits"``.

        A class that is empty is omitted rather than reported as zero, so an
        ordinary letters-only paste reads ``"891 symbols: 891 letters"`` and
        does not invite the reader to wonder what a digit was doing there.
        Punctuation and whitespace are not mentioned: they are layout, and the
        count in front of the colon is the symbol stream, not the file.

        Unless there is no symbol stream at all, in which case the marks are
        not layout -- they are everything that was pasted, and saying "0
        symbols" over 12,935 of them is the filter describing its own
        leftovers again. An all-zero inventory has no marks either, so NOT
        MEASURED still reads "0 symbols" exactly as before.
        """
        if not self.symbols and self.other:
            return (
                f"{self.other} mark{'' if self.other == 1 else 's'}, "
                "none of them letters or digits"
            )

        parts: list[str] = []
        if self.letters:
            parts.append(f"{self.letters} letter{'' if self.letters == 1 else 's'}")
        if self.digits:
            parts.append(f"{self.digits} digit{'' if self.digits == 1 else 's'}")
        head = f"{self.symbols} symbol{'' if self.symbols == 1 else 's'}"
        if not parts:
            return head
        return f"{head}: {' and '.join(parts)}"


def inventory_of(text: str) -> Inventory:
    """Count the character classes of *text* without normalising it."""
    return normalize(text).inventory


@dataclass(frozen=True)
class NormalizedText:
    """Two synchronised views of one piece of input text.

    Attributes
    ----------
    original:
        The untouched input.
    letters:
        Uppercase A-Z only. This is what every cryptanalysis routine reads.
    positions:
        ``positions[i]`` is the index in ``original`` of ``letters[i]``.
        This is what makes :meth:`relayout` possible.
    groups:
        The whitespace-separated tokens of the original, with their own
        non-letter characters removed. Used only for reporting (for example
        "are all displayed groups the same length?"), never as word evidence.
    symbols:
        Uppercase A-Z and 0-9, in order. ``""`` means NOT MEASURED, which is
        what a hand-built instance gets -- never read it as "no symbols".
    symbol_positions:
        ``symbol_positions[i]`` is the index in ``original`` of ``symbols[i]``.
    inventory:
        What the input contained, by class. All zeroes means NOT MEASURED.
    marks:
        The non-whitespace characters that are neither letters nor digits, in
        order. ``""`` means NOT MEASURED. Layout in an ordinary paste, and the
        whole message in a mark-notation one -- see the module docstring.

    The last four are appended with defaults on purpose. Every existing
    construction of this class, positional or keyword, keeps working, and
    every predicate written over the inventory must fail closed onto the
    toolkit's older behaviour when it sees zeroes.
    """

    original: str
    letters: str
    positions: tuple[int, ...]
    groups: tuple[str, ...]
    symbols: str = ""
    symbol_positions: tuple[int, ...] = ()
    #: Frozen and immutable, so one shared default instance is safe here.
    inventory: Inventory = Inventory()
    marks: str = ""

    # -- convenience -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.letters)

    @property
    def length(self) -> int:
        """Number of alphabetic characters."""
        return len(self.letters)

    @property
    def is_empty(self) -> bool:
        """True when the input contained no letters at all."""
        return not self.letters

    @property
    def has_symbols(self) -> bool:
        """True when the symbol stream was measured and is not empty."""
        return bool(self.symbols)

    @property
    def has_marks(self) -> bool:
        """True when the mark stream was measured and is not empty."""
        return bool(self.marks)

    @property
    def digit_fraction(self) -> float:
        """Digits as a fraction of the symbol stream; 0.0 when not measured."""
        return self.inventory.digit_fraction

    def describe_input(self) -> str:
        """What was pasted, in one phrase, for a screen to print verbatim."""
        return self.inventory.describe()

    def grouped(self, size: int = 5, per_line: int = 10) -> str:
        """Render the normalised letters in fixed-size groups for display."""
        return group_text(self.letters, size=size, per_line=per_line)

    def relayout(self, plaintext: str) -> str:
        """Pour *plaintext* back into the punctuation/spacing of the original.

        ``plaintext`` must be the same length as :attr:`letters`; each of its
        characters replaces the letter that stood at the same position in the
        original. Non-letter characters of the original (spaces, punctuation,
        line breaks) are preserved.

        This is a *display* aid only. It deliberately does not claim that the
        original spacing matches plaintext words -- for a five-letter-grouped
        input it will not, and the caller is expected to know that.
        """
        if len(plaintext) != len(self.letters):
            raise ValueError(
                f"relayout needs {len(self.letters)} characters, got {len(plaintext)}"
            )
        chars = list(self.original)
        for source_index, position in enumerate(self.positions):
            chars[position] = plaintext[source_index]
        return "".join(chars)

    def uniform_group_length(self) -> int | None:
        """Return the common group length, or ``None`` if groups differ.

        A block of equal-length groups (classically five) is weak evidence
        that the spacing is transcription formatting rather than real words.
        Reported as an observation, never acted on automatically.
        """
        lengths = {len(g) for g in self.groups if g}
        if len(lengths) == 1:
            return lengths.pop()
        return None


def normalize(text: str) -> NormalizedText:
    """Build a :class:`NormalizedText` from arbitrary user input.

    Handles upper/lower case, spaces, tabs, line breaks, punctuation, digits,
    five-character grouping and any other arbitrary grouping. Nothing is
    rejected; anything that is not a letter is simply not part of the
    cryptanalytic view.

    Byte-order marks are the one exception to "the original is preserved
    exactly". A BOM is a record of how the file was *saved*, not something the
    sender wrote, and keeping it would put an unprintable character into
    ``original`` -- which is echoed verbatim by ``show``.
    """
    text = strip_bom(text)
    folded = fold_to_ascii(text)

    letters: list[str] = []
    positions: list[int] = []
    symbols: list[str] = []
    symbol_positions: list[int] = []
    marks: list[str] = []
    letter_count = digit_count = other_count = space_count = 0
    # Index into `text` (the untouched original). Folding can change string
    # length, so walk the *original* and fold one character at a time to keep
    # the index mapping exact.
    #
    # The symbol stream is built in this SAME pass rather than by a second
    # walk: the two position maps must agree about which character of the
    # original produced which entry, and two passes over a folded string is
    # exactly how they would stop agreeing.
    for index, raw_char in enumerate(text):
        candidate = fold_to_ascii(raw_char).upper()
        # A folded character may expand (rare) -- take its first ASCII letter
        # or digit, whichever comes first.
        for ch in candidate:
            if "A" <= ch <= "Z":
                letters.append(ch)
                positions.append(index)
                symbols.append(ch)
                symbol_positions.append(index)
                letter_count += 1
                break
            if "0" <= ch <= "9":
                symbols.append(ch)
                symbol_positions.append(index)
                digit_count += 1
                break
        else:
            # Not a symbol. Whitespace is layout; anything else is
            # punctuation the sender or the transcriber put there.
            if raw_char.isspace():
                space_count += 1
            else:
                # Recorded as written, not folded: the reader needs to see the
                # character they typed when a screen names the alphabet back
                # to them.
                marks.append(raw_char)
                other_count += 1

    groups = tuple(
        cleaned
        for token in folded.split()
        if (cleaned := "".join(c for c in token.upper() if "A" <= c <= "Z"))
    )

    return NormalizedText(
        original=text,
        letters="".join(letters),
        positions=tuple(positions),
        groups=groups,
        symbols="".join(symbols),
        symbol_positions=tuple(symbol_positions),
        inventory=Inventory(
            letters=letter_count,
            digits=digit_count,
            other=other_count,
            spaces=space_count,
        ),
        marks="".join(marks),
    )


def group_text(text: str, size: int = 5, per_line: int = 10) -> str:
    """Format *text* into groups of *size* characters, *per_line* per line.

    ``group_text("ATTACKATDAWN", 5, 10)`` -> ``"ATTAC KATDA WN"``.
    A *size* of zero or less returns the text unchanged.
    """
    if size <= 0 or not text:
        return text
    groups = [text[i : i + size] for i in range(0, len(text), size)]
    if per_line <= 0:
        return " ".join(groups)
    lines = [
        " ".join(groups[i : i + per_line]) for i in range(0, len(groups), per_line)
    ]
    return "\n".join(lines)


def chunks(text: str, size: int) -> Iterator[str]:
    """Yield successive *size*-character chunks of *text* (last may be short)."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for i in range(0, len(text), size):
        yield text[i : i + size]


def columns(text: str, count: int) -> list[str]:
    """Split *text* into *count* interleaved columns.

    ``columns("ABCDEFG", 3)`` -> ``["ADG", "BE", "CF"]``.

    This is the "take every nth letter" split used by Vigenere key-length
    analysis: column *i* contains every character enciphered by key position
    *i*, so each column is a simple Caesar shift.
    """
    if count <= 0:
        raise ValueError("column count must be positive")
    buckets = ["" for _ in range(count)]
    for index, ch in enumerate(text):
        buckets[index % count] += ch
    return buckets


def to_numbers(text: str) -> list[int]:
    """Map ``A..Z`` to ``0..25``. Input must already be letters-only."""
    return [ord(ch) - 65 for ch in text]


def from_numbers(values: Iterable[int]) -> str:
    """Inverse of :func:`to_numbers`; values are reduced modulo 26."""
    return "".join(chr(65 + (v % ALPHABET_SIZE)) for v in values)
