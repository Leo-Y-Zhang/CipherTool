"""Input normalisation for ciphertext.

Design rule that everything else in the toolkit depends on:

    The original text is never destroyed, and whitespace in the input is
    never treated as evidence about plaintext word boundaries.

Competition ciphertext is usually printed in five-letter groups. Those groups
are a transcription convenience, not information about the plaintext. Several
classical attacks would be badly misled by treating them as words, so this
module keeps two parallel views of the input:

* ``original``      -- exactly what the user gave us, byte for byte.
* ``letters_only``  -- uppercase A-Z with everything else removed.

A position map ties the two together so a recovered plaintext can be re-laid
into the original layout for human reading.
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


def clean_key(text: str) -> str:
    """Normalise a user-supplied alphabetic key the same way as ciphertext."""
    return letters_only(text)


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
    """

    original: str
    letters: str
    positions: tuple[int, ...]
    groups: tuple[str, ...]

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
    """
    folded = fold_to_ascii(text)

    letters: list[str] = []
    positions: list[int] = []
    # Index into `text` (the untouched original). Folding can change string
    # length, so walk the *original* and fold one character at a time to keep
    # the index mapping exact.
    for index, raw_char in enumerate(text):
        candidate = fold_to_ascii(raw_char).upper()
        # A folded character may expand (rare) -- take its first ASCII letter.
        for ch in candidate:
            if "A" <= ch <= "Z":
                letters.append(ch)
                positions.append(index)
                break

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
