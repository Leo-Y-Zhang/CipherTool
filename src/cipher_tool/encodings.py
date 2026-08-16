"""Encodings: hexadecimal, binary, decimal ASCII, Base64 and Morse.

    NOTHING IN THIS MODULE IS ENCRYPTION.

Every format here is a *reversible representation* of text. There is no key,
there is no secret, and anybody who recognises the format can read the
message. A puzzle that arrives as ``48 45 4C 4C 4F`` has not been encrypted;
it has been written down in a different notation. That is why this module has
no ``encrypt``/``decrypt`` pair: it offers ``encode_*``/``decode_*``, and every
result it hands back is labelled so that a reader cannot mistake "decoded" for
"broken".

Why the toolkit needs it anyway
-------------------------------
Competition ciphertext is frequently wrapped in one of these notations, either
as the first layer of a multi-stage puzzle or simply as the way the text was
transcribed. Recognising the wrapper and peeling it off is the step before the
cryptanalysis starts.

The detection problem, and why it is a real problem
---------------------------------------------------
Detection is where a tool like this can do genuine damage, because these
notations overlap with ordinary text:

* Every letter of ``DEFACEDBEEF`` is a hexadecimal digit.
* Every letter of an uppercase English ciphertext is a Base64 character.
* ``01001000`` is a valid Base64 string, a valid hexadecimal string and a
  valid decimal number as well as being valid binary.

So "this text *could* be parsed as X" is worth nothing. Guessing wrongly here
is worse than not guessing: it invites the reader to throw away real
ciphertext and analyse rubbish. The rule this module follows is therefore:

    A format is only reported when the text could not plausibly be anything
    else, and the strongest evidence for that is what the decode *produces*.

Concretely, every detector must clear all of these before it says a word:

1. **Alphabet.** Not one character outside the format's alphabet, once
   layout (spaces, ``:``/``-`` separators, ``0x`` prefixes, line breaks) has
   been removed.
2. **Shape.** The length has to be consistent with the format: an even number
   of hexadecimal digits, a multiple of 7 or 8 binary digits, a multiple of
   four Base64 characters, and so on. Half the false positives die here,
   because arbitrary text has an arbitrary length.
3. **Product.** The decode must produce mostly printable ASCII. This is the
   test that does the real work. Twelve arbitrary bytes are all printable
   with probability of roughly (95/256)^12, which is about one in a hundred
   thousand, and the odds collapse further with every extra byte. Text that
   merely *looks* like hexadecimal decodes to high-bit noise and is rejected.

Base64 gets one extra requirement, because its alphabet contains the whole
Latin alphabet and so overlaps plain ciphertext completely: the text must
contain a lowercase letter, a digit, ``+``, ``/`` or ``=``. Real Base64 of six
or more bytes of text essentially always does; an uppercase-only ciphertext
never does. The cost of that rule is stated under "Known limitations" below.

Confidence is deliberately only two-valued, ``"possible"`` and ``"likely"``.
There is no "certain": a short input can satisfy every test above by accident,
and the module says so rather than pretending otherwise.

Known limitations, stated rather than hidden
--------------------------------------------
* Base64 whose original text is drawn only from the uppercase alphabet is not
  detected (see the extra rule above). ``decode_base64`` still decodes it when
  asked directly.
* URL-safe Base64 (``-`` and ``_`` instead of ``+`` and ``/``) is not decoded;
  those two characters are treated as layout separators elsewhere in the
  module and the ambiguity is not worth the false positives.
* A list of numbers that all lie in 1..26 is far more likely to be A1Z26
  letter numbering than decimal ASCII. Such a text decodes to control
  characters, so requirement 3 rejects it and no guess is made. This module
  does not implement A1Z26.
* Decoded bytes that are not printable ASCII are shown as ``?``. The decoded
  string is for reading, not for round-tripping binary data.

Base64 and the competition rules
--------------------------------
The Base64 helpers call the standard library's :mod:`base64` module. That is
part of Python itself, not a third-party package and not a deciphering tool:
Base64 is a transport notation defined in RFC 4648, and decoding it is no more
cryptanalysis than reading a number written in hexadecimal. Every actual
cryptanalytic algorithm in this toolkit is written from scratch.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from typing import Iterable

from .candidates import Candidate, CandidateSet
from .normalize import NormalizedText, normalize
from .scoring import EnglishScorer, annotate, default_scorer

__all__ = [
    "NOT_ENCRYPTION",
    "FORMAT_NAMES",
    "CONFIDENCE_LEVELS",
    "MORSE_TABLE",
    "EncodingGuess",
    "is_hex",
    "decode_hex",
    "encode_hex",
    "is_binary",
    "decode_binary",
    "encode_binary",
    "is_decimal",
    "decode_decimal",
    "encode_decimal",
    "is_base64",
    "decode_base64",
    "encode_base64",
    "is_morse",
    "decode_morse",
    "encode_morse",
    "identify",
    "describe_guesses",
    "solve",
]

#: Attached to every guess and every candidate this module produces.
NOT_ENCRYPTION = (
    "This is an encoding, not a cipher. It has no key and hides nothing; "
    "anyone who recognises the notation can read the text."
)

#: Canonical format names, in the order :func:`identify` reports ties.
FORMAT_NAMES: tuple[str, ...] = (
    "hexadecimal",
    "binary",
    "decimal ASCII",
    "Base64",
    "Morse code",
)

#: The only two confidence labels. There is deliberately no "certain".
CONFIDENCE_LEVELS: tuple[str, ...] = ("possible", "likely")

# ---------------------------------------------------------------------------
# Thresholds. Each one is a judgement about false positives, so each one is
# explained where it is used rather than left as a bare number.
# ---------------------------------------------------------------------------

#: Fewest hexadecimal digits (two bytes) before a text is called hexadecimal.
MIN_HEX_DIGITS = 4
#: Fewest binary groups before a text is called binary.
MIN_BINARY_GROUPS = 2
#: Fewest numbers before a text is called decimal ASCII.
MIN_DECIMAL_VALUES = 2
#: Fewest Base64 characters, and fewest decoded bytes, for Base64.
MIN_BASE64_CHARS = 8
MIN_BASE64_BYTES = 6
#: Fewest dot/dash tokens before a text is called Morse.
MIN_MORSE_TOKENS = 3

#: Proportion of decoded bytes that must be printable ASCII. Base64 is held to
#: a stricter standard (see the module docstring) and requires all of them.
MIN_PRINTABLE_RATIO = 0.9

# ---------------------------------------------------------------------------
# Byte helpers
# ---------------------------------------------------------------------------

#: Printable ASCII plus tab, newline and carriage return.
_PRINTABLE_BYTES = frozenset(range(32, 127)) | {9, 10, 13}

#: Characters that ordinary English prose is actually made of. Used only to
#: separate "decodes to text" from "decodes to printable but odd bytes".
_TEXT_BYTES = frozenset(
    ord(ch)
    for ch in (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        " .,;:'!?-()/\"\n\t\r"
    )
)


def _printable_ratio(data: bytes) -> float:
    """Fraction of *data* that is printable ASCII; 0.0 for empty input."""
    if not data:
        return 0.0
    return sum(1 for value in data if value in _PRINTABLE_BYTES) / len(data)


def _text_ratio(data: bytes) -> float:
    """Fraction of *data* that is letters, digits, spaces or plain punctuation."""
    if not data:
        return 0.0
    return sum(1 for value in data if value in _TEXT_BYTES) / len(data)


def _bytes_to_text(data: bytes) -> str:
    """Render bytes for a human reader, showing unprintable bytes as ``?``.

    Substitution rather than an exception, because a decode that is 95 per
    cent readable is exactly the evidence a solver wants to see; the ``?``
    marks make the damage visible instead of silently dropping it.
    """
    return "".join(
        chr(value) if value in _PRINTABLE_BYTES else "?" for value in data
    )


def _text_to_bytes(text: str) -> bytes:
    """ASCII bytes of *text*, refusing anything this notation cannot carry."""
    try:
        return text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "only ASCII text can be encoded by this module; "
            f"character {text[exc.start]!r} at position {exc.start} is not ASCII"
        ) from exc


def _confidence_for(data: bytes) -> str:
    """Label a decode: ``likely`` when it reads as text, else ``possible``.

    A decode that is wholly printable *and* made of the characters English is
    written with is strong evidence that the notation was found, not imagined.
    Anything weaker still passed the hard gates, so it is reported -- with the
    weaker label.
    """
    if (
        len(data) >= 4
        and _printable_ratio(data) == 1.0
        and _text_ratio(data) >= 0.9
    ):
        return "likely"
    return "possible"


# ---------------------------------------------------------------------------
# Guesses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodingGuess:
    """One conservative claim that *text* is written in a given notation.

    Attributes
    ----------
    format:
        One of :data:`FORMAT_NAMES`.
    confidence:
        ``"possible"`` or ``"likely"``. Never anything stronger.
    reason:
        The evidence, in words, including the counts that were checked.
    decoded:
        The decoded text, or ``None`` if decoding did not succeed.
    """

    format: str
    confidence: str
    reason: str
    decoded: str | None = None

    def __post_init__(self) -> None:
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"confidence must be one of {CONFIDENCE_LEVELS}, "
                f"got {self.confidence!r}"
            )

    def describe(self) -> str:
        """One line for a report, carrying the not-encryption warning."""
        return (
            f"{self.format}: {self.confidence} -- {self.reason} "
            "[encoding, NOT encryption]"
        )


# ---------------------------------------------------------------------------
# Hexadecimal
# ---------------------------------------------------------------------------

#: Layout characters that a transcription may put between hexadecimal bytes.
_SEPARATORS = re.compile(r"[\s,:;|_-]+")
#: ``0x`` and ``\x`` byte prefixes. ``x`` is not a hexadecimal digit, so these
#: can only ever be prefixes and removing them cannot destroy data.
_HEX_PREFIX = re.compile(r"(?i)0x|\\x")

_HEX_ALPHABET = frozenset("0123456789abcdefABCDEF")


def _hex_digits(text: str) -> str:
    """Strip layout from a hexadecimal transcription."""
    return _HEX_PREFIX.sub("", _SEPARATORS.sub("", text))


def _hex_bytes(text: str) -> bytes:
    """Decode hexadecimal to raw bytes, or raise :class:`ValueError`."""
    digits = _hex_digits(text)
    if not digits:
        return b""
    invalid = sorted({ch for ch in digits if ch not in _HEX_ALPHABET})
    if invalid:
        raise ValueError(
            "not hexadecimal: character(s) "
            + ", ".join(repr(ch) for ch in invalid)
            + " are not hexadecimal digits"
        )
    if len(digits) % 2:
        raise ValueError(
            f"hexadecimal needs an even number of digits (two per byte), "
            f"got {len(digits)}"
        )
    return bytes.fromhex(digits)


def decode_hex(text: str) -> str:
    """Decode hexadecimal, with or without separators or ``0x`` prefixes.

    ``decode_hex("48 45 4C")``, ``decode_hex("48454c")`` and
    ``decode_hex("0x48:0x45:0x4C")`` all give ``"HEL"``. Empty input gives
    ``""``. Raises :class:`ValueError` for non-hexadecimal characters or an
    odd number of digits.
    """
    return _bytes_to_text(_hex_bytes(text))


def encode_hex(text: str, *, separator: str = "", upper: bool = True) -> str:
    """Write ASCII *text* as hexadecimal, e.g. ``"HELLO"`` -> ``"48454C4C4F"``."""
    data = _text_to_bytes(text)
    pairs = [f"{value:02X}" if upper else f"{value:02x}" for value in data]
    return separator.join(pairs)


def _examine_hex(text: str) -> EncodingGuess | None:
    """Apply the alphabet/shape/product tests described in the module docstring."""
    digits = _hex_digits(text)
    if len(digits) < MIN_HEX_DIGITS or len(digits) % 2:
        return None
    if any(ch not in _HEX_ALPHABET for ch in digits):
        return None
    data = bytes.fromhex(digits)
    ratio = _printable_ratio(data)
    if ratio < MIN_PRINTABLE_RATIO:
        return None
    reason = (
        f"{len(digits)} hexadecimal digits ({len(data)} bytes), nothing outside "
        f"0-9 and A-F, and {ratio:.0%} of the decoded bytes are printable ASCII"
    )
    return EncodingGuess(
        format="hexadecimal",
        confidence=_confidence_for(data),
        reason=reason,
        decoded=_bytes_to_text(data),
    )


def is_hex(text: str) -> bool:
    """True only when *text* is hexadecimal on the evidence, not merely legal.

    An even number of hexadecimal digits, no character outside the alphabet,
    and a decode that is at least 90 per cent printable ASCII. ``"DEADBEEF"``
    is therefore **not** hexadecimal by this test: it parses, but it decodes
    to four high-bit bytes that no message would contain.
    """
    return _examine_hex(text) is not None


# ---------------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------------


def _binary_tokens(text: str) -> list[str]:
    """The whitespace/punctuation separated groups of a binary transcription."""
    return [token for token in _SEPARATORS.split(text.strip()) if token]


def _pack_bits(digits: str, bits: int) -> bytes:
    """Cut a bit string into *bits*-wide groups and turn each into a byte."""
    return bytes(
        int(digits[start : start + bits], 2) for start in range(0, len(digits), bits)
    )


def _binary_bytes(text: str, bits: int | None = None) -> tuple[bytes, int]:
    """Decode binary to bytes; returns the bytes and the group width used.

    Choosing the width is the only interesting decision. Modern transcriptions
    use 8 bits per character; puzzle transcriptions often use 7, because ASCII
    only ever needed 7. Evidence is used in this order:

    1. an explicit *bits* argument;
    2. the transcription's own grouping, when every group is the same width
       and that width is 7 or 8 -- the writer told us directly;
    3. divisibility of the total length, preferring 8 when both fit unless
       the 7-bit reading is visibly more printable.
    """
    tokens = _binary_tokens(text)
    digits = "".join(tokens)
    if not digits:
        return b"", bits if bits is not None else 8

    invalid = sorted({ch for ch in digits if ch not in "01"})
    if invalid:
        raise ValueError(
            "not binary: character(s) "
            + ", ".join(repr(ch) for ch in invalid)
            + " are not 0 or 1"
        )

    if bits is not None:
        if bits not in (7, 8):
            raise ValueError(f"binary groups must be 7 or 8 bits wide, got {bits}")
        if len(digits) % bits:
            raise ValueError(
                f"{len(digits)} binary digits is not a multiple of {bits}"
            )
        return _pack_bits(digits, bits), bits

    widths = {len(token) for token in tokens}
    if len(tokens) >= 2 and len(widths) == 1:
        width = widths.pop()
        if width in (7, 8):
            return _pack_bits(digits, width), width

    fits_eight = len(digits) % 8 == 0
    fits_seven = len(digits) % 7 == 0
    if fits_eight and fits_seven:
        eight = _pack_bits(digits, 8)
        seven = _pack_bits(digits, 7)
        if _printable_ratio(seven) > _printable_ratio(eight):
            return seven, 7
        return eight, 8
    if fits_eight:
        return _pack_bits(digits, 8), 8
    if fits_seven:
        return _pack_bits(digits, 7), 7
    raise ValueError(
        f"{len(digits)} binary digits is not a multiple of 7 or 8, "
        "so it cannot be a whole number of characters"
    )


def decode_binary(text: str, *, bits: int | None = None) -> str:
    """Decode 7-bit or 8-bit binary, with or without separators.

    ``decode_binary("01001000 01001001")`` gives ``"HI"``. With *bits* left as
    ``None`` the group width is inferred; pass 7 or 8 to force it. Empty input
    gives ``""``. Raises :class:`ValueError` for non-binary characters, an
    impossible length, or a *bits* value other than 7 or 8.
    """
    data, _ = _binary_bytes(text, bits)
    return _bytes_to_text(data)


def encode_binary(text: str, *, bits: int = 8, separator: str = " ") -> str:
    """Write ASCII *text* as binary, e.g. ``"HI"`` -> ``"01001000 01001001"``."""
    if bits not in (7, 8):
        raise ValueError(f"binary groups must be 7 or 8 bits wide, got {bits}")
    data = _text_to_bytes(text)
    if bits == 7 and any(value > 127 for value in data):
        raise ValueError("7-bit encoding cannot carry a byte above 127")
    return separator.join(format(value, f"0{bits}b") for value in data)


def _examine_binary(text: str) -> EncodingGuess | None:
    tokens = _binary_tokens(text)
    digits = "".join(tokens)
    if not digits or any(ch not in "01" for ch in digits):
        return None
    try:
        data, bits = _binary_bytes(text)
    except ValueError:
        return None
    if len(data) < MIN_BINARY_GROUPS:
        return None
    ratio = _printable_ratio(data)
    if ratio < MIN_PRINTABLE_RATIO:
        return None
    reason = (
        f"{len(digits)} binary digits, nothing but 0 and 1, read as {len(data)} "
        f"groups of {bits} bits, of which {ratio:.0%} are printable ASCII"
    )
    return EncodingGuess(
        format="binary",
        confidence=_confidence_for(data),
        reason=reason,
        decoded=_bytes_to_text(data),
    )


def is_binary(text: str) -> bool:
    """True only when *text* is binary on the evidence: 0s and 1s, a length
    that is a whole number of 7-bit or 8-bit characters, and a mostly
    printable decode."""
    return _examine_binary(text) is not None


# ---------------------------------------------------------------------------
# Decimal ASCII
# ---------------------------------------------------------------------------

_NUMBER_SPLIT = re.compile(r"[\s,;|]+")


def _decimal_values(text: str) -> list[int]:
    """Parse a space or comma separated list of ASCII codes."""
    tokens = [token for token in _NUMBER_SPLIT.split(text.strip()) if token]
    values: list[int] = []
    for token in tokens:
        if not token.isdigit():
            raise ValueError(
                f"not decimal ASCII: {token!r} is not a number "
                "(expected numbers separated by spaces or commas)"
            )
        value = int(token)
        if value > 255:
            raise ValueError(
                f"decimal ASCII values must be 0-255, got {value}; "
                "this module does not decode wider code points"
            )
        values.append(value)
    return values


def decode_decimal(text: str) -> str:
    """Decode space- or comma-separated decimal ASCII codes.

    ``decode_decimal("72 69 76 76 79")`` and ``decode_decimal("72,69,76,76,79")``
    both give ``"HELLO"``. Empty input gives ``""``. Raises
    :class:`ValueError` for a non-numeric token or a value above 255.
    """
    return _bytes_to_text(bytes(_decimal_values(text)))


def encode_decimal(text: str, *, separator: str = " ") -> str:
    """Write ASCII *text* as decimal codes, e.g. ``"HELLO"`` -> ``"72 69 76 76 79"``."""
    return separator.join(str(value) for value in _text_to_bytes(text))


def _examine_decimal(text: str) -> EncodingGuess | None:
    try:
        values = _decimal_values(text)
    except ValueError:
        return None
    if len(values) < MIN_DECIMAL_VALUES:
        return None
    data = bytes(values)
    ratio = _printable_ratio(data)
    if ratio < MIN_PRINTABLE_RATIO:
        # This is also what rejects A1Z26 letter numbering (values 1..26),
        # which would decode to control characters. See the module docstring.
        return None
    reason = (
        f"{len(values)} numbers, every one of them 0-255, of which {ratio:.0%} "
        "are printable ASCII codes"
    )
    return EncodingGuess(
        format="decimal ASCII",
        confidence=_confidence_for(data),
        reason=reason,
        decoded=_bytes_to_text(data),
    )


def is_decimal(text: str) -> bool:
    """True only when *text* is a list of numbers that decode to printable ASCII."""
    return _examine_decimal(text) is not None


# ---------------------------------------------------------------------------
# Base64
#
# Uses the standard library's `base64` module: part of Python, not a
# third-party package and not a deciphering tool. See the module docstring
# for the compliance note.
# ---------------------------------------------------------------------------

_BASE64_ALPHABET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
)
#: The characters that separate real Base64 from a plain uppercase ciphertext.
_BASE64_EVIDENCE = frozenset("abcdefghijklmnopqrstuvwxyz0123456789+/=")


def _base64_bytes(text: str) -> bytes:
    """Decode Base64 to raw bytes, or raise :class:`ValueError`."""
    compact = "".join(text.split())  # Base64 is routinely wrapped at 76 columns
    if not compact:
        return b""

    body = compact.rstrip("=")
    padding = len(compact) - len(body)
    invalid = sorted({ch for ch in body if ch not in _BASE64_ALPHABET})
    if invalid:
        raise ValueError(
            "not Base64: character(s) "
            + ", ".join(repr(ch) for ch in invalid)
            + " are outside the Base64 alphabet"
        )
    if padding > 2:
        raise ValueError(f"Base64 allows at most two '=' padding characters, got {padding}")
    if len(compact) % 4:
        raise ValueError(
            f"Base64 length must be a multiple of 4 characters, got {len(compact)}"
        )
    try:
        return base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"not valid Base64: {exc}") from exc


def decode_base64(text: str) -> str:
    """Decode standard Base64, tolerating line breaks.

    ``decode_base64("TWFu")`` gives ``"Man"``. Empty input gives ``""``.
    Raises :class:`ValueError` for characters outside the alphabet, a length
    that is not a multiple of four, or malformed padding. URL-safe Base64
    (``-`` and ``_``) is not supported; see the module docstring.
    """
    return _bytes_to_text(_base64_bytes(text))


def encode_base64(text: str) -> str:
    """Write ASCII *text* as standard Base64, e.g. ``"Man"`` -> ``"TWFu"``."""
    return base64.b64encode(_text_to_bytes(text)).decode("ascii")


def _examine_base64(text: str) -> EncodingGuess | None:
    compact = "".join(text.split())
    if len(compact) < MIN_BASE64_CHARS or len(compact) % 4:
        return None
    if not any(ch in _BASE64_EVIDENCE for ch in compact):
        # Uppercase letters only: this is what plain ciphertext looks like, and
        # the Base64 alphabet contains all of it. Refuse rather than guess.
        return None
    try:
        data = _base64_bytes(compact)
    except ValueError:
        return None
    if len(data) < MIN_BASE64_BYTES:
        return None
    # Base64's alphabet overlaps ordinary text completely, so it is held to the
    # strictest product test in the module: every decoded byte must be
    # printable, not merely most of them.
    if _printable_ratio(data) < 1.0:
        return None
    reason = (
        f"{len(compact)} Base64 characters (a multiple of 4), containing "
        "lowercase/digit/padding characters that a plain uppercase ciphertext "
        f"would not, decoding to {len(data)} bytes that are all printable ASCII"
    )
    return EncodingGuess(
        format="Base64",
        confidence=_confidence_for(data),
        reason=reason,
        decoded=_bytes_to_text(data),
    )


def is_base64(text: str) -> bool:
    """True only when *text* is Base64 on the evidence.

    Requires the alphabet, a length that is a multiple of four, at least one
    character a plain uppercase ciphertext could not contain, and a decode
    that is entirely printable ASCII.
    """
    return _examine_base64(text) is not None


# ---------------------------------------------------------------------------
# Morse code
#
# The table is the ITU standard, typed out here rather than imported from
# anywhere. Letters, digits and the punctuation that ITU actually defines.
# ---------------------------------------------------------------------------

_MORSE_PAIRS: tuple[tuple[str, str], ...] = (
    ("A", ".-"), ("B", "-..."), ("C", "-.-."), ("D", "-.."),
    ("E", "."), ("F", "..-."), ("G", "--."), ("H", "...."),
    ("I", ".."), ("J", ".---"), ("K", "-.-"), ("L", ".-.."),
    ("M", "--"), ("N", "-."), ("O", "---"), ("P", ".--."),
    ("Q", "--.-"), ("R", ".-."), ("S", "..."), ("T", "-"),
    ("U", "..-"), ("V", "...-"), ("W", ".--"), ("X", "-..-"),
    ("Y", "-.--"), ("Z", "--.."),
    ("0", "-----"), ("1", ".----"), ("2", "..---"), ("3", "...--"),
    ("4", "....-"), ("5", "....."), ("6", "-...."), ("7", "--..."),
    ("8", "---.."), ("9", "----."),
    (".", ".-.-.-"), (",", "--..--"), ("?", "..--.."), ("'", ".----."),
    ("!", "-.-.--"), ("/", "-..-."), ("(", "-.--."), (")", "-.--.-"),
    ("&", ".-..."), (":", "---..."), (";", "-.-.-."), ("=", "-...-"),
    ("+", ".-.-."), ("-", "-....-"), ("_", "..--.-"), ('"', ".-..-."),
    ("$", "...-..-"), ("@", ".--.-."),
)

#: Character -> dot/dash string.
MORSE_TABLE: dict[str, str] = dict(_MORSE_PAIRS)
#: Dot/dash string -> character.
_MORSE_REVERSE: dict[str, str] = {code: ch for ch, code in _MORSE_PAIRS}

# A duplicated entry would silently make one character undecodable, so check
# the table is one-to-one at import rather than discovering it in a puzzle.
if len(MORSE_TABLE) != len(_MORSE_PAIRS) or len(_MORSE_REVERSE) != len(_MORSE_PAIRS):
    raise ValueError("Morse table is not one-to-one: a character or code was typed twice")

#: Word separator: a literal '/' (optionally spaced) or a run of two or more
#: spaces. The '/' character is itself encodable as "-..-.", so a literal '/'
#: in the stream can only ever be a separator -- there is no ambiguity.
_MORSE_WORD_SPLIT = re.compile(r"\s*/\s*|\s{2,}")


def _morse_normalise(text: str) -> str:
    """Fold the two common ASCII renderings of a dash into one.

    Transcriptions use either ``-`` or ``_`` for the dash. A dash is never
    written as a literal underscore *inside* a code, and the underscore
    character itself is encoded as ``..--.-``, so folding ``_`` to ``-``
    cannot destroy information.

    A line break is folded to a WORD gap rather than a letter gap. Morse is
    conventionally transcribed a line at a time and nobody splits one
    letter's dots across two lines, so a newline almost always falls on a
    word boundary. Folding it to a single space instead would silently run
    the last word of each line into the first word of the next. Two spaces
    is what :data:`_MORSE_WORD_SPLIT` reads as a word separator.
    """
    return (
        text.replace("_", "-")
        .replace("\t", " ")
        .replace("\r\n", "  ")
        .replace("\r", "  ")
        .replace("\n", "  ")
    )


def _morse_tokens(text: str) -> list[list[str]]:
    """Split normalised Morse into words, each a list of dot/dash tokens."""
    cleaned = _morse_normalise(text).strip()
    if not cleaned:
        return []
    words: list[list[str]] = []
    for word in _MORSE_WORD_SPLIT.split(cleaned):
        tokens = [token for token in word.split() if token]
        if tokens:
            words.append(tokens)
    return words


def decode_morse(text: str) -> str:
    """Decode Morse code written with ``.`` and ``-`` (or ``_``).

    Letters are separated by a single space, words by ``/`` or by two or more
    spaces. ``decode_morse(".... . .-.. .-.. --- / .-- --- .-. .-.. -..")``
    gives ``"HELLO WORLD"``. Empty input gives ``""``. Raises
    :class:`ValueError` naming the offending token when a group is not a
    Morse symbol.
    """
    words = _morse_tokens(text)
    if not words:
        return ""
    out: list[str] = []
    for tokens in words:
        letters: list[str] = []
        for token in tokens:
            if any(ch not in ".-" for ch in token):
                raise ValueError(
                    f"not Morse code: {token!r} contains characters other than "
                    "dots and dashes"
                )
            character = _MORSE_REVERSE.get(token)
            if character is None:
                raise ValueError(
                    f"not Morse code: {token!r} is not a Morse symbol "
                    "(no letter, digit or punctuation mark has that pattern)"
                )
            letters.append(character)
        out.append("".join(letters))
    return " ".join(out)


def encode_morse(
    text: str, *, letter_separator: str = " ", word_separator: str = " / "
) -> str:
    """Write *text* as Morse code, e.g. ``"SOS"`` -> ``"... --- ..."``.

    Raises :class:`ValueError` for any character with no Morse symbol.
    """
    words = text.upper().split()
    encoded_words: list[str] = []
    for word in words:
        symbols: list[str] = []
        for character in word:
            code = MORSE_TABLE.get(character)
            if code is None:
                raise ValueError(
                    f"no Morse symbol exists for character {character!r}"
                )
            symbols.append(code)
        encoded_words.append(letter_separator.join(symbols))
    return word_separator.join(encoded_words)


def _examine_morse(text: str) -> EncodingGuess | None:
    cleaned = _morse_normalise(text)
    if not cleaned.strip():
        return None
    # Alphabet test: a Morse transcription contains dots, dashes, word
    # separators and whitespace, and nothing else. One stray letter and it is
    # not Morse, which is what keeps English prose (and its ellipses) out.
    if any(ch not in ".-/ " for ch in cleaned):
        return None
    words = _morse_tokens(text)
    token_count = sum(len(word) for word in words)
    if token_count < MIN_MORSE_TOKENS:
        return None
    try:
        decoded = decode_morse(text)
    except ValueError:
        return None
    data = decoded.encode("ascii", errors="replace")
    reason = (
        f"{token_count} dot/dash groups in {len(words)} word(s), every group a "
        "valid Morse symbol, nothing but dots, dashes and separators present"
    )
    return EncodingGuess(
        format="Morse code",
        confidence=_confidence_for(data),
        reason=reason,
        decoded=decoded,
    )


def is_morse(text: str) -> bool:
    """True only when *text* is nothing but dots, dashes and separators, and
    every group is a real Morse symbol."""
    return _examine_morse(text) is not None


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------

_CONFIDENCE_RANK = {"likely": 0, "possible": 1}


def identify(text: str) -> list[EncodingGuess]:
    """Report every notation *text* passes the evidence tests for.

    Returns an empty list -- the common and correct answer for ciphertext --
    when nothing clears the bar. More than one guess is possible and honest:
    ``"72 69 76 76 79"`` really is both a valid decimal ASCII message and a
    valid hexadecimal one once the spaces are removed, and the caller is told
    so rather than being sold whichever the code happened to test first.

    Guesses are ordered by confidence, then by :data:`FORMAT_NAMES`.
    """
    if not text or not text.strip():
        return []
    found: list[EncodingGuess] = [
        guess
        for guess in (
            _examine_hex(text),
            _examine_binary(text),
            _examine_decimal(text),
            _examine_base64(text),
            _examine_morse(text),
        )
        if guess is not None
    ]
    found.sort(
        key=lambda guess: (
            _CONFIDENCE_RANK[guess.confidence],
            FORMAT_NAMES.index(guess.format),
        )
    )
    return found


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    include_possible: bool = True,
    time_budget: float | None = None,
    seed: int | None = None,
    **options: object,
) -> CandidateSet:
    """Identify the notation and decode it, returning ranked candidates.

    "Solve" is the wrong word and the candidates say so: each one is named
    ``"Encoding: <format> (NOT encryption)"`` and carries
    :data:`NOT_ENCRYPTION` in its diagnostics. Ranking is by English score, so
    when a text parses as two notations the one that produces English comes
    first -- but both are returned.

    Parameters
    ----------
    source:
        The text, as a string or a :class:`NormalizedText`.
    scorer:
        English scorer; the shared default is used when ``None``.
    top:
        Keep at most this many candidates.
    include_possible:
        When ``False``, only ``"likely"`` guesses are turned into candidates.
    time_budget, seed:
        Accepted so that every solver in the toolkit has one signature. This
        search is exhaustive (five fixed tests) and deterministic, so neither
        has any effect and no time budget can be hit.
    """
    if options:
        raise ValueError(
            "unknown option(s) for encodings.solve: " + ", ".join(sorted(options))
        )

    engine = scorer if scorer is not None else default_scorer()
    normalised = source if isinstance(source, NormalizedText) else normalize(source)
    # Encodings live in the digits, spacing and punctuation that `letters`
    # throws away -- "01001000" has no letters at all -- so this is one of the
    # few places in the toolkit that must read the original text.
    raw = normalised.original

    results = CandidateSet()
    for guess in identify(raw):
        if guess.decoded is None or not guess.decoded:
            continue
        if not include_possible and guess.confidence != "likely":
            continue
        plaintext = guess.decoded
        diagnostics: dict[str, object] = {
            "not_encryption": True,
            "warning": NOT_ENCRYPTION,
            "detector_confidence": guess.confidence,
            "evidence": guess.reason,
            "decoded_characters": len(plaintext),
        }
        annotate(diagnostics, plaintext, engine)
        results.add(
            Candidate(
                method=f"Encoding: {guess.format} (NOT encryption)",
                key=f"format={guess.format}",
                score=engine.score(plaintext),
                plaintext=plaintext,
                diagnostics=diagnostics,
                # No `display`: the decoded text is a different length from the
                # input, so it cannot be poured back into the input's layout.
                display=None,
            )
        )

    if top > 0 and len(results) > top:
        return CandidateSet(results.top(top))
    return results


def describe_guesses(guesses: Iterable[EncodingGuess]) -> str:
    """Render :func:`identify` output as plain lines, warning and all."""
    lines = [guess.describe() for guess in guesses]
    if not lines:
        return "No encoding was identified. Treat the text as ciphertext."
    lines.append(f"NOTE: {NOT_ENCRYPTION}")
    return "\n".join(lines)
