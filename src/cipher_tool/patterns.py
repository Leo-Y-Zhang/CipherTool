"""Word pattern signatures for monoalphabetic analysis.

A *pattern signature* records the shape of repeated letters in a word while
throwing away the letters themselves:

    HELLO   -> 0-1-2-2-3
    PEOPLE  -> 0-1-2-0-3-1
    ATTACK  -> 0-1-1-0-2-3

Why this works
--------------
A monoalphabetic substitution replaces each letter consistently, so it cannot
change the *shape* of a word. Whatever ``HELLO`` becomes, it is still five
letters with the third and fourth identical and every other letter distinct.
So if we know a ciphertext token is a whole plaintext word, the plaintext must
be an English word with the same signature. That usually cuts the candidate
list from thousands of words to a handful, and each surviving word supplies a
set of letter equations we can test for consistency.

Important caveat, enforced by the CLI
-------------------------------------
This only helps when the ciphertext genuinely preserves word divisions. The
National Cipher Challenge normally publishes ciphertext in five-letter groups,
where the spacing means nothing. Every entry point here therefore takes the
candidate words explicitly; nothing in this module ever splits ciphertext on
whitespace by itself.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

SEPARATOR = "-"


def pattern_signature(word: str, separator: str = SEPARATOR) -> str:
    """Return the repeated-letter signature of *word*.

    The first distinct letter is numbered 0, the next new letter 1, and so on,
    so the signature depends only on the pattern of repeats.

    >>> pattern_signature("HELLO")
    '0-1-2-2-3'
    >>> pattern_signature("hello") == pattern_signature("QIVVA")
    True

    A multi-character separator is used because words longer than ten distinct
    letters would otherwise be ambiguous ("0110" could be 0,1,1,0 or 0,11,0).
    """
    seen: dict[str, int] = {}
    codes: list[str] = []
    for char in word.upper():
        if char not in seen:
            seen[char] = len(seen)
        codes.append(str(seen[char]))
    return separator.join(codes)


def has_repeat(word: str) -> bool:
    """True if *word* contains any repeated letter.

    Words with no repeats have an uninformative signature (0-1-2-3-...) that
    matches every other word of the same length, so callers usually skip them.
    """
    upper = word.upper()
    return len(set(upper)) != len(upper)


def signature_selectivity(signature: str) -> int:
    """How many letter positions the signature actually constrains.

    A signature with *n* symbols and *d* distinct codes pins down ``n - d``
    equalities. Higher is more selective, so callers can prefer the most
    informative ciphertext words first.
    """
    codes = signature.split(SEPARATOR)
    return len(codes) - len(set(codes))


class PatternIndex:
    """A dictionary indexed by pattern signature.

    Built once from a word list and then queried per ciphertext token. Lookup
    is a dictionary hit, so testing hundreds of tokens against a word list of
    tens of thousands of entries is instant.
    """

    def __init__(self, words: Iterable[str]) -> None:
        self._by_signature: dict[str, list[str]] = defaultdict(list)
        self._words: set[str] = set()
        for raw in words:
            word = "".join(c for c in raw.upper() if "A" <= c <= "Z")
            if not word or word in self._words:
                continue
            self._words.add(word)
            self._by_signature[pattern_signature(word)].append(word)
        # Sort each bucket so results are deterministic run to run.
        for bucket in self._by_signature.values():
            bucket.sort()

    def __len__(self) -> int:
        return len(self._words)

    def __contains__(self, word: str) -> bool:
        return word.upper() in self._words

    @property
    def words(self) -> set[str]:
        """Every distinct word held in the index."""
        return self._words

    def matches(self, cipher_word: str, limit: int | None = None) -> list[str]:
        """English words whose letter pattern matches *cipher_word*.

        Returns candidates only; it makes no claim that any of them is right.
        """
        cleaned = "".join(c for c in cipher_word.upper() if "A" <= c <= "Z")
        if not cleaned:
            return []
        found = self._by_signature.get(pattern_signature(cleaned), [])
        return found[:limit] if limit else list(found)

    def signature_counts(self) -> dict[str, int]:
        """How many words share each signature (diagnostics/reporting)."""
        return {sig: len(words) for sig, words in self._by_signature.items()}


# ---------------------------------------------------------------------------
# Turning a pattern match into letter equations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatternMatch:
    """One hypothesis: *cipher_word* decrypts to *plain_word*.

    ``mapping`` is the implied partial substitution key, cipher letter ->
    plain letter. ``constrains`` is how many distinct cipher letters it fixes.
    """

    cipher_word: str
    plain_word: str
    mapping: Mapping[str, str]

    @property
    def constrains(self) -> int:
        """How many distinct cipher letters this match would pin down."""
        return len(self.mapping)


def mapping_from_pair(cipher_word: str, plain_word: str) -> dict[str, str] | None:
    """Derive the cipher->plain mapping implied by a word pair.

    Returns ``None`` when the pair cannot come from a monoalphabetic cipher,
    which happens in three ways:

    1. different lengths;
    2. one cipher letter would have to mean two different plain letters;
    3. two different cipher letters would have to mean the same plain letter
       (a substitution alphabet is a bijection, so this is impossible too).

    Rejecting case 3 is what makes this useful: it prunes most spurious
    signature matches that a naive check would accept.
    """
    if len(cipher_word) != len(plain_word):
        return None

    forward: dict[str, str] = {}
    backward: dict[str, str] = {}
    for cipher_char, plain_char in zip(cipher_word.upper(), plain_word.upper()):
        if forward.setdefault(cipher_char, plain_char) != plain_char:
            return None
        if backward.setdefault(plain_char, cipher_char) != cipher_char:
            return None
    return forward


def match_word(
    cipher_word: str,
    index: PatternIndex,
    *,
    known: Mapping[str, str] | None = None,
    limit: int | None = None,
) -> list[PatternMatch]:
    """All English words consistent with *cipher_word* and *known* so far.

    *known* is a partial cipher->plain mapping already believed. A candidate
    English word is kept only if it agrees with every letter already decided,
    in both directions, so the list shrinks as the solve progresses.
    """
    cleaned = "".join(c for c in cipher_word.upper() if "A" <= c <= "Z")
    if not cleaned:
        return []

    known_forward = {k.upper(): v.upper() for k, v in (known or {}).items()}
    known_backward = {v: k for k, v in known_forward.items()}

    results: list[PatternMatch] = []
    for word in index.matches(cleaned):
        mapping = mapping_from_pair(cleaned, word)
        if mapping is None:
            continue
        if not _agrees(mapping, known_forward, known_backward):
            continue
        results.append(PatternMatch(cleaned, word, mapping))
        if limit and len(results) >= limit:
            break
    return results


def _agrees(
    mapping: Mapping[str, str],
    known_forward: Mapping[str, str],
    known_backward: Mapping[str, str],
) -> bool:
    """True if *mapping* contradicts nothing already fixed."""
    for cipher_char, plain_char in mapping.items():
        if known_forward.get(cipher_char, plain_char) != plain_char:
            return False
        if known_backward.get(plain_char, cipher_char) != cipher_char:
            return False
    return True


def rank_by_selectivity(cipher_words: Sequence[str]) -> list[str]:
    """Order ciphertext tokens most-informative first.

    Long words with several repeated letters constrain the key hardest, so a
    human working by hand should attack them first. Ties break on length and
    then alphabetically so the output is stable.
    """
    unique = {
        cleaned
        for word in cipher_words
        if (cleaned := "".join(c for c in word.upper() if "A" <= c <= "Z"))
    }
    return sorted(
        unique,
        key=lambda w: (-signature_selectivity(pattern_signature(w)), -len(w), w),
    )
