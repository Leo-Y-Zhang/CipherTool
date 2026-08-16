"""General monoalphabetic substitution: the cipher, and the attack on it.

The cipher
----------
A monoalphabetic substitution replaces every occurrence of one plaintext
letter with one fixed ciphertext letter, and never changes that choice. The
key is therefore a *bijection* of the 26-letter alphabet onto itself: one
plain letter in, one cipher letter out, and no two plain letters share a
cipher letter. There are 26! = 403,291,461,126,605,635,584,000,000 such keys,
which is why the cipher cannot be broken by trying them all, and why it is
nevertheless broken in seconds by the attack below.

Caesar, Atbash, affine and keyword ciphers are all special cases: they are
substitutions whose key happens to be describable in a few characters. This
module knows nothing about that structure and searches the whole space, so it
solves those too (less efficiently than their own dedicated solvers).

What the cipher does NOT change
-------------------------------
Because a substitution only relabels letters, it preserves:

* the Index of Coincidence (see ``statistics.py``) -- an English-like IC with
  a scrambled letter distribution is the fingerprint of this family;
* the shape of every word, which is what ``patterns.py`` exploits;
* the position of every letter, which is why a recovered plaintext can be
  poured straight back into the original layout with ``relayout``.

The attack: hill climbing with restarts
---------------------------------------
Searching 26! keys directly is hopeless, but the space is not flat. A key
that is mostly right produces text that is mostly English, and the English
score (see ``scoring.py``) rises smoothly as the key improves. That makes
local search work:

1. Start from a key -- either the frequency guess (commonest ciphertext
   letter stands for E, next for T, and so on down ETAOIN SHRDLU) or a random
   bijection.
2. Consider swapping the plain letters currently assigned to two cipher
   letters. There are 26 * 25 / 2 = 325 such swaps. Score the decryption that
   each swap would produce; keep the swap if the score rises, otherwise undo
   it and move on.
3. When a complete pass over all 325 pairs improves nothing, no single swap
   helps: the key is a *local* optimum. Record it.
4. Restart from a fresh random key and climb again.

Step 4 is the important one. A single climb often stalls on a key with two or
three letters transposed, because escaping needs two simultaneous swaps and
the search only makes one at a time. Different starting points stall in
different places, so we run many climbs and compare the results. The evidence
that the search converged is not the score of the winner -- a hill climber
always returns *something* -- but **how many independent restarts arrived at
the same answer**. That count is reported as ``agreements`` on every
candidate, and a solve where the best key was found once out of twenty-five
deserves far less trust than one where it was found twenty times.

Why the score is a quadgram model
---------------------------------
Single-letter frequencies are not enough: many wrong keys reproduce English
letter frequencies almost exactly (they only have to permute the rare letters
among themselves). Four-letter statistics are much harder to fake, because
they encode which letters may follow which. We therefore climb on the sum of
log10 P(d | abc) over every four-letter window, taken from
``EnglishScorer.table()``.

Incremental scoring
-------------------
See ``_HillClimber``. Rescoring the whole text for each of the 325 candidate
swaps would make the search hundreds of times slower than it needs to be, and
almost all of that work would be recomputing windows that cannot have
changed.

Honest limits
-------------
* Below roughly 150 letters the quadgram signal is weak and the climb will
  confidently return a wrong key. ``solve`` records a warning in the
  diagnostics rather than hiding it.
* The climber optimises an English-likeness score, not correctness. A key
  that scores well on gibberish is still gibberish; read the plaintext.
* Nothing in this module ever splits ciphertext on whitespace to find words.
  Competition ciphertext is printed in five-letter groups and its spacing
  carries no information about the plaintext.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from typing import Iterable, Mapping, Sequence

from .candidates import Candidate, CandidateSet
from .normalize import (
    ALPHABET,
    ALPHABET_SIZE,
    NormalizedText,
    letters_only,
    normalize,
)
from .patterns import PatternIndex, mapping_from_pair, match_word
from .reference import ENGLISH_FREQUENCY_ORDER
from .scoring import EnglishScorer, annotate, default_scorer

__all__ = [
    "SubstitutionKey",
    "METHOD",
    "encrypt",
    "decrypt",
    "solve",
    "frequency_guess",
    "window_score",
    "apply_crib",
    "crib_positions",
    "analyse_words",
]

#: The method name attached to every candidate this module produces.
METHOD = "Monoalphabetic substitution (hill climbing)"

#: Default character shown by :meth:`SubstitutionKey.apply` for a cipher
#: letter the key says nothing about.
PLACEHOLDER = "."

#: Below this many letters the quadgram statistics are too thin for a hill
#: climb to be trusted. Measured, not guessed: see the module docstring and
#: tests/test_substitution.py.
RELIABLE_CLIMB_LETTERS = 150

#: A quadgram window is four letters wide, so a text shorter than this has no
#: windows at all and there is nothing for the climber to optimise.
WINDOW = 4

#: Hard ceiling on the passes one climb may make.
#:
#: A correct climb cannot reach this. Every accepted swap strictly increases
#: the score, the score is bounded above, and there are finitely many keys,
#: so the loop has to stop by itself; on competition-length text it settles
#: in well under ten passes. The cap exists because that termination argument
#: depends on the incremental arithmetic being right, and a search that never
#: returns is worse on competition night than one that returns a poor answer
#: and says so. Reaching it is recorded in the diagnostics as a bug signal.
MAX_CLIMB_PASSES = 500


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _letters_of(source: str | NormalizedText) -> str:
    """Uppercase A-Z view of *source*, accepting either supported input type."""
    if isinstance(source, NormalizedText):
        return source.letters
    if isinstance(source, str):
        return letters_only(source)
    raise ValueError(
        "expected a string or a NormalizedText, got " + type(source).__name__
    )


def _single_letter(value: object, role: str) -> str:
    """Clean *value* to exactly one A-Z letter or explain why it is not one."""
    cleaned = letters_only(str(value))
    if len(cleaned) != 1:
        raise ValueError(
            f"{role} must be a single letter A-Z, got {value!r}"
        )
    return cleaned


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


class SubstitutionKey:
    """A partial or complete substitution alphabet, cipher letter -> plain.

    The direction matters and is easy to get backwards, so it is stated once
    here and repeated on every method that could be misread: **a
    SubstitutionKey maps CIPHERTEXT letters to PLAINTEXT letters**, because
    that is the direction a solver works in and the direction a human fills in
    while breaking a message by hand. :meth:`inverse` gives the enciphering
    direction.

    A key may be partial. That is the normal state during a manual solve: you
    know that Q stands for E and X stands for T, and nothing else yet.
    :meth:`apply` renders the letters you have not decided as a placeholder so
    the shape of the message becomes readable as you go.

    Instances are immutable. :meth:`with_pair` returns a new key rather than
    modifying this one, so a search can hold onto a key without worrying that
    something else will change it underneath.

    Consistency is enforced at construction. A substitution alphabet is a
    bijection, so BOTH of these are rejected:

    * one cipher letter standing for two different plain letters;
    * two different cipher letters standing for the same plain letter.

    The second is the one people forget, and rejecting it is what makes
    pattern matching and crib fitting worth doing at all -- most wrong guesses
    die on it.
    """

    __slots__ = ("_mapping", "_reverse")

    def __init__(self, mapping: Mapping[str, str] | None = None) -> None:
        """Build a key from a cipher letter -> plain letter mapping.

        Raises ``ValueError`` naming the offending letters if the mapping is
        not a bijection, or if any key or value is not a single letter.
        """
        forward: dict[str, str] = {}
        reverse: dict[str, str] = {}
        for raw_cipher, raw_plain in dict(mapping or {}).items():
            cipher = _single_letter(raw_cipher, "cipher letter")
            plain = _single_letter(raw_plain, "plain letter")
            existing = forward.get(cipher)
            if existing is not None and existing != plain:
                raise ValueError(
                    f"inconsistent key: cipher letter {cipher} is already "
                    f"mapped to {existing}, so it cannot also mean {plain}"
                )
            owner = reverse.get(plain)
            if owner is not None and owner != cipher:
                raise ValueError(
                    f"inconsistent key: plain letter {plain} is already the "
                    f"meaning of cipher letter {owner}, so cipher letter "
                    f"{cipher} cannot mean it too (a substitution alphabet is "
                    "a bijection)"
                )
            forward[cipher] = plain
            reverse[plain] = cipher
        self._mapping = dict(sorted(forward.items()))
        self._reverse = dict(sorted(reverse.items()))

    # -- construction ------------------------------------------------------

    @classmethod
    def from_pairs(cls, pairs: Iterable[object]) -> SubstitutionKey:
        """Build a key from an iterable of cipher/plain pairs.

        Each item may be a two-element sequence ``("Q", "E")`` or a two-letter
        string ``"QE"``, read as "cipher Q stands for plain E".

        >>> SubstitutionKey.from_pairs(["QE", ("X", "T")]).apply("QX")
        'ET'
        """
        mapping: dict[str, str] = {}
        for item in pairs:
            if isinstance(item, str):
                cleaned = letters_only(item)
                if len(cleaned) != 2:
                    raise ValueError(
                        f"pair {item!r} must be exactly two letters, "
                        "cipher letter then plain letter (for example 'QE')"
                    )
                cipher, plain = cleaned[0], cleaned[1]
            else:
                try:
                    cipher_raw, plain_raw = tuple(item)  # type: ignore[misc]
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"pair {item!r} must be a two-letter string or a "
                        "(cipher, plain) pair"
                    ) from error
                cipher = _single_letter(cipher_raw, "cipher letter")
                plain = _single_letter(plain_raw, "plain letter")
            if cipher in mapping and mapping[cipher] != plain:
                raise ValueError(
                    f"inconsistent key: cipher letter {cipher} is given twice, "
                    f"as {mapping[cipher]} and as {plain}"
                )
            mapping[cipher] = plain
        return cls(mapping)

    @classmethod
    def from_string(cls, text: str) -> SubstitutionKey:
        """Parse a human-typed list of pairs such as ``"QE XT"``.

        Pairs are separated by whitespace or commas. Within a pair any
        non-letter is ignored, so ``"Q=E, X->T"`` and ``"QE XT"`` mean the
        same thing: cipher Q stands for plain E, cipher X stands for plain T.

        An empty string gives an empty key, which is a legitimate starting
        point rather than an error. A token that is not exactly two letters is
        an error, and says so -- in particular a full 26-letter alphabet
        belongs in :meth:`from_alphabet`, not here.
        """
        if not isinstance(text, str):
            raise ValueError("from_string needs a string of letter pairs")
        tokens = [token for token in text.replace(",", " ").split() if token]
        pairs: list[str] = []
        for token in tokens:
            cleaned = letters_only(token)
            if len(cleaned) == 26:
                raise ValueError(
                    f"{token!r} looks like a full 26-letter alphabet; use "
                    "SubstitutionKey.from_alphabet for that, or separate the "
                    "pairs with spaces (for example 'QE XT')"
                )
            if len(cleaned) != 2:
                raise ValueError(
                    f"{token!r} is not a cipher/plain pair: expected exactly "
                    "two letters such as 'QE', found "
                    f"{len(cleaned)} letters"
                )
            pairs.append(cleaned)
        return cls.from_pairs(pairs)

    @classmethod
    def from_alphabet(cls, alphabet: str) -> SubstitutionKey:
        """Build a complete key from a 26-letter DECRYPTION alphabet.

        ``alphabet[i]`` is the plaintext letter that ciphertext letter
        ``ALPHABET[i]`` stands for. So ``from_alphabet("GHAIE...")`` means
        cipher A decrypts to G, cipher B decrypts to H, cipher C decrypts to A,
        and so on.

        This is the direction ``solve`` reports, so the ``key=...`` string
        printed for a candidate can be pasted straight back in here.
        Use :meth:`from_encipher_alphabet` for the other convention.
        """
        cleaned = letters_only(alphabet)
        if len(cleaned) != ALPHABET_SIZE:
            raise ValueError(
                f"a substitution alphabet needs exactly {ALPHABET_SIZE} "
                f"letters, got {len(cleaned)} from {alphabet!r}"
            )
        missing = sorted(set(ALPHABET) - set(cleaned))
        if missing:
            duplicated = sorted(
                letter for letter, count in Counter(cleaned).items() if count > 1
            )
            raise ValueError(
                "a substitution alphabet must use every letter exactly once; "
                f"{''.join(duplicated)} repeated, {''.join(missing)} missing"
            )
        return cls(dict(zip(ALPHABET, cleaned)))

    @classmethod
    def from_encipher_alphabet(cls, alphabet: str) -> SubstitutionKey:
        """Build a complete key from a 26-letter ENCIPHERING alphabet.

        ``alphabet[i]`` is the ciphertext letter that plaintext letter
        ``ALPHABET[i]`` becomes -- the convention used when a keyword-mixed
        alphabet is written out, for example ``"CIPHERABDFGJKLMNOQSTUVWXYZ"``
        for the keyword CIPHER. The result is still a cipher -> plain key;
        only the reading of the argument differs.
        """
        return cls.from_alphabet(alphabet).inverse()

    # -- container protocol ------------------------------------------------

    def __len__(self) -> int:
        """How many cipher letters this key decides."""
        return len(self._mapping)

    def __contains__(self, cipher: str) -> bool:
        return letters_only(str(cipher)) in self._mapping

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubstitutionKey):
            return NotImplemented
        return self._mapping == other._mapping

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._mapping.items())))

    def __repr__(self) -> str:
        return f"SubstitutionKey.from_alphabet({self.to_alphabet()!r})"

    # -- reading -----------------------------------------------------------

    @property
    def mapping(self) -> dict[str, str]:
        """A copy of the cipher -> plain mapping, sorted by cipher letter."""
        return dict(self._mapping)

    @property
    def is_complete(self) -> bool:
        """True when all 26 cipher letters are decided."""
        return len(self._mapping) == ALPHABET_SIZE

    @property
    def undecided_cipher_letters(self) -> tuple[str, ...]:
        """Cipher letters this key says nothing about, alphabetically."""
        return tuple(letter for letter in ALPHABET if letter not in self._mapping)

    @property
    def unused_plain_letters(self) -> tuple[str, ...]:
        """Plain letters not yet claimed by any cipher letter."""
        return tuple(letter for letter in ALPHABET if letter not in self._reverse)

    def get(self, cipher: str, default: str | None = None) -> str | None:
        """The plain letter *cipher* stands for, or *default* if undecided."""
        return self._mapping.get(letters_only(str(cipher)), default)

    def inverse(self) -> SubstitutionKey:
        """The plain -> cipher key, i.e. the enciphering direction.

        A partial key inverts to an equally partial key: only the pairs that
        were decided appear, which is correct -- an undecided letter is
        undecided in both directions.
        """
        return SubstitutionKey({plain: cipher for cipher, plain in self._mapping.items()})

    def with_pair(self, cipher: str, plain: str) -> SubstitutionKey:
        """A NEW key with one more pair decided.

        Raises ``ValueError`` naming the conflicting letters if the pair
        contradicts what this key already says, in either direction. Adding a
        pair the key already contains is not a contradiction and simply
        returns an equal key.
        """
        cipher_letter = _single_letter(cipher, "cipher letter")
        plain_letter = _single_letter(plain, "plain letter")
        existing = self._mapping.get(cipher_letter)
        if existing is not None and existing != plain_letter:
            raise ValueError(
                f"cannot map cipher letter {cipher_letter} to {plain_letter}: "
                f"it already stands for {existing}"
            )
        owner = self._reverse.get(plain_letter)
        if owner is not None and owner != cipher_letter:
            raise ValueError(
                f"cannot map cipher letter {cipher_letter} to {plain_letter}: "
                f"cipher letter {owner} already stands for {plain_letter}, and "
                "a substitution alphabet is a bijection"
            )
        merged = dict(self._mapping)
        merged[cipher_letter] = plain_letter
        return SubstitutionKey(merged)

    def apply(
        self, text: str | NormalizedText, placeholder: str = PLACEHOLDER
    ) -> str:
        """Decrypt *text* as far as this key allows, letters only.

        Cipher letters the key has not decided become *placeholder*, which
        must be a single non-letter character: rendering an unknown letter as
        a letter would quietly turn a guess into a claim.
        """
        if not isinstance(placeholder, str) or len(placeholder) != 1:
            raise ValueError(
                f"placeholder must be a single character, got {placeholder!r}"
            )
        if "A" <= placeholder.upper() <= "Z":
            raise ValueError(
                f"placeholder {placeholder!r} is a letter; use a symbol such "
                "as '.' so undecided letters cannot be mistaken for solved ones"
            )
        mapping = self._mapping
        return "".join(mapping.get(char, placeholder) for char in _letters_of(text))

    def to_alphabet(self, placeholder: str = PLACEHOLDER) -> str:
        """The 26-character decryption alphabet, undecided letters as dots.

        The inverse of :meth:`from_alphabet` for a complete key.
        """
        return self.apply(ALPHABET, placeholder=placeholder)

    def describe(self, placeholder: str = PLACEHOLDER) -> str:
        """Two aligned lines, the way a solver writes a key out by hand."""
        return (
            f"cipher: {ALPHABET}\n"
            f"plain : {self.to_alphabet(placeholder)}"
        )


# ---------------------------------------------------------------------------
# Key coercion for the public functions
# ---------------------------------------------------------------------------


def _coerce_key(
    key: SubstitutionKey | str | Mapping[str, str], role: str
) -> SubstitutionKey:
    """Accept the several shapes a caller may reasonably supply for a key.

    A string of exactly 26 letters is read as a full decryption alphabet; any
    other string is read as space-separated pairs (``"QE XT"``). The two
    cannot be confused because a pair list of 26 letters would be 13 pairs and
    is spelled with spaces.
    """
    if isinstance(key, SubstitutionKey):
        return key
    if isinstance(key, str):
        if len(letters_only(key)) == ALPHABET_SIZE and not key.strip().count(" "):
            return SubstitutionKey.from_alphabet(key)
        return SubstitutionKey.from_string(key)
    if isinstance(key, Mapping):
        return SubstitutionKey(key)
    raise ValueError(
        f"{role} must be a SubstitutionKey, a 26-letter alphabet, a pair "
        f"string such as 'QE XT', or a mapping; got {type(key).__name__}"
    )


def _require_complete(key: SubstitutionKey, role: str) -> SubstitutionKey:
    """Insist on a complete key, explaining what to do with a partial one."""
    if not key.is_complete:
        missing = "".join(key.undecided_cipher_letters)
        raise ValueError(
            f"{role} needs a complete 26-letter key; these cipher letters are "
            f"still undecided: {missing}. Use SubstitutionKey.apply() to see a "
            "partial decryption instead."
        )
    return key


# ---------------------------------------------------------------------------
# Encryption and decryption
# ---------------------------------------------------------------------------


def decrypt(
    text: str | NormalizedText, key: SubstitutionKey | str | Mapping[str, str]
) -> str:
    """Decrypt with a complete key. Operates on letters only, returns letters.

    *key* maps cipher letters to plain letters, so a 26-letter string argument
    is read as a decryption alphabet: ``decrypt(text, "GHAIE...")`` treats
    ciphertext A as plaintext G.
    """
    return _require_complete(_coerce_key(key, "decrypt"), "decrypt").apply(text)


def encrypt(
    text: str | NormalizedText, key: SubstitutionKey | str | Mapping[str, str]
) -> str:
    """Encrypt with a complete key. Exact inverse of :func:`decrypt`.

    The key is still a cipher -> plain map, so encryption applies its inverse:
    with ``"GHAIE..."``, ciphertext A means plaintext G, therefore plaintext G
    enciphers to A.
    """
    resolved = _require_complete(_coerce_key(key, "encrypt"), "encrypt")
    return resolved.inverse().apply(text)


# ---------------------------------------------------------------------------
# Frequency analysis
# ---------------------------------------------------------------------------


def frequency_guess(source: str | NormalizedText) -> SubstitutionKey:
    """Rank the ciphertext letters by frequency and read off ETAOIN SHRDLU.

    The commonest ciphertext letter is guessed to be E, the next T, the next
    A, and so on down :data:`reference.ENGLISH_FREQUENCY_ORDER`.

    This is a *starting point for a search and nothing more*. On a few hundred
    letters it typically gets three or four of the top letters right and the
    rest wrong, because the frequency gaps between (for instance) H, R and D
    are smaller than the sampling noise. Ties are broken alphabetically so the
    result is reproducible.

    Cipher letters that do not occur in the text are left undecided: there is
    no evidence about them, so the key does not pretend to have any.
    """
    letters = _letters_of(source)
    counts = Counter(letters)
    ordered = sorted(counts, key=lambda letter: (-counts[letter], letter))
    return SubstitutionKey(dict(zip(ordered, ENGLISH_FREQUENCY_ORDER)))


# ---------------------------------------------------------------------------
# The objective function
# ---------------------------------------------------------------------------


def _window_total(
    values: Sequence[int], table: Sequence[float], plain_of: Sequence[int]
) -> float:
    """Sum log10 P(d | abc) over every four-letter window of the decryption.

    *values* are the ciphertext letters as 0..25, *plain_of* maps a cipher
    letter index to its plain letter index, and *table* is the flat order-3
    table from :meth:`EnglishScorer.table`, indexed
    ``((a * 26 + b) * 26 + c) * 26 + d``.

    This is the quantity the hill climber maximises. It deliberately ignores
    the first three letters (which have no full context): they contribute a
    handful of terms out of hundreds and leaving them out keeps the
    incremental arithmetic exact.
    """
    decoded = [plain_of[value] for value in values]
    total = 0.0
    for start in range(len(decoded) - WINDOW + 1):
        total += table[
            ((decoded[start] * 26 + decoded[start + 1]) * 26
             + decoded[start + 2]) * 26 + decoded[start + 3]
        ]
    return total


def window_score(
    text: str | NormalizedText,
    key: SubstitutionKey | str | Mapping[str, str],
    *,
    scorer: EnglishScorer | None = None,
) -> float:
    """The quadgram-window score of decrypting *text* with *key*.

    Exposed because it is the exact objective :func:`solve` optimises, so a
    human comparing two keys by hand can compare them on the same footing the
    search used. It is NOT the same number as ``scorer.score(plaintext)``,
    which also scores the first three letters and is what a candidate reports.
    """
    engine = scorer or default_scorer()
    resolved = _require_complete(_coerce_key(key, "window_score"), "window_score")
    values = engine.encode(_letters_of(text))
    plain_of = _plain_indices(resolved.mapping)
    return _window_total(values, engine.table(), plain_of)


def _plain_indices(mapping: Mapping[str, str]) -> list[int]:
    """Turn a complete cipher -> plain mapping into a list of 26 indices."""
    return [ord(mapping[letter]) - 65 for letter in ALPHABET]


# ---------------------------------------------------------------------------
# Hill climbing
# ---------------------------------------------------------------------------


class _HillClimber:
    """One text, prepared once, climbed many times.

    Incremental scoring
    -------------------
    A full rescore costs one table lookup per four-letter window, so a single
    pass over the 325 candidate swaps would cost 325 * (n - 3) lookups, and a
    solve makes hundreds of passes. Nearly all of that is wasted work.

    Swapping the plain letters assigned to cipher letters X and Y changes the
    decryption ONLY at positions holding an X or a Y. A window starting at p
    covers positions p, p+1, p+2 and p+3, so a position i can only affect
    windows starting in the range [i - 3, i]. We therefore precompute, for
    each cipher letter, the set of window starts it touches, and for each pair
    of letters the union of their two sets. Scoring a swap then costs
    |union| lookups instead of n - 3.

    For a 500-letter text an average letter occurs about 19 times and touches
    about 4 * 19 = 76 windows, so a pair union is roughly 150 windows against
    497 for a full rescore -- and much better than that for the rare letters
    which most of the 325 swaps involve.

    The "before" figure costs nothing at all: a running per-window score array
    is kept in step with the current key, so the old score of the affected
    windows is a sum of cached floats rather than a rescore. Only the "after"
    figure needs table lookups, and only over the affected windows.

    The running total is therefore maintained by addition of deltas. That is
    the classic place for a silent bug, so ``full_score`` recomputes it from
    scratch and tests/test_substitution.py asserts the two agree.
    """

    def __init__(
        self,
        values: Sequence[int],
        table: Sequence[float],
        free: Sequence[int],
    ) -> None:
        self.values = list(values)
        self.table = table
        self.count = len(self.values)
        #: Index of the last valid window start; -1 when the text is shorter
        #: than one window, in which case there is nothing to optimise.
        self.last_window = self.count - WINDOW
        self.free = sorted(set(free))
        self.swap_tests = 0
        self.passes = 0
        self.pass_limit_hits = 0

        self.positions_of: list[list[int]] = [[] for _ in range(ALPHABET_SIZE)]
        for index, value in enumerate(self.values):
            self.positions_of[value].append(index)

        # A text shorter than one window has nothing to score, so it has no
        # swaps worth considering either. Saying so here keeps `pairs` and
        # `pair_windows` in step: every pair the climber will try has an entry.
        self.pairs: list[tuple[int, int]] = (
            []
            if self.last_window < 0
            else [
                (first, second)
                for offset, first in enumerate(self.free)
                for second in self.free[offset + 1:]
            ]
        )

        self.pair_windows: dict[tuple[int, int], list[int]] = {}
        if self.pairs:
            touched: list[set[int]] = [set() for _ in range(ALPHABET_SIZE)]
            for index, value in enumerate(self.values):
                low = index - (WINDOW - 1)
                if low < 0:
                    low = 0
                high = index if index < self.last_window else self.last_window
                touched[value].update(range(low, high + 1))
            for pair in self.pairs:
                self.pair_windows[pair] = sorted(touched[pair[0]] | touched[pair[1]])

    # -- scoring -----------------------------------------------------------

    def full_score(self, plain_of: Sequence[int]) -> float:
        """Independent recomputation of the objective, used to audit the climb."""
        return _window_total(self.values, self.table, plain_of)

    # -- the climb ---------------------------------------------------------

    def climb(
        self,
        plain_of: Sequence[int],
        deadline: float | None = None,
        max_passes: int = MAX_CLIMB_PASSES,
    ) -> tuple[list[int], float, bool]:
        """Climb to a local optimum from *plain_of*.

        Returns the resulting key (as 26 plain-letter indices), its window
        score, and whether the climb was cut short by *deadline* (a
        ``time.monotonic()`` reading) rather than reaching a local optimum.

        *max_passes* is a safety net, not a tuning knob: see
        :data:`MAX_CLIMB_PASSES`. Reaching it increments
        ``pass_limit_hits`` and stops the climb.
        """
        current = list(plain_of)
        table = self.table
        positions_of = self.positions_of
        pair_windows = self.pair_windows

        # Per-position decryption and per-window score, kept in step with
        # `current` at all times. Every swap updates both or reverts both.
        decoded = [current[value] for value in self.values]
        window_count = self.last_window + 1
        window_score_of = [0.0] * (window_count if window_count > 0 else 0)
        for start in range(window_count):
            window_score_of[start] = table[
                ((decoded[start] * 26 + decoded[start + 1]) * 26
                 + decoded[start + 2]) * 26 + decoded[start + 3]
            ]
        total = sum(window_score_of)

        cut_short = False
        improved = True
        passes_here = 0
        while improved and not cut_short:
            if passes_here >= max_passes:
                # Unreachable with correct arithmetic (see MAX_CLIMB_PASSES).
                # Recorded rather than ignored so that a bug shows up as a
                # diagnostic instead of a machine that never answers.
                self.pass_limit_hits += 1
                break
            improved = False
            passes_here += 1
            self.passes += 1
            for counter, pair in enumerate(self.pairs):
                # Checking the clock costs a syscall, so do it once per 64
                # candidate swaps rather than on every one.
                if (
                    deadline is not None
                    and not counter & 63
                    and time.monotonic() >= deadline
                ):
                    cut_short = True
                    break

                windows = pair_windows[pair]
                if not windows:
                    continue  # neither letter occurs in the text
                self.swap_tests += 1

                first, second = pair
                before = sum(map(window_score_of.__getitem__, windows))

                # Apply the swap to the decryption only. `current` is not
                # touched until we know we are keeping the swap.
                plain_first = current[first]
                plain_second = current[second]
                for index in positions_of[first]:
                    decoded[index] = plain_second
                for index in positions_of[second]:
                    decoded[index] = plain_first

                after = 0.0
                for start in windows:
                    after += table[
                        ((decoded[start] * 26 + decoded[start + 1]) * 26
                         + decoded[start + 2]) * 26 + decoded[start + 3]
                    ]

                if after > before:
                    current[first] = plain_second
                    current[second] = plain_first
                    # Accepts are the minority, so recomputing the cache here
                    # is cheaper than building a save-list on every test.
                    # `decoded` already holds the post-swap decryption, so the
                    # affected windows rescore straight from it. Skipping this
                    # would leave the cache stale and make every later
                    # `before` wrong, silently corrupting the whole climb.
                    for start in windows:
                        window_score_of[start] = table[
                            ((decoded[start] * 26 + decoded[start + 1]) * 26
                             + decoded[start + 2]) * 26 + decoded[start + 3]
                        ]
                    total += after - before
                    improved = True
                else:
                    for index in positions_of[first]:
                        decoded[index] = plain_first
                    for index in positions_of[second]:
                        decoded[index] = plain_second

        return current, total, cut_short


# ---------------------------------------------------------------------------
# Building start keys
# ---------------------------------------------------------------------------


def _merge_over_fixed(
    base: Mapping[str, str],
    fixed: Mapping[str, str],
    *,
    strict: bool,
) -> dict[str, str]:
    """Combine a start mapping with pairs the human has fixed.

    The fixed pairs always win. In *strict* mode a disagreement is an error,
    which is what we want when the caller supplied both explicitly; in lenient
    mode the disagreeing part of *base* is simply dropped, which is what we
    want when *base* is only a frequency heuristic.
    """
    merged = dict(fixed)
    used = set(fixed.values())
    for cipher, plain in base.items():
        if cipher in merged:
            if strict and merged[cipher] != plain:
                raise ValueError(
                    f"start_key says cipher letter {cipher} means {plain}, but "
                    f"fixed says it means {merged[cipher]}"
                )
            continue
        if plain in used:
            if strict:
                raise ValueError(
                    f"start_key maps cipher letter {cipher} to {plain}, which "
                    "fixed has already assigned to another cipher letter"
                )
            continue
        merged[cipher] = plain
        used.add(plain)
    return merged


def _complete_deterministically(partial: Mapping[str, str]) -> dict[str, str]:
    """Fill a partial mapping out to a bijection, reproducibly.

    Undecided cipher letters are taken alphabetically and given the unused
    plain letters in English frequency order, so the completion is a mild
    guess rather than an arbitrary one, and identical on every run.
    """
    used = set(partial.values())
    free_cipher = [letter for letter in ALPHABET if letter not in partial]
    free_plain = [
        letter for letter in ENGLISH_FREQUENCY_ORDER if letter not in used
    ]
    completed = dict(partial)
    completed.update(zip(free_cipher, free_plain))
    return completed


def _random_key(fixed: Mapping[str, str], rng: random.Random) -> dict[str, str]:
    """A uniformly random bijection that respects the fixed pairs."""
    used = set(fixed.values())
    free_cipher = [letter for letter in ALPHABET if letter not in fixed]
    free_plain = [letter for letter in ALPHABET if letter not in used]
    rng.shuffle(free_plain)
    key = dict(fixed)
    key.update(zip(free_cipher, free_plain))
    return key


# ---------------------------------------------------------------------------
# Cribs
# ---------------------------------------------------------------------------


def apply_crib(
    text: str | NormalizedText,
    crib: str,
    position: int,
    *,
    known: SubstitutionKey | None = None,
) -> SubstitutionKey | None:
    """The key forced by assuming *crib* is the plaintext at *position*.

    *position* counts letters, not characters of the original: it indexes the
    normalised letters-only text, which is the only position that survives
    five-letter grouping.

    Returns ``None`` -- not an exception -- when the assumption is impossible,
    because a crib that does not fit is an ordinary and expected result. It is
    impossible when the crib would make one cipher letter mean two different
    plain letters, or two cipher letters mean the same plain letter, or when
    it contradicts *known*.

    A returned key is the mapping the crib *would* imply. It is a hypothesis
    to be tested, never a conclusion: several wrong positions usually survive
    this test, which is exactly why :func:`crib_positions` returns a list.
    """
    letters = _letters_of(text)
    cleaned_crib = letters_only(crib)
    if not cleaned_crib:
        raise ValueError("crib must contain at least one letter")
    if position < 0:
        raise ValueError(f"crib position must not be negative, got {position}")
    if position + len(cleaned_crib) > len(letters):
        raise ValueError(
            f"crib of {len(cleaned_crib)} letters does not fit at position "
            f"{position} in a text of {len(letters)} letters"
        )

    window = letters[position: position + len(cleaned_crib)]
    mapping = mapping_from_pair(window, cleaned_crib)
    if mapping is None:
        return None
    if known is None:
        return SubstitutionKey(mapping)

    merged = known
    for cipher, plain in mapping.items():
        try:
            merged = merged.with_pair(cipher, plain)
        except ValueError:
            return None
    return merged


def crib_positions(
    text: str | NormalizedText,
    crib: str,
    *,
    known: SubstitutionKey | None = None,
) -> list[int]:
    """Every letter position where *crib* could sit without contradiction.

    A position survives only if the implied mapping is a bijection and agrees
    with *known*. This prunes hard -- a six-letter crib with a repeated letter
    typically leaves a few per cent of positions -- but surviving is not
    evidence that a position is right, only that it is not yet ruled out.

    Returns an empty list when the crib fits nowhere, which is itself
    informative: the crib, or the assumption that the cipher is
    monoalphabetic, is wrong.
    """
    letters = _letters_of(text)
    cleaned_crib = letters_only(crib)
    if not cleaned_crib:
        raise ValueError("crib must contain at least one letter")
    if len(cleaned_crib) > len(letters):
        return []
    return [
        position
        for position in range(len(letters) - len(cleaned_crib) + 1)
        if apply_crib(letters, cleaned_crib, position, known=known) is not None
    ]


# ---------------------------------------------------------------------------
# Word patterns
# ---------------------------------------------------------------------------


def analyse_words(
    cipher_words: Sequence[str],
    scorer: EnglishScorer | None = None,
    *,
    known: SubstitutionKey | None = None,
    limit: int | None = 20,
) -> dict[str, list[str]]:
    """English words whose letter pattern matches each supplied cipher word.

    WARNING, and the reason this function takes its words explicitly: the
    spacing in competition ciphertext is transcription formatting, not word
    boundaries. National Cipher Challenge texts are normally printed in
    five-letter groups. This function therefore never splits anything on
    whitespace, and neither does anything else in this module. Pass the tokens
    only when you have a positive reason to believe they are real words -- for
    instance because the challenge published the message with its punctuation.

    A substitution cannot change the shape of a word, so ``HELLO`` must
    decrypt from something with the pattern 0-1-2-2-3. Matching on that
    signature usually cuts thousands of candidate words down to a handful, and
    each survivor supplies a set of letter equations. Candidates that would
    need two cipher letters to mean the same plain letter are discarded, since
    a substitution alphabet is a bijection.

    Returns a mapping from each cleaned cipher word to its candidate plain
    words, in the input order. A word with no repeated letters matches every
    English word of its length, so the list will be long and nearly useless;
    that is a property of the evidence, not a bug.
    """
    if isinstance(cipher_words, str):
        raise ValueError(
            "analyse_words takes a sequence of words, not a single string. "
            "Splitting ciphertext on whitespace is exactly the mistake this "
            "function refuses to make for you: five-letter groups are not "
            "words. Pass ['QEBOB', 'FP'] rather than 'QEBOB FP'."
        )
    engine = scorer or default_scorer()
    index = PatternIndex(engine.lexicon)
    known_mapping = known.mapping if known is not None else None

    results: dict[str, list[str]] = {}
    for raw in cipher_words:
        word = letters_only(str(raw))
        if not word or word in results:
            continue
        matches = match_word(word, index, known=known_mapping, limit=limit)
        results[word] = [match.plain_word for match in matches]
    return results


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    restarts: int = 25,
    seed: int | None = None,
    time_budget: float | None = None,
    fixed: SubstitutionKey | str | Mapping[str, str] | None = None,
    start_key: SubstitutionKey | str | Mapping[str, str] | None = None,
) -> CandidateSet:
    """Attack a monoalphabetic substitution and return ranked candidates.

    Parameters
    ----------
    source:
        Ciphertext, as a string or an already-normalised text.
    scorer:
        English model to climb on. ``None`` uses the shared default.
    top:
        How many candidates to return. The search always considers all of
        them; this only trims the report.
    restarts:
        Independent hill climbs. The first starts from *start_key* if given,
        otherwise from the frequency guess; the rest start from random
        bijections. More restarts buy confidence, not speed: the evidence a
        solve worked is several restarts agreeing.
    seed:
        Seeds a private ``random.Random``, so a seeded run is reproducible.
        The global ``random`` module is never touched.
    time_budget:
        Seconds. The search stops cleanly when it expires, mid-climb if
        necessary, and records ``time_budget_hit`` in the diagnostics.
    fixed:
        A partial key held constant throughout every climb -- a human crib.
        Those cipher letters are never swapped, so the search runs over a
        smaller space. A wrong crib will make every restart wrong, which is
        why the diagnostics report what was held.
    start_key:
        Where the first climb begins. Must not contradict *fixed*.

    Returns
    -------
    A :class:`CandidateSet`, best first, or an empty one for input with no
    letters. Each candidate carries the number of restarts that agreed on it
    (``agreements``), which is the real evidence of convergence, plus the
    spread of the restart scores so a run that never converged is visible.
    """
    if restarts < 1:
        raise ValueError(f"restarts must be at least 1, got {restarts}")
    if top < 1:
        raise ValueError(f"top must be at least 1, got {top}")
    if time_budget is not None and time_budget <= 0:
        raise ValueError(
            f"time_budget must be a positive number of seconds, got {time_budget}"
        )

    if isinstance(source, NormalizedText):
        normalized = source
    elif isinstance(source, str):
        normalized = normalize(source)
    else:
        raise ValueError(
            "solve needs a string or a NormalizedText, got "
            + type(source).__name__
        )

    engine = scorer or default_scorer()
    letters = normalized.letters
    found = CandidateSet()
    if not letters:
        return found

    fixed_key = _coerce_key(fixed, "fixed") if fixed is not None else SubstitutionKey()
    fixed_map = fixed_key.mapping

    if start_key is not None:
        start_map = _merge_over_fixed(
            _coerce_key(start_key, "start_key").mapping, fixed_map, strict=True
        )
    else:
        start_map = _merge_over_fixed(
            frequency_guess(letters).mapping, fixed_map, strict=False
        )
    first_start = _plain_indices(_complete_deterministically(start_map))

    free = [
        ord(letter) - 65 for letter in ALPHABET if letter not in fixed_map
    ]
    climber = _HillClimber(engine.encode(letters), engine.table(), free)

    # The climber reports no pairs when the text is shorter than one quadgram
    # window, or when the fixed key leaves fewer than two letters free. There
    # is then nothing to search, and running 25 identical restarts would
    # manufacture agreement that means nothing.
    searchable = bool(climber.pairs)
    attempts = restarts if searchable else 1

    rng = random.Random(seed)
    deadline = time.monotonic() + time_budget if time_budget is not None else None
    budget_hit = False
    restart_scores: list[float] = []
    outcomes: list[tuple[tuple[int, ...], float]] = []

    for attempt in range(attempts):
        start = first_start if attempt == 0 else _plain_indices(
            _random_key(fixed_map, rng)
        )
        final, value, cut_short = climber.climb(start, deadline)
        restart_scores.append(value)
        outcomes.append((tuple(final), value))
        if cut_short:
            budget_hit = True
            break
        if deadline is not None and time.monotonic() >= deadline:
            budget_hit = True
            break

    summary: dict[str, object] = {
        "restarts_run": len(outcomes),
        "restarts_requested": restarts,
        "distinct_local_optima": len({key for key, _ in outcomes}),
        "time_budget_hit": budget_hit,
        "letters": len(letters),
    }
    if restart_scores:
        summary["restart_score_best"] = max(restart_scores)
        summary["restart_score_worst"] = min(restart_scores)
        summary["restart_score_mean"] = sum(restart_scores) / len(restart_scores)
    if seed is not None:
        summary["seed"] = seed
    if fixed_map:
        summary["held_fixed"] = " ".join(
            f"{cipher}{plain}" for cipher, plain in sorted(fixed_map.items())
        )
    if not searchable:
        summary["searchable"] = False
    if climber.pass_limit_hits:
        # Should be impossible. If it ever appears, the incremental scoring in
        # _HillClimber disagrees with a full rescore and the key below is not
        # to be trusted.
        summary["climb_pass_limit_hit"] = climber.pass_limit_hits
    if len(letters) < RELIABLE_CLIMB_LETTERS:
        summary["short_text_warning"] = (
            f"only {len(letters)} letters; a hill climb needs roughly "
            f"{RELIABLE_CLIMB_LETTERS} before its answer means much"
        )
        # And weaken the headline to match the warning. A twenty-six letter
        # key has more freedom than a short ciphertext has evidence, so the
        # climber can land on a fluent English sentence that is not the
        # plaintext -- AHTLDSGAETSPNBLPFNPN enciphers ERANDSHEWASGOINGTOGO
        # and reads as YFORMANYCOASTERSITST, which scores -0.85 per letter
        # with 80 per cent word coverage and so cleared both "strong"
        # thresholds. The scorer cannot see that; this solver can.
        summary["confidence_cap"] = "promising"

    for plain_indices, value in outcomes:
        key = SubstitutionKey(
            dict(zip(ALPHABET, (ALPHABET[index] for index in plain_indices)))
        )
        plaintext = key.apply(letters)
        diagnostics: dict[str, object] = {"quadgram_window_score": value}
        annotate(diagnostics, plaintext, engine)
        found.add(
            Candidate(
                method=METHOD,
                key=f"key={key.to_alphabet()}",
                score=engine.score(plaintext),
                plaintext=plaintext,
                diagnostics=diagnostics,
                display=normalized.relayout(plaintext),
            )
        )

    # Every candidate carries the whole search's evidence, so a report on one
    # of them shows how much company it had.
    for candidate in found.ranked():
        candidate.diagnostics.update(summary)

    return CandidateSet(found.top(top))
