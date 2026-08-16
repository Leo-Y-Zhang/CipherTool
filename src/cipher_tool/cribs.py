"""Crib placement: testing WHERE a guessed plaintext fragment COULD sit.

A *crib* is a guess at a piece of the plaintext. Cipher Challenge stories hand
them out freely: a letter that almost certainly ends YOURS SINCERELY, a name
the narrator keeps repeating, a place the previous message named. A crib is
worth far more than its length suggests, because a classical cipher is a
*deterministic* function of the key: fixing a few plaintext letters usually
fixes a large part of the key, and every position where the crib cannot sit is
a position the search never has to visit again.

What this module does, and what it refuses to do
------------------------------------------------
Every function here answers one question only:

    "If this crib really is in the plaintext, where could it be, and what
     would the key have to look like there?"

It never answers "the crib is here". Under a substitution a five-letter crib
typically leaves dozens of surviving offsets in a few hundred letters; under a
Caesar it leaves the handful of shifts that produce the crib's letters
anywhere at all. Surviving a placement test means *not yet ruled out*, which
is a much weaker statement than *right*, and every report in this module says
so in as many words. The one genuinely strong result available here is a
negative one: if a crib fits nowhere, then either the crib is wrong or the
assumed cipher family is.

The four families, and why each one is attacked differently
-----------------------------------------------------------
**Monoalphabetic substitution.** Each ciphertext letter always stands for the
same plaintext letter, so the crib and the ciphertext window under it must
define a *bijection* -- one cipher letter cannot mean two plain letters, and
two cipher letters cannot mean the same plain letter. That second condition is
the one that does the pruning; :func:`patterns.mapping_from_pair` enforces
both. Worked example: the crib MEETING has signature 0-1-1-2-3-4-5, so a
window fits only if its second and third letters are equal and its other six
letters are all distinct. Over uniformly random letters that is
(1/26) * (26*25*24*23*22*21)/26**6 = 0.021, about one window in fifty. Real
ciphertext is not uniform, but the order of magnitude holds: a crib with a
repeated letter throws away roughly 98 per cent of the offsets before any
scoring happens at all.

**Caesar and affine.** The key space is tiny (26 shifts, 312 affine keys), so
we do not need to reason backwards at all: encipher the crib under every key
and look for the result in the ciphertext. That is complete by construction --
every key that could place the crib anywhere is reported, with every offset.

**Vigenere.** The most productive of the four. Ciphertext is plaintext plus a
repeating key, letter by letter modulo 26, so the key is simply the difference
of the two: K = C - P. Assume the crib sits at offset i, subtract it from the
ciphertext there, and the result *is* a stretch of the key -- aligned at key
position i mod L. With the key length already known, a crib as long as the key
hands over the whole key in one subtraction, and a crib shorter than the key
still fixes several of its letters. Without the key length the fragments are
still useful: two offsets produce the same fragment exactly when the two
ciphertext windows are identical, so a repeated fragment at spacing d is
Kasiski's argument applied at the crib's length, and the key length is a
divisor of d.

**Transposition.** Much weaker, and honestly labelled as such. Rearranging
letters does not change which letters are present, so all we can check is that
the ciphertext holds enough copies of each crib letter, and all we can offer a
human is the list of positions each crib letter occupies. There is no offset
to test, because the crib's letters are no longer adjacent. The letter-count
check is worth running anyway: when it fails it rules the entire transposition
family out, which is the strongest single result any function in this module
can produce.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from . import affine as affine_cipher
from . import caesar as caesar_cipher
from .columnar import plausible_column_counts
from .context import MINIMUM_CRIB_LENGTH
from .normalize import (
    ALPHABET_SIZE,
    NormalizedText,
    from_numbers,
    letters_only,
    to_numbers,
)
from .patterns import mapping_from_pair
from .reference import SUGGESTED_CRIBS
from .scoring import EnglishScorer, default_scorer
from .statistics import TextStatistics, divisors
from .substitution import SubstitutionKey

#: The cipher families this module can place a crib under, in report order.
METHODS: tuple[str, ...] = (
    "substitution",
    "caesar",
    "affine",
    "vigenere",
    "transposition",
)

#: How many rows of any one list a rendered report prints before summarising.
DEFAULT_LIMIT = 12

#: Longest key length worth voting for from a repeated key fragment. Matches
#: vigenere.MAXIMUM_KEY_LENGTH in spirit; kept local so this module does not
#: depend on the Vigenere solver's search limits.
MAXIMUM_VOTED_KEY_LENGTH = 20


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _letters_of(source: str | NormalizedText) -> str:
    """Letters-only view of either kind of input."""
    if isinstance(source, NormalizedText):
        return source.letters
    if isinstance(source, str):
        return letters_only(source)
    raise ValueError(
        "ciphertext must be a string or a NormalizedText, got "
        f"{type(source).__name__}"
    )


def _clean_crib(crib: str) -> str:
    """Normalise a crib the same way as ciphertext, or explain the refusal.

    A crib is a guess about *letters*, so case, spacing and punctuation are
    discarded exactly as they are for ciphertext: ``"meet me"`` and
    ``"MEETME"`` are the same crib. A crib with no letters in it is an error
    rather than a silently empty search, because "matches everywhere" would be
    a badly misleading answer.
    """
    if not isinstance(crib, str):
        raise ValueError(
            f"crib must be a string of letters, got {type(crib).__name__}"
        )
    cleaned = letters_only(crib)
    if not cleaned:
        raise ValueError(
            f"crib {crib!r} contains no letters A-Z. A crib is a guess at part "
            "of the plaintext, such as MEETING or YOURSSINCERELY."
        )
    return cleaned


def _as_key(
    known: SubstitutionKey | Mapping[str, str] | None,
) -> SubstitutionKey | None:
    """Accept either shape of partial key, or ``None`` for "nothing known"."""
    if known is None:
        return None
    if isinstance(known, SubstitutionKey):
        return known
    if isinstance(known, Mapping):
        return SubstitutionKey(known)
    raise ValueError(
        "known must be a SubstitutionKey, a cipher->plain mapping, or None; "
        f"got {type(known).__name__}"
    )


def _merge(
    known: SubstitutionKey, mapping: Mapping[str, str]
) -> SubstitutionKey | None:
    """Add *mapping* to *known*, or ``None`` if the two contradict each other.

    Contradiction is not an error here: it is the ordinary result of testing a
    guess that happens to be wrong, so it is reported as a value.
    """
    merged = known
    for cipher, plain in mapping.items():
        try:
            merged = merged.with_pair(cipher, plain)
        except ValueError:
            return None
    return merged


def _occurrences(haystack: str, needle: str) -> tuple[int, ...]:
    """Every start index of *needle* in *haystack*, overlaps included.

    ``str.find`` in a loop rather than a regular expression, because
    overlapping matches matter: AA occurs twice in AAA, and a crib such as
    THATTHAT can genuinely overlap itself.
    """
    if not needle or len(needle) > len(haystack):
        return ()
    found: list[int] = []
    start = haystack.find(needle)
    while start != -1:
        found.append(start)
        start = haystack.find(needle, start + 1)
    return tuple(found)


# ---------------------------------------------------------------------------
# Monoalphabetic substitution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    """One offset at which a crib COULD sit under a monoalphabetic cipher.

    A Placement is a hypothesis with its evidence attached, never a finding.
    It records where the crib would start, the ciphertext window it would sit
    under, and the partial substitution alphabet that window and crib together
    force.

    Attributes
    ----------
    position:
        Index into the *letters-only* ciphertext, not into the original
        string. Letter positions are the only ones that survive five-letter
        grouping, so they are what every function here uses.
    crib:
        The cleaned crib.
    window:
        The ciphertext letters the crib would cover.
    pairs:
        The implied key as a sorted tuple of ``(cipher, plain)`` pairs. Stored
        as a tuple rather than a dict so a Placement is hashable and can be
        deduplicated.
    """

    position: int
    crib: str
    window: str
    pairs: tuple[tuple[str, str], ...]

    @property
    def mapping(self) -> dict[str, str]:
        """The implied partial key as a cipher letter -> plain letter dict."""
        return dict(self.pairs)

    @property
    def fixes(self) -> int:
        """How many distinct cipher letters this placement would pin down.

        The useful measure of a crib's strength: a nine-letter crib with three
        repeated letters fixes six letters of the alphabet, not nine.
        """
        return len(self.pairs)

    @property
    def fixed_points(self) -> tuple[str, ...]:
        """Cipher letters this placement would have standing for themselves.

        Reported because some ciphers cannot have any (see the
        ``no_fixed_points`` option of :func:`substitution_placements`), and
        because a placement needing several of them is mildly suspicious even
        when they are allowed.
        """
        return tuple(cipher for cipher, plain in self.pairs if cipher == plain)

    def key(
        self, known: SubstitutionKey | Mapping[str, str] | None = None
    ) -> SubstitutionKey:
        """The partial :class:`substitution.SubstitutionKey` this implies.

        Pass *known* to fold this placement into a key you already believe.
        Raises ``ValueError`` if it contradicts *known* -- which cannot happen
        for a placement obtained from :func:`substitution_placements` with the
        same *known*, because contradictory offsets are dropped there.
        """
        base = _as_key(known) or SubstitutionKey()
        merged = _merge(base, self.mapping)
        if merged is None:
            raise ValueError(
                f"the placement at offset {self.position} contradicts the key "
                "you supplied"
            )
        return merged

    def describe(self) -> str:
        """One line: offset, letters fixed, and the implied pairs."""
        pairs = " ".join(f"{cipher}={plain}" for cipher, plain in self.pairs)
        return f"offset {self.position}: fixes {self.fixes} letters -- {pairs}"


def substitution_placements(
    ciphertext: str | NormalizedText,
    crib: str,
    *,
    known: SubstitutionKey | Mapping[str, str] | None = None,
    no_fixed_points: bool = False,
) -> list[Placement]:
    """Every offset where *crib* could sit under a monoalphabetic substitution.

    An offset survives only if the ciphertext window and the crib define a
    consistent *bijection*: no cipher letter standing for two plain letters,
    and no two cipher letters standing for the same plain letter. Both tests
    are :func:`patterns.mapping_from_pair`, and the second is the one that
    does most of the work -- it is what kills the spurious matches a naive
    "same shape" test would accept.

    Surviving offsets are possibilities, not findings. A crib with no repeated
    letters constrains only the bijection condition and will survive at a large
    fraction of offsets; that is a property of the crib, not a fault here. Use
    the longest, most repetitive crib available.

    Parameters
    ----------
    known:
        A partial key already believed, as a :class:`SubstitutionKey` or a
        cipher->plain mapping. Offsets contradicting it are dropped, in both
        directions.
    no_fixed_points:
        When ``True``, drop any offset that would need a ciphertext letter to
        stand for itself. **The default is False and should usually stay
        there.** This flag exists for the famous special case: an Enigma
        machine can never encipher a letter as itself, which is exactly what
        let Bletchley Park slide a crib along a message and reject every
        position where a letter agreed with the ciphertext. Ordinary
        pencil-and-paper substitutions have no such property -- a keyword
        alphabet routinely leaves several letters where they started -- so
        switching this on without a reason will discard the true placement.

    Returns an empty list when the crib fits nowhere. That is informative: the
    crib is wrong, or the cipher is not monoalphabetic, or *known* is wrong.
    """
    letters = _letters_of(ciphertext)
    cleaned = _clean_crib(crib)
    key = _as_key(known)

    if len(cleaned) > len(letters):
        return []

    placements: list[Placement] = []
    for position in range(len(letters) - len(cleaned) + 1):
        window = letters[position: position + len(cleaned)]
        mapping = mapping_from_pair(window, cleaned)
        if mapping is None:
            continue
        if no_fixed_points and any(
            cipher == plain for cipher, plain in mapping.items()
        ):
            continue
        if key is not None and _merge(key, mapping) is None:
            continue
        placements.append(
            Placement(
                position=position,
                crib=cleaned,
                window=window,
                pairs=tuple(sorted(mapping.items())),
            )
        )
    return placements


# ---------------------------------------------------------------------------
# Caesar and affine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShiftPlacement:
    """One Caesar shift that would put the crib somewhere in the ciphertext."""

    shift: int
    positions: tuple[int, ...]

    @property
    def key(self) -> str:
        """The key in the toolkit's copy-pasteable form, e.g. ``shift=3``."""
        return f"shift={self.shift}"

    def describe(self) -> str:
        """One line: the key and every offset it would place the crib at."""
        offsets = ", ".join(str(position) for position in self.positions)
        return f"{self.key}: offsets {offsets}"


@dataclass(frozen=True)
class AffinePlacement:
    """One affine key ``(a, b)`` that would place the crib somewhere."""

    a: int
    b: int
    positions: tuple[int, ...]

    @property
    def key(self) -> str:
        """The key in the toolkit's copy-pasteable form, e.g. ``a=5 b=8``."""
        return f"a={self.a} b={self.b}"

    def describe(self) -> str:
        """One line: the key, any well-known name for it, and the offsets."""
        offsets = ", ".join(str(position) for position in self.positions)
        note = affine_cipher.describe_key(self.a, self.b)
        suffix = f"  [{note}]" if note else ""
        return f"{self.key}: offsets {offsets}{suffix}"


def caesar_placements(
    ciphertext: str | NormalizedText, crib: str
) -> list[ShiftPlacement]:
    """Which Caesar shifts are consistent with *crib* appearing anywhere.

    A Caesar key is a single number, so there is nothing to reason backwards
    about: encipher the crib under each of the 26 shifts and look for the
    result in the ciphertext. Any shift whose enciphered crib appears is
    consistent; any shift whose enciphered crib does not appear is *ruled out*
    on the assumption that the crib is present at all.

    The shift is the enciphering shift, matching the rest of the toolkit:
    ``caesar.decrypt(ciphertext, shift)`` is the reading that would contain
    the crib.

    Completeness is what makes this worth doing. With only 26 keys the answer
    is exhaustive, so an empty list is a real negative: under no shift does
    this ciphertext contain this crib.
    """
    letters = _letters_of(ciphertext)
    cleaned = _clean_crib(crib)

    results: list[ShiftPlacement] = []
    for shift in range(ALPHABET_SIZE):
        positions = _occurrences(letters, caesar_cipher.encrypt(cleaned, shift))
        if positions:
            results.append(ShiftPlacement(shift=shift, positions=positions))
    return results


def affine_placements(
    ciphertext: str | NormalizedText, crib: str
) -> list[AffinePlacement]:
    """Which affine keys ``(a, b)`` are consistent with *crib* appearing.

    Same idea as :func:`caesar_placements` over the 12 * 26 = 312 usable
    affine keys (``a`` must be coprime with 26, see
    :func:`affine.valid_multipliers`). Enciphering a short crib 312 times
    costs nothing, and brute force is exhaustive, so the result is complete
    rather than heuristic.

    Solving algebraically would also work -- two plaintext/ciphertext letter
    pairs give ``a * (p1 - p2) = c1 - c2`` mod 26 and then ``b`` follows -- but
    it needs care when ``p1 - p2`` shares a factor with 26, and it gains
    nothing measurable here. The obviously-correct method wins.

    A warning the caller should read: a crib made of a single repeated letter
    (``AAA``) constrains one equation in two unknowns, so a great many keys
    will "fit". Short cribs are weak evidence under any cipher, and weakest of
    all here.
    """
    letters = _letters_of(ciphertext)
    cleaned = _clean_crib(crib)

    results: list[AffinePlacement] = []
    for multiplier in affine_cipher.valid_multipliers():
        for offset in range(ALPHABET_SIZE):
            positions = _occurrences(
                letters, affine_cipher.encrypt(cleaned, multiplier, offset)
            )
            if positions:
                results.append(
                    AffinePlacement(a=multiplier, b=offset, positions=positions)
                )
    return results


# ---------------------------------------------------------------------------
# Vigenere
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyFragment:
    """The key stretch that would produce *crib* at one offset.

    Attributes
    ----------
    position:
        Offset into the letters-only ciphertext.
    crib, window:
        The crib and the ciphertext letters it would cover.
    fragment:
        ``window - crib`` letter by letter, modulo 26: the key letters that
        would have to be in force across those positions.
    key_length:
        The assumed key length, or ``None`` if none was assumed.
    consistent:
        With an assumed key length, whether the fragment can come from a
        repeating key of that length -- i.e. whether every key position the
        fragment touches more than once is given the same letter each time.
        Always ``True`` when no key length was assumed, since then there is
        nothing to be inconsistent with.
    partial_key:
        With an assumed key length and a consistent fragment, the key with the
        letters this offset would fix and ``.`` everywhere else. ``None``
        otherwise.
    """

    position: int
    crib: str
    window: str
    fragment: str
    key_length: int | None = None
    consistent: bool = True
    partial_key: str | None = None

    @property
    def fixes(self) -> int:
        """How many key positions this offset would decide.

        Zero when no key length was assumed, because without a length there
        are no key positions to speak of.
        """
        if self.partial_key is None:
            return 0
        return sum(1 for char in self.partial_key if char != ".")

    def describe(self) -> str:
        """One line: offset, fragment, and the partial key if there is one."""
        if self.partial_key is not None:
            return (
                f"offset {self.position}: fragment {self.fragment} -> "
                f"key {self.partial_key}"
            )
        verdict = "" if self.consistent else "  [inconsistent]"
        return f"offset {self.position}: fragment {self.fragment}{verdict}"


def vigenere_placements(
    ciphertext: str | NormalizedText,
    crib: str,
    key_length: int | None = None,
) -> list[KeyFragment]:
    """The key fragment *crib* would need at every offset of the ciphertext.

    This is the strongest attack in this module, and the reason is worth
    stating carefully. A Vigenere ciphertext is the plaintext plus a repeating
    key, letter by letter modulo 26:

        C[i] = (P[i] + K[i mod L]) mod 26

    Rearranged, ``K[i mod L] = (C[i] - P[i]) mod 26``. So if the crib really
    is the plaintext at offset i, subtracting it from the ciphertext there does
    not merely *test* a hypothesis -- it *hands over a piece of the key*,
    already aligned at key position ``i mod L``. A crib as long as the key
    gives the whole key from one subtraction. A crib shorter than the key still
    fixes several of its letters, and the rest can be finished by frequency
    fitting on the remaining columns.

    With *key_length* supplied, each offset is checked for internal
    consistency: if the fragment is longer than the key it wraps round, and
    every key position it lands on twice must receive the same letter. That
    test alone throws away most offsets. Consistent offsets carry a
    ``partial_key`` ready to hand to :func:`vigenere.decrypt` once completed.

    Without *key_length*, every offset is reported with its fragment and
    nothing is rejected -- there is nothing yet to reject it with. The
    fragments are still evidence: two offsets produce the *same* fragment
    exactly when the two ciphertext windows are identical (the crib cancels
    out of the subtraction), so a fragment repeating at spacing d says the key
    length divides d. That is Kasiski's argument narrowed to the crib's
    length, and :func:`key_length_votes` turns it into a vote per divisor.

    Every offset is returned rather than only the plausible ones, so the caller
    can see how hard the test actually pruned; filter on ``.consistent``, or
    use :func:`consistent_fragments`. An offset surviving this test is still
    only a possibility: nothing here says the crib is at any of them.

    Raises ``ValueError`` for a key length that is not a positive integer.
    """
    letters = _letters_of(ciphertext)
    cleaned = _clean_crib(crib)

    if key_length is not None:
        if isinstance(key_length, bool) or not isinstance(key_length, int):
            raise ValueError(
                "key_length must be a whole number of letters, got "
                f"{key_length!r} ({type(key_length).__name__})"
            )
        if key_length < 1:
            raise ValueError(
                f"key_length must be at least 1, got {key_length}"
            )

    if len(cleaned) > len(letters):
        return []

    crib_values = to_numbers(cleaned)
    fragments: list[KeyFragment] = []
    for position in range(len(letters) - len(cleaned) + 1):
        window = letters[position: position + len(cleaned)]
        # K = C - P, the whole attack in one line.
        fragment = from_numbers(
            cipher_value - plain_value
            for cipher_value, plain_value in zip(to_numbers(window), crib_values)
        )

        if key_length is None:
            fragments.append(
                KeyFragment(
                    position=position,
                    crib=cleaned,
                    window=window,
                    fragment=fragment,
                )
            )
            continue

        # Fold the fragment onto the key positions it would occupy. A clash
        # means a repeating key of this length cannot produce this crib here.
        slots: dict[int, str] = {}
        consistent = True
        for index, letter in enumerate(fragment):
            slot = (position + index) % key_length
            if slots.setdefault(slot, letter) != letter:
                consistent = False
                break
        partial = (
            "".join(slots.get(slot, ".") for slot in range(key_length))
            if consistent
            else None
        )
        fragments.append(
            KeyFragment(
                position=position,
                crib=cleaned,
                window=window,
                fragment=fragment,
                key_length=key_length,
                consistent=consistent,
                partial_key=partial,
            )
        )
    return fragments


def consistent_fragments(placements: Sequence[KeyFragment]) -> list[KeyFragment]:
    """Just the fragments a repeating key of the assumed length could produce.

    With no assumed key length every fragment is consistent and this returns
    the input unchanged, which is the honest answer rather than an empty one.
    """
    return [placement for placement in placements if placement.consistent]


def repeated_fragments(
    placements: Sequence[KeyFragment],
) -> dict[str, tuple[int, ...]]:
    """Key fragments that occur at more than one offset, with those offsets.

    Two offsets share a fragment exactly when the underlying ciphertext
    windows are identical, because the crib cancels out of ``C - P``. So this
    is a repeated-substring search at the crib's length, restricted to the
    windows the crib could plausibly explain -- and repeated ciphertext is the
    classic evidence for a periodic key.
    """
    grouped: dict[str, list[int]] = {}
    for placement in placements:
        grouped.setdefault(placement.fragment, []).append(placement.position)
    return {
        fragment: tuple(positions)
        for fragment, positions in sorted(grouped.items())
        if len(positions) > 1
    }


def key_length_votes(
    placements: Sequence[KeyFragment],
    *,
    maximum: int = MAXIMUM_VOTED_KEY_LENGTH,
) -> Counter:
    """Vote for key lengths implied by repeated key fragments.

    If the same fragment appears at offsets i and j then the ciphertext repeats
    at spacing ``d = j - i``. Under a repeating key of length L, an identical
    ciphertext stretch is what happens when identical plaintext meets identical
    key, and the key is identical again only after a whole number of periods --
    so L divides d. Each divisor of each spacing gets one vote.

    A vote is a hint, not a measurement. Short repeats happen by chance, and
    1 is a divisor of everything, so it is excluded along with anything longer
    than *maximum*. For serious key-length work use
    :func:`vigenere.estimate_key_lengths`, which combines Kasiski votes with
    the index of coincidence over the whole text rather than one crib.
    """
    votes: Counter = Counter()
    for positions in repeated_fragments(placements).values():
        for index, first in enumerate(positions):
            for second in positions[index + 1:]:
                for divisor in divisors(second - first):
                    if 2 <= divisor <= maximum:
                        votes[divisor] += 1
    return votes


def word_key_fragments(
    placements: Sequence[KeyFragment],
    *,
    lexicon: Iterable[str] | None = None,
    scorer: EnglishScorer | None = None,
) -> list[KeyFragment]:
    """Fragments whose key reads as an English word, as a heuristic filter.

    Keys chosen by humans are nearly always words, so a fragment that reads
    LEMON is more interesting than one that reads QXZVB. This is a *prior about
    the person who chose the key*, not a fact about the cipher: a random key
    would never trip it, and a real key can fail it whenever the crib straddles
    a wrap-round of the key, since the fragment is then a rotation of the word
    (MONLE for LEMON). Treat a hit as a place to look first, nothing more.

    Both readings are tested: the raw fragment, and -- when a key length was
    assumed and this offset fixed every position of it -- the completed key,
    which is the fragment rotated back into its proper order and is the one
    that usually reads as a word.

    Strings shorter than :data:`MINIMUM_CRIB_LENGTH` are skipped: two-letter
    "words" match constantly and would bury the useful hits.
    """
    if lexicon is None:
        engine = scorer if scorer is not None else default_scorer()
        words = engine.lexicon
    else:
        words = frozenset(word.upper() for word in lexicon)

    found: list[KeyFragment] = []
    for placement in placements:
        readings = [placement.fragment]
        if placement.partial_key is not None and "." not in placement.partial_key:
            readings.append(placement.partial_key)
        if any(
            len(reading) >= MINIMUM_CRIB_LENGTH and reading in words
            for reading in readings
        ):
            found.append(placement)
    return found


# ---------------------------------------------------------------------------
# Transposition
# ---------------------------------------------------------------------------


@dataclass
class TranspositionCribHelp:
    """What can honestly be said about a crib under a transposition.

    Attributes
    ----------
    crib, length:
        The cleaned crib, and the ciphertext letter count.
    possible:
        Whether the ciphertext contains enough copies of every crib letter.
        ``False`` rules out every transposition of this ciphertext, which is
        the strongest result in this module.
    needed, available, missing:
        Per-letter counts required, present, and short (missing is empty when
        ``possible`` is true).
    positions:
        Every position of each crib letter in the ciphertext, rarest letter
        first -- the list a human actually works from.
    column_counts:
        Column counts that would fill a complete rectangle of this length,
        from :func:`columnar.plausible_column_counts`. Context only: an
        incomplete columnar leaves every count possible.
    """

    crib: str
    length: int
    possible: bool
    needed: dict[str, int] = field(default_factory=dict)
    available: dict[str, int] = field(default_factory=dict)
    missing: dict[str, int] = field(default_factory=dict)
    positions: dict[str, tuple[int, ...]] = field(default_factory=dict)
    column_counts: tuple[int, ...] = ()

    def rarest_letters(self, count: int = 3) -> tuple[str, ...]:
        """The crib letters with the fewest positions to check, rarest first.

        These are where a human should start: a crib letter occurring twice
        gives two places to look, one occurring ninety times gives ninety.
        """
        order = sorted(
            self.positions,
            key=lambda letter: (len(self.positions[letter]), letter),
        )
        return tuple(order[:count])

    def render(self, *, limit: int = DEFAULT_LIMIT) -> str:
        """The human-readable report, including its own honesty warning."""
        lines: list[str] = []
        if not self.possible:
            shortfall = ", ".join(
                f"{letter} needs {self.needed[letter]} but only "
                f"{self.available.get(letter, 0)} present"
                for letter in sorted(self.missing)
            )
            lines.append(
                "RULED OUT: a transposition only moves letters, so the "
                "ciphertext must already contain every letter of the crib."
            )
            lines.append(f"  {shortfall}")
            lines.append(
                "  Therefore no transposition of this ciphertext contains "
                "this crib. Either the crib is wrong or the cipher is not a "
                "pure transposition."
            )
            return "\n".join(lines)

        lines.append(
            "Not ruled out: every crib letter is present often enough. This is "
            "a weak result -- it is a letter count, not a placement."
        )
        lines.append("Positions of each crib letter (rarest first):")
        order = sorted(
            self.positions,
            key=lambda letter: (len(self.positions[letter]), letter),
        )
        for letter in order:
            found = self.positions[letter]
            shown = ", ".join(str(position) for position in found[:limit])
            more = f", ... {len(found) - limit} more" if len(found) > limit else ""
            lines.append(
                f"  {letter} x{len(found):<4} needs {self.needed[letter]}: "
                f"{shown}{more}"
            )
        if self.column_counts:
            counts = ", ".join(str(count) for count in self.column_counts)
            lines.append(
                f"Complete-rectangle column counts for {self.length} letters: "
                f"{counts}"
            )
            lines.append(
                "  (Only relevant if the sender padded to a full rectangle; "
                "an incomplete columnar allows any count.)"
            )
        lines.append(
            "Automated crib placement is much weaker under a transposition "
            "than under a substitution: rearranging letters destroys "
            "adjacency, so there is no offset to test and no key fragment to "
            "read off. Work from the rarest crib letter's positions by hand, "
            "or feed the crib to the columnar and rail fence solvers."
        )
        return "\n".join(lines)


def transposition_crib_help(
    ciphertext: str | NormalizedText, crib: str
) -> TranspositionCribHelp:
    """What a crib can and cannot tell you about a transposition.

    A transposition rearranges the letters and changes none of them, so the
    multiset of letters is an invariant of the whole cipher family. Two things
    follow, one strong and one weak.

    **Strong (a negative).** If the ciphertext does not contain enough copies
    of some crib letter, then no rearrangement of it can contain the crib, and
    the entire transposition family is eliminated at a stroke. Almost nothing
    else in classical cryptanalysis rules a family out this cleanly, and it
    costs one pass of letter counting.

    **Weak (a positive).** If the letters are all present, that is not evidence
    the crib is there. Any English text of a few hundred letters contains
    enough Es and Ts for almost any crib. All this function can then offer is
    the position of every crib letter, so a human can hunt for the pattern --
    typically by looking at the rarest crib letter first, since it gives the
    fewest places to check.

    Nothing here attempts automated placement, because under a transposition
    the crib's letters are no longer adjacent and there is no offset to slide.
    """
    letters = _letters_of(ciphertext)
    cleaned = _clean_crib(crib)

    needed = Counter(cleaned)
    available = Counter(letters)
    missing = {
        letter: count - available.get(letter, 0)
        for letter, count in needed.items()
        if available.get(letter, 0) < count
    }

    positions: dict[str, tuple[int, ...]] = {}
    for letter in sorted(set(cleaned)):
        positions[letter] = tuple(
            index for index, char in enumerate(letters) if char == letter
        )

    return TranspositionCribHelp(
        crib=cleaned,
        length=len(letters),
        possible=not missing,
        needed=dict(sorted(needed.items())),
        available={letter: available.get(letter, 0) for letter in sorted(needed)},
        missing=dict(sorted(missing.items())),
        positions=positions,
        column_counts=tuple(plausible_column_counts(len(letters))),
    )


# ---------------------------------------------------------------------------
# Suggested cribs
# ---------------------------------------------------------------------------


def suggest_cribs(
    stats: TextStatistics | str | NormalizedText,
    *,
    minimum_length: int = MINIMUM_CRIB_LENGTH,
) -> list[str]:
    """Stock cribs worth trying on a text of this length. GUESSES, not evidence.

    These come from :data:`reference.SUGGESTED_CRIBS`, a hand-typed list of
    words and phrases that turn up constantly in Cipher Challenge stories. The
    list is a prior about the competition's writing style and nothing else.
    **Not one of these words is known to be in your ciphertext**, and a
    placement found for one of them is exactly as speculative as the guess
    that put it there. A crib taken from the story that accompanies the
    message beats every word on this list.

    Filtering is deliberately crude, because there is nothing subtle to be
    done: a crib must fit in the text at all, must be long enough to rule
    anything out (a two-letter crib fits almost everywhere), and should not be
    longer than half the message, since a crib that long is a guess about most
    of the plaintext rather than a foothold in it. Longest first, because a
    longer crib constrains harder.

    Accepts a :class:`statistics.TextStatistics`, or plain text if you have not
    run :func:`statistics.analyse`.
    """
    if isinstance(stats, TextStatistics):
        length = stats.length
    else:
        length = len(_letters_of(stats))

    maximum = max(minimum_length, length // 2)
    chosen = [
        word
        for word in SUGGESTED_CRIBS
        if minimum_length <= len(word) <= min(maximum, length)
    ]
    return sorted(set(chosen), key=lambda word: (-len(word), word))


def describe_suggestions(
    stats: TextStatistics | str | NormalizedText,
    *,
    minimum_length: int = MINIMUM_CRIB_LENGTH,
    limit: int = 24,
) -> str:
    """Render :func:`suggest_cribs` with the label those guesses need."""
    words = suggest_cribs(stats, minimum_length=minimum_length)
    if not words:
        return (
            "Suggested cribs: none. The text is too short for any stock crib "
            "to be worth testing."
        )
    shown = ", ".join(words[:limit])
    more = f" ... and {len(words) - limit} more" if len(words) > limit else ""
    return (
        "Suggested cribs -- GUESSES from a stock list, not evidence, and not "
        "known to be in this message:\n"
        f"  {shown}{more}\n"
        "  A word from the story published with the cipher is worth more than "
        "any of these."
    )


# ---------------------------------------------------------------------------
# The combined report
# ---------------------------------------------------------------------------


@dataclass
class CribReport:
    """Everything the placement tests found for one crib, and nothing more.

    Each field is ``None`` when its method was not run, and an empty list when
    the method ran and found nothing -- a distinction that matters, because
    "not tested" and "ruled out" are very different answers.
    """

    crib: str
    letters: str
    methods: tuple[str, ...]
    offsets_tested: int
    substitution: list[Placement] | None = None
    caesar: list[ShiftPlacement] | None = None
    affine: list[AffinePlacement] | None = None
    vigenere: list[KeyFragment] | None = None
    vigenere_key_length: int | None = None
    vigenere_words: list[KeyFragment] | None = None
    vigenere_votes: Counter | None = None
    transposition: TranspositionCribHelp | None = None
    no_fixed_points: bool = False
    limit: int = DEFAULT_LIMIT

    # -- reading -----------------------------------------------------------

    def possible_methods(self) -> tuple[str, ...]:
        """Families this crib has not ruled out, in report order.

        Absence from this tuple is the useful half: a family that was tested
        and produced nothing cannot hold this crib. Presence means only "still
        possible".
        """
        alive: list[str] = []
        if self.substitution:
            alive.append("substitution")
        if self.caesar:
            alive.append("caesar")
        if self.affine:
            alive.append("affine")
        if self.vigenere and consistent_fragments(self.vigenere):
            alive.append("vigenere")
        if self.transposition is not None and self.transposition.possible:
            alive.append("transposition")
        return tuple(name for name in METHODS if name in alive)

    def summary(self) -> str:
        """One line naming the families still standing after these tests."""
        if self.offsets_tested == 0:
            return (
                f"The crib is {len(self.crib)} letters and the ciphertext is "
                f"{len(self.letters)}, so it does not fit at all. Nothing was "
                "tested and nothing is ruled out."
            )
        alive = self.possible_methods()
        if not alive:
            return (
                f"No tested family can place {self.crib} in this ciphertext. "
                "The crib is wrong, or the message is not one of these."
            )
        return (
            f"{self.crib} is still possible under: " + ", ".join(alive) +
            ". Possible is not probable -- read the evidence below."
        )

    # -- rendering ---------------------------------------------------------

    def render(self) -> str:
        """The full report as plain text, honest about every uncertainty."""
        lines: list[str] = []
        title = f'Crib test: "{self.crib}" ({len(self.crib)} letters)'
        lines.append(title)
        lines.append("=" * len(title))
        lines.append(
            f"Ciphertext: {len(self.letters)} letters, "
            f"{self.offsets_tested} offsets to test."
        )
        lines.append("Methods tested: " + ", ".join(self.methods))
        if len(self.crib) < MINIMUM_CRIB_LENGTH:
            lines.append(
                f"WARNING: a crib of {len(self.crib)} letters fits almost "
                "anywhere and rules out almost nothing."
            )

        if self.substitution is not None:
            lines.extend(self._substitution_section())
        if self.caesar is not None:
            lines.extend(self._caesar_section())
        if self.affine is not None:
            lines.extend(self._affine_section())
        if self.vigenere is not None:
            lines.extend(self._vigenere_section())
        if self.transposition is not None:
            lines.append("")
            lines.append("Transposition")
            lines.append("-------------")
            lines.append(self.transposition.render(limit=self.limit))

        lines.append("")
        lines.append(self.summary())
        lines.append(
            "NOTE: every line above is a POSSIBILITY. Nothing here says the "
            "crib is at any particular place; a placement is only an offset "
            "that has not yet been ruled out."
        )
        return "\n".join(lines)

    def _heading(self, title: str) -> list[str]:
        """Blank line, title, underline -- the toolkit's usual section head."""
        return ["", title, "-" * len(title)]

    def _does_not_fit(self) -> str:
        """The line every section prints when there is no offset to test.

        Saying "ruled out" here would be a lie: a crib longer than the
        ciphertext was never tested against anything.
        """
        return (
            f"The crib does not fit in {len(self.letters)} letters, so there "
            "is no offset to test and nothing is ruled out."
        )

    def _substitution_section(self) -> list[str]:
        """Render the monoalphabetic placements."""
        placements = self.substitution or []
        lines = self._heading("Monoalphabetic substitution")
        if self.offsets_tested == 0:
            lines.append(self._does_not_fit())
            return lines
        if not placements:
            lines.append(
                "No offset survives. Under a monoalphabetic substitution this "
                "ciphertext cannot contain this crib: at every offset the "
                "implied alphabet contradicts itself. Strong negative "
                "evidence -- against the crib, or against the family."
            )
            return lines

        share = (
            100.0 * len(placements) / self.offsets_tested
            if self.offsets_tested
            else 0.0
        )
        lines.append(
            f"{len(placements)} of {self.offsets_tested} offsets survive "
            f"({share:.1f}%). Surviving means the implied alphabet is "
            "self-consistent and bijective, nothing more."
        )
        if self.no_fixed_points:
            lines.append(
                "Offsets needing a letter to stand for itself were discarded "
                "(no_fixed_points=True). Most pencil-and-paper substitutions "
                "DO have fixed points, so this may have discarded the truth."
            )
        ordered = sorted(placements, key=lambda p: (-p.fixes, p.position))
        for placement in ordered[: self.limit]:
            marker = "*" if placement.fixed_points else " "
            lines.append(f" {marker}{placement.describe()}")
        if len(ordered) > self.limit:
            lines.append(f"  ... and {len(ordered) - self.limit} more offsets.")
        if any(placement.fixed_points for placement in ordered[: self.limit]):
            lines.append(
                "  * this offset needs at least one letter to stand for "
                "itself; allowed by an ordinary substitution, impossible for "
                "some machine ciphers."
            )
        return lines

    def _caesar_section(self) -> list[str]:
        """Render the Caesar shifts consistent with the crib."""
        shifts = self.caesar or []
        lines = self._heading("Caesar shift")
        if self.offsets_tested == 0:
            lines.append(self._does_not_fit())
            return lines
        if not shifts:
            lines.append(
                "No shift places the crib. All 26 keys were enciphered and "
                "searched for, so this rules the Caesar cipher out unless the "
                "crib itself is wrong."
            )
            return lines
        lines.append(
            f"{len(shifts)} of {ALPHABET_SIZE} shifts place the crib "
            "somewhere:"
        )
        for placement in shifts[: self.limit]:
            lines.append(f"  {placement.describe()}")
        if len(shifts) > self.limit:
            lines.append(f"  ... and {len(shifts) - self.limit} more shifts.")
        lines.append(
            "  Decrypt with: cipher_tool caesar <file> --shift N"
        )
        return lines

    def _affine_section(self) -> list[str]:
        """Render the affine keys consistent with the crib."""
        keys = self.affine or []
        lines = self._heading("Affine")
        if self.offsets_tested == 0:
            lines.append(self._does_not_fit())
            return lines
        if not keys:
            lines.append(
                "No affine key places the crib. All 312 usable keys were "
                "tested, so the affine cipher is ruled out unless the crib is "
                "wrong."
            )
            return lines
        lines.append(f"{len(keys)} of 312 usable keys place the crib somewhere:")
        for placement in keys[: self.limit]:
            lines.append(f"  {placement.describe()}")
        if len(keys) > self.limit:
            lines.append(f"  ... and {len(keys) - self.limit} more keys.")
        return lines

    def _vigenere_section(self) -> list[str]:
        """Render the Vigenere key fragments and the key-length evidence."""
        placements = self.vigenere or []
        lines = self._heading("Vigenere")
        if not placements:
            lines.append(self._does_not_fit())
            return lines

        if self.vigenere_key_length is None:
            lines.append(
                f"No key length assumed. Each of the {len(placements)} offsets "
                "gives the key fragment it would need (fragment = ciphertext "
                "minus crib, modulo 26)."
            )
            for placement in placements[: self.limit]:
                lines.append(f"  {placement.describe()}")
            if len(placements) > self.limit:
                lines.append(
                    f"  ... and {len(placements) - self.limit} more offsets."
                )
        else:
            good = consistent_fragments(placements)
            lines.append(
                f"Key length {self.vigenere_key_length} assumed: "
                f"{len(good)} of {len(placements)} offsets are consistent with "
                "a repeating key of that length."
            )
            if not good:
                lines.append(
                    "  None survive. Either the key length is wrong or the "
                    "crib is not in this message."
                )
            for placement in sorted(
                good, key=lambda p: (-p.fixes, p.position)
            )[: self.limit]:
                lines.append(f"  {placement.describe()}")
            if len(good) > self.limit:
                lines.append(f"  ... and {len(good) - self.limit} more offsets.")

        repeats = repeated_fragments(placements)
        if repeats:
            lines.append(
                "Repeated fragments (identical ciphertext windows -- the crib "
                "cancels out, so this is Kasiski at the crib's length):"
            )
            for fragment, positions in list(repeats.items())[: self.limit]:
                offsets = ", ".join(str(position) for position in positions)
                lines.append(f"  {fragment} at offsets {offsets}")
        votes = self.vigenere_votes or Counter()
        if votes:
            ranked = ", ".join(
                f"{length} ({count})"
                for length, count in sorted(
                    votes.items(), key=lambda row: (-row[1], row[0])
                )[:8]
            )
            lines.append(f"  Key lengths those spacings would allow: {ranked}")
            lines.append(
                "  Votes are hints. Confirm with cipher_tool vigenere "
                "--key-length, which measures the whole text."
            )
        if self.vigenere_words:
            lines.append(
                "  Fragments whose key reads as an English word (keys usually "
                "do, so look here first -- a hint about the key's author, not "
                "evidence about the cipher):"
            )
            for placement in self.vigenere_words[: self.limit]:
                lines.append(f"    {placement.describe()}")
        return lines


def test_crib(
    source: str | NormalizedText,
    crib: str,
    *,
    methods: Sequence[str] | None = None,
    key_length: int | None = None,
    known: SubstitutionKey | Mapping[str, str] | None = None,
    no_fixed_points: bool = False,
    scorer: EnglishScorer | None = None,
    limit: int = DEFAULT_LIMIT,
) -> CribReport:
    """Run every applicable placement test for one crib and collect the results.

    This is what ``cipher_tool crib <file> "THE"`` calls. It runs the tests,
    reports what each one found, and picks nothing: the report lists
    possibilities per family and states plainly that a surviving placement is
    an offset not yet ruled out rather than an answer.

    Parameters
    ----------
    methods:
        Which families to test, from :data:`METHODS`. ``None`` or an empty
        sequence runs all of them. An unrecognised name is a ``ValueError``
        rather than a silent skip, because silently testing nothing would look
        exactly like a crib that fits nowhere.
    key_length:
        Assumed Vigenere key length, if the key-length analysis has already
        suggested one. Sharpens the Vigenere section considerably.
    known:
        A partial substitution key already believed; placements contradicting
        it are dropped.
    no_fixed_points:
        Passed to :func:`substitution_placements`. Read its warning first.
    scorer:
        Used only to look up whether a Vigenere key fragment is an English
        word. ``None`` uses the shared :func:`scoring.default_scorer`.

    Empty ciphertext is not an error: every test simply reports that nothing
    fits. An empty crib IS an error, because "where could nothing sit" has no
    useful answer.
    """
    letters = _letters_of(source)
    cleaned = _clean_crib(crib)

    if methods is None or len(methods) == 0:
        chosen = METHODS
    else:
        if isinstance(methods, str):
            raise ValueError(
                "methods must be a sequence of family names such as "
                f"['substitution', 'vigenere'], not the single string "
                f"{methods!r}"
            )
        unknown = sorted({name.lower() for name in methods} - set(METHODS))
        if unknown:
            raise ValueError(
                f"unknown crib method(s): {', '.join(unknown)}. Valid names "
                f"are: {', '.join(METHODS)}"
            )
        wanted = {name.lower() for name in methods}
        chosen = tuple(name for name in METHODS if name in wanted)

    offsets = max(0, len(letters) - len(cleaned) + 1)
    report = CribReport(
        crib=cleaned,
        letters=letters,
        methods=chosen,
        offsets_tested=offsets,
        vigenere_key_length=key_length,
        no_fixed_points=no_fixed_points,
        limit=limit,
    )

    if "substitution" in chosen:
        report.substitution = substitution_placements(
            letters, cleaned, known=known, no_fixed_points=no_fixed_points
        )
    if "caesar" in chosen:
        report.caesar = caesar_placements(letters, cleaned)
    if "affine" in chosen:
        report.affine = affine_placements(letters, cleaned)
    if "vigenere" in chosen:
        fragments = vigenere_placements(letters, cleaned, key_length)
        report.vigenere = fragments
        report.vigenere_votes = key_length_votes(fragments)
        report.vigenere_words = word_key_fragments(
            consistent_fragments(fragments), scorer=scorer
        )
    if "transposition" in chosen:
        report.transposition = transposition_crib_help(letters, cleaned)

    return report
