"""Keyword substitution: the cipher, and the attack on it.

(The module is called ``keyword_cipher`` and not ``keyword`` because
``keyword`` is a standard library module. Shadowing it would break any tool
in the same process that inspects Python reserved words.)

The cipher
----------
A keyword substitution is an ordinary monoalphabetic substitution whose
cipher alphabet is not chosen at random but grown from a memorable word.
Write the keyword down with any repeated letters dropped, then follow it
with every letter of the alphabet that has not been used yet, in order::

    keyword   SECRET
    reduced   SECRT
    remainder ABDFGHIJKLMNOPQUVWXYZ
    cipher    SECRTABDFGHIJKLMNOPQUVWXYZ

The cipher alphabet is then written under the plain alphabet, and enciphering
is a table lookup::

    plain   A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
    cipher  S E C R T A B D F G H I J K L M N O P Q U V W X Y Z

so ATTACKATDAWN becomes SQQSCHSQRSWK.

Two variants are in common use and both are supported here.

*Offset* (``start_letter``). The keyword need not begin under plain A. If it
begins under plain D the whole cipher row is rotated three places to the
right and the remainder wraps round the end back to the start. This costs the
cryptographer nothing and multiplies the keyspace by 26. It also removes the
cipher's most obvious weakness: with ``start_letter='A'`` the tail of the
alphabet is usually a long stretch of letters that encipher to themselves.

*Reversed remainder* (``reverse_remainder``). Instead of filling the unused
letters forwards from A, fill them backwards from Z::

    keyword   SECRET
    cipher    SECRTZYXWVUQPONMLKJIHGFDBA

This is sometimes called a decimated or reversed-remainder alphabet. It is
popular precisely because the plain fill leaves the end of the alphabet
almost unchanged.

Why the cipher is weak
----------------------
As a substitution it is exactly as strong as any monoalphabetic cipher
against a hill climber -- letter frequencies leak, and the keyspace of 26!
(about 4 x 10^26) is irrelevant because the search never needs to enumerate
it. But choosing the alphabet from a word throws the keyspace away outright.
There are only about four thousand plausible English keywords; times 26 start
letters times two remainder directions that is around two hundred thousand
alphabets, and we can test each one with a single decryption and an n-gram
score. In practice the default search here -- one start letter, both
remainder directions, every dictionary word of three to twelve letters -- is
a few thousand trials and finishes in seconds.

The two attacks in this module
------------------------------
1. *Dictionary search* (:func:`solve` with the default ``words=None``).
   Build the alphabet for every candidate keyword, decrypt, score with the
   local English model, keep the best. This is complete with respect to the
   word list: if the keyword is an English word we hold, we will find it.

2. *Keyword recovery from a recovered alphabet*
   (:func:`candidate_keywords`). Suppose a general substitution solver -- the
   hill climber in ``substitution.py`` -- has already recovered the cipher
   alphabet. Was it built from a keyword, and if so which? The structure to
   look for is the tail: everything after the keyword is in strict
   alphabetical order. So find the longest strictly ordered suffix; whatever
   precedes it is the keyword.

   How much evidence is a long ordered tail? For a permutation chosen at
   random the chance that its last T letters happen to ascend is exactly
   1/T!, because all T! orderings of those letters are equally likely. A tail
   of six is 1 in 720, a tail of ten is 1 in 3.6 million, and a tail of
   twenty is 1 in 2 x 10^18. That is why :func:`candidate_keywords` reports
   the tail length and 1/T! alongside every suggestion, and why it refuses by
   default to report a tail shorter than six letters: it would be noise. Note
   that we test 52 alignments (26 rotations by two directions), so multiply
   1/T! by 52 for the chance that *some* alignment fires by accident.

   Three honest limitations of the tail method:

   * The keyword loses its repeated letters when the alphabet is built, so
     SECRET can only ever be recovered as SECRT. We therefore also report
     which dictionary words reduce to the recovered stem, which is how
     SECRET is put back.
   * If the keyword's last letter happens to fall alphabetically before the
     first letter of the remainder, the ordered tail swallows it. ZEBRA
     gives ZEBR|ACDFGH..., so the longest tail suggests the keyword ZEBR.
     For that reason we report the shortest keyword and a few one-letter
     extensions of it, ranked with dictionary words first.
   * A keyword alphabet does not have one description, it has a family of
     them. Cutting the same cyclic row one place further along turns
     SECRT|ABDF...YZ into Z|SECRT|ABDF...Y, which is the perfectly valid key
     "keyword ZSECRT starting under Z" and rebuilds the identical alphabet.
     Those rotations are marked (:attr:`KeywordRecovery.rotated_duplicate`)
     and ranked below the un-rotated readings, because the reading with the
     longest tail is the one a human is likely to have started from.

   One practical warning about attack 2. A hill climber recovers the common
   letters reliably and the rare ones barely at all: if J, Q, X and Z do not
   occur in the plaintext, nothing in the ciphertext says where they belong,
   so the recovered alphabet is usually right except for a couple of rare
   letters in the wrong places. That is enough to break the ordered tail and
   shift the recovered keyword by a letter or two -- PORTCULIS can come back
   as APORTCULISZ. Read the suggestion as a strong hint, then confirm it by
   putting the keyword back through :func:`decrypt`.

Recovering the keyword is worth doing even after the plaintext is already
readable: in the National Cipher Challenge the keyword is usually a word from
the story, and reading it back confirms the solve and often gives away the
setting of the next part.
"""

from __future__ import annotations

import heapq
import math
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

from .candidates import Candidate, CandidateSet
from .normalize import (
    ALPHABET,
    ALPHABET_SIZE,
    NormalizedText,
    clean_key,
    letters_only,
    normalize,
)
from .scoring import EnglishScorer, annotate, default_scorer
from .statistics import chi_squared_english, index_of_coincidence

__all__ = [
    "deduplicate",
    "keyword_alphabet",
    "invert_alphabet",
    "validate_alphabet",
    "encrypt",
    "decrypt",
    "encrypt_with_alphabet",
    "decrypt_with_alphabet",
    "KeywordRecovery",
    "candidate_keywords",
    "describe_alphabet",
    "solve",
]

#: Shortest ordered tail we are willing to call evidence of a keyword.
#: 1/6! = 1 in 720 per alignment, so about one false alarm in fourteen
#: alphabets given the 52 alignments we test. Below this the method is noise.
DEFAULT_MINIMUM_TAIL = 6

#: Dictionary words shorter than this make hopeless keywords (a two-letter
#: keyword barely disturbs the alphabet), and longer than this are vanishingly
#: rare in practice.
DEFAULT_MIN_KEYWORD = 3
DEFAULT_MAX_KEYWORD = 12


# ---------------------------------------------------------------------------
# Building the alphabet
# ---------------------------------------------------------------------------


def deduplicate(keyword: str) -> str:
    """Drop repeated letters from *keyword*, keeping the first occurrence.

    ``deduplicate("SECRET")`` -> ``"SECRT"``. Input is cleaned first, so
    lower case, spaces and punctuation are all acceptable.
    """
    seen: set[str] = set()
    out: list[str] = []
    for letter in clean_key(keyword):
        if letter not in seen:
            seen.add(letter)
            out.append(letter)
    return "".join(out)


def keyword_alphabet(
    keyword: str,
    start_letter: str = "A",
    reverse_remainder: bool = False,
) -> str:
    """Build the 26-letter cipher alphabet generated by *keyword*.

    Parameters
    ----------
    keyword:
        The key word or phrase. Cleaned to A-Z and de-duplicated. Must
        contain at least one letter.
    start_letter:
        Plain letter under which the keyword is written. ``'A'`` is the
        textbook construction; anything else rotates the finished row so that
        the keyword's first letter sits under *start_letter* and the
        remainder wraps round the end of the alphabet.
    reverse_remainder:
        If true, fill the letters not used by the keyword backwards from Z
        instead of forwards from A.

    Returns
    -------
    A 26-character string. Position *i* holds the cipher letter for plain
    letter *i*, i.e. it is the "cipher" row of the enciphering table.

    Examples
    --------
    >>> keyword_alphabet("SECRET")
    'SECRTABDFGHIJKLMNOPQUVWXYZ'
    >>> keyword_alphabet("SECRET", reverse_remainder=True)
    'SECRTZYXWVUQPONMLKJIHGFDBA'

    Raises
    ------
    ValueError
        If the keyword has no letters, or *start_letter* is not exactly one
        letter.
    """
    stem = deduplicate(keyword)
    if not stem:
        raise ValueError(
            "keyword must contain at least one letter A-Z; "
            f"got {keyword!r}, which cleans to an empty string"
        )

    start = clean_key(start_letter)
    if len(start) != 1:
        raise ValueError(
            "start_letter must be exactly one letter A-Z; "
            f"got {start_letter!r}, which cleans to {start!r}"
        )

    remainder = [letter for letter in ALPHABET if letter not in set(stem)]
    if reverse_remainder:
        remainder.reverse()

    # `sequence` is the alphabet as it would be written with the keyword at
    # position 0; the offset then rotates it so the keyword starts under
    # `start_letter`, with the tail wrapping round to the front.
    sequence = stem + "".join(remainder)
    offset = ord(start) - 65
    row = [""] * ALPHABET_SIZE
    for index, letter in enumerate(sequence):
        row[(offset + index) % ALPHABET_SIZE] = letter
    return "".join(row)


def validate_alphabet(alphabet: str) -> str:
    """Check *alphabet* is a genuine permutation of A-Z and return it cleaned.

    Raises ``ValueError`` naming the problem, because a substitution table
    with a missing or duplicated letter is not invertible and every silent
    consequence of that is worse than an early failure.
    """
    cleaned = letters_only(alphabet)
    if len(cleaned) != ALPHABET_SIZE:
        raise ValueError(
            f"cipher alphabet must have {ALPHABET_SIZE} letters, "
            f"got {len(cleaned)} from {alphabet!r}"
        )
    missing = sorted(set(ALPHABET) - set(cleaned))
    if missing:
        duplicated = sorted({c for c in cleaned if cleaned.count(c) > 1})
        raise ValueError(
            "cipher alphabet must use each letter exactly once; "
            f"missing {''.join(missing)}, repeated {''.join(duplicated)}"
        )
    return cleaned


def invert_alphabet(alphabet: str) -> str:
    """Return the alphabet that undoes *alphabet*.

    If ``alphabet[i]`` is the cipher letter for plain letter *i*, the inverse
    has ``inverse[j]`` = the plain letter for cipher letter *j*. Inverting
    twice gives the original back.
    """
    cleaned = validate_alphabet(alphabet)
    row = [""] * ALPHABET_SIZE
    for plain_index, cipher_letter in enumerate(cleaned):
        row[ord(cipher_letter) - 65] = ALPHABET[plain_index]
    return "".join(row)


# ---------------------------------------------------------------------------
# Encryption and decryption
# ---------------------------------------------------------------------------


def encrypt_with_alphabet(text: str, alphabet: str) -> str:
    """Encipher *text* with an explicit cipher alphabet (letters only, upper)."""
    cleaned = validate_alphabet(alphabet)
    return "".join(cleaned[ord(ch) - 65] for ch in letters_only(text))


def decrypt_with_alphabet(text: str, alphabet: str) -> str:
    """Decipher *text* with an explicit cipher alphabet (letters only, upper)."""
    return encrypt_with_alphabet(text, invert_alphabet(alphabet))


def encrypt(
    text: str,
    keyword: str,
    start_letter: str = "A",
    reverse_remainder: bool = False,
) -> str:
    """Encrypt. Operates on letters only; returns letters only, uppercase.

    >>> encrypt("attack at dawn", "SECRET")
    'SQQSCHSQRSWK'
    """
    return encrypt_with_alphabet(
        text, keyword_alphabet(keyword, start_letter, reverse_remainder)
    )


def decrypt(
    text: str,
    keyword: str,
    start_letter: str = "A",
    reverse_remainder: bool = False,
) -> str:
    """Exact inverse of :func:`encrypt` for the same key.

    >>> decrypt("SQQSCHSQRSWK", "SECRET")
    'ATTACKATDAWN'
    """
    return decrypt_with_alphabet(
        text, keyword_alphabet(keyword, start_letter, reverse_remainder)
    )


# ---------------------------------------------------------------------------
# Keyword recovery from a recovered alphabet
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeywordRecovery:
    """One hypothesis about which keyword generated a cipher alphabet.

    Attributes
    ----------
    keyword:
        The recovered stem, with repeated letters already gone. Feed it back
        to :func:`keyword_alphabet` with the same *start_letter* and
        *reverse_remainder* to reproduce the alphabet exactly.
    start_letter:
        Plain letter the keyword sits under.
    reverse_remainder:
        Whether the unused letters were filled backwards from Z.
    tail_length:
        Length of the strictly ordered run that gave the keyword away. This
        is the evidence; everything else is bookkeeping.
    chance_probability:
        1/tail_length! -- the chance a random permutation would show a tail
        this long at this one alignment. Small means "unlikely to be luck".
        Multiply by 52 for the chance across every alignment we test.
    lexicon_words:
        Dictionary words that reduce to :attr:`keyword` when their repeated
        letters are removed. This is how SECRT is read back as SECRET.
    rotated_duplicate:
        True when this reading is the same cyclic row cut in a worse place:
        its first keyword letter would itself continue the ordered tail, so
        another reading of the same alphabet has a longer tail. Still a valid
        key, but usually an artefact rather than what the setter wrote down.
    alphabet:
        The alphabet this hypothesis reproduces.
    """

    keyword: str
    start_letter: str
    reverse_remainder: bool
    tail_length: int
    chance_probability: float
    lexicon_words: tuple[str, ...]
    rotated_duplicate: bool
    alphabet: str

    @property
    def is_word(self) -> bool:
        """True if some real word reduces to this keyword stem."""
        return bool(self.lexicon_words)

    def key_string(self) -> str:
        """Copy-pasteable key description, matching :func:`solve` output."""
        word = self.lexicon_words[0] if self.lexicon_words else self.keyword
        parts = [f"keyword={word}"]
        if self.start_letter != "A":
            parts.append(f"start={self.start_letter}")
        if self.reverse_remainder:
            parts.append("remainder=reverse")
        return " ".join(parts)

    def describe(self) -> str:
        """One-line human summary including the strength of the evidence."""
        words = f" (from {', '.join(self.lexicon_words)})" if self.lexicon_words else ""
        rotated = " [rotation of a longer-tailed reading]" if self.rotated_duplicate else ""
        return (
            f"{self.key_string()}{words}; ordered tail of {self.tail_length}, "
            f"1 in {1.0 / self.chance_probability:,.0f} by chance{rotated}"
        )


def _ordered_run_start(sequence: str, descending: bool) -> int:
    """Index at which the longest strictly ordered suffix of *sequence* begins.

    Walk backwards while each letter still continues the run. Returns 0 when
    the whole sequence is ordered.
    """
    index = len(sequence) - 1
    while index > 0:
        previous, current = sequence[index - 1], sequence[index]
        in_order = previous > current if descending else previous < current
        if not in_order:
            break
        index -= 1
    return index


@lru_cache(maxsize=4)
def _stem_index(lexicon: frozenset[str]) -> dict[str, tuple[str, ...]]:
    """Map each de-duplicated word stem to the words that produce it.

    Cached because it is rebuilt identically for every alphabet analysed and
    the lexicon does not change within a run.
    """
    index: dict[str, list[str]] = {}
    for word in lexicon:
        index.setdefault(deduplicate(word), []).append(word)
    # Shortest first, then alphabetical: SECRT -> SECRET before SECRETARY.
    return {
        stem: tuple(sorted(words, key=lambda w: (len(w), w)))
        for stem, words in index.items()
    }


def candidate_keywords(
    alphabet: str,
    *,
    minimum_tail: int = DEFAULT_MINIMUM_TAIL,
    extend: int = 3,
    maximum_keyword: int = 15,
    lexicon: Iterable[str] | None = None,
    limit: int | None = 10,
) -> list[KeywordRecovery]:
    """Work backwards from a recovered cipher alphabet to its keyword.

    Given a full 26-letter substitution alphabet -- typically one a hill
    climber has just recovered -- decide whether it could have been built
    from a keyword, and if so from which.

    Method. For each of the 26 possible start letters and both remainder
    directions, undo the rotation and look for the longest strictly ordered
    suffix. If the suffix has *minimum_tail* letters or more the alphabet has
    keyword structure at that alignment, and everything before the suffix is
    the keyword. Because a keyword's final letter can be swallowed by the
    ordered tail (see the module docstring: ZEBRA -> ZEBR), we also offer up
    to *extend* one-letter-longer readings of the same alignment; each of
    them regenerates the identical alphabet, so they are equally valid keys
    and only the dictionary can choose between them.

    Results are ranked: hypotheses whose stem matches a real word first, then
    shorter keywords (a shorter keyword means a longer ordered tail, which is
    stronger evidence), then the plain construction before the variants.

    Parameters
    ----------
    alphabet:
        26 letters, each exactly once. Anything else raises ``ValueError``.
    minimum_tail:
        Refuse to report an alignment whose ordered tail is shorter than
        this. The default of 6 corresponds to 1 chance in 720 per alignment.
    extend:
        How many longer readings of each alignment to offer.
    maximum_keyword:
        Never report a keyword longer than this.
    lexicon:
        Words used to recognise a stem. Defaults to the shared scorer's
        lexicon; pass an empty sequence to switch dictionary ranking off.
    limit:
        Truncate the ranked list. Defaults to 10 because one keyword alphabet
        has dozens of equally valid rotated readings and only the head of the
        list is worth a human's time; pass ``None`` for all of them.

    Returns
    -------
    A ranked list, empty if the alphabet shows no keyword structure at all --
    which is the honest answer for an alphabet that was chosen at random.
    """
    cleaned = validate_alphabet(alphabet)
    if minimum_tail < 1:
        raise ValueError(f"minimum_tail must be at least 1, got {minimum_tail}")
    if extend < 0:
        raise ValueError(f"extend must not be negative, got {extend}")

    words = frozenset(lexicon) if lexicon is not None else default_scorer().lexicon
    stems = _stem_index(words)

    found: list[KeywordRecovery] = []
    for descending in (False, True):
        for offset in range(ALPHABET_SIZE):
            # Undo the rotation: `sequence` is keyword followed by remainder.
            sequence = cleaned[offset:] + cleaned[:offset]
            shortest = _ordered_run_start(sequence, descending)
            tail = ALPHABET_SIZE - shortest
            if tail < minimum_tail:
                continue

            # Is this just the same cyclic row cut one place too early? It is
            # if the first keyword letter would carry on the ordered tail --
            # cutting one place further along moves that letter to the end of
            # the tail and makes the tail longer. Neither letter changes when
            # we extend the keyword below, so this is a property of the
            # alignment and is computed once.
            head, last = sequence[0], sequence[-1]
            rotated = head < last if descending else head > last

            # Every length from `shortest` up is a genuine key, because a
            # suffix of an ordered run is still ordered. Each one rebuilds
            # `cleaned` exactly: the stem's letters are distinct (it is a
            # slice of a permutation) so nothing is lost to de-duplication,
            # what follows it is by construction the unused letters in the
            # required order, and the rotation puts them back where they
            # came from. tests/test_keyword_cipher.py checks that claim
            # against every suggestion rather than trusting the argument.
            longest = min(shortest + extend, maximum_keyword, ALPHABET_SIZE - 1)
            for length in range(max(shortest, 1), longest + 1):
                stem = sequence[:length]
                start_letter = ALPHABET[offset]
                found.append(
                    KeywordRecovery(
                        keyword=stem,
                        start_letter=start_letter,
                        reverse_remainder=descending,
                        tail_length=ALPHABET_SIZE - length,
                        chance_probability=1.0
                        / math.factorial(ALPHABET_SIZE - length),
                        lexicon_words=stems.get(stem, ()),
                        rotated_duplicate=rotated,
                        alphabet=cleaned,
                    )
                )

    # Ranking, most believable first. A stem that is a real word beats
    # everything: keyword alphabets are built from words, and a random
    # alignment landing on one is a coincidence twice over. After that,
    # prefer the readings that are not merely rotations, then the longest
    # ordered tail (shortest keyword), then the textbook construction.
    found.sort(
        key=lambda r: (
            not r.lexicon_words,
            r.rotated_duplicate,
            len(r.keyword),
            r.start_letter != "A",
            r.reverse_remainder,
            r.keyword,
        )
    )
    return found[:limit] if limit is not None else found


def describe_alphabet(alphabet: str, *, limit: int | None = 10) -> str:
    """Human-readable verdict on whether an alphabet came from a keyword."""
    results = candidate_keywords(alphabet, limit=limit)
    if not results:
        return (
            f"{validate_alphabet(alphabet)}: no keyword structure "
            "(no long alphabetical tail at any alignment)"
        )
    lines = [f"{validate_alphabet(alphabet)}: keyword structure found"]
    lines.extend(f"  - {result.describe()}" for result in results)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The attack
# ---------------------------------------------------------------------------


def _default_words(scorer: EnglishScorer) -> list[str]:
    """Every dictionary word of a plausible keyword length, SHORTEST first.

    Order decides the answer, because several words can reduce to the same
    cipher alphabet and whichever is tried first becomes the label reported.
    Repeated letters are dropped when a keyword alphabet is built, so
    LIGHTHOUSE and LIGHTHOUSES both give LIGHTOUSE followed by the remaining
    alphabet, and no ciphertext can tell them apart.

    This ran longest first, on the reasoning that the longer word was "more
    informative". It is not. It is a longer claim explaining exactly the same
    evidence, and reporting keyword=LIGHTHOUSES for a message enciphered with
    LIGHTHOUSE is simply wrong where the shorter answer would have been
    right. Shortest first prefers the smallest claim that fits the evidence.
    """
    return sorted(
        (
            word
            for word in scorer.lexicon
            if DEFAULT_MIN_KEYWORD <= len(word) <= DEFAULT_MAX_KEYWORD
        ),
        key=lambda w: (len(w), w),
    )


_ALPHABET_RUN = re.compile(r"[A-Za-z]{26,}")


def _alphabets_in(candidate: Candidate) -> list[str]:
    """Pull any 26-letter permutation out of another solver's candidate.

    ``substitution.solve`` is free to report its key however it likes, and it
    may report either direction of the table, so we look in the diagnostics
    and in the key string, and we analyse both the alphabet and its inverse.
    Anything that is not a permutation of A-Z is ignored.
    """
    texts: list[str] = []
    for name in ("alphabet", "cipher_alphabet", "plain_alphabet", "key"):
        value = candidate.diagnostics.get(name)
        if isinstance(value, str):
            texts.append(value)
    texts.append(candidate.key)

    out: list[str] = []
    for text in texts:
        for run in _ALPHABET_RUN.findall(text):
            upper = run.upper()
            for start in range(len(upper) - ALPHABET_SIZE + 1):
                window = upper[start : start + ALPHABET_SIZE]
                if len(set(window)) == ALPHABET_SIZE:
                    for form in (window, invert_alphabet(window)):
                        if form not in out:
                            out.append(form)
    return out


def _keyword_note(alphabet: str, lexicon: Iterable[str] | None) -> tuple[str, str]:
    """Return ``(key_string, evidence)`` describing an alphabet's keyword."""
    results = candidate_keywords(alphabet, lexicon=lexicon, limit=3)
    if not results:
        return f"alphabet={alphabet}", "no keyword structure"
    best = results[0]
    others = "; ".join(r.key_string() for r in results[1:])
    evidence = best.describe()
    if others:
        evidence += f"; also possible: {others}"
    return best.key_string(), evidence


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    words: Sequence[str] | None = None,
    start_letters: str = "A",
    try_reversed: bool = True,
    alphabets: Sequence[str] | None = None,
    hill_climb: bool = False,
    hill_climb_top: int = 5,
    seed: int | None = None,
    time_budget: float | None = None,
) -> CandidateSet:
    """Attack a keyword substitution and return ranked candidates.

    The main attack is exhaustive over a word list: build the alphabet for
    every candidate keyword, decrypt, and score with the local English model.
    With the default word list this is a few thousand trials, so it is worth
    running even when the keyword is a long shot.

    Options
    -------
    words:
        Keywords to try. ``None`` (the default) means every word in the
        scorer's lexicon of length 3 to 12 -- around 3800 of them.
    start_letters:
        Which plain letters to try the keyword under. ``"A"`` (the default)
        is the textbook construction. Pass ``ALPHABET`` to sweep all 26,
        which costs 26 times as much.
    try_reversed:
        Also try the reversed-remainder variant. On by default; it only
        doubles the work.
    alphabets:
        Cipher alphabets recovered by some other means. Each is decrypted
        with and analysed for keyword structure. Use this to ask "the hill
        climber found this alphabet -- what keyword made it?".
    hill_climb:
        Run ``substitution.solve`` as well and analyse the alphabets it
        recovers. Off by default because it is far slower than the dictionary
        search and because it only helps when the keyword is *not* an English
        word we hold. Raises ``ImportError`` with an explanation if the
        substitution module is not present. Remember that a climbed alphabet
        is usually wrong in the letters that never occur, so the keyword it
        suggests can be a letter or two adrift.
    hill_climb_top:
        How many of the hill climber's candidates to analyse.
    seed:
        Forwarded to ``substitution.solve`` so hill-climbing runs reproduce.
    time_budget:
        Seconds. The dictionary search checks the clock as it goes and stops
        cleanly, recording ``time_budget_hit`` in the diagnostics of every
        candidate it returns.

    Returns
    -------
    A :class:`CandidateSet`, empty if the input has no letters. The evidence
    on each candidate records the alphabet used, how the keyword was found,
    and how much of the search actually ran.
    """
    if top < 1:
        raise ValueError(f"top must be at least 1, got {top}")
    if time_budget is not None and time_budget <= 0:
        raise ValueError(f"time_budget must be positive, got {time_budget}")

    starts = clean_key(start_letters)
    if not starts:
        raise ValueError(
            f"start_letters must contain at least one letter A-Z, got {start_letters!r}"
        )

    engine = scorer or default_scorer()
    text = normalize(source) if isinstance(source, str) else source
    letters = text.letters
    results = CandidateSet()
    if not letters:
        return results

    values = engine.encode(letters)
    started = time.monotonic()
    budget_hit = False

    # -- 1. dictionary search over keywords --------------------------------
    #
    # Each trial is one alphabet, one table-driven decryption and one n-gram
    # score. We keep only a small heap of the best trials, because the
    # expensive part of judging a candidate (word-coverage segmentation) is
    # not worth paying 7000 times.
    keep = max(top * 3, 10)
    heap: list[tuple[float, int, str, str, str, bool]] = []
    counter = 0
    seen_alphabets: set[str] = set()
    duplicates = 0

    wordlist = list(words) if words is not None else _default_words(engine)
    cleaned_words = [w for w in (clean_key(word) for word in wordlist) if w]
    if words is not None and not cleaned_words:
        raise ValueError(
            "words was given but contains no usable keywords "
            "(every entry cleaned to an empty string)"
        )

    directions = (False, True) if try_reversed else (False,)
    tried = 0
    for word in cleaned_words:
        if budget_hit:
            break
        for start_letter in starts:
            for reversed_tail in directions:
                if time_budget is not None and (counter & 0x3F) == 0:
                    if time.monotonic() - started > time_budget:
                        budget_hit = True
                        break
                counter += 1
                alphabet = keyword_alphabet(word, start_letter, reversed_tail)
                if alphabet in seen_alphabets:
                    duplicates += 1
                    continue
                seen_alphabets.add(alphabet)
                tried += 1

                # Decrypt in the integer domain: inverse[c] is the plain
                # value for cipher value c.
                inverse = [0] * ALPHABET_SIZE
                for plain_index, cipher_letter in enumerate(alphabet):
                    inverse[ord(cipher_letter) - 65] = plain_index
                plain_values = [inverse[value] for value in values]
                score = engine.score_values(plain_values)

                entry = (score, counter, word, alphabet, start_letter, reversed_tail)
                if len(heap) < keep:
                    heapq.heappush(heap, entry)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, entry)
            if budget_hit:
                break

    for score, _, word, alphabet, start_letter, reversed_tail in heap:
        plaintext = decrypt_with_alphabet(letters, alphabet)
        key_parts = [f"keyword={word}"]
        if start_letter != "A":
            key_parts.append(f"start={start_letter}")
        if reversed_tail:
            key_parts.append("remainder=reverse")
        diagnostics: dict[str, object] = {
            "alphabet": alphabet,
            "keyword_stem": deduplicate(word),
            "start_letter": start_letter,
            "remainder": "reverse" if reversed_tail else "forward",
            "keywords_tried": tried,
            "duplicate_alphabets_skipped": duplicates,
            "source": "dictionary search",
            "ciphertext_ic": index_of_coincidence(letters),
            "chi_squared_plain": chi_squared_english(plaintext),
        }
        if budget_hit:
            diagnostics["time_budget_hit"] = True
        annotate(diagnostics, plaintext, engine)
        results.add(
            Candidate(
                method="Keyword substitution",
                key=" ".join(key_parts),
                score=score,
                plaintext=plaintext,
                diagnostics=diagnostics,
                display=text.relayout(plaintext),
            )
        )

    # -- 2. analyse alphabets recovered elsewhere ---------------------------
    recovered: list[tuple[str, str]] = [
        (alphabet, "supplied alphabet") for alphabet in (alphabets or ())
    ]

    if hill_climb:
        try:
            from . import substitution  # noqa: PLC0415  (optional companion module)
        except ImportError as error:  # pragma: no cover - depends on install
            raise ImportError(
                "hill_climb=True needs cipher_tool.substitution, which is not "
                "importable. Run solve() without hill_climb to use the "
                "dictionary search alone."
            ) from error
        remaining = None
        if time_budget is not None:
            remaining = max(0.1, time_budget - (time.monotonic() - started))
        climbed = substitution.solve(
            text,
            scorer=engine,
            top=hill_climb_top,
            seed=seed,
            time_budget=remaining,
        )
        for candidate in climbed.top(hill_climb_top):
            for alphabet in _alphabets_in(candidate):
                recovered.append((alphabet, "substitution hill climb"))

    for alphabet, provenance in recovered:
        cleaned_alphabet = validate_alphabet(alphabet)
        plaintext = decrypt_with_alphabet(letters, cleaned_alphabet)
        key_string, evidence = _keyword_note(cleaned_alphabet, engine.lexicon)
        diagnostics = {
            "alphabet": cleaned_alphabet,
            "source": provenance,
            "keyword_analysis": evidence,
            "ciphertext_ic": index_of_coincidence(letters),
            "chi_squared_plain": chi_squared_english(plaintext),
        }
        annotate(diagnostics, plaintext, engine)
        results.add(
            Candidate(
                method="Keyword substitution",
                key=key_string,
                score=engine.score(plaintext),
                plaintext=plaintext,
                diagnostics=diagnostics,
                display=text.relayout(plaintext),
            )
        )

    return results
