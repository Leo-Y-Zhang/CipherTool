"""Atbash: the reversed alphabet. A<->Z, B<->Y, ..., M<->N.

The cipher
----------
Atbash writes the alphabet backwards under itself and swaps each letter for
the one below it. Numbering A..Z as 0..25 the whole cipher is one line of
arithmetic:

    E(x) = 25 - x

and that is also the decryption, because 25 - (25 - x) = x. Atbash is an
*involution*: enciphering twice returns the original text. It is the oldest
substitution cipher we know of, from Hebrew scribal practice, and it has no
key at all.

Where it sits in the family
---------------------------
Atbash is a special case of the affine cipher E(x) = (a*x + b) mod 26 with
a = 25 and b = 25, since

    25x + 25 = -x - 1 = 25 - x   (mod 26, because 25 = -1)

so ``affine.encrypt(text, 25, 25)`` and ``atbash.encrypt(text)`` agree letter
for letter (``tests/test_affine.py`` asserts exactly that). It is *not* a
Caesar shift: a shift moves the whole alphabet along, Atbash reflects it.

"Attacking" a cipher with no key
--------------------------------
There is nothing to search. The only useful work :func:`solve` can do is
decipher the one possible way and then report *honestly* whether the result
looks like English, so that an operator who tried Atbash on something that is
not Atbash gets told so instead of being handed confident nonsense. The
candidate therefore carries three pieces of evidence:

* the n-gram score and word coverage of the one reading (via ``annotate``);
* the chi-squared distance of that reading from English letter frequencies;
* the index of coincidence of the *ciphertext*, which is a prior check --
  Atbash only relabels letters, so IC is untouched by it. If the ciphertext IC
  is far from the English value of about 0.0667 then no monoalphabetic cipher,
  Atbash included, can turn it into English, and the operator should be
  looking at a polyalphabetic or fractionating cipher instead.
"""

from __future__ import annotations

from typing import Any

from .candidates import Candidate, CandidateSet
from .normalize import (
    ALPHABET_SIZE,
    NormalizedText,
    from_numbers,
    letters_only,
    normalize,
    to_numbers,
)
from .reference import ENGLISH_IC
from .scoring import EnglishScorer, annotate, default_scorer
from .statistics import chi_squared_english, index_of_coincidence

METHOD = "Atbash"

#: The fixed cipher alphabet, "ZYXWVUTSRQPONMLKJIHGFEDCBA".
ATBASH_ALPHABET = from_numbers(
    ALPHABET_SIZE - 1 - value for value in range(ALPHABET_SIZE)
)


def _require_text(text: str) -> str:
    """Reject non-string input loudly rather than half-processing it."""
    if not isinstance(text, str):
        raise ValueError(
            f"text must be a string, got {type(text).__name__}. Read the file "
            "first, or pass NormalizedText.letters."
        )
    return text


def encrypt(text: str) -> str:
    """Encrypt. Operates on letters only; returns letters only, uppercase.

    Atbash takes no key: every letter is reflected through the middle of the
    alphabet by ``x -> 25 - x``.
    """
    letters = letters_only(_require_text(text))
    return from_numbers(
        ALPHABET_SIZE - 1 - value for value in to_numbers(letters)
    )


def decrypt(text: str) -> str:
    """Exact inverse of :func:`encrypt`, which is :func:`encrypt` itself.

    Kept as a separate named function so that calling code reads the way the
    operator is thinking, and so the involution can be asserted in tests
    rather than assumed.
    """
    return encrypt(text)


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    time_budget: float | None = None,
    **options: Any,
) -> CandidateSet:
    """Decipher the one possible way and report honestly how English it looks.

    Returns a :class:`CandidateSet` holding exactly one candidate (or none, if
    the input has no letters). There is no search and therefore no ranking:
    read ``confidence()`` and the plaintext. A ``weak`` or ``unlikely`` label
    here means the ciphertext is simply not Atbash.

    ``top`` and ``time_budget`` are accepted for a uniform solver interface
    and have nothing to do here. Unknown options are ignored but recorded in
    the diagnostics as ``options_ignored``.
    """
    engine = scorer if scorer is not None else default_scorer()
    if isinstance(source, str):
        normalized = normalize(source)
    elif isinstance(source, NormalizedText):
        normalized = source
    else:
        raise ValueError(
            "solve() needs ciphertext as a string or a NormalizedText, got "
            f"{type(source).__name__}."
        )

    letters = normalized.letters
    if not letters:
        return CandidateSet()

    plaintext = encrypt(letters)
    ciphertext_ic = index_of_coincidence(letters)

    diagnostics: dict[str, Any] = {
        "keys_tested": 1,
        "fixed_key": True,
        "chi_squared": chi_squared_english(plaintext),
        "ciphertext_ic": ciphertext_ic,
        "english_ic": ENGLISH_IC,
        "note": (
            "Atbash has no key, so this is the only reading it can produce. "
            "Nothing was searched and nothing was optimised: judge this by "
            "the plaintext and the scores, not by its rank."
        ),
    }
    if abs(ciphertext_ic - ENGLISH_IC) > 0.015:
        # A monoalphabetic cipher preserves IC exactly, so an un-English IC in
        # the ciphertext rules Atbash out before the plaintext is even read.
        diagnostics["warning"] = (
            f"Ciphertext IC {ciphertext_ic:.4f} is far from English "
            f"({ENGLISH_IC:.4f}). Atbash only relabels letters and cannot "
            "change IC, so this text is unlikely to be a monoalphabetic "
            "cipher at all."
        )
    if options:
        diagnostics["options_ignored"] = ", ".join(sorted(options))

    annotate(diagnostics, plaintext, engine)

    candidates = CandidateSet()
    candidates.add(
        Candidate(
            method=METHOD,
            key="atbash (no key)",
            score=engine.score(plaintext),
            plaintext=plaintext,
            diagnostics=diagnostics,
            # One cipher letter per plaintext letter, so the recovered text
            # fits back into the original spacing and punctuation.
            display=normalized.relayout(plaintext),
        )
    )
    return candidates
