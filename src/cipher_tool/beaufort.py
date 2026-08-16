"""Beaufort and variant Beaufort: the ciphers and the attack on both.

Three periodic ciphers, one alphabet, three different sums
----------------------------------------------------------
All three of the classical periodic ciphers below take a short key, repeat it
under the plaintext, and add or subtract letter values modulo 26. They differ
only in the direction of the arithmetic, and that small difference changes how
each one must be attacked.

    Vigenere            C_i = (P_i + K_i) mod 26
    variant Beaufort    C_i = (P_i - K_i) mod 26
    Beaufort            C_i = (K_i - P_i) mod 26

Beaufort is self-reciprocal (its own inverse)
---------------------------------------------
Substitute the Beaufort rule into itself:

    encrypt(encrypt(P)) = K - (K - P) = P   (mod 26)

so encrypting a Beaufort ciphertext a second time with the same key returns
the plaintext. One routine does both jobs, which is exactly why the cipher was
convenient on a Royal Navy slide rule. Variant Beaufort is NOT self-reciprocal:
its inverse is P_i = (C_i + K_i) mod 26, which is Vigenere encryption. Put
another way, variant Beaufort encryption and Vigenere decryption are the same
operation, and Beaufort is the only one of the three whose encryption and
decryption routines coincide.

Why the column attack needs care
--------------------------------
Write the ciphertext into L columns, where column j holds every letter
enciphered by key letter K_j. If L is the key length then each column was
enciphered by a single fixed rule, so each column can be attacked on its own by
letter frequency. That much is the same for all three ciphers. What is NOT the
same is the rule inside a column:

    Vigenere            P = C - K        a shift of the column by -K
    variant Beaufort    P = C + K        a shift of the column by +K
    Beaufort            P = K - C        a REFLECTION of the column, then +K

The Beaufort case is not a shift at all: the map C -> K - C reverses the
alphabet as well as moving it. A solver that assumes "each column is a Caesar
shift, so try all 26 shifts and take the best chi-squared" therefore finds
nothing on Beaufort ciphertext, because no shift of the column produces
English -- the reflection has to be undone first. This module derives the
column rule separately for each variant instead of reusing one shift solver,
and ``tests/test_beaufort.py`` contains the experiment that observes the wrong
solver failing on Beaufort text.

Key length is found the same way for both
-----------------------------------------
The index of coincidence counts how often two letters drawn from a text agree.
Any bijection of the alphabet -- a shift, a reflection, or both -- relabels
letters without merging or splitting them, so it leaves every letter count
intact under a permutation and therefore leaves IC unchanged. That is why
:func:`cipher_tool.statistics.ic_by_period` works on Beaufort exactly as it
works on Vigenere: at the true period every column is monoalphabetic and shows
English-like IC (~0.067), and at a wrong period the columns mix alphabets and
fall back towards random (~0.038). Kasiski works for the same reason: a
repeated plaintext run that lands on the same key offset still gives an
identical ciphertext run, whichever direction the arithmetic runs in.

The attack in this module, in order
-----------------------------------
1. For every candidate key length, derive one key letter per column by
   minimising chi-squared against English letter frequencies, using the
   correct per-variant rule above.
2. Score the resulting decryption with the order-3 English model.
3. Take the most promising few and hill-climb the key letter by letter under
   the n-gram score. Chi-squared uses single-letter statistics only, so it
   misreads short columns; the n-gram model reads context and fixes the one or
   two letters chi-squared usually gets wrong.
4. Return every key length tried, for both variants, ranked together.
"""

from __future__ import annotations

import time
from typing import Sequence

from .candidates import Candidate, CandidateSet
from .normalize import (
    ALPHABET_SIZE,
    NormalizedText,
    clean_key,
    columns,
    from_numbers,
    letters_only,
    normalize,
    to_numbers,
)
from .scoring import EnglishScorer, annotate, default_scorer
from .statistics import chi_squared_english, ic_by_period, kasiski_factor_votes

#: Canonical variant names accepted by :func:`solve`.
BEAUFORT = "beaufort"
VARIANT = "variant"
VARIANTS: tuple[str, str] = (BEAUFORT, VARIANT)

#: Spellings a user or the CLI might reasonably pass for each variant.
_VARIANT_ALIASES = {
    "beaufort": BEAUFORT,
    "true": BEAUFORT,
    "standard": BEAUFORT,
    "variant": VARIANT,
    "variant-beaufort": VARIANT,
    "variant_beaufort": VARIANT,
    "beaufort-variant": VARIANT,
    "german": VARIANT,
}

#: Human-readable labels used in ``Candidate.method`` and ``Candidate.key``.
_METHOD_NAME = {BEAUFORT: "Beaufort", VARIANT: "Variant Beaufort"}
_KEY_LABEL = {BEAUFORT: "beaufort", VARIANT: "variant-beaufort"}

#: A candidate at or above this per-letter n-gram score is worth reading.
#: Measured English sits near -1.0 and wrong keys near -2.7 (see
#: ``candidates.py``), so this threshold is deliberately generous.
ENGLISH_THRESHOLD = -1.80


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------


def _require_key(key: str) -> str:
    """Normalise a key and refuse an empty one.

    An empty key would silently become "no encryption at all", which is the
    kind of quiet nonsense that loses a competition round.
    """
    cleaned = clean_key(key)
    if not cleaned:
        raise ValueError(
            f"A Beaufort key must contain at least one letter A-Z; {key!r} "
            "normalises to an empty key."
        )
    return cleaned


def _resolve_variant(name: str) -> str:
    """Map a user-supplied variant name onto :data:`BEAUFORT` or :data:`VARIANT`."""
    resolved = _VARIANT_ALIASES.get(str(name).strip().lower())
    if resolved is None:
        raise ValueError(
            f"Unknown Beaufort variant {name!r}. Use one of: "
            + ", ".join(sorted(set(_VARIANT_ALIASES)))
        )
    return resolved


def _combine(text: str, key: str, operation: str) -> str:
    """Apply one of the three periodic rules to *text* under *key*.

    ``operation`` selects the arithmetic:

    * ``"key_minus_text"`` -- Beaufort, ``K - X``
    * ``"text_minus_key"`` -- variant Beaufort encryption / Vigenere decryption
    * ``"text_plus_key"``  -- Vigenere encryption / variant Beaufort decryption
    """
    letters = letters_only(text)
    key_values = to_numbers(_require_key(key))
    length = len(key_values)
    values = to_numbers(letters)

    if operation == "key_minus_text":
        out = [key_values[i % length] - value for i, value in enumerate(values)]
    elif operation == "text_minus_key":
        out = [value - key_values[i % length] for i, value in enumerate(values)]
    elif operation == "text_plus_key":
        out = [value + key_values[i % length] for i, value in enumerate(values)]
    else:  # pragma: no cover - programming error, not user input
        raise ValueError(f"Unknown combination rule {operation!r}")
    # from_numbers reduces modulo 26, so negative values are handled there.
    return from_numbers(out)


# ---------------------------------------------------------------------------
# The ciphers
# ---------------------------------------------------------------------------


def beaufort_encrypt(text: str, key: str) -> str:
    """Beaufort encryption: ``C_i = (K_i - P_i) mod 26``.

    Operates on letters only; returns letters only, uppercase.
    """
    return _combine(text, key, "key_minus_text")


def beaufort_decrypt(text: str, key: str) -> str:
    """Beaufort decryption: ``P_i = (K_i - C_i) mod 26``.

    Identical arithmetic to :func:`beaufort_encrypt`, because Beaufort is
    self-reciprocal. The separate name exists so that calling code reads
    honestly; it is not a different computation.
    """
    return _combine(text, key, "key_minus_text")


def variant_encrypt(text: str, key: str) -> str:
    """Variant Beaufort encryption: ``C_i = (P_i - K_i) mod 26``.

    This is the same arithmetic as Vigenere *decryption*.
    """
    return _combine(text, key, "text_minus_key")


def variant_decrypt(text: str, key: str) -> str:
    """Variant Beaufort decryption: ``P_i = (C_i + K_i) mod 26``.

    This is the same arithmetic as Vigenere *encryption*, which is why the
    variant is not self-reciprocal.
    """
    return _combine(text, key, "text_plus_key")


def vigenere_encrypt(text: str, key: str) -> str:
    """Vigenere encryption: ``C_i = (P_i + K_i) mod 26``.

    Provided for contrast and for tests that compare the three rules. The
    Vigenere solver lives in its own module.
    """
    return _combine(text, key, "text_plus_key")


def vigenere_decrypt(text: str, key: str) -> str:
    """Vigenere decryption: ``P_i = (C_i - K_i) mod 26``."""
    return _combine(text, key, "text_minus_key")


def encrypt(text: str, key: str, *, variant: bool = False) -> str:
    """Encrypt with Beaufort, or with variant Beaufort when *variant* is true.

    Operates on letters only; returns letters only, uppercase.
    """
    return variant_encrypt(text, key) if variant else beaufort_encrypt(text, key)


def decrypt(text: str, key: str, *, variant: bool = False) -> str:
    """Exact inverse of :func:`encrypt` for the same key and *variant* flag."""
    return variant_decrypt(text, key) if variant else beaufort_decrypt(text, key)


# ---------------------------------------------------------------------------
# Column solving
# ---------------------------------------------------------------------------


def _decrypt_values(
    cipher_values: Sequence[int], key_values: Sequence[int], variant_name: str
) -> list[int]:
    """Decrypt encoded ciphertext under an encoded key, for one variant.

    Beaufort:         ``P = K - C``  (reflect, then shift)
    Variant Beaufort: ``P = C + K``  (shift only)
    """
    length = len(key_values)
    if variant_name == BEAUFORT:
        return [
            (key_values[i % length] - value) % ALPHABET_SIZE
            for i, value in enumerate(cipher_values)
        ]
    return [
        (value + key_values[i % length]) % ALPHABET_SIZE
        for i, value in enumerate(cipher_values)
    ]


def column_key_letter(column: str, variant_name: str) -> tuple[int, float, float]:
    """Best key letter for one column, with its chi-squared evidence.

    Returns ``(key value, chi-squared of the winner, margin over the runner-up)``.

    For each of the 26 possible key letters we decrypt the column with the rule
    that belongs to this variant and measure how far the result is from English
    letter frequencies. The key letter that fits English best wins.

    The margin matters as much as the winner: a column whose best and
    second-best key letters score almost the same has not really been solved,
    it has been guessed, and the caller reports that rather than hiding it.
    """
    variant_name = _resolve_variant(variant_name)
    if not column:
        raise ValueError("Cannot derive a key letter from an empty column")

    values = to_numbers(column)
    scores: list[tuple[float, int]] = []
    for key_value in range(ALPHABET_SIZE):
        if variant_name == BEAUFORT:
            plain = [(key_value - value) % ALPHABET_SIZE for value in values]
        else:
            plain = [(value + key_value) % ALPHABET_SIZE for value in values]
        scores.append((chi_squared_english(from_numbers(plain)), key_value))
    scores.sort()
    best_score, best_key = scores[0]
    margin = scores[1][0] - best_score if len(scores) > 1 else 0.0
    return best_key, best_score, margin


def derive_key(
    letters: str, key_length: int, variant_name: str
) -> tuple[list[int], float, float]:
    """Derive a whole key of *key_length* letters by solving each column.

    Returns ``(key values, total chi-squared, mean column margin)``.
    """
    if key_length < 1:
        raise ValueError(f"Key length must be at least 1, got {key_length}")
    if key_length > len(letters):
        raise ValueError(
            f"Key length {key_length} exceeds the {len(letters)} letters of "
            "ciphertext available; there would be empty columns."
        )
    variant_name = _resolve_variant(variant_name)

    key_values: list[int] = []
    total_chi = 0.0
    margins: list[float] = []
    for column in columns(letters, key_length):
        key_value, chi, margin = column_key_letter(column, variant_name)
        key_values.append(key_value)
        total_chi += chi
        margins.append(margin)
    mean_margin = sum(margins) / len(margins) if margins else 0.0
    return key_values, total_chi, mean_margin


def _hill_climb(
    cipher_values: Sequence[int],
    key_values: list[int],
    variant_name: str,
    scorer: EnglishScorer,
    deadline: float | None,
) -> tuple[list[int], float, int, bool]:
    """Improve a key one letter at a time under the n-gram score.

    Chi-squared judges a column by single-letter frequencies alone, which is a
    weak signal when a column holds only a few dozen letters. The n-gram model
    judges the whole decryption in context, so it can see that one key letter
    is wrong even when that letter's column looks frequency-plausible.

    Changing key letter *j* changes only the letters at positions congruent to
    *j*, but those letters sit inside quadgrams that span the whole text, so we
    rescore the full decryption. We sweep every position, keep any change that
    improves the score, and repeat until a whole sweep changes nothing --
    a steepest-ascent climb with a guaranteed finish, since the score strictly
    increases at every accepted step.

    Returns ``(key, score, sweeps, ran out of time)``.
    """
    current = list(key_values)
    best_score = scorer.score_values(
        _decrypt_values(cipher_values, current, variant_name)
    )
    sweeps = 0
    budget_hit = False

    improved = True
    while improved:
        improved = False
        sweeps += 1
        for position in range(len(current)):
            if deadline is not None and time.monotonic() >= deadline:
                return current, best_score, sweeps, True
            original = current[position]
            best_letter = original
            for candidate_value in range(ALPHABET_SIZE):
                if candidate_value == original:
                    continue
                current[position] = candidate_value
                score = scorer.score_values(
                    _decrypt_values(cipher_values, current, variant_name)
                )
                if score > best_score:
                    best_score = score
                    best_letter = candidate_value
            current[position] = best_letter
            if best_letter != original:
                improved = True
        # A key of length 1 has nothing left to explore after one sweep.
        if len(current) == 1:
            break
    return current, best_score, sweeps, budget_hit


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    key_length: int | None = None,
    min_key_length: int = 1,
    max_key_length: int = 20,
    variants: Sequence[str] = VARIANTS,
    refine: bool = True,
    refine_top: int = 5,
    time_budget: float | None = None,
) -> CandidateSet:
    """Attack Beaufort and variant Beaufort ciphertext; return ranked candidates.

    Options
    -------
    key_length:
        Solve this key length only. ``None`` tries every length in range.
    min_key_length, max_key_length:
        Inclusive range of key lengths to try when *key_length* is ``None``.
    variants:
        Which variants to attack. Defaults to both, ranked in one list, with
        the variant named in every ``Candidate.key``.
    refine:
        Hill-climb the best few chi-squared keys under the n-gram model.
    refine_top:
        How many (variant, key length) pairs to hill-climb.
    time_budget:
        Seconds. The search stops cleanly when it is exceeded and every
        candidate records ``time_budget_hit``.

    Every key length tried becomes a candidate, so the caller can see the
    runners-up and judge the margin rather than being handed one answer.
    """
    engine = scorer if scorer is not None else default_scorer()
    normalized = normalize(source) if isinstance(source, str) else source
    letters = normalized.letters
    results = CandidateSet()
    if not letters:
        return results

    if min_key_length < 1:
        raise ValueError(f"min_key_length must be at least 1, got {min_key_length}")
    if max_key_length < min_key_length:
        raise ValueError(
            f"max_key_length ({max_key_length}) is below min_key_length "
            f"({min_key_length})"
        )
    if key_length is not None and key_length < 1:
        raise ValueError(f"key_length must be at least 1, got {key_length}")

    wanted = [_resolve_variant(name) for name in variants]
    if not wanted:
        raise ValueError("At least one variant must be requested")

    deadline = time.monotonic() + time_budget if time_budget is not None else None
    budget_hit = False

    if key_length is not None:
        if key_length > len(letters):
            raise ValueError(
                f"key_length {key_length} exceeds the {len(letters)} letters of "
                "ciphertext; there is nothing to solve."
            )
        lengths = [key_length]
    else:
        lengths = [
            length
            for length in range(min_key_length, max_key_length + 1)
            if length <= len(letters)
        ]
    if not lengths:
        return results

    # Independent evidence about the period, measured once and attached to
    # every candidate so a human can see whether the statistics agree with the
    # key length that happened to score best.
    ic_rows = dict(ic_by_period(letters, max(lengths)))
    votes = kasiski_factor_votes(letters, maximum_factor=max(2, max(lengths)))

    cipher_values = to_numbers(letters)

    # -- stage 1: one chi-squared key per (variant, length) ----------------
    attempts: list[dict] = []
    for variant_name in wanted:
        for length in lengths:
            if deadline is not None and time.monotonic() >= deadline:
                budget_hit = True
                break
            key_values, chi_total, margin = derive_key(letters, length, variant_name)
            plain_values = _decrypt_values(cipher_values, key_values, variant_name)
            attempts.append(
                {
                    "variant": variant_name,
                    "length": length,
                    "key": key_values,
                    "chi": chi_total,
                    "margin": margin,
                    "score": engine.score_values(plain_values),
                    "sweeps": 0,
                    "refined": False,
                }
            )
        if budget_hit:
            break

    # -- stage 2: hill-climb the most promising few -------------------------
    if refine and refine_top > 0:
        order = sorted(attempts, key=lambda row: row["score"], reverse=True)
        for row in order[:refine_top]:
            if deadline is not None and time.monotonic() >= deadline:
                budget_hit = True
                break
            key_values, score, sweeps, hit = _hill_climb(
                cipher_values, row["key"], row["variant"], engine, deadline
            )
            budget_hit = budget_hit or hit
            if score > row["score"]:
                row["key"] = key_values
                row["score"] = score
            row["sweeps"] = sweeps
            row["refined"] = True

    # -- report -------------------------------------------------------------
    for row in attempts:
        variant_name = row["variant"]
        length = row["length"]
        plain_values = _decrypt_values(cipher_values, row["key"], variant_name)
        plaintext = from_numbers(plain_values)
        key_text = from_numbers(row["key"])
        diagnostics: dict = {
            "variant": _METHOD_NAME[variant_name],
            "key_length": length,
            "column_letters": len(letters) // length,
            "chi_squared_total": row["chi"],
            "column_margin_mean": row["margin"],
            "column_ic_at_period": ic_rows.get(length),
            "kasiski_votes_at_length": votes.get(length, 0),
            "refined": row["refined"],
            "hill_climb_sweeps": row["sweeps"],
            "letters": len(letters),
        }
        if budget_hit:
            diagnostics["time_budget_hit"] = True
        annotate(diagnostics, plaintext, engine)
        diagnostics["meets_english_threshold"] = (
            diagnostics["normalised_score"] > ENGLISH_THRESHOLD
        )
        results.add(
            Candidate(
                method=_METHOD_NAME[variant_name],
                key=f"{_KEY_LABEL[variant_name]} key={key_text} ({length})",
                score=engine.score(plaintext),
                plaintext=plaintext,
                diagnostics=diagnostics,
                display=normalized.relayout(plaintext),
            )
        )

    if top and top > 0:
        return CandidateSet(results.top(top))
    return results
