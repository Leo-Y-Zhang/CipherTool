"""Affine cipher: E(x) = (a*x + b) mod 26, and its exhaustive attack.

The cipher
----------
Number the letters A..Z as 0..25. The affine cipher multiplies and then adds:

    E(x) = (a * x + b) mod 26

It contains two ciphers this toolkit implements separately:

* a = 1 gives E(x) = x + b, which is exactly Caesar with shift b;
* a = 25, b = 25 gives E(x) = -x - 1 = 25 - x, which is exactly Atbash.

Why a must be coprime with 26
-----------------------------
Encryption is only useful if it can be undone, which means E must be a
bijection on the 26 letters. Multiplication by a is a bijection modulo 26
exactly when gcd(a, 26) = 1. To see why it fails otherwise, let
d = gcd(a, 26) > 1. Then

    a * (x + 26/d) = a*x + 26 * (a/d) = a*x   (mod 26)

so x and x + 26/d encipher to the same letter and the cipher collapses two
plaintext letters onto one ciphertext letter. Concretely, a = 2 sends both x
and x + 13 to the same place (2x + 26 = 2x mod 26), and a = 13 sends x and
x + 2 to the same place. Since 26 = 2 * 13, the multipliers that fail are the
even ones and 13, leaving 12 usable values of a:

    1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25

The key space is therefore 12 * 26 = 312 keys (311 useful ones; a = 1, b = 0
is the identity). That is small enough to try exhaustively, which is what
:func:`solve` does.

Decryption, and the extended Euclidean algorithm
------------------------------------------------
To invert y = a*x + b we need a number a_inv with a * a_inv = 1 (mod 26),
because then

    a_inv * (y - b) = a_inv * a * x = x   (mod 26)

so D(y) = a_inv * (y - b) mod 26.

We compute a_inv ourselves rather than calling ``pow(a, -1, 26)``, because the
algorithm that finds it is also the proof that it exists. Ordinary Euclid
finds gcd(a, m) by repeatedly replacing the pair (r_prev, r) with
(r, r_prev mod r) until the remainder is 0. The extended version additionally
remembers how every remainder is built out of the two original numbers,
maintaining at each step a coefficient s with

    a * s + m * t = r

for the current remainder r. It starts from the two trivial identities

    a * 1 + m * 0 = a        (remainder a, coefficient s = 1)
    a * 0 + m * 1 = m        (remainder m, coefficient s = 0)

and then applies to the coefficients exactly the same subtraction it applies
to the remainders: if r_next = r_prev - q * r then s_next = s_prev - q * s.
When the remainder hits 0, the previous remainder is gcd(a, m) and its
coefficient s satisfies a * s + m * t = gcd(a, m). If the gcd is 1 then

    a * s = 1 - m * t

and since m * t vanishes modulo m, a * s leaves remainder 1 modulo m. So
s mod m is the inverse. If the gcd is not 1 there is no inverse at all, and
the same computation tells us so -- which is why :func:`modular_inverse` can
raise a truthful error instead of guessing.

The attack
----------
Brute force over all 312 keys, ranked by the n-gram model with chi-squared
against English letter frequencies reported alongside as a second, order-blind
signal. No frequency shortcut is needed at this size, and a shortcut would
only add a failure mode: an exhaustive search cannot miss the key, so if the
best candidate still reads as nonsense then the text is not an affine cipher,
and that is a finding worth reporting rather than hiding.
"""

from __future__ import annotations

import time
from typing import Any, Final, Sequence

from .candidates import Candidate, CandidateSet
from .normalize import (
    ALPHABET,
    ALPHABET_SIZE,
    NormalizedText,
    from_numbers,
    letters_only,
    normalize,
    to_numbers,
)
from .scoring import EnglishScorer, annotate, default_scorer
from .statistics import chi_squared_english

METHOD = "Affine"


# ---------------------------------------------------------------------------
# Modular arithmetic, written out rather than imported
# ---------------------------------------------------------------------------


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return ``(g, s, t)`` with ``a*s + b*t == g == gcd(a, b)``.

    Iterative form of the algorithm described in the module docstring. The
    two coefficient columns are updated with the same subtraction as the
    remainder column, which is the entire trick.
    """
    # (remainder, coefficient of a) for the previous and current rows.
    old_remainder, remainder = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while remainder != 0:
        quotient = old_remainder // remainder
        old_remainder, remainder = remainder, old_remainder - quotient * remainder
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    return old_remainder, old_s, old_t


def modular_inverse(a: int, m: int) -> int:
    """The inverse of *a* modulo *m*: the value ``v`` with ``a*v % m == 1``.

    Raises
    ------
    ValueError
        If *m* is not at least 2, or if ``gcd(a, m) != 1``, in which case no
        inverse exists -- multiplication by *a* is not reversible modulo *m*.
    """
    if isinstance(m, bool) or not isinstance(m, int) or m < 2:
        raise ValueError(
            f"modulus must be an integer of at least 2, got {m!r}."
        )
    if isinstance(a, bool) or not isinstance(a, int):
        raise ValueError(
            f"value to invert must be an integer, got {a!r} "
            f"({type(a).__name__})."
        )
    divisor, coefficient, _ = extended_gcd(a % m, m)
    if divisor != 1:
        raise ValueError(
            f"{a} has no inverse modulo {m}: gcd({a % m}, {m}) = {divisor}, "
            f"so multiplying by {a} maps different values onto the same "
            "result and cannot be undone."
        )
    # `coefficient` may be negative; % m brings it into 0..m-1.
    return coefficient % m


#: The 12 multipliers coprime with 26, computed rather than typed in so that
#: the list cannot drift away from the arithmetic that justifies it.
VALID_MULTIPLIERS: Final[tuple[int, ...]] = tuple(
    a for a in range(1, ALPHABET_SIZE)
    if extended_gcd(a, ALPHABET_SIZE)[0] == 1
)


def valid_multipliers() -> tuple[int, ...]:
    """The 12 values of *a* that give an invertible affine cipher mod 26.

    ``(1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25)`` -- the odd numbers below
    26 except 13, because 26 = 2 * 13.
    """
    return VALID_MULTIPLIERS


# ---------------------------------------------------------------------------
# Key and input validation
# ---------------------------------------------------------------------------


def _check_key(a: int, b: int) -> tuple[int, int]:
    """Validate an affine key and reduce both halves modulo 26.

    Reducing first means a = 31 is accepted as the same key as a = 5, which it
    genuinely is; what is rejected is a multiplier that no reduction can save.
    """
    for name, value in (("a", a), ("b", b)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"affine key part {name} must be a whole number, got "
                f"{value!r} ({type(value).__name__}). The key is two "
                "integers, as in a=5 b=8."
            )
    reduced_a = a % ALPHABET_SIZE
    reduced_b = b % ALPHABET_SIZE
    divisor = extended_gcd(reduced_a, ALPHABET_SIZE)[0]
    if divisor != 1:
        raise ValueError(
            f"affine multiplier a={a} is unusable: gcd({reduced_a}, 26) = "
            f"{divisor}, so it maps {divisor} different plaintext letters "
            "onto each ciphertext letter and cannot be decrypted. Usable "
            "values of a are: "
            + ", ".join(str(value) for value in VALID_MULTIPLIERS)
            + "."
        )
    return reduced_a, reduced_b


def _require_text(text: str) -> str:
    """Reject non-string input loudly rather than half-processing it."""
    if not isinstance(text, str):
        raise ValueError(
            f"text must be a string, got {type(text).__name__}. Read the file "
            "first, or pass NormalizedText.letters."
        )
    return text


# ---------------------------------------------------------------------------
# Encryption and decryption
# ---------------------------------------------------------------------------


def encrypt(text: str, a: int, b: int) -> str:
    """Encrypt. Operates on letters only; returns letters only, uppercase.

    ``a`` must be coprime with 26 (see :func:`valid_multipliers`); ``b`` may
    be any integer. Both are reduced modulo 26.
    """
    multiplier, offset = _check_key(a, b)
    letters = letters_only(_require_text(text))
    return from_numbers(
        multiplier * value + offset for value in to_numbers(letters)
    )


def decrypt(text: str, a: int, b: int) -> str:
    """Exact inverse of :func:`encrypt` for the same key.

    Applies ``D(y) = a_inv * (y - b) mod 26`` with ``a_inv`` from
    :func:`modular_inverse`.
    """
    multiplier, offset = _check_key(a, b)
    inverse = modular_inverse(multiplier, ALPHABET_SIZE)
    letters = letters_only(_require_text(text))
    return from_numbers(
        inverse * (value - offset) for value in to_numbers(letters)
    )


def cipher_alphabet(a: int, b: int) -> str:
    """The alphabet A-Z maps onto under key ``(a, b)``, for reporting."""
    return encrypt(ALPHABET, a, b)


def describe_key(a: int, b: int) -> str:
    """Name the well-known special cases of an affine key.

    Returns an empty string for an ordinary key. This is how the solver tells
    an operator that the "affine" answer it found is really Caesar or Atbash,
    which matters when deciding what the next round of a puzzle will be.
    """
    multiplier, offset = _check_key(a, b)
    if multiplier == 1 and offset == 0:
        return "identity (a=1 b=0 leaves the text unchanged)"
    if multiplier == 1:
        return f"Caesar shift={offset} (a=1 is a pure shift)"
    if multiplier == 25 and offset == 25:
        return "Atbash (a=25 b=25 reverses the alphabet)"
    return ""


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _ranks(values: Sequence[float], *, best_is_lowest: bool) -> list[int]:
    """1-based rank of each value; rank 1 is the best, ties break by index."""
    direction = 1.0 if best_is_lowest else -1.0
    order = sorted(
        range(len(values)),
        key=lambda index: (direction * values[index], index),
    )
    ranks = [0] * len(values)
    for position, index in enumerate(order, start=1):
        ranks[index] = position
    return ranks


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    time_budget: float | None = None,
    **options: Any,
) -> CandidateSet:
    """Try all 12 * 26 = 312 affine keys and return them all, ranked.

    The whole key space is returned rather than the best few, so that the
    operator can see the margin between the winner and the field
    (``CandidateSet.score_gap()``) and trim for display with
    ``CandidateSet.top(n)``.

    Parameters
    ----------
    source:
        Ciphertext, or an already-normalised :class:`NormalizedText`.
    scorer:
        English scorer; ``None`` uses the shared :func:`default_scorer`.
    top:
        Accepted for a uniform solver interface; not used for filtering here.
    time_budget:
        Seconds. Decrypting is instant but word-coverage scoring is not, so
        on a very long ciphertext 312 full annotations can take a few
        seconds. When the budget runs out the search stops cleanly, every
        candidate records ``time_budget_hit`` and ``keys_tested`` shows how
        much of the key space was actually covered. At least one key is
        always tried, so a budget of 0 returns one honest candidate rather
        than an empty set.
    **options:
        Anything else is ignored and recorded as ``options_ignored``.

    Empty input gives an empty :class:`CandidateSet`.
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

    if time_budget is not None and (
        isinstance(time_budget, bool)
        or not isinstance(time_budget, (int, float))
        or time_budget < 0
    ):
        raise ValueError(
            f"time_budget must be a non-negative number of seconds or None, "
            f"got {time_budget!r}."
        )

    ignored = ", ".join(sorted(options)) if options else ""
    values = to_numbers(letters)
    started = time.monotonic()
    budget_hit = False

    keys: list[tuple[int, int]] = []
    plaintexts: list[str] = []
    scores: list[float] = []
    chi_values: list[float] = []
    diagnostics_list: list[dict[str, Any]] = []

    for multiplier in VALID_MULTIPLIERS:
        inverse = modular_inverse(multiplier, ALPHABET_SIZE)
        for offset in range(ALPHABET_SIZE):
            # Stop cleanly, but only after at least one key has been tried,
            # so the caller always gets something it can read.
            if (
                time_budget is not None
                and keys
                and time.monotonic() - started >= time_budget
            ):
                budget_hit = True
                break
            plaintext = from_numbers(
                inverse * (value - offset) for value in values
            )
            keys.append((multiplier, offset))
            plaintexts.append(plaintext)
            scores.append(engine.score(plaintext))
            chi_values.append(chi_squared_english(plaintext))
            diagnostics: dict[str, Any] = {
                "a": multiplier,
                "b": offset,
                "a_inverse": inverse,
                "chi_squared": chi_values[-1],
                "cipher_alphabet": cipher_alphabet(multiplier, offset),
            }
            # Annotation (word-coverage segmentation) is the expensive part of
            # a key trial, so it happens inside the loop the time budget
            # guards. Ranking below is arithmetic on numbers already computed.
            annotate(diagnostics, plaintext, engine)
            diagnostics_list.append(diagnostics)
        if budget_hit:
            break

    chi_ranks = _ranks(chi_values, best_is_lowest=True)
    score_ranks = _ranks(scores, best_is_lowest=False)
    chi_best = min(range(len(chi_values)), key=lambda i: (chi_values[i], i))
    score_best = max(range(len(scores)), key=lambda i: (scores[i], -i))
    agree = chi_best == score_best

    candidates = CandidateSet()
    for index, (multiplier, offset) in enumerate(keys):
        diagnostics = diagnostics_list[index]
        diagnostics["keys_tested"] = len(keys)
        diagnostics["keys_possible"] = len(VALID_MULTIPLIERS) * ALPHABET_SIZE
        diagnostics["rank_by_chi2"] = chi_ranks[index]
        diagnostics["rank_by_ngram"] = score_ranks[index]
        diagnostics["chi2_best_key"] = "a={} b={}".format(*keys[chi_best])
        diagnostics["ngram_best_key"] = "a={} b={}".format(*keys[score_best])
        diagnostics["measures_agree"] = agree
        if budget_hit:
            diagnostics["time_budget_hit"] = True
        special = describe_key(multiplier, offset)
        if special:
            diagnostics["equivalent_to"] = special
        if ignored:
            diagnostics["options_ignored"] = ignored
        candidates.add(
            Candidate(
                method=METHOD,
                key=f"a={multiplier} b={offset}",
                score=scores[index],
                plaintext=plaintexts[index],
                diagnostics=diagnostics,
                # One cipher letter per plaintext letter, so the recovered
                # text fits back into the original layout.
                display=normalized.relayout(plaintexts[index]),
            )
        )
    return candidates
