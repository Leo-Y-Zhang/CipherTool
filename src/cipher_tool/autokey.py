"""Autokey ciphers: the plaintext and ciphertext forms, and attacks on both.

The idea
--------
A Vigenere key of length m repeats every m letters, and that repetition is the
whole reason Vigenere falls: it makes the ciphertext a set of m interleaved
Caesar shifts, and Kasiski plus the index of coincidence find m without any
guessing. Vigenere himself proposed the fix in 1586: do not repeat the key,
extend it with the message. A short *primer* starts the key stream and the
message itself continues it, so the key never repeats and the periodic
statistics have nothing to lock onto.

There are two classical forms, and they behave completely differently under
attack, so this module keeps them firmly apart.

**Plaintext autokey** -- the key stream is the primer followed by the plaintext.

    K = primer + P
    K_i = primer_i          for i < m
    K_i = P_{i-m}           for i >= m
    C_i = (P_i + K_i) mod 26

**Ciphertext autokey** -- the key stream is the primer followed by the
ciphertext.

    K = primer + C
    K_i = primer_i          for i < m
    K_i = C_{i-m}           for i >= m
    C_i = (P_i + K_i) mod 26

Decryption, and why the plaintext form is the interesting one
-------------------------------------------------------------
Ciphertext autokey decrypts in one pass with no bookkeeping: every key letter
after the primer is a ciphertext letter, and the receiver has the whole
ciphertext in front of them from the start.

The plaintext form cannot do that. The key letter for position i is the
plaintext letter from position i - m, which the receiver does not have until
they have decrypted that far. So decryption *builds its own key as it goes*:
use the primer for the first m letters, and from then on feed each plaintext
letter you recover back in as the key letter for the position m places later.
The consequence is error propagation -- one wrong letter poisons every m-th
letter after it -- and that is exactly what makes the attack below work, since
a wrong primer letter produces visible nonsense rather than a near miss.

How each form is attacked
-------------------------
**Ciphertext autokey is fatally weak.** Substitute K_i = C_{i-m} into the
decryption rule:

    P_i = (C_i - K_i) = (C_i - C_{i-m}) mod 26      for i >= m

The primer has vanished. Every plaintext letter after the first m is a
difference of two ciphertext letters and needs no key at all, so the only
unknown in the whole message is m itself, which is found by trying each value
and scoring the result. The primer merely hides the first m letters, and those
are recovered afterwards by choosing the m opening plaintext letters that read
best -- the primer follows from primer_i = (C_i - P_i) mod 26.

Note the asymmetry that leaves, because the solver reports it rather than
glossing over it: everything from position m onwards is *forced* and does not
depend on the primer at all, while the opening m letters are only a language
model's opinion. Any opening whatsoever is consistent with some primer, so
those letters are not recoverable by cryptanalysis at all -- only by guessing
which English opening reads best. Measured on this toolkit's own corpus, a
420-letter message with a six-letter primer decrypts perfectly from position
six, while the model prefers the opening ENINEL to the true ALMOST because it
scores fractionally higher. So a ciphertext-autokey candidate can carry a
completely correct message and a wrong primer, and the diagnostics say exactly
which letters are forced and which are guessed.

**Plaintext autokey is genuinely harder.** Substituting K_i = P_{i-m} gives

    P_i = (C_i - P_{i-m}) mod 26

which is a recurrence, not a formula: each plaintext letter depends on one m
places earlier. Guess the m primer letters and the whole message unrolls, so
the cipher is exactly as strong as the primer is long. Two structural facts
drive the attack here:

1. *The message splits into m independent chains.* Positions j, j+m, j+2m, ...
   depend only on primer letter j. So each primer letter can be tested on its
   own chain without knowing any of the others -- and a chain is ordinary
   English sampled every m letters, so its single-letter frequencies are still
   English. Chi-squared on each chain therefore gives a usable first guess at
   every primer letter independently, which is what :func:`initial_primer`
   does.
2. *A wrong primer letter alternates sign down its chain.* Unrolling the
   recurrence with a wrong guess g' = g + e:

       P'_j     = P_j - e
       P'_{j+m} = P_{j+m} + e
       P'_{j+2m}= P_{j+2m} - e

   the error flips sign at every step. So a wrong guess does not shift a chain
   uniformly the way a wrong Caesar key would; it produces a text that no
   single shift can repair. This is why the ordinary per-column Vigenere
   attack finds nothing on autokey ciphertext.

The chain guess fixes letters one at a time using only single-letter
statistics, so it is usually right about most of the primer and wrong about
some of it. The classical way to finish the job is to guess a stretch of key,
decrypt what it gives you, and extend the guess where the decryption looks
like English. The version implemented here is the systematic one: for short
primers try every possible primer exhaustively, judging each on the n-gram
score of the decrypted prefix (a wrong letter shows up within a few dozen
letters), and then hill-climb the primer letter by letter under the full
n-gram score, with seeded random restarts to escape a local maximum.

HONEST LIMITATIONS -- READ THIS
-------------------------------
Autokey attacks are markedly weaker than Vigenere attacks and they fail often.

* There is no key length to find by Kasiski or by the index of coincidence.
  Those measurements, which do most of the work against Vigenere, say nothing
  useful here, so the search leans entirely on the English language model.
* The search is exhaustive only for short primers. Beyond that it is a
  hill-climb, and a hill-climb can stop at a wrong primer that happens to
  score well.
* A primer longer than ``max_primer`` (default 8) is not tried at all, and
  the solver cannot tell you that is what happened.
* On short texts -- under roughly 200 letters -- the n-gram score of a wrong
  primer is often as good as the right one, and this solver will confidently
  rank nonsense first.

Because of that, every candidate carries
``diagnostics["meets_english_threshold"]``, which is true only when the
per-letter n-gram score beats -1.80. When it is false, the solver has NOT
solved the message, whatever the ranking says.
"""

from __future__ import annotations

import itertools
import random
import time
from typing import Sequence

from .candidates import Candidate, CandidateSet
from .normalize import (
    ALPHABET_SIZE,
    NormalizedText,
    clean_key,
    from_numbers,
    letters_only,
    normalize,
    to_numbers,
)
from .scoring import EnglishScorer, annotate, default_scorer
from .statistics import chi_squared_english

#: Canonical mode names accepted by every public function here.
PLAINTEXT = "plaintext"
CIPHERTEXT = "ciphertext"
MODES: tuple[str, str] = (PLAINTEXT, CIPHERTEXT)

_MODE_ALIASES = {
    "plaintext": PLAINTEXT,
    "plain": PLAINTEXT,
    "p": PLAINTEXT,
    "ciphertext": CIPHERTEXT,
    "cipher": CIPHERTEXT,
    "c": CIPHERTEXT,
}

_METHOD_NAME = {
    PLAINTEXT: "Autokey (plaintext)",
    CIPHERTEXT: "Autokey (ciphertext)",
}

#: A candidate at or above this per-letter n-gram score is worth reading.
#: Below it, the solver has not solved anything.
ENGLISH_THRESHOLD = -1.80

#: Attached to every candidate so the caveat travels with the result.
ATTACK_CAVEAT = (
    "Autokey attacks are weaker than Vigenere attacks: the key never repeats, "
    "so Kasiski and the index of coincidence give no key length and the search "
    "relies entirely on the English model. It fails often, especially under "
    "about 200 letters. Treat a candidate as unsolved unless "
    "meets_english_threshold is true and the plaintext reads."
)


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------


def _require_primer(primer: str) -> str:
    """Normalise a primer and refuse an empty one.

    A zero-length primer is not a degenerate autokey, it is undefined: there
    would be no key letter at all for the first position.
    """
    cleaned = clean_key(primer)
    if not cleaned:
        raise ValueError(
            f"An autokey primer must contain at least one letter A-Z; "
            f"{primer!r} normalises to an empty primer."
        )
    return cleaned


def _resolve_mode(mode: str) -> str:
    """Map a user-supplied mode name onto :data:`PLAINTEXT` or :data:`CIPHERTEXT`."""
    resolved = _MODE_ALIASES.get(str(mode).strip().lower())
    if resolved is None:
        raise ValueError(
            f"Unknown autokey mode {mode!r}. Use 'plaintext' (key = primer + "
            "plaintext) or 'ciphertext' (key = primer + ciphertext)."
        )
    return resolved


# ---------------------------------------------------------------------------
# The ciphers
# ---------------------------------------------------------------------------


def plaintext_autokey_encrypt(text: str, primer: str) -> str:
    """Encrypt with a plaintext autokey: ``K = primer + P``.

    Operates on letters only; returns letters only, uppercase. The sender has
    the whole plaintext, so the key stream is available immediately.
    """
    values = to_numbers(letters_only(text))
    primer_values = to_numbers(_require_primer(primer))
    size = len(primer_values)
    out: list[int] = []
    for position, plain_value in enumerate(values):
        key_value = (
            primer_values[position] if position < size else values[position - size]
        )
        out.append((plain_value + key_value) % ALPHABET_SIZE)
    return from_numbers(out)


def plaintext_autokey_decrypt(text: str, primer: str) -> str:
    """Decrypt a plaintext autokey, recovering the running key as it goes.

    This is the part worth understanding. The key letter for position i is the
    plaintext letter from position i - m, and the receiver does not have that
    letter until they have decrypted it. So the loop below keeps the plaintext
    it has recovered so far and reads its own output back in as key material:

        positions 0 .. m-1   use the primer, the only key we were given
        positions m onwards  use ``plain[position - m]``, a letter this very
                             loop produced a moment ago

    The key stream is therefore *reconstructed*, never known in advance. One
    wrong primer letter corrupts P_j, which becomes the key for P_{j+m}, which
    corrupts that, and so on down the chain -- error propagation that is a
    nuisance for a legitimate receiver and a gift to the cryptanalyst, because
    a nearly-right primer still produces obvious nonsense.
    """
    values = to_numbers(letters_only(text))
    primer_values = to_numbers(_require_primer(primer))
    size = len(primer_values)
    plain: list[int] = []
    for position, cipher_value in enumerate(values):
        key_value = (
            primer_values[position] if position < size else plain[position - size]
        )
        plain.append((cipher_value - key_value) % ALPHABET_SIZE)
    return from_numbers(plain)


def ciphertext_autokey_encrypt(text: str, primer: str) -> str:
    """Encrypt with a ciphertext autokey: ``K = primer + C``.

    The key stream is the ciphertext this very loop is producing, so it is
    built up alongside the output.
    """
    values = to_numbers(letters_only(text))
    primer_values = to_numbers(_require_primer(primer))
    size = len(primer_values)
    out: list[int] = []
    for position, plain_value in enumerate(values):
        key_value = primer_values[position] if position < size else out[position - size]
        out.append((plain_value + key_value) % ALPHABET_SIZE)
    return from_numbers(out)


def ciphertext_autokey_decrypt(text: str, primer: str) -> str:
    """Decrypt a ciphertext autokey.

    Unlike the plaintext form, every key letter after the primer is a letter of
    the ciphertext, which the receiver already has in full. Nothing needs to be
    recovered first, and a transmission error corrupts only two letters rather
    than a whole chain.
    """
    values = to_numbers(letters_only(text))
    primer_values = to_numbers(_require_primer(primer))
    size = len(primer_values)
    plain: list[int] = []
    for position, cipher_value in enumerate(values):
        key_value = (
            primer_values[position] if position < size else values[position - size]
        )
        plain.append((cipher_value - key_value) % ALPHABET_SIZE)
    return from_numbers(plain)


def encrypt(text: str, primer: str, *, mode: str = PLAINTEXT) -> str:
    """Encrypt with the plaintext (default) or ciphertext autokey."""
    if _resolve_mode(mode) == PLAINTEXT:
        return plaintext_autokey_encrypt(text, primer)
    return ciphertext_autokey_encrypt(text, primer)


def decrypt(text: str, primer: str, *, mode: str = PLAINTEXT) -> str:
    """Exact inverse of :func:`encrypt` for the same primer and mode."""
    if _resolve_mode(mode) == PLAINTEXT:
        return plaintext_autokey_decrypt(text, primer)
    return ciphertext_autokey_decrypt(text, primer)


# ---------------------------------------------------------------------------
# Working on encoded values (the solver's hot path)
# ---------------------------------------------------------------------------


def _plaintext_values(
    cipher_values: Sequence[int], primer_values: Sequence[int]
) -> list[int]:
    """Plaintext-autokey decryption on encoded values."""
    size = len(primer_values)
    plain: list[int] = []
    for position, cipher_value in enumerate(cipher_values):
        key_value = (
            primer_values[position] if position < size else plain[position - size]
        )
        plain.append((cipher_value - key_value) % ALPHABET_SIZE)
    return plain


def _plaintext_prefix(
    cipher_values: Sequence[int], primer_values: Sequence[int], limit: int
) -> list[int]:
    """Decrypt only the first *limit* letters, for cheap candidate screening.

    A wrong primer letter corrupts its chain from its very first position, so a
    prefix of a few dozen letters already separates a right primer from a wrong
    one. Screening on a prefix is what makes an exhaustive search over 26^m
    primers affordable.
    """
    size = len(primer_values)
    plain: list[int] = []
    for position in range(min(limit, len(cipher_values))):
        key_value = (
            primer_values[position] if position < size else plain[position - size]
        )
        plain.append((cipher_values[position] - key_value) % ALPHABET_SIZE)
    return plain


def _chain_values(
    cipher_values: Sequence[int], size: int, residue: int, primer_value: int
) -> list[int]:
    """Unroll one independent chain of a plaintext autokey.

    Positions ``residue, residue + size, residue + 2*size, ...`` depend only on
    primer letter ``residue``, through ``P_next = C_next - P_previous``.
    """
    out: list[int] = []
    previous = primer_value
    for position in range(residue, len(cipher_values), size):
        value = (cipher_values[position] - previous) % ALPHABET_SIZE
        out.append(value)
        previous = value
    return out


# ---------------------------------------------------------------------------
# Attack on the plaintext autokey
# ---------------------------------------------------------------------------


def initial_primer(letters: str, size: int) -> tuple[list[int], list[float]]:
    """First guess at a plaintext-autokey primer, one letter at a time.

    Each primer letter owns one chain of the message and no other letter
    touches it, so each can be tested alone. A chain is English sampled every
    *size* letters, which leaves single-letter frequencies untouched, so the
    primer letter whose chain looks most like English by chi-squared is the
    best guess available from frequency alone.

    Returns the primer values and, for each position, the chi-squared margin
    between the best and second-best letter. A small margin is the solver
    admitting it could not tell those letters apart.
    """
    if size < 1:
        raise ValueError(f"Primer length must be at least 1, got {size}")
    if size > len(letters):
        raise ValueError(
            f"Primer length {size} exceeds the {len(letters)} letters of "
            "ciphertext available."
        )
    cipher_values = to_numbers(letters)
    primer: list[int] = []
    margins: list[float] = []
    for residue in range(size):
        scored = sorted(
            (
                chi_squared_english(
                    from_numbers(_chain_values(cipher_values, size, residue, guess))
                ),
                guess,
            )
            for guess in range(ALPHABET_SIZE)
        )
        primer.append(scored[0][1])
        margins.append(scored[1][0] - scored[0][0])
    return primer, margins


def _climb_primer(
    cipher_values: Sequence[int],
    primer_values: Sequence[int],
    scorer: EnglishScorer,
    deadline: float | None,
) -> tuple[list[int], float, int, bool]:
    """Hill-climb a plaintext-autokey primer under the n-gram score.

    Chi-squared can only see single-letter frequencies within one chain. The
    n-gram model sees the message as a whole, so it can tell that one primer
    letter is wrong from the words its chain breaks. We sweep the primer,
    keeping any single-letter change that raises the score of the whole
    decryption, and repeat until a full sweep changes nothing. The score
    strictly increases at every accepted step, so this always terminates.

    Returns ``(primer, score, sweeps, ran out of time)``.
    """
    current = list(primer_values)
    best_score = scorer.score_values(_plaintext_values(cipher_values, current))
    sweeps = 0
    improved = True
    while improved:
        improved = False
        sweeps += 1
        for position in range(len(current)):
            if deadline is not None and time.monotonic() > deadline:
                return current, best_score, sweeps, True
            original = current[position]
            chosen = original
            for guess in range(ALPHABET_SIZE):
                if guess == original:
                    continue
                current[position] = guess
                score = scorer.score_values(_plaintext_values(cipher_values, current))
                if score > best_score:
                    best_score = score
                    chosen = guess
            current[position] = chosen
            if chosen != original:
                improved = True
    return current, best_score, sweeps, False


def _exhaustive_primer(
    cipher_values: Sequence[int],
    size: int,
    scorer: EnglishScorer,
    probe_length: int,
    keep: int,
    deadline: float | None,
) -> tuple[list[int] | None, float, bool]:
    """Try every 26^size primer, screening on a decrypted prefix.

    Two passes: score every primer on the first *probe_length* letters only,
    then rescore the best *keep* of them on the whole message. The prefix is
    enough to eliminate almost everything because a wrong primer letter goes
    wrong immediately and stays wrong.
    """
    probe = min(max(probe_length, size * 4), len(cipher_values))
    shortlist: list[tuple[float, tuple[int, ...]]] = []
    checked = 0
    for combination in itertools.product(range(ALPHABET_SIZE), repeat=size):
        checked += 1
        # Checking the clock on every primer would cost more than the search.
        if deadline is not None and checked % 512 == 0:
            if time.monotonic() > deadline:
                return None, float("-inf"), True
        score = scorer.score_values(
            _plaintext_prefix(cipher_values, combination, probe)
        )
        shortlist.append((score, combination))
    shortlist.sort(key=lambda row: row[0], reverse=True)

    best_primer: list[int] | None = None
    best_score = float("-inf")
    for _, combination in shortlist[:keep]:
        score = scorer.score_values(_plaintext_values(cipher_values, combination))
        if score > best_score:
            best_score = score
            best_primer = list(combination)
    return best_primer, best_score, False


# ---------------------------------------------------------------------------
# Attack on the ciphertext autokey
# ---------------------------------------------------------------------------


def _ciphertext_tail(cipher_values: Sequence[int], size: int) -> list[int]:
    """The part of a ciphertext-autokey message that needs no key at all.

    ``P_i = (C_i - C_{i-m}) mod 26`` for i >= m. The primer cancels out
    completely, which is why this cipher is so much weaker than it looks.
    """
    return [
        (cipher_values[position] - cipher_values[position - size]) % ALPHABET_SIZE
        for position in range(size, len(cipher_values))
    ]


def _solve_ciphertext_head(
    cipher_values: Sequence[int],
    size: int,
    scorer: EnglishScorer,
    deadline: float | None,
    restarts: int = 0,
    rng: random.Random | None = None,
) -> tuple[list[int], list[int], float, bool]:
    """Choose the first *size* plaintext letters, and read off the primer.

    Everything past position *size* is forced by :func:`_ciphertext_tail`, so
    the only freedom left is the opening letters. Choosing them is equivalent
    to choosing the primer, because ``primer_i = (C_i - P_i) mod 26``.

    The opening letters are searched RIGHT TO LEFT, and the direction matters.
    The letter immediately before the forced tail sits inside quadgrams that
    are otherwise entirely known, so the language model has real evidence about
    it; the first letter of the message has almost none until its neighbours
    are settled. Sweeping left to right instead means every early decision is
    taken with the least context, and the climb sticks in a local maximum --
    measured on this toolkit's corpus, a left-to-right sweep settled on the
    opening ETWOSTEVERY where the true ALMOSTEVERY scored better.

    Even right to left this is a hill-climb over a handful of letters, so the
    caller may ask for seeded random restarts.

    Returns ``(plaintext values, primer values, score, ran out of time)``.
    """
    tail = _ciphertext_tail(cipher_values, size)
    head_length = min(size, len(cipher_values))
    budget_hit = False

    def climb(start: list[int]) -> tuple[list[int], float, bool]:
        plain = list(start) + tail
        score_now = scorer.score_values(plain)
        improved = True
        while improved:
            improved = False
            # Right to left: settle the letters with the most context first.
            for position in range(head_length - 1, -1, -1):
                if deadline is not None and time.monotonic() > deadline:
                    return plain, score_now, True
                original = plain[position]
                chosen = original
                for guess in range(ALPHABET_SIZE):
                    if guess == original:
                        continue
                    plain[position] = guess
                    score = scorer.score_values(plain)
                    if score > score_now:
                        score_now = score
                        chosen = guess
                plain[position] = chosen
                if chosen != original:
                    improved = True
        return plain, score_now, False

    best_plain, best_score, budget_hit = climb([0] * head_length)
    if not budget_hit and rng is not None:
        for _ in range(max(0, restarts)):
            seeded = [rng.randrange(ALPHABET_SIZE) for _ in range(head_length)]
            plain, score, hit = climb(seeded)
            budget_hit = budget_hit or hit
            if score > best_score:
                best_plain, best_score = plain, score
            if hit:
                break

    primer = [
        (cipher_values[i] - best_plain[i]) % ALPHABET_SIZE
        for i in range(head_length)
    ]
    return best_plain, primer, best_score, budget_hit


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    max_primer: int = 8,
    min_primer: int = 1,
    modes: Sequence[str] = MODES,
    exhaustive_limit: int = 20000,
    probe_length: int = 60,
    climb_top: int = 4,
    restarts: int = 2,
    seed: int | None = None,
    time_budget: float | None = None,
) -> CandidateSet:
    """Attack autokey ciphertext in both forms; return ranked candidates.

    Options
    -------
    max_primer, min_primer:
        Inclusive range of primer lengths to try. A longer primer than
        *max_primer* is simply not attempted, and the solver has no way of
        detecting that, so ``diagnostics["max_primer_tried"]`` records it.
    modes:
        Which forms to attack; both by default, ranked in one list with the
        mode named in every ``Candidate.key``.
    exhaustive_limit:
        Try every primer whenever 26^m does not exceed this. Above it, the
        search starts from the per-chain chi-squared guess instead.
    probe_length:
        Letters of decrypted prefix used to screen primers during the
        exhaustive pass.
    climb_top:
        How many of the primer lengths to polish with the hill-climb.
    restarts, seed:
        Extra hill-climbs from random primers, to escape a local maximum.
        *seed* drives a private ``random.Random`` so runs are reproducible;
        the global random module is never touched.
    time_budget:
        Seconds. The search stops cleanly when exceeded and every candidate
        records ``time_budget_hit``.

    Remember what the module docstring says: these attacks fail often. Check
    ``diagnostics["meets_english_threshold"]`` and read the plaintext.
    """
    engine = scorer if scorer is not None else default_scorer()
    normalized = normalize(source) if isinstance(source, str) else source
    letters = normalized.letters
    results = CandidateSet()
    if not letters:
        return results

    if min_primer < 1:
        raise ValueError(f"min_primer must be at least 1, got {min_primer}")
    if max_primer < min_primer:
        raise ValueError(
            f"max_primer ({max_primer}) is below min_primer ({min_primer})"
        )
    wanted = [_resolve_mode(mode) for mode in modes]
    if not wanted:
        raise ValueError("At least one autokey mode must be requested")

    rng = random.Random(seed)
    deadline = time.monotonic() + time_budget if time_budget is not None else None
    budget_hit = False

    cipher_values = to_numbers(letters)
    sizes = [
        size
        for size in range(min_primer, max_primer + 1)
        if size < len(letters)  # a primer as long as the message hides everything
    ]
    if not sizes:
        return results

    attempts: list[dict] = []

    # -- ciphertext autokey: the tail is forced, so only m is unknown -------
    if CIPHERTEXT in wanted:
        for size in sizes:
            if deadline is not None and time.monotonic() > deadline:
                budget_hit = True
                break
            plain, primer, score, hit = _solve_ciphertext_head(
                cipher_values, size, engine, deadline, restarts, rng
            )
            budget_hit = budget_hit or hit
            attempts.append(
                {
                    "mode": CIPHERTEXT,
                    "size": size,
                    "primer": primer,
                    "plain": plain,
                    "score": score,
                    "stages": ["forced tail", "right-to-left head hill-climb"],
                    "margin": None,
                    "sweeps": 0,
                    "restarts": 0,
                }
            )

    # -- plaintext autokey: exhaustive where affordable, else guess + climb --
    if PLAINTEXT in wanted:
        starts: list[dict] = []
        for size in sizes:
            if deadline is not None and time.monotonic() > deadline:
                budget_hit = True
                break
            primer, margins = initial_primer(letters, size)
            # Record every stage that RAN, not only the stage that happened to
            # win. "What did you actually try?" is the question a diagnostic
            # has to answer.
            stages = ["per-chain chi-squared"]
            score = engine.score_values(_plaintext_values(cipher_values, primer))
            if ALPHABET_SIZE**size <= exhaustive_limit:
                found, found_score, hit = _exhaustive_primer(
                    cipher_values, size, engine, probe_length, 12, deadline
                )
                budget_hit = budget_hit or hit
                stages.append(
                    "exhaustive search cut short by the time budget"
                    if hit
                    else "exhaustive over all 26^%d primers" % size
                )
                if found is not None and found_score > score:
                    primer, score = found, found_score
            starts.append(
                {
                    "mode": PLAINTEXT,
                    "size": size,
                    "primer": primer,
                    "score": score,
                    "stages": stages,
                    "margin": sum(margins) / len(margins) if margins else 0.0,
                    "sweeps": 0,
                    "restarts": 0,
                }
            )

        # Polish only the most promising lengths: a hill-climb costs 26*m full
        # rescorings per sweep, which is far too expensive to spend on primer
        # lengths that the first pass already shows to be hopeless.
        ranked = sorted(starts, key=lambda row: row["score"], reverse=True)
        for row in ranked[: max(0, climb_top)]:
            if deadline is not None and time.monotonic() > deadline:
                budget_hit = True
                break
            primer, score, sweeps, hit = _climb_primer(
                cipher_values, row["primer"], engine, deadline
            )
            budget_hit = budget_hit or hit
            row["stages"].append("hill-climb")
            if score > row["score"]:
                row["primer"], row["score"] = primer, score
            row["sweeps"] = sweeps

            for _ in range(max(0, restarts)):
                if deadline is not None and time.monotonic() > deadline:
                    budget_hit = True
                    break
                seeded = [rng.randrange(ALPHABET_SIZE) for _ in range(row["size"])]
                primer, score, sweeps, hit = _climb_primer(
                    cipher_values, seeded, engine, deadline
                )
                budget_hit = budget_hit or hit
                row["restarts"] += 1
                row["sweeps"] += sweeps
                if score > row["score"]:
                    row["primer"], row["score"] = primer, score
                    row["stages"].append("seeded random restart won")

        for row in starts:
            row["plain"] = _plaintext_values(cipher_values, row["primer"])
            attempts.append(row)

    # -- report -------------------------------------------------------------
    for row in attempts:
        mode = row["mode"]
        plaintext = from_numbers(row["plain"])
        primer_text = from_numbers(row["primer"])
        diagnostics: dict = {
            "mode": mode,
            "primer_length": row["size"],
            "search": " + ".join(row["stages"]),
            "hill_climb_sweeps": row["sweeps"],
            "random_restarts": row["restarts"],
            "max_primer_tried": max(sizes),
            "letters": len(letters),
            "caveat": ATTACK_CAVEAT,
        }
        if row["margin"] is not None:
            diagnostics["chain_margin_mean"] = row["margin"]
        if mode == CIPHERTEXT:
            # Be explicit about which half of this answer is evidence and which
            # half is opinion. The tail needs no key; the head is a guess.
            diagnostics["forced_letters"] = max(0, len(letters) - row["size"])
            diagnostics["head_note"] = (
                f"letters {row['size'] + 1} onwards are forced by the "
                "ciphertext alone (P_i = C_i - C_{i-m}) and do not depend on "
                f"the primer; the first {row['size']} letters, and therefore "
                "the primer itself, are only the English model's preferred "
                "reading and may be wrong even when the rest is right"
            )
        if len(letters) < 200:
            diagnostics["short_text_warning"] = (
                f"only {len(letters)} letters -- an autokey attack on a text "
                "this short is unreliable and a wrong primer often outscores "
                "the right one"
            )
        if budget_hit:
            diagnostics["time_budget_hit"] = True
        annotate(diagnostics, plaintext, engine)
        diagnostics["meets_english_threshold"] = (
            diagnostics["normalised_score"] > ENGLISH_THRESHOLD
        )
        results.add(
            Candidate(
                method=_METHOD_NAME[mode],
                key=f"mode={mode} primer={primer_text}",
                score=engine.score(plaintext),
                plaintext=plaintext,
                diagnostics=diagnostics,
                display=normalized.relayout(plaintext),
            )
        )

    if top and top > 0:
        return CandidateSet(results.top(top))
    return results
