"""Four-square encryption and a ciphertext-only simulated annealing attack.

The top-left and bottom-right squares are the alphabet without J. The
top-right and bottom-left squares are independently keyed. A rectangle sends
each plaintext pair to those keyed squares; unlike Playfair, doubles stay
together. J folds to I and an odd final letter is padded with X. Decryption
keeps that X, since an inserted pad cannot be distinguished from a real X.

The search follows Playfair and crossed-coordinate annealing: swap letters,
rows or columns in either square, accept downhill moves with the Metropolis
rule, cool and reheat, and keep the best keys across seeded restarts. The
shared English scorer judges each decryption; its word coverage and the
candidate confidence ladder judge the result. Short readings are leads, even
when they sound fluent.
"""

from __future__ import annotations

import math
import random
import time

from .candidates import Candidate, CandidateSet
from .normalize import NormalizedText, letters_only, normalize
from .scoring import EnglishScorer, annotate, default_scorer

ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
_VALUES = tuple(ord(letter) - 65 for letter in ALPHABET)
DEFAULT_RESTARTS = 6
DEFAULT_ITERATIONS = 60000
RELIABLE_CLIMB_LETTERS = 800


def keyed_square(keyword: str = "") -> str:
    """Return a keyword's 25 letters in row order, merging J into I.

    A complete square can also be supplied as the keyword. An empty keyword
    gives the standard square used for both plaintext squares.
    """
    key = letters_only(keyword).replace("J", "I")
    return "".join(dict.fromkeys(key + ALPHABET))


def encrypt(plaintext: str, key_one: str, key_two: str) -> str:
    """Encrypt using top-right *key_one* and bottom-left *key_two*.

    Drop nonletters, merge I/J, preserve doubled pairs, and pad with X when
    needed. Keys may be keywords or complete 25-letter squares.
    """
    one, two = keyed_square(key_one), keyed_square(key_two)
    text = letters_only(plaintext).replace("J", "I")
    if len(text) % 2:
        text += "X"
    positions = {letter: divmod(i, 5) for i, letter in enumerate(ALPHABET)}
    output = []
    for i in range(0, len(text), 2):
        row_one, col_one = positions[text[i]]
        row_two, col_two = positions[text[i + 1]]
        output.extend((one[5 * row_one + col_two],
                       two[5 * row_two + col_one]))
    return "".join(output)


def decrypt(ciphertext: str, key_one: str, key_two: str) -> str:
    """Decrypt two keyed squares, retaining any final padding X.

    Nonletters are ignored. Odd lengths and J in ciphertext raise ValueError:
    they cannot be produced by this version of four-square.
    """
    text = letters_only(ciphertext)
    if len(text) % 2 or "J" in text:
        raise ValueError("Four-square ciphertext needs even length and no J.")
    one, two = keyed_square(key_one), keyed_square(key_two)
    first = {letter: divmod(i, 5) for i, letter in enumerate(one)}
    second = {letter: divmod(i, 5) for i, letter in enumerate(two)}
    output = []
    for i in range(0, len(text), 2):
        row_one, col_two = first[text[i]]
        row_two, col_one = second[text[i + 1]]
        output.extend((ALPHABET[5 * row_one + col_one],
                       ALPHABET[5 * row_two + col_two]))
    return "".join(output)


def recognises(text: str) -> bool:
    """Whether text has the 25-letter, doubled-digraph four-square shape.

    This is evidence for an attack, not proof of the cipher family. Requiring
    all 25 letters is only a pipeline gate; solve() can attack smaller alphabets.
    """
    letters = letters_only(text)
    return (len(letters) >= 40 and len(letters) % 2 == 0
            and set(letters) == set(ALPHABET)
            and any(letters[i] == letters[i + 1]
                    for i in range(0, len(letters), 2)))


def _decrypt_values(pairs, one, two, buffer):
    """Decrypt inverse-square positions into the shared scorer's alphabet."""
    values = _VALUES
    for i, (left, right) in enumerate(pairs):
        first, second = one[left], two[right]
        buffer[2 * i] = values[first // 5 * 5 + second % 5]
        buffer[2 * i + 1] = values[second // 5 * 5 + first % 5]
    return buffer


def _square_from_positions(positions):
    """Invert letter-to-cell positions for the human-readable recovered key."""
    square = [""] * 25
    for letter, position in zip(ALPHABET, positions):
        square[position] = letter
    return "".join(square)


def solve(source: str | NormalizedText, *, scorer: EnglishScorer | None = None,
          top: int = 5, restarts: int = DEFAULT_RESTARTS,
          iterations: int = DEFAULT_ITERATIONS, seed: int | None = None,
          time_budget: float | None = None,
          require_shape: bool = False) -> CandidateSet:
    """Recover plaintext and both squares by annealing, without keywords.

    Return ranked Candidates with both squares in the key and diagnostics,
    and confidence through Candidate.confidence(). Restarts and iterations
    bound the search; seed makes it reproducible without a wall-clock limit.
    A time budget may interrupt a restart and is recorded in its diagnostics.
    The automatic pipeline uses require_shape to avoid speculative searches.
    """
    if restarts < 1 or iterations < 1 or top < 1:
        raise ValueError("restarts, iterations and top must be positive.")
    normalized = normalize(source) if isinstance(source, str) else source
    letters = normalized.letters
    found = CandidateSet(source_letters=letters)
    if (len(letters) < 40 or len(letters) % 2 or "J" in letters
            or (require_shape and not recognises(letters))):
        return found
    engine = scorer if scorer is not None else default_scorer()
    engine.table()
    rng = random.Random(seed)
    deadline = (time.monotonic() + max(0.0, time_budget)
                if time_budget is not None else None)
    pairs = [(ALPHABET.index(letters[i]), ALPHABET.index(letters[i + 1]))
             for i in range(0, len(letters), 2)]
    buffer = [0] * len(letters)
    score_values = engine.score_values
    cycle = max(iterations // 3, 1)

    for attempt in range(restarts):
        one, two = list(range(25)), list(range(25))
        rng.shuffle(one)
        rng.shuffle(two)
        current = score_values(_decrypt_values(pairs, one, two, buffer))
        best_score, best_one, best_two = current, list(one), list(two)
        budget_hit = False
        evaluated = 0
        for step in range(iterations):
            if (step & 255) == 0 and deadline is not None:
                if time.monotonic() >= deadline:
                    budget_hit = True
                    break
            # As in crossed._anneal, three cooling cycles let a stalled
            # search cross another valley without discarding its best keys.
            phase = (step % cycle) / cycle
            temperature = 0.04 * len(pairs) * (1.0 - phase) ** 2 + 0.2
            square = one if rng.random() < 0.5 else two
            if rng.random() < 0.1:
                # Move a whole row or column across a valley that five
                # individual swaps would have to cross one letter at a time.
                first, second = rng.sample(range(5), 2)
                previous = list(square)
                rows = rng.random() < 0.5
                for i, position in enumerate(square):
                    coordinate = position // 5 if rows else position % 5
                    if coordinate == first:
                        square[i] += (second - first) * (5 if rows else 1)
                    elif coordinate == second:
                        square[i] += (first - second) * (5 if rows else 1)
            else:
                previous = None
                first, second = rng.sample(range(25), 2)
                square[first], square[second] = square[second], square[first]
            value = score_values(_decrypt_values(pairs, one, two, buffer))
            evaluated += 1
            delta = value - current
            if delta >= 0 or rng.random() < math.exp(delta / temperature):
                current = value
                if current > best_score:
                    best_score, best_one, best_two = current, list(one), list(two)
            else:
                if previous is None:
                    square[first], square[second] = square[second], square[first]
                else:
                    square[:] = previous

        key_one = _square_from_positions(best_one)
        key_two = _square_from_positions(best_two)
        plaintext = decrypt(letters, key_one, key_two)
        diagnostics = {
            "square_one": key_one,
            "square_two": key_two,
            "ciphertext_letters": len(letters),
            "restart": attempt + 1,
            "moves_evaluated": evaluated,
            "time_budget_hit": budget_hit,
            "search": "simulated annealing over two squares; not exhaustive",
        }
        annotate(diagnostics, plaintext, engine)
        if len(letters) < RELIABLE_CLIMB_LETTERS:
            diagnostics["confidence_cap"] = "promising"
            diagnostics["short_ciphertext"] = (
                "Two unknown squares leave room for near-miss readings; "
                f"below {RELIABLE_CLIMB_LETTERS} letters treat this as a lead."
            )
        if diagnostics.get("word_coverage", 0.0) < 0.7:
            diagnostics["confidence_cap"] = "weak"
        candidate = Candidate(
            method="Four-square", key=f"square_one={key_one}; square_two={key_two}",
            score=engine.score(plaintext), plaintext=plaintext,
            diagnostics=diagnostics,
        )
        found.add(candidate)
        if budget_hit or candidate.confidence() == "strong":
            break
    return CandidateSet(found.ranked()[:top], source_letters=letters)
