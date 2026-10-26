"""Delastelle Trifid, using a keyed 3x3x3 cube and ternary fractionation.

Coordinates are zero based, in layer/row/column order. Within each period
all first coordinates precede all second coordinates and then all thirds.
Period zero (or None) means the whole message. Raw ciphertext is that same
transposed coordinate stream, before regrouping into ciphertext letters.
Searches are heuristic; an unsuccessful search does not rule out Trifid.
"""

from __future__ import annotations

import itertools
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .candidates import Candidate, CandidateSet
from .normalize import ALPHABET, NormalizedText, letters_only
from .scoring import EnglishScorer, annotate, default_scorer

DEFAULT_ALPHABET = ALPHABET + "+"
DEFAULT_RESTARTS = 4
DEFAULT_ITERATIONS = 20000


@dataclass(frozen=True)
class TrifidCube:
    """Twenty-six letters and one ASCII filler in layer/row/column order."""

    letters: str = DEFAULT_ALPHABET

    def __post_init__(self) -> None:
        if (len(self.letters) != 27 or len(set(self.letters)) != 27
                or not set(ALPHABET) <= set(self.letters)
                or not self.letters.isascii()
                or self.filler.isspace() or self.filler.isalnum()):
            raise ValueError("a cube needs A-Z and one distinct ASCII punctuation filler")

    @property
    def filler(self) -> str:
        """The cube's twenty-seventh symbol."""
        return next((ch for ch in self.letters if ch not in ALPHABET), "")

    def position(self, letter: str) -> tuple[int, int, int]:
        """Return a symbol's zero-based layer, row and column."""
        index = self.letters.index(letter.upper())
        return index // 9, index // 3 % 3, index % 3

    def at(self, layer: int, row: int, column: int) -> str:
        """Return the symbol at three coordinates, each in 0..2."""
        if any(value not in range(3) for value in (layer, row, column)):
            raise ValueError("cube coordinates must be in 0..2")
        return self.letters[9 * layer + 3 * row + column]


def trifid_cube(keyword: str = "", *, filler: str = "+") -> TrifidCube:
    """Build a cube from a keyword followed by unused letters and the filler."""
    if len(filler) != 1 or filler in ALPHABET or not filler.isascii():
        raise ValueError("filler must be one ASCII punctuation character")
    return TrifidCube("".join(dict.fromkeys(letters_only(keyword) + ALPHABET + filler)))


def _cube(key: TrifidCube | str | None) -> TrifidCube:
    """Accept a cube, a keyword, or the unkeyed default."""
    if isinstance(key, TrifidCube):
        return key
    if key is not None and len(key) == 27 and set(ALPHABET) <= set(key.upper()):
        return TrifidCube(key.upper())
    return trifid_cube(key or "")


def _clean(text: str, cube: TrifidCube) -> str:
    """Preserve letters and the chosen filler, ignoring layout."""
    return "".join(ch for ch in text.upper() if ch in cube.letters)


def _period(period: int | None, length: int) -> int:
    """Validate a period and resolve whole-message notation."""
    if period is not None and (not isinstance(period, int) or period < 0):
        raise ValueError("period must be a nonnegative integer or None")
    return period or max(1, length)


def letters_to_trits(text: str, cube: TrifidCube | str | None = None) -> str:
    """Encode cube symbols as consecutive triples of digits 0, 1 and 2."""
    board = _cube(cube)
    return "".join(str(value) for ch in _clean(text, board)
                   for value in board.position(ch))


def trits_to_letters(stream: str, cube: TrifidCube | str | None = None,
                     *, symbols: str = "012") -> str:
    """Decode triples under an explicit ordered three-symbol alphabet.

    Whitespace is layout. Invalid symbols and incomplete triples are errors.
    """
    values = _trits(stream, symbols)
    board = _cube(cube)
    return "".join(board.letters[9 * values[i] + 3 * values[i + 1] + values[i + 2]]
                   for i in range(0, len(values), 3))


def _trits(stream: str, symbols: str) -> list[int]:
    """Validate an explicitly mapped ternary stream."""
    if len(symbols) != 3 or len(set(symbols)) != 3 or any(c.isspace() for c in symbols):
        raise ValueError("symbols must contain three distinct non-whitespace characters")
    body = "".join(stream.split())
    if len(body) % 3 or set(body) - set(symbols):
        raise ValueError("a trit stream needs complete triples from its three symbols")
    return [symbols.index(ch) for ch in body]


def encrypt(text: str, cube: TrifidCube | str | None = None,
            period: int | None = 5) -> str:
    """Encrypt letters and the cube filler using classic Delastelle Trifid."""
    board = _cube(cube)
    plain = _clean(text, board)
    size = _period(period, len(plain))
    stream = []
    for start in range(0, len(plain), size):
        coordinates = [board.position(ch) for ch in plain[start:start + size]]
        stream.extend(str(point[axis]) for axis in range(3) for point in coordinates)
    return trits_to_letters("".join(stream), board)


def _undo(values: Sequence[int], period: int | None) -> list[int]:
    """Undo period transposition, returning plaintext cube cell indices."""
    size = 3 * _period(period, len(values) // 3)
    cells = []
    for start in range(0, len(values), size):
        block = values[start:start + size]
        width = len(block) // 3
        cells.extend(9 * block[i] + 3 * block[width + i] + block[2 * width + i]
                     for i in range(width))
    return cells


def decrypt(text: str, cube: TrifidCube | str | None = None,
            period: int | None = 5) -> str:
    """Decrypt letter ciphertext, preserving the cube's filler symbol."""
    board = _cube(cube)
    values = [int(ch) for ch in letters_to_trits(text, board)]
    return "".join(board.letters[cell] for cell in _undo(values, period))


def decrypt_trits(stream: str, cube: TrifidCube | str | None = None,
                  period: int | None = 5, *, symbols: str = "012") -> str:
    """Undo raw ciphertext fractionation with an explicitly ordered alphabet."""
    board = _cube(cube)
    return "".join(board.letters[cell] for cell in _undo(_trits(stream, symbols), period))


def recognise(source: str | NormalizedText) -> bool:
    """Recognise three-symbol streams of balanced counts or complete triples.

    This is a routing hint, never evidence that a decryption is correct.
    Whitespace is ignored; every other character belongs to the stream.
    """
    raw = source.original if isinstance(source, NormalizedText) else source
    body = "".join(raw.split())
    counts = Counter(body)
    return (len(counts) == 3
            and (len(body) % 3 == 0 or max(counts.values()) <= 1.25 * min(counts.values())))


def _anneal(evaluate, start, rng, iterations, deadline, temperature):
    """Anneal a permutation, retaining its best state across accepted moves."""
    cells = list(start)
    current = best = evaluate(cells)
    best_cells = cells[:]
    for step in range(iterations):
        if step % 64 == 0 and deadline is not None and time.monotonic() >= deadline:
            break
        first, second = rng.sample(range(27), 2)
        cells[first], cells[second] = cells[second], cells[first]
        score = evaluate(cells)
        delta = score - current
        heat = temperature * (1.0 - step / iterations) + 0.01
        if delta >= 0 or rng.random() < math.exp(delta / heat):
            current = score
            if score > best:
                best, best_cells = score, cells[:]
        else:
            cells[first], cells[second] = cells[second], cells[first]
    return best, best_cells


def _evaluator(cipher, period, engine, raw):
    """Build a fast integer decryption/scoring closure for one period."""
    score = engine.score_values
    if raw:
        indices = _undo(cipher, period)

        def evaluate(cells):
            # Penalise filler rather than deleting it and creating false joins.
            values = [cells[i] for i in indices]
            return score([v if v < 26 else 25 for v in values]) - 8 * values.count(26)
    else:
        length = len(cipher)
        # Precompute which ciphertext trits make each plaintext cell.
        triples = []
        size = _period(period, length)
        for start in range(0, length, size):
            width = min(size, length - start)
            triples.extend((3 * start + i, 3 * start + width + i,
                            3 * start + 2 * width + i) for i in range(width))

        def evaluate(cells):
            inverse = [0] * 27
            for i, value in enumerate(cells):
                inverse[value] = i
            coords = [(i // 9, i // 3 % 3, i % 3) for i in inverse]
            trits = [v for ch in cipher for v in coords[ch]]
            values = [cells[9 * trits[a] + 3 * trits[b] + trits[c]]
                      for a, b, c in triples]
            return score([v if v < 26 else 25 for v in values]) - 8 * values.count(26)
    return evaluate


def solve(source: str | NormalizedText, *, scorer: EnglishScorer | None = None,
          top: int = 5, period: int | None = None,
          periods: Sequence[int] | None = None, raw: bool | None = None,
          filler: str = "+", seed: int | None = None,
          restarts: int = DEFAULT_RESTARTS, iterations: int = DEFAULT_ITERATIONS,
          time_budget: float | None = 60.0) -> CandidateSet:
    """Search unknown Trifid cubes and periods, returning ranked candidates.

    By default try periods 3..20 and zero (the whole message). ``period=0``
    pins whole-message fractionation. ``raw`` is inferred from recognition,
    or can be supplied explicitly. All six raw symbol mappings are tested;
    they are equivalent cube coordinate relabellings, so share their climb.
    Seeded runs with no time limit are reproducible. A time limit can cut a
    restart short, and diagnostics report the periods and work completed.
    Keyword cubes from the local lexicon seed letter-ciphertext searches;
    annealing can change any of the 27 cells and also uses random restarts.
    """
    if restarts < 1 or iterations < 1 or top < 1:
        raise ValueError("restarts, iterations and top must be positive")
    if time_budget is not None and time_budget <= 0:
        raise ValueError("time_budget must be positive or None")
    board = trifid_cube(filler=filler)
    text = source.original if isinstance(source, NormalizedText) else source
    body = "".join(text.split())
    is_raw = recognise(source) if raw is None else raw
    results = CandidateSet()
    if is_raw and (len(set(body)) != 3 or len(body) % 3):
        return results
    alphabet = ALPHABET + board.filler
    cipher = (_trits(body, "".join(sorted(set(body)))) if is_raw
              else [alphabet.index(ch) for ch in _clean(text, board)])
    length = len(cipher) // 3 if is_raw else len(cipher)
    if length < 20:
        return results
    values = list(dict.fromkeys(periods if periods is not None else
                               [period] if period is not None else [*range(3, 21), 0]))
    for value in values:
        _period(value, length)
    engine = scorer or default_scorer()
    engine.table()
    rng = random.Random(seed)
    deadline = time.monotonic() + time_budget if time_budget is not None else None
    expired = lambda: deadline is not None and time.monotonic() >= deadline
    screened = []
    mappings = list(itertools.permutations(sorted(set(body)))) if is_raw else []
    words = sorted(engine.lexicon) if not is_raw else []
    for value in values:
        if expired():
            break
        evaluate = _evaluator(cipher, value, engine, is_raw)
        start = list(range(27))
        if is_raw:
            counts = Counter(_undo(cipher, value))
            ranked = sorted(range(27), key=lambda i: (-counts[i], i))
            frequencies = sorted(range(26), key=lambda i: (-engine.score_values([i]), i)) + [26]
            for cell, letter in zip(ranked, frequencies):
                start[cell] = letter
        else:
            # A keyword is only a starting hypothesis, not a restriction on
            # the recovered cube. Score a short prefix during this cheap pass.
            prefix = length if not value else min(length, max(value, 180 // value * value))
            quick = _evaluator(cipher[:prefix], value, engine, False)
            best = quick(start)
            for index, word in enumerate(words):
                if index % 64 == 0 and expired():
                    break
                key = [alphabet.index(ch) for ch in trifid_cube(word, filler=filler).letters]
                score = quick(key)
                if score > best:
                    best, start = score, key
        score, cells = _anneal(evaluate, start, rng, min(iterations, 1200),
                               deadline, max(2.0, length * 0.015))
        screened.append((score, value, cells, evaluate))
    screened.sort(key=lambda row: row[0], reverse=True)
    for score, value, cells, evaluate in screened[:3]:
        attempts = 0
        for attempt in range(restarts):
            if expired():
                break
            attempts += 1
            start = cells[:] if attempt == 0 else rng.sample(range(27), 27)
            trial, key = _anneal(evaluate, start, rng, iterations, deadline,
                                 max(2.0, length * 0.015))
            if trial > score:
                score, cells = trial, key
            if score / length > -1.05:
                break
        cube = TrifidCube("".join(alphabet[i] for i in cells))
        orders = mappings if is_raw else [None]
        seen_plaintexts = set()
        for order in orders:
            if order is not None:
                # Applying one trit permutation to all three axes commutes
                # with period transposition. This represents all six maps.
                base = sorted(set(body))
                perm = [base.index(ch) for ch in order]
                remapped = []
                for i in range(27):
                    a, b, c = i // 9, i // 3 % 3, i % 3
                    remapped.append(cube.letters[9 * perm[a] + 3 * perm[b] + perm[c]])
                recovered = TrifidCube("".join(remapped))
                plaintext = decrypt_trits(body, recovered, value, symbols="".join(order))
            else:
                recovered = cube
                plaintext = decrypt(text, cube, value)
            if plaintext in seen_plaintexts:
                continue
            seen_plaintexts.add(plaintext)
            diagnostics = {"period": value, "cube": recovered.letters,
                           "reads": "symbols" if is_raw else "letters",
                           "periods_tested": [row[1] for row in screened],
                           "symbol_mappings_tested": len(mappings),
                           "symbol_mappings": ["".join(order) for order in mappings],
                           "restarts_requested": restarts,
                           "restarts_run": attempts,
                           "iterations_per_restart": iterations,
                           "time_budget_hit": expired(),
                           "search": "simulated annealing over 27 cells (not exhaustive)"}
            if order is not None:
                diagnostics["symbols"] = "".join(order)
            annotate(diagnostics, plaintext, engine)
            # A filler-rich or short reading has insufficient evidence.
            if length < 300 or plaintext.count(filler) > max(1, length // 100):
                diagnostics["confidence_cap"] = "promising"
            description = f"period={value} cube={recovered.letters}"
            if order is not None:
                description += f" symbols={''.join(order)!r}"
            results.add(Candidate("Trifid", description,
                                  engine.score(plaintext), plaintext, diagnostics))
    return CandidateSet(results.ranked()[:top])
