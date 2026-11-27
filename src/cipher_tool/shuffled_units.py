"""Shuffled units: a mark code shuffled in fixed blocks after encoding.

2024 challenge 10B wrote each letter as two units of three marks, every unit
holding each of | / \\ once, then applied one shuffle to every block of 21
marks. Three steps undo it:

1. Period and units. Two marks that began in the same unit are never equal,
   so at the right period each position has exactly two partners it always
   differs from. Those trios are the units.
2. Unit order. A letter is two adjacent units, so the pair of units that
   really make a letter only ever shows about 25 of the 36 possible
   combinations, whatever order the marks inside each unit are read in.
   Every order of the units is scored by that count.
3. Order inside each unit. With the units in place, a climb picks the
   reading of each unit that leaves the fewest distinct letter codes (one
   alphabet instead of several). What is left is a substitution.

Only three-mark units, two to a letter, are handled.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from dataclasses import dataclass

from . import substitution
from .candidates import Candidate, CandidateSet
from .normalize import NormalizedText
from .scoring import EnglishScorer

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
UNIT = 3
MAX_PERIOD = 120
MAX_UNITS = 8          # unit orders are enumerated, so 8! at most
MIN_BLOCKS = 40
ALWAYS = 0.97          # "never equal", allowing a few transcription slips
READINGS = list(itertools.permutations(range(UNIT)))


@dataclass(frozen=True)
class Shuffle:
    period: int
    groups: tuple[tuple[int, ...], ...]
    blocks: tuple[str, ...]


def _stream(source: str | NormalizedText) -> str:
    raw = source.original if isinstance(source, NormalizedText) else source
    return "".join(raw.split())


def _differ(blocks: list[str], a: int, b: int) -> float:
    return sum(block[a] != block[b] for block in blocks) / len(blocks)


def detect(source: str | NormalizedText) -> Shuffle | None:
    """Find the shuffle period and its units, or None."""
    stream = _stream(source)
    if len(set(stream)) != UNIT:
        return None
    for period in range(2 * UNIT, MAX_PERIOD + 1, UNIT):
        count = len(stream) // period
        if count < MIN_BLOCKS or period // UNIT > MAX_UNITS:
            continue
        blocks = [stream[k * period:(k + 1) * period] for k in range(count)]
        # Cheap screen on position 0 before testing every pair.
        if sum(_differ(blocks, 0, b) >= ALWAYS
               for b in range(1, period)) != UNIT - 1:
            continue
        partners = {a: {b for b in range(period)
                        if b != a and _differ(blocks, a, b) >= ALWAYS}
                    for a in range(period)}
        groups = {tuple(sorted({a} | mates)) for a, mates in partners.items()}
        if (len(groups) == period // UNIT
                and all(len(g) == UNIT for g in groups)
                and sorted(i for g in groups for i in g) == list(range(period))):
            return Shuffle(period, tuple(sorted(groups)), tuple(blocks))
    return None


def _unit(block: str, group: tuple[int, ...], reading: int) -> str:
    return "".join(block[group[i]] for i in READINGS[reading])


def _combinations(found: Shuffle, first: int, second: int, *,
                  parity: int | None, across: bool) -> int:
    """How many different unit pairs this pair of groups shows."""
    seen = set()
    blocks = found.blocks
    for k in range(len(blocks) - (1 if across else 0)):
        if parity is not None and k % 2 != parity:
            continue
        a = _unit(blocks[k], found.groups[first], 0)
        b = _unit(blocks[k + 1] if across else blocks[k], found.groups[second], 0)
        seen.add((a, b))
    return len(seen)


def unit_orders(found: Shuffle, keep: int = 4) -> list[tuple[int, ...]]:
    """Orders of the units, best first, scored on the letter pairs they imply."""
    units = len(found.groups)
    cache: dict[tuple, int] = {}

    def cost(first: int, second: int, parity: int | None, across: bool) -> int:
        key = (first, second, parity, across)
        if key not in cache:
            cache[key] = _combinations(found, first, second,
                                       parity=parity, across=across)
        return cache[key]

    scored = []
    for order in itertools.permutations(range(units)):
        total = 0
        for parity in (0, 1):
            start = parity * units      # global unit index of the block start
            for j in range(units - 1):
                if (start + j) % 2 == 0:
                    total += cost(order[j], order[j + 1],
                                  parity if units % 2 else None, False)
            if (start + units - 1) % 2 == 0:
                total += cost(order[-1], order[0],
                              parity if units % 2 else None, True)
        scored.append((total, order))
    scored.sort()
    return [order for _, order in scored[:keep]]


def _letters(found: Shuffle, order: tuple[int, ...],
             readings: list[int]) -> list[str]:
    units = [_unit(block, found.groups[g], readings[g])
             for block in found.blocks for g in order]
    return [units[i] + units[i + 1] for i in range(0, len(units) - 1, 2)]


def _fit(codes: list[str]) -> tuple[int, float]:
    counts = Counter(codes)
    n = len(codes)
    ic = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))
    return (-len(counts), ic)


def unit_readings(found: Shuffle, order: tuple[int, ...],
                  rng: random.Random, restarts: int = 6) -> list[int]:
    """Choose how to read each unit so one alphabet covers every letter."""
    groups = len(found.groups)
    best, best_fit = None, None
    for attempt in range(restarts):
        readings = ([0] * groups if attempt == 0
                    else [rng.randrange(len(READINGS)) for _ in range(groups)])
        readings[order[0]] = 0      # one unit fixed: relabelling all is free
        fit = _fit(_letters(found, order, readings))
        improved = True
        while improved:
            improved = False
            for g in order[1:]:
                for r in range(len(READINGS)):
                    if r == readings[g]:
                        continue
                    trial = readings[:]
                    trial[g] = r
                    trial_fit = _fit(_letters(found, order, trial))
                    if trial_fit > fit:
                        readings, fit, improved = trial, trial_fit, True
        if best_fit is None or fit > best_fit:
            best, best_fit = readings, fit
    return best if best is not None else [0] * groups


def solve(source: str | NormalizedText, *, scorer: EnglishScorer | None = None,
          top: int = 5, seed: int | None = None,
          time_budget: float | None = 60.0) -> CandidateSet:
    """Undo the shuffle, read the units as letters, solve the substitution."""
    out = CandidateSet()
    found = detect(source)
    if found is None:
        return out
    rng = random.Random(seed)
    orders = unit_orders(found)
    share = time_budget / len(orders) if time_budget else None
    for order in orders:
        readings = unit_readings(found, order, rng)
        codes = _letters(found, order, readings)
        ranked = [code for code, _ in Counter(codes).most_common()]
        if len(ranked) > len(ALPHABET):
            continue
        label = dict(zip(ranked, ALPHABET))
        letters = "".join(label[code] for code in codes)
        solved = substitution.solve(letters, scorer=scorer, top=top, seed=seed,
                                    time_budget=share)
        for candidate in solved.ranked()[:top]:
            out.add(Candidate(
                method=(f"Shuffled mark code (blocks of {found.period}, "
                        "two units a letter) + substitution"),
                key=(f"unit order={[found.groups[g] for g in order]} "
                     f"readings={[READINGS[readings[g]] for g in order]}"),
                score=candidate.score,
                plaintext=candidate.plaintext,
                diagnostics={**candidate.diagnostics, "reads": "symbols",
                             "period": found.period,
                             "distinct_codes": len(ranked)},
            ))
    return out
