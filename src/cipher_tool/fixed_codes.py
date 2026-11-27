"""Fixed-length codes: each letter written as the same number of marks.

Baconian is the classic case (five A/B marks a letter). 2024 challenge 9B
used six of | / and \\. Once the stream is cut at the right length and
offset, what is left is a monoalphabetic substitution, so the work here is
finding the cut; the substitution solver does the rest.

A reading is kept when it gives 18 to 26 distinct codes and their
frequencies are uneven the way letters are. Uniform random marks fail the
second test at every length.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from . import substitution
from .candidates import Candidate, CandidateSet
from .normalize import NormalizedText
from .scoring import EnglishScorer

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SIZES = range(2, 9)
MIN_CODES = 18
MAX_CODES = 26
MIN_TOKENS = 100
# Index of coincidence times the number of distinct codes: 1.0 for uniform
# noise, about 1.7 for English letters.
MIN_ROUGHNESS = 1.4


@dataclass(frozen=True)
class Reading:
    size: int
    offset: int
    tokens: tuple[str, ...]
    roughness: float

    @property
    def codes(self) -> int:
        """How many different codes this cut produces."""
        return len(set(self.tokens))

    def as_letters(self) -> tuple[str, dict[str, str]]:
        """Relabel the codes as letters, most frequent first."""
        order = [code for code, _ in Counter(self.tokens).most_common()]
        label = dict(zip(order, ALPHABET))
        return "".join(label[t] for t in self.tokens), label


def _stream(source: str | NormalizedText) -> str:
    raw = source.original if isinstance(source, NormalizedText) else source
    return "".join(raw.split())


def _roughness(tokens: list[str]) -> float:
    counts = Counter(tokens)
    n = len(tokens)
    ic = sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))
    return ic * len(counts)


def readings(source: str | NormalizedText) -> list[Reading]:
    """Every plausible cut of the stream into equal codes, best first."""
    stream = _stream(source)
    # Digit streams belong to the Polybius and Nihilist routes, which also
    # report a dropped or odd digit; this one would hide that.
    if stream.isdigit() or not 2 <= len(set(stream)) <= 6:
        return []
    found = []
    for size in SIZES:
        for offset in range(size):
            usable = (len(stream) - offset) // size
            if usable < MIN_TOKENS:
                continue
            tokens = [stream[offset + i * size: offset + (i + 1) * size]
                      for i in range(usable)]
            if not MIN_CODES <= len(set(tokens)) <= MAX_CODES:
                continue
            rough = _roughness(tokens)
            if rough >= MIN_ROUGHNESS:
                found.append(Reading(size, offset, tuple(tokens), rough))
    # Halves or thirds of a real code can pass too, so the longest cut that
    # still gives few enough codes wins. Joining two real codes never does:
    # pairs of letters give hundreds of distinct tokens.
    return sorted(found, key=lambda r: (-r.size, -r.roughness, r.offset))


def solve(source: str | NormalizedText, *, scorer: EnglishScorer | None = None,
          top: int = 5, seed: int | None = None,
          time_budget: float | None = 60.0, limit: int = 3) -> CandidateSet:
    """Cut the stream into codes and solve the substitution underneath."""
    out = CandidateSet()
    options = readings(source)[:limit]
    share = time_budget / len(options) if time_budget and options else None
    for reading in options:
        letters, label = reading.as_letters()
        found = substitution.solve(letters, scorer=scorer, top=top, seed=seed,
                                   time_budget=share)
        back = {letter: code for code, letter in label.items()}
        for candidate in found.ranked()[:top]:
            book = {}
            for code_letter, plain in zip(letters, candidate.plaintext):
                book.setdefault(back[code_letter], plain)
            key = " ".join(f"{code}={plain}" for code, plain
                           in sorted(book.items(), key=lambda kv: kv[1]))
            out.add(Candidate(
                method=(f"Fixed-length code ({reading.size} marks per letter)"
                        " + substitution"),
                key=key + (f" (start at mark {reading.offset + 1})"
                           if reading.offset else ""),
                score=candidate.score,
                plaintext=candidate.plaintext,
                diagnostics={**candidate.diagnostics, "reads": "symbols",
                             "code_size": reading.size,
                             "code_offset": reading.offset,
                             "distinct_codes": reading.codes},
            ))
    return out
