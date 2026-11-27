"""Shuffled units: a mark code whose marks were then shuffled in fixed blocks.

2024 challenge 10B wrote each letter as two groups of three marks, each group
holding one | one / and one \\, and then shuffled every block of 21 marks the
same way. The marks that started in one group never repeat a mark, which is
what gives the shuffle away.
"""

from __future__ import annotations

import itertools
import random
import unittest

from cipher_tool import shuffled_units
from cipher_tool.auto import auto_solve
from cipher_tool.normalize import letters_only
from cipher_tool.scoring import corpus_files

MARKS = "\\/|"


def sample_plaintext(length: int) -> str:
    return letters_only(corpus_files()[2].read_text(encoding="utf-8"))[:length]


def encode(plaintext: str, period: int, seed: int) -> str:
    rng = random.Random(seed)
    units = ["".join(p) for p in itertools.permutations(MARKS)]
    pairs = [a + b for a in units for b in units]
    book = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", rng.sample(pairs, 26)))
    stream = "".join(book[ch] for ch in plaintext)
    shuffle = list(range(period))
    rng.shuffle(shuffle)
    out = []
    for start in range(0, len(stream), period):
        block = stream[start:start + period]
        out.append("".join(block[i] for i in shuffle))
    joined = "".join(out)
    return " ".join(joined[i:i + 5] for i in range(0, len(joined), 5))


class TestDetect(unittest.TestCase):
    def test_finds_the_period_and_groups(self) -> None:
        found = shuffled_units.detect(encode(sample_plaintext(1505), 21, 4))
        self.assertIsNotNone(found)
        self.assertEqual(found.period, 21)
        self.assertEqual(len(found.groups), 7)

    def test_ignores_random_marks(self) -> None:
        rng = random.Random(2)
        text = "".join(rng.choice(MARKS) for _ in range(9030))
        self.assertIsNone(shuffled_units.detect(text))

    def test_ignores_letters(self) -> None:
        self.assertIsNone(shuffled_units.detect(sample_plaintext(900)))


class TestSolve(unittest.TestCase):
    def test_solves_a_shuffled_code(self) -> None:
        plain = sample_plaintext(1505)
        best = shuffled_units.solve(encode(plain, 21, 7), seed=1).best()
        self.assertEqual(best.plaintext, plain)
        self.assertEqual(best.confidence(), "strong")

    def test_auto_solve_reaches_it(self) -> None:
        plain = sample_plaintext(1505)
        result = auto_solve(encode(plain, 21, 8), seed=1, max_time=90)
        self.assertEqual(result.candidates.best().plaintext, plain)


if __name__ == "__main__":
    unittest.main()
