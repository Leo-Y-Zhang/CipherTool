"""Fixed-length symbol codes: every letter written as the same number of marks.

2024 challenge 9B wrote each letter as six of the marks | / and \\. Read in
threes it looks like a Trifid stream, so the pipeline used to send it there
and get nothing. Read in sixes it is a plain substitution with 24 codes.
"""

from __future__ import annotations

import itertools
import random
import unittest

from cipher_tool import fixed_codes
from cipher_tool.auto import auto_solve
from cipher_tool.normalize import letters_only
from cipher_tool.scoring import corpus_files


def sample_plaintext(length: int) -> str:
    return letters_only(corpus_files()[2].read_text(encoding="utf-8"))[:length]


def encode(plaintext: str, marks: str, size: int, seed: int) -> str:
    rng = random.Random(seed)
    codes = ["".join(c) for c in itertools.product(marks, repeat=size)]
    book = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", rng.sample(codes, 26)))
    stream = "".join(book[ch] for ch in plaintext)
    return " ".join(stream[i:i + 5] for i in range(0, len(stream), 5))


class TestReadings(unittest.TestCase):
    def test_finds_the_code_length(self) -> None:
        text = encode(sample_plaintext(1200), "\\/|", 6, seed=3)
        best = fixed_codes.readings(text)[0]
        self.assertEqual(best.size, 6)
        self.assertEqual(best.offset, 0)
        self.assertLessEqual(len(set(best.tokens)), 26)

    def test_finds_an_offset_start(self) -> None:
        # A stray mark at the front shifts every code by one.
        text = "|" + encode(sample_plaintext(1200), "\\/|", 4, seed=5)
        best = fixed_codes.readings(text)[0]
        self.assertEqual((best.size, best.offset), (4, 1))

    def test_ignores_ordinary_letters(self) -> None:
        self.assertEqual(fixed_codes.readings(sample_plaintext(800)), [])

    def test_ignores_a_random_mark_stream(self) -> None:
        rng = random.Random(9)
        text = "".join(rng.choice("\\/|") for _ in range(6000))
        self.assertEqual(fixed_codes.readings(text), [])


class TestSolve(unittest.TestCase):
    def test_solves_a_six_mark_code(self) -> None:
        plain = sample_plaintext(1500)
        found = fixed_codes.solve(encode(plain, "\\/|", 6, seed=11), seed=1)
        best = found.best()
        self.assertEqual(best.plaintext, plain)
        self.assertEqual(best.confidence(), "strong")
        self.assertIn("6 marks per letter", best.method)

    def test_auto_solve_reaches_it_before_trifid(self) -> None:
        plain = sample_plaintext(1500)
        result = auto_solve(encode(plain, "\\/|", 6, seed=12), seed=1,
                            max_time=60)
        best = result.candidates.best()
        self.assertEqual(best.plaintext, plain)
        self.assertEqual(best.confidence(), "strong")


if __name__ == "__main__":
    unittest.main()
