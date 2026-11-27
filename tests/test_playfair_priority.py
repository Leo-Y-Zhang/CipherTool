"""A Playfair-shaped ciphertext gets Playfair first and a real share of time.

2025 challenge 9B is a Playfair that the solver cracks in about half a
minute on its own. Inside the deep pipeline it ran seventh with an 18 second
share and came back weak, so the pipeline answered with a substitution.
"""

from __future__ import annotations

import unittest

from cipher_tool import playfair
from cipher_tool.auto import build_stages, order_stages
from cipher_tool.normalize import letters_only
from cipher_tool.scoring import corpus_files
from cipher_tool.statistics import analyse


def playfair_text() -> str:
    plain = letters_only(corpus_files()[2].read_text(encoding="utf-8"))[:900]
    return playfair.encrypt(plain, "MONARCHY")


class TestPlayfairPriority(unittest.TestCase):
    def test_signature_puts_playfair_first_with_a_third_of_the_weight(self) -> None:
        text = playfair_text()
        ordered = order_stages(build_stages("deep", 5, 1), analyse(text))
        self.assertEqual(ordered[0].name, "Playfair")
        others = sum(stage.weight for stage in ordered[1:])
        self.assertGreaterEqual(ordered[0].weight, others / 2)

    def test_no_signature_leaves_playfair_alone(self) -> None:
        text = letters_only(corpus_files()[2].read_text(encoding="utf-8"))[:900]
        stages = build_stages("deep", 5, 1)
        ordered = order_stages(stages, analyse(text))
        before = next(s for s in stages if s.name == "Playfair")
        after = next(s for s in ordered if s.name == "Playfair")
        self.assertEqual(after.weight, before.weight)
        self.assertNotEqual(ordered[0].name, "Playfair")


if __name__ == "__main__":
    unittest.main()
