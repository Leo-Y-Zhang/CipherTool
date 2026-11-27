"""Deep effort reaches long autokey primers.

2024 challenge 10A was a plaintext autokey with the 11-letter primer
CATACLYSMIC. Deep effort stopped at 8, so the chain attack that solves it in
seconds was never asked to try 11. Long primers have their own cheap stage.
"""

from __future__ import annotations

import unittest

from cipher_tool import autokey
from cipher_tool.auto import build_stages
from cipher_tool.normalize import letters_only
from cipher_tool.scoring import corpus_files


class TestLongPrimer(unittest.TestCase):
    def test_deep_stage_solves_an_eleven_letter_primer(self) -> None:
        plain = letters_only(corpus_files()[2].read_text(encoding="utf-8"))[:900]
        cipher = autokey.encrypt(plain, "CATACLYSMIC")
        stage = next(s for s in build_stages("deep", 5, 1) if s.name == "autokey (long primers)")
        best = stage.run(cipher, **stage.options).best()
        self.assertEqual(best.plaintext, plain)
        self.assertEqual(best.confidence(), "strong")


if __name__ == "__main__":
    unittest.main()
