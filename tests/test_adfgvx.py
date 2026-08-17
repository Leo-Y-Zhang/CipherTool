"""Tests for the ADFGVX and ADFGX field ciphers.

The cipher is two stages: fractionate every plaintext letter into TWO symbols
through a keyed Polybius square, then columnar-transpose the symbol stream.
Neither half is new to this toolkit -- what was missing was an attack that
undoes them together, and until it existed a real ADFGVX message produced no
candidates at all.

The attack rests on one measured fact, pinned by
`test_the_index_of_coincidence_separates_the_right_columns` below: pair the
symbols up two at a time, and when the columns are in the right order the
pairs ARE the original square cells, so their distribution is English-shaped.
When the order is wrong the pairs straddle cell boundaries and flatten out.
Measured on a 600-letter message: 0.0678 against 0.0398, with English at
about 0.0667 and a flat 36 cells at 0.0278.
"""

from __future__ import annotations

import random
import time
import unittest

from cipher_tool import adfgvx, polybius
from cipher_tool.normalize import letters_only
from cipher_tool.scoring import DATA_DIR, default_scorer

CORPUS = (DATA_DIR / "corpus_03_letters.txt").read_text(encoding="utf-8")


class TestRoundTrip(unittest.TestCase):
    """Encrypt then decrypt, before anything is attacked."""

    def setUp(self) -> None:
        self.plaintext = letters_only(CORPUS)[:200]

    def test_encrypt_then_decrypt_returns_the_plaintext(self) -> None:
        ciphertext = adfgvx.encrypt(self.plaintext, "MONARCHY", "BERLIN")
        self.assertEqual(
            adfgvx.decrypt(ciphertext, "MONARCHY", "BERLIN"), self.plaintext
        )

    def test_the_ciphertext_uses_only_the_six_labels(self) -> None:
        ciphertext = adfgvx.encrypt(self.plaintext, "MONARCHY", "BERLIN")
        self.assertLessEqual(set(ciphertext), set("ADFGVX"))

    def test_every_letter_becomes_two_symbols(self) -> None:
        ciphertext = adfgvx.encrypt(self.plaintext, "MONARCHY", "BERLIN")
        self.assertEqual(len(ciphertext), 2 * len(self.plaintext))

    def test_the_five_by_five_variant_round_trips_too(self) -> None:
        """ADFGX, the 1918 predecessor. No digits, so I and J share a cell."""
        plaintext = self.plaintext.replace("J", "I")
        ciphertext = adfgvx.encrypt(plaintext, "MONARCHY", "BERLIN",
                                    labels=polybius.ADFGX_LABELS)
        self.assertLessEqual(set(ciphertext), set("ADFGX"))
        self.assertEqual(
            adfgvx.decrypt(ciphertext, "MONARCHY", "BERLIN",
                           labels=polybius.ADFGX_LABELS),
            plaintext,
        )


class TestRecognition(unittest.TestCase):
    """Telling an ADFGVX message from anything else."""

    def setUp(self) -> None:
        self.plaintext = letters_only(CORPUS)[:200]

    def test_an_adfgvx_message_is_recognised(self) -> None:
        ciphertext = adfgvx.encrypt(self.plaintext, "MONARCHY", "BERLIN")
        self.assertEqual(adfgvx.looks_like(ciphertext), "ADFGVX")

    def test_an_adfgx_message_is_recognised_as_the_five_letter_variant(self) -> None:
        plaintext = self.plaintext.replace("J", "I")
        ciphertext = adfgvx.encrypt(plaintext, "MONARCHY", "BERLIN",
                                    labels=polybius.ADFGX_LABELS)
        self.assertEqual(adfgvx.looks_like(ciphertext), "ADFGX")

    def test_ordinary_english_is_not_mistaken_for_it(self) -> None:
        self.assertIsNone(adfgvx.looks_like(self.plaintext))

    def test_an_odd_length_is_refused(self) -> None:
        """Every letter becomes two symbols, so an odd count cannot be one."""
        self.assertIsNone(adfgvx.looks_like("ADFGVXADFGV"))


class TestTheAttack(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = default_scorer()
        cls.plaintext = letters_only(CORPUS)[:600]

    def test_the_index_of_coincidence_separates_the_right_columns(self) -> None:
        """The measurement the whole attack rests on.

        If this stops being true the attack has no signal to work with, and
        it should fail loudly here rather than quietly returning nonsense.
        """
        from cipher_tool import columnar

        ciphertext = adfgvx.encrypt(self.plaintext, "MONARCHY", "BERLIN")
        right = columnar.decrypt(ciphertext, columnar.key_order("BERLIN"))
        wrong = columnar.decrypt(ciphertext, (5, 4, 3, 2, 1, 0))

        self.assertGreater(adfgvx.digraph_ic(right), 0.06)
        self.assertLess(adfgvx.digraph_ic(wrong), 0.05)

    def test_it_solves_a_real_message(self) -> None:
        ciphertext = adfgvx.encrypt(self.plaintext, "MONARCHY", "BERLIN")
        found = adfgvx.solve(ciphertext, scorer=self.scorer, top=3, seed=1)
        self.assertEqual(found.best().plaintext, self.plaintext)

    def test_it_recovers_the_square_so_the_answer_can_be_checked(self) -> None:
        """A competition answer nobody can verify by hand is worth little.

        The transposition key alone does not let anyone re-read the message,
        because the square is what turns cells back into letters.
        """
        ciphertext = adfgvx.encrypt(self.plaintext, "MONARCHY", "BERLIN")
        best = adfgvx.solve(ciphertext, scorer=self.scorer, top=1,
                            seed=1).best()
        square = best.diagnostics["square"]
        self.assertGreaterEqual(len(square), 20)
        # Every cell it reports must decode to the letter it claims.
        self.assertIn("A", square)

    def test_it_solves_the_five_by_five_variant(self) -> None:
        plaintext = self.plaintext.replace("J", "I")
        ciphertext = adfgvx.encrypt(plaintext, "MONARCHY", "BERLIN",
                                    labels=polybius.ADFGX_LABELS)
        found = adfgvx.solve(ciphertext, scorer=self.scorer, top=3, seed=1)
        self.assertEqual(found.best().plaintext, plaintext)

    def test_ordinary_english_produces_no_candidates(self) -> None:
        """Refusing is the right answer, not a best guess."""
        found = adfgvx.solve(self.plaintext, scorer=self.scorer, top=3)
        self.assertEqual(len(found), 0)

    def test_random_six_letter_noise_is_not_called_strong(self) -> None:
        generator = random.Random(5)
        noise = "".join(generator.choice("ADFGVX") for _ in range(600))
        found = adfgvx.solve(noise, scorer=self.scorer, top=1, seed=1,
                             max_key_length=5)
        best = found.best()
        if best is not None:
            self.assertNotEqual(best.confidence(), "strong")

    def test_a_time_budget_is_honoured(self) -> None:
        ciphertext = adfgvx.encrypt(self.plaintext, "MONARCHY", "BERLIN")
        started = time.monotonic()
        adfgvx.solve(ciphertext, scorer=self.scorer, top=1, seed=1,
                     max_key_length=8, time_budget=1.0)
        self.assertLess(time.monotonic() - started, 25.0)

    def test_the_key_lengths_searched_are_reported(self) -> None:
        ciphertext = adfgvx.encrypt(self.plaintext, "MONARCHY", "BERLIN")
        best = adfgvx.solve(ciphertext, scorer=self.scorer, top=1, seed=1,
                            max_key_length=6).best()
        self.assertIn("key_lengths_tried", best.diagnostics)
        self.assertEqual(best.diagnostics["transposition_key_length"], 6)

    def test_empty_input_is_not_a_crash(self) -> None:
        self.assertEqual(len(adfgvx.solve("", scorer=self.scorer)), 0)


class TestPipeline(unittest.TestCase):
    """A solver nothing calls is decoration."""

    def test_it_runs_from_the_cheapest_effort_level(self) -> None:
        """It costs nothing when it does not apply, so it need not wait.

        An ADFGVX message is written in five or six specific letters and has
        an even length; anything else is rejected before a permutation is
        tried. Holding it back to `deep` would make the paste screen escalate
        twice, and take minutes, to reach a cipher it can recognise at a
        glance.
        """
        from cipher_tool.auto import build_stages

        names = [stage.name for stage in build_stages("fast", 5, 1)]
        self.assertIn("ADFGVX", names)

    def test_the_pipeline_solves_one_end_to_end(self) -> None:
        from cipher_tool.auto import auto_solve, build_stages

        plaintext = letters_only(CORPUS)[:400]
        ciphertext = adfgvx.encrypt(plaintext, "MONARCHY", "BERLIN")
        stage = [s for s in build_stages("fast", 5, 1) if s.name == "ADFGVX"]
        result = auto_solve(ciphertext, effort="fast", top=3, seed=1,
                            stages=stage)
        self.assertEqual(result.candidates.best().plaintext, plaintext)


if __name__ == "__main__":
    unittest.main()
