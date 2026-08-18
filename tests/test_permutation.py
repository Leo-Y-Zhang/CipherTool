"""Tests for the permutation cipher: one fixed shuffle inside every block.

This family was missing outright. The toolkit could break columnar, double
columnar, rail fence and every route through a grid, and none of them can
describe "swap these five positions, over and over" -- so the 2018 National
Cipher Challenge 6B message came back `weak` from a solver that had searched
column counts up to 63. `test_the_columnar_solver_cannot_express_this` below
is the test that pins WHY the module exists; if it ever starts passing
through columnar, this module is redundant and should go.

The other measurement worth stating: swept over five different texts at each
of five block sizes and six lengths from 60 to 500 letters, the solver
returned the exact plaintext in every one of the 150 runs, and produced NO
case of a wrong answer labelled `strong`. That is why there is no confidence
cap here, unlike Playfair -- a wrong block permutation is noise, not
near-English, so the score catches it. Decided by measurement, not argument.
"""

from __future__ import annotations

import random
import time
import unittest

from cipher_tool import columnar, permutation
from cipher_tool.normalize import letters_only
from cipher_tool.scoring import DATA_DIR, default_scorer

CORPUS = (DATA_DIR / "corpus_03_letters.txt").read_text(encoding="utf-8")


class TestTheCipher(unittest.TestCase):
    """Encrypt and decrypt, before anything is attacked."""

    def setUp(self) -> None:
        self.plaintext = letters_only(CORPUS)[:200]

    def test_encrypt_then_decrypt_returns_the_plaintext(self) -> None:
        for key in ("BAEDC", "ZEBRAS", (2, 0, 3, 1), (1, 0)):
            with self.subTest(key=key):
                ciphertext = permutation.encrypt(self.plaintext, key)
                self.assertEqual(
                    permutation.decrypt(ciphertext, key), self.plaintext
                )

    def test_it_moves_letters_without_changing_them(self) -> None:
        ciphertext = permutation.encrypt(self.plaintext, "BAEDC")
        self.assertNotEqual(ciphertext, self.plaintext)
        self.assertEqual(sorted(ciphertext), sorted(self.plaintext))

    def test_a_keyword_and_its_read_order_agree(self) -> None:
        """BAEDC sorts to ABCDE, which reads positions 1, 0, 4, 3, 2."""
        self.assertEqual(columnar.key_order("BAEDC"), (1, 0, 4, 3, 2))
        self.assertEqual(
            permutation.encrypt(self.plaintext, "BAEDC"),
            permutation.encrypt(self.plaintext, (1, 0, 4, 3, 2)),
        )

    def test_no_letter_leaves_its_own_block(self) -> None:
        """The property that distinguishes this from every columnar cipher."""
        period = 5
        ciphertext = permutation.encrypt(self.plaintext, "BAEDC")
        for start in range(0, len(self.plaintext) - period, period):
            block_in = sorted(self.plaintext[start:start + period])
            block_out = sorted(ciphertext[start:start + period])
            self.assertEqual(block_in, block_out)

    def test_the_ragged_last_block_is_permuted_too(self) -> None:
        """A short block keeps the key's relative order; it is not left alone.

        Found on the real message: the 2018 6B plaintext ends CONSTANTINOPLE
        and its last two letters are a block of two. Leaving a short block
        untouched spells it CONSTANTINOPEL -- right for 2,140 letters and
        wrong for the two a reader looks at last.
        """
        plaintext = "CONSTANTINOPLE"          # 14 letters: two full blocks + 4
        ciphertext = permutation.encrypt(plaintext, "BAEDC")
        self.assertEqual(permutation.decrypt(ciphertext, "BAEDC"), plaintext)

        tail = "LE"
        self.assertNotEqual(permutation.encrypt(tail, "BAEDC"), tail)

    def test_a_key_that_is_not_a_permutation_is_refused(self) -> None:
        for bad in ((0, 1, 1), (0, 2), (0,), (0, 1, 3)):
            with self.subTest(key=bad):
                with self.assertRaises(ValueError):
                    permutation.encrypt("HELLOTHERE", bad)

    def test_a_fractional_offset_is_refused_rather_than_rounded(self) -> None:
        with self.assertRaises(ValueError):
            permutation.encrypt("HELLOTHERE", (0, 1.5, 2))


class TestTheAttack(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = default_scorer()
        cls.plaintext = letters_only(CORPUS)[:400]

    def test_it_recovers_a_known_key(self) -> None:
        ciphertext = permutation.encrypt(self.plaintext, "BAEDC")
        best = permutation.solve(ciphertext, scorer=self.scorer, top=1,
                                 seed=1).best()
        self.assertEqual(best.plaintext, self.plaintext)
        self.assertEqual(best.diagnostics["period"], 5)

    def test_it_recovers_keys_of_several_lengths(self) -> None:
        for key in ("BAEDC", "ZEBRAS", "CATION", (3, 1, 0, 2)):
            with self.subTest(key=key):
                ciphertext = permutation.encrypt(self.plaintext, key)
                best = permutation.solve(ciphertext, scorer=self.scorer,
                                         top=1, seed=1).best()
                self.assertEqual(best.plaintext, self.plaintext)

    def test_the_key_it_reports_actually_decrypts_the_message(self) -> None:
        """A key that does not reproduce the answer is worse than no key.

        This needs a permutation that is NOT its own inverse, and that is the
        whole point of the test. BAEDC reads (1, 0, 4, 3, 2), which is a pair
        of swaps and therefore an involution -- so a solver that forgot to
        invert the arrangement before reporting it still passes every test
        built on BAEDC. Deleting the inversion was caught only here.

        The first assertion guards the premise: if someone later changes the
        key to an involution the test goes quiet rather than wrong.
        """
        key = (2, 0, 3, 1)
        inverse = [0] * len(key)
        for position, offset in enumerate(key):
            inverse[offset] = position
        self.assertNotEqual(list(key), inverse)

        ciphertext = permutation.encrypt(self.plaintext, key)
        best = permutation.solve(ciphertext, scorer=self.scorer, top=1,
                                 seed=1).best()
        self.assertEqual(best.plaintext, self.plaintext)
        reported = tuple(
            int(value) for value in best.diagnostics["read_order"].split(",")
        )
        self.assertEqual(
            permutation.decrypt(ciphertext, reported), self.plaintext
        )
        self.assertEqual(
            permutation.decrypt(ciphertext, best.key.split(" ")[0]),
            self.plaintext,
        )

    def test_it_is_confident_about_a_right_answer(self) -> None:
        ciphertext = permutation.encrypt(self.plaintext, "BAEDC")
        best = permutation.solve(ciphertext, scorer=self.scorer, top=1,
                                 seed=1).best()
        self.assertEqual(best.confidence(), "strong")

    def test_the_columnar_solver_cannot_express_this(self) -> None:
        """Why this module exists at all.

        A period-5 permutation is not a columnar transposition of any width,
        so the columnar solver has no key that describes it and cannot reach
        the plaintext. If this ever fails, the family is covered elsewhere.
        """
        ciphertext = permutation.encrypt(self.plaintext, "BAEDC")
        found = columnar.solve(ciphertext, scorer=self.scorer, top=5, seed=1,
                               max_key_length=9)
        for candidate in found.ranked():
            self.assertNotEqual(candidate.plaintext, self.plaintext)

    def test_it_never_offers_the_identity(self) -> None:
        """Otherwise every piece of plain English "decrypts" to itself."""
        found = permutation.solve(self.plaintext, scorer=self.scorer, top=5,
                                  seed=1)
        for candidate in found.ranked():
            self.assertNotEqual(candidate.plaintext, self.plaintext)

    def test_random_letters_are_not_called_strong(self) -> None:
        generator = random.Random(11)
        noise = "".join(generator.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                        for _ in range(400))
        best = permutation.solve(noise, scorer=self.scorer, top=1,
                                 seed=1).best()
        if best is not None:
            self.assertNotEqual(best.confidence(), "strong")

    def test_a_wrong_answer_is_never_labelled_strong(self) -> None:
        """The property that matters most, swept across DIFFERENT texts.

        Five texts at each of three block sizes. Sweeping seeds instead of
        texts is what made the first Playfair confidence fix wrong.
        """
        corpus = letters_only(CORPUS)
        generator = random.Random(20260818)
        for period in (4, 5, 6):
            for index in range(5):
                at = generator.randrange(0, len(corpus) - 200)
                plaintext = corpus[at:at + 150]
                order = list(range(period))
                while tuple(order) == tuple(range(period)):
                    generator.shuffle(order)
                ciphertext = permutation.encrypt(plaintext, order)
                best = permutation.solve(ciphertext, scorer=self.scorer,
                                         top=1, seed=index).best()
                with self.subTest(period=period, index=index):
                    if best is not None and best.confidence() == "strong":
                        self.assertEqual(best.plaintext, plaintext)

    def test_pinning_the_period_searches_only_that_one(self) -> None:
        ciphertext = permutation.encrypt(self.plaintext, "BAEDC")
        found = permutation.solve(ciphertext, scorer=self.scorer, top=5,
                                  seed=1, period=5)
        for candidate in found.ranked():
            self.assertEqual(candidate.diagnostics["period"], 5)

    def test_a_time_budget_is_honoured(self) -> None:
        ciphertext = permutation.encrypt(self.plaintext, "BAEDC")
        started = time.monotonic()
        permutation.solve(ciphertext, scorer=self.scorer, top=1, seed=1,
                          max_period=12, time_budget=1.0)
        self.assertLess(time.monotonic() - started, 20.0)

    def test_a_time_budget_is_accepted_by_every_public_entry_point(self) -> None:
        """A stage that refuses the clock is silently dropped from the run.

        `polybius.solve_unknown_square` raised on `time_budget` and auto_solve
        only retried on TypeError, so the stage vanished from every real run
        while passing every test -- because no test set a clock.
        """
        ciphertext = permutation.encrypt(self.plaintext, "BAEDC")
        found = permutation.solve(ciphertext, scorer=self.scorer, top=1,
                                  seed=1, time_budget=30.0)
        self.assertGreater(len(found), 0)

    def test_an_unknown_option_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            permutation.solve("HELLOTHERE", scorer=self.scorer, wibble=3)

    def test_empty_input_is_not_a_crash(self) -> None:
        self.assertEqual(len(permutation.solve("", scorer=self.scorer)), 0)

    def test_text_shorter_than_two_blocks_is_refused(self) -> None:
        found = permutation.solve("ABCDE", scorer=self.scorer, period=5)
        self.assertEqual(len(found), 0)

    def test_it_reports_what_it_searched(self) -> None:
        ciphertext = permutation.encrypt(self.plaintext, "BAEDC")
        best = permutation.solve(ciphertext, scorer=self.scorer, top=1,
                                 seed=1).best()
        self.assertIn("periods_tried", best.diagnostics)
        self.assertIn("read_order", best.diagnostics)
        self.assertIn("exhaustive", best.diagnostics["search"])


class TestPipeline(unittest.TestCase):
    """A solver nothing calls is decoration."""

    def test_it_runs_from_the_cheapest_effort_level(self) -> None:
        from cipher_tool.auto import build_stages

        names = [stage.name for stage in build_stages("fast", 5, 1)]
        self.assertIn("block permutation", names)

    def test_the_pipeline_solves_one_end_to_end(self) -> None:
        from cipher_tool.auto import auto_solve, build_stages

        plaintext = letters_only(CORPUS)[:400]
        ciphertext = permutation.encrypt(plaintext, "BAEDC")
        stage = [s for s in build_stages("fast", 5, 1)
                 if s.name == "block permutation"]
        result = auto_solve(ciphertext, effort="fast", top=3, seed=1,
                            stages=stage)
        self.assertEqual(result.candidates.best().plaintext, plaintext)

    def test_the_whole_fast_pipeline_solves_one(self) -> None:
        """Through the real ladder, not one hand-picked stage.

        The library call passing while the user's journey fails is exactly
        the gap the double columnar ceiling left, so this runs the pipeline
        the way the paste screen does.
        """
        from cipher_tool.auto import auto_solve

        plaintext = letters_only(CORPUS)[:400]
        ciphertext = permutation.encrypt(plaintext, "BAEDC")
        result = auto_solve(ciphertext, effort="fast", top=3, seed=1)
        self.assertEqual(result.candidates.best().plaintext, plaintext)


if __name__ == "__main__":
    unittest.main()
