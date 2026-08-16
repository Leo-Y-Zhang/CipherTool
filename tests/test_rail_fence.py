"""Tests for the rail fence (zigzag) transposition and its exhaustive attack.

The hand-computed cases at the top of this file were worked out on paper
before the module was written:

    rails = 3, ATTACKATDAWN

        A . . . C . . . D . . .        rail 0 = A C D
        . T . A . K . T . A . N        rail 1 = T A K T A N
        . . T . . . A . . . W .        rail 2 = T A W

    ciphertext = ACD TAKTAN TAW

    rails = 3, offset = 1, ATTACKATDAWN
    (the walk starts one step into the zigzag, on the middle rail going down)

        . . . A . . . T . . . N        rail 0 = A T N
        A . T . C . A . D . W .        rail 1 = A T C A D W
        . T . . . K . . . A . .        rail 2 = T K A

    ciphertext = ATN ATCADW TKA
"""

from __future__ import annotations

import random
import unittest

from cipher_tool.normalize import group_text, letters_only, normalize
from cipher_tool.rail_fence import (
    cycle_length,
    decrypt,
    encrypt,
    rail_counts,
    rail_sequence,
    solve,
)
from cipher_tool.scoring import DATA_DIR, default_scorer

CORPUS = (DATA_DIR / "corpus_04_expository.txt").read_text(encoding="utf-8")


class TestZigzag(unittest.TestCase):
    """The rail walk itself, which everything else is built on."""

    def test_cycle_length(self) -> None:
        self.assertEqual(cycle_length(2), 2)
        self.assertEqual(cycle_length(3), 4)
        self.assertEqual(cycle_length(7), 12)

    def test_rail_sequence_bounces(self) -> None:
        self.assertEqual(rail_sequence(6, 3), [0, 1, 2, 1, 0, 1])
        self.assertEqual(rail_sequence(5, 2), [0, 1, 0, 1, 0])
        self.assertEqual(rail_sequence(9, 4), [0, 1, 2, 3, 2, 1, 0, 1, 2])

    def test_rail_sequence_offset_starts_part_way_down(self) -> None:
        self.assertEqual(rail_sequence(6, 3, 1), [1, 2, 1, 0, 1, 2])
        # An offset of a whole cycle is the same walk as no offset at all.
        self.assertEqual(rail_sequence(12, 3, 4), rail_sequence(12, 3, 0))
        self.assertEqual(rail_sequence(12, 5, 8), rail_sequence(12, 5, 0))

    def test_rail_counts_are_not_uniform(self) -> None:
        # Middle rails are visited twice per cycle, outer rails once.
        self.assertEqual(rail_counts(12, 3), [3, 6, 3])
        self.assertEqual(sum(rail_counts(37, 6)), 37)


class TestKnownPairs(unittest.TestCase):
    """Hand-computed plaintext/key/ciphertext triples."""

    def test_known_pair(self) -> None:
        self.assertEqual(encrypt("ATTACKATDAWN", 3), "ACDTAKTANTAW")

    def test_known_pair_with_offset(self) -> None:
        self.assertEqual(encrypt("ATTACKATDAWN", 3, 1), "ATNATCADWTKA")

    def test_offset_really_changes_the_cipher(self) -> None:
        self.assertNotEqual(
            encrypt("ATTACKATDAWN", 3, 0), encrypt("ATTACKATDAWN", 3, 1)
        )

    def test_known_pair_four_rails(self) -> None:
        # WEAREDISCOVEREDFLEEATONCE on four rails (cycle 6), on paper:
        #   rail 0 : W . . . . . I . . . . . R . . . . . E . . . . . E
        #   rail 1 : . E . . . D . S . . . E . E . . . E . A . . . C .
        #   rail 2 : . . A . E . . . C . V . . . D . L . . . T . N . .
        #   rail 3 : . . . R . . . . . O . . . . . F . . . . . O . . .
        self.assertEqual(
            encrypt("WEAREDISCOVEREDFLEEATONCE", 4),
            "WIREE" + "EDSEEEAC" + "AECVDLTN" + "ROFO",
        )

    def test_decrypt_reverses_the_known_pair(self) -> None:
        self.assertEqual(decrypt("ACDTAKTANTAW", 3), "ATTACKATDAWN")
        self.assertEqual(decrypt("ATNATCADWTKA", 3, 1), "ATTACKATDAWN")


class TestRoundTrip(unittest.TestCase):
    def test_round_trip_many_keys(self) -> None:
        plaintext = letters_only(CORPUS)[:137]  # deliberately an odd length
        for rails in (2, 3, 4, 5, 7, 11, 20):
            for offset in range(cycle_length(rails)):
                with self.subTest(rails=rails, offset=offset):
                    self.assertEqual(
                        decrypt(encrypt(plaintext, rails, offset), rails, offset),
                        plaintext,
                    )

    def test_encryption_is_a_permutation_of_the_letters(self) -> None:
        plaintext = letters_only(CORPUS)[:200]
        ciphertext = encrypt(plaintext, 6, 2)
        self.assertEqual(sorted(ciphertext), sorted(plaintext))


class TestInputRobustness(unittest.TestCase):
    def test_layout_does_not_change_the_result(self) -> None:
        clean = encrypt("ATTACKATDAWN", 4)
        for variant in (
            "attackatdawn",
            "Attack at dawn!",
            "ATTAC KATDA WN",
            "ATTAC\nKATDA\nWN",
            "  A t t a c k , a t   d a w n .  ",
            "attack at dawn (1805)",
        ):
            with self.subTest(variant=variant):
                self.assertEqual(encrypt(variant, 4), clean)

    def test_decrypt_ignores_layout_too(self) -> None:
        self.assertEqual(decrypt("acdta ktant aw", 3), "ATTACKATDAWN")


class TestEmptyInput(unittest.TestCase):
    def test_empty_encrypt_and_decrypt(self) -> None:
        self.assertEqual(encrypt("", 4), "")
        self.assertEqual(decrypt("", 4), "")
        self.assertEqual(encrypt("...!...", 4), "")

    def test_empty_solve(self) -> None:
        self.assertEqual(len(solve("", scorer=default_scorer())), 0)
        self.assertEqual(len(solve("AB", scorer=default_scorer())), 0)


class TestInvalidKeys(unittest.TestCase):
    def _assert_explains(self, call, *args, **kwargs) -> str:
        with self.assertRaises(ValueError) as caught:
            call(*args, **kwargs)
        message = str(caught.exception)
        self.assertTrue(message.strip(), "ValueError must carry an explanation")
        return message

    def test_too_few_rails(self) -> None:
        for rails in (1, 0, -3):
            with self.subTest(rails=rails):
                message = self._assert_explains(encrypt, "ATTACKATDAWN", rails)
                self.assertIn("at least 2", message)

    def test_rails_not_an_integer(self) -> None:
        message = self._assert_explains(encrypt, "ATTACKATDAWN", "3")
        self.assertIn("integer", message)
        self._assert_explains(encrypt, "ATTACKATDAWN", 3.0)

    def test_rails_not_fewer_than_the_letters(self) -> None:
        # 12 letters cannot be written on 12 rails: nothing would move.
        message = self._assert_explains(encrypt, "ATTACKATDAWN", 12)
        self.assertIn("fewer than", message)
        self._assert_explains(decrypt, "ATTACKATDAWN", 40)

    def test_negative_offset(self) -> None:
        message = self._assert_explains(encrypt, "ATTACKATDAWN", 3, -1)
        self.assertIn("negative", message)

    def test_none_options_mean_not_supplied(self) -> None:
        # A command line hands us None for arguments the user left out.
        result = solve(
            encrypt(letters_only(CORPUS)[:120], 5),
            scorer=default_scorer(),
            top=2,
            max_rails=None,
            time_budget=None,
            seed=None,
        )
        self.assertTrue(result)

    def test_unknown_solver_option(self) -> None:
        message = self._assert_explains(
            solve, "ATTACKATDAWN", scorer=default_scorer(), rail_count=3
        )
        self.assertIn("rail_count", message)


class TestSolver(unittest.TestCase):
    """The exhaustive attack, on text long enough to be honestly solvable."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = default_scorer()
        cls.plaintext = letters_only(CORPUS)[:320]
        assert len(cls.plaintext) >= 300

    def test_recovers_plaintext_and_key(self) -> None:
        ciphertext = encrypt(self.plaintext, 7, 3)
        result = solve(ciphertext, scorer=self.scorer, top=5)
        best = result.best()
        self.assertEqual(best.plaintext, self.plaintext)
        self.assertEqual(best.key, "rails=7 offset=3")
        self.assertEqual(best.confidence(), "strong")

    def test_recovers_several_configurations(self) -> None:
        for rails, offset in ((2, 0), (4, 0), (9, 5), (13, 2), (20, 11)):
            with self.subTest(rails=rails, offset=offset):
                ciphertext = encrypt(self.plaintext, rails, offset)
                best = solve(ciphertext, scorer=self.scorer, top=3).best()
                self.assertEqual(best.plaintext, self.plaintext)

    def test_reports_the_search_it_actually_ran(self) -> None:
        result = solve(
            encrypt(self.plaintext, 5), scorer=self.scorer, top=3, max_rails=8
        )
        evidence = result.best().diagnostics
        self.assertEqual(evidence["rails_tested"], "2-8")
        # sum over rails 2..8 of (2*rails - 2) offsets
        self.assertEqual(evidence["configurations_tested"], 56)
        self.assertIn("rail_lengths", evidence)

    def test_accepts_a_normalized_text(self) -> None:
        # The command line normalises before calling us, so both entry points
        # must behave identically.
        ciphertext = encrypt(self.plaintext, 7, 3)
        grouped = normalize(group_text(ciphertext))
        from_text = solve(ciphertext, scorer=self.scorer, top=3)
        from_object = solve(grouped, scorer=self.scorer, top=3)
        self.assertEqual(from_object.best().plaintext, self.plaintext)
        self.assertEqual(
            [candidate.key for candidate in from_text],
            [candidate.key for candidate in from_object],
        )

    def test_top_limits_the_returned_set(self) -> None:
        result = solve(encrypt(self.plaintext, 6), scorer=self.scorer, top=2)
        self.assertEqual(len(result), 2)

    def test_display_is_not_the_original_layout(self) -> None:
        # A transposition moves letters, so the original spacing tells us
        # nothing about the plaintext; display must be plain five-letter
        # groups rather than a relayout that would imply otherwise.
        result = solve(
            encrypt(self.plaintext, 5), scorer=self.scorer, top=1
        )
        best = result.best()
        self.assertIsNotNone(best.display)
        self.assertEqual(letters_only(best.display), best.plaintext)
        self.assertIn(" ", best.display)


class TestFailureModes(unittest.TestCase):
    """What the tool does when the text is NOT a rail fence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = default_scorer()

    def test_random_letters_are_not_reported_as_solved(self) -> None:
        rng = random.Random(7)
        noise = "".join(
            rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(300)
        )
        best = solve(noise, scorer=self.scorer, top=3).best()
        # There is always a best-scoring rail count; the point is that the
        # tool must not dress it up as an answer.
        self.assertIsNotNone(best)
        self.assertIn(best.confidence(), ("weak", "unlikely"))
        self.assertLess(best.diagnostics["word_coverage"], 0.5)
        self.assertLess(best.diagnostics["normalised_score"], -2.0)

    def test_wrong_rail_count_gives_junk_not_plaintext(self) -> None:
        plaintext = letters_only(CORPUS)[:300]
        ciphertext = encrypt(plaintext, 7, 0)
        wrong = decrypt(ciphertext, 6, 0)
        self.assertNotEqual(wrong, plaintext)
        self.assertLess(
            self.scorer.normalised(wrong), self.scorer.normalised(plaintext)
        )

    def test_wrong_offset_gives_junk_not_plaintext(self) -> None:
        plaintext = letters_only(CORPUS)[:300]
        ciphertext = encrypt(plaintext, 7, 3)
        self.assertNotEqual(decrypt(ciphertext, 7, 4), plaintext)

    def test_english_plaintext_is_returned_untouched_by_two_rails(self) -> None:
        # Sanity check on the scorer's ability to separate: the true
        # decryption must outscore every other configuration.
        plaintext = letters_only(CORPUS)[:300]
        result = solve(encrypt(plaintext, 3, 0), scorer=self.scorer, top=5)
        ranked = result.ranked()
        self.assertEqual(ranked[0].plaintext, plaintext)
        self.assertGreater(ranked[0].score, ranked[1].score)


if __name__ == "__main__":
    unittest.main()
