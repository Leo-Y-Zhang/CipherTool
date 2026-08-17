"""Tests for the Bifid cipher.

The hand-computed cases use the standard square::

        1  2  3  4  5
     1  A  B  C  D  E
     2  F  G  H  I  K
     3  L  M  N  O  P
     4  Q  R  S  T  U
     5  V  W  X  Y  Z

ATTACK, fractionated as a whole:

    plaintext   A  T  T  A  C  K
    row         1  4  4  1  1  2
    column      1  4  4  1  3  5
    stream      1 4 4 1 1 2 | 1 4 4 1 3 5
    pairs       (1,4)(4,1)(1,2)(1,4)(4,1)(3,5)  ->  D Q B D Q P

ATTACKAT with period 3 splits into ATT, ACK, AT:

    ATT -> rows 1 4 4, columns 1 4 4 -> (1,4)(4,1)(4,4) -> D Q T
    ACK -> rows 1 1 2, columns 1 3 5 -> (1,1)(2,1)(3,5) -> A F P
    AT  -> rows 1 4,   columns 1 4   -> (1,4)(1,4)      -> D D
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path

from cipher_tool.bifid import decrypt, encrypt, solve, solve_unknown_square
from cipher_tool.normalize import group_text, letters_only, normalize
from cipher_tool.polybius import PolybiusSquare
from cipher_tool.scoring import default_scorer

CORPUS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "cipher_tool"
    / "data"
    / "corpus_04_expository.txt"
)


def sample_text(count: int = 400) -> str:
    """A run of real English long enough for statistical attacks to work."""
    return letters_only(CORPUS.read_text(encoding="utf-8"))[:count]


class TestKnownAnswers(unittest.TestCase):
    def test_whole_message_fractionation_by_hand(self) -> None:
        self.assertEqual(encrypt("ATTACK"), "DQBDQP")
        self.assertEqual(decrypt("DQBDQP"), "ATTACK")

    def test_period_three_with_a_ragged_final_block_by_hand(self) -> None:
        self.assertEqual(encrypt("ATTACKAT", None, 3), "DQTAFPDD")
        self.assertEqual(decrypt("DQTAFPDD", None, 3), "ATTACKAT")

    def test_the_ragged_tail_is_enciphered_like_a_block_of_its_own_length(
        self,
    ) -> None:
        # The two-letter tail AT gives DD, exactly as a lone AT does: block
        # arithmetic depends only on the length of the block in hand.
        self.assertEqual(encrypt("ATTACKAT", None, 3)[-2:], "DD")
        self.assertEqual(encrypt("AT", None, 3), "DD")

    def test_period_one_is_the_identity(self) -> None:
        # A one-letter block has stream [row, column], which re-pairs to the
        # same cell. Worth knowing: the solver reporting period 1 means "this
        # text was never fractionated".
        self.assertEqual(encrypt("ATTACKATDAWN", None, 1), "ATTACKATDAWN")
        self.assertEqual(decrypt("ATTACKATDAWN", None, 1), "ATTACKATDAWN")

    def test_a_period_longer_than_the_message_is_the_whole_message(self) -> None:
        self.assertEqual(encrypt("ATTACK", None, 500), encrypt("ATTACK", None, 0))
        self.assertEqual(encrypt("ATTACK", None, 0), encrypt("ATTACK", None, None))


class TestRoundTrip(unittest.TestCase):
    def test_round_trip_over_periods_and_squares(self) -> None:
        message = "MEETMEATTHEHARBOURATMIDNIGHTANDBRINGTHECHARTS"
        squares = [
            None,
            "MONARCHY",
            PolybiusSquare.standard(),
            PolybiusSquare.standard("SPHINXOFBLACKQUARTZ"),
            PolybiusSquare.without_q(),
            PolybiusSquare.six_by_six("CIPHER"),
            PolybiusSquare.adfgx("HARBOUR"),
        ]
        for square in squares:
            for period in (None, 0, 1, 2, 3, 5, 7, 13, 44, 45, 46):
                with self.subTest(square=square, period=period):
                    board = square if isinstance(square, PolybiusSquare) else None
                    expected = (
                        board.prepare(message)
                        if board is not None
                        else PolybiusSquare.standard(
                            square if isinstance(square, str) else None
                        ).prepare(message)
                    )
                    cipher = encrypt(message, square, period)
                    self.assertEqual(len(cipher), len(expected))
                    self.assertEqual(decrypt(cipher, square, period), expected)

    def test_round_trip_on_the_full_sample_with_a_ragged_tail(self) -> None:
        text = sample_text(403)  # 403 = 7 * 57 + 4, so the last block is short
        square = PolybiusSquare.standard("HARBOUR")
        cipher = encrypt(text, square, 7)
        self.assertEqual(len(cipher), len(square.prepare(text)))
        self.assertEqual(decrypt(cipher, square, 7), square.prepare(text))

    def test_layout_of_the_input_does_not_matter(self) -> None:
        clean = encrypt("ATTACKATDAWN", None, 5)
        self.assertEqual(encrypt("attack at dawn", None, 5), clean)
        self.assertEqual(encrypt("Attack, at dawn!", None, 5), clean)
        self.assertEqual(encrypt("ATTAC KATDA WN", None, 5), clean)
        self.assertEqual(encrypt("ATTACKAT\nDAWN\n", None, 5), clean)
        self.assertEqual(decrypt(group_text(clean), None, 5), "ATTACKATDAWN")

    def test_empty_input_is_not_an_error(self) -> None:
        self.assertEqual(encrypt(""), "")
        self.assertEqual(decrypt(""), "")
        self.assertEqual(encrypt("  ,,, "), "")
        self.assertEqual(encrypt("", None, 5), "")
        self.assertEqual(len(solve("")), 0)


class TestInvalidInput(unittest.TestCase):
    def assertRaisesWithMessage(self, callable_object) -> None:
        """Every refusal must explain itself, not just fail."""
        with self.assertRaises(ValueError) as caught:
            callable_object()
        self.assertTrue(str(caught.exception).strip())

    def test_negative_period_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            encrypt("ATTACK", None, -3)
        self.assertIn("negative", str(caught.exception))

    def test_non_integer_period_is_rejected(self) -> None:
        self.assertRaisesWithMessage(lambda: encrypt("ATTACK", None, "5"))
        self.assertRaisesWithMessage(lambda: encrypt("ATTACK", None, 2.5))
        self.assertRaisesWithMessage(lambda: decrypt("DQBDQP", None, "5"))

    def test_bad_square_argument_is_rejected(self) -> None:
        self.assertRaisesWithMessage(lambda: encrypt("ATTACK", 25))

    def test_unusable_keyword_is_rejected(self) -> None:
        self.assertRaisesWithMessage(lambda: encrypt("ATTACK", "1234"))
        self.assertRaisesWithMessage(lambda: solve("ATTACK", keywords=["----"]))

    def test_max_period_must_be_positive(self) -> None:
        self.assertRaisesWithMessage(lambda: solve("ATTACK", max_period=0))
        self.assertRaisesWithMessage(lambda: solve("ATTACK", max_period=-1))
        self.assertRaisesWithMessage(lambda: solve("ATTACK", max_period="7"))

    def test_letter_the_square_cannot_hold_is_refused(self) -> None:
        # A drop-Q square has nowhere to put the Q of QUARTZ.
        square = PolybiusSquare.without_q()
        with self.assertRaises(ValueError) as caught:
            encrypt("QUARTZ", square, 5)
        self.assertIn("Q", str(caught.exception))


class TestSolve(unittest.TestCase):
    def test_recovers_the_period_and_reads_the_plaintext(self) -> None:
        square = PolybiusSquare.standard()
        expected = square.prepare(sample_text())
        result = solve(encrypt(expected, square, 7))
        best = result.best()
        self.assertIsNotNone(best)
        self.assertEqual(best.plaintext, expected)
        self.assertEqual(best.diagnostics["period"], "7")
        self.assertIn("period=7", best.key)
        self.assertEqual(best.confidence(), "strong")

    def test_recovers_whole_message_fractionation(self) -> None:
        square = PolybiusSquare.standard()
        expected = square.prepare(sample_text())
        best = solve(encrypt(expected, square, 0)).best()
        self.assertEqual(best.plaintext, expected)
        self.assertEqual(best.diagnostics["period"], "whole message")

    def test_recovers_a_keyed_square_when_the_keyword_is_offered(self) -> None:
        square = PolybiusSquare.standard("SPHINX")
        expected = square.prepare(sample_text())
        result = solve(
            encrypt(expected, square, 9), keywords=["HARBOUR", "SPHINX"]
        )
        best = result.best()
        self.assertEqual(best.plaintext, expected)
        self.assertIn("SPHINX", best.key)
        self.assertEqual(best.confidence(), "strong")

    def test_reports_every_period_it_tried(self) -> None:
        square = PolybiusSquare.standard()
        cipher = encrypt(square.prepare(sample_text()), square, 4)
        best = solve(cipher, max_period=6).best()
        self.assertEqual(
            best.diagnostics["periods_tested"], "whole message, 1, 2, 3, 4, 5, 6"
        )
        self.assertGreaterEqual(best.diagnostics["squares_tested"], 1)

    def test_display_preserves_the_layout_of_the_ciphertext(self) -> None:
        square = PolybiusSquare.standard()
        expected = square.prepare(sample_text(200))
        source = group_text(encrypt(expected, square, 5))
        best = solve(source).best()
        self.assertIsNotNone(best.display)
        self.assertEqual(letters_only(best.display), expected)
        self.assertEqual(len(best.display), len(source))

    def test_accepts_a_normalized_text(self) -> None:
        square = PolybiusSquare.standard()
        expected = square.prepare(sample_text())
        source = normalize(group_text(encrypt(expected, square, 6)))
        self.assertEqual(solve(source).best().plaintext, expected)

    def test_top_limits_the_returned_set(self) -> None:
        cipher = encrypt(sample_text(), None, 5)
        self.assertLessEqual(len(solve(cipher, top=3)), 3)
        self.assertGreater(len(solve(cipher, top=0)), 3)

    def test_time_budget_is_honoured(self) -> None:
        cipher = encrypt(sample_text(), None, 5)
        result = solve(cipher, time_budget=0.0)
        self.assertGreater(len(result), 0)
        for candidate in result:
            self.assertTrue(candidate.diagnostics.get("time_budget_hit"))


class TestFailureModes(unittest.TestCase):
    """What the tool does when the answer is not there to be found."""

    def test_the_wrong_period_does_not_give_the_plaintext(self) -> None:
        square = PolybiusSquare.standard()
        expected = square.prepare(sample_text(120))
        cipher = encrypt(expected, square, 7)
        self.assertNotEqual(decrypt(cipher, square, 6), expected)
        self.assertNotEqual(decrypt(cipher, square, 8), expected)

    def test_keyed_square_without_its_keyword_is_not_reported_as_solved(self) -> None:
        square = PolybiusSquare.standard("SPHINX")
        truth = square.prepare(sample_text())
        best = solve(encrypt(truth, square, 9)).best()
        self.assertIsNotNone(best)
        self.assertNotEqual(best.plaintext, truth)
        self.assertNotEqual(best.confidence(), "strong")
        self.assertIn(best.confidence(), ("weak", "unlikely"))

    def test_a_square_that_cannot_hold_the_ciphertext_is_ruled_out_loudly(
        self,
    ) -> None:
        # A ciphertext containing Q cannot have come from a drop-Q square, so
        # that square is discarded and the reason recorded.
        cipher = encrypt("ATTACKATDAWNTOMORROW", None, 5)
        self.assertIn("Q", cipher)
        best = solve(cipher).best()
        self.assertIn("squares_ruled_out", best.diagnostics)
        self.assertIn("Q dropped", str(best.diagnostics["squares_ruled_out"]))
        self.assertEqual(best.diagnostics["squares_tested"], 1)

        # ...and a ciphertext it could have produced keeps it in the running.
        innocent = encrypt("MEETMEATTHEHARBOUR", None, 5)
        self.assertNotIn("Q", innocent)
        other = solve(innocent).best()
        self.assertNotIn("squares_ruled_out", other.diagnostics)
        self.assertEqual(other.diagnostics["squares_tested"], 2)

    def test_plain_english_is_reported_as_period_one_not_as_a_break(self) -> None:
        # Running Bifid on text that was never enciphered must not manufacture
        # a key: period 1 is the identity, and the diagnostics say so.
        text = sample_text(300)
        best = solve(text).best()
        self.assertEqual(best.plaintext, text)
        self.assertEqual(best.diagnostics["period"], "1")
        self.assertIn("identity", str(best.diagnostics["note"]))


class TestUnknownSquare(unittest.TestCase):
    """Recovering a KEYED square nobody supplied.

    `solve` tries the squares it is handed and no others, so a keyed message
    with no keyword available was out of reach. Measured, the pipeline gave a
    `weak` reading of a Bifid message and the README said as much -- honest,
    and still a hole, because a competition does not hand over the keyword.

    The square is hill-climbed instead, exactly as Playfair's already is.
    MEASURED: 500 letters at period 7 recovered in a few seconds.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = default_scorer()
        # J folds into I in a 5x5 square, so the plaintext must not contain
        # one if the round trip is to be character-exact.
        cls.plaintext = sample_text(500).replace("J", "I")
        cls.square = PolybiusSquare.standard("TEMPEST")
        cls.ciphertext = encrypt(cls.plaintext, cls.square, 7)

    def test_it_recovers_a_keyed_square_with_the_period_known(self) -> None:
        found = solve_unknown_square(
            self.ciphertext, scorer=self.scorer, top=1, period=7, seed=1,
        )
        self.assertEqual(found.best().plaintext, self.plaintext)

    def test_it_finds_the_period_when_not_told(self) -> None:
        # Full strength deliberately. Cutting this to two restarts of 6,000
        # steps to save half a minute made it fail: identifying the period
        # is only half the job, and the winning period still has to be
        # climbed properly before its plaintext is right.
        found = solve_unknown_square(
            self.ciphertext, scorer=self.scorer, top=1, seed=1, max_period=9,
        )
        best = found.best()
        self.assertEqual(best.plaintext, self.plaintext)
        self.assertEqual(best.diagnostics["period"], 7)

    def test_it_reports_the_square_so_the_answer_can_be_checked(self) -> None:
        # Cheap on purpose: the invariant holds whether or not the search
        # succeeded, so it does not need a full-strength climb to prove.
        best = solve_unknown_square(
            self.ciphertext, scorer=self.scorer, top=1, period=7, seed=1,
            restarts=1, iterations=1200,
        ).best()
        recovered = PolybiusSquare(best.diagnostics["square"])
        self.assertEqual(
            decrypt(self.ciphertext, recovered, 7), best.plaintext,
            "the square it prints must reproduce the answer it prints",
        )

    def test_it_never_claims_the_search_was_exhaustive(self) -> None:
        best = solve_unknown_square(
            self.ciphertext, scorer=self.scorer, top=1, period=7, seed=1,
            restarts=1, iterations=1200,
        ).best()
        self.assertIn("not exhaustive", best.diagnostics["search"])

    def test_the_seed_makes_a_run_reproducible(self) -> None:
        options = dict(scorer=self.scorer, top=1, period=7, seed=9,
                       restarts=1, iterations=600)
        first = solve_unknown_square(self.ciphertext, **options).best()
        second = solve_unknown_square(self.ciphertext, **options).best()
        self.assertEqual(first.plaintext, second.plaintext)

    def test_random_letters_are_not_reported_as_solved(self) -> None:
        import random

        generator = random.Random(3)
        noise = "".join(generator.choice("ABCDEFGHIKLMNOPQRSTUVWXYZ")
                        for _ in range(400))
        best = solve_unknown_square(
            noise, scorer=self.scorer, top=1, period=7, seed=1,
            restarts=1, iterations=800,
        ).best()
        if best is not None:
            self.assertNotEqual(best.confidence(), "strong")

    def test_a_time_budget_is_honoured(self) -> None:
        started = time.monotonic()
        solve_unknown_square(
            self.ciphertext, scorer=self.scorer, top=1, seed=1,
            max_period=12, time_budget=1.0,
        )
        self.assertLess(time.monotonic() - started, 25.0)

    def test_empty_input_is_not_a_crash(self) -> None:
        self.assertEqual(len(solve_unknown_square("", scorer=self.scorer)), 0)

    def test_the_pipeline_reaches_it_at_deep(self) -> None:
        """A solver nothing calls is decoration."""
        from cipher_tool.auto import build_stages

        names = [stage.name for stage in build_stages("deep", 5, 1)]
        self.assertIn("Bifid (unknown square)", names)

    def test_the_pipeline_runs_it_hard_enough_to_succeed(self) -> None:
        """A stage that cannot reach the answer is worse than no stage.

        MEASURED on a 400-letter keyed message at period 7: three restarts of
        6,000 steps failed inside a 45 second budget and reported `weak`,
        while four of 8,000 solved it at `strong` in 42.7 seconds. The first
        version of this stage used the weaker numbers and looked like
        coverage while providing none.
        """
        from cipher_tool.auto import build_stages
        from cipher_tool.bifid import (DEFAULT_CLIMB_ITERATIONS,
                                       DEFAULT_CLIMB_RESTARTS)

        stage = next(s for s in build_stages("deep", 5, 1)
                     if s.name == "Bifid (unknown square)")
        self.assertGreaterEqual(stage.options["restarts"],
                                DEFAULT_CLIMB_RESTARTS)
        self.assertGreaterEqual(stage.options["iterations"],
                                DEFAULT_CLIMB_ITERATIONS)
        self.assertGreaterEqual(stage.options["time_budget"], 60.0)


if __name__ == "__main__":
    unittest.main()
