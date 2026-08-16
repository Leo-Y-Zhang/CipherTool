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

import unittest
from pathlib import Path

from cipher_tool.bifid import decrypt, encrypt, solve
from cipher_tool.normalize import group_text, letters_only, normalize
from cipher_tool.polybius import PolybiusSquare

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


if __name__ == "__main__":
    unittest.main()
