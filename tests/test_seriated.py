"""Tests for the split-coordinate Polybius.

The detector has to fire on one shape and refuse everything else, and the
refusals are the harder half. Three of the tests below are controls that were
written because they FAILED first:

* handed an ordinary interleaved Polybius message whose plaintext repeated
  with a period dividing the half-length, the detector paired every symbol
  with an identical one and reported an index of coincidence of 0.204 -- the
  highest number anywhere in this file, from five distinct cells, on the wrong
  cipher. A high index of coincidence is also what a collapse looks like;
* the same construction on a SHUFFLE of the same symbols is the null the
  finding has to beat, because the raw statistic partly measures how many
  cells there are rather than what order they are in;
* and the ordinary interleaved form of a real message must be refused, since
  the toolkit already has a solver for that and two readings of one message is
  not an improvement.
"""

from __future__ import annotations

import random
import unittest

from cipher_tool import seriated

SQUARE = "ZEROPQSTUVWXYABCDFGHIKLMN"

PLAIN = (
    "MYDEARCHARLESIGUESSYOUHAVEBYNOWWORKEDOUTTHECHALLENGESETBYKAISERANDSOLVED"
    "ALREADYBYOURDEARMRSLOVELACEIWASKICKINGMYSELFWHENIFINALLYSAWITSPECIALLY"
    "SINCEIWASTHEONEWHOINSPECTEDTHECARGOINLIVERPOOLTHEBOXESWEREPACKEDEDGETO"
    "EDGEINTHECRATESWITHONECRATEEMPTIEDOFALLBUTTWOHUNDREDROUNDSANDTHEREST"
    "FILLEDWITHTHECONTRABANDWEHAVEBEENSEEKINGFORSOMANYWEEKSNOWMYAPOLOGIES"
) * 4


def encrypt(text: str = PLAIN) -> str:
    return seriated.encipher(text, SQUARE)


class TestRoundTrip(unittest.TestCase):
    """The cipher itself, before any cryptanalysis."""

    def test_encipher_then_decipher_returns_the_message(self) -> None:
        kept = "".join(c for c in PLAIN if c in SQUARE)
        self.assertEqual(seriated.decipher(encrypt(), SQUARE), kept)

    def test_the_two_halves_are_the_two_coordinates(self) -> None:
        """Not an implementation detail -- it is why the split is at the middle."""
        cipher = encrypt()
        half = len(cipher) // 2
        self.assertEqual(len(cipher) % 2, 0)
        self.assertEqual(len(set(cipher[:half])), 5)
        self.assertEqual(len(set(cipher[half:])), 5)

    def test_a_letter_outside_the_square_is_dropped_not_invented(self) -> None:
        self.assertEqual(len(seriated.encipher("JJJ" + PLAIN[:10], SQUARE)),
                         20)


class TestDetector(unittest.TestCase):
    """Recognition, which happens before any key is known."""

    def test_it_finds_the_split(self) -> None:
        found = seriated.detect(encrypt())
        self.assertIsNotNone(found)
        self.assertEqual(found.split, len(encrypt()) // 2)
        self.assertGreaterEqual(found.index_of_coincidence,
                                seriated.MINIMUM_IC)
        self.assertGreaterEqual(found.gap, seriated.MINIMUM_GAP)

    def test_a_shuffle_of_the_same_symbols_is_refused(self) -> None:
        """The null control. The statistic must be about ORDER, not inventory."""
        symbols = list(encrypt())
        random.Random(1).shuffle(symbols)
        self.assertIsNone(seriated.detect("".join(symbols)))

    def test_the_ordinary_interleaved_form_is_refused(self) -> None:
        """The same message written the usual way belongs to `polybius`.

        Two readings of one message is not an improvement, and this detector
        firing on an ordinary Polybius would put a second, worse answer beside
        a solver that already gets it right.
        """
        cipher = encrypt()
        half = len(cipher) // 2
        interleaved = "".join(a + b for a, b in zip(cipher[:half],
                                                    cipher[half:]))
        self.assertIsNone(seriated.detect(interleaved))

    def test_a_collapsed_pairing_is_refused_however_well_it_scores(self) -> None:
        """FOUND BY A CONTROL, and it scored higher than any real finding.

        A stream that repeats with a period dividing its half-length pairs
        every symbol with an identical one. That reads as an index of
        coincidence of 0.204 over five distinct cells -- the strongest signal
        in this file, on nothing at all.
        """
        collapsed = "12345" * 400
        found = seriated.detect(collapsed)
        self.assertIsNone(found)

    def test_a_letter_ciphertext_is_refused_without_work(self) -> None:
        self.assertIsNone(seriated.detect("ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 100))

    def test_too_short_a_message_is_refused(self) -> None:
        self.assertIsNone(seriated.detect(encrypt()[:400]))


class TestSolve(unittest.TestCase):
    """End to end, and the ways it must decline."""

    def test_it_reads_the_message(self) -> None:
        found = seriated.solve(encrypt(), seed=1)
        best = found.best()
        self.assertIsNotNone(best)
        self.assertEqual(best.method, seriated.METHOD)
        self.assertTrue(best.plaintext.startswith("MYDEARCHARLESIGUESS"))
        self.assertEqual(best.confidence(), "strong")

    def test_the_split_is_reported_so_the_answer_can_be_checked(self) -> None:
        best = seriated.solve(encrypt(), seed=1).best()
        self.assertIn(f"start at symbol {len(encrypt()) // 2}", best.key)
        self.assertIn("square=", best.key)

    def test_it_returns_nothing_on_anything_else(self) -> None:
        self.assertEqual(seriated.solve("ABCDEFGHIJ" * 200).ranked(), [])

    def test_it_accepts_a_time_budget_without_raising(self) -> None:
        """A stage that RAISES on ``time_budget`` is silently dropped."""
        self.assertEqual(seriated.solve("12345" * 20, time_budget=1.0).ranked(),
                         [])


if __name__ == "__main__":
    unittest.main()
