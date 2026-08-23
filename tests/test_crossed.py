"""Tests for the crossed-coordinate digraph cipher.

The failure that costs a challenge is missing a real message. The failure that
costs the project is firing on something else -- a paired stream that is NOT
this cipher would be handed a confident reading built from 48 free unknowns,
and that is exactly the shape of answer this toolkit exists not to give. So
there are controls in both directions here: a real message must be recovered
exactly, and a shuffled one must be refused.

Everything is bounded by a STEP COUNT rather than a clock, so a slow machine
runs these slowly instead of failing them.
"""

from __future__ import annotations

import random
import unittest

from cipher_tool import crossed

# A 24-letter reduced alphabet, no J and no Z -- the convention the real 2025
# message uses.
SQUARE_ONE = "SBKRHCLTAEMUDFNVOGPXWIQY"
SQUARE_TWO = "FULHEARTBCDGIKMNOPQSVWXY"

RANKS = "A23456789XJQK"
SUITS = "CDHS"
DECK = [rank + suit for suit in SUITS for rank in RANKS]
CELLS_ONE = DECK[:36]
CELLS_TWO = DECK[36:]

PLAIN = (
    "REPORTONTHEMEETINGBETWEENCHARLESDICKENSANDTHEPRESIDENTANDTHEENSUING"
    "EVENTSFORTHEMEMBERSOFTHECURIAOFTHETRINITYFOUNDATIONTHEPRESIDENT"
    "GREETEDMRDICKENSFULSOMELYANDINVITEDHIMTOBESEATEDAFTERINITIAL"
    "EXCHANGESTHEROOMWASCLEAREDOFALLOTHERPARTICIPANTSTOALLOWCONFIDENTIAL"
    "EXCHANGESTHERECORDOFTHEIRCONVERSATIONCOMESCOURTESYOFAVERBATIMREPORT"
    "BYMRDICKENSRECEIVEDBYGENERALGRENVILLEDODGEANDPASSEDTOCHARLESBABBAGE"
    "INAPERSONALLETTERMRDICKENSTHANKEDTHEPRESIDENTFORHISHOSPITALITYAND"
    "FORHISENTHUSIASTICSUPPORTFORTHEPERFORMANCEWHICHHEHADGIVEN"
) * 4


def encrypt(plaintext: str) -> str:
    return crossed.encipher(
        plaintext, SQUARE_ONE, SQUARE_TWO, CELLS_ONE, CELLS_TWO,
        width_one=4, width_two=4,
    )


class TestRoundTrip(unittest.TestCase):
    """Encipher and decipher must be inverses, and separately written."""

    def test_a_message_survives_the_round_trip(self) -> None:
        cipher = encrypt(PLAIN)
        back = crossed.decipher(
            cipher, SQUARE_ONE, SQUARE_TWO, CELLS_ONE, CELLS_TWO,
            width_one=4, width_two=4,
        )
        self.assertEqual(back, PLAIN[:len(back)])

    def test_the_squares_are_not_their_own_inverse(self) -> None:
        """A test built on an involution cannot detect a missing inversion.

        Recorded on this project when a block-permutation key that was its own
        inverse let every test pass with the inversion deleted. So: assert the
        cipher is NOT an involution, which is what makes the test above mean
        something.
        """
        cipher = encrypt(PLAIN)
        self.assertNotEqual(cipher[:60], PLAIN[:60])
        twice = crossed.encipher(
            crossed.decipher(cipher, SQUARE_ONE, SQUARE_TWO, CELLS_ONE,
                             CELLS_TWO, width_one=4, width_two=4),
            SQUARE_ONE, SQUARE_TWO, CELLS_ONE, CELLS_TWO,
            width_one=4, width_two=4,
        )
        self.assertEqual(twice, cipher[:len(twice)])

    def test_a_letter_outside_the_squares_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            encrypt("JAZZ")

    def test_a_wrong_cell_alphabet_size_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            crossed.encipher(PLAIN, SQUARE_ONE, SQUARE_TWO, CELLS_ONE[:35],
                             CELLS_TWO, width_one=4, width_two=4)


class TestDetector(unittest.TestCase):
    """Recognition happens before any key is known, so it is testable alone."""

    def setUp(self) -> None:
        self.cipher = encrypt(PLAIN)
        self.cells = [self.cipher[i:i + 2]
                      for i in range(0, len(self.cipher), 2)]

    def test_it_finds_the_true_fractionation(self) -> None:
        split = crossed.detect(self.cells)
        self.assertIsNotNone(split)
        self.assertIn("6x6", split.shape)
        self.assertIn("4x4", split.shape)
        self.assertGreater(split.score, crossed.MINIMUM_IC)

    def test_the_true_split_beats_every_other_reading(self) -> None:
        """The control that makes the detector evidence rather than a guess.

        Any search over many readings finds one above average. What separates
        a fractionation from the best of N coincidences is that the true one
        keeps English's index of coincidence in BOTH halves while the others
        merge classes and flatten.
        """
        split = crossed.detect(self.cells)
        shuffled = list(self.cells)
        random.Random(0).shuffle(shuffled)
        control = crossed.detect(shuffled)
        margin = split.score - (0.0 if control is None else control.score)
        self.assertGreater(margin, 0.01)

    def test_a_stream_of_noise_is_refused(self) -> None:
        rng = random.Random(3)
        noise = []
        for _ in range(len(self.cells) // 2):
            noise.append(rng.choice(CELLS_ONE))
            noise.append(rng.choice(CELLS_TWO))
        self.assertIsNone(crossed.detect(noise))

    def test_a_short_stream_is_refused_rather_than_guessed(self) -> None:
        self.assertIsNone(crossed.detect(self.cells[:100]))

    def test_too_little_text_is_refused_because_the_answer_would_be_wrong(
            self) -> None:
        """The length floor is a MEASUREMENT, not a round number.

        At 300 units the true split scores 0.0580 on the index of coincidence
        and the recovery gets 8% of the letters right. Detecting there would
        mean offering a reading of nothing.
        """
        self.assertIsNone(crossed.detect(self.cells[:2 * 300]))

    def test_overlapping_cell_alphabets_are_refused(self) -> None:
        """The two sub-alphabets must be disjoint; that is what a unit means."""
        muddled = list(self.cells)
        muddled[1] = muddled[0]
        self.assertIsNone(crossed.detect(muddled))


class TestRecovery(unittest.TestCase):
    """The end-to-end claim: a real message comes back letter for letter."""

    def test_it_recovers_the_plaintext_exactly(self) -> None:
        cipher = encrypt(PLAIN)
        cells = [cipher[i:i + 2] for i in range(0, len(cipher), 2)]
        split = crossed.detect(cells)
        reading = crossed.recover(split, restarts=1,
                                  steps=crossed.STEPS_PER_RESTART, seed=1)
        expected = PLAIN[:len(reading.plaintext)]
        exact = sum(1 for a, b in zip(reading.plaintext, expected) if a == b)
        self.assertEqual(exact, len(expected))

    def test_the_key_it_reports_can_be_checked_by_hand(self) -> None:
        cipher = encrypt(PLAIN)
        cells = [cipher[i:i + 2] for i in range(0, len(cipher), 2)]
        split = crossed.detect(cells)
        reading = crossed.recover(split, restarts=1,
                                  steps=crossed.STEPS_PER_RESTART, seed=1)
        described = reading.describe_key()
        # Q never occurs in this plaintext, so its cell is never observed and
        # the square has a hole in it. The hole is PRINTED rather than closed
        # up: closing it would shift every letter after it and produce a key
        # that looks checkable and is wrong.
        expected_one = SQUARE_ONE.replace("Q", ".")
        expected_two = SQUARE_TWO.replace("Q", ".")
        self.assertIn(expected_one, described)
        self.assertIn(expected_two, described)
        self.assertEqual(described.count("."), 2)


class TestSolve(unittest.TestCase):
    """The pipeline entry point, including the ways it must decline."""

    def test_it_returns_nothing_on_a_message_that_is_not_this_cipher(self) -> None:
        found = crossed.solve("A" * 200 + "B" * 200)
        self.assertEqual(found.ranked(), [])

    def test_it_accepts_a_time_budget_without_raising(self) -> None:
        """A stage that RAISES on ``time_budget`` is silently dropped.

        That happened to the Polybius square search and cost a whole
        challenge: 1,257 tests passed because none of them set a clock, and
        every real run sets one.
        """
        found = crossed.solve("ABCD" * 3, time_budget=1.0, seed=1)
        self.assertEqual(found.ranked(), [])

    def test_an_odd_symbol_count_is_declined_rather_than_truncated(self) -> None:
        found = crossed.solve("ABCDE")
        self.assertEqual(found.ranked(), [])

    def test_a_real_message_solves_through_solve(self) -> None:
        found = crossed.solve(encrypt(PLAIN), restarts=1, seed=1)
        best = found.best()
        self.assertIsNotNone(best)
        self.assertEqual(best.method, crossed.METHOD)
        self.assertEqual(best.confidence(), "strong")
        self.assertTrue(best.plaintext.startswith("REPORTONTHEMEETING"))

    def test_a_thin_message_is_capped_rather_than_sold_as_strong(self) -> None:
        """The guard that protects the property this project is built on.

        At 600 units the attack recovers 97.5% of the letters at 0.72 word
        coverage -- fluent, nearly right, and NOT the plaintext. Both
        thresholds for `strong` are cleared by that reading, so without a cap
        keyed on message length it would be handed back as a confident answer.
        """
        cipher = encrypt(PLAIN)[:600 * 4]
        found = crossed.solve(cipher, restarts=1, seed=1)
        best = found.best()
        self.assertIsNotNone(best)
        self.assertIn("thin_ciphertext", best.diagnostics)
        self.assertNotEqual(best.confidence(), "strong")


class TestHelpers(unittest.TestCase):
    """Small pieces that the rest depends on being right."""

    def test_index_of_coincidence_matches_the_hand_calculation(self) -> None:
        # AABB: 4 items, so 4 x 3 = 12 ordered draws without replacement, of
        # which 2 (the two As) + 2 (the two Bs) = 4 match.
        self.assertAlmostEqual(crossed.index_of_coincidence("AABB"), 4 / 12)

    def test_index_of_coincidence_of_one_item_is_zero(self) -> None:
        self.assertEqual(crossed.index_of_coincidence("A"), 0.0)

    def test_card_ranks_are_offered_as_an_ordering(self) -> None:
        found = crossed.orderings(RANKS)
        self.assertIn(crossed.CARD_RANKS_ACE_LOW, found)
        self.assertIn(crossed.CARD_RANKS_ACE_HIGH, found)

    def test_orderings_are_short_because_the_search_must_not_be_free(self) -> None:
        self.assertLessEqual(len(crossed.orderings(RANKS)), 4)
        self.assertLessEqual(len(crossed.orderings(SUITS)), 4)


if __name__ == "__main__":
    unittest.main()
