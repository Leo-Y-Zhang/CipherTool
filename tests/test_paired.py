"""Tests for the paired-alphabet recogniser.

The recogniser makes a claim about a message it cannot solve, which is the
most dangerous kind of output this toolkit produces: nobody checks a
description the way they check a plaintext. So the negative tests matter more
than the positive ones, and the shuffled-stream control is the important one --
it is what separates "these symbols alternate" from "these symbols are of two
kinds".
"""

from __future__ import annotations

import random
import unittest

from cipher_tool import paired

RANKS = "23456789XJQKA"
SUITS = "CDHS"


def card_stream(cards: int, seed: int = 1) -> str:
    """A clean rank-then-suit stream of *cards* cards."""
    generator = random.Random(seed)
    return "".join(
        generator.choice(RANKS) + generator.choice(SUITS)
        for _ in range(cards)
    )


def sample_english() -> str:
    """Six hundred letters of ordinary English, letters only."""
    from tests.test_auto import sample_plaintext

    return sample_plaintext(600)


class TestACleanDeckIsRecognised(unittest.TestCase):
    def setUp(self) -> None:
        self.report = paired.recognise(card_stream(300))

    def test_it_is_detected(self) -> None:
        self.assertTrue(self.report.detected, self.report.reason)

    def test_the_two_classes_are_ranks_and_suits(self) -> None:
        level = self.report.levels[0]
        sizes = sorted((len(level.first_class), len(level.second_class)))
        self.assertEqual(sizes, [4, 13])

    def test_the_cell_count_is_reported_not_the_symbol_count(self) -> None:
        level = self.report.levels[0]
        self.assertEqual(level.tokens, 600)
        self.assertEqual(level.units, 300)
        self.assertEqual(level.cells_available, 52)

    def test_it_names_the_deck(self) -> None:
        self.assertEqual(self.report.inventory_name, "playing-card deck")
        self.assertIn("playing-card deck", self.report.description)

    def test_no_breaks(self) -> None:
        self.assertTrue(self.report.levels[0].clean)
        self.assertEqual(self.report.levels[0].breaks, ())

    def test_the_shuffle_control_is_zero(self) -> None:
        """The claim, made falsifiable.

        Same symbols, same counts, order destroyed: if shuffles alternated
        too, the recogniser would be reading the inventory rather than the
        structure, and its description would be worthless.
        """
        self.assertEqual(self.report.shuffle_control, 0.0)

    def test_the_description_never_calls_the_symbols_letters(self) -> None:
        self.assertIn("300 cards", self.report.description)
        self.assertIn("not 600 letters", self.report.description)


class TestASingleTranscriptionSlipIsReportedNotRepaired(unittest.TestCase):
    """One symbol missing is the ordinary state of a hand transcription.

    Refusing on it would make the recogniser useless on real material;
    repairing it would mean inventing a symbol. It reports where.
    """

    def setUp(self) -> None:
        clean = card_stream(300)
        # Delete one suit symbol from the middle. Everything after it is now
        # on the wrong parity, so a recogniser that only tests disjointness
        # sees two classes that share every symbol.
        self.slipped = clean[:201] + clean[202:]
        self.report = paired.recognise(self.slipped)

    def test_still_detected(self) -> None:
        self.assertTrue(self.report.detected, self.report.reason)

    def test_it_says_where(self) -> None:
        level = self.report.levels[0]
        self.assertFalse(level.clean)
        self.assertEqual(len(level.breaks), 1)
        self.assertEqual(level.repairable_at, level.breaks[0])
        self.assertEqual(level.breaks[0], 201)
        self.assertIn("201", self.report.description)

    def test_the_two_classes_survive_the_slip(self) -> None:
        """The assertion that catches a crossed parity hypothesis.

        A deleted token flips the parity of everything after it, so the tail's
        two classes have to be matched to the head's the other way round.
        MEASURED with that matching crossed: the break index is still right,
        detection still fires, and the report says one 17-symbol alphabet
        against another instead of 13 ranks against 4 suits -- so the deck
        goes unnamed and the cell count reads 289 rather than 52. Everything
        that is easy to assert still passes.
        """
        level = self.report.levels[0]
        sizes = sorted((len(level.first_class), len(level.second_class)))
        self.assertEqual(sizes, [4, 13])
        self.assertEqual(level.cells_available, 52)
        self.assertEqual(self.report.inventory_name, "playing-card deck")

    def test_the_cell_count_cannot_exceed_the_cells_that_exist(self) -> None:
        """An impossible number in a description is worse than no number.

        Past the break the naive pairing is off by one, so it manufactures
        rank-suit AND suit-rank cells. Counted over the whole stream that
        gives 96 distinct cells out of the 52 the two alphabets allow, and the
        report says so in one sentence, in plain English, to somebody who
        cannot check it.
        """
        level = self.report.levels[0]
        self.assertLessEqual(level.distinct_units, level.cells_available)
        self.assertIn("before the break", self.report.description)

    def test_it_did_not_touch_the_stream(self) -> None:
        self.assertIn("Nothing was changed", self.report.description)
        self.assertEqual(len(self.slipped), 599)

    def test_the_odd_tail_symbol_is_reported_never_paired_with_nothing(
        self,
    ) -> None:
        level = self.report.levels[0]
        self.assertEqual(level.tokens, 599)
        self.assertEqual(level.units, 299)
        self.assertIn("one symbol left over", self.report.description)


class TestASecondLevelOfPairing(unittest.TestCase):
    """When the CELLS alternate too, the unit is two cells, not one."""

    def setUp(self) -> None:
        generator = random.Random(7)
        first = "23456789"
        second = "XJQKA"
        stream = []
        for index in range(300):
            ranks = first if index % 2 == 0 else second
            stream.append(generator.choice(ranks) + generator.choice(SUITS))
        self.report = paired.recognise("".join(stream))

    def test_two_levels_are_reported(self) -> None:
        self.assertTrue(self.report.detected, self.report.reason)
        self.assertEqual(len(self.report.levels), 2)

    def test_the_cells_themselves_are_two_disjoint_sets(self) -> None:
        level = self.report.levels[1]
        self.assertEqual(level.tokens, 300)
        self.assertEqual(len(level.first_class), 32)   # 8 ranks x 4 suits
        self.assertEqual(len(level.second_class), 20)  # 5 ranks x 4 suits

    def test_it_says_what_that_implies(self) -> None:
        self.assertIn("TWO cells", self.report.description)
        self.assertIn("more than one letter", self.report.description)

    def test_a_broken_stream_gets_no_second_level_claim(self) -> None:
        """Past a slip the cells are fabricated, so they cannot be evidence.

        Level 1 is a claim about the CELLS, and every cell past the break is
        one the naive pairing invented. Looking there at all would let the
        report announce a digraph on evidence that is partly made up.
        """
        generator = random.Random(7)
        stream = []
        for index in range(300):
            ranks = "23456789" if index % 2 == 0 else "XJQKA"
            stream.append(generator.choice(ranks) + generator.choice(SUITS))
        flat = "".join(stream)
        report = paired.recognise(flat[:201] + flat[202:])
        self.assertTrue(report.detected, report.reason)
        self.assertEqual(len(report.levels), 1)
        self.assertNotIn("TWO cells", report.description)


class TestItRefusesToClaimStructureThatIsNotThere(unittest.TestCase):
    def test_english_is_not_a_paired_alphabet(self) -> None:
        report = paired.recognise(sample_english())
        self.assertFalse(report.detected)
        self.assertIn("share", report.reason)

    def test_an_adfgvx_stream_is_not_a_paired_alphabet(self) -> None:
        generator = random.Random(3)
        stream = "".join(generator.choice("ADFGVX") for _ in range(600))
        report = paired.recognise(stream)
        self.assertFalse(report.detected)

    def test_a_separator_is_not_a_paired_alphabet(self) -> None:
        """A1B1C1D1 alternates perfectly and means nothing.

        One class of size one is a delimiter between symbols, not half of a
        two-part alphabet, and selling it as structure would be exactly the
        confident-description failure this module exists to avoid.
        """
        generator = random.Random(5)
        stream = "".join(generator.choice("ABCDEFGH") + "1" for _ in range(300))
        report = paired.recognise(stream)
        self.assertFalse(report.detected)
        self.assertIn("separator", report.reason)

    def test_the_same_symbols_shuffled_are_not_detected(self) -> None:
        """The control, as a test rather than as a statistic.

        Identical multiset, identical counts, order destroyed. A recogniser
        that still fired here would be measuring the inventory.
        """
        symbols = list(card_stream(300))
        random.Random(11).shuffle(symbols)
        report = paired.recognise("".join(symbols))
        self.assertFalse(report.detected, report.description)

    def test_one_symbol_repeated_is_not_structure(self) -> None:
        report = paired.recognise("A" * 1000)
        self.assertFalse(report.detected)

    def test_a_short_stream_refuses_on_length(self) -> None:
        report = paired.recognise(card_stream(300)[: paired.MINIMUM_SYMBOLS - 1])
        self.assertFalse(report.detected)
        self.assertIn(str(paired.MINIMUM_SYMBOLS - 1), report.reason)

    def test_at_the_minimum_length_it_may_detect(self) -> None:
        report = paired.recognise(card_stream(300)[: paired.MINIMUM_SYMBOLS])
        self.assertTrue(report.detected, report.reason)

    def test_an_empty_stream_is_refused_not_crashed(self) -> None:
        report = paired.recognise("")
        self.assertFalse(report.detected)
        self.assertEqual(report.levels, ())


class TestCells(unittest.TestCase):
    def test_pairs_up(self) -> None:
        self.assertEqual(paired.cells("7C3H"), ["7C", "3H"])

    def test_a_lone_tail_symbol_is_dropped_not_padded(self) -> None:
        self.assertEqual(paired.cells("7C3H5"), ["7C", "3H"])


if __name__ == "__main__":
    unittest.main()
