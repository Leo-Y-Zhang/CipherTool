"""Tests for the Nihilist cipher.

The attack here is a CONSTRAINT, not a score, and the tests are written to
keep it that way. A wrong period does not merely score badly: it leaves a
class of values that no single key coordinate can decode, and the period is
excluded outright. So the tests assert exclusion, not ranking.

The other thing pinned here is the reason the cipher was unreadable at all.
Its tokens are whitespace-separated and of variable width, and normalising the
paste keeps the digits while throwing the separators away -- which turns
``"97 26 57"`` into ``"972657"`` and destroys the message. ``parse`` reads the
RAW text for that reason, and there is a test that says so.
"""

from __future__ import annotations

import unittest

from cipher_tool import nihilist

SQUARE = "ZEROPQSTUVWXYABCDFGHIKLMN"
KEY = "NOTHING"

# Repeated to a competition length, and carrying its rare letters inside
# ORDINARY WORDS on purpose. The period and the key are deduced from a handful
# of values, but the substitution underneath is an ordinary hill climb. The
# first draft of this text used X only in the proper noun LENNOX and used Q not
# at all, and the climb read LENNOQ -- neither reading is a dictionary word, so
# nothing in the scorer could separate them. That is a fair limit of the
# substitution solver, not of this module, and testing it here would be testing
# the wrong thing.
PLAIN = (
    "DEARMRROGERSYOUWILLNEVERKNOWHOWRELIEVEDIAMTOHEARTHATTHELIBRARYWILL"
    "REMAININTACTANDHEREATTHEMANORIKNOWTHATYOUWEREKEENTOSUPPORTOURCITY"
    "INITSAMBITIONTORIVALNEWYORKWITHITSASTORANDLENNOXLIBRARIESBUTTHEBOOKS"
    "ARESAFERHEREANDIAMGRATEFULTHATYOUAGREETHENEXTBOXOFTEXTSWASEXAMINED"
    "BYTHEEXPERTWHOQUESTIONEDTHEQUALITYOFSIXOFTHEMANDASKEDMEAQUESTION"
    "ABOUTTHEEXACTSIZEOFTHEQUANTITYWEEXPECTTOEXAMINENEXTMONTH"
) * 3


class TestRoundTrip(unittest.TestCase):
    """The cipher itself, before any cryptanalysis."""

    def test_encrypt_then_decrypt_returns_the_message(self) -> None:
        values = nihilist.encrypt(PLAIN, SQUARE, KEY)
        self.assertEqual(nihilist.decrypt(values, SQUARE, KEY), PLAIN)

    def test_every_value_is_a_sum_with_no_carry(self) -> None:
        """The property the whole attack rests on.

        No carry between the digits means the tens and units run 2..10
        independently, so a value ending in 1 is impossible whatever the key.
        """
        for value in nihilist.encrypt(PLAIN, SQUARE, KEY):
            tens = [row for row in range(2, 11) if 2 <= value - 10 * row <= 10]
            self.assertTrue(tens, f"{value} is not a carry-free sum")
        # ... and the values a carry-free sum can NEVER take.
        possible = {10 * r + c for r in range(2, 11) for c in range(2, 11)}
        for impossible in (21, 31, 41, 51, 61, 71, 81, 91, 101, 111):
            self.assertNotIn(impossible, possible)

    def test_a_letter_outside_the_square_is_dropped_not_invented(self) -> None:
        values = nihilist.encrypt("JJJ" + PLAIN[:10], SQUARE, KEY)
        self.assertEqual(len(values), 10)

    def test_an_empty_key_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            nihilist.encrypt(PLAIN, SQUARE, "")

    def test_a_value_that_cannot_decode_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            nihilist.decrypt([21], SQUARE, KEY)


class TestParsing(unittest.TestCase):
    """The reason this cipher was invisible to the toolkit."""

    def test_variable_width_tokens_survive(self) -> None:
        self.assertEqual(nihilist.parse("97 26 57 105 68"),
                         [97, 26, 57, 105, 68])

    def test_it_reads_the_raw_text_not_a_normalised_one(self) -> None:
        """Normalising keeps the digits and destroys the boundaries.

        ``"97 26 57"`` normalised is ``"972657"``, which is a different
        message: six single digits instead of three numbers. Reading the raw
        text is not a convenience, it is the whole point.
        """
        joined = "".join("97 26 57".split())
        self.assertNotEqual(nihilist.parse("97 26 57"),
                            nihilist.parse(" ".join(joined)))

    def test_prose_around_the_numbers_is_ignored(self) -> None:
        self.assertEqual(nihilist.parse("Nothing to go on? 97 26 57"),
                         [97, 26, 57])


class TestDetection(unittest.TestCase):
    """Period and key are DEDUCED, so the tests assert exclusion."""

    def setUp(self) -> None:
        self.values = nihilist.encrypt(PLAIN, SQUARE, KEY)

    def test_it_finds_the_true_period(self) -> None:
        found = nihilist.detect(self.values)
        self.assertIsNotNone(found)
        self.assertEqual(found.period, len(KEY))

    def test_it_recovers_the_key_coordinates_exactly(self) -> None:
        found = nihilist.detect(self.values)
        expected = []
        for character in KEY:
            index = SQUARE.index(character)
            expected.append(10 * (index // 5 + 1) + (index % 5 + 1))
        self.assertEqual(list(found.key_coordinates), expected)

    def test_every_wrong_period_is_excluded_not_merely_outscored(self) -> None:
        for period in range(1, len(KEY)):
            classes = [self.values[start::period] for start in range(period)]
            survivors = [nihilist.feasible_keys(group) for group in classes]
            self.assertTrue(any(not group for group in survivors),
                            f"period {period} was not excluded")

    def test_the_shortest_period_wins_over_its_multiples(self) -> None:
        """A multiple of the true period always works too.

        Reporting it would name a longer key than the message uses, which is
        the same mistake the block-permutation detector had to be taught not
        to make.
        """
        doubled = [nihilist.feasible_keys(self.values[start::2 * len(KEY)])
                   for start in range(2 * len(KEY))]
        self.assertTrue(all(doubled))
        self.assertEqual(nihilist.detect(self.values).period, len(KEY))

    def test_a_stream_that_is_not_nihilist_is_refused(self) -> None:
        self.assertIsNone(nihilist.detect([21] * 200))

    def test_too_few_tokens_are_refused_rather_than_guessed(self) -> None:
        self.assertIsNone(nihilist.detect(self.values[:20]))


class TestSolve(unittest.TestCase):
    """End to end, including the ways it must decline."""

    def test_it_reads_a_real_shaped_message(self) -> None:
        values = nihilist.encrypt(PLAIN, SQUARE, KEY)
        found = nihilist.solve(" ".join(str(v) for v in values), seed=1)
        best = found.best()
        self.assertIsNotNone(best)
        self.assertEqual(best.method, nihilist.METHOD)
        self.assertEqual(best.plaintext, PLAIN)
        self.assertIn(f"key={KEY}", best.key)

    def test_the_whole_square_is_reported_when_every_cell_is_used(self) -> None:
        values = nihilist.encrypt(PLAIN, SQUARE, KEY)
        best = nihilist.solve(" ".join(str(v) for v in values), seed=1).best()
        self.assertIn("square=" + SQUARE, best.key)

    def test_a_cell_the_message_never_used_is_shown_as_a_hole(self) -> None:
        """The hole is printed rather than closed up.

        Closing it shifts every later cell to the wrong coordinate and
        produces a key that looks checkable by hand and is not.
        """
        without_z = PLAIN.replace("Z", "S")
        values = nihilist.encrypt(without_z, SQUARE, KEY)
        best = nihilist.solve(" ".join(str(v) for v in values), seed=1).best()
        self.assertIn("square=" + SQUARE.replace("Z", "."), best.key)

    def test_single_digits_are_left_to_the_polybius_solver(self) -> None:
        found = nihilist.solve(" ".join("12345" * 40))
        self.assertEqual(found.ranked(), [])

    def test_text_with_no_numbers_is_declined(self) -> None:
        self.assertEqual(nihilist.solve("THEQUICKBROWNFOX").ranked(), [])

    def test_it_accepts_a_time_budget_without_raising(self) -> None:
        """A stage that RAISES on ``time_budget`` is silently dropped."""
        self.assertEqual(
            nihilist.solve("1 2 3", time_budget=1.0).ranked(), [])


if __name__ == "__main__":
    unittest.main()
