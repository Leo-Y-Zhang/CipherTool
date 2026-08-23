"""The Playfair same-column variant, and the guard that keeps it honest.

WHY THIS EXISTS, because the number is the argument. On a real 1,502-letter
competition message the shipped square search converged on one square from 16
of 40 independent restarts, and that square decrypted:

    rectangle digraphs    541 of 541 correct
    same-row digraphs     111 of 111 correct
    same-column digraphs    0 of  99 correct

The search had already done its whole job. One rule out of three was wrong,
and because a same-column digraph is only about one in seven, the resulting
reading was fluent English at 0.61 word coverage -- readable, plausible, and
wrong in 99 places. That is the most dangerous shape of answer this toolkit
can produce, and no amount of extra searching would have fixed it.

The second half of the tests is the guard: an ORDINARY Playfair message must
not gain a second, worse answer. The variant reading is offered only when it
scores better than the ordinary rule on the same square.
"""

from __future__ import annotations

import unittest

from cipher_tool import playfair
from cipher_tool.scoring import default_scorer

KEY = "MONARCHY"

PLAIN = (
    "CHARLESIFINDTHATICANNOTWRITEDEARCHARLESORANYOTHERTERMOFENDEARMENT"
    "INTHISLETTERANDSTILLRETAINMYHONOURIWILNOTBELIEVETHATYOUHONESTLY"
    "EXPECTMETOSUSPECTMYCONFIDANTEMISSTERNANOFANYTHINGSOUNDERHANDAS"
    "THEFOULTREACHERYYOUHAVESUGGESTEDELLENISADEARFRIENDANDHASBEEN"
    "NOTHINGBUTFAIRINHERDEALINGSWITHMESHEWANTSONLYWHATISBESTANDI"
    "CANNOTBELIEVEANYTHINGEVILOFHER"
) * 3


class TestTheRule(unittest.TestCase):
    """The cipher itself."""

    def test_the_variant_round_trips(self) -> None:
        cipher = playfair.encrypt(
            PLAIN, KEY, column_rule=playfair.COLUMN_RULE_SIDEWAYS)
        back = playfair.decrypt(
            cipher, KEY, column_rule=playfair.COLUMN_RULE_SIDEWAYS)
        self.assertEqual(
            back, playfair.prepare_text(PLAIN, square=playfair.as_square(KEY)))

    def test_the_ordinary_rule_is_untouched(self) -> None:
        self.assertEqual(
            playfair.decrypt(playfair.encrypt("ATTACKATDAWN", KEY), KEY),
            "ATTACKATDAWN")

    def test_the_two_rules_differ_only_on_same_column_pairs(self) -> None:
        """The whole reason the variant is invisible to a square search."""
        square = playfair.as_square(KEY)
        cipher = playfair.encrypt(PLAIN, KEY)
        ordinary = playfair.decrypt(cipher, KEY)
        variant = playfair.decrypt(
            cipher, KEY, column_rule=playfair.COLUMN_RULE_SIDEWAYS)
        differed = shared_column = 0
        for index in range(0, len(cipher) - 1, 2):
            row_a, column_a = square.position(cipher[index])
            row_b, column_b = square.position(cipher[index + 1])
            same_column = row_a != row_b and column_a == column_b
            shared_column += same_column
            if ordinary[index:index + 2] != variant[index:index + 2]:
                differed += 1
                self.assertTrue(same_column,
                                "a rule changed something outside its case")
        self.assertEqual(differed, shared_column)
        self.assertGreater(shared_column, 0, "no same-column pairs to test")

    def test_most_of_the_message_reads_the_same_either_way(self) -> None:
        """Which is exactly why the wrong rule produces fluent, wrong English."""
        cipher = playfair.encrypt(PLAIN, KEY)
        ordinary = playfair.decrypt(cipher, KEY)
        variant = playfair.decrypt(
            cipher, KEY, column_rule=playfair.COLUMN_RULE_SIDEWAYS)
        same = sum(1 for a, b in zip(ordinary, variant) if a == b)
        self.assertGreater(same / len(ordinary), 0.6)

    def test_an_unknown_column_rule_falls_back_to_the_ordinary_one(self) -> None:
        self.assertEqual(
            playfair.decrypt(playfair.encrypt("ATTACKATDAWN", KEY), KEY,
                             column_rule="nonsense"),
            "ATTACKATDAWN")


class TestTheSearch(unittest.TestCase):
    """What `solve` does with it, in both directions."""

    def setUp(self) -> None:
        self.scorer = default_scorer()

    def test_a_variant_message_is_read_correctly(self) -> None:
        cipher = playfair.encrypt(
            PLAIN, KEY, column_rule=playfair.COLUMN_RULE_SIDEWAYS)
        found = playfair.solve(cipher, scorer=self.scorer, top=5,
                               restarts=8, seed=1, stop_when_strong=False)
        best = found.best()
        self.assertIsNotNone(best)
        self.assertIn("same-column rule reversed", best.method)
        self.assertTrue(best.plaintext.startswith("CHARLESIFINDTHATICANNOT"))

    def test_an_ordinary_message_gains_no_variant_answer(self) -> None:
        """The guard. A second, worse reading beside a correct one is a cost."""
        cipher = playfair.encrypt(PLAIN, KEY)
        found = playfair.solve(cipher, scorer=self.scorer, top=5,
                               restarts=8, seed=1, stop_when_strong=False)
        best = found.best()
        self.assertIsNotNone(best)
        self.assertEqual(best.method, "Playfair")
        self.assertTrue(best.plaintext.startswith("CHARLESIFINDTHATICANNOT"))

    def test_the_variant_answer_says_which_rule_it_used(self) -> None:
        cipher = playfair.encrypt(
            PLAIN, KEY, column_rule=playfair.COLUMN_RULE_SIDEWAYS)
        found = playfair.solve(cipher, scorer=self.scorer, top=5,
                               restarts=8, seed=1, stop_when_strong=False)
        best = found.best()
        self.assertEqual(best.diagnostics.get("column_rule"),
                         playfair.COLUMN_RULE_SIDEWAYS)
        self.assertIn("ALONG THE ROW",
                      best.diagnostics["column_rule_note"])
        self.assertIn("column rule=sideways", best.key)


class TestTheSecondPassClock(unittest.TestCase):
    """A DEFECT IN THIS FEATURE, found before it shipped and pinned here.

    The second pass was originally handed the caller's ORIGINAL
    ``time_budget``, which restarts the clock: a stage given eight seconds
    could take sixteen. ``auto_solve`` shares one deadline across every stage
    by weight, so that overrun is taken straight out of the searches that come
    after it -- the stage does not merely run long, it makes other stages fail.

    Tested through a pure helper rather than a stopwatch, because a randomised
    search asserted against a wall clock is a flaky test by construction and
    this repository has already been bitten by one.
    """

    def test_no_budget_means_the_second_pass_has_none_either(self) -> None:
        self.assertEqual(playfair.variant_pass_budget(None, 100.0),
                         (True, None))

    def test_the_second_pass_gets_what_is_left_not_a_fresh_budget(self) -> None:
        may_run, remaining = playfair.variant_pass_budget(110.0, 100.0)
        self.assertTrue(may_run)
        self.assertAlmostEqual(remaining, 10.0)

    def test_an_exhausted_budget_skips_the_second_pass_entirely(self) -> None:
        self.assertEqual(playfair.variant_pass_budget(100.0, 100.0),
                         (False, 0.0))

    def test_a_sliver_of_time_is_not_worth_starting_on(self) -> None:
        may_run, _ = playfair.variant_pass_budget(
            100.0, 100.0 - playfair.MINIMUM_SECOND_PASS_SECONDS / 2)
        self.assertFalse(may_run)


if __name__ == "__main__":
    unittest.main()
