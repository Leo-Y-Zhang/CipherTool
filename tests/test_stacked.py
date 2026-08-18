"""Tests for two ciphers piled up: a polyalphabetic, then a transposition.

The reason this module exists is a mistake worth pinning down. The 2017
National Cipher Challenge 7B is a Vigenere under SCYTALE followed by a
six-column transposition. An earlier attempt recovered the Vigenere key
CORRECTLY, read the gibberish it produced, and concluded that the alphabets
must be mixed -- filing the second cipher as evidence for a harder version of
the first. `test_the_polyalphabetic_solver_alone_cannot_read_it` is that
mistake as a test: the single-layer solvers must fail, and the stacked one
must not.

The measurement the attack rests on is pinned by
`test_the_signal_separates_the_right_shape`. Split the ciphertext into
`width` contiguous blocks and take the mean index of coincidence of the
cosets at spacing `period` inside each block: right shape near 0.066, wrong
shape near 0.040, and no knowledge of the key or the column order needed.
"""

from __future__ import annotations

import time
import unittest
from collections import Counter

from cipher_tool import columnar, stacked, vigenere
from cipher_tool.normalize import letters_only
from cipher_tool.scoring import DATA_DIR, default_scorer

CORPUS = letters_only(
    (DATA_DIR / "corpus_03_letters.txt").read_text(encoding="utf-8")
)


class TestTheDetector(unittest.TestCase):
    """Finding the shape without knowing anything about the key."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.plaintext = CORPUS[:3000]
        cls.ciphertext = columnar.encrypt(
            vigenere.encrypt(cls.plaintext, "SCYTALE"), "ZEBRAS"
        )

    def test_the_signal_separates_the_right_shape(self) -> None:
        """If this stops being true the attack has no signal to work on."""
        right = stacked.block_signal(self.ciphertext, 6, 7)
        for width, period in ((6, 5), (5, 7), (4, 4), (3, 9)):
            with self.subTest(width=width, period=period):
                self.assertGreater(
                    right, stacked.block_signal(self.ciphertext, width, period)
                )
        self.assertGreater(right, 0.060)
        self.assertLess(stacked.block_signal(self.ciphertext, 5, 7), 0.055)

    def test_it_detects_the_shape(self) -> None:
        shape = stacked.detect(self.ciphertext)
        self.assertIsNotNone(shape)
        width, period, _ = shape
        self.assertEqual((width, period), (6, 7))

    def test_it_prefers_the_smallest_shape_not_the_highest_score(self) -> None:
        """Multiples of the true shape peak too, and must not win.

        Splitting a real column in half leaves the key phase intact inside
        each half, so a multiple of the true width scores just as well.

        This case is chosen because the multiple actually WINS on it: a
        5-letter key under a 6-column transposition, 2,000 letters, and the
        highest-scoring shape is width 12, not width 6. A case where the
        truth happens to score highest cannot tell the two rules apart, and
        the first version of this test used one -- it passed with the rule
        deleted. Measured over 120 constructions, the highest-scoring shape
        was NOT the true one in 73 of them, so this is the common case
        rather than a corner.
        """
        plaintext = CORPUS[:2000]
        ciphertext = columnar.encrypt(
            vigenere.encrypt(plaintext, "ZEBRA"), "ZEBRAS"
        )
        scan = [(stacked.block_signal(ciphertext, width, period),
                 width, period)
                for width in range(1, 13) for period in range(2, 13)]
        _, best_width, best_period = max(scan)
        self.assertEqual((best_width, best_period), (12, 5),
                         "premise: the multiple outscores the truth here")
        self.assertEqual(stacked.detect(ciphertext)[:2], (6, 5))

    def test_plain_english_scores_highly_at_every_shape(self) -> None:
        """Which is why the solver must refuse English rather than sweep it.

        English cosets have an English index of coincidence however you cut
        them, so the detector cannot tell prose from a peeled stack. The
        guard is the whole-message IC, not the detector.
        """
        worst = min(
            stacked.block_signal(CORPUS[:3000], width, period)
            for width in (2, 6, 10) for period in (3, 7, 11)
        )
        self.assertGreater(worst, stacked.MIN_SIGNAL)
        self.assertGreater(
            stacked.index_of_coincidence(CORPUS[:3000]), stacked.MAX_SOURCE_IC
        )


class TestTheAttack(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = default_scorer()
        cls.plaintext = CORPUS[:3000]
        cls.middle = vigenere.encrypt(cls.plaintext, "SCYTALE")
        cls.ciphertext = columnar.encrypt(cls.middle, "ZEBRAS")

    def test_it_solves_a_stacked_message_exactly(self) -> None:
        best = stacked.solve(self.ciphertext, scorer=self.scorer, top=1,
                             seed=1).best()
        self.assertIsNotNone(best)
        self.assertEqual(best.plaintext, self.plaintext)
        self.assertEqual(best.confidence(), "strong")

    def test_it_reports_the_period_and_the_column_count(self) -> None:
        best = stacked.solve(self.ciphertext, scorer=self.scorer, top=1,
                             seed=1).best()
        self.assertEqual(best.diagnostics["columns"], 6)
        self.assertEqual(best.diagnostics["polyalphabetic_period"], 7)
        self.assertIn("polyalphabetic_key", best.diagnostics)

    def test_the_recovered_key_is_the_keyword_decimated_by_the_width(self) -> None:
        """It is NOT simply a rotation of the keyword, and the reason is real.

        The key was applied along the plaintext, but this attack reads it
        along a COLUMN of the transposition -- and going down a column steps
        the plaintext index by `width`. So the key comes back sampled with
        stride `width mod period`, from an unknown starting point.

        Here that stride is 6 mod 7, which is -1, so the recovered key is
        SCYTALE BACKWARDS: ELATYCS, and what the solver reports is ATYCSEL,
        a rotation of it. Asserting a rotation of SCYTALE looks obviously
        right and fails. The same thing shows up on the real 2017 7B
        message, where the recovered key is SELATYC -- also a rotation of
        ELATYCS.

        Worth knowing before reporting a key to a competition: the letters
        are right, the reading order is a decimation.
        """
        best = stacked.solve(self.ciphertext, scorer=self.scorer, top=1,
                             seed=1).best()
        key = best.diagnostics["polyalphabetic_key"]
        keyword, width, period = "SCYTALE", 6, 7
        stride = width % period
        decimations = {
            "".join(keyword[(start + step * stride) % period]
                    for step in range(period))
            for start in range(period)
        }
        self.assertIn(key, decimations)
        self.assertNotIn(
            key, {keyword[i:] + keyword[:i] for i in range(period)},
            "premise: a plain rotation is NOT what comes back",
        )

    def test_the_polyalphabetic_solver_alone_cannot_read_it(self) -> None:
        """The mistake this module was built to stop, as a test.

        A Vigenere attack on a stacked message cannot reach the plaintext,
        because undoing the Vigenere leaves a transposition. Reading that as
        "the key must be wrong" is what sent an earlier attempt looking for
        mixed alphabets that were never there.
        """
        found = vigenere.solve(self.ciphertext, scorer=self.scorer, top=5)
        for candidate in found.ranked():
            self.assertNotEqual(candidate.plaintext, self.plaintext)

    def test_the_transposition_solver_alone_cannot_read_it_either(self) -> None:
        found = columnar.solve(self.ciphertext, scorer=self.scorer, top=5,
                               seed=1, max_key_length=9)
        for candidate in found.ranked():
            self.assertNotEqual(candidate.plaintext, self.plaintext)

    def test_peeling_leaves_a_transposition_of_the_plaintext(self) -> None:
        """What comes off is letter-for-letter a transposition of English.

        Not of the middle layer -- peeling REMOVES the polyalphabetic, so
        what is left has the plaintext's letters in the wrong order. Getting
        that backwards is easy and the assertion is the only thing that says
        which is meant.

        Counters, not sorted lists: comparing two three-thousand-element
        lists is fine when it passes and takes minutes to PRINT when it
        fails, because unittest sits there building a diff of them.
        """
        stripped, key = stacked.peel(self.ciphertext, 6, 7)
        self.assertEqual(len(stripped), len(self.ciphertext))
        self.assertEqual(Counter(stripped), Counter(self.plaintext))
        self.assertEqual(len(key), 7)

    # -- the guards --------------------------------------------------------

    def test_plain_english_is_refused(self) -> None:
        self.assertEqual(
            len(stacked.solve(self.plaintext, scorer=self.scorer)), 0
        )

    def test_english_is_refused_by_the_shape_rule_as_well(self) -> None:
        """The index-of-coincidence guard is an early exit, not the net.

        Deleting that guard changes no answer, and this test is what says
        so. English-shaped text is refused further down anyway, because its
        cosets look English at EVERY shape, so the smallest tied shape is a
        width of 1 -- no transposition, nothing here to attack. Measured
        over sixteen English-shaped texts, prose and three transpositions,
        1,000 to 5,000 letters: not one got past with a width of 2 or more.

        Without this, the guard's comment would be an unchecked claim about
        a line whose removal breaks nothing.
        """
        for label, text in (
            ("prose", self.plaintext),
            ("columnar", columnar.encrypt(self.plaintext, "ZEBRAS")),
        ):
            with self.subTest(text=label):
                shape = stacked.detect(text)
                self.assertIsNotNone(shape)
                self.assertEqual(shape[0], 1)

    def test_a_plain_polyalphabetic_is_refused(self) -> None:
        """No transposition means no stack; the Vigenere solver owns this."""
        self.assertEqual(
            len(stacked.solve(self.middle, scorer=self.scorer)), 0
        )

    def test_a_plain_transposition_is_refused(self) -> None:
        moved = columnar.encrypt(self.plaintext, "ZEBRAS")
        self.assertEqual(len(stacked.solve(moved, scorer=self.scorer)), 0)

    def test_a_short_message_is_refused_rather_than_guessed_at(self) -> None:
        """Measured: 600 letters of this construction is not detectable.

        Returning nothing is the honest answer. The failure mode this avoids
        is a stage that runs, spends time and cannot reach the answer, whose
        presence then reads as coverage.
        """
        self.assertEqual(
            len(stacked.solve(self.ciphertext[:600], scorer=self.scorer)), 0
        )

    def test_random_letters_are_refused_or_not_called_strong(self) -> None:
        import random

        generator = random.Random(7)
        noise = "".join(generator.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                        for _ in range(3000))
        best = stacked.solve(noise, scorer=self.scorer, top=1, seed=1).best()
        if best is not None:
            self.assertNotEqual(best.confidence(), "strong")

    # -- housekeeping ------------------------------------------------------

    def test_a_time_budget_is_accepted_and_honoured(self) -> None:
        """A stage that refuses the clock is silently dropped from real runs."""
        started = time.monotonic()
        found = stacked.solve(self.ciphertext, scorer=self.scorer, top=1,
                              seed=1, time_budget=30.0)
        self.assertGreater(len(found), 0)
        self.assertLess(time.monotonic() - started, 60.0)

    def test_an_unknown_option_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            stacked.solve(self.ciphertext, scorer=self.scorer, wibble=1)

    def test_empty_input_is_not_a_crash(self) -> None:
        self.assertEqual(len(stacked.solve("", scorer=self.scorer)), 0)

    def test_the_shape_can_be_pinned(self) -> None:
        best = stacked.solve(self.ciphertext, scorer=self.scorer, top=1,
                             seed=1, width=6, period=7).best()
        self.assertEqual(best.plaintext, self.plaintext)


class TestPipeline(unittest.TestCase):
    """A solver nothing calls is decoration."""

    def test_it_is_in_the_pipeline(self) -> None:
        from cipher_tool.auto import build_stages

        names = [stage.name for stage in build_stages("normal", 5, 1)]
        self.assertIn("stacked (polyalphabetic + transposition)", names)

    def test_it_does_not_run_at_the_cheapest_level(self) -> None:
        """It cannot recognise its cipher for free, so it waits its turn."""
        from cipher_tool.auto import build_stages

        names = [stage.name for stage in build_stages("fast", 5, 1)]
        self.assertNotIn("stacked (polyalphabetic + transposition)", names)

    def test_the_pipeline_solves_one_end_to_end(self) -> None:
        """Through auto_solve, but with this stage alone.

        Deliberately not the whole `normal` ladder. A stacked message has to
        be long -- about 900 letters at the least -- and running every normal
        stage over three thousand letters took more than thirteen minutes,
        which is not a test, it is a tax on every future run. What matters
        here is that the stage is reachable through auto_solve's machinery:
        that it accepts the options the pipeline passes it, tolerates the
        clock, and gets its candidates into the merged set. That the ladder
        contains it is checked separately and for free.
        """
        from cipher_tool.auto import auto_solve, build_stages

        plaintext = CORPUS[:3000]
        ciphertext = columnar.encrypt(
            vigenere.encrypt(plaintext, "SCYTALE"), "ZEBRAS"
        )
        stage = [s for s in build_stages("normal", 5, 1)
                 if s.name == "stacked (polyalphabetic + transposition)"]
        self.assertEqual(len(stage), 1)
        result = auto_solve(ciphertext, effort="normal", top=3, seed=1,
                            stages=stage)
        self.assertEqual(result.candidates.best().plaintext, plaintext)


if __name__ == "__main__":
    unittest.main()
