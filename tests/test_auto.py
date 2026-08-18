"""Tests for the cross-solver pipeline.

The behaviour that matters most here is honesty about coverage: the report
must say which stages ran and which did not, and it must not present a
partial run as a complete one.
"""

from __future__ import annotations

import unittest

from cipher_tool import caesar, vigenere
from cipher_tool.auto import (
    EFFORT_LEVELS,
    Stage,
    auto_solve,
    build_stages,
    non_letters_are_material,
    order_stages,
    quick_triage,
)
from cipher_tool.candidates import Candidate, CandidateSet
from cipher_tool.normalize import NormalizedText, letters_only, normalize
from cipher_tool.scoring import corpus_files
from cipher_tool.statistics import analyse


def sample_plaintext(length: int = 500) -> str:
    return letters_only(corpus_files()[2].read_text(encoding="utf-8"))[:length]


class TestStagePlanning(unittest.TestCase):
    def test_effort_levels_are_nested(self) -> None:
        fast = {stage.name for stage in build_stages("fast", 5, None)}
        normal = {stage.name for stage in build_stages("normal", 5, None)}
        deep = {stage.name for stage in build_stages("deep", 5, None)}
        self.assertTrue(fast < normal < deep)

    def test_fast_excludes_the_expensive_searches(self) -> None:
        fast = {stage.name for stage in build_stages("fast", 5, None)}
        self.assertNotIn("Playfair", fast)
        self.assertNotIn("Hill 2x2", fast)

    def test_deep_searches_harder_than_fast(self) -> None:
        def restarts(effort: str) -> int:
            for stage in build_stages(effort, 5, None):
                if stage.name == "substitution":
                    return stage.options["restarts"]
            raise AssertionError("substitution stage missing")

        self.assertLess(restarts("fast"), restarts("normal"))
        self.assertLess(restarts("normal"), restarts("deep"))

    def test_ordering_promotes_the_suggested_family(self) -> None:
        # Statistics choose the ORDER so that a truncated run does the most
        # useful work first. They must not remove anything.
        # Uses "deep" because the digraphic stages only exist at that level.
        stats = analyse(vigenere.encrypt(sample_plaintext(900), "KEYWORD"))
        stages = build_stages("deep", 5, None)
        ordered = order_stages(stages, stats)
        self.assertEqual(len(ordered), len(stages))
        self.assertEqual({s.name for s in ordered}, {s.name for s in stages})
        families = [stage.family for stage in ordered]
        self.assertLess(families.index("polyalphabetic"),
                        families.index("digraphic"))

    def test_ordering_never_drops_a_stage(self) -> None:
        for text in (sample_plaintext(400),
                     caesar.encrypt(sample_plaintext(400), 4),
                     sample_plaintext(400)[::-1]):
            stats = analyse(text)
            stages = build_stages("deep", 5, None)
            self.assertEqual(len(order_stages(stages, stats)), len(stages))


class TestPipeline(unittest.TestCase):
    def test_finds_a_caesar_quickly(self) -> None:
        result = auto_solve(caesar.encrypt(sample_plaintext(400), 9),
                            effort="fast", top=5, seed=1)
        best = result.candidates.best()
        self.assertIsNotNone(best)
        self.assertIn(sample_plaintext(60), best.plaintext)
        self.assertEqual(best.confidence(), "strong")

    def test_finds_a_vigenere(self) -> None:
        result = auto_solve(vigenere.encrypt(sample_plaintext(700), "ORCHARD"),
                            effort="fast", top=5, seed=1)
        best = result.candidates.best()
        self.assertIsNotNone(best)
        self.assertIn(sample_plaintext(60), best.plaintext)

    def test_rejects_an_unknown_effort_level(self) -> None:
        with self.assertRaises(ValueError) as context:
            auto_solve("ABCDEF", effort="thorough")
        self.assertIn("thorough", str(context.exception))
        for level in EFFORT_LEVELS:
            self.assertIn(level, str(context.exception))

    def test_empty_input_is_handled_and_explained(self) -> None:
        result = auto_solve("1234 !!!")
        self.assertEqual(len(result.candidates), 0)
        self.assertEqual(len(result.stages), 1)
        self.assertFalse(result.stages[0].ran)
        self.assertIn("no letters", result.stages[0].note)

    def test_a_broken_solver_does_not_kill_the_run(self) -> None:
        from cipher_tool.auto import Stage

        def explode(*args: object, **kwargs: object):
            raise RuntimeError("deliberate failure")

        stages = [
            Stage("boom", "monoalphabetic", "fast", 1.0, explode),
            Stage("Caesar", "monoalphabetic", "fast", 0.2, caesar.solve,
                  {"top": 3}),
        ]
        result = auto_solve(caesar.encrypt(sample_plaintext(300), 3),
                            effort="fast", stages=stages)
        names = {stage.name: stage for stage in result.stages}
        self.assertFalse(names["boom"].ran)
        self.assertIn("deliberate failure", names["boom"].note)
        self.assertTrue(names["Caesar"].ran)
        self.assertTrue(result.candidates.ranked())


class TestTimeBudget(unittest.TestCase):
    def test_a_tiny_budget_skips_stages_and_says_so(self) -> None:
        result = auto_solve(sample_plaintext(600), effort="deep", top=3,
                            max_time=0.6, seed=1)
        skipped = [stage for stage in result.stages if not stage.ran]
        self.assertTrue(skipped, "a 0.6s deep run must skip something")
        self.assertTrue(result.budget_exhausted)
        self.assertTrue(any("time budget" in stage.note for stage in skipped))

    def test_skipped_stages_appear_in_the_report(self) -> None:
        # A report that silently omitted them would read as "we tried
        # everything" when it did not.
        result = auto_solve(sample_plaintext(600), effort="deep", top=3,
                            max_time=0.6, seed=1)
        rendered = result.render(top=3)
        self.assertIn("NOT TRIED", rendered)
        for stage in result.stages:
            self.assertIn(stage.name, rendered)

    def test_budget_is_broadly_respected(self) -> None:
        # Not a hard real-time guarantee -- a stage already running cannot be
        # interrupted -- but it must not overrun wildly.
        result = auto_solve(sample_plaintext(800), effort="deep", top=3,
                            max_time=8.0, seed=1)
        self.assertLess(result.seconds, 45.0)


class TestReporting(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = auto_solve(caesar.encrypt(sample_plaintext(400), 9),
                                effort="fast", top=5, seed=1)
        cls.rendered = cls.result.render(top=5)

    def test_lists_every_stage(self) -> None:
        for stage in self.result.stages:
            self.assertIn(stage.name, self.rendered)

    def test_labels_the_family_guesses_as_heuristics(self) -> None:
        self.assertIn("HEURISTIC", self.rendered)
        self.assertIn("often wrong", self.rendered)

    def test_reports_the_score_gap(self) -> None:
        self.assertIn("per letter", self.rendered)

    def test_warns_when_the_top_two_are_close(self) -> None:
        # Random letters produce a field of equally bad candidates, which is
        # exactly when the number means nothing.
        import random

        generator = random.Random(4)
        noise = "".join(generator.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                        for _ in range(300))
        result = auto_solve(noise, effort="fast", top=5, seed=1)
        rendered = result.render(top=5)
        self.assertTrue(
            "WARNING" in rendered or "Nothing scored well" in rendered,
            "a hopeless ciphertext must be reported as such",
        )

    def test_says_so_when_nothing_scored_well(self) -> None:
        import random

        generator = random.Random(8)
        noise = "".join(generator.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                        for _ in range(300))
        rendered = auto_solve(noise, effort="fast", top=3,
                              seed=1).render(top=3)
        self.assertIn("Nothing scored well", rendered)


class TestQuickTriage(unittest.TestCase):
    def test_reports_a_cheap_win(self) -> None:
        summary = quick_triage(caesar.encrypt(sample_plaintext(400), 6))
        self.assertIn("Caesar", summary)

    def test_reports_when_nothing_cheap_worked(self) -> None:
        summary = quick_triage(vigenere.encrypt(sample_plaintext(400),
                                                "LONGERKEYWORD"))
        self.assertTrue(summary.strip())


class TestReversedPlaintextIsFound(unittest.TestCase):
    """Writing the message backwards is a real competition trick.

    Found by running the tool against the published National Cipher
    Challenge archive rather than against anything invented here. The 2018
    challenge 8A ciphertext decrypted correctly and then read:

        MQYRAMNIAGAREHTEGOTKROWOTOTINUTROPPOEHTEVAHEWEPOHYUO

    which is -1.999 per letter with 31 per cent word coverage, so the tool
    called it `weak` and moved on. Backwards it is -1.176 with 73 per cent:

        OUY HOPE WE HAVE THE OPPORTUNIT[Y] TO WORK TOGETHER AGAIN MARY

    The decryption was right and the answer was on the screen; nothing
    thought to read it the other way round. Reversing a candidate costs one
    extra score, which is nothing against the searches that produced it.
    """

    def test_a_reversed_message_is_read_the_right_way_round(self) -> None:
        plain = sample_plaintext(400)
        ciphertext = caesar.encrypt(plain[::-1], 7)
        best = auto_solve(ciphertext, effort="fast", top=3, seed=1).candidates.best()
        self.assertEqual(best.plaintext, plain)
        self.assertIn("reversed", best.method.lower())

    def test_it_says_the_text_was_reversed(self) -> None:
        plain = sample_plaintext(400)
        ciphertext = caesar.encrypt(plain[::-1], 7)
        best = auto_solve(ciphertext, effort="fast", top=3, seed=1).candidates.best()
        self.assertTrue(best.diagnostics.get("plaintext_reversed"))

    def test_ordinary_messages_are_not_turned_around(self) -> None:
        """The guard: a message that reads correctly must stay that way."""
        plain = sample_plaintext(400)
        best = auto_solve(caesar.encrypt(plain, 7), effort="fast", top=3,
                          seed=1).candidates.best()
        self.assertEqual(best.plaintext, plain)
        self.assertNotIn("reversed", best.method.lower())


class TestEmbeddedNonProseBlock(unittest.TestCase):
    """A message that is part prose and part something else.

    From the 2017 challenge 5A, which the toolkit decrypted perfectly and
    reported as weak. 900 letters of English, 1,500 of a steganographic
    frieze enciphered with the words, 300 more of English.

    The block does not merely spoil the verdict, it spoils the SEARCH: a key
    that is wrong everywhere scored -1.52 per letter against the CORRECT
    key's -2.07, because the correct one has to carry the tiles. Measured,
    the climber returned JUDIE/GUNE/OWT for JODIE/GONE/BUT no matter how
    many restarts it was given -- and solved it immediately once the block
    was out of the way.
    """

    @staticmethod
    def _message() -> tuple[str, str]:
        from cipher_tool import substitution

        prose = sample_plaintext(1200)
        plain = prose[:900] + "WWWB" * 375 + prose[900:1200]
        key = substitution.SubstitutionKey.from_alphabet(
            "QWERTYUIOPASDFGHJKLZXCVBNM")
        return plain, substitution.encrypt(plain, key)

    def test_the_prose_is_recovered(self) -> None:
        plain, ciphertext = self._message()
        best = auto_solve(ciphertext, effort="fast", top=1,
                          seed=1).candidates.best()
        self.assertIn(plain[:60], best.plaintext)

    def test_it_says_a_block_was_set_aside(self) -> None:
        _plain, ciphertext = self._message()
        best = auto_solve(ciphertext, effort="fast", top=1,
                          seed=1).candidates.best()
        self.assertIn("non_prose_block", best.diagnostics)

    def test_an_ordinary_message_is_untouched(self) -> None:
        from cipher_tool import caesar

        plain = sample_plaintext(600)
        best = auto_solve(caesar.encrypt(plain, 5), effort="fast", top=1,
                          seed=1).candidates.best()
        self.assertEqual(best.plaintext, plain)
        self.assertNotIn("non_prose_block", best.diagnostics)


class TestEveryStageToleratesATimeBudget(unittest.TestCase):
    """Every stage must survive being handed a clock.

    auto_solve gives each stage a share of the remaining time whenever the
    caller sets max_time. A solver that refuses the argument is dropped and
    its whole family goes unsearched -- silently, from the user's side.

    MEASURED: `polybius.solve_unknown_square` rejected `time_budget` with a
    ValueError, and auto_solve only retries on TypeError, so the stage never
    ran under a budget at all. The 2017 challenge 3A -- a 4,846-letter
    Polybius written in Roman numerals -- solves at `strong` in seconds when
    that stage runs, and came back `unlikely` in the pipeline because it did
    not. It passed every test I had, because none of them set a budget.
    """

    def test_no_stage_refuses_time_budget(self) -> None:
        from cipher_tool.scoring import default_scorer

        engine = default_scorer()
        text = normalize(sample_plaintext(300))
        for stage in build_stages("deep", 3, 1):
            options = dict(stage.options)
            options["time_budget"] = 2.0
            with self.subTest(stage=stage.name):
                try:
                    stage.run(text, scorer=engine, **options)
                except TypeError:
                    # auto_solve retries without the argument, so this is
                    # survivable -- ValueError is not.
                    pass


def digit_bearing(letters: str, every: int = 3, digit: str = "7") -> str:
    """*letters* with a digit dropped in after every *every* letters."""
    out: list[str] = []
    for index, character in enumerate(letters, start=1):
        out.append(character)
        if not index % every:
            out.append(digit)
    return "".join(out)


class TestMateriality(unittest.TestCase):
    """When is the letters-only view an unfair reading of the paste?

    MEASURED 2026-08-18 over the 40 official archive ciphertexts: digit
    fraction 0.0000 on every one of them. No positive threshold can move the
    scoreboard, so the only question these answer is how tolerant to be of a
    date or a page number inside an ordinary message.
    """

    def test_a_card_stream_is_material(self) -> None:
        self.assertTrue(non_letters_are_material(normalize("7CX S3H6" * 200)))

    def test_an_ordinary_message_carrying_a_date_is_not(self) -> None:
        # 900 letters and 8 digits: 0.9 per cent, and fewer digits than the
        # count floor. This must take the letters path exactly as today.
        text = sample_plaintext(900) + " 12 03 19 07"
        self.assertFalse(non_letters_are_material(normalize(text)))

    def test_exactly_at_the_fraction_threshold(self) -> None:
        # 190 letters and 10 digits is 5.0 per cent of 200 symbols.
        material = "A" * 190 + "1" * 10
        self.assertTrue(non_letters_are_material(normalize(material)))

    def test_one_digit_below_the_count_threshold(self) -> None:
        self.assertFalse(non_letters_are_material(normalize("A" * 191 + "1" * 9)))

    def test_four_per_cent_of_a_thousand_symbols_is_not_material(self) -> None:
        self.assertFalse(
            non_letters_are_material(normalize("A" * 960 + "1" * 40))
        )

    def test_a_hand_built_normalized_text_is_never_material(self) -> None:
        """The designed null case: a zeroed inventory means NOT MEASURED.

        Anything that builds a NormalizedText without going through
        normalize() gets zeroes, and must then behave exactly as the toolkit
        did before any of this existed.
        """
        legacy = NormalizedText("A1B2", "AB", (0, 2), ("AB",))
        self.assertFalse(non_letters_are_material(legacy))


def one_candidate(normalized, *, scorer=None, **options) -> CandidateSet:
    """A stand-in solver: always returns exactly one candidate."""
    return CandidateSet([
        Candidate(method="stand-in", key="none", score=-1.0,
                  plaintext="THISISNOTAREADINGOFANYTHING"),
    ])


class TestALettersOnlyReadingOfASymbolStreamCannotClaim(unittest.TestCase):
    """The second, independent guard behind the routing fix.

    The paste screen refuses; this is what stops the LIBRARY and the `auto`
    command handing back a confident monoalphabetic reading of a message
    whose digits were never in the search.
    """

    def setUp(self) -> None:
        self.text = digit_bearing(caesar.encrypt(sample_plaintext(300), 3))

    def _run(self, reads: str):
        stage = Stage("stand-in", "monoalphabetic", "fast", 1.0,
                      one_candidate, {}, reads=reads)
        return auto_solve(self.text, effort="fast", top=5, stages=[stage])

    def test_a_letters_stage_is_capped_and_says_why(self) -> None:
        result = self._run("letters")
        self.assertTrue(result.candidates.ranked())
        for candidate in result.candidates.ranked():
            self.assertEqual(candidate.diagnostics["confidence_cap"], "weak")
            self.assertIn("digits", candidate.diagnostics["discarded_symbols"])
            self.assertEqual(candidate.confidence(), "weak")

    def test_a_symbols_stage_is_not_capped(self) -> None:
        """A stage that DID read the digits must not be punished for them."""
        result = self._run("symbols")
        self.assertTrue(result.candidates.ranked())
        for candidate in result.candidates.ranked():
            self.assertNotIn("confidence_cap", candidate.diagnostics)
            self.assertNotIn("discarded_symbols", candidate.diagnostics)

    def test_a_letters_only_message_is_untouched(self) -> None:
        stage = Stage("stand-in", "monoalphabetic", "fast", 1.0,
                      one_candidate, {}, reads="letters")
        result = auto_solve(caesar.encrypt(sample_plaintext(300), 3),
                            effort="fast", top=5, stages=[stage])
        for candidate in result.candidates.ranked():
            self.assertNotIn("confidence_cap", candidate.diagnostics)

    def test_the_stage_table_agrees_with_the_candidates(self) -> None:
        """One screen must not say two things.

        The stage report is written inside the loop and the cap is applied
        after it, so the table said `promising` about a candidate the ranking
        below it called `weak`. Observed on the real message: `substitution
        1.14s 2 candidate(s), best promising` above a candidate whose
        confidence line read `weak`.
        """
        result = self._run("letters")
        for report in result.stages:
            if report.ran and report.candidates:
                self.assertEqual(report.best_confidence, "weak")

    def test_every_stage_records_what_it_read(self) -> None:
        result = self._run("symbols")
        for candidate in result.candidates.ranked():
            self.assertEqual(candidate.diagnostics["reads"], "symbols")


class TestTheStructureReportReachesTheAutoReport(unittest.TestCase):
    def test_a_card_stream_is_described_not_solved(self) -> None:
        import random

        generator = random.Random(1)
        stream = "".join(
            generator.choice("23456789XJQKA") + generator.choice("CDHS")
            for _ in range(300)
        )
        stage = Stage("stand-in", "monoalphabetic", "fast", 1.0,
                      one_candidate, {}, reads="letters")
        result = auto_solve(stream, effort="fast", top=3, stages=[stage])
        self.assertIsNotNone(result.structure)
        self.assertTrue(result.structure.detected)
        self.assertIn("playing-card deck", result.render())

    def test_ordinary_english_gets_no_structure_claim(self) -> None:
        stage = Stage("stand-in", "monoalphabetic", "fast", 1.0,
                      one_candidate, {}, reads="letters")
        result = auto_solve(sample_plaintext(400), effort="fast", top=3,
                            stages=[stage])
        self.assertFalse(result.structure.detected)
        self.assertNotIn("paired alphabet", result.render())


class TestTheHomophonicStageIsInThePlan(unittest.TestCase):
    def test_it_runs_from_fast(self) -> None:
        names = {stage.name for stage in build_stages("fast", 5, None)}
        self.assertIn("homophonic", names)

    def test_it_is_marked_as_reading_the_symbol_stream(self) -> None:
        stages = {stage.name: stage for stage in build_stages("fast", 5, None)}
        self.assertEqual(stages["homophonic"].reads, "symbols")
        for name in ("encodings", "Polybius (unknown square)"):
            self.assertEqual(stages[name].reads, "symbols")
        self.assertEqual(stages["Caesar"].reads, "letters")


if __name__ == "__main__":
    unittest.main()
