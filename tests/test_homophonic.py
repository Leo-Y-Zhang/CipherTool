"""Tests for the homophonic substitution family.

The load-bearing test in this file is not one of the recoveries. It is
:class:`TestTheSlotConstraintIsTheWholeAlgorithm`, which watches the
UNCONSTRAINED search collapse. Without that, "the search works" is a claim
about a search whose most important property has never been observed doing
anything -- and a constraint that is never seen mattering is decoration.
"""

from __future__ import annotations

import random
import unittest

from cipher_tool import candidates, homophonic
from cipher_tool.scoring import default_scorer
from tests.test_auto import sample_plaintext


def build(symbol_count: int, units: int, *, seed: int = 5,
          model: str = "uniform") -> tuple[str, list[str], dict[str, str]]:
    """A homophonic ciphertext of known plaintext, key and shape."""
    plaintext = sample_plaintext(units)
    key = homophonic.random_key(symbol_count, model=model, seed=seed)
    stream = homophonic.encrypt(plaintext, key, seed=seed)
    mapping = {
        token: letter for letter, tokens in key.items() for token in tokens
    }
    return plaintext, stream, mapping


def accuracy(recovered: str, plaintext: str) -> float:
    """Fraction of letters recovered exactly."""
    if not plaintext:
        return 0.0
    pairs = zip(recovered, plaintext)
    return sum(a == b for a, b in pairs) / len(plaintext)


class TestSlotMultiset(unittest.TestCase):
    def test_uniform_over_fifty_two_is_two_per_letter(self) -> None:
        slots = homophonic.slot_multiset(52)
        self.assertEqual(sorted(set(slots.values())), [2])
        self.assertEqual(sum(slots.values()), 52)

    def test_uniform_spends_the_remainder_on_the_commonest_letters(self) -> None:
        slots = homophonic.slot_multiset(36)
        self.assertEqual(sum(slots.values()), 36)
        self.assertEqual(slots["E"], 2)
        self.assertEqual(slots["Z"], 1)

    def test_frequency_model_gives_every_letter_at_least_one(self) -> None:
        slots = homophonic.slot_multiset(52, model="frequency")
        self.assertEqual(sum(slots.values()), 52)
        self.assertEqual(min(slots.values()), 1)
        self.assertGreater(slots["E"], slots["Z"])

    def test_an_unknown_model_is_refused_rather_than_guessed(self) -> None:
        with self.assertRaises(ValueError):
            homophonic.slot_multiset(52, model="vibes")


class TestEncryptDecryptRoundTrip(unittest.TestCase):
    def test_a_known_key_round_trips(self) -> None:
        plaintext, stream, mapping = build(52, 300)
        self.assertEqual(len(stream), 300)
        self.assertEqual(homophonic.decrypt(stream, mapping), plaintext)

    def test_the_same_letter_is_written_several_ways(self) -> None:
        """That is the whole point of the family, so assert it happened."""
        plaintext, stream, _ = build(52, 300)
        written_as = {
            token for token, letter in zip(stream, plaintext) if letter == "E"
        }
        self.assertGreater(len(written_as), 1)


class TestItRecoversTheMeasuredShapes(unittest.TestCase):
    """The two shapes the design was measured on, from a fixed seed."""

    def _recover(self, symbols: int, units: int, **options: object) -> float:
        plaintext, stream, _ = build(symbols, units)
        found = homophonic.solve(
            stream, scorer=default_scorer(), top=1, seed=3, **options
        )
        best = found.best()
        self.assertIsNotNone(best, f"notes: {getattr(found, 'notes', ())}")
        return accuracy(best.plaintext, plaintext)

    def test_fifty_two_symbols_from_four_hundred_units(self) -> None:
        # MEASURED 2026-08-18 at the default search size: 0.983 exact letters.
        # Only 44 of the 52 symbols occur in 400 units -- the rare letters'
        # symbols never come up -- so the multiset the search assumes is the
        # uniform one over 44, not the one the message was written with, and
        # it still recovers. That is the ordinary case for this family.
        self.assertGreaterEqual(self._recover(52, 400), 0.90)

    def test_thirty_six_symbols_from_three_hundred_and_thirteen(self) -> None:
        # MEASURED: 0.856 at six restarts, 0.901 at twelve. The shape is
        # thinner than it looks -- 33 of the 36 symbols occur -- and this is
        # the size of search it takes, which is why the number is here rather
        # than in a comment about the default.
        self.assertGreaterEqual(self._recover(36, 313, restarts=12), 0.90)

    def test_the_incremental_total_agrees_with_a_full_rescore(self) -> None:
        """A delta-maintained total is the classic place for a silent bug.

        Reported on every real run as ``score_audit_gap``, not only under
        test, so that a drift shows up in the field rather than in a suite
        somebody might not run.
        """
        plaintext, stream, _ = build(52, 400)
        found = homophonic.solve(
            stream, scorer=default_scorer(), top=1, seed=3
        )
        best = found.best()
        self.assertIsNotNone(best)
        self.assertLess(abs(best.diagnostics["score_audit_gap"]), 1e-6)


class TestTheSlotConstraintIsTheWholeAlgorithm(unittest.TestCase):
    """Watch the unconstrained search collapse, and watch where it does it.

    ``constrain_slots=False`` is a real mutation of the move set -- any symbol
    may take any letter -- not a constant read differently. It exists so this
    test can exist.

    MEASURED 2026-08-18, and NOT where the design expected. On genuine
    homophonic ciphertext the unconstrained search does no harm at all: at
    52/400, 52/600, 36/313, 40/320 and 52/340 it matched or beat the
    constrained one, reaching 1.000 exact letters where the constrained
    search reached 0.983. The collapse is real, but it happens on a stream
    that is NOT homophonic, which is the case that matters: that is when a
    search manufactures a reading of something it cannot read.

    On 600 units of uniformly random cards -- no plaintext, nothing to find:

        constrained     26 letters used, -2.179 per letter, `weak`
        UNCONSTRAINED    8 letters used, -1.248 per letter, `promising`
                        "SEEITSSESINILETSTATSSITSSITSSSTSEETSISASSITSLESS..."

    That is the five-letter-language failure the design describes, and the
    fixed multiset is what forbids it: a key that must spend a symbol on Z
    cannot become a key that only writes AEILNRST.

    The `promising` in that table is what the label WAS. There are now two
    independent defences and the tests below assert both: the constraint
    stops the collapse being searched for at all, and
    `candidates.looks_degenerate` stops a collapsed reading being sold even
    if one is produced. The second matters because the first is a property of
    this solver, while the scorer's willingness to reward a handful of
    repeated letters was toolkit-wide.
    """

    def setUp(self) -> None:
        self.engine = default_scorer()
        generator = random.Random(19)
        deck = homophonic.deck_tokens(52)
        self.noise = [generator.choice(deck) for _ in range(600)]
        self.plaintext, self.stream, _ = build(52, 400)

    def _best(self, stream: object, **options: object):
        found = homophonic.solve(
            stream, scorer=self.engine, top=1, seed=3, **options
        )
        best = found.best()
        self.assertIsNotNone(best, f"notes: {getattr(found, 'notes', ())}")
        return best

    def test_unconstrained_manufactures_a_reading_of_pure_noise(self) -> None:
        loose = self._best(self.noise, constrain_slots=False)
        self.assertLess(
            len(set(loose.plaintext)), 9,
            "the unconstrained search was expected to collapse onto a handful "
            f"of letters, and used {sorted(set(loose.plaintext))}",
        )
        # The collapse still happens -- that is what the assertion above
        # measures, and it is why the slot constraint exists. What has changed
        # is that the result is no longer SOLD: `candidates.looks_degenerate`
        # refuses to call a reading built from a handful of letters anything
        # better than weak, however well it scores. Before that guard this
        # same call returned `promising` on -1.248 per letter.
        self.assertEqual(loose.confidence(), "weak")
        self.assertIsNotNone(
            candidates.looks_degenerate(loose.plaintext),
            "the collapsed reading should be recognised as degenerate",
        )

    def test_constrained_calls_the_same_noise_what_it_is(self) -> None:
        tight = self._best(self.noise)
        self.assertEqual(len(set(tight.plaintext)), 26)
        self.assertEqual(tight.confidence(), "weak")

    def test_the_mutation_makes_the_nonsense_score_better(self) -> None:
        """The trap, stated as a number.

        The collapsed reading scores nearly a whole log unit per letter
        BETTER than the honest one. Score alone cannot tell them apart; only
        the constraint can, and only because it is applied before the search
        rather than judged after it.
        """
        tight = self._best(self.noise)
        loose = self._best(self.noise, constrain_slots=False)
        self.assertGreater(loose.normalised_score, tight.normalised_score)

    def test_the_constrained_search_still_recovers_real_ciphertext(self) -> None:
        tight = self._best(self.stream)
        self.assertGreaterEqual(accuracy(tight.plaintext, self.plaintext), 0.90)

    def test_the_mutation_is_recorded_in_the_diagnostics(self) -> None:
        loose = self._best(self.noise, constrain_slots=False)
        self.assertIn("DISABLED", loose.diagnostics["slot_constraint"])


class TestItRefusesBeforeDoingAnyWork(unittest.TestCase):
    """Every refusal returns an empty set and says why in ``notes``."""

    def _refusal(self, stream: object, **options: object):
        found = homophonic.solve(
            stream, scorer=default_scorer(), top=1, seed=3, **options
        )
        self.assertEqual(len(found), 0)
        self.assertTrue(getattr(found, "notes", ()), "a refusal must say why")
        return " ".join(found.notes)

    def test_twenty_six_or_fewer_symbols_is_the_substitution_solver_s_job(
        self,
    ) -> None:
        generator = random.Random(2)
        stream = [generator.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                  for _ in range(600)]
        self.assertIn("substitution", self._refusal(stream))

    def test_too_many_symbols_is_not_a_homophonic_key(self) -> None:
        stream = [f"T{index}" for index in range(homophonic.MAX_SYMBOLS + 1)]
        stream = stream * 40
        self.assertIn("symbols", self._refusal(stream))

    def test_too_few_units_outright(self) -> None:
        plaintext, stream, _ = build(52, 120)
        self.assertIn("units", self._refusal(stream))

    def test_too_few_units_per_symbol(self) -> None:
        # MEASURED: 52 symbols and 220 units gives 39 distinct symbols and
        # 5.6 units per symbol, below the floor of 6.0.
        _, stream, _ = build(52, 220)
        message = self._refusal(stream)
        self.assertIn("per symbol", message)

    def test_a_broken_pairing_is_refused_not_repaired(self) -> None:
        """One missing symbol makes every unit past it a fiction.

        Pairing on regardless invents cells that were never written -- on the
        real 1,251-symbol message, 96 distinct 'cards' out of a 52-card deck
        -- and then reports a reading of them. The recogniser found the break;
        this refuses on it and names the index.
        """
        _, stream, _ = build(52, 400)
        flat = "".join(stream)
        broken = flat[:401] + flat[402:]
        message = self._refusal(broken)
        self.assertIn("alternation", message.lower())


class TestItNeverRaisesOnATimeBudget(unittest.TestCase):
    """The trap that silently dropped a whole stage from every real run.

    auto_solve hands every stage a share of the clock and retries only on
    TypeError. A ValueError here means the stage is dropped, its family goes
    unsearched, and the report says nothing about it.
    """

    def test_zero_and_negative_budgets_return_an_empty_set(self) -> None:
        _, stream, _ = build(52, 400)
        for budget in (0, -1, 0.0):
            with self.subTest(budget=budget):
                found = homophonic.solve(stream, top=1, time_budget=budget)
                self.assertEqual(len(found), 0)

    def test_a_real_budget_is_honoured_not_refused(self) -> None:
        _, stream, _ = build(52, 400)
        found = homophonic.solve(stream, top=1, seed=3, time_budget=2.0,
                                 restarts=6, iterations=200_000)
        best = found.best()
        self.assertIsNotNone(best)
        self.assertTrue(best.diagnostics.get("time_budget_hit"))


class TestAThinCipherCannotClaimStrength(unittest.TestCase):
    def test_below_the_confident_regime_the_label_is_capped(self) -> None:
        # MEASURED: 52 symbols and 260 units gives 42 distinct symbols and
        # 6.19 units per symbol -- above the floor the search will run at,
        # below the 8.0 at which the measurements succeeded. The search has
        # more key freedom than the ciphertext can pay for, and must not be
        # able to say `strong` whatever it finds.
        _, stream, _ = build(52, 260)
        found = homophonic.solve(stream, scorer=default_scorer(), top=1,
                                 seed=3)
        best = found.best()
        self.assertIsNotNone(best, f"notes: {getattr(found, 'notes', ())}")
        self.assertEqual(best.diagnostics["confidence_cap"], "promising")
        self.assertNotEqual(best.confidence(), "strong")


class TestDiagnosticsSayWhatWasDone(unittest.TestCase):
    def test_the_search_names_itself_as_not_exhaustive(self) -> None:
        _, stream, _ = build(52, 400)
        best = homophonic.solve(stream, scorer=default_scorer(), top=1,
                                seed=3).best()
        self.assertIsNotNone(best)
        for name in ("symbols", "units", "units_per_symbol", "restarts",
                     "iterations", "slot_model", "unit_size", "search"):
            self.assertIn(name, best.diagnostics)
        self.assertIn("not exhaustive", best.diagnostics["search"])


if __name__ == "__main__":
    unittest.main()
