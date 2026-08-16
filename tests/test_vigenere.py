"""Tests for the Vigenere cipher and its cryptanalysis.

The interesting tests here are the ones that pin down *observed* behaviour
rather than round trips:

* ``test_chi_squared_near_miss_is_repaired_by_refinement`` records a real
  failure of the per-column chi-squared attack on a 200-letter ciphertext and
  checks that the whole-text refinement pass repairs it. That is the whole
  justification for the refinement pass existing.
* ``test_column_shift_fits_agrees_with_chi_squared_english`` checks the
  rotated-counts optimisation against the straightforward implementation in
  ``statistics``, because a fast path that quietly disagrees with the slow one
  is worse than no fast path.
* The ``FailureMode`` class asserts that the toolkit stays honest when it
  cannot solve something: random letters, a wrong supplied key, a forced wrong
  key length and a text too short for its key must all come back weak.
"""

from __future__ import annotations

import random
import unittest
from pathlib import Path

from cipher_tool.normalize import ALPHABET, letters_only, normalize
from cipher_tool.statistics import chi_squared_english
from cipher_tool.vigenere import (
    MAXIMUM_BRUTE_FORCE_LENGTH,
    best_shift_for_column,
    brute_force_length,
    column_fit_analysis,
    column_shift_fits,
    decrypt,
    decrypt_with_key,
    describe_key_lengths,
    encrypt,
    estimate_key_lengths,
    ic_analysis,
    kasiski_analysis,
    refine_key,
    render_tabula_recta,
    solve,
    solve_key_for_length,
    tabula_recta,
)

DATA = Path(__file__).resolve().parents[1] / "src" / "cipher_tool" / "data"
CORPUS = DATA / "corpus_04_expository.txt"


def sample_plaintext(count: int) -> str:
    """The first *count* letters of the expository corpus file."""
    return letters_only(CORPUS.read_text(encoding="utf-8"))[:count]


class TestEncryptDecrypt(unittest.TestCase):
    """Hand-computed vectors and the algebra that has to hold around them."""

    def test_known_pair_worked_by_hand(self) -> None:
        # ATTACKATDAWN under LEMON, one letter at a time:
        #   A+L = 0+11 = 11 -> L      K+L = 10+11 = 21 -> V
        #   T+E = 19+4  = 23 -> X     A+E = 0+4   =  4 -> E
        #   T+M = 19+12 = 31 ->  5 F  T+M = 19+12 = 31 ->  5 F
        #   A+O = 0+14  = 14 -> O     D+O = 3+14  = 17 -> R
        #   C+N = 2+13  = 15 -> P     A+N = 0+13  = 13 -> N
        #   W+L = 22+11 = 33 ->  7 H  N+E = 13+4  = 17 -> R
        self.assertEqual(encrypt("ATTACKATDAWN", "LEMON"), "LXFOPVEFRNHR")
        self.assertEqual(decrypt("LXFOPVEFRNHR", "LEMON"), "ATTACKATDAWN")

    def test_known_pair_single_letter_key_is_a_caesar(self) -> None:
        # A one-letter key adds the same shift everywhere, so B means +1.
        self.assertEqual(encrypt("ATTACKATDAWN", "B"), "BUUBDLBUEBXO")
        self.assertEqual(decrypt("BUUBDLBUEBXO", "B"), "ATTACKATDAWN")

    def test_wraparound_past_z(self) -> None:
        # Z+Z = 25+25 = 50, and 50 mod 26 = 24 -> Y.
        self.assertEqual(encrypt("ZZZ", "Z"), "YYY")
        self.assertEqual(decrypt("YYY", "Z"), "ZZZ")

    def test_all_a_key_is_the_identity(self) -> None:
        """A key of A adds zero. Worth asserting: it is what a bug looks like."""
        plain = sample_plaintext(120)
        for key in ("A", "AA", "AAAAAAA"):
            self.assertEqual(encrypt(plain, key), plain)
            self.assertEqual(decrypt(plain, key), plain)

    def test_round_trip_for_several_keys(self) -> None:
        plain = sample_plaintext(300)
        for key in ("B", "ZQ", "SKY", "LEMON", "FALCON", "WINTERGALE", "X" * 19):
            with self.subTest(key=key):
                self.assertEqual(decrypt(encrypt(plain, key), key), plain)

    def test_decrypt_with_key_is_the_same_function(self) -> None:
        cipher = encrypt(sample_plaintext(80), "LEMON")
        self.assertEqual(decrypt_with_key(cipher, "LEMON"), decrypt(cipher, "LEMON"))

    def test_key_longer_than_the_text_only_uses_its_prefix(self) -> None:
        self.assertEqual(encrypt("AB", "BCDEFGHIJ"), "BD")

    def test_input_robustness(self) -> None:
        """Case, punctuation, grouping and line breaks must not change a thing."""
        expected = encrypt("ATTACKATDAWN", "LEMON")
        for variant in (
            "attackatdawn",
            "Attack at dawn!",
            "ATTAC KATDA WN",
            "ATTAC KATDA\nWN\n",
            "  a t t a c k, a t   d a w n.  ",
            "ATTACK-AT-DAWN (1917)",
        ):
            with self.subTest(variant=variant):
                self.assertEqual(encrypt(variant, "LEMON"), expected)

    def test_key_is_cleaned_the_same_way_as_text(self) -> None:
        expected = encrypt("ATTACKATDAWN", "LEMON")
        for key in ("lemon", "Le Mon", "L-E-M-O-N", " lemon! "):
            with self.subTest(key=key):
                self.assertEqual(encrypt("ATTACKATDAWN", key), expected)

    def test_empty_input_does_not_crash(self) -> None:
        self.assertEqual(encrypt("", "LEMON"), "")
        self.assertEqual(decrypt("", "LEMON"), "")
        self.assertEqual(encrypt("12345 !?", "LEMON"), "")

    def test_invalid_keys_raise_value_error_with_a_message(self) -> None:
        for bad in ("", "   ", "1234", "!!!", "\n"):
            with self.subTest(key=bad):
                with self.assertRaises(ValueError) as caught:
                    encrypt("ATTACKATDAWN", bad)
                self.assertTrue(str(caught.exception).strip())
                self.assertIn("no letters", str(caught.exception))
                with self.assertRaises(ValueError):
                    decrypt("ATTACKATDAWN", bad)

    def test_non_string_key_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as caught:
            encrypt("ATTACKATDAWN", 5)  # type: ignore[arg-type]
        self.assertIn("int", str(caught.exception))


class TestTabulaRecta(unittest.TestCase):
    def test_shape_and_rows(self) -> None:
        square = tabula_recta()
        self.assertEqual(len(square), 26)
        self.assertEqual(square[0], ALPHABET)
        self.assertEqual(square[1], "BCDEFGHIJKLMNOPQRSTUVWXYZA")
        self.assertEqual(square[25], "ZABCDEFGHIJKLMNOPQRSTUVWXY")
        for row in square:
            self.assertEqual(len(row), 26)
            self.assertEqual(set(row), set(ALPHABET))

    def test_square_agrees_with_encrypt(self) -> None:
        """Row = key letter, column = plaintext letter, cell = ciphertext."""
        square = tabula_recta()
        for key_index, key_letter in enumerate(ALPHABET):
            for plain_index, plain_letter in enumerate(ALPHABET):
                self.assertEqual(
                    square[key_index][plain_index],
                    encrypt(plain_letter, key_letter),
                )

    def test_rendered_square_is_ascii_and_labelled(self) -> None:
        rendered = render_tabula_recta()
        self.assertTrue(rendered.isascii())
        self.assertEqual(len(rendered.splitlines()), 28)  # header, rule, 26 rows


class TestColumnFitting(unittest.TestCase):
    def test_column_shift_fits_agrees_with_chi_squared_english(self) -> None:
        """The rotated-counts fast path must match the plain implementation.

        ``column_shift_fits`` never builds the 26 decrypted strings; it rotates
        a 26-entry count vector instead. If that arithmetic were wrong every
        key-length estimate in the module would be quietly wrong too, so it is
        checked against ``statistics.chi_squared_english`` on real text.
        """
        column = encrypt(sample_plaintext(200), "Q")
        fits = column_shift_fits(column)
        self.assertEqual(len(fits), 26)
        for shift, letter in enumerate(ALPHABET):
            with self.subTest(shift=shift):
                slow = chi_squared_english(decrypt(column, letter))
                self.assertAlmostEqual(fits[shift], slow, places=9)

    def test_best_shift_recovers_a_known_caesar(self) -> None:
        column = encrypt(sample_plaintext(300), "Q")
        shift, best, average = best_shift_for_column(column)
        self.assertEqual(shift, ord("Q") - 65)
        self.assertLess(best, average)
        # The winner should be far better than the field, not marginally so.
        self.assertLess(best / average, 0.2)

    def test_empty_column_is_infinitely_bad_rather_than_zero(self) -> None:
        fits = column_shift_fits("")
        self.assertEqual(len(fits), 26)
        self.assertTrue(all(value == float("inf") for value in fits))


class TestKeyLengthEstimation(unittest.TestCase):
    """Each of the three methods, then the three combined."""

    def setUp(self) -> None:
        self.plain = sample_plaintext(600)
        self.key = "HARBOUR"  # seven letters, no repeats
        self.cipher = encrypt(self.plain, self.key)

    def test_kasiski_finds_the_key_length_and_shows_its_working(self) -> None:
        report = kasiski_analysis(self.cipher)
        self.assertEqual(report.best_length(), len(self.key))
        winner = report.candidates[0]
        self.assertTrue(winner.supporting, "no supporting repeats were reported")
        for gap in winner.gaps:
            self.assertEqual(gap % winner.length, 0)
        # Every supporting repeat must actually appear twice in the ciphertext.
        for repeat in winner.supporting:
            self.assertGreaterEqual(self.cipher.count(repeat.text), 2)
        self.assertIn("Kasiski", report.describe())

    def test_kasiski_demotes_multiples_of_a_winner(self) -> None:
        report = kasiski_analysis(self.cipher)
        for candidate in report.candidates:
            if candidate.length % len(self.key) == 0 and candidate.length != len(
                self.key
            ):
                self.assertIsNotNone(candidate.demoted_of)

    def test_ic_analysis_peaks_at_the_key_length(self) -> None:
        rows = ic_analysis(self.cipher)
        by_period = {row.period: row for row in rows}
        self.assertIn(len(self.key), by_period)
        english_like = [
            row.period for row in rows if row.mean_ic > 0.06
        ]
        self.assertIn(len(self.key), english_like)
        # Wrong periods must sit near the flat-random value, not near English.
        self.assertLess(by_period[2].mean_ic, 0.055)
        self.assertLess(by_period[3].mean_ic, 0.055)
        # Only periods with enough letters per column are reported at all.
        for row in rows:
            self.assertGreaterEqual(row.shortest_column, 20)

    def test_column_fit_prefers_the_key_length(self) -> None:
        rows = {row.period: row for row in column_fit_analysis(self.cipher)}
        right = rows[len(self.key)]
        self.assertEqual(right.key_guess, self.key)
        self.assertLess(right.mean_ratio, 0.15)
        self.assertLess(right.worst_ratio, 0.25)
        for period, row in rows.items():
            if period % len(self.key):
                with self.subTest(period=period):
                    self.assertGreater(row.mean_ratio, right.mean_ratio)

    def test_estimate_ranks_the_true_length_first(self) -> None:
        evidence = estimate_key_lengths(self.cipher)
        self.assertEqual(evidence[0].length, len(self.key))
        self.assertEqual(evidence[0].rank, 1)
        self.assertIsNone(evidence[0].demoted_of)
        self.assertGreater(evidence[0].combined, evidence[1].combined)
        self.assertEqual(
            set(evidence[0].evidence_available), {"kasiski", "ic", "column-fit"}
        )

    def test_multiples_are_demoted_not_presented_as_findings(self) -> None:
        """14 explains the text as well as 7 does, because 7 divides it."""
        evidence = estimate_key_lengths(self.cipher)
        by_length = {row.length: row for row in evidence}
        self.assertEqual(by_length[14].demoted_of, 7)
        # Demoted rows sort below every row that is not demoted.
        first_demoted = min(
            row.rank for row in evidence if row.demoted_of is not None
        )
        last_clean = max(row.rank for row in evidence if row.demoted_of is None)
        self.assertGreater(first_demoted, last_clean)

    def test_short_text_reports_missing_evidence_rather_than_inventing_it(self) -> None:
        cipher = encrypt(sample_plaintext(60), "SKY")
        evidence = estimate_key_lengths(cipher, max_key_length=20)
        self.assertTrue(evidence)
        for row in evidence:
            # 60 letters cannot support an IC measurement at any useful period.
            if row.mean_ic is None:
                self.assertIsNone(row.ic_distance)
                self.assertNotIn("ic", row.evidence_available)
        self.assertTrue(any(row.mean_ic is None for row in evidence))

    def test_describe_renders_a_table(self) -> None:
        evidence = estimate_key_lengths(self.cipher)
        text = describe_key_lengths(evidence)
        self.assertTrue(text.isascii())
        self.assertIn("Candidate key lengths", text)
        self.assertIn("demoted", text)
        self.assertIn(evidence[0].describe(), text)

    def test_empty_input_gives_no_evidence(self) -> None:
        self.assertEqual(estimate_key_lengths(""), [])
        self.assertEqual(describe_key_lengths([]), (
            "No key-length evidence: the input contains no letters."
        ))

    def test_bad_bounds_raise_value_error(self) -> None:
        for bad in (0, -1):
            with self.subTest(bound=bad):
                with self.assertRaises(ValueError) as caught:
                    estimate_key_lengths(self.cipher, bad)
                self.assertTrue(str(caught.exception).strip())


class TestKeyRecovery(unittest.TestCase):
    def test_chi_squared_alone_recovers_a_long_text_key(self) -> None:
        cipher = encrypt(sample_plaintext(600), "HARBOUR")
        self.assertEqual(solve_key_for_length(cipher, 7), "HARBOUR")

    def test_chi_squared_near_miss_is_repaired_by_refinement(self) -> None:
        """The observed failure the refinement pass exists to fix.

        On 200 letters a seven-letter key leaves columns of under thirty
        letters, and chi-squared picks the wrong letter for one of them. The
        whole-text n-gram pass sees the damage that one wrong key letter does
        to every seventh letter of the plaintext and repairs it.
        """
        plain = sample_plaintext(200)
        cipher = encrypt(plain, "KESTREL")

        chi_key = solve_key_for_length(cipher, 7)
        self.assertNotEqual(chi_key, "KESTREL", "chi-squared unexpectedly right")
        wrong = sum(1 for a, b in zip(chi_key, "KESTREL") if a != b)
        self.assertEqual(wrong, 1)

        refinement = refine_key(cipher, chi_key)
        self.assertEqual(refinement.key, "KESTREL")
        self.assertEqual(refinement.changes, 1)
        self.assertGreater(refinement.score_after, refinement.score_before)
        self.assertEqual(decrypt(cipher, refinement.key), plain)
        self.assertIn("->", refinement.describe())

    def test_refinement_leaves_a_correct_key_alone(self) -> None:
        cipher = encrypt(sample_plaintext(400), "FALCON")
        refinement = refine_key(cipher, "FALCON")
        self.assertEqual(refinement.key, "FALCON")
        self.assertEqual(refinement.changes, 0)
        self.assertIn("unchanged", refinement.describe())

    def test_solve_key_for_length_can_refine_in_one_call(self) -> None:
        cipher = encrypt(sample_plaintext(200), "KESTREL")
        self.assertEqual(solve_key_for_length(cipher, 7, refine=True), "KESTREL")

    def test_invalid_lengths_raise_value_error(self) -> None:
        cipher = encrypt(sample_plaintext(100), "SKY")
        for bad in (0, -3):
            with self.subTest(length=bad):
                with self.assertRaises(ValueError) as caught:
                    solve_key_for_length(cipher, bad)
                self.assertIn("at least 1", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            solve_key_for_length(cipher, 500)
        self.assertIn("exceeds", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            solve_key_for_length("", 3)
        self.assertTrue(str(caught.exception).strip())

    def test_refine_rejects_an_empty_key(self) -> None:
        cipher = encrypt(sample_plaintext(100), "SKY")
        with self.assertRaises(ValueError) as caught:
            refine_key(cipher, "1234")
        self.assertIn("no letters", str(caught.exception))


class TestBruteForce(unittest.TestCase):
    def test_exhaustive_search_finds_a_two_letter_key(self) -> None:
        plain = sample_plaintext(150)
        found = brute_force_length(encrypt(plain, "ZQ"), 2)
        self.assertEqual(found.keys_possible, 676)
        self.assertEqual(found.keys_tried, 676)
        self.assertTrue(found.exhaustive)
        self.assertEqual(found.best[0][0], "ZQ")

    def test_exhaustive_search_finds_a_three_letter_key(self) -> None:
        plain = sample_plaintext(300)
        found = brute_force_length(encrypt(plain, "XYZ"), 3)
        self.assertEqual(found.keys_possible, 17576)
        self.assertEqual(found.best[0][0], "XYZ")

    def test_long_keys_are_refused_with_an_explanation(self) -> None:
        """26**5 is not a search, it is a hang. Say so instead of starting it."""
        with self.assertRaises(ValueError) as caught:
            brute_force_length("ABCDEFGH", MAXIMUM_BRUTE_FORCE_LENGTH + 1)
        message = str(caught.exception)
        self.assertIn("refusing", message)
        self.assertIn("11,881,376", message)

    def test_empty_input_is_handled(self) -> None:
        found = brute_force_length("", 2)
        self.assertEqual(found.keys_tried, 0)
        self.assertEqual(found.best, ())


class TestSolve(unittest.TestCase):
    def test_recovers_key_and_plaintext_from_statistics(self) -> None:
        plain = sample_plaintext(400)
        cipher = encrypt(plain, "FALCON")
        result = solve(cipher, brute_force_up_to=0)
        best = result.best()
        self.assertIsNotNone(best)
        self.assertEqual(best.plaintext, plain)
        self.assertEqual(best.key, "key=FALCON")
        self.assertEqual(best.confidence(), "strong")
        self.assertGreater(best.diagnostics["word_coverage"], 0.7)
        self.assertEqual(best.diagnostics["key_length"], 6)
        self.assertEqual(best.diagnostics["length_rank"], 1)
        self.assertIn("kasiski_votes", best.diagnostics)
        self.assertIn("mean_column_ic", best.diagnostics)
        self.assertIn("column_fit_ratio", best.diagnostics)

    def test_reports_several_key_lengths_not_one_answer(self) -> None:
        cipher = encrypt(sample_plaintext(400), "FALCON")
        result = solve(cipher, brute_force_up_to=0, lengths_to_try=4, top=10)
        self.assertGreaterEqual(len(result), 3)
        lengths = {candidate.diagnostics["key_length"] for candidate in result}
        self.assertGreaterEqual(len(lengths), 3)
        self.assertIsNotNone(result.score_gap())

    def test_short_text_is_solved_by_the_exhaustive_search(self) -> None:
        """90 letters and a two-letter key: statistics thin, brute force fine."""
        plain = sample_plaintext(90)
        cipher = encrypt(plain, "ZQ")
        result = solve(cipher, brute_force_up_to=2)
        best = result.best()
        self.assertEqual(best.plaintext, plain)
        self.assertEqual(best.key, "key=ZQ")

    def test_supplied_key_skips_the_search(self) -> None:
        plain = sample_plaintext(200)
        cipher = encrypt(plain, "LEMON")
        result = solve(cipher, key="le mon!")
        self.assertEqual(len(result), 1)
        best = result.best()
        self.assertEqual(best.plaintext, plain)
        self.assertEqual(best.key, "key=LEMON")
        self.assertIn("supplied", best.diagnostics["search"])

    def test_forced_key_length_is_honoured(self) -> None:
        plain = sample_plaintext(400)
        cipher = encrypt(plain, "FALCON")
        result = solve(cipher, key_length=6, brute_force_up_to=0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.best().plaintext, plain)

    def test_identity_key_is_flagged(self) -> None:
        """Plaintext in, plaintext out: say so rather than claiming a solve."""
        plain = sample_plaintext(300)
        result = solve(plain, key="AAAA")
        best = result.best()
        self.assertEqual(best.plaintext, plain)
        self.assertIn("identity_key", best.diagnostics)

    def test_accepts_normalized_text_and_preserves_layout(self) -> None:
        plain = sample_plaintext(200)
        original = " ".join(
            encrypt(plain, "LEMON")[i : i + 5]
            for i in range(0, 200, 5)
        )
        result = solve(normalize(original), brute_force_up_to=0)
        best = result.best()
        self.assertEqual(best.plaintext, plain)
        self.assertIsNotNone(best.display)
        self.assertEqual(len(best.display), len(original))
        # Spacing of the original survives; only the letters changed.
        for index, character in enumerate(original):
            if character == " ":
                self.assertEqual(best.display[index], " ")

    def test_empty_input_returns_an_empty_candidate_set(self) -> None:
        self.assertEqual(len(solve("")), 0)
        self.assertEqual(len(solve("1234 !!")), 0)
        self.assertIsNone(solve("").best())

    def test_time_budget_is_respected_and_recorded(self) -> None:
        cipher = encrypt(sample_plaintext(600), "HARBOUR")
        result = solve(cipher, time_budget=0.0)
        # One key length is always attempted, so there is still an answer...
        self.assertGreaterEqual(len(result), 1)
        # ...and every candidate says the search was cut short.
        for candidate in result:
            self.assertTrue(candidate.diagnostics["time_budget_hit"])

    def test_bad_options_raise_value_error(self) -> None:
        cipher = encrypt(sample_plaintext(200), "SKY")
        with self.assertRaises(ValueError) as caught:
            solve(cipher, brute_force_up_to=5)
        self.assertIn("11,881,376", str(caught.exception))
        with self.assertRaises(ValueError):
            solve(cipher, brute_force_up_to=-1)
        with self.assertRaises(ValueError) as caught:
            solve(cipher, key_length=0)
        self.assertIn("at least 1", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            solve(cipher, key_length=5000)
        self.assertIn("exceeds", str(caught.exception))
        with self.assertRaises(ValueError):
            solve(cipher, max_key_length=0)
        with self.assertRaises(ValueError) as caught:
            solve(cipher, key="!!!")
        self.assertIn("no letters", str(caught.exception))


class TestFailureMode(unittest.TestCase):
    """What the toolkit does when it cannot solve something.

    Every one of these would pass silently if the module returned its best
    guess with a confident label attached, which is exactly the behaviour that
    would waste a competition afternoon.
    """

    def test_random_letters_are_not_reported_as_solved(self) -> None:
        generator = random.Random(20260816)
        junk = "".join(generator.choice(ALPHABET) for _ in range(400))
        result = solve(junk, max_key_length=8, brute_force_up_to=1)
        best = result.best()
        self.assertIsNotNone(best)
        self.assertIn(best.confidence(), {"weak", "unlikely"})
        self.assertLess(best.diagnostics["word_coverage"], 0.4)
        self.assertLess(best.diagnostics["normalised_score"], -2.0)

    def test_wrong_supplied_key_is_reported_honestly(self) -> None:
        plain = sample_plaintext(400)
        cipher = encrypt(plain, "FALCON")
        result = solve(cipher, key="WRONGKEY")
        best = result.best()
        self.assertNotEqual(best.plaintext, plain)
        self.assertIn(best.confidence(), {"weak", "unlikely"})
        self.assertLess(best.diagnostics["word_coverage"], 0.4)

    def test_forcing_the_wrong_key_length_produces_no_confident_answer(self) -> None:
        plain = sample_plaintext(400)
        cipher = encrypt(plain, "FALCON")
        result = solve(cipher, key_length=5, brute_force_up_to=0)
        best = result.best()
        self.assertNotEqual(best.plaintext, plain)
        self.assertIn(best.confidence(), {"weak", "unlikely"})
        self.assertLess(best.diagnostics["word_coverage"], 0.5)

    def test_text_too_short_for_its_key_is_not_claimed_as_solved(self) -> None:
        """80 letters and a ten-letter key: eight letters per column.

        The refinement pass gets most of the key and stalls in a local
        maximum. The plaintext is wrong, and the confidence label must not say
        strong.
        """
        plain = sample_plaintext(80)
        cipher = encrypt(plain, "WINTERGALE")
        result = solve(cipher, max_key_length=12, brute_force_up_to=0)
        best = result.best()
        self.assertNotEqual(best.plaintext, plain)
        self.assertNotEqual(best.confidence(), "strong")

    def test_a_warning_is_attached_when_columns_are_too_short(self) -> None:
        cipher = encrypt(sample_plaintext(60), "WINTERGALE")
        result = solve(cipher, key_length=10, brute_force_up_to=0)
        best = result.best()
        self.assertIn("warning", best.diagnostics)
        self.assertIn("too few", best.diagnostics["warning"])


if __name__ == "__main__":
    unittest.main()
