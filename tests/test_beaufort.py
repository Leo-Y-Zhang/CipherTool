"""Tests for the Beaufort and variant Beaufort module.

The hand-computed cases below were worked out on paper with a letter/number
table before any code was run, so they test the arithmetic rather than merely
agreeing with it. ATTACKATDAWN under LEMON is used throughout because its
Vigenere ciphertext (LXFOPVEFRNHR) is the standard textbook example, which
pins down the sign conventions of all three ciphers at once.
"""

from __future__ import annotations

import time
import unittest

from cipher_tool.beaufort import (
    BEAUFORT,
    VARIANT,
    _hill_climb,
    beaufort_decrypt,
    beaufort_encrypt,
    column_key_letter,
    decrypt,
    derive_key,
    encrypt,
    solve,
    variant_decrypt,
    variant_encrypt,
    vigenere_decrypt,
    vigenere_encrypt,
)
from cipher_tool.normalize import group_text, letters_only, normalize, to_numbers
from cipher_tool.scoring import DATA_DIR, default_scorer

# Build the order-3 table once, up front. It is built lazily on first use, and
# leaving that to happen inside a timed test would corrupt the measurement the
# time-budget test calibrates against.
default_scorer().table()


def corpus_letters(count: int) -> str:
    """The first *count* letters of the expository corpus file."""
    text = (DATA_DIR / "corpus_04_expository.txt").read_text(encoding="utf-8")
    letters = letters_only(text)
    if len(letters) < count:  # pragma: no cover - guards a broken checkout
        raise AssertionError(
            f"corpus_04_expository.txt has only {len(letters)} letters, "
            f"needed {count}"
        )
    return letters[:count]


class TestHandComputedCases(unittest.TestCase):
    """Known plaintext/key/ciphertext triples, computed by hand."""

    def test_beaufort_known_triple(self) -> None:
        # Key LEMON repeats as L E M O N L E M O N L E under ATTACKATDAWN.
        # C = K - P: L-A=L(11-0=11), E-T=L(4-19=-15=11), M-T=T(12-19=-7=19),
        # O-A=O, N-C=L(13-2=11), L-K=B(11-10=1), E-A=E, M-T=T, O-D=L(14-3=11),
        # N-A=N, L-W=P(11-22=-11=15), E-N=R(4-13=-9=17).
        self.assertEqual(beaufort_encrypt("ATTACKATDAWN", "LEMON"), "LLTOLBETLNPR")

    def test_variant_beaufort_known_triple(self) -> None:
        # C = P - K: A-L=P(0-11=-11=15), T-E=P(19-4=15), T-M=H(7), A-O=M(-14=12),
        # C-N=P(-11=15), K-L=Z(-1=25), A-E=W(-4=22), T-M=H, D-O=P(-11=15),
        # A-N=N(-13=13), W-L=L(11), N-E=J(9).
        self.assertEqual(variant_encrypt("ATTACKATDAWN", "LEMON"), "PPHMPZWHPNLJ")

    def test_vigenere_contrast_is_the_textbook_example(self) -> None:
        # The three ciphers differ only in the direction of the arithmetic, so
        # the classic Vigenere answer is the anchor for the other two.
        self.assertEqual(vigenere_encrypt("ATTACKATDAWN", "LEMON"), "LXFOPVEFRNHR")

    def test_the_three_ciphers_disagree(self) -> None:
        beaufort = beaufort_encrypt("ATTACKATDAWN", "LEMON")
        variant = variant_encrypt("ATTACKATDAWN", "LEMON")
        vigenere = vigenere_encrypt("ATTACKATDAWN", "LEMON")
        self.assertEqual(len({beaufort, variant, vigenere}), 3)

    def test_beaufort_decrypt_recovers_the_hand_case(self) -> None:
        self.assertEqual(beaufort_decrypt("LLTOLBETLNPR", "LEMON"), "ATTACKATDAWN")

    def test_variant_decrypt_recovers_the_hand_case(self) -> None:
        self.assertEqual(variant_decrypt("PPHMPZWHPNLJ", "LEMON"), "ATTACKATDAWN")


class TestSelfReciprocity(unittest.TestCase):
    """Beaufort is its own inverse; variant Beaufort is not."""

    def test_beaufort_encrypting_twice_returns_the_plaintext(self) -> None:
        for key in ("LEMON", "A", "ZEBRA", "CIPHERCHALLENGE"):
            once = beaufort_encrypt("ATTACKATDAWN", key)
            twice = beaufort_encrypt(once, key)
            self.assertEqual(twice, "ATTACKATDAWN", f"key={key}")

    def test_beaufort_encrypt_and_decrypt_are_the_same_operation(self) -> None:
        text = corpus_letters(120)
        self.assertEqual(
            beaufort_encrypt(text, "SLIDERULE"), beaufort_decrypt(text, "SLIDERULE")
        )

    def test_variant_beaufort_is_not_self_reciprocal(self) -> None:
        # Encrypting twice with the variant subtracts the key twice, so it can
        # only return the plaintext if the key is all A (subtracting zero).
        once = variant_encrypt("ATTACKATDAWN", "LEMON")
        self.assertNotEqual(variant_encrypt(once, "LEMON"), "ATTACKATDAWN")

    def test_variant_encryption_equals_vigenere_decryption(self) -> None:
        text = corpus_letters(120)
        self.assertEqual(
            variant_encrypt(text, "LEMON"), vigenere_decrypt(text, "LEMON")
        )

    def test_variant_decryption_equals_vigenere_encryption(self) -> None:
        text = corpus_letters(120)
        self.assertEqual(variant_decrypt(text, "LEMON"), vigenere_encrypt(text, "LEMON"))


class TestRoundTrip(unittest.TestCase):
    """decrypt(encrypt(x, k), k) == x for several keys and both variants."""

    PLAIN = corpus_letters(300)

    def test_round_trip_beaufort(self) -> None:
        for key in ("A", "BC", "LEMON", "SEVENTEEN", "THEQUICKBROWNFOX"):
            with self.subTest(key=key):
                self.assertEqual(
                    decrypt(encrypt(self.PLAIN, key), key), self.PLAIN
                )

    def test_round_trip_variant(self) -> None:
        for key in ("A", "BC", "LEMON", "SEVENTEEN", "THEQUICKBROWNFOX"):
            with self.subTest(key=key):
                self.assertEqual(
                    decrypt(encrypt(self.PLAIN, key, variant=True), key, variant=True),
                    self.PLAIN,
                )

    def test_the_variant_flag_actually_changes_the_cipher(self) -> None:
        self.assertNotEqual(
            encrypt(self.PLAIN, "LEMON"), encrypt(self.PLAIN, "LEMON", variant=True)
        )

    def test_a_key_longer_than_the_text_still_round_trips(self) -> None:
        self.assertEqual(decrypt(encrypt("HELLO", "VERYLONGKEYINDEED"),
                                 "VERYLONGKEYINDEED"), "HELLO")


class TestInputRobustness(unittest.TestCase):
    """Case, spacing, punctuation and grouping must not change the answer."""

    EXPECTED = "LLTOLBETLNPR"

    def test_lowercase(self) -> None:
        self.assertEqual(beaufort_encrypt("attackatdawn", "LEMON"), self.EXPECTED)

    def test_spaces_and_punctuation(self) -> None:
        self.assertEqual(
            beaufort_encrypt("Attack at dawn!", "LEMON"), self.EXPECTED
        )

    def test_five_letter_groups_and_line_breaks(self) -> None:
        grouped = group_text("ATTACKATDAWN", size=5, per_line=2)
        # "ATTAC KATDA\nWN" -- both a group separator and a line break.
        self.assertIn(" ", grouped)
        self.assertIn("\n", grouped)
        self.assertEqual(beaufort_encrypt(grouped, "LEMON"), self.EXPECTED)

    def test_digits_and_symbols_are_ignored(self) -> None:
        self.assertEqual(
            beaufort_encrypt("ATTACK 42 AT-DAWN (#3)", "LEMON"), self.EXPECTED
        )

    def test_key_is_normalised_the_same_way(self) -> None:
        self.assertEqual(
            beaufort_encrypt("ATTACKATDAWN", "l e-m o n!"), self.EXPECTED
        )

    def test_decrypting_grouped_ciphertext(self) -> None:
        self.assertEqual(
            beaufort_decrypt("LLTOL BETLN PR", "LEMON"), "ATTACKATDAWN"
        )

    def test_solver_accepts_grouped_input_and_a_normalized_text(self) -> None:
        plain = corpus_letters(300)
        ciphertext = beaufort_encrypt(plain, "OTTER")
        grouped = group_text(ciphertext, size=5, per_line=10)
        from_string = solve(grouped, key_length=5, top=1).best()
        from_object = solve(normalize(grouped), key_length=5, top=1).best()
        self.assertIsNotNone(from_string)
        self.assertIsNotNone(from_object)
        self.assertEqual(from_string.plaintext, plain)
        self.assertEqual(from_string.plaintext, from_object.plaintext)


class TestEmptyInput(unittest.TestCase):
    """Empty input must be handled, not crashed on."""

    def test_encrypt_empty(self) -> None:
        self.assertEqual(beaufort_encrypt("", "LEMON"), "")
        self.assertEqual(variant_encrypt("", "LEMON"), "")

    def test_encrypt_text_with_no_letters(self) -> None:
        self.assertEqual(beaufort_encrypt("123 !!! ???", "LEMON"), "")

    def test_solve_empty_returns_an_empty_candidate_set(self) -> None:
        result = solve("")
        self.assertEqual(len(result), 0)
        self.assertIsNone(result.best())
        self.assertFalse(result)


class TestInvalidInput(unittest.TestCase):
    """Bad keys and impossible configurations raise ValueError, with a reason."""

    def _assert_explains(self, context: unittest.case._AssertRaisesContext) -> None:
        message = str(context.exception)
        self.assertTrue(message.strip(), "ValueError carried no explanation")

    def test_empty_key_rejected(self) -> None:
        with self.assertRaises(ValueError) as context:
            beaufort_encrypt("ATTACKATDAWN", "")
        self._assert_explains(context)

    def test_key_with_no_letters_rejected(self) -> None:
        with self.assertRaises(ValueError) as context:
            variant_encrypt("ATTACKATDAWN", "1234 !!")
        self._assert_explains(context)
        self.assertIn("1234", str(context.exception))

    def test_unknown_variant_rejected(self) -> None:
        with self.assertRaises(ValueError) as context:
            solve("ATTACKATDAWN", variants=("rot13",))
        self._assert_explains(context)
        self.assertIn("rot13", str(context.exception))

    def test_key_length_longer_than_ciphertext_rejected(self) -> None:
        with self.assertRaises(ValueError) as context:
            solve("SHORT", key_length=99)
        self._assert_explains(context)

    def test_derive_key_rejects_a_zero_length_key(self) -> None:
        with self.assertRaises(ValueError) as context:
            derive_key("ATTACKATDAWN", 0, BEAUFORT)
        self._assert_explains(context)

    def test_column_key_letter_rejects_an_empty_column(self) -> None:
        with self.assertRaises(ValueError) as context:
            column_key_letter("", BEAUFORT)
        self._assert_explains(context)

    def test_inverted_key_length_range_rejected(self) -> None:
        with self.assertRaises(ValueError) as context:
            solve(corpus_letters(200), min_key_length=9, max_key_length=4)
        self._assert_explains(context)


class TestColumnRule(unittest.TestCase):
    """The per-column rule must be derived per variant, not shared."""

    def test_beaufort_column_letter_is_recovered(self) -> None:
        plain = corpus_letters(400)
        # One column of a Beaufort message enciphered by a single key letter.
        column = beaufort_encrypt(plain, "Q")
        key_value, _, margin = column_key_letter(column, BEAUFORT)
        self.assertEqual(key_value, ord("Q") - 65)
        self.assertGreater(margin, 0.0)

    def test_variant_column_letter_is_recovered(self) -> None:
        plain = corpus_letters(400)
        column = variant_encrypt(plain, "Q")
        key_value, _, _ = column_key_letter(column, VARIANT)
        self.assertEqual(key_value, ord("Q") - 65)

    def test_the_two_rules_disagree_about_the_same_column(self) -> None:
        # This is the whole point of not reusing the Vigenere column solver:
        # on Beaufort ciphertext the shift-only rule reads a different letter.
        column = beaufort_encrypt(corpus_letters(400), "Q")
        beaufort_letter, _, _ = column_key_letter(column, BEAUFORT)
        variant_letter, _, _ = column_key_letter(column, VARIANT)
        self.assertNotEqual(beaufort_letter, variant_letter)

    def test_derive_key_finds_the_whole_key(self) -> None:
        plain = corpus_letters(500)
        key_values, _, _ = derive_key(beaufort_encrypt(plain, "OTTER"), 5, BEAUFORT)
        self.assertEqual(key_values, [ord(c) - 65 for c in "OTTER"])


class TestSolver(unittest.TestCase):
    """The solver must recover a known key from a realistic ciphertext."""

    def test_solves_a_400_letter_beaufort_message(self) -> None:
        plain = corpus_letters(420)
        ciphertext = beaufort_encrypt(plain, "FALCON")
        best = solve(ciphertext).best()
        self.assertIsNotNone(best)
        self.assertEqual(best.plaintext, plain)
        self.assertEqual(best.method, "Beaufort")
        self.assertIn("FALCON", best.key)
        self.assertIn("beaufort", best.key)
        self.assertEqual(best.diagnostics["key_length"], 6)
        self.assertTrue(best.diagnostics["meets_english_threshold"])
        self.assertEqual(best.confidence(), "strong")

    def test_solves_a_400_letter_variant_beaufort_message(self) -> None:
        plain = corpus_letters(420)
        ciphertext = variant_encrypt(plain, "RIVER")
        best = solve(ciphertext, max_key_length=12).best()
        self.assertIsNotNone(best)
        self.assertEqual(best.plaintext, plain)
        self.assertEqual(best.method, "Variant Beaufort")
        self.assertIn("variant-beaufort", best.key)
        self.assertIn("RIVER", best.key)

    def test_both_variants_are_ranked_in_one_list(self) -> None:
        plain = corpus_letters(420)
        everything = solve(beaufort_encrypt(plain, "FALCON"), top=0, max_key_length=8)
        methods = {candidate.method for candidate in everything}
        self.assertEqual(methods, {"Beaufort", "Variant Beaufort"})
        # The correct variant must win, and by a visible margin.
        self.assertEqual(everything.best().method, "Beaufort")
        self.assertGreater(everything.score_gap(), 0.5)

    def test_candidates_carry_their_evidence(self) -> None:
        plain = corpus_letters(420)
        best = solve(beaufort_encrypt(plain, "FALCON"), max_key_length=8).best()
        for field in (
            "key_length",
            "chi_squared_total",
            "column_margin_mean",
            "column_ic_at_period",
            "kasiski_votes_at_length",
            "word_coverage",
            "normalised_score",
        ):
            self.assertIn(field, best.diagnostics)
        # The independent period evidence should agree with the winning key.
        self.assertGreater(best.diagnostics["column_ic_at_period"], 0.055)

    def test_the_hill_climb_rescues_a_key_chi_squared_gets_wrong(self) -> None:
        """Observe the refinement stage doing work the first stage cannot.

        A nine-letter key over 180 letters leaves only 20 letters per column.
        Chi-squared reads single-letter frequencies only, and on a sample that
        small it misreads one of the columns. The n-gram hill-climb sees the
        broken words that mistake produces and repairs the key.
        """
        plain = corpus_letters(180)
        ciphertext = beaufort_encrypt(plain, "ADMIRALTY")

        unrefined = solve(ciphertext, key_length=9, refine=False, top=1).best()
        recovered = unrefined.key.split("key=")[1].split(" ")[0]
        self.assertNotEqual(unrefined.plaintext, plain)
        self.assertNotEqual(recovered, "ADMIRALTY")
        # It should be close, though -- most columns are read correctly.
        wrong_letters = sum(
            1 for got, want in zip(recovered, "ADMIRALTY") if got != want
        )
        self.assertGreaterEqual(wrong_letters, 1)
        self.assertLessEqual(wrong_letters, 3)

        refined = solve(ciphertext, key_length=9, refine=True, top=1).best()
        self.assertEqual(refined.plaintext, plain)
        self.assertIn("ADMIRALTY", refined.key)
        self.assertTrue(refined.diagnostics["refined"])
        self.assertGreater(refined.score, unrefined.score)

    def test_display_preserves_the_original_layout(self) -> None:
        plain = corpus_letters(300)
        grouped = group_text(beaufort_encrypt(plain, "OTTER"), size=5, per_line=8)
        best = solve(grouped, key_length=5, top=1).best()
        self.assertIsNotNone(best.display)
        self.assertEqual(len(best.display), len(grouped))
        self.assertEqual(letters_only(best.display), best.plaintext)

    def test_top_limits_the_returned_candidates(self) -> None:
        plain = corpus_letters(300)
        result = solve(beaufort_encrypt(plain, "OTTER"), top=3, max_key_length=8)
        self.assertEqual(len(result), 3)

    def test_zero_time_budget_returns_nothing_rather_than_junk(self) -> None:
        # No time to search at all means no candidates. Reporting an unsearched
        # guess here would be worse than reporting nothing.
        result = solve(beaufort_encrypt(corpus_letters(420), "FALCON"),
                       time_budget=0.0)
        self.assertEqual(len(result), 0)

    def test_hill_climb_reports_a_blown_budget(self) -> None:
        # Deterministic: hand the climber a deadline that has already passed.
        ciphertext = beaufort_encrypt(corpus_letters(300), "OTTER")
        values = to_numbers(ciphertext)
        _, _, _, hit = _hill_climb(
            values, [0, 0, 0, 0, 0], BEAUFORT, default_scorer(),
            time.monotonic() - 1.0,
        )
        self.assertTrue(hit)

    def test_a_partial_time_budget_flags_every_candidate(self) -> None:
        # Calibrate against this machine: measure a full search, then allow an
        # eighth of it, which must cut the search short mid-way.
        ciphertext = beaufort_encrypt(corpus_letters(420), "FALCON")
        start = time.monotonic()
        solve(ciphertext)
        full = time.monotonic() - start

        result = solve(ciphertext, top=0, time_budget=full / 8.0)
        self.assertGreater(len(result), 0, "the search stopped before it began")
        for candidate in result:
            self.assertTrue(
                candidate.diagnostics.get("time_budget_hit"),
                "a truncated search must say so on every candidate",
            )


class TestFailureModes(unittest.TestCase):
    """Things that must NOT solve, and must say so."""

    def test_the_shift_only_rule_fails_on_beaufort_ciphertext(self) -> None:
        """Observing the failure that makes the per-variant derivation necessary.

        Variant Beaufort decrypts a column with a pure shift, which is what a
        Vigenere column solver does. Beaufort needs a reflection first, so the
        shift-only solver cannot read Beaufort ciphertext no matter which key
        it picks -- it can only produce a reversed alphabet, and reversed
        English is not English.
        """
        plain = corpus_letters(420)
        ciphertext = beaufort_encrypt(plain, "FALCON")
        wrong = solve(ciphertext, variants=(VARIANT,), max_key_length=12).best()
        self.assertIsNotNone(wrong)
        self.assertNotEqual(wrong.plaintext, plain)
        self.assertFalse(wrong.diagnostics["meets_english_threshold"])
        self.assertIn(wrong.confidence(), {"weak", "unlikely"})

    def test_the_correct_rule_succeeds_on_the_same_ciphertext(self) -> None:
        # The control for the test above: same text, right rule, solved.
        plain = corpus_letters(420)
        ciphertext = beaufort_encrypt(plain, "FALCON")
        right = solve(ciphertext, variants=(BEAUFORT,), max_key_length=12).best()
        self.assertEqual(right.plaintext, plain)
        self.assertTrue(right.diagnostics["meets_english_threshold"])

    def test_random_letters_are_reported_as_unsolved(self) -> None:
        import random

        generator = random.Random(20260816)
        junk = "".join(
            generator.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(400)
        )
        best = solve(junk, max_key_length=8).best()
        self.assertIsNotNone(best)
        self.assertFalse(best.diagnostics["meets_english_threshold"])
        self.assertIn(best.confidence(), {"weak", "unlikely"})

    def test_a_wrong_key_scores_far_below_the_right_one(self) -> None:
        plain = corpus_letters(300)
        ciphertext = beaufort_encrypt(plain, "FALCON")
        scorer = default_scorer()
        right = scorer.normalised(beaufort_decrypt(ciphertext, "FALCON"))
        wrong = scorer.normalised(beaufort_decrypt(ciphertext, "FALCOM"))
        self.assertGreater(right, -1.5)
        self.assertLess(wrong, right - 0.3)


if __name__ == "__main__":
    unittest.main()
