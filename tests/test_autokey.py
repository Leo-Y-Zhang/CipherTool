"""Tests for the autokey module, in both its plaintext and ciphertext forms.

The hand-computed cases were worked out on paper before any code ran. Both use
the plaintext ATTACKATDAWN with the primer QUEEN, which makes the difference
between the two forms visible: the key streams agree for the first five letters
(the primer) and diverge from the sixth, so the two ciphertexts share the
prefix QNXEP and then part company.
"""

from __future__ import annotations

import random
import time
import unittest

from cipher_tool.autokey import (
    ATTACK_CAVEAT,
    CIPHERTEXT,
    PLAINTEXT,
    ciphertext_autokey_decrypt,
    ciphertext_autokey_encrypt,
    decrypt,
    encrypt,
    initial_primer,
    plaintext_autokey_decrypt,
    plaintext_autokey_encrypt,
    solve,
)
from cipher_tool.normalize import group_text, letters_only, normalize, to_numbers
from cipher_tool.scoring import DATA_DIR, default_scorer

# Build the order-3 table once so the timed test measures the search, not the
# one-off cost of constructing the language model.
default_scorer().table()


def corpus_letters(count: int, start: int = 0) -> str:
    """*count* letters of the expository corpus, from offset *start*."""
    text = (DATA_DIR / "corpus_04_expository.txt").read_text(encoding="utf-8")
    letters = letters_only(text)
    if len(letters) < start + count:  # pragma: no cover - guards a bad checkout
        raise AssertionError("corpus_04_expository.txt is shorter than the test needs")
    return letters[start : start + count]


class TestHandComputedCases(unittest.TestCase):
    """Known plaintext/primer/ciphertext triples, computed by hand."""

    def test_plaintext_autokey_known_triple(self) -> None:
        # Key stream = QUEEN + ATTACKA (the plaintext, shifted five places):
        #   Q U E E N A T T A C K A
        #   A T T A C K A T D A W N   plaintext
        # C = P + K: A+Q=Q, T+U=N(19+20=39-26=13), T+E=X(23), A+E=E, C+N=P(15),
        # K+A=K, A+T=T, T+T=M(38-26=12), D+A=D, A+C=C, W+K=G(32-26=6), N+A=N.
        self.assertEqual(
            plaintext_autokey_encrypt("ATTACKATDAWN", "QUEEN"), "QNXEPKTMDCGN"
        )

    def test_ciphertext_autokey_known_triple(self) -> None:
        # Key stream = QUEEN + QNXEP (the ciphertext as it is produced):
        # first five as above, then K+Q=A(10+16=26=0), A+N=N(13), T+X=Q(42-26=16),
        # D+E=H(7), A+P=P(15), W+A=W(22), N+N=A(26=0).
        self.assertEqual(
            ciphertext_autokey_encrypt("ATTACKATDAWN", "QUEEN"), "QNXEPANQHPWA"
        )

    def test_the_two_forms_share_the_primer_prefix_then_diverge(self) -> None:
        plaintext_form = plaintext_autokey_encrypt("ATTACKATDAWN", "QUEEN")
        ciphertext_form = ciphertext_autokey_encrypt("ATTACKATDAWN", "QUEEN")
        self.assertEqual(plaintext_form[:5], ciphertext_form[:5])
        self.assertEqual(plaintext_form[:5], "QNXEP")
        self.assertNotEqual(plaintext_form[5:], ciphertext_form[5:])

    def test_plaintext_autokey_decrypt_recovers_the_hand_case(self) -> None:
        self.assertEqual(
            plaintext_autokey_decrypt("QNXEPKTMDCGN", "QUEEN"), "ATTACKATDAWN"
        )

    def test_ciphertext_autokey_decrypt_recovers_the_hand_case(self) -> None:
        self.assertEqual(
            ciphertext_autokey_decrypt("QNXEPANQHPWA", "QUEEN"), "ATTACKATDAWN"
        )

    def test_a_primer_longer_than_the_message_is_plain_vigenere(self) -> None:
        # If the primer covers the whole message the key never autokeys, so
        # both forms collapse to Vigenere with a non-repeating key.
        from cipher_tool.beaufort import vigenere_encrypt

        expected = vigenere_encrypt("ATTACK", "LONGERPRIMER")
        self.assertEqual(plaintext_autokey_encrypt("ATTACK", "LONGERPRIMER"), expected)
        self.assertEqual(ciphertext_autokey_encrypt("ATTACK", "LONGERPRIMER"), expected)


class TestStructure(unittest.TestCase):
    """The structural facts the attacks rely on."""

    def test_ciphertext_autokey_tail_needs_no_key(self) -> None:
        """P_i = C_i - C_{i-m}: the primer cancels out completely."""
        plain = corpus_letters(200)
        ciphertext = ciphertext_autokey_encrypt(plain, "MARLIN")
        values = to_numbers(ciphertext)
        recovered = [
            (values[i] - values[i - 6]) % 26 for i in range(6, len(values))
        ]
        self.assertEqual(
            "".join(chr(65 + v) for v in recovered), plain[6:]
        )

    def test_the_tail_is_the_same_whatever_the_primer_was(self) -> None:
        # Same plaintext, different primers of the same length: the letters
        # after position m decrypt identically without knowing either primer.
        plain = corpus_letters(200)
        first = to_numbers(ciphertext_autokey_encrypt(plain, "MARLIN"))
        second = to_numbers(ciphertext_autokey_encrypt(plain, "ZEPHYR"))
        tail_one = [(first[i] - first[i - 6]) % 26 for i in range(6, len(first))]
        tail_two = [(second[i] - second[i - 6]) % 26 for i in range(6, len(second))]
        self.assertEqual(tail_one, tail_two)

    def test_plaintext_autokey_splits_into_independent_chains(self) -> None:
        """Primer letter j affects only positions congruent to j modulo m."""
        plain = corpus_letters(120)
        ciphertext = plaintext_autokey_encrypt(plain, "QUEEN")
        base = plaintext_autokey_decrypt(ciphertext, "QUEEN")
        changed = plaintext_autokey_decrypt(ciphertext, "QUFEN")  # position 2
        differing = {i for i, (a, b) in enumerate(zip(base, changed)) if a != b}
        self.assertTrue(differing)
        self.assertTrue(all(index % 5 == 2 for index in differing))

    def test_a_wrong_primer_letter_alternates_sign_down_its_chain(self) -> None:
        """The error flips sign at each step, so no single shift repairs it."""
        plain = corpus_letters(120)
        ciphertext = plaintext_autokey_encrypt(plain, "QUEEN")
        wrong = plaintext_autokey_decrypt(ciphertext, "RUEEN")  # position 0, +1
        errors = [
            (ord(wrong[i]) - ord(plain[i])) % 26
            for i in range(0, len(plain), 5)
        ]
        # +1 error at the primer becomes -1, +1, -1, ... down the chain.
        expected = [(-1) ** (step + 1) % 26 for step in range(len(errors))]
        self.assertEqual(errors, [value % 26 for value in expected])

    def test_initial_primer_guesses_most_of_a_primer_from_frequencies_alone(
        self,
    ) -> None:
        plain = corpus_letters(600)
        ciphertext = plaintext_autokey_encrypt(plain, "ADMIRAL")
        guess, margins = initial_primer(ciphertext, 7)
        recovered = "".join(chr(65 + value) for value in guess)
        correct = sum(1 for got, want in zip(recovered, "ADMIRAL") if got == want)
        self.assertGreaterEqual(correct, 5, f"chi-squared guessed {recovered}")
        self.assertEqual(len(margins), 7)


class TestRoundTrip(unittest.TestCase):
    """decrypt(encrypt(x, p), p) == x for several primers and both modes."""

    PLAIN = corpus_letters(300)

    def test_round_trip_plaintext_mode(self) -> None:
        for primer in ("A", "QU", "QUEEN", "ADMIRALTY", "AVERYLONGPRIMERINDEED"):
            with self.subTest(primer=primer):
                self.assertEqual(
                    decrypt(encrypt(self.PLAIN, primer), primer), self.PLAIN
                )

    def test_round_trip_ciphertext_mode(self) -> None:
        for primer in ("A", "QU", "QUEEN", "ADMIRALTY", "AVERYLONGPRIMERINDEED"):
            with self.subTest(primer=primer):
                self.assertEqual(
                    decrypt(
                        encrypt(self.PLAIN, primer, mode=CIPHERTEXT),
                        primer,
                        mode=CIPHERTEXT,
                    ),
                    self.PLAIN,
                )

    def test_the_mode_flag_actually_changes_the_cipher(self) -> None:
        self.assertNotEqual(
            encrypt(self.PLAIN, "QUEEN"),
            encrypt(self.PLAIN, "QUEEN", mode=CIPHERTEXT),
        )

    def test_decrypting_in_the_wrong_mode_gives_nonsense(self) -> None:
        ciphertext = encrypt(self.PLAIN, "QUEEN", mode=PLAINTEXT)
        wrong = decrypt(ciphertext, "QUEEN", mode=CIPHERTEXT)
        self.assertNotEqual(wrong, self.PLAIN)
        self.assertLess(default_scorer().normalised(wrong), -1.80)


class TestInputRobustness(unittest.TestCase):
    """Case, spacing, punctuation and grouping must not change the answer."""

    EXPECTED = "QNXEPKTMDCGN"

    def test_lowercase(self) -> None:
        self.assertEqual(
            plaintext_autokey_encrypt("attackatdawn", "QUEEN"), self.EXPECTED
        )

    def test_spaces_and_punctuation(self) -> None:
        self.assertEqual(
            plaintext_autokey_encrypt("Attack at dawn!", "QUEEN"), self.EXPECTED
        )

    def test_five_letter_groups_and_line_breaks(self) -> None:
        grouped = group_text("ATTACKATDAWN", size=5, per_line=2)
        self.assertIn(" ", grouped)
        self.assertIn("\n", grouped)
        self.assertEqual(plaintext_autokey_encrypt(grouped, "QUEEN"), self.EXPECTED)

    def test_digits_and_symbols_are_ignored(self) -> None:
        self.assertEqual(
            plaintext_autokey_encrypt("ATTACK 42 AT-DAWN (#3)", "QUEEN"),
            self.EXPECTED,
        )

    def test_primer_is_normalised_the_same_way(self) -> None:
        self.assertEqual(
            plaintext_autokey_encrypt("ATTACKATDAWN", "q u-e e n!"), self.EXPECTED
        )

    def test_ciphertext_mode_is_equally_robust(self) -> None:
        self.assertEqual(
            ciphertext_autokey_encrypt("attack at dawn!", "queen"), "QNXEPANQHPWA"
        )

    def test_solver_accepts_grouped_input_and_a_normalized_text(self) -> None:
        plain = corpus_letters(300)
        grouped = group_text(
            plaintext_autokey_encrypt(plain, "KEY"), size=5, per_line=10
        )
        from_string = solve(grouped, max_primer=3, modes=(PLAINTEXT,), top=1).best()
        from_object = solve(
            normalize(grouped), max_primer=3, modes=(PLAINTEXT,), top=1
        ).best()
        self.assertEqual(from_string.plaintext, plain)
        self.assertEqual(from_string.plaintext, from_object.plaintext)


class TestEmptyInput(unittest.TestCase):
    """Empty input must be handled, not crashed on."""

    def test_encrypt_empty(self) -> None:
        self.assertEqual(plaintext_autokey_encrypt("", "QUEEN"), "")
        self.assertEqual(ciphertext_autokey_encrypt("", "QUEEN"), "")

    def test_decrypt_empty(self) -> None:
        self.assertEqual(plaintext_autokey_decrypt("", "QUEEN"), "")
        self.assertEqual(ciphertext_autokey_decrypt("", "QUEEN"), "")

    def test_text_with_no_letters(self) -> None:
        self.assertEqual(plaintext_autokey_encrypt("123 !!! ???", "QUEEN"), "")

    def test_solve_empty_returns_an_empty_candidate_set(self) -> None:
        result = solve("")
        self.assertEqual(len(result), 0)
        self.assertIsNone(result.best())
        self.assertFalse(result)

    def test_solve_on_a_single_letter_does_not_crash(self) -> None:
        # Every primer length is at least as long as the message, so there is
        # nothing to attack; the solver must say nothing rather than invent.
        self.assertEqual(len(solve("A")), 0)


class TestInvalidInput(unittest.TestCase):
    """Bad primers and modes raise ValueError, with a reason."""

    def _assert_explains(self, context: unittest.case._AssertRaisesContext) -> None:
        self.assertTrue(str(context.exception).strip(), "ValueError explained nothing")

    def test_empty_primer_rejected(self) -> None:
        with self.assertRaises(ValueError) as context:
            plaintext_autokey_encrypt("ATTACKATDAWN", "")
        self._assert_explains(context)

    def test_primer_with_no_letters_rejected(self) -> None:
        with self.assertRaises(ValueError) as context:
            ciphertext_autokey_encrypt("ATTACKATDAWN", "12 34")
        self._assert_explains(context)
        self.assertIn("12 34", str(context.exception))

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError) as context:
            encrypt("ATTACKATDAWN", "QUEEN", mode="autokey")
        self._assert_explains(context)
        self.assertIn("autokey", str(context.exception))

    def test_unknown_mode_rejected_by_the_solver(self) -> None:
        with self.assertRaises(ValueError) as context:
            solve("ATTACKATDAWN", modes=("running-key",))
        self._assert_explains(context)

    def test_inverted_primer_range_rejected(self) -> None:
        with self.assertRaises(ValueError) as context:
            solve(corpus_letters(200), min_primer=6, max_primer=2)
        self._assert_explains(context)

    def test_zero_primer_length_rejected(self) -> None:
        with self.assertRaises(ValueError) as context:
            solve(corpus_letters(200), min_primer=0)
        self._assert_explains(context)

    def test_initial_primer_rejects_an_impossible_length(self) -> None:
        with self.assertRaises(ValueError) as context:
            initial_primer("SHORT", 99)
        self._assert_explains(context)


class TestPlaintextAutokeySolver(unittest.TestCase):
    """Recovering a primer and a message from plaintext-autokey ciphertext."""

    def test_solves_a_400_letter_message_with_a_short_primer(self) -> None:
        plain = corpus_letters(420)
        best = solve(plaintext_autokey_encrypt(plain, "KEY")).best()
        self.assertIsNotNone(best)
        self.assertEqual(best.plaintext, plain)
        self.assertEqual(best.method, "Autokey (plaintext)")
        self.assertIn("primer=KEY", best.key)
        self.assertIn("mode=plaintext", best.key)
        self.assertTrue(best.diagnostics["meets_english_threshold"])
        self.assertEqual(best.diagnostics["primer_length"], 3)

    def test_solves_a_400_letter_message_with_a_six_letter_primer(self) -> None:
        plain = corpus_letters(420)
        best = solve(plaintext_autokey_encrypt(plain, "FALCON")).best()
        self.assertEqual(best.plaintext, plain)
        self.assertIn("primer=FALCON", best.key)
        self.assertTrue(best.diagnostics["meets_english_threshold"])

    def test_short_primers_are_searched_exhaustively(self) -> None:
        plain = corpus_letters(420)
        best = solve(
            plaintext_autokey_encrypt(plain, "QU"), max_primer=2, modes=(PLAINTEXT,)
        ).best()
        self.assertEqual(best.plaintext, plain)
        self.assertIn("exhaustive", best.diagnostics["search"])

    def test_both_modes_are_ranked_in_one_list(self) -> None:
        plain = corpus_letters(420)
        everything = solve(
            plaintext_autokey_encrypt(plain, "FALCON"), top=0, max_primer=6
        )
        methods = {candidate.method for candidate in everything}
        self.assertEqual(
            methods, {"Autokey (plaintext)", "Autokey (ciphertext)"}
        )
        self.assertEqual(everything.best().method, "Autokey (plaintext)")

    def test_candidates_carry_their_evidence_and_the_caveat(self) -> None:
        plain = corpus_letters(420)
        best = solve(plaintext_autokey_encrypt(plain, "KEY"), max_primer=4).best()
        for field in (
            "mode",
            "primer_length",
            "search",
            "max_primer_tried",
            "caveat",
            "word_coverage",
            "normalised_score",
            "meets_english_threshold",
        ):
            self.assertIn(field, best.diagnostics)
        self.assertEqual(best.diagnostics["caveat"], ATTACK_CAVEAT)
        self.assertIn("weaker than Vigenere", best.diagnostics["caveat"])

    def test_display_preserves_the_original_layout(self) -> None:
        plain = corpus_letters(300)
        grouped = group_text(
            plaintext_autokey_encrypt(plain, "KEY"), size=5, per_line=8
        )
        best = solve(grouped, max_primer=3, modes=(PLAINTEXT,), top=1).best()
        self.assertEqual(len(best.display), len(grouped))
        self.assertEqual(letters_only(best.display), best.plaintext)

    def test_the_seed_makes_the_search_reproducible(self) -> None:
        ciphertext = plaintext_autokey_encrypt(corpus_letters(300), "ADMIRAL")
        first = solve(ciphertext, max_primer=7, seed=99, top=0)
        second = solve(ciphertext, max_primer=7, seed=99, top=0)
        self.assertEqual(
            [(c.key, round(c.score, 6)) for c in first],
            [(c.key, round(c.score, 6)) for c in second],
        )

    def test_the_global_random_module_is_never_used(self) -> None:
        # Seeding the global module must not change a seeded solve at all.
        ciphertext = plaintext_autokey_encrypt(corpus_letters(300), "ADMIRAL")
        random.seed(1)
        first = [c.key for c in solve(ciphertext, max_primer=7, seed=5, top=0)]
        random.seed(9999)
        second = [c.key for c in solve(ciphertext, max_primer=7, seed=5, top=0)]
        self.assertEqual(first, second)

    def test_top_limits_the_returned_candidates(self) -> None:
        plain = corpus_letters(300)
        result = solve(plaintext_autokey_encrypt(plain, "KEY"), top=2, max_primer=5)
        self.assertEqual(len(result), 2)


class TestCiphertextAutokeySolver(unittest.TestCase):
    """Ciphertext autokey: the message is forced, the opening is a guess."""

    def test_recovers_the_whole_message_after_the_primer(self) -> None:
        plain = corpus_letters(420)
        ciphertext = ciphertext_autokey_encrypt(plain, "MARLIN")
        best = solve(ciphertext, modes=(CIPHERTEXT,), seed=1).best()
        self.assertEqual(best.diagnostics["primer_length"], 6)
        # Everything from position six is forced by the ciphertext alone.
        self.assertEqual(best.plaintext[6:], plain[6:])
        self.assertEqual(best.diagnostics["forced_letters"], len(plain) - 6)
        self.assertTrue(best.diagnostics["meets_english_threshold"])

    def test_the_opening_is_reported_as_a_guess_not_as_evidence(self) -> None:
        """The one thing this attack genuinely cannot recover.

        Every possible opening is consistent with some primer, so the first m
        letters are not cryptanalysis at all -- they are whichever English
        opening the model likes best, and it is often wrong. The solver must
        say so rather than presenting the primer as a finding.
        """
        plain = corpus_letters(420)
        ciphertext = ciphertext_autokey_encrypt(plain, "MARLIN")
        best = solve(ciphertext, modes=(CIPHERTEXT,), seed=1).best()
        note = best.diagnostics["head_note"]
        self.assertIn("forced", note)
        self.assertIn("may be wrong", note)
        # On this passage the model does prefer a wrong opening, which is the
        # honest outcome this note exists to warn about.
        self.assertNotEqual(best.plaintext[:6], plain[:6])
        # ... and it is a genuine preference, not a search failure: the model
        # scores its own answer above the truth.
        scorer = default_scorer()
        self.assertGreater(scorer.score(best.plaintext), scorer.score(plain))

    def test_a_one_letter_primer_is_recovered_exactly(self) -> None:
        # With a single free letter the model has little room to go wrong.
        plain = corpus_letters(420)
        ciphertext = ciphertext_autokey_encrypt(plain, "A")
        best = solve(ciphertext, modes=(CIPHERTEXT,), max_primer=4, seed=1).best()
        self.assertEqual(best.plaintext, plain)
        self.assertIn("primer=A", best.key)


class TestFailureModes(unittest.TestCase):
    """Things that must NOT solve, and must say so."""

    def test_a_primer_longer_than_max_primer_is_not_solved(self) -> None:
        """An impossible configuration, honestly reported.

        The message was enciphered with a fourteen-letter primer and the
        solver is only allowed to try up to six. There is no way to succeed,
        and the solver must not dress up its best guess as an answer.
        """
        plain = corpus_letters(420)
        ciphertext = plaintext_autokey_encrypt(plain, "ADMIRALTYBOARD")
        best = solve(ciphertext, max_primer=6).best()
        self.assertIsNotNone(best)
        self.assertNotEqual(best.plaintext, plain)
        self.assertFalse(best.diagnostics["meets_english_threshold"])
        self.assertIn(best.confidence(), {"weak", "unlikely"})
        self.assertEqual(best.diagnostics["max_primer_tried"], 6)

    def test_random_letters_are_reported_as_unsolved(self) -> None:
        generator = random.Random(20260816)
        junk = "".join(
            generator.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(400)
        )
        best = solve(junk, max_primer=4).best()
        self.assertIsNotNone(best)
        self.assertFalse(best.diagnostics["meets_english_threshold"])
        self.assertIn(best.confidence(), {"weak", "unlikely"})

    def test_a_short_text_carries_the_short_text_warning(self) -> None:
        plain = corpus_letters(120)
        best = solve(plaintext_autokey_encrypt(plain, "KEY"), max_primer=4).best()
        self.assertIn("short_text_warning", best.diagnostics)
        self.assertIn("120 letters", best.diagnostics["short_text_warning"])

    def test_a_long_text_carries_no_short_text_warning(self) -> None:
        plain = corpus_letters(420)
        best = solve(plaintext_autokey_encrypt(plain, "KEY"), max_primer=4).best()
        self.assertNotIn("short_text_warning", best.diagnostics)

    def test_a_wrong_primer_scores_far_below_the_right_one(self) -> None:
        plain = corpus_letters(300)
        ciphertext = plaintext_autokey_encrypt(plain, "FALCON")
        scorer = default_scorer()
        right = scorer.normalised(plaintext_autokey_decrypt(ciphertext, "FALCON"))
        wrong = scorer.normalised(plaintext_autokey_decrypt(ciphertext, "FALCOM"))
        self.assertGreater(right, -1.5)
        self.assertLess(wrong, right - 0.2)

    def test_zero_time_budget_returns_nothing_rather_than_junk(self) -> None:
        result = solve(
            plaintext_autokey_encrypt(corpus_letters(420), "KEY"), time_budget=0.0
        )
        self.assertEqual(len(result), 0)

    def test_a_partial_time_budget_flags_every_candidate(self) -> None:
        ciphertext = plaintext_autokey_encrypt(corpus_letters(420), "FALCON")
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


if __name__ == "__main__":
    unittest.main()
