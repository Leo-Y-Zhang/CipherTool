"""Tests for cipher_tool.substitution.

The hand-computed triple below was worked out on paper before any code ran,
using the keyword CIPHER:

    keyword letters, then the rest of the alphabet in order, skipping any
    letter already used:

        C I P H E R | A B D F G J K L M N O Q S T U V W X Y Z

    so plaintext A enciphers to C, B to I, C to P, D to H, E to E, F to R,
    G to A, H to B, I to D, J to F, K to G, L to J, M to K, N to L, O to M,
    P to N, Q to O, R to Q, and S..Z stand still.

    ATTACKATDAWN
    A->C  T->T  T->T  A->C  C->P  K->G  A->C  T->T  D->H  A->C  W->W  N->L
    = CTTCPGCTHCWL

Inverting that gives the DECRYPTION alphabet, which is what a SubstitutionKey
holds and what from_alphabet takes:

    cipher : ABCDEFGHIJKLMNOPQRSTUVWXYZ
    plain  : GHAIEJKDBLMNOPQCRFSTUVWXYZ
"""

from __future__ import annotations

import random
import time
import unittest

from cipher_tool.candidates import Candidate
from cipher_tool.normalize import ALPHABET, group_text, letters_only, normalize
from cipher_tool.scoring import DATA_DIR, default_scorer
from cipher_tool.substitution import (
    MAX_CLIMB_PASSES,
    METHOD,
    RELIABLE_CLIMB_LETTERS,
    SubstitutionKey,
    _HillClimber,
    _plain_indices,
    _random_key,
    analyse_words,
    apply_crib,
    crib_positions,
    decrypt,
    encrypt,
    frequency_guess,
    solve,
    window_score,
)

# The hand-computed triple (see the module docstring).
KEYWORD_ENCIPHER = "CIPHERABDFGJKLMNOQSTUVWXYZ"
KEYWORD_DECRYPT = "GHAIEJKDBLMNOPQCRFSTUVWXYZ"
KNOWN_PLAIN = "ATTACKATDAWN"
KNOWN_CIPHER = "CTTCPGCTHCWL"

# A random-looking key for the statistical tests. Written out rather than
# shuffled at import time so the test data does not depend on the behaviour of
# any particular Python release.
SAMPLE_ENCIPHER = "TQCKBAMOEDVLWUYZHRINGJSXFP"
SAMPLE_KEY = SubstitutionKey.from_encipher_alphabet(SAMPLE_ENCIPHER)


def sample_plaintext(length: int = 400) -> str:
    """*length* letters of the team's own expository prose."""
    raw = (DATA_DIR / "corpus_04_expository.txt").read_text(encoding="utf-8")
    letters = letters_only(raw)
    if len(letters) < length:  # pragma: no cover - the corpus is far longer
        raise AssertionError("corpus_04_expository.txt is shorter than expected")
    return letters[:length]


PLAIN = sample_plaintext(400)
CIPHER = encrypt(PLAIN, SAMPLE_KEY)


class TestKeyConstruction(unittest.TestCase):
    """Building keys the several ways a human might type one."""

    def test_from_alphabet_is_cipher_to_plain(self) -> None:
        key = SubstitutionKey.from_alphabet(KEYWORD_DECRYPT)
        self.assertEqual(key.get("A"), "G")
        self.assertEqual(key.get("C"), "A")
        self.assertTrue(key.is_complete)
        self.assertEqual(key.to_alphabet(), KEYWORD_DECRYPT)

    def test_from_encipher_alphabet_is_the_other_direction(self) -> None:
        by_encipher = SubstitutionKey.from_encipher_alphabet(KEYWORD_ENCIPHER)
        by_decrypt = SubstitutionKey.from_alphabet(KEYWORD_DECRYPT)
        self.assertEqual(by_encipher, by_decrypt)

    def test_from_string_reads_pairs(self) -> None:
        key = SubstitutionKey.from_string("QE XT")
        self.assertEqual(key.mapping, {"Q": "E", "X": "T"})

    def test_from_string_tolerates_punctuation_and_case(self) -> None:
        self.assertEqual(
            SubstitutionKey.from_string("q=e, x->t").mapping,
            {"Q": "E", "X": "T"},
        )

    def test_from_string_empty_gives_empty_key(self) -> None:
        key = SubstitutionKey.from_string("   ")
        self.assertEqual(len(key), 0)
        self.assertFalse(key.is_complete)

    def test_from_pairs_accepts_tuples_and_strings(self) -> None:
        key = SubstitutionKey.from_pairs(["QE", ("x", "t")])
        self.assertEqual(key.mapping, {"Q": "E", "X": "T"})

    def test_partial_key_reports_what_it_does_not_know(self) -> None:
        key = SubstitutionKey.from_string("QE XT")
        self.assertFalse(key.is_complete)
        self.assertEqual(len(key), 2)
        self.assertNotIn("A", key.mapping)
        self.assertIn("A", key.undecided_cipher_letters)
        self.assertNotIn("E", key.unused_plain_letters)

    def test_inverse_round_trips(self) -> None:
        key = SubstitutionKey.from_alphabet(KEYWORD_DECRYPT)
        self.assertEqual(key.inverse().inverse(), key)
        self.assertEqual(key.inverse().get("G"), "A")

    def test_inverse_of_a_partial_key_is_partial(self) -> None:
        key = SubstitutionKey.from_string("QE XT")
        self.assertEqual(key.inverse().mapping, {"E": "Q", "T": "X"})

    def test_apply_marks_undecided_letters(self) -> None:
        key = SubstitutionKey.from_string("QE XT")
        self.assertEqual(key.apply("QXZ"), "ET.")
        self.assertEqual(key.apply("QXZ", placeholder="_"), "ET_")

    def test_apply_ignores_layout(self) -> None:
        key = SubstitutionKey.from_alphabet(KEYWORD_DECRYPT)
        self.assertEqual(key.apply("cttcp gcthc wl!"), KNOWN_PLAIN)

    def test_with_pair_returns_a_new_key(self) -> None:
        original = SubstitutionKey.from_string("QE")
        extended = original.with_pair("X", "T")
        self.assertEqual(len(original), 1, "with_pair must not mutate the original")
        self.assertEqual(extended.mapping, {"Q": "E", "X": "T"})

    def test_to_alphabet_shows_gaps(self) -> None:
        key = SubstitutionKey.from_string("AE BT")
        self.assertEqual(key.to_alphabet(), "ET" + "." * 24)


class TestKeyConsistency(unittest.TestCase):
    """A substitution alphabet is a bijection, and both ways of breaking it
    have to be rejected by name."""

    def test_one_cipher_letter_cannot_mean_two_plain_letters(self) -> None:
        with self.assertRaises(ValueError) as caught:
            SubstitutionKey.from_pairs([("A", "E"), ("A", "T")])
        message = str(caught.exception)
        self.assertTrue(message)
        self.assertIn("A", message)

    def test_two_cipher_letters_cannot_mean_the_same_plain_letter(self) -> None:
        with self.assertRaises(ValueError) as caught:
            SubstitutionKey({"A": "E", "B": "E"})
        message = str(caught.exception)
        self.assertTrue(message)
        self.assertIn("E", message)
        self.assertIn("bijection", message)

    def test_with_pair_rejects_a_contradictory_cipher_letter(self) -> None:
        key = SubstitutionKey.from_string("QE")
        with self.assertRaises(ValueError) as caught:
            key.with_pair("Q", "T")
        self.assertIn("Q", str(caught.exception))
        self.assertTrue(str(caught.exception))

    def test_with_pair_rejects_a_plain_letter_already_taken(self) -> None:
        key = SubstitutionKey.from_string("QE")
        with self.assertRaises(ValueError) as caught:
            key.with_pair("X", "E")
        message = str(caught.exception)
        self.assertIn("E", message)
        self.assertIn("Q", message)

    def test_with_pair_accepts_a_pair_already_present(self) -> None:
        key = SubstitutionKey.from_string("QE")
        self.assertEqual(key.with_pair("Q", "E"), key)

    def test_from_alphabet_rejects_the_wrong_length(self) -> None:
        with self.assertRaises(ValueError) as caught:
            SubstitutionKey.from_alphabet("ABC")
        self.assertIn("26", str(caught.exception))

    def test_from_alphabet_rejects_a_repeated_letter(self) -> None:
        with self.assertRaises(ValueError) as caught:
            SubstitutionKey.from_alphabet("AABCDEFGHIJKLMNOPQRSTUVWXY")
        message = str(caught.exception)
        self.assertIn("A", message)
        self.assertIn("Z", message)

    def test_from_string_rejects_a_token_that_is_not_a_pair(self) -> None:
        with self.assertRaises(ValueError) as caught:
            SubstitutionKey.from_string("QE XTZ")
        self.assertIn("XTZ", str(caught.exception))

    def test_from_string_points_a_full_alphabet_at_from_alphabet(self) -> None:
        with self.assertRaises(ValueError) as caught:
            SubstitutionKey.from_string(KEYWORD_DECRYPT)
        self.assertIn("from_alphabet", str(caught.exception))

    def test_placeholder_must_not_be_a_letter(self) -> None:
        key = SubstitutionKey.from_string("QE")
        with self.assertRaises(ValueError) as caught:
            key.apply("QZ", placeholder="X")
        self.assertTrue(str(caught.exception))

    def test_a_key_letter_must_be_a_single_letter(self) -> None:
        with self.assertRaises(ValueError) as caught:
            SubstitutionKey({"AB": "E"})
        self.assertTrue(str(caught.exception))


class TestEncryptDecrypt(unittest.TestCase):
    """Requirement 1-5: a hand-computed triple, round trips, messy input,
    empty input and invalid keys."""

    def test_known_triple(self) -> None:
        self.assertEqual(encrypt(KNOWN_PLAIN, KEYWORD_DECRYPT), KNOWN_CIPHER)
        self.assertEqual(decrypt(KNOWN_CIPHER, KEYWORD_DECRYPT), KNOWN_PLAIN)

    def test_known_triple_through_the_key_object(self) -> None:
        key = SubstitutionKey.from_encipher_alphabet(KEYWORD_ENCIPHER)
        self.assertEqual(encrypt(KNOWN_PLAIN, key), KNOWN_CIPHER)
        self.assertEqual(key.apply(KNOWN_CIPHER), KNOWN_PLAIN)

    def test_round_trip_for_several_keys(self) -> None:
        rng = random.Random(4242)
        message = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGANDTHENSOMEMORE"
        for _ in range(6):
            letters = list(ALPHABET)
            rng.shuffle(letters)
            key = SubstitutionKey.from_alphabet("".join(letters))
            self.assertEqual(decrypt(encrypt(message, key), key), message)
            self.assertEqual(encrypt(decrypt(message, key), key), message)

    def test_identity_key_changes_nothing(self) -> None:
        self.assertEqual(encrypt(KNOWN_PLAIN, ALPHABET), KNOWN_PLAIN)

    def test_input_layout_does_not_matter(self) -> None:
        expected = encrypt(KNOWN_PLAIN, KEYWORD_DECRYPT)
        variants = [
            KNOWN_PLAIN.lower(),
            "attack at dawn",
            "Attack, at dawn!",
            group_text(KNOWN_PLAIN, size=5, per_line=10),
            "ATTAC\nKATDA\nWN",
            "  ATTACK\tAT\nDAWN  ",
            "attack at dawn 12345",
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertEqual(encrypt(variant, KEYWORD_DECRYPT), expected)

    def test_decrypt_accepts_grouped_ciphertext(self) -> None:
        grouped = group_text(KNOWN_CIPHER, size=5, per_line=10)
        self.assertEqual(decrypt(grouped, KEYWORD_DECRYPT), KNOWN_PLAIN)

    def test_empty_input(self) -> None:
        self.assertEqual(encrypt("", KEYWORD_DECRYPT), "")
        self.assertEqual(decrypt("", KEYWORD_DECRYPT), "")
        self.assertEqual(decrypt("!!! 123 ???", KEYWORD_DECRYPT), "")

    def test_incomplete_key_is_refused_with_advice(self) -> None:
        partial = SubstitutionKey.from_string("QE XT")
        with self.assertRaises(ValueError) as caught:
            decrypt("QXZ", partial)
        message = str(caught.exception)
        self.assertTrue(message)
        self.assertIn("apply", message)

    def test_nonsense_key_type_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            decrypt("ABC", 17)  # type: ignore[arg-type]
        self.assertTrue(str(caught.exception))


class TestFrequencyGuess(unittest.TestCase):
    def test_commonest_letter_is_guessed_to_be_e(self) -> None:
        key = frequency_guess("AAAA BBB CC D")
        self.assertEqual(key.get("A"), "E")
        self.assertEqual(key.get("B"), "T")
        self.assertEqual(key.get("C"), "A")
        self.assertEqual(key.get("D"), "O")

    def test_absent_letters_are_left_undecided(self) -> None:
        key = frequency_guess("AAAA BBB")
        self.assertEqual(len(key), 2)
        self.assertIsNone(key.get("Z"))

    def test_empty_text_gives_an_empty_key(self) -> None:
        self.assertEqual(len(frequency_guess("")), 0)

    def test_ties_break_deterministically(self) -> None:
        self.assertEqual(frequency_guess("BA"), frequency_guess("AB"))

    def test_it_is_only_a_starting_point(self) -> None:
        # On real ciphertext the frequency guess gets a few letters right and
        # most of them wrong. If this ever became a solved decryption the
        # module docstring would be lying, so the test states the expectation.
        guess = frequency_guess(CIPHER)
        correct = sum(
            1
            for cipher, plain in guess.mapping.items()
            if SAMPLE_KEY.get(cipher) == plain
        )
        self.assertLess(correct, 26)
        self.assertLess(default_scorer().word_coverage(guess.apply(CIPHER)), 0.6)


class TestIncrementalScoring(unittest.TestCase):
    """The incremental swap arithmetic is the easiest place in the module to
    introduce a silent bug, so it is checked against an independent full
    rescore from two directions."""

    def setUp(self) -> None:
        self.scorer = default_scorer()
        self.climber = _HillClimber(
            self.scorer.encode(CIPHER), self.scorer.table(), range(26)
        )

    def test_running_total_matches_a_full_rescore(self) -> None:
        rng = random.Random(31)
        for attempt in range(5):
            with self.subTest(attempt=attempt):
                start = _plain_indices(_random_key({}, rng))
                final, running, cut_short = self.climber.climb(start)
                self.assertFalse(cut_short)
                self.assertAlmostEqual(
                    running, self.climber.full_score(final), delta=1e-6
                )

    def test_the_climb_actually_improves_the_score(self) -> None:
        start = _plain_indices(_random_key({}, random.Random(5)))
        before = self.climber.full_score(start)
        final, running, _ = self.climber.climb(start)
        self.assertGreater(running, before)
        self.assertGreater(self.climber.passes, 1)
        self.assertGreater(self.climber.swap_tests, 300)

    def test_the_climb_really_stops_at_a_local_optimum(self) -> None:
        # Independent check of the whole incremental machinery: if a swap
        # delta were computed wrongly the climber would stop somewhere that a
        # full rescore can still improve on. Here we try all 325 swaps by
        # brute force, full rescore each time, and demand that none helps.
        #
        # The reference is deliberately full_score(final) and NOT the
        # climber's own running total. A climber with broken arithmetic
        # reports an inflated total, and comparing against that number would
        # let every swap look like no improvement -- the test would pass
        # precisely when the code is most broken. Measured: using `running`
        # here let four separate incremental-scoring mutations survive.
        #
        # The tolerance is for floating point only. Summing several hundred
        # log probabilities in two different orders disagrees in the last bit
        # or two (about 1e-13 on a total near -700), which can make a swap
        # look like an improvement of 2e-13. A swap the climber genuinely
        # missed is worth a substantial fraction of a log unit, so 1e-6
        # separates the two cases by seven orders of magnitude.
        tolerance = 1e-6
        start = _plain_indices(_random_key({}, random.Random(17)))
        final, running, _ = self.climber.climb(start)
        best = self.climber.full_score(final)
        self.assertAlmostEqual(running, best, delta=tolerance)
        for first in range(26):
            for second in range(first + 1, 26):
                swapped = list(final)
                swapped[first], swapped[second] = swapped[second], swapped[first]
                self.assertLessEqual(
                    self.climber.full_score(swapped),
                    best + tolerance,
                    f"swapping {ALPHABET[first]} and {ALPHABET[second]} still "
                    "improves the score, so the climb stopped too early",
                )

    def test_window_score_matches_an_independent_quadgram_sum(self) -> None:
        # The expected value is built here, from the scorer's table, by a
        # different route: decrypt first, then walk the plaintext. Comparing
        # window_score() with the module's own full_score() would just be the
        # same summation twice, and a transposed index inside it would cancel
        # out on both sides -- measured, that mutation survived such a test.
        plaintext = SAMPLE_KEY.apply(CIPHER)
        values = self.scorer.encode(plaintext)
        table = self.scorer.table()
        expected = 0.0
        for position in range(len(values) - 3):
            first, second, third, fourth = values[position: position + 4]
            expected += table[((first * 26 + second) * 26 + third) * 26 + fourth]
        self.assertAlmostEqual(window_score(CIPHER, SAMPLE_KEY), expected, delta=1e-9)

    def test_window_score_needs_a_complete_key(self) -> None:
        with self.assertRaises(ValueError) as caught:
            window_score(CIPHER, SubstitutionKey.from_string("QE"))
        self.assertTrue(str(caught.exception))

    def test_the_pass_limit_stops_a_climb_and_is_recorded(self) -> None:
        # The climb loop runs until a pass makes no improvement. That only
        # terminates because the incremental arithmetic is correct, so there
        # is a hard cap behind it. Here the cap is set to one pass to watch it
        # fire: the climb must stop, say that it stopped, and still return a
        # key whose reported score matches an independent rescore.
        start = _plain_indices(_random_key({}, random.Random(23)))
        final, running, cut_short = self.climber.climb(start, max_passes=1)
        self.assertEqual(self.climber.pass_limit_hits, 1)
        self.assertEqual(self.climber.passes, 1)
        self.assertFalse(cut_short, "the pass cap is not a time budget")
        self.assertAlmostEqual(running, self.climber.full_score(final), delta=1e-6)

    def test_a_normal_climb_never_reaches_the_pass_limit(self) -> None:
        start = _plain_indices(_random_key({}, random.Random(29)))
        self.climber.climb(start)
        self.assertEqual(self.climber.pass_limit_hits, 0)
        self.assertLess(self.climber.passes, MAX_CLIMB_PASSES)

    def test_a_text_shorter_than_one_window_has_nothing_to_climb(self) -> None:
        climber = _HillClimber(self.scorer.encode("ABC"), self.scorer.table(), range(26))
        self.assertEqual(climber.pairs, [])
        start = _plain_indices(_random_key({}, random.Random(1)))
        final, running, cut_short = climber.climb(start)
        self.assertEqual(running, 0.0)
        self.assertEqual(final, start)
        self.assertFalse(cut_short)


class TestCribs(unittest.TestCase):
    def setUp(self) -> None:
        self.position = PLAIN.find("MACHINE")
        self.assertGreater(self.position, 0, "test fixture: crib not in sample")

    def test_a_correct_crib_yields_the_correct_letters(self) -> None:
        key = apply_crib(CIPHER, "MACHINE", self.position)
        self.assertIsNotNone(key)
        assert key is not None
        for cipher, plain in key.mapping.items():
            self.assertEqual(SAMPLE_KEY.get(cipher), plain)

    def test_a_crib_that_cannot_fit_returns_none(self) -> None:
        # SEE has the pattern 0-1-1; ABC has 0-1-2, so no substitution can
        # turn one into the other.
        self.assertIsNone(apply_crib("ABC", "SEE", 0))

    def test_a_crib_cannot_reuse_a_plain_letter(self) -> None:
        # AB -> EE would need two cipher letters to mean the same plain
        # letter, which a bijection forbids.
        self.assertIsNone(apply_crib("AB", "EE", 0))

    def test_crib_positions_contains_the_true_position(self) -> None:
        places = crib_positions(CIPHER, "MACHINE")
        self.assertIn(self.position, places)
        self.assertLess(
            len(places),
            len(CIPHER),
            "a crib with a repeated letter must rule some positions out",
        )

    def test_crib_positions_are_all_genuinely_consistent(self) -> None:
        for position in crib_positions(CIPHER, "MACHINE"):
            self.assertIsNotNone(apply_crib(CIPHER, "MACHINE", position))

    def test_a_crib_that_fits_nowhere_reports_nothing(self) -> None:
        # Honest failure: no exception, no invented answer, an empty list.
        self.assertEqual(crib_positions("ABCDEF", "SEE"), [])
        self.assertEqual(crib_positions("ABC", "LONGERTHANTHETEXT"), [])

    def test_known_letters_prune_positions(self) -> None:
        known = SubstitutionKey({CIPHER[self.position]: "M"})
        pruned = crib_positions(CIPHER, "MACHINE", known=known)
        self.assertIn(self.position, pruned)
        self.assertLess(len(pruned), len(crib_positions(CIPHER, "MACHINE")))

    def test_a_crib_contradicting_known_letters_returns_none(self) -> None:
        wrong = SubstitutionKey({CIPHER[self.position]: "Q"})
        self.assertIsNone(
            apply_crib(CIPHER, "MACHINE", self.position, known=wrong)
        )

    def test_invalid_crib_input_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            apply_crib(CIPHER, "!!!", 0)
        self.assertTrue(str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            apply_crib("ABC", "SEEN", 0)
        self.assertTrue(str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            apply_crib("ABC", "SEE", -1)
        self.assertTrue(str(caught.exception))


class TestWordPatterns(unittest.TestCase):
    def test_the_true_word_is_among_the_pattern_matches(self) -> None:
        cipher_word = encrypt("PEOPLE", SAMPLE_KEY)
        matches = analyse_words([cipher_word])
        self.assertIn("PEOPLE", matches[cipher_word])

    def test_matches_all_share_the_pattern(self) -> None:
        cipher_word = encrypt("ATTACK", SAMPLE_KEY)
        for word in analyse_words([cipher_word])[cipher_word]:
            self.assertEqual(len(word), 6)
            self.assertEqual(word[1], word[2])
            self.assertEqual(word[0], word[3])

    def test_known_letters_narrow_the_matches(self) -> None:
        cipher_word = encrypt("PEOPLE", SAMPLE_KEY)
        wide = analyse_words([cipher_word])[cipher_word]
        known = SubstitutionKey({cipher_word[0]: "P"})
        narrow = analyse_words([cipher_word], known=known)[cipher_word]
        self.assertIn("PEOPLE", narrow)
        self.assertLessEqual(len(narrow), len(wide))
        for word in narrow:
            self.assertTrue(word.startswith("P"))

    def test_it_refuses_to_split_a_string_for_you(self) -> None:
        # The whole point: five-letter groups are not words, so passing a
        # block of ciphertext must be an error rather than a silent split.
        with self.assertRaises(ValueError) as caught:
            analyse_words("QEBOB FP")  # type: ignore[arg-type]
        self.assertTrue(str(caught.exception))

    def test_unknown_shapes_simply_return_nothing(self) -> None:
        self.assertEqual(analyse_words(["QQQQQQQQQQQQQQQQQQ"]), {"QQQQQQQQQQQQQQQQQQ": []})
        self.assertEqual(analyse_words([]), {})


class TestSolve(unittest.TestCase):
    """Requirement 6: recover a known key from ciphertext long enough to be
    solvable, and assert on the recovered plaintext."""

    def test_solves_four_hundred_letters(self) -> None:
        found = solve(CIPHER, restarts=25, seed=7)
        best = found.best()
        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best.plaintext, PLAIN)
        self.assertGreater(best.diagnostics["word_coverage"], 0.9)
        self.assertEqual(best.confidence(), "strong")
        self.assertEqual(best.method, METHOD)

    def test_the_reported_key_reproduces_the_plaintext(self) -> None:
        best = solve(CIPHER, restarts=25, seed=7).best()
        assert best is not None
        alphabet = best.key.split("=", 1)[1]
        self.assertEqual(decrypt(CIPHER, alphabet), PLAIN)

    def test_several_restarts_agree(self) -> None:
        best = solve(CIPHER, restarts=25, seed=7).best()
        assert best is not None
        self.assertGreater(
            best.diagnostics["agreements"],
            1,
            "one restart agreeing with itself is not evidence of convergence",
        )
        self.assertEqual(best.diagnostics["restarts_run"], 25)
        self.assertFalse(best.diagnostics["time_budget_hit"])

    def test_display_preserves_the_original_layout(self) -> None:
        laid_out = group_text(CIPHER, size=5, per_line=8)
        best = solve(laid_out, restarts=25, seed=7).best()
        assert best is not None
        self.assertIsNotNone(best.display)
        assert best.display is not None
        self.assertEqual(letters_only(best.display), PLAIN)
        self.assertEqual(len(best.display), len(laid_out))
        self.assertEqual(best.display.count("\n"), laid_out.count("\n"))

    def test_layout_does_not_change_the_answer(self) -> None:
        messy = group_text(CIPHER.lower(), size=5, per_line=8) + "\n  ...  "
        clean = solve(CIPHER, restarts=12, seed=3).best()
        dirty = solve(messy, restarts=12, seed=3).best()
        assert clean is not None and dirty is not None
        self.assertEqual(clean.plaintext, dirty.plaintext)

    def test_accepts_a_normalized_text(self) -> None:
        best = solve(normalize(CIPHER), restarts=12, seed=3).best()
        assert best is not None
        self.assertEqual(best.plaintext, PLAIN)

    def test_a_seed_makes_a_run_reproducible(self) -> None:
        first = [(c.key, c.score) for c in solve(CIPHER, restarts=8, seed=11)]
        second = [(c.key, c.score) for c in solve(CIPHER, restarts=8, seed=11)]
        self.assertEqual(first, second)

    def test_top_limits_the_report_not_the_search(self) -> None:
        found = solve(CIPHER, restarts=12, seed=3, top=2)
        self.assertLessEqual(len(found), 2)
        best = found.best()
        assert best is not None
        self.assertEqual(best.diagnostics["restarts_run"], 12)

    def test_a_correct_crib_is_held_and_helps(self) -> None:
        truth = SAMPLE_KEY.mapping
        crib = SubstitutionKey({key: truth[key] for key in list(truth)[:5]})
        found = solve(CIPHER, restarts=6, seed=7, fixed=crib)
        best = found.best()
        assert best is not None
        self.assertEqual(best.plaintext, PLAIN)
        for candidate in found:
            reported = SubstitutionKey.from_alphabet(candidate.key.split("=", 1)[1])
            for cipher, plain in crib.mapping.items():
                self.assertEqual(reported.get(cipher), plain)
        self.assertIn("held_fixed", best.diagnostics)

    def test_start_key_must_not_contradict_fixed(self) -> None:
        with self.assertRaises(ValueError) as caught:
            solve(CIPHER, restarts=2, fixed="AB", start_key="AC")
        self.assertTrue(str(caught.exception))

    def test_start_key_is_used(self) -> None:
        best = solve(
            CIPHER, restarts=1, seed=1, start_key=SAMPLE_KEY
        ).best()
        assert best is not None
        self.assertEqual(best.plaintext, PLAIN)

    def test_time_budget_stops_the_search_and_says_so(self) -> None:
        began = time.monotonic()
        found = solve(CIPHER, restarts=500, seed=7, time_budget=0.05)
        elapsed = time.monotonic() - began
        best = found.best()
        assert best is not None
        self.assertTrue(best.diagnostics["time_budget_hit"])
        self.assertLess(best.diagnostics["restarts_run"], 500)
        self.assertLess(elapsed, 20.0)
        self.assertGreaterEqual(len(found), 1)

    def test_empty_input_gives_an_empty_candidate_set(self) -> None:
        self.assertEqual(len(solve("")), 0)
        self.assertEqual(len(solve("1234 !!!")), 0)
        self.assertIsNone(solve("").best())

    def test_short_input_is_flagged_rather_than_dressed_up(self) -> None:
        best = solve("ABCDEFGH", restarts=4, seed=1).best()
        assert best is not None
        self.assertIn("short_text_warning", best.diagnostics)
        self.assertIn(str(RELIABLE_CLIMB_LETTERS), best.diagnostics["short_text_warning"])

    def test_text_shorter_than_a_window_is_not_searched(self) -> None:
        found = solve("ABC", restarts=25, seed=1)
        best = found.best()
        assert best is not None
        self.assertEqual(best.diagnostics["restarts_run"], 1)
        self.assertIs(best.diagnostics["searchable"], False)

    def test_invalid_options_raise(self) -> None:
        for options in ({"restarts": 0}, {"top": 0}, {"time_budget": 0.0}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError) as caught:
                    solve(CIPHER, **options)  # type: ignore[arg-type]
                self.assertTrue(str(caught.exception))
        with self.assertRaises(ValueError):
            solve(1234)  # type: ignore[arg-type]

    def test_it_never_touches_the_global_random_module(self) -> None:
        random.seed(1)
        before = random.random()
        random.seed(1)
        solve(CIPHER, restarts=6, seed=None)
        self.assertEqual(random.random(), before)


class TestFailureModes(unittest.TestCase):
    """Requirement 7: watch the tool fail, and check it says so."""

    def test_random_letters_are_not_reported_as_solved(self) -> None:
        rng = random.Random(7)
        noise = "".join(rng.choice(ALPHABET) for _ in range(300))
        found = solve(noise, restarts=8, seed=3)
        best = found.best()
        assert best is not None

        # The climber always returns something. What it must not do is claim
        # the something is English.
        self.assertIn(best.confidence(), {"weak", "unlikely"})
        self.assertLess(best.diagnostics["word_coverage"], 0.5)

        # And the honest evidence of non-convergence is right there: every
        # restart landed somewhere different, so nothing agreed with anything.
        self.assertEqual(best.diagnostics["agreements"], 1)
        self.assertEqual(
            best.diagnostics["distinct_local_optima"],
            best.diagnostics["restarts_run"],
        )

    def test_a_wrong_crib_produces_a_wrong_answer_that_looks_wrong(self) -> None:
        # Force four cipher letters to plain letters they certainly are not.
        truth = SAMPLE_KEY
        wrong_pairs = {}
        for cipher, plain in zip(sorted(set(CIPHER))[:4], "QZJX"):
            self.assertNotEqual(truth.get(cipher), plain)
            wrong_pairs[cipher] = plain
        wrong = SubstitutionKey(wrong_pairs)

        found = solve(CIPHER, restarts=12, seed=5, fixed=wrong)
        best = found.best()
        assert best is not None
        self.assertNotEqual(best.plaintext, PLAIN)
        self.assertNotEqual(best.confidence(), "strong")
        self.assertLess(best.diagnostics["word_coverage"], 0.5)
        # The crib it was told to trust is reported, so a human can see what
        # to blame.
        self.assertIn("held_fixed", best.diagnostics)

    def test_a_correct_solve_scores_far_above_a_wrong_one(self) -> None:
        # The separation is the whole basis for ranking candidates, so it is
        # worth asserting rather than assuming.
        scorer = default_scorer()
        right = scorer.normalised(PLAIN)
        wrong = scorer.normalised(SubstitutionKey.from_alphabet(ALPHABET).apply(CIPHER))
        self.assertGreater(right - wrong, 1.0)

    def test_decrypting_with_the_wrong_key_is_visibly_wrong(self) -> None:
        scrambled = SubstitutionKey.from_alphabet(ALPHABET[::-1])
        junk = decrypt(CIPHER, scrambled)
        candidate = Candidate(
            method=METHOD, key="key=" + ALPHABET[::-1], score=0.0, plaintext=junk
        )
        candidate.diagnostics["normalised_score"] = default_scorer().normalised(junk)
        candidate.diagnostics["word_coverage"] = default_scorer().word_coverage(junk)
        self.assertIn(candidate.confidence(), {"weak", "unlikely"})


if __name__ == "__main__":
    unittest.main()
