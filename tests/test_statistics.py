"""Tests for ciphertext statistics and the heuristic family report.

The measurements are tested against values that can be worked out by hand.
The heuristics are tested for the property that actually matters: that they
point at the right family for texts whose cipher we know, and that they never
claim certainty.
"""

from __future__ import annotations

import unittest

from cipher_tool.normalize import ALPHABET, letters_only
from cipher_tool.scoring import corpus_files
from cipher_tool.statistics import (
    _kasiski_shortlist,
    analyse,
    chi_squared_english,
    divisors,
    find_repeats,
    ic_by_period,
    index_of_coincidence,
    kasiski_factor_votes,
    letter_counts,
    letter_frequencies,
    normalised_ic,
    prime_factors,
    render_report,
    repeat_distances,
    repeated_ngrams,
)


def sample_english(length: int = 900) -> str:
    """Real English, taken from our own corpus, for statistical tests."""
    text = corpus_files()[3].read_text(encoding="utf-8")
    return letters_only(text)[:length]


def vigenere(text: str, key: str) -> str:
    return "".join(
        ALPHABET[(ord(c) - 65 + ord(key[i % len(key)]) - 65) % 26]
        for i, c in enumerate(text)
    )


def caesar(text: str, shift: int) -> str:
    return "".join(ALPHABET[(ord(c) - 65 + shift) % 26] for c in text)


class TestIndexOfCoincidence(unittest.TestCase):
    def test_hand_computed(self) -> None:
        # "AABB": pairs of equal letters = 2*1 + 2*1 = 4; total pairs = 4*3.
        self.assertAlmostEqual(index_of_coincidence("AABB"), 4 / 12)

    def test_all_same_letter_is_one(self) -> None:
        self.assertAlmostEqual(index_of_coincidence("AAAAA"), 1.0)

    def test_all_distinct_is_zero(self) -> None:
        self.assertAlmostEqual(index_of_coincidence("ABCDE"), 0.0)

    def test_undefined_for_short_text(self) -> None:
        self.assertEqual(index_of_coincidence(""), 0.0)
        self.assertEqual(index_of_coincidence("A"), 0.0)

    def test_english_is_near_the_reference(self) -> None:
        value = index_of_coincidence(sample_english())
        self.assertGreater(value, 0.058)
        self.assertLess(value, 0.078)

    def test_monoalphabetic_ciphers_preserve_it(self) -> None:
        # This invariance is the reason IC is the first measurement we take.
        english = sample_english()
        self.assertAlmostEqual(
            index_of_coincidence(english),
            index_of_coincidence(caesar(english, 11)),
            places=12,
        )

    def test_vigenere_flattens_it(self) -> None:
        english = sample_english()
        self.assertLess(index_of_coincidence(vigenere(english, "KEYWORD")), 0.050)

    def test_normalised_ic(self) -> None:
        self.assertAlmostEqual(normalised_ic("AABB"), 26 * 4 / 12)


class TestChiSquared(unittest.TestCase):
    def test_english_is_small(self) -> None:
        self.assertLess(chi_squared_english(sample_english()), 0.12)

    def test_transposition_leaves_it_unchanged(self) -> None:
        english = sample_english()
        reversed_text = english[::-1]
        self.assertAlmostEqual(
            chi_squared_english(english),
            chi_squared_english(reversed_text),
            places=12,
        )

    def test_substitution_raises_it(self) -> None:
        self.assertGreater(chi_squared_english(caesar(sample_english(), 7)), 1.0)

    def test_empty_is_infinite(self) -> None:
        self.assertEqual(chi_squared_english(""), float("inf"))


class TestCounts(unittest.TestCase):
    def test_letter_counts_include_zeros(self) -> None:
        counts = letter_counts("AAB")
        self.assertEqual(counts["A"], 2)
        self.assertEqual(counts["B"], 1)
        self.assertEqual(counts["Z"], 0)
        self.assertEqual(len(counts), 26)

    def test_frequencies_are_percentages(self) -> None:
        frequencies = letter_frequencies("AAAB")
        self.assertAlmostEqual(frequencies["A"], 75.0)
        self.assertAlmostEqual(frequencies["B"], 25.0)

    def test_frequencies_of_empty_text(self) -> None:
        self.assertEqual(set(letter_frequencies("").values()), {0.0})


class TestFactors(unittest.TestCase):
    def test_divisors(self) -> None:
        self.assertEqual(divisors(12), [1, 2, 3, 4, 6, 12])
        self.assertEqual(divisors(13), [1, 13])
        self.assertEqual(divisors(1), [1])

    def test_divisors_of_non_positive(self) -> None:
        self.assertEqual(divisors(0), [])
        self.assertEqual(divisors(-6), [])

    def test_prime_factors(self) -> None:
        self.assertEqual(prime_factors(900), [2, 2, 3, 3, 5, 5])
        self.assertEqual(prime_factors(97), [97])
        self.assertEqual(prime_factors(1), [])
        self.assertEqual(prime_factors(0), [])


class TestRepeats(unittest.TestCase):
    def test_finds_a_repeat(self) -> None:
        found = repeated_ngrams("ABCXABC", 3)
        self.assertEqual(found, {"ABC": [0, 4]})

    def test_ignores_singletons(self) -> None:
        self.assertEqual(repeated_ngrams("ABCDEF", 3), {})

    def test_size_larger_than_text(self) -> None:
        self.assertEqual(repeated_ngrams("AB", 5), {})

    def test_distances_are_consecutive_gaps_only(self) -> None:
        # Positions 0, 4, 10 give gaps 4 and 6, NOT also 10: every pairwise
        # distance is a sum of consecutive gaps, so counting them all would
        # double-count the same evidence.
        self.assertEqual(repeat_distances([0, 4, 10]), [4, 6])

    def test_find_repeats_sorted_by_frequency(self) -> None:
        found = find_repeats("ABABABXYXY", 2)
        self.assertEqual(found[0].text, "AB")
        self.assertEqual(found[0].count, 3)


class TestKasiski(unittest.TestCase):
    def test_recovers_a_known_key_length(self) -> None:
        ciphertext = vigenere(sample_english(1200), "KEYWORD")  # length 7
        shortlist = _kasiski_shortlist(kasiski_factor_votes(ciphertext))
        self.assertEqual(shortlist[0][0], 7)

    def test_no_repeats_gives_no_votes(self) -> None:
        self.assertEqual(kasiski_factor_votes("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), {})

    def test_shortlist_demotes_multiples_of_a_winner(self) -> None:
        from collections import Counter

        votes = Counter({3: 100, 6: 90, 9: 80, 5: 40})
        shortlist = dict(_kasiski_shortlist(votes))
        self.assertIn(3, shortlist)
        self.assertIn(5, shortlist)
        self.assertNotIn(6, shortlist)
        self.assertNotIn(9, shortlist)


class TestIcByPeriod(unittest.TestCase):
    def test_the_true_period_stands_out(self) -> None:
        ciphertext = vigenere(sample_english(1200), "KEYWORD")
        periods = dict(ic_by_period(ciphertext, 20))
        self.assertGreater(periods[7], 0.058)
        for period in (2, 3, 4, 5, 6, 8, 9):
            self.assertLess(periods[period], 0.050)

    def test_skips_periods_that_leave_tiny_columns(self) -> None:
        # 100 letters at period 6 leaves 16 per column, which is too few for
        # IC to mean anything; reporting it anyway is how people convince
        # themselves of a wrong key length.
        periods = [row[0] for row in ic_by_period("A" * 100, 20)]
        self.assertEqual(max(periods), 5)

    def test_short_text_gives_nothing(self) -> None:
        self.assertEqual(ic_by_period("ABCDE", 20), [])


class TestAnalyse(unittest.TestCase):
    def test_empty_input_is_handled(self) -> None:
        stats = analyse("1234 !!!")
        self.assertEqual(stats.length, 0)
        self.assertEqual(stats.hypotheses[0].family, "none")
        self.assertIn("no alphabetic", stats.hypotheses[0].reason)

    def test_five_letter_grouping_is_flagged_as_formatting(self) -> None:
        stats = analyse("HEALI OPASD EHANS XQKTR")
        self.assertEqual(stats.uniform_group_length, 5)
        self.assertIn("NOT word boundaries", render_report(stats))

    def test_original_is_retained(self) -> None:
        raw = "Hello, world!"
        self.assertEqual(analyse(raw).original, raw)

    def test_counts_and_uniqueness(self) -> None:
        stats = analyse("ABCABC")
        self.assertEqual(stats.length, 6)
        self.assertEqual(stats.unique_letters, 3)
        self.assertEqual(len(stats.missing_letters), 23)

    def test_length_factors(self) -> None:
        stats = analyse("A" * 900)
        self.assertEqual(stats.length_prime_factors, [2, 2, 3, 3, 5, 5])
        self.assertIn(30, stats.length_divisors)


class TestHypotheses(unittest.TestCase):
    """The heuristics must point the right way -- and stay heuristics."""

    def families(self, text: str) -> str:
        return " | ".join(h.family for h in analyse(text).hypotheses)

    def test_transposition_is_suggested_for_reordered_english(self) -> None:
        self.assertIn("Transposition", self.families(sample_english()[::-1]))

    def test_substitution_is_suggested_for_a_caesar_shift(self) -> None:
        families = self.families(caesar(sample_english(), 7))
        self.assertIn("Monoalphabetic", families)
        self.assertNotIn("Transposition", families)

    def test_polyalphabetic_is_suggested_for_vigenere(self) -> None:
        families = self.families(vigenere(sample_english(), "KEYWORD"))
        self.assertIn("Polyalphabetic", families)

    def test_agreeing_key_length_evidence_is_reported_once(self) -> None:
        # Kasiski and column-IC both find 7 here. Reporting that as two
        # findings would overstate the amount of independent evidence.
        stats = analyse(vigenere(sample_english(1200), "KEYWORD"))
        key_length_findings = [
            h for h in stats.hypotheses if "Repeating key" in h.family
        ]
        self.assertEqual(len(key_length_findings), 1)
        self.assertIn("length 7", key_length_findings[0].family)
        self.assertIn("agree", key_length_findings[0].reason)

    def test_short_text_is_downgraded_and_says_why(self) -> None:
        stats = analyse(caesar(sample_english(60), 7))
        monoalphabetic = [
            h for h in stats.hypotheses if "Monoalphabetic" in h.family
        ][0]
        self.assertEqual(monoalphabetic.confidence, "possible")
        self.assertIn("unreliable", monoalphabetic.reason)

    def test_no_hypothesis_ever_claims_certainty(self) -> None:
        for text in (sample_english(), caesar(sample_english(), 7),
                     vigenere(sample_english(), "KEYWORD")):
            for hypothesis in analyse(text).hypotheses:
                self.assertIn(hypothesis.confidence,
                              {"consider", "possible", "likely"})

    def test_report_labels_the_suggestions_as_heuristics(self) -> None:
        report = render_report(analyse(sample_english()))
        self.assertIn("HEURISTIC", report)
        self.assertIn("not an identification", report)

    def test_report_renders_for_every_family(self) -> None:
        for text in ("", "HEALI OPASD", sample_english(),
                     caesar(sample_english(), 7),
                     vigenere(sample_english(), "KEYWORD"),
                     "12345 67890 12345 67890 12345 67890 12345 67890"):
            self.assertIsInstance(render_report(analyse(text)), str)


class TestLowAlphabetBlock(unittest.TestCase):
    """Finding a stretch that was never prose.

    From the 2017 challenge 5A: 900 letters of English, then 1,500 letters
    of a steganographic frieze -- black and white tiles, enciphered with the
    words -- then 300 more of English. MEASURED distinct letters per
    100-letter window: 18 to 23 across the prose, exactly 2 across the
    frieze.

    This matters more than it looks. The block does not merely spoil the
    score of a correct answer; it spoils the SEARCH, because a key that is
    wrong everywhere scored better (-1.52 per letter) than the correct key
    (-2.07), which has to carry 1,500 letters of tiles. The right answer was
    not findable while the block was in the text.
    """

    def test_it_finds_the_block(self) -> None:
        from cipher_tool.statistics import find_low_alphabet_block

        prose = letters_only(corpus_files()[0].read_text(encoding="utf-8"))
        text = prose[:900] + "WWWB" * 375 + prose[900:1200]
        span = find_low_alphabet_block(text)
        self.assertIsNotNone(span)
        start, end = span
        self.assertLess(abs(start - 900), 150)
        self.assertLess(abs(end - 2400), 150)

    def test_ordinary_english_has_no_such_block(self) -> None:
        from cipher_tool.statistics import find_low_alphabet_block

        prose = letters_only(corpus_files()[0].read_text(encoding="utf-8"))
        self.assertIsNone(find_low_alphabet_block(prose[:2000]))

    def test_a_short_run_is_not_worth_reporting(self) -> None:
        """Two dozen letters of numbers spelled out is not a block."""
        from cipher_tool.statistics import find_low_alphabet_block

        prose = letters_only(corpus_files()[0].read_text(encoding="utf-8"))
        text = prose[:1000] + "ABABAB" * 8 + prose[1000:2000]
        self.assertIsNone(find_low_alphabet_block(text))

    def test_random_letters_have_no_block(self) -> None:
        import random

        from cipher_tool.statistics import find_low_alphabet_block

        generator = random.Random(2)
        noise = "".join(generator.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                        for _ in range(2000))
        self.assertIsNone(find_low_alphabet_block(noise))


if __name__ == "__main__":
    unittest.main()
