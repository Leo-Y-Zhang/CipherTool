"""Tests for word pattern signatures and pattern matching."""

from __future__ import annotations

import unittest

from cipher_tool.patterns import (
    PatternIndex,
    has_repeat,
    mapping_from_pair,
    match_word,
    pattern_signature,
    rank_by_selectivity,
    signature_selectivity,
)


class TestSignature(unittest.TestCase):
    def test_worked_example(self) -> None:
        self.assertEqual(pattern_signature("HELLO"), "0-1-2-2-3")

    def test_more_examples(self) -> None:
        self.assertEqual(pattern_signature("PEOPLE"), "0-1-2-0-3-1")
        self.assertEqual(pattern_signature("ATTACK"), "0-1-1-0-2-3")
        self.assertEqual(pattern_signature("ABC"), "0-1-2")

    def test_case_insensitive(self) -> None:
        self.assertEqual(pattern_signature("hello"), pattern_signature("HELLO"))

    def test_substitution_preserves_the_signature(self) -> None:
        # This is the whole reason the technique works.
        table = str.maketrans("HELO", "QIVA")
        self.assertEqual(
            pattern_signature("HELLO"),
            pattern_signature("HELLO".translate(table)),
        )

    def test_empty(self) -> None:
        self.assertEqual(pattern_signature(""), "")

    def test_separator_avoids_double_digit_ambiguity(self) -> None:
        # Without a separator, an eleven-distinct-letter word would produce
        # digits that could be read two ways.
        long_word = "ABCDEFGHIJKLA"
        self.assertEqual(
            pattern_signature(long_word),
            "0-1-2-3-4-5-6-7-8-9-10-11-0",
        )

    def test_has_repeat(self) -> None:
        self.assertTrue(has_repeat("HELLO"))
        self.assertFalse(has_repeat("ABCDE"))

    def test_selectivity(self) -> None:
        self.assertEqual(signature_selectivity(pattern_signature("ABCDE")), 0)
        self.assertEqual(signature_selectivity(pattern_signature("HELLO")), 1)
        self.assertEqual(signature_selectivity(pattern_signature("PEOPLE")), 2)


class TestMappingFromPair(unittest.TestCase):
    def test_consistent_pair(self) -> None:
        self.assertEqual(
            mapping_from_pair("QIVVA", "HELLO"),
            {"Q": "H", "I": "E", "V": "L", "A": "O"},
        )

    def test_length_mismatch(self) -> None:
        self.assertIsNone(mapping_from_pair("ABC", "ABCD"))

    def test_one_cipher_letter_cannot_mean_two_plain_letters(self) -> None:
        self.assertIsNone(mapping_from_pair("AA", "BC"))

    def test_two_cipher_letters_cannot_mean_the_same_plain_letter(self) -> None:
        # A substitution alphabet is a bijection, so this is impossible even
        # though a naive forward-only check would accept it.
        self.assertIsNone(mapping_from_pair("BC", "AA"))


class TestPatternIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = PatternIndex(
            ["HELLO", "PEOPLE", "ATTACK", "MELLOW", "CELLO", "THE", "AND"]
        )

    def test_matches_by_shape(self) -> None:
        matches = self.index.matches("QIVVA")
        self.assertIn("HELLO", matches)
        self.assertIn("CELLO", matches)
        self.assertNotIn("PEOPLE", matches)

    def test_no_match(self) -> None:
        self.assertEqual(self.index.matches("QQQQQQQQ"), [])

    def test_empty_word(self) -> None:
        self.assertEqual(self.index.matches(""), [])

    def test_deduplicates_and_cleans(self) -> None:
        index = PatternIndex(["hello", "HELLO", "he-llo"])
        self.assertEqual(len(index), 1)
        self.assertIn("HELLO", index)

    def test_results_are_deterministic(self) -> None:
        first = PatternIndex(["CELLO", "HELLO"]).matches("QIVVA")
        second = PatternIndex(["HELLO", "CELLO"]).matches("QIVVA")
        self.assertEqual(first, second)


class TestMatchWord(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = PatternIndex(["HELLO", "CELLO", "MELLOW"])

    def test_returns_mappings(self) -> None:
        results = match_word("QIVVA", self.index)
        words = {result.plain_word for result in results}
        self.assertEqual(words, {"HELLO", "CELLO"})

    def test_known_mapping_filters_candidates(self) -> None:
        results = match_word("QIVVA", self.index, known={"Q": "H"})
        self.assertEqual([r.plain_word for r in results], ["HELLO"])

    def test_known_mapping_rejects_a_backwards_conflict(self) -> None:
        # If some other cipher letter already means H, then Q cannot also
        # mean H, so HELLO must be ruled out.
        results = match_word("QIVVA", self.index, known={"Z": "H"})
        self.assertEqual([r.plain_word for r in results], ["CELLO"])

    def test_limit(self) -> None:
        self.assertEqual(len(match_word("QIVVA", self.index, limit=1)), 1)

    def test_constrains_count(self) -> None:
        result = match_word("QIVVA", self.index)[0]
        self.assertEqual(result.constrains, 4)


class TestRankBySelectivity(unittest.TestCase):
    def test_most_informative_first(self) -> None:
        ranked = rank_by_selectivity(["ABC", "PEOPLE", "HELLO"])
        self.assertEqual(ranked[0], "PEOPLE")
        self.assertEqual(ranked[1], "HELLO")
        self.assertEqual(ranked[2], "ABC")

    def test_deduplicates(self) -> None:
        self.assertEqual(rank_by_selectivity(["ABC", "abc", "A-B-C"]), ["ABC"])

    def test_empty(self) -> None:
        self.assertEqual(rank_by_selectivity([]), [])


if __name__ == "__main__":
    unittest.main()
