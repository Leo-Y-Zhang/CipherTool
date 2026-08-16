"""Tests for input normalisation.

The rule these tests defend: the original input is never destroyed, and
whitespace is never treated as a plaintext word boundary.
"""

from __future__ import annotations

import unittest

from cipher_tool.normalize import (
    NormalizedText,
    chunks,
    clean_key,
    columns,
    from_numbers,
    group_text,
    letters_only,
    normalize,
    strip_bom,
    to_numbers,
)


class TestLettersOnly(unittest.TestCase):
    def test_uppercases_and_strips(self) -> None:
        self.assertEqual(letters_only("Hello, World!"), "HELLOWORLD")

    def test_five_letter_grouping(self) -> None:
        self.assertEqual(letters_only("HEALI OPASD EHANS"), "HEALIOPASDEHANS")

    def test_line_breaks_and_tabs(self) -> None:
        self.assertEqual(letters_only("AB\nCD\tEF\r\nGH"), "ABCDEFGH")

    def test_digits_and_punctuation_removed(self) -> None:
        self.assertEqual(letters_only("A1B2-C3.D4"), "ABCD")

    def test_accents_folded_not_dropped(self) -> None:
        # A paste from a PDF must not silently lose letters. The accented
        # characters are written as escapes to keep this file pure ASCII.
        self.assertEqual(letters_only("CAF\u00c9 NA\u00cfVE"), "CAFENAIVE")

    def test_empty(self) -> None:
        self.assertEqual(letters_only(""), "")
        self.assertEqual(letters_only("1234 !!!"), "")


class TestNormalize(unittest.TestCase):
    def test_original_is_preserved_exactly(self) -> None:
        raw = "  Attack at dawn!\n\nSecond line.  "
        result = normalize(raw)
        self.assertEqual(result.original, raw)
        self.assertEqual(result.letters, "ATTACKATDAWNSECONDLINE")

    def test_positions_point_at_the_right_characters(self) -> None:
        raw = "a-b-c"
        result = normalize(raw)
        self.assertEqual(result.letters, "ABC")
        self.assertEqual(result.positions, (0, 2, 4))
        for index, position in enumerate(result.positions):
            self.assertEqual(raw[position].upper(), result.letters[index])

    def test_relayout_restores_punctuation(self) -> None:
        # "Xyz, abc!" has six letters at positions 0,1,2,5,6,7. Pouring
        # "HELLOW" back in must keep the comma, the space and the bang.
        result = normalize("Xyz, abc!")
        self.assertEqual(result.letters, "XYZABC")
        self.assertEqual(result.relayout("HELLOW"), "HEL, LOW!")

    def test_relayout_keeps_five_letter_grouping(self) -> None:
        result = normalize("HEALI OPASD")
        self.assertEqual(result.relayout("ATTACKATDA"), "ATTAC KATDA")

    def test_relayout_rejects_wrong_length(self) -> None:
        result = normalize("ABCDE")
        with self.assertRaises(ValueError) as context:
            result.relayout("ABC")
        self.assertTrue(str(context.exception))

    def test_groups_ignore_punctuation_but_are_reported(self) -> None:
        result = normalize("HEALI OPASD EHANS")
        self.assertEqual(result.groups, ("HEALI", "OPASD", "EHANS"))
        self.assertEqual(result.uniform_group_length(), 5)

    def test_mixed_group_lengths_report_none(self) -> None:
        result = normalize("THE QUICK BROWN FOX")
        self.assertIsNone(result.uniform_group_length())

    def test_empty_input(self) -> None:
        result = normalize("")
        self.assertTrue(result.is_empty)
        self.assertEqual(result.length, 0)
        self.assertEqual(result.groups, ())
        self.assertIsNone(result.uniform_group_length())

    def test_length_matches_letters(self) -> None:
        result = normalize("Hello there, friend.")
        self.assertEqual(len(result), len(result.letters))
        self.assertEqual(result.letters, "HELLOTHEREFRIEND")
        self.assertEqual(result.length, 16)


class TestByteOrderMark(unittest.TestCase):
    """A BOM is how the file was saved, not something the sender wrote.

    Notepad writes one for every file saved as "UTF-8", so it arrives often.
    Left in the normalised text it reaches the terminal, where a code page
    that cannot represent it raises UnicodeEncodeError -- a crash blamed on
    the toolkit for something the user did not do.
    """

    def test_strip_bom_removes_marks_anywhere_not_just_the_first(self) -> None:
        self.assertEqual(strip_bom("\ufeffAB\ufeffCD\ufeff"), "ABCD")

    def test_strip_bom_leaves_ordinary_text_alone(self) -> None:
        self.assertEqual(strip_bom("Attack at dawn!"), "Attack at dawn!")

    def test_normalize_keeps_no_bom_in_the_original(self) -> None:
        result = normalize("\ufeffHEALI OPASD")
        self.assertNotIn("\ufeff", result.original)
        self.assertEqual(result.original, "HEALI OPASD")
        self.assertEqual(result.letters, "HEALIOPASD")

    def test_positions_and_relayout_survive_a_bom(self) -> None:
        # The position map indexes `original`, so stripping the BOM before
        # building it is what keeps relayout aligned.
        result = normalize("\ufeffa-b")
        self.assertEqual(result.letters, "AB")
        self.assertEqual(result.positions, (0, 2))
        self.assertEqual(result.relayout("XY"), "X-Y")


class TestGrouping(unittest.TestCase):
    def test_group_text(self) -> None:
        self.assertEqual(group_text("ATTACKATDAWN", 5, 10), "ATTAC KATDA WN")

    def test_group_text_wraps_lines(self) -> None:
        text = "A" * 60
        rendered = group_text(text, 5, 4)
        self.assertEqual(len(rendered.splitlines()), 3)

    def test_group_size_zero_is_identity(self) -> None:
        self.assertEqual(group_text("ABCDEF", 0), "ABCDEF")

    def test_chunks(self) -> None:
        self.assertEqual(list(chunks("ABCDEFG", 3)), ["ABC", "DEF", "G"])

    def test_chunks_rejects_zero(self) -> None:
        with self.assertRaises(ValueError):
            list(chunks("ABC", 0))


class TestColumns(unittest.TestCase):
    def test_interleaved_split(self) -> None:
        self.assertEqual(columns("ABCDEFG", 3), ["ADG", "BE", "CF"])

    def test_one_column_is_identity(self) -> None:
        self.assertEqual(columns("ABCDEF", 1), ["ABCDEF"])

    def test_rejects_zero(self) -> None:
        with self.assertRaises(ValueError):
            columns("ABC", 0)


class TestNumbers(unittest.TestCase):
    def test_round_trip(self) -> None:
        self.assertEqual(to_numbers("ABZ"), [0, 1, 25])
        self.assertEqual(from_numbers([0, 1, 25]), "ABZ")

    def test_from_numbers_reduces_modulo_26(self) -> None:
        self.assertEqual(from_numbers([26, 27, -1]), "ABZ")

    def test_clean_key(self) -> None:
        self.assertEqual(clean_key("le mon!"), "LEMON")


if __name__ == "__main__":
    unittest.main()
