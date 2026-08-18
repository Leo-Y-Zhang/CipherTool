"""Tests for input normalisation.

The rule these tests defend: the original input is never destroyed, and
whitespace is never treated as a plaintext word boundary.
"""

from __future__ import annotations

import unittest

from cipher_tool.normalize import (
    Inventory,
    NormalizedText,
    chunks,
    clean_key,
    columns,
    from_numbers,
    group_text,
    inventory_of,
    letters_only,
    normalize,
    strip_bom,
    symbols_only,
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


class TestSymbolsOnly(unittest.TestCase):
    def test_keeps_letters_and_digits_in_order(self) -> None:
        self.assertEqual(symbols_only("A1B 2C, 3D"), "A1B2C3D")

    def test_uppercases_and_folds_accents(self) -> None:
        self.assertEqual(symbols_only("caf\u00e9 1907"), "CAFE1907")

    def test_empty(self) -> None:
        self.assertEqual(symbols_only(""), "")
        self.assertEqual(symbols_only("!!! ..."), "")


class TestInventoryCounts(unittest.TestCase):
    """What was in the paste, not what survived the filter.

    The bug this defends against: a message of 891 letters and 360 digits
    printed "Read 891 letters", which describes the leftovers of a filter as
    though it described the input.
    """

    def test_every_class_is_counted(self) -> None:
        found = inventory_of("7CX S3, H6\n")
        self.assertEqual(found.letters, 4)
        self.assertEqual(found.digits, 3)
        self.assertEqual(found.other, 1)      # the comma
        self.assertEqual(found.spaces, 3)     # two spaces and the newline
        self.assertEqual(found.symbols, 7)
        self.assertEqual(found.total, 11)

    def test_digit_fraction_is_over_the_symbol_stream(self) -> None:
        # Not over the whole input: spaces and punctuation are layout, and
        # dividing by them would make the same message look less numeric
        # simply for being printed in five-symbol groups.
        found = inventory_of("AAAAAAAA 11")
        self.assertAlmostEqual(found.digit_fraction, 0.2)

    def test_digit_fraction_of_nothing_is_zero_not_an_error(self) -> None:
        self.assertEqual(inventory_of("!!!").digit_fraction, 0.0)

    def test_describe_names_both_classes(self) -> None:
        described = inventory_of("7CX S3, H6").describe()
        self.assertIn("7 symbols", described)
        self.assertIn("4 letters", described)
        self.assertIn("3 digits", described)

    def test_describe_omits_a_class_that_is_empty(self) -> None:
        described = inventory_of("ATTACK AT DAWN").describe()
        self.assertIn("12 letters", described)
        self.assertNotIn("digit", described)

    def test_a_hand_built_inventory_is_all_zeroes(self) -> None:
        """The NOT MEASURED case, and it must read as zero, never as counts."""
        self.assertEqual(Inventory().symbols, 0)
        self.assertEqual(Inventory().total, 0)
        self.assertEqual(Inventory().digit_fraction, 0.0)


class TestNormalizedTextCarriesTheInventory(unittest.TestCase):
    def test_symbols_view_and_its_position_map(self) -> None:
        raw = "A1B 2C, 3D"
        result = normalize(raw)
        self.assertEqual(result.symbols, "A1B2C3D")
        self.assertEqual(result.symbol_positions, (0, 1, 2, 4, 5, 8, 9))
        for index, position in enumerate(result.symbol_positions):
            self.assertEqual(raw[position].upper(), result.symbols[index])

    def test_the_letters_of_the_symbol_stream_are_the_letters_view(self) -> None:
        result = normalize("Attack at dawn! 42 times.")
        self.assertEqual(
            "".join(c for c in result.symbols if not c.isdigit()),
            result.letters,
        )

    def test_lengths_agree_with_the_inventory(self) -> None:
        result = normalize("CAF\u00c9 na\u00efve, 1907")
        self.assertEqual(len(result.symbols), result.inventory.symbols)
        self.assertEqual(len(result.symbol_positions), len(result.symbols))
        self.assertEqual(
            result.inventory.letters + result.inventory.digits,
            len(result.symbols),
        )

    def test_derived_members(self) -> None:
        result = normalize("7CX S3, H6")
        self.assertTrue(result.has_symbols)
        self.assertAlmostEqual(result.digit_fraction, 3 / 7)
        self.assertIn("7 symbols", result.describe_input())
        self.assertFalse(normalize("!!!").has_symbols)

    def test_a_hand_built_normalized_text_still_works(self) -> None:
        """The legacy call site: positional construction, no inventory.

        Anything that builds a NormalizedText without going through
        normalize() gets a zeroed inventory, and every predicate over it must
        read that as NOT MEASURED and behave exactly as the tool does today.
        """
        legacy = NormalizedText("ABC", "ABC", (0, 1, 2), ("ABC",))
        self.assertEqual(legacy.symbols, "")
        self.assertEqual(legacy.symbol_positions, ())
        self.assertEqual(legacy.inventory, Inventory())
        self.assertFalse(legacy.has_symbols)
        self.assertEqual(legacy.digit_fraction, 0.0)


class TestTheExistingViewsAreByteIdentical(unittest.TestCase):
    """The additive-first rule, as a test.

    These values were recorded from the tree BEFORE the inventory existed.
    If any of them moves, the change was not additive and every solver in the
    toolkit is reading something different from what it read yesterday.
    """

    GOLDEN = (
        (
            "Attack at dawn! 42 times.",
            "ATTACKATDAWNTIMES",
            (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 19, 20, 21, 22, 23),
            ("ATTACK", "AT", "DAWN", "TIMES"),
        ),
        (
            "CAF\u00c9 na\u00efve, 1907",
            "CAFENAIVE",
            (0, 1, 2, 3, 5, 6, 7, 8, 9),
            ("CAFE", "NAIVE"),
        ),
        (
            "\ufeffABC 123 def",
            "ABCDEF",
            (0, 1, 2, 8, 9, 10),
            ("ABC", "DEF"),
        ),
        ("7CX S3, H6\n", "CXSH", (1, 2, 4, 8), ("CX", "S", "H")),
        ("", "", (), ()),
        ("   ", "", (), ()),
    )

    def test_letters_positions_and_groups_are_unchanged(self) -> None:
        for raw, letters, positions, groups in self.GOLDEN:
            with self.subTest(raw=raw):
                result = normalize(raw)
                self.assertEqual(result.original, strip_bom(raw))
                self.assertEqual(result.letters, letters)
                self.assertEqual(result.positions, positions)
                self.assertEqual(result.groups, groups)


if __name__ == "__main__":
    unittest.main()
