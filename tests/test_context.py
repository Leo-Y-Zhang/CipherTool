"""Tests for team-supplied story context.

The behaviour that matters: context supplies candidates to test, and never
constrains or asserts anything by itself.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cipher_tool.context import (
    FIELDS,
    ContextNotes,
    merge_cribs,
    parse_assignment,
)


class TestEditing(unittest.TestCase):
    def test_add_and_read_back(self) -> None:
        notes = ContextNotes()
        notes.add("people", "Admiral Harrow")
        self.assertEqual(notes.people, ["Admiral Harrow"])

    def test_duplicates_are_ignored(self) -> None:
        notes = ContextNotes()
        notes.add("places", "Portsmouth")
        notes.add("places", "Portsmouth")
        self.assertEqual(notes.places, ["Portsmouth"])

    def test_unknown_field_is_rejected_with_a_helpful_message(self) -> None:
        notes = ContextNotes()
        with self.assertRaises(ValueError) as context:
            notes.add("suspects", "someone")
        message = str(context.exception)
        self.assertIn("suspects", message)
        self.assertIn("people", message)

    def test_empty_entry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ContextNotes().add("people", "   ")

    def test_is_empty(self) -> None:
        notes = ContextNotes()
        self.assertTrue(notes.is_empty())
        notes.add("notes", "something")
        self.assertFalse(notes.is_empty())


class TestPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.cipher_path = Path(self.directory.name) / "message.txt"
        self.cipher_path.write_text("HEALI OPASD", encoding="utf-8")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_path_is_beside_the_ciphertext(self) -> None:
        self.assertEqual(
            ContextNotes.path_for(self.cipher_path).name,
            "message.txt.context.json",
        )

    def test_path_for_a_file_without_a_suffix(self) -> None:
        self.assertEqual(
            ContextNotes.path_for(Path("/tmp/message")).name,
            "message.context.json",
        )

    def test_round_trip(self) -> None:
        notes = ContextNotes()
        notes.add("people", "Admiral Harrow")
        notes.add("fragments", "MEET ME AT THE HARBOUR")
        notes.save(self.cipher_path)

        reloaded = ContextNotes.load(self.cipher_path)
        self.assertEqual(reloaded.people, ["Admiral Harrow"])
        self.assertEqual(reloaded.fragments, ["MEET ME AT THE HARBOUR"])

    def test_missing_file_is_not_an_error(self) -> None:
        self.assertTrue(ContextNotes.load(self.cipher_path).is_empty())

    def test_broken_json_is_reported_clearly(self) -> None:
        ContextNotes.path_for(self.cipher_path).write_text("{oh dear",
                                                           encoding="utf-8")
        with self.assertRaises(ValueError) as context:
            ContextNotes.load(self.cipher_path)
        self.assertIn("not valid JSON", str(context.exception))

    def test_unknown_fields_are_reported_rather_than_ignored(self) -> None:
        # Silently dropping a field the team spent time typing would be worse
        # than refusing to load.
        ContextNotes.path_for(self.cipher_path).write_text(
            json.dumps({"peple": ["typo"]}), encoding="utf-8"
        )
        with self.assertRaises(ValueError) as context:
            ContextNotes.load(self.cipher_path)
        self.assertIn("peple", str(context.exception))

    def test_saved_file_is_ascii(self) -> None:
        notes = ContextNotes()
        notes.add("people", "Se\u00f1or Vega")
        path = notes.save(self.cipher_path)
        raw = path.read_bytes()
        self.assertTrue(all(byte < 128 for byte in raw))


class TestDerivedMaterial(unittest.TestCase):
    def setUp(self) -> None:
        self.notes = ContextNotes()
        self.notes.add("people", "Admiral Harrow")
        self.notes.add("places", "Portsmouth")
        self.notes.add("fragments", "MEET ME AT THE HARBOUR")
        self.notes.add("keywords", "tempest")

    def test_cribs_include_whole_entries_and_their_words(self) -> None:
        cribs = self.notes.crib_candidates()
        self.assertIn("ADMIRALHARROW", cribs)
        self.assertIn("ADMIRAL", cribs)
        self.assertIn("HARROW", cribs)
        self.assertIn("MEETMEATTHEHARBOUR", cribs)

    def test_cribs_are_longest_first(self) -> None:
        cribs = self.notes.crib_candidates()
        lengths = [len(crib) for crib in cribs]
        self.assertEqual(lengths, sorted(lengths, reverse=True))

    def test_very_short_words_are_dropped(self) -> None:
        # "ME" and "AT" fit almost anywhere and rule out almost nothing.
        cribs = self.notes.crib_candidates()
        self.assertNotIn("ME", cribs)
        self.assertNotIn("AT", cribs)

    def test_keywords_include_everything_written_down(self) -> None:
        keys = self.notes.keyword_candidates()
        self.assertIn("TEMPEST", keys)
        self.assertIn("PORTSMOUTH", keys)
        self.assertIn("HARROW", keys)

    def test_empty_notes_produce_nothing(self) -> None:
        empty = ContextNotes()
        self.assertEqual(empty.crib_candidates(), [])
        self.assertEqual(empty.keyword_candidates(), [])

    def test_merge_cribs_puts_explicit_ones_first(self) -> None:
        merged = merge_cribs(self.notes, extra=["THE"])
        self.assertEqual(merged[0], "THE")

    def test_merge_cribs_deduplicates(self) -> None:
        merged = merge_cribs(self.notes, extra=["harrow"])
        self.assertEqual(merged.count("HARROW"), 1)


class TestRendering(unittest.TestCase):
    def test_empty_notes_explain_how_to_add_some(self) -> None:
        rendered = ContextNotes().render("message.txt")
        self.assertIn("No context has been recorded", rendered)
        self.assertIn("--add", rendered)
        for name, _ in FIELDS:
            self.assertIn(name, rendered)

    def test_populated_notes_never_claim_correctness(self) -> None:
        notes = ContextNotes()
        notes.add("people", "Admiral Harrow")
        rendered = notes.render("message.txt")
        self.assertIn("candidates", rendered.lower())
        self.assertIn("none of this is assumed to be correct",
                      rendered.lower())


class TestParseAssignment(unittest.TestCase):
    def test_splits_on_the_first_equals(self) -> None:
        self.assertEqual(
            parse_assignment("fragments=A=B is the clue"),
            ("fragments", "A=B is the clue"),
        )

    def test_strips_whitespace(self) -> None:
        self.assertEqual(parse_assignment(" people = Harrow "),
                         ("people", "Harrow"))

    def test_missing_equals_is_reported_with_the_field_list(self) -> None:
        with self.assertRaises(ValueError) as context:
            parse_assignment("people Harrow")
        self.assertIn("field=value", str(context.exception))
        self.assertIn("people", str(context.exception))


if __name__ == "__main__":
    unittest.main()
