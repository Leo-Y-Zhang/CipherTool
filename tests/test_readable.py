"""Tests for turning a recovered plaintext into something readable.

The rule these tests defend: a reading may add spaces and change case, and it
may do NOTHING else. Every letter the decryption produced must survive, in the
same order, so that stripping the reading back down returns the plaintext
exactly. A reading that quietly drops or invents a letter would be a wrong
answer wearing the clothes of a right one.

The second rule: capitals are only ever placed where English guarantees one.
Two cleverer schemes were measured against held-out prose and both were far
worse than doing nothing -- the numbers are recorded in `readable.py`.
"""

from __future__ import annotations

import unittest

from cipher_tool.candidates import (
    Candidate,
    render_candidate,
    render_candidates,
)
from cipher_tool.normalize import group_text
from cipher_tool.readable import (
    Reading,
    read_plaintext,
    sentence_case,
)
from cipher_tool.scoring import default_scorer


class TestSentenceCase(unittest.TestCase):
    """Casing alone, with the word split already decided."""

    def test_lowercases_everything_but_the_opening_letter(self) -> None:
        self.assertEqual(
            sentence_case(["ATTACK", "AT", "DAWN"]),
            "Attack at dawn",
        )

    def test_the_pronoun_i_keeps_its_capital(self) -> None:
        self.assertEqual(
            sentence_case(["THE", "LETTER", "I", "SENT"]),
            "The letter I sent",
        )

    def test_i_inside_a_word_is_not_capitalised(self) -> None:
        """Only the standalone pronoun, never an I that happens to be there."""
        self.assertEqual(
            sentence_case(["IT", "IS", "MINE"]),
            "It is mine",
        )

    def test_a_leading_pronoun_is_capitalised_once_not_twice(self) -> None:
        self.assertEqual(sentence_case(["I", "AGREE"]), "I agree")

    def test_no_other_word_is_capitalised(self) -> None:
        """Names come out lowercase, deliberately.

        Capitalising a name we cannot identify means capitalising ordinary
        words too, and the measured cost of that is 1,197 wrong capitals for
        172 right ones. A lowercase name reads as the tool declining to
        guess; a capitalised MARE reads as a broken decryption.
        """
        self.assertEqual(
            sentence_case(["MEET", "MIRA", "SALT", "ON", "TUESDAY"]),
            "Meet mira salt on tuesday",
        )

    def test_empty_input(self) -> None:
        self.assertEqual(sentence_case([]), "")

    def test_single_word(self) -> None:
        self.assertEqual(sentence_case(["HELLO"]), "Hello")

    def test_an_unexplained_chunk_is_lowercased_like_anything_else(self) -> None:
        self.assertEqual(sentence_case(["THE", "XQZPT", "IS"]), "The xqzpt is")


class TestLettersSurvive(unittest.TestCase):
    """The invariant that makes a reading safe to show at all."""

    def _assert_letters_preserved(self, plaintext: str) -> Reading:
        reading = read_plaintext(plaintext)
        stripped = "".join(c for c in reading.text if c.isalpha()).upper()
        self.assertEqual(
            stripped,
            plaintext,
            "the reading changed the letters, not just the spacing and case",
        )
        return reading

    def test_ordinary_english(self) -> None:
        self._assert_letters_preserved(
            "TIMBERANDBRUSHWOODSOMEFOURHUNDREDYARDSDOWNSTREAM"
        )

    def test_text_the_lexicon_cannot_explain(self) -> None:
        self._assert_letters_preserved("XQZPTWVBKJXQZPTWVBKJXQZPTWVBKJ")

    def test_mixed_known_and_unknown(self) -> None:
        self._assert_letters_preserved("THEMESSAGEFROMXQZPTARRIVEDATDAWN")

    def test_single_letter(self) -> None:
        self._assert_letters_preserved("A")

    def test_empty_plaintext(self) -> None:
        reading = read_plaintext("")
        self.assertEqual(reading.text, "")
        self.assertFalse(reading.trustworthy)

    def test_input_that_is_already_spaced_and_cased(self) -> None:
        """Callers should be able to hand us anything and get letters back."""
        reading = read_plaintext("Attack at dawn!")
        stripped = "".join(c for c in reading.text if c.isalpha()).upper()
        self.assertEqual(stripped, "ATTACKATDAWN")


class TestReadPlaintext(unittest.TestCase):
    def test_real_english_becomes_readable(self) -> None:
        reading = read_plaintext("ATTACKATDAWN")
        self.assertEqual(reading.text, "Attack at dawn")
        self.assertTrue(reading.trustworthy)

    def test_a_longer_passage(self) -> None:
        reading = read_plaintext("THEMESSAGEWASSENTONTHEFIRSTOFTHEMONTH")
        self.assertEqual(
            reading.text, "The message was sent on the first of the month"
        )
        self.assertTrue(reading.trustworthy)

    def test_gibberish_is_not_trustworthy(self) -> None:
        """The gate exists so a bad split is never shown as a reading."""
        reading = read_plaintext("XQZPTWVBKJXQZPTWVBKJXQZPTWVBKJXQZPTWVBKJ")
        self.assertFalse(reading.trustworthy)

    def test_coverage_is_reported(self) -> None:
        good = read_plaintext("ATTACKATDAWN")
        bad = read_plaintext("XQZPTWVBKJXQZPTWVBKJ")
        self.assertGreater(good.coverage, bad.coverage)
        self.assertGreaterEqual(good.coverage, 0.0)
        self.assertLessEqual(good.coverage, 1.0)

    def test_submission_safe_is_stricter_than_trustworthy(self) -> None:
        """A name makes a reading unsafe to paste, but still worth showing.

        Measured on real competition text: a split can be 0.889 trustworthy
        by letters while 43 of its 670 tokens are not words at all. Coverage
        cannot see a word cut in half; counting whole tokens can.
        """
        reading = read_plaintext("THEMESSAGEFROMXQZPTARRIVEDATDAWN")
        self.assertFalse(reading.submission_safe)

    def test_clean_text_is_submission_safe(self) -> None:
        reading = read_plaintext("ATTACKATDAWN")
        self.assertTrue(reading.submission_safe)

    def test_an_explicit_scorer_is_honoured(self) -> None:
        """Callers that already built a scorer must not pay for another."""
        scorer = default_scorer()
        reading = read_plaintext("ATTACKATDAWN", scorer=scorer)
        self.assertEqual(reading.text, "Attack at dawn")

    def test_pronoun_survives_the_whole_pipeline(self) -> None:
        reading = read_plaintext("IAMWRITINGTOYOUABOUTTHEMATTER")
        self.assertTrue(reading.text.startswith("I am writing"))


class TestWrapping(unittest.TestCase):
    def test_wrapped_lines_respect_the_width(self) -> None:
        reading = read_plaintext(
            "THEMESSAGEWASSENTONTHEFIRSTOFTHEMONTHANDITARRIVEDBEFORENIGHTFALL"
        )
        lines = reading.wrapped(28)
        self.assertTrue(lines)
        for line in lines:
            self.assertLessEqual(len(line), 28)

    def test_wrapping_never_breaks_a_word(self) -> None:
        reading = read_plaintext(
            "THEMESSAGEWASSENTONTHEFIRSTOFTHEMONTHANDITARRIVEDBEFORENIGHTFALL"
        )
        rejoined = " ".join(reading.wrapped(24))
        self.assertEqual(rejoined.split(), reading.text.split())

    def test_a_word_longer_than_the_width_is_not_lost(self) -> None:
        """An unexplained run can be longer than the column. Keep it whole."""
        reading = read_plaintext("XQZPTWVBKJXQZPTWVBKJXQZPTWVBKJ")
        rejoined = " ".join(reading.wrapped(8))
        self.assertEqual(rejoined.split(), reading.text.split())

    def test_empty_reading_wraps_to_nothing(self) -> None:
        self.assertEqual(read_plaintext("").wrapped(40), [])


class TestItIsActuallyWiredIn(unittest.TestCase):
    """The feature is worthless if the render path does not call it.

    Every test above could pass while the toolkit still printed nothing but
    five-letter groups, so these drive the real rendering functions and look
    at what a person would actually see.
    """

    def _candidate(self, plaintext: str) -> Candidate:
        return Candidate(
            method="Caesar",
            key="shift=19",
            score=-1.0 * len(plaintext),
            plaintext=plaintext,
            display=group_text(plaintext),
        )

    def test_the_reading_appears_beneath_the_plaintext(self) -> None:
        block = render_candidate(self._candidate("ATTACKATDAWN"))
        self.assertIn("Plaintext:", block)
        self.assertIn("Reading:", block)
        self.assertIn("Attack at dawn", block)
        self.assertLess(
            block.index("Plaintext:"),
            block.index("Reading:"),
            "the decryption must be printed before the toolkit's opinion of it",
        )

    def test_the_five_letter_groups_are_still_there(self) -> None:
        """The reading is an addition. It must never replace the answer."""
        block = render_candidate(self._candidate("ATTACKATDAWN"))
        self.assertIn("ATTAC KATDA WN", block)

    def test_gibberish_gets_no_reading(self) -> None:
        block = render_candidate(self._candidate("XQZPTWVBKJXQZPTWVBKJXQZPTWVBKJ"))
        self.assertIn("Plaintext:", block)
        self.assertNotIn("Reading:", block)

    def test_full_text_mode_renders_the_reading_too(self) -> None:
        block = render_candidate(
            self._candidate("THEMESSAGEWASSENTONTHEFIRSTOFTHEMONTH"),
            full_text=True,
        )
        self.assertIn("Reading:", block)
        self.assertIn("The message was sent", block)

    def test_the_caveat_is_printed_when_a_reading_is(self) -> None:
        rendered = render_candidates([self._candidate("ATTACKATDAWN")])
        self.assertIn("Reading:", rendered)
        self.assertIn("Submit the Plaintext, never the Reading", rendered)

    def test_no_caveat_when_no_reading_was_shown(self) -> None:
        """A caveat about a line that is not there teaches people to skim."""
        rendered = render_candidates(
            [self._candidate("XQZPTWVBKJXQZPTWVBKJXQZPTWVBKJ")]
        )
        self.assertNotIn("Reading:", rendered)
        self.assertNotIn("Submit the Plaintext, never the Reading", rendered)

    def test_a_plaintext_with_no_letters_does_not_crash_the_renderer(self) -> None:
        """Some solvers produce digit streams. There is nothing to read there."""
        block = render_candidate(self._candidate("12345678901234567890"))
        self.assertNotIn("Reading:", block)

    def test_an_empty_plaintext_does_not_crash_the_renderer(self) -> None:
        block = render_candidate(self._candidate(""))
        self.assertNotIn("Reading:", block)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
