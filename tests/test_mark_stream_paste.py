"""A paste made of marks is not an empty paste.

Fifth variant of one bug. The first four were: a numeric ciphertext read as an
empty paste; "no letters" and "nothing pasted" printed identically; a message
of letters AND digits described by the count of what survived the letters-only
filter; and a symbol stream refused because the guard keyed on digits rather
than on the pairing.

This one is the same mistake made about punctuation. ``normalize`` keeps A-Z
and 0-9 and counts everything else as ``other``, and ``Inventory.describe``
never mentions ``other`` -- correctly, because in an ordinary paste those
characters are transcription layout. But 2024 challenge 9B is written in the
three marks ``|`` ``/`` and ``\\``, so for that message the marks ARE the
message: 12,935 of them, described by the paste screen as "Read 0 symbols".

The rule this pins is narrow on purpose: **marks are layout while a symbol
stream exists, and are the message when none does.** An inventory that has
never been measured is all zeroes, so it takes the old path and reads "0
symbols" exactly as before.

Two of these classes are guards rather than regressions -- they pinned
behaviour that was already right before the fix, so that widening the
description could not quietly widen it onto ordinary text.
"""

from __future__ import annotations

import unittest

from cipher_tool import cli, encodings
from cipher_tool.normalize import Inventory, normalize

#: The shape of 2024 challenge 9B: a ternary alphabet, no letters, no digits.
TERNARY = "|/\\ /||\\ \\\\/ |//\n\\|/ //| \\/| |\\\\" * 4

#: Twenty-six distinct marks: a one-to-one substitution written in symbols,
#: which is the opposite shape to a fractionation and must not be called one.
WIDE_ALPHABET = "!\"#$%&'()*+,-.:;<=>?@[]^_`{}~" * 3

#: HELLO WORLD. Morse is dots and dashes, so it too is a pure mark stream --
#: and ``encodings`` reads it from the original text, which means the toolkit
#: SOLVES this paste while describing it as nothing.
MORSE = ".... . .-.. .-.. --- / .-- --- .-. .-.. -.."


def marks_in(text: str) -> str:
    """The non-whitespace characters of *text*, which here are all marks."""
    return "".join(ch for ch in text if not ch.isspace())


class AMarkOnlyPasteIsCounted(unittest.TestCase):
    """What was pasted must be reported, whatever notation it is written in."""

    def test_the_marks_are_counted_rather_than_reported_as_zero_symbols(
        self,
    ) -> None:
        """The recorded defect: 12,935 marks announced as "Read 0 symbols"."""
        described = normalize(TERNARY).describe_input()
        self.assertIn(str(len(marks_in(TERNARY))), described)
        self.assertNotIn("0 symbols", described)

    def test_the_description_does_not_call_marks_letters_or_digits(self) -> None:
        """Counting them is only half of it; naming them wrongly is the rest."""
        described = normalize(TERNARY).describe_input()
        self.assertNotIn("letters", described.replace("letters or digits", ""))
        self.assertNotIn("digits", described.replace("letters or digits", ""))

    def test_the_mark_stream_is_kept_as_a_view(self) -> None:
        """A count cannot name an alphabet; a screen that refuses needs to."""
        self.assertEqual(normalize(TERNARY).marks, marks_in(TERNARY))

    def test_the_view_holds_the_alphabet_the_message_is_written_in(self) -> None:
        """2024 9B is ternary, and that is the fact a reader needs first."""
        self.assertEqual(set(normalize(TERNARY).marks), {"|", "/", "\\"})

    def test_has_marks_is_false_when_nothing_was_measured(self) -> None:
        """A hand-built instance must read as NOT MEASURED, never as empty."""
        self.assertFalse(normalize("").has_marks)


class OrdinaryTextIsUnaffected(unittest.TestCase):
    """Guards. Marks stay layout wherever a symbol stream exists."""

    def test_punctuation_in_a_letter_paste_is_still_layout(self) -> None:
        """A comma is not evidence, and must not be reported as though it is."""
        described = normalize("ATTACK AT DAWN, at once.").describe_input()
        self.assertEqual(described, "18 symbols: 18 letters")

    def test_a_digit_paste_is_described_exactly_as_before(self) -> None:
        """The Polybius family, fixed earlier, must not shift underneath."""
        described = normalize("12345 54321 11223").describe_input()
        self.assertEqual(described, "15 symbols: 15 digits")

    def test_a_mixed_paste_is_described_exactly_as_before(self) -> None:
        """The 891-letters-and-360-digits shape, which started all of this."""
        described = normalize("7CX S3, H6").describe_input()
        self.assertEqual(described, "7 symbols: 4 letters and 3 digits")

    def test_an_unmeasured_inventory_still_reads_as_zero_symbols(self) -> None:
        """All zeroes means NOT MEASURED, so it must take the old path."""
        self.assertEqual(Inventory().describe(), "0 symbols")


class AMorsePasteIsSolvedAndSaysSo(unittest.TestCase):
    """The worst version of the defect: the screen lied about a paste it got
    RIGHT. ``encodings.solve`` reads the original text, so a Morse paste has
    always decoded -- underneath a banner reading "Read 0 symbols"."""

    def test_a_morse_paste_reports_what_was_read(self) -> None:
        """The description must survive contact with a solvable mark stream."""
        described = normalize(MORSE).describe_input()
        self.assertIn(str(len(marks_in(MORSE))), described)
        self.assertNotIn("0 symbols", described)

    def test_morse_still_decodes(self) -> None:
        """Guard: the description changed, the decoding must not have."""
        best = encodings.solve(MORSE, top=3).best()
        self.assertIsNotNone(best, "Morse stopped decoding")
        assert best is not None
        self.assertIn("HELLOWORLD", best.plaintext.upper().replace(" ", ""))


class TheRefusalNamesTheShape(unittest.TestCase):
    """A refusal that cannot say what it saw sends the reader back to paste
    the same thing again."""

    def test_the_refusal_counts_the_distinct_marks(self) -> None:
        """Three distinct marks is the single most useful fact about 9B."""
        rendered = cli._render_letterless(TERNARY)
        self.assertIn("3 distinct", rendered)

    def test_the_refusal_does_not_describe_marks_as_digits(self) -> None:
        """It used to end on "(0 are digits)", which explains nothing."""
        rendered = cli._render_letterless(TERNARY)
        self.assertNotIn("0 are digits", rendered)

    def test_a_digit_refusal_is_unchanged(self) -> None:
        """Guard: the numeric wording earned its place and keeps it."""
        rendered = cli._render_letterless("12345 54321")
        self.assertIn("are digits", rendered)

    def test_a_five_mark_alphabet_is_told_that_transcribing_reaches_a_solver(
        self,
    ) -> None:
        """Five symbols is a Polybius square, and the toolkit solves those.

        Written in digits it is 2023 challenge 8B, solved in seven seconds.
        Written in marks it is the same cipher, and the only thing between
        the reader and the answer is a find-and-replace.
        """
        rendered = cli._render_letterless("!@#$% %$#@! @#$%! #$%!@ $%!@#")
        self.assertIn("Polybius square", rendered)

    def test_a_three_mark_alphabet_is_not_promised_a_solver(self) -> None:
        """2024 9B is ternary and nothing here reads it. Say that.

        The refusal is reached only after every symbol solver has already
        declined, so sending the reader to run those same solvers by hand is
        homework with a known ending.
        """
        rendered = cli._render_letterless(TERNARY)
        self.assertIn("no solver in this toolkit reads", rendered)
        self.assertNotIn("Polybius square", rendered)

    def test_a_three_mark_alphabet_is_not_sent_to_a_command_that_will_refuse(
        self,
    ) -> None:
        """Naming the gap and then listing the solver is the same homework.

        The numeric-square command cannot read three marks any more than the
        paste screen could, and printing it under a paragraph that has just
        said so invites exactly one thing: running it, and being refused
        again.
        """
        rendered = cli._render_letterless(TERNARY)
        self.assertNotIn("polybius", rendered)
        self.assertIn("encodings", rendered)

    def test_a_five_mark_alphabet_still_gets_the_numeric_square_command(
        self,
    ) -> None:
        """Guard: the command belongs wherever transcription can reach it."""
        rendered = cli._render_letterless("!@#$% %$#@! @#$%! #$%!@ $%!@#")
        self.assertIn("polybius", rendered)

    def test_a_large_mark_alphabet_is_not_called_a_fractionation(self) -> None:
        """A fractionation writes each letter as a GROUP of marks, so its
        alphabet is small -- two for binary, three for 9B, five for a Polybius
        square, six for ADFGVX. Twenty-six distinct symbols is the opposite
        shape: one mark per letter. Calling it a fractionation would send the
        reader looking for groups that are not there.
        """
        rendered = cli._render_letterless(WIDE_ALPHABET)
        self.assertNotIn("fractionation", rendered)

    def test_a_large_mark_alphabet_is_told_what_shape_it_is(self) -> None:
        """Naming what it is NOT leaves the reader exactly where they were."""
        rendered = cli._render_letterless(WIDE_ALPHABET)
        self.assertIn("stands for one letter", rendered)


if __name__ == "__main__":
    unittest.main()
