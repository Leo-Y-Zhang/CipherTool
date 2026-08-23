"""A letterless paste must not be treated as an empty one.

``auto_solve`` used to stop at ``normalized.is_empty`` with "the input contains
no letters, so nothing was attempted", and return before a single stage ran.
Polybius, Nihilist and straddling-checkerboard ciphertexts are ALL DIGITS, so
that early exit hid a whole family: 2023 challenge 8B, 2,714 digits over five
symbols, was refused in 0.0 seconds by a toolkit that already contained the
solver that cracks it in seven.

These tests pin both halves of the fix, and the second half is the one that was
wrong on the first attempt.
"""

from __future__ import annotations

import unittest

from cipher_tool import auto, polybius

PLAIN = (
    "MYINVESTIGATIONSOFTHEINCIDENTHAVEBEENMOREFRUITFULTHANICOULDHAVEHOPED"
    "PINKERTONSHAVEPROVIDEDCLEAREVIDENCETHATTHEREWASACONSPIRACYTOSTRIKE"
    "ATTHEHEARTOFGOVERNMENTANDTHATTHEPLOTREACHEDFURTHERTHANWEFEARED"
    "IHAVESETOUTBELOWWHATWEKNOWANDWHATWEONLYSUSPECTSOTHATYOUMAYJUDGE"
    "FORYOURSELFWHETHERTOBRINGTHISBEFORETHECABINET"
) * 3


def as_digits(text: str) -> str:
    """Encrypt through the classical numeric 5x5 square."""
    return polybius.encrypt(text)


class LetterlessPaste(unittest.TestCase):
    def test_a_pure_digit_paste_is_solved_not_refused(self) -> None:
        """The 2023 8B shape: no letters at all, a real message underneath."""
        result = auto.auto_solve(as_digits(PLAIN), effort="fast", top=3,
                                 seed=1)
        best = result.candidates.best()
        self.assertIsNotNone(best, "a letterless paste was refused outright")
        assert best is not None
        self.assertIn("MYINVESTIGATIONS", best.plaintext)
        self.assertTrue(
            any(stage.ran for stage in result.stages),
            "no stage ran at all -- the early exit is back",
        )

    def test_an_odd_digit_count_still_solves(self) -> None:
        """One stray digit must not cost the whole message.

        The first version of the fix required an EVEN count and threw away two
        more real challenges -- 2024 7B (2,315 digits) and 8B (3,025) --
        because hand-copied competition material routinely gains or loses a
        symbol. The final unpaired digit is dropped; a leading one would
        misalign every pair after it.
        """
        result = auto.auto_solve(as_digits(PLAIN) + "3", effort="fast",
                                 top=3, seed=1)
        best = result.candidates.best()
        self.assertIsNotNone(best,
                             "an odd digit count refused the whole message")
        assert best is not None
        self.assertIn("MYINVESTIGATIONS", best.plaintext)

    def test_the_oddity_is_reported_not_hidden(self) -> None:
        result = auto.auto_solve(as_digits(PLAIN) + "3", effort="fast",
                                 top=1, seed=1)
        ran = [stage for stage in result.stages if stage.ran]
        self.assertTrue(
            any("odd count" in stage.note for stage in ran),
            "a digit was dropped and the report did not say so",
        )


class ThePasteScreenPathToo(unittest.TestCase):
    """The fix above was in ``auto_solve``. The paste screen does not use it.

    DEFECT, found 2026-08-23. ``handle_paste`` routes a letterless paste to
    ``cli._solve_symbol_stream``, which called the Polybius square search with
    the raw text -- and that search returns NOTHING on an odd symbol count.
    So the odd-count fix above was real, tested, and unreachable from the only
    screen the operator uses. 2024 challenge 8B, 3,025 digits, was refused
    with "this is not a letter cipher" and a suggestion to run
    ``cipher_tool polybius``, which would have refused it identically.

    The test drives the CLI helper, not ``auto_solve``, because that is the
    distinction the defect turned on: the library call passed while the user
    journey failed.
    """

    def _args(self):
        import argparse
        return argparse.Namespace(top=3, seed=1, max_time=None)

    def test_the_paste_path_solves_an_odd_digit_count(self) -> None:
        from cipher_tool import cli

        found = cli._solve_symbol_stream(as_digits(PLAIN) + "3", self._args())
        self.assertIsNotNone(
            found, "the paste screen refused an odd digit count outright")
        best = found.candidates.best()
        self.assertIsNotNone(best)
        self.assertIn("MYINVESTIGATIONS", best.plaintext)

    def test_the_paste_path_still_solves_an_even_digit_count(self) -> None:
        from cipher_tool import cli

        found = cli._solve_symbol_stream(as_digits(PLAIN), self._args())
        self.assertIsNotNone(found)
        self.assertIn("MYINVESTIGATIONS", found.candidates.best().plaintext)

    def test_only_the_last_symbol_is_ever_dropped(self) -> None:
        """A leading drop misaligns every pair after it.

        Pinned as its own test because the two mistakes look equally harmless
        and only one of them destroys the message.
        """
        from cipher_tool import cli

        found = cli._solve_symbol_stream("3" + as_digits(PLAIN), self._args())
        if found is not None:
            best = found.candidates.best()
            self.assertNotIn("MYINVESTIGATIONS", best.plaintext)


class StillRefused(unittest.TestCase):
    """The early exit must survive for the cases it was written for."""

    def test_a_genuinely_empty_paste_is_still_refused(self) -> None:
        result = auto.auto_solve("....  ----  ....", effort="fast", top=1,
                                 seed=1)
        self.assertIsNone(result.candidates.best())
        self.assertFalse(any(stage.ran for stage in result.stages))

    def test_a_short_digit_run_is_not_treated_as_a_cipher(self) -> None:
        """A date or a reference number is not a Polybius ciphertext."""
        result = auto.auto_solve("1234 5678 2345", effort="fast", top=1,
                                 seed=1)
        self.assertIsNone(result.candidates.best())


if __name__ == "__main__":
    unittest.main()
