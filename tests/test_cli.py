"""Tests for the command line interface.

Covers the plumbing (input handling, argument parsing, error reporting) and
runs each command end to end on a small ciphertext to prove it does not
crash and prints what it promises.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

from cipher_tool import caesar, cli
from cipher_tool.normalize import letters_only
from cipher_tool.scoring import corpus_files


def sample_plaintext(length: int = 400) -> str:
    """Real English from our own corpus."""
    return letters_only(corpus_files()[1].read_text(encoding="utf-8"))[:length]


def run(*argv: str) -> tuple[int, str]:
    """Run the CLI and capture everything it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        code = cli.main(list(argv))
    return code, buffer.getvalue()


def run_rejecting(*argv: str) -> tuple[int, str]:
    """Run the CLI expecting argparse to reject the line.

    argparse reports a bad value by raising ``SystemExit(2)`` from inside
    ``parse_args``, so it never reaches ``main``'s return statement.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            code = cli.main(list(argv))
        except SystemExit as stop:
            code = stop.code
    return code, buffer.getvalue()


def run_shell_session(*lines: str) -> str:
    """Type *lines* at the interactive shell and return everything printed.

    Nothing here catches exceptions: a line that kills the shell must fail
    the test that types it, which is the whole point of most of these.
    """
    script = io.StringIO("\n".join(lines) + "\n")
    buffer = io.StringIO()
    original_stdin = sys.stdin
    sys.stdin = script
    try:
        with contextlib.redirect_stdout(buffer), \
                contextlib.redirect_stderr(buffer):
            cli.main(["shell"])
    finally:
        sys.stdin = original_stdin
    return buffer.getvalue()


class TestReadSource(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "message.txt"
        self.path.write_text("HEALI OPASD", encoding="utf-8")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_reads_a_file(self) -> None:
        code, output = run("show", str(self.path))
        self.assertEqual(code, 0)
        self.assertIn("HEALIOPASD", output)

    def test_reads_text_directly(self) -> None:
        code, output = run("show", "--text", "heali opasd")
        self.assertEqual(code, 0)
        self.assertIn("HEALIOPASD", output)

    def test_missing_file_is_an_error_not_a_guess(self) -> None:
        # Treating a bad path as literal ciphertext would turn a typo into a
        # confusing solve of the filename.
        code, output = run("show", "nosuchfile.txt")
        self.assertEqual(code, 2)
        self.assertIn("No such file", output)
        self.assertIn("--text", output)

    def test_directory_is_reported(self) -> None:
        code, output = run("show", self.directory.name)
        self.assertEqual(code, 2)
        self.assertIn("directory", output)

    def test_no_input_at_all_is_reported(self) -> None:
        code, output = run("show")
        self.assertEqual(code, 2)
        self.assertIn("No ciphertext given", output)


class TestShow(unittest.TestCase):
    def test_prints_original_and_normalised(self) -> None:
        _, output = run("show", "--text", "Attack, at dawn!")
        self.assertIn("ORIGINAL INPUT", output)
        self.assertIn("Attack, at dawn!", output)
        self.assertIn("NORMALISED", output)
        self.assertIn("ATTACKATDAWN", output)

    def test_flags_uniform_grouping_as_formatting(self) -> None:
        _, output = run("show", "--text", "HEALI OPASD EHANS")
        self.assertIn("NOT", output)
        self.assertIn("word", output.lower())

    def test_single_group_is_not_flagged(self) -> None:
        # One group is not evidence of anything.
        _, output = run("show", "--text", "HEALI")
        self.assertNotIn("transcription formatting", output)

    def test_empty_input_does_not_crash(self) -> None:
        code, output = run("show", "--text", "1234 !!!")
        self.assertEqual(code, 0)
        self.assertIn("(no letters)", output)


class TestSuppliedKeys(unittest.TestCase):
    """Supplying a key means 'do this'; omitting it means 'search'."""

    def test_caesar_with_a_shift_decrypts(self) -> None:
        ciphertext = caesar.encrypt(sample_plaintext(120), 5)
        _, output = run("caesar", "--text", ciphertext, "--shift", "5",
                        "--full")
        self.assertIn(sample_plaintext(60), output.replace("\n", ""))

    def test_caesar_without_a_shift_searches(self) -> None:
        ciphertext = caesar.encrypt(sample_plaintext(200), 5)
        _, output = run("caesar", "--text", ciphertext, "--top", "3")
        self.assertIn("shift=5", output)

    def test_caesar_all_shows_every_shift(self) -> None:
        ciphertext = caesar.encrypt(sample_plaintext(200), 5)
        _, output = run("caesar", "--text", ciphertext, "--all", "--top", "26")
        self.assertIn("Candidate 26", output)

    def test_encrypt_runs_the_cipher_forwards(self) -> None:
        _, output = run("caesar", "--text", "ATTACKATDAWN", "--shift", "3",
                        "--encrypt")
        self.assertIn("DWWDFNDWGDZQ", output.replace("\n", "").replace(" ", ""))

    def test_a_wrong_supplied_key_is_reported_honestly(self) -> None:
        # The tool must not present a user's wrong key as an answer.
        ciphertext = caesar.encrypt(sample_plaintext(200), 5)
        _, output = run("caesar", "--text", ciphertext, "--shift", "9")
        self.assertRegex(output, r"Confidence:\s+(weak|unlikely)")

    def test_affine_needs_both_parameters(self) -> None:
        code, output = run("affine", "--text", "ABCDEF", "-a", "5")
        self.assertEqual(code, 2)
        self.assertIn("both", output)

    def test_affine_rejects_an_invalid_multiplier(self) -> None:
        code, output = run("affine", "--text", "ABCDEF", "-a", "2", "-b", "1")
        self.assertEqual(code, 2)
        self.assertTrue(output.strip())


class TestParsingHelpers(unittest.TestCase):
    def test_parse_matrix_from_commas(self) -> None:
        self.assertEqual(cli._parse_matrix("3,3,2,5"), [[3, 3], [2, 5]])

    def test_parse_matrix_from_semicolons(self) -> None:
        self.assertEqual(cli._parse_matrix("3 3; 2 5"), [[3, 3], [2, 5]])

    def test_parse_matrix_rejects_non_square(self) -> None:
        with self.assertRaises(cli.InputError) as context:
            cli._parse_matrix("1,2,3")
        self.assertIn("square", str(context.exception))

    def test_parse_matrix_rejects_words(self) -> None:
        with self.assertRaises(cli.InputError) as context:
            cli._parse_matrix("3,3,two,5")
        self.assertIn("whole number", str(context.exception))

    def test_parse_mapping_accepts_repeated_flags(self) -> None:
        key = cli._parse_mapping(["Q=E", "X=T"])
        self.assertEqual(key.apply("QX"), "ET")

    def test_parse_mapping_accepts_a_comma_list(self) -> None:
        key = cli._parse_mapping(["Q=E,X=T"])
        self.assertEqual(key.apply("QX"), "ET")

    def test_parse_mapping_rejects_an_inconsistent_pair(self) -> None:
        # Two cipher letters cannot both mean E.
        with self.assertRaises(cli.InputError):
            cli._parse_mapping(["Q=E", "X=E"])

    def test_split_arguments_honours_quotes(self) -> None:
        self.assertEqual(
            cli._split_arguments('--key "two words" --top 5'),
            ["--key", "two words", "--top", "5"],
        )

    def test_split_arguments_on_empty(self) -> None:
        self.assertEqual(cli._split_arguments(""), [])


class TestOutputContract(unittest.TestCase):
    def test_every_solve_ends_with_the_disclaimer(self) -> None:
        _, output = run("caesar", "--text", sample_plaintext(200))
        self.assertIn("Verify the current rules", output)

    def test_quiet_suppresses_the_disclaimer(self) -> None:
        _, output = run("caesar", "--text", sample_plaintext(200), "--quiet")
        self.assertNotIn("Verify the current rules", output)

    def test_output_file_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.txt"
            _, output = run("analyse", "--text", sample_plaintext(200),
                            "--output", str(target))
            self.assertIn(f"Written to {target}", output)
            self.assertIn("Index of Coincidence", target.read_text("utf-8"))

    def test_confidence_is_always_labelled_a_heuristic(self) -> None:
        _, output = run("caesar", "--text", sample_plaintext(200))
        self.assertIn("[heuristic]", output)

    def test_candidate_lists_carry_the_uncertainty_note(self) -> None:
        _, output = run("caesar", "--text", sample_plaintext(200))
        self.assertIn("ranked guesses", output)


class TestEveryCommandRuns(unittest.TestCase):
    """Smoke test: every command produces output and exits cleanly."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = caesar.encrypt(sample_plaintext(240), 7)

    def test_commands(self) -> None:
        commands = [
            ["analyse"], ["analyze"], ["show"],
            ["caesar"], ["atbash"], ["affine"],
            ["substitution", "--restarts", "2", "--seed", "1"],
            ["keyword", "--words", "SECRET,TEMPEST"],
            ["vigenere", "--max-key-length", "6"],
            ["vigenere", "--evidence"],
            ["beaufort", "--max-key-length", "4"],
            ["autokey", "--max-primer", "2"],
            ["railfence"],
            ["columnar", "--max-key-length", "4"],
            # Pinned lengths and a short budget: this is a smoke test that
            # the command runs and prints, not that the search succeeds.
            ["columnar", "--double", "--first-length", "3",
             "--second-length", "2", "--max-time", "2", "--seed", "1"],
            ["transposition", "--max-key-length", "4"],
            ["transposition", "--routes"],
            ["polybius"], ["bifid", "--max-period", "4"],
            # A short budget: this proves the flag is wired to the search,
            # not that the search succeeds.
            ["bifid", "--search-square", "--period", "5",
             "--max-time", "2", "--seed", "1"],
            # Ordinary English is not ADFGVX, so this exercises the refusal
            # path: it must say so rather than print nothing.
            ["adfgvx"],
            ["playfair", "--check"],
            ["hill", "--matrix", "3,3,2,5"],
            ["encodings"],
            ["crib", "THE"],
        ]
        for command in commands:
            with self.subTest(command=command[0] + " " + " ".join(command[1:])):
                code, output = run(*command, "--text", self.text, "--top", "2",
                                   "--quiet")
                self.assertEqual(code, 0, f"{command} exited {code}: {output}")
                self.assertTrue(output.strip(), f"{command} printed nothing")

    def test_playfair_decrypt_on_valid_text(self) -> None:
        from cipher_tool import playfair

        ciphertext = playfair.encrypt(sample_plaintext(200), "MONARCHY")
        code, output = run("playfair", "--text", ciphertext,
                           "--key", "MONARCHY", "--top", "1", "--quiet")
        self.assertEqual(code, 0)
        self.assertIn("Playfair", output)

    def test_playfair_rejects_a_j_and_explains_why(self) -> None:
        # A Playfair square that merges I/J can never emit a J, so a
        # ciphertext containing one cannot have come from it. Saying so
        # plainly is more useful than decrypting it into nonsense.
        code, output = run("playfair", "--text", "ABJDEFGH",
                           "--key", "MONARCHY")
        self.assertEqual(code, 2)
        self.assertIn("J", output)
        self.assertIn("merges I/J", output)

    def test_model_command(self) -> None:
        code, output = run("model")
        self.assertEqual(code, 0)
        self.assertIn("Network use", output)
        self.assertIn("none", output)

    def test_version(self) -> None:
        with self.assertRaises(SystemExit):
            run("--version")

    def test_no_command_prints_help(self) -> None:
        code, output = run()
        self.assertEqual(code, 1)
        self.assertIn("COMMAND", output)


class TestContextCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "message.txt"
        self.path.write_text("HEALI OPASD EHANS", encoding="utf-8")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_add_then_show(self) -> None:
        code, output = run("context", str(self.path),
                           "--add", "people=Admiral Harrow")
        self.assertEqual(code, 0)
        self.assertIn("Saved to", output)

        _, output = run("context", str(self.path))
        self.assertIn("Admiral Harrow", output)
        self.assertIn("ADMIRALHARROW", output)

    def test_clear(self) -> None:
        run("context", str(self.path), "--add", "places=Portsmouth")
        run("context", str(self.path), "--clear", "places")
        _, output = run("context", str(self.path))
        self.assertNotIn("Portsmouth", output)

    def test_requires_a_file_not_inline_text(self) -> None:
        code, output = run("context", "--text", "HEALI")
        self.assertEqual(code, 2)
        self.assertIn("file", output)

    def test_unknown_field_is_rejected(self) -> None:
        code, output = run("context", str(self.path), "--add", "wrong=thing")
        self.assertEqual(code, 2)
        self.assertIn("wrong", output)


class TestByteOrderMark(unittest.TestCase):
    """Notepad's "UTF-8" writes a BOM, so BOM files are the common case."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "notepad.txt"
        # Written as bytes, because the point is a real file on disk saved
        # the way Notepad saves one, not a Python string with an escape in it.
        self.path.write_bytes(b"\xef\xbb\xbfHEALI OPASD EHANS")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_show_reads_a_bom_file(self) -> None:
        code, output = run("show", str(self.path), "--quiet")
        self.assertEqual(code, 0)
        self.assertIn("HEALIOPASDEHANS", output)
        self.assertNotIn("\ufeff", output)

    def test_caesar_reads_a_bom_file(self) -> None:
        code, output = run("caesar", str(self.path), "--top", "1", "--quiet")
        self.assertEqual(code, 0)
        self.assertNotIn("\ufeff", output)

    def test_auto_reads_a_bom_file(self) -> None:
        code, output = run("auto", str(self.path), "--fast", "--top", "1",
                           "--quiet")
        self.assertEqual(code, 0)
        self.assertNotIn("\ufeff", output)

    def test_the_shell_reads_a_bom_file(self) -> None:
        output = run_shell_session(f"load {self.path}", "show", "quit")
        self.assertIn("HEALIOPASDEHANS", output)
        self.assertNotIn("\ufeff", output)

    def test_a_bom_in_inline_text_never_reaches_the_output(self) -> None:
        code, output = run("show", "--text", "\ufeffHEALI OPASD", "--quiet")
        self.assertEqual(code, 0)
        self.assertNotIn("\ufeff", output)

    def test_read_file_hands_on_no_mark(self) -> None:
        # The mark is stopped at the boundary rather than only where it
        # happens to be printed today: every command reads through here, and
        # some of them (analyse, encodings, polybius) work on the raw text
        # rather than the normalised view.
        text, path = cli.read_file(str(self.path))
        self.assertNotIn("\ufeff", text)
        self.assertTrue(text.startswith("HEALI"), text[:10])
        self.assertEqual(path, str(self.path))

    def test_read_source_hands_on_no_mark_from_inline_text(self) -> None:
        arguments = argparse.Namespace(text="\ufeffHEALI OPASD", input=None)
        text, path = cli.read_source(arguments)
        self.assertEqual(text, "HEALI OPASD")
        self.assertIsNone(path)


class TestNonPositiveLimits(unittest.TestCase):
    """One reading of a bad limit, at the one layer that sees every command.

    ``--max-time 0`` used to raise a traceback in three commands, be ignored
    by two more and produce "no candidates" in the rest. Only one of those can
    be right, and none of them is: a deadline of zero is a user error.
    """

    TEXT = "HEALIOPASDEHANSTHEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"

    #: Every command that accepts --max-time and used to disagree about zero.
    TIMED = ("substitution", "columnar", "transposition", "vigenere",
             "bifid", "hill", "auto")

    def test_zero_max_time_is_rejected_by_every_command(self) -> None:
        for command in self.TIMED:
            with self.subTest(command=command):
                code, output = run_rejecting(command, "--text", self.TEXT,
                                             "--max-time", "0")
                self.assertEqual(code, 2)
                self.assertIn("greater than zero seconds", output)

    def test_negative_max_time_is_rejected(self) -> None:
        code, output = run_rejecting("vigenere", "--text", self.TEXT,
                                     "--max-time", "-2")
        self.assertEqual(code, 2)
        self.assertIn("greater than zero seconds", output)

    def test_nan_max_time_is_rejected(self) -> None:
        # NaN compares false against everything, so a `<= 0` guard would let
        # it through and hand a meaningless deadline to the solvers.
        code, output = run_rejecting("vigenere", "--text", self.TEXT,
                                     "--max-time", "nan")
        self.assertEqual(code, 2)
        self.assertIn("greater than zero seconds", output)

    def test_a_real_max_time_still_works(self) -> None:
        code, output = run("vigenere", "--text", self.TEXT, "--max-time",
                           "5", "--max-key-length", "4", "--top", "1",
                           "--quiet")
        self.assertEqual(code, 0)
        self.assertTrue(output.strip())

    def test_zero_top_is_rejected_rather_than_showing_everything(self) -> None:
        code, output = run_rejecting("caesar", "--text", self.TEXT, "--top",
                                     "0")
        self.assertEqual(code, 2)
        self.assertIn("at least 1", output)

    def test_negative_top_is_rejected(self) -> None:
        code, output = run_rejecting("caesar", "--text", self.TEXT,
                                     "--top=-5")
        self.assertEqual(code, 2)
        self.assertIn("at least 1", output)

    def test_shell_top_zero_is_rejected(self) -> None:
        output = run_shell_session("text ABCDEFGHIJ", "top 0", "quit")
        self.assertIn("at least 1", output)
        self.assertNotIn("Showing up to", output)

    def test_shell_top_still_accepts_a_real_number(self) -> None:
        output = run_shell_session("text ABCDEFGHIJ", "top 3", "quit")
        self.assertIn("Showing up to 3 candidates", output)


class TestShellCribRouting(unittest.TestCase):
    """`crib THE` must test THE, and nothing else.

    Every command shares an optional file positional and argparse fills it
    first, so in the shell the crib word landed in the file slot and the
    filename was adopted as the crib -- printing confident verdicts ("this
    rules the Caesar cipher out") about a crib nobody typed.
    """

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "message.txt"
        self.path.write_text(
            "HEALIOPASDEHANSTHEQUICKBROWNFOXJUMPSOVERTHELAZYDOG",
            encoding="utf-8")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_crib_after_load_tests_the_crib_not_the_filename(self) -> None:
        output = run_shell_session(f"load {self.path}", "crib THE", "quit")
        self.assertIn('Crib test: "THE" (3 letters)', output)
        # "message.txt" normalises to MESSAGETXT; its appearance anywhere
        # means the filename was tested as the crib.
        self.assertNotIn("MESSAGETXT", output)

    def test_crib_after_text_finds_the_crib(self) -> None:
        output = run_shell_session("text HEALIOPASDEHANS", "crib THE", "quit")
        self.assertIn('Crib test: "THE" (3 letters)', output)
        self.assertNotIn("Give at least one crib", output)

    def test_several_cribs_in_the_shell(self) -> None:
        output = run_shell_session("text HEALIOPASDEHANS", "crib THE AND",
                                   "quit")
        self.assertIn('Crib test: "THE"', output)
        self.assertIn('Crib test: "AND"', output)

    def test_a_file_and_a_crib_from_the_terminal(self) -> None:
        code, output = run("crib", str(self.path), "THE", "--quiet")
        self.assertEqual(code, 0)
        self.assertIn('Crib test: "THE" (3 letters)', output)

    def test_text_and_a_crib_from_the_terminal(self) -> None:
        code, output = run("crib", "THE", "--text", "HEALIOPASD", "--quiet")
        self.assertEqual(code, 0)
        self.assertIn('Crib test: "THE" (3 letters)', output)

    def test_a_stray_word_is_reported_rather_than_ignored(self) -> None:
        # Silently dropping it would let the user believe an argument they
        # typed had been honoured.
        output = run_shell_session("text ABCDEFGHIJ", "caesar nonsense.txt",
                                   "quit")
        self.assertIn("not an argument", output)
        self.assertNotIn("Candidate 1", output)


class TestShellSurvivesEverything(unittest.TestCase):
    """Nothing typed at the prompt may end the session."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_bare_load_does_not_kill_the_session(self) -> None:
        # Path("") resolves to the current directory, which exists, so a bare
        # `load` used to try to read a directory and take the shell with it.
        output = run_shell_session("load", "text ATTACKATDAWN", "show", "quit")
        self.assertIn("load needs a filename", output)
        self.assertIn("ORIGINAL INPUT", output)
        self.assertIn("Bye.", output)

    def test_loading_a_directory_does_not_kill_the_session(self) -> None:
        output = run_shell_session(f"load {self.directory.name}",
                                   "text ATTACKATDAWN", "show", "quit")
        self.assertIn("directory", output)
        self.assertIn("ORIGINAL INPUT", output)
        self.assertIn("Bye.", output)

    def test_a_missing_file_does_not_kill_the_session(self) -> None:
        output = run_shell_session("load no_such_file.txt",
                                   "text ATTACKATDAWN", "show", "quit")
        self.assertIn("No such file", output)
        self.assertIn("ORIGINAL INPUT", output)

    def test_a_solver_error_does_not_kill_the_session(self) -> None:
        output = run_shell_session("text HEALIOPASD", "vigenere --key 123",
                                   "show", "quit")
        self.assertIn("contains no letters", output)
        self.assertIn("ORIGINAL INPUT", output)


class TestShellErrorReporting(unittest.TestCase):
    def test_unknown_command_says_so_before_nothing_loaded(self) -> None:
        # "nothing loaded" would send the user to load a file that could not
        # have helped: the command does not exist either way.
        output = run_shell_session("wibble", "quit")
        self.assertIn("unknown command", output)
        self.assertNotIn("nothing loaded", output)

    def test_a_real_command_with_nothing_loaded_still_says_so(self) -> None:
        output = run_shell_session("caesar", "quit")
        self.assertIn("nothing loaded", output)
        self.assertNotIn("unknown command", output)

    def test_errors_do_not_carry_the_exception_class_name(self) -> None:
        output = run_shell_session("text HEALIOPASD", "vigenere --key 123",
                                   "quit")
        self.assertIn("Vigenere key '123' contains no letters", output)
        self.assertNotIn("ValueError", output)

    def test_an_exception_with_no_message_is_still_named(self) -> None:
        # The class name is the fallback, not the default: printing "error:"
        # and nothing else would be worse than printing the class.
        self.assertEqual(cli._error_message(ValueError()),
                         "ValueError (no further detail)")
        self.assertEqual(cli._error_message(ValueError("bad key")), "bad key")


class TestCaesarShiftLabel(unittest.TestCase):
    """A key that is not the key the user typed must not be reported as it."""

    def test_a_shift_beyond_25_reports_the_shift_actually_used(self) -> None:
        _, output = run("caesar", "--text", "ATTACKATDAWN", "--shift", "99",
                        "--quiet")
        self.assertIn("shift=21", output)

    def test_a_negative_shift_reports_the_shift_actually_used(self) -> None:
        _, output = run("caesar", "--text", "ATTACKATDAWN", "--shift", "-3",
                        "--quiet")
        self.assertIn("shift=23", output)

    def test_encrypting_reports_it_too(self) -> None:
        _, output = run("caesar", "--text", "ATTACKATDAWN", "--shift", "99",
                        "--encrypt", "--quiet")
        self.assertIn("shift=21", output)

    def test_an_ordinary_shift_is_reported_plainly(self) -> None:
        _, output = run("caesar", "--text", "ATTACKATDAWN", "--shift", "3",
                        "--quiet")
        self.assertIn("shift=3", output)
        self.assertNotIn("only 26 shifts exist", output)


if __name__ == "__main__":
    unittest.main()


class TestPasteSession(unittest.TestCase):
    """The double-click flow: paste a ciphertext, read the plaintext.

    This exists because a competition ciphertext arrives as a block of
    five-letter groups over several lines, and pasting that into the command
    shell made every line an unknown command.
    """

    def _paste(self, *lines: str) -> str:
        script = io.StringIO("\n".join(lines) + "\n")
        buffer = io.StringIO()
        original = sys.stdin
        sys.stdin = script
        try:
            with contextlib.redirect_stdout(buffer), \
                    contextlib.redirect_stderr(buffer):
                cli.main(["paste"])
        finally:
            sys.stdin = original
        return buffer.getvalue()

    def test_multi_line_paste_is_one_message(self) -> None:
        ciphertext = caesar.encrypt(sample_plaintext(200), 3)
        grouped = [ciphertext[i:i + 25] for i in range(0, len(ciphertext), 25)]
        output = self._paste(*grouped, "", "q")
        self.assertIn("BEST ANSWER", output)
        self.assertIn(sample_plaintext(40), output.replace("\n", "")
                      .replace("  ", ""))
        self.assertNotIn("unknown command", output)

    def test_the_plaintext_is_shown_unbroken(self) -> None:
        # Not poured back into the ciphertext's five-letter groups: those are
        # not word boundaries, and "ISNOT HINGS" reads like a bad decryption.
        ciphertext = caesar.encrypt(sample_plaintext(120), 3)
        output = self._paste(ciphertext, "", "q")
        collapsed = "".join(output.split())
        self.assertIn(sample_plaintext(60), collapsed)

    def test_it_says_read_it_before_submitting(self) -> None:
        output = self._paste(caesar.encrypt(sample_plaintext(200), 3), "", "q")
        self.assertIn("READ IT BEFORE YOU SUBMIT IT", output)
        self.assertIn("not a verdict", output)

    def test_nothing_pasted_is_explained(self) -> None:
        output = self._paste("", "")
        self.assertIn("No letters were pasted", output)

    def test_q_at_the_paste_prompt_backs_out(self) -> None:
        # Someone who opens this by mistake types 'q'. It used to be
        # analysed as a one-letter ciphertext, complete with a key.
        output = self._paste("q")
        self.assertIn("No letters were pasted", output)
        self.assertNotIn("BEST ANSWER", output)

    def test_a_ciphertext_starting_with_q_still_works(self) -> None:
        # The quit word is only honoured as the very first thing typed, so a
        # real message is never mistaken for it.
        ciphertext = caesar.encrypt(sample_plaintext(150), 3)
        output = self._paste(ciphertext, "Q", "", "q")
        self.assertIn("BEST ANSWER", output)

    def test_unsolvable_input_is_not_sold_as_an_answer(self) -> None:
        import random

        generator = random.Random(4)
        noise = "".join(generator.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                        for _ in range(300))
        output = self._paste(noise, "", "q")
        self.assertIn("probably NOT the plaintext", output)
        self.assertIn("did not score like English", output)

    def test_unencrypted_input_says_so(self) -> None:
        output = self._paste(sample_plaintext(300), "", "q")
        self.assertIn("DOES NOT APPEAR TO BE ENCRYPTED", output)

    def test_leading_blank_lines_do_not_end_the_paste(self) -> None:
        # People press Enter before pasting. That must not read as "done".
        ciphertext = caesar.encrypt(sample_plaintext(150), 3)
        output = self._paste("", "", ciphertext, "", "q")
        self.assertIn("BEST ANSWER", output)

    def test_quit_leaves_immediately(self) -> None:
        output = self._paste(caesar.encrypt(sample_plaintext(120), 3), "", "q")
        self.assertNotIn("Searching harder", output)


def paste_session(*lines: str) -> str:
    """Drive the paste flow with a scripted stdin and capture its output."""
    script = io.StringIO("\n".join(lines) + "\n")
    buffer = io.StringIO()
    original = sys.stdin
    sys.stdin = script
    try:
        with contextlib.redirect_stdout(buffer), \
                contextlib.redirect_stderr(buffer):
            cli.main(["paste"])
    finally:
        sys.stdin = original
    return buffer.getvalue()


class TestPasteMenuDoesNotSwallowInput(unittest.TestCase):
    """The answer menu must not silently eat what is typed at it.

    Found by watching a real first run. The menu loop ended with

        # Anything else, including a bare Enter, means "search harder".

    so every unrecognised input became "try harder" with no reply. Two
    separate failures came out of that one line:

    1. Someone who has just been told their paste was not encrypted pastes
       the REAL ciphertext at the ``>`` prompt -- the obvious thing to do,
       since the prompt is where you type. The message was discarded without
       a word, and the tool re-searched the previous text. The user watched
       it think for a while and print an answer, so nothing looked wrong;
       the answer just had nothing to do with what they pasted.
    2. Typing anything else -- a stray line copied off the page, a comment --
       silently advanced the effort level, and once at 'deep' printed
       "Already searched as hard as this tool goes" no matter what was typed.
       The menu looked broken because it never said "I do not know that one".
    """

    def _corpus(self) -> tuple[str, str]:
        corpus = sample_plaintext(400)
        return corpus[:150], corpus[200:350]

    def test_a_ciphertext_pasted_at_the_menu_is_solved_not_discarded(self) -> None:
        first_plain, second_plain = self._corpus()
        output = paste_session(
            caesar.encrypt(first_plain, 3), "",
            caesar.encrypt(second_plain, 7), "",
            "q",
        )
        collapsed = "".join(output.split())
        self.assertIn(second_plain[:60], collapsed,
                      "the second message must actually be solved")

    def test_a_multi_line_ciphertext_at_the_menu_is_one_message(self) -> None:
        """A website paste arrives as several lines, not one."""
        first_plain, second_plain = self._corpus()
        second = caesar.encrypt(second_plain, 7)
        groups = [second[i:i + 25] for i in range(0, len(second), 25)]
        output = paste_session(
            caesar.encrypt(first_plain, 3), "",
            *groups, "",
            "q",
        )
        collapsed = "".join(output.split())
        self.assertIn(second_plain[:60], collapsed)

    def test_unrecognised_input_is_answered_not_silently_obeyed(self) -> None:
        first_plain, _ = self._corpus()
        output = paste_session(caesar.encrypt(first_plain, 3), "",
                               "Mea culpa", "q")
        self.assertIn("not one of the options", output)
        self.assertNotIn("Searching harder", output)

    def test_a_bare_enter_starts_another_message(self) -> None:
        """Enter takes the next message; it no longer means "search harder".

        Changed deliberately on the operator's instruction, 2026-08-22: "the
        pasting and enter on the new blank line is fine but the other options
        are not needed... just get me the answer". Making the reader press
        Enter to obtain a real search is asking them to do the tool's job, and
        the ordinary path already climbs fast -> normal -> deep by itself
        before the answer is printed.

        This replaces ``test_a_bare_enter_still_means_try_harder``, which
        pinned the old meaning. The coverage is kept rather than dropped: the
        assertion is simply the other way round, and "harder" survives as a
        typed word for anyone who wants it.
        """
        first_plain, _ = self._corpus()
        output = paste_session(caesar.encrypt(first_plain, 3), "", "", "q")
        self.assertNotIn("Searching harder", output)
        self.assertIn("Paste your ciphertext below", output.split(
            "BEST ANSWER")[-1], "Enter did not start a new message")

    def test_harder_still_works_as_a_typed_word(self) -> None:
        """Nothing was removed from the menu, only stopped shouting."""
        first_plain, _ = self._corpus()
        output = paste_session(caesar.encrypt(first_plain, 3), "",
                               "harder", "q")
        self.assertIn("Searching harder", output)


#: A progressive-shift (Trithemius) cipher, expressed as the 26-letter
#: Vigenere key it is equivalent to. This is what National Cipher Challenge
#: 2025 challenge 10B turned out to be, and it is the case that exposed the
#: paste flow stopping short of the answer.
PROGRESSIVE_SHIFT_KEY = "DEFGHIJKLMNOPQRSTUVWXYZABC"


class TestPasteFlowReachesTheAnswerItself(unittest.TestCase):
    """Paste in, answer out. The user should not have to drive the search.

    The paste flow ran one `fast` pass and printed whatever came back, even
    when it knew the reading was weak. On a real competition ciphertext that
    meant a screenful of gibberish under the headline BEST ANSWER, plus an
    instruction to try 'normal' or 'deep' -- which in this screen means
    pressing Enter, something the message never said. The tool had the answer
    within reach and asked the user to go and get it.
    """

    def test_promising_is_not_good_enough_to_stop_on(self) -> None:
        """The trap that makes the obvious fix wrong.

        Escalating only while the reading is 'weak' looks right and is not.
        Measured on the progressive-shift cipher below: `fast` gives weak and
        wrong, `normal` gives PROMISING and still wrong, `deep` gives strong
        and correct. Stopping at the first non-weak label would hand back a
        confident-sounding wrong answer -- worse than the weak one, because
        the label no longer warns anybody.
        """
        self.assertTrue(cli._should_search_harder("unlikely"))
        self.assertTrue(cli._should_search_harder("weak"))
        self.assertTrue(cli._should_search_harder("promising"))
        self.assertFalse(cli._should_search_harder("strong"))

    def test_a_cipher_fast_cannot_solve_is_escalated_without_being_asked(self) -> None:
        from cipher_tool import vigenere

        plain = sample_plaintext(600)
        ciphertext = vigenere.encrypt(plain, PROGRESSIVE_SHIFT_KEY)
        # Nothing typed but the ciphertext and a quit: no Enter presses.
        output = paste_session(ciphertext, "", "q")
        collapsed = "".join(output.split())
        self.assertIn(plain[:60], collapsed,
                      "the flow must reach the answer on its own")
        self.assertIn("(deep)", output,
                      "it must not stop at 'normal', which is wrong here")

    def test_a_message_solved_at_fast_is_not_searched_again(self) -> None:
        """Escalation must cost nothing when the first pass already won."""
        output = paste_session(caesar.encrypt(sample_plaintext(200), 3),
                               "", "q")
        self.assertNotIn("Searching harder", output)

    def test_text_that_was_never_encrypted_is_not_escalated(self) -> None:
        """Nothing is gained by searching harder for a cipher that is absent,
        and this is the exact path where a deeper search used to invent one.
        """
        output = paste_session(sample_plaintext(300), "", "q")
        self.assertIn("DOES NOT APPEAR TO BE ENCRYPTED", output)
        self.assertNotIn("Searching harder", output)

    def test_a_weak_reading_is_not_printed_in_full(self) -> None:
        """A screenful of gibberish is not an answer, and printing it under
        the word ANSWER buries the one line that actually helps.
        """
        import random

        from cipher_tool.auto import auto_solve

        generator = random.Random(4)
        noise = "".join(generator.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                        for _ in range(300))
        result = auto_solve(noise, effort="fast", top=5, seed=1)
        rendered = cli._render_answer(result)
        best = result.candidates.best()

        self.assertNotIn("".join(best.plaintext[-40:].split()),
                         "".join(rendered.split()),
                         "the whole wrong plaintext should not be dumped")
        self.assertIn("probably NOT the plaintext", rendered)


class TestNumericCiphertextIsNotCalledNothing(unittest.TestCase):
    """A numeric ciphertext is a cipher, not an empty paste.

    The paste screen works on letters, so a Polybius ciphertext -- digits,
    no letters -- normalised to nothing and the user was told "No letters
    were pasted, so there is nothing to work on. Run it again and paste the
    ciphertext when prompted." They had pasted the ciphertext. It is the one
    reply that guarantees the person tries the exact same thing again.

    This matters for the National Cipher Challenge specifically: Polybius,
    Nihilist and ADFGVX-written-with-digits all arrive as numbers, and the
    toolkit HAS solvers for that shape -- just not on this screen.
    """

    # These use digits the tool CANNOT read as a cipher. A readable numeric
    # message is now solved outright rather than explained -- see
    # TestPasteSolvesNonLetterMessages -- so the explanation is reserved for
    # what is genuinely beyond it, which is where it belongs.
    UNREADABLE = "2 4 6 8 1 3 5 7 9 0 2 4 6 8 1 3"

    def test_digits_are_not_reported_as_an_empty_paste(self) -> None:
        output = paste_session(self.UNREADABLE, "", "q")
        self.assertNotIn("No letters were pasted", output)

    def test_it_says_what_was_actually_pasted(self) -> None:
        output = paste_session(self.UNREADABLE, "", "q")
        self.assertIn("digits", output.lower())

    def test_it_names_a_command_that_can_work_on_numbers(self) -> None:
        """A dead end is worse than a wrong guess. Point somewhere real."""
        output = paste_session(self.UNREADABLE, "", "q")
        self.assertIn("polybius", output.lower())

    def test_a_genuinely_empty_paste_still_says_so(self) -> None:
        """The fix must not swallow the case the old message was written for."""
        output = paste_session("", "")
        self.assertIn("No letters were pasted", output)


class TestCopyReadyBlockIsSubmissionSafe(unittest.TestCase):
    """One wrong space in a competition answer is a rejected answer.

    The letters a decryption produces are exact. The SPACES are a guess made
    afterwards by a lexicon of 8,342 words, and a competition message is full
    of names -- CHARLES, TERNAN, DROOD -- that no lexicon written for a
    cipher toolkit will ever hold. So the spaced form is a reading aid and
    must never be the thing someone pastes into an answer box.

    The gate existed but measured the wrong thing: the fraction of LETTERS
    inside known words. Measured on a real solve, that was comfortably over
    the threshold while 9.5 per cent of the TOKENS were not words at all --
    `MY DEARCH A RLES` for `MY DEAR CHARLES`. Letters inside known words
    cannot see a word that has been cut in half.
    """

    def _scorer(self):
        from cipher_tool.scoring import default_scorer

        return default_scorer()

    @classmethod
    def setUpClass(cls) -> None:
        # Ordinary covered prose with a few names dropped into it, which is
        # what a competition message actually looks like. Measured on the
        # real one: the display gate passed at 0.889 while 43 of its 670
        # tokens were not words at all.
        cls.WITH_NAMES = (
            sample_plaintext(400) + "CHARLESTERNANEDWINDROODDODGE"
        )

    def test_a_message_with_names_is_not_offered_spaced_for_submission(self) -> None:
        from cipher_tool import cli as cli_module

        self.assertFalse(
            cli_module._segmentation_is_submission_safe(
                self.WITH_NAMES, self._scorer()),
            "guessed spacing must never reach the block someone pastes into "
            "an answer box when one wrong space is a rejected answer",
        )

    def test_that_message_is_still_readable_on_screen(self) -> None:
        """The reading aid must survive; only the COPY block is restricted.

        These are different jobs. On screen the spacing helps somebody check
        an answer they can see is right. In the copy block it is a liability,
        because nobody proof-reads what they paste.
        """
        from cipher_tool import cli as cli_module

        self.assertTrue(
            cli_module._segmentation_is_trustworthy(
                self.WITH_NAMES, self._scorer())
        )

    def test_ordinary_covered_english_still_gets_its_spaces(self) -> None:
        """The strict gate must not refuse everything -- that would be
        useless, and would quietly remove a feature rather than fix it."""
        from cipher_tool import cli as cli_module

        plaintext = "IHAVETHEHONOURTOREPORTTHATTHECOLUMNREACHEDTHEBRIDGE"
        self.assertTrue(
            cli_module._segmentation_is_submission_safe(
                plaintext, self._scorer())
        )

    def test_the_letters_are_always_exact(self) -> None:
        """Whatever else it prints, the copy block contains the decryption."""
        from cipher_tool import cli as cli_module
        from cipher_tool.auto import auto_solve

        plain = sample_plaintext(300)
        result = auto_solve(caesar.encrypt(plain, 5), effort="fast", top=1,
                            seed=1)
        block = cli_module.render_submission(result)
        self.assertIn(plain[:80], "".join(block.split()))


class TestPasteSolvesNonLetterMessages(unittest.TestCase):
    """Paste in, answer out -- even when the message is not letters.

    Pointing someone at the right command was an improvement on telling them
    they had pasted nothing, but it is still homework. The solvers that can
    read a symbol stream already exist, so the paste screen runs them.
    """

    def test_a_numeric_polybius_message_is_solved_not_just_explained(self) -> None:
        from cipher_tool import polybius

        plaintext = polybius.PolybiusSquare.standard().prepare(
            sample_plaintext(300))
        ciphertext = polybius.encrypt(
            plaintext, polybius.PolybiusSquare.standard("TEMPEST"))
        output = paste_session(ciphertext, "", "q")
        self.assertIn(plaintext[:50], "".join(output.split()))

    def test_morse_is_decoded_rather_than_refused(self) -> None:
        output = paste_session("- .... . / .-. .- .. -.", "", "q")
        self.assertIn("THE", output)
        self.assertIn("RAIN", output)

    def test_symbols_it_cannot_read_still_get_the_explanation(self) -> None:
        """The guidance must survive for the cases that remain unreadable."""
        output = paste_session("2 4 6 8 1 3 5 7 9 0 2 4", "", "q")
        self.assertIn("NOT A LETTER CIPHER", output.upper())


def card_message(cards: int = 300, seed: int = 1, slip: int | None = 401) -> str:
    """A paired-symbol ciphertext of the shape the operator actually pasted.

    Ranks and suits alternate, ranks 2 to 9 are digits, and one symbol is
    missing -- which is the ordinary state of a hand transcription and the
    reason the real message had an odd symbol count.
    """
    import random

    generator = random.Random(seed)
    stream = "".join(
        generator.choice("23456789XJQKA") + generator.choice("CDHS")
        for _ in range(cards)
    )
    if slip is not None:
        stream = stream[:slip] + stream[slip + 1:]
    return " ".join(stream[i:i + 5] for i in range(0, len(stream), 5))


class TestASymbolStreamIsNotReadAsLetters(unittest.TestCase):
    """The bug this whole change exists for, driven through the real screen.

    A message of 1,251 alphanumeric symbols -- 891 letters and 360 digits --
    had its digits dropped in silence, the wreckage solved as a monoalphabetic
    substitution, and the result offered at `promising`. The screen printed
    "Read 891 letters. Working...", which describes the leftovers of a filter
    as though it described the paste.
    """

    def setUp(self) -> None:
        self.output = paste_session(card_message(), "", "q")

    def test_it_reports_what_was_pasted_not_what_survived(self) -> None:
        self.assertIn("symbols", self.output)
        self.assertIn("digits", self.output)
        self.assertNotIn("Read 369 letters", self.output)

    def test_it_names_the_structure_it_found(self) -> None:
        self.assertIn("playing-card deck", self.output)
        self.assertIn("Nothing was changed", self.output)

    def test_it_does_not_offer_a_letters_only_reading_as_the_answer(self) -> None:
        self.assertNotIn("BEST ANSWER", self.output)
        self.assertNotIn("Monoalphabetic substitution", self.output)

    def test_it_refuses_with_a_reason_and_a_next_step(self) -> None:
        self.assertIn("IT IS NOT SOLVED", self.output.upper())
        self.assertIn("homophonic", self.output)

    def test_it_says_what_a_letters_only_reading_would_cost(self) -> None:
        self.assertIn("would discard", self.output)

    def test_trying_harder_does_not_fall_back_to_the_letters_pipeline(
        self,
    ) -> None:
        """A bare Enter is 'search harder', never 'search something else'.

        Self-escalation on this path is how the defect spent three minutes
        climbing fast to normal to deep and arriving at a confident wrong
        answer.
        """
        output = paste_session(card_message(), "", "", "q")
        self.assertNotIn("BEST ANSWER", output)
        self.assertNotIn("Monoalphabetic substitution", output)

    def test_the_letters_only_reading_is_behind_an_explicit_choice(self) -> None:
        output = paste_session(card_message(), "", "l", "q")
        self.assertIn("NOT an answer", output)
        self.assertIn("discard", output.lower())


class TestAnalyseReportsTheSymbolStreamToo(unittest.TestCase):
    """The refusal screen sends people to `analyse`; it must have the answer.

    `analyse` reads the letters view like everything else, so on a card
    cipher it reported 369 letters and nine distinct symbols and said nothing
    about the 630 symbols that were actually there.
    """

    def test_it_names_the_structure_and_the_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cards.txt"
            path.write_text(card_message(), encoding="utf-8")
            code, output = run("analyse", str(path))
        self.assertEqual(code, 0)
        self.assertIn("symbols:", output)
        self.assertIn("digits", output)
        self.assertIn("playing-card deck", output)

    def test_an_ordinary_ciphertext_gets_no_structure_claim(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "caesar.txt"
            path.write_text(caesar.encrypt(sample_plaintext(300), 3),
                            encoding="utf-8")
            code, output = run("analyse", str(path))
        self.assertEqual(code, 0)
        self.assertNotIn("paired alphabet", output)
        self.assertNotIn("playing-card deck", output)


class TestTheInventoryLineIsAlwaysHonest(unittest.TestCase):
    def test_a_letters_only_paste_still_says_what_it_read(self) -> None:
        output = paste_session(caesar.encrypt(sample_plaintext(200), 3),
                               "", "q")
        self.assertIn("200 symbols: 200 letters", output)
        self.assertIn("BEST ANSWER", output)

    def test_an_ordinary_message_carrying_a_date_takes_the_letters_path(
        self,
    ) -> None:
        """Over-refusal is the second-worst outcome, so measure the boundary.

        900 letters and 8 digits is a message with a date in it, not a symbol
        cipher, and it must be solved exactly as it is today.
        """
        ciphertext = caesar.encrypt(sample_plaintext(900), 3) + " 12 03 19 07"
        output = paste_session(ciphertext, "", "q")
        self.assertIn("BEST ANSWER", output)
        self.assertIn("908 symbols: 900 letters and 8 digits", output)
