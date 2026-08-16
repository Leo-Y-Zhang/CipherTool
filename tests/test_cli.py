"""Tests for the command line interface.

Covers the plumbing (input handling, argument parsing, error reporting) and
runs each command end to end on a small ciphertext to prove it does not
crash and prints what it promises.
"""

from __future__ import annotations

import contextlib
import io
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
            ["transposition", "--max-key-length", "4"],
            ["transposition", "--routes"],
            ["polybius"], ["bifid", "--max-period", "4"],
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


if __name__ == "__main__":
    unittest.main()
