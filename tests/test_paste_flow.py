"""Clipboard delivery, repeated pastes and the shared escalation deadline."""

from __future__ import annotations

import argparse
import contextlib
import io
import unittest
from unittest.mock import Mock, patch

from cipher_tool import auto, cli, clipboard, crossed, polybius, substitution
from cipher_tool.candidates import Candidate, CandidateSet
from cipher_tool.normalize import normalize
from cipher_tool.statistics import analyse


class TestClipboard(unittest.TestCase):
    def test_platform_commands_and_exact_payload(self) -> None:
        for platform, available, command, encoding in (
            ("win32", "clip", ["/tool/clip"], "utf-16le"),
            ("darwin", "pbcopy", ["/tool/pbcopy"], "utf-8"),
            ("linux", "wl-copy", ["/tool/wl-copy"], "utf-8"),
            ("linux", "xclip", ["/tool/xclip", "-selection", "clipboard"],
             "utf-8"),
        ):
            with self.subTest(platform=platform, available=available):
                with patch.object(clipboard.sys, "platform", platform), \
                        patch.object(clipboard.shutil, "which", side_effect=
                                     lambda name: "/tool/" + name
                                     if name == available else None), \
                        patch.object(clipboard.subprocess, "run",
                                     return_value=Mock(returncode=0)) as run:
                    self.assertTrue(clipboard.copy_to_clipboard("EXACTLETTERS"))
                self.assertEqual(run.call_args.args[0], command)
                self.assertEqual(run.call_args.kwargs["input"],
                                 "EXACTLETTERS".encode(encoding))
                self.assertNotIn("shell", run.call_args.kwargs)
                self.assertEqual(run.call_args.kwargs["timeout"], 2)

    def test_missing_clipboard_never_raises(self) -> None:
        for platform in ("win32", "darwin", "linux", "unknown"):
            with self.subTest(platform=platform):
                with patch.object(clipboard.sys, "platform", platform), \
                        patch.object(clipboard.shutil, "which", return_value=None), \
                        patch.object(clipboard.subprocess, "run") as run:
                    self.assertFalse(clipboard.copy_to_clipboard("ANSWER"))
                run.assert_not_called()

    def test_broken_clipboard_never_raises(self) -> None:
        for error in (OSError("no display"),
                      clipboard.subprocess.TimeoutExpired("pbcopy", 2)):
            with self.subTest(error=error):
                with patch.object(clipboard.sys, "platform", "darwin"), \
                        patch.object(clipboard.shutil, "which", return_value="pbcopy"), \
                        patch.object(clipboard.subprocess, "run", side_effect=error):
                    self.assertFalse(clipboard.copy_to_clipboard("ANSWER"))

    def test_linux_falls_back_after_wayland_failure(self) -> None:
        with patch.object(clipboard.sys, "platform", "linux"), \
                patch.object(clipboard.shutil, "which", side_effect=lambda name: name), \
                patch.object(clipboard.subprocess, "run", side_effect=[
                    Mock(returncode=1), Mock(returncode=0),
                ]) as run:
            self.assertTrue(clipboard.copy_to_clipboard("ANSWER"))
        self.assertEqual([call.args[0] for call in run.call_args_list],
                         [["wl-copy"], ["xclip", "-selection", "clipboard"]])


class TestPasteFlow(unittest.TestCase):
    def result(self, confidence: str = "strong", plaintext: str | None = None,
               score: float = 0) -> auto.AutoResult:
        normalized = normalize("WKHUHLVQRWKLQJVRIDWDOWRFKDUDFWHU")
        candidate = Candidate(
            method="Vigenere", key="key=LEMON", score=score,
            plaintext=plaintext or "THEREISNOTHINGSOFATALTOCHARACTERASHALFFINISHEDTASK",
            diagnostics={"normalised_score": -0.5, "word_coverage": 1.0,
                         "confidence_cap": confidence},
        )
        return auto.AutoResult(normalized=normalized, stats=analyse(normalized),
                               candidates=CandidateSet([candidate]), effort="fast")

    def session(self, script: str, *options: str) -> tuple[int, str]:
        output = io.StringIO()
        with patch("sys.stdin", io.StringIO(script)), \
                contextlib.redirect_stdout(output):
            code = cli.main(["paste", *options])
        return code, output.getvalue()

    def test_blank_first_line_exits_without_search(self) -> None:
        with patch.object(cli.auto_module, "auto_solve") as solve:
            code, _ = self.session("\nIGNORED\n")
        self.assertEqual(code, 0)
        solve.assert_not_called()

    def test_each_message_starts_fast_and_copies_its_own_exact_letters(self) -> None:
        results = [self.result(plaintext="FIRSTEXACTANSWER"),
                   self.result(plaintext="SECONDEXACTANSWER")]
        with patch.object(cli.auto_module, "auto_solve", side_effect=results) as solve, \
                patch.object(cli, "copy_to_clipboard", return_value=True) as copy:
            code, output = self.session("FIRSTCIPHERTEXTMESSAGELINE\n\n"
                                        "SECONDCIPHERTEXTMESSAGELINE\n\n\n")
        self.assertEqual(code, 0)
        self.assertEqual([call.kwargs["effort"] for call in solve.call_args_list],
                         ["fast", "fast"])
        self.assertEqual([call.args[0] for call in copy.call_args_list],
                         ["FIRSTEXACTANSWER", "SECONDEXACTANSWER"])
        self.assertEqual(output.count("BEST ANSWER"), 2)
        self.assertEqual(output.count("Copied the exact"), 2)
        self.assertIn("Paste the next message (or press Enter on a blank line to quit)",
                      output)

    def test_no_copy_opts_out_and_keeps_the_answer(self) -> None:
        with patch.object(cli.auto_module, "auto_solve", return_value=self.result()), \
                patch.object(cli, "copy_to_clipboard") as copy:
            _, output = self.session("CIPHERTEXT\n\n\n", "--no-copy")
        copy.assert_not_called()
        self.assertIn("BEST ANSWER", output)
        self.assertIn("Not copied to clipboard (--no-copy)", output)

    def test_copy_removes_only_the_exact_plaintexts_line_breaks(self) -> None:
        result = self.result(plaintext="EXACT\r\nDECRYPTED\nLETTERS")
        with patch.object(cli.auto_module, "auto_solve", return_value=result), \
                patch.object(cli, "copy_to_clipboard", return_value=True) as copy:
            self.session("CIPHERTEXT\n\n\n")
        copy.assert_called_once_with("EXACTDECRYPTEDLETTERS")

    def test_short_and_symbol_messages_can_be_pasted_next(self) -> None:
        for line in ("ABC DEF", "12345 54321", ".... . .-.. .-.. ---"):
            with self.subTest(line=line):
                self.assertTrue(cli._looks_like_a_pasted_message(line))

    def test_many_messages_do_not_grow_the_call_stack(self) -> None:
        count = 1100
        with patch.object(cli.auto_module, "auto_solve", return_value=self.result()) as solve, \
                patch.object(cli, "_render_answer", return_value="ANSWER"):
            code, _ = self.session("CIPHERTEXTMESSAGELINE\n\n" * count + "\n",
                                   "--no-copy")
        self.assertEqual(code, 0)
        self.assertEqual(solve.call_count, count)

    def test_clipboard_failure_keeps_the_answer(self) -> None:
        with patch.object(cli.auto_module, "auto_solve", return_value=self.result()), \
                patch.object(cli, "copy_to_clipboard", return_value=False):
            code, output = self.session("CIPHERTEXT\n\n\n")
        self.assertEqual(code, 0)
        self.assertIn("BEST ANSWER", output)
        self.assertIn("tool unavailable or failed", output)

    def test_summary_precedes_the_plaintext(self) -> None:
        result = self.result()
        output = cli._render_answer(result, elapsed=1.234)
        self.assertIn("Solved in 1.23s | Vigenere | key=LEMON", output)
        self.assertLess(output.index("Solved in"),
                        output.index(result.candidates.best().plaintext[:30]))
        self.assertIn("  Cipher      : Vigenere", output)
        self.assertIn("  Key         : key=LEMON", output)

    def test_escalation_stops_at_first_clear_english(self) -> None:
        for confidences in (("strong",), ("weak", "strong"),
                            ("weak", "promising", "strong")):
            with self.subTest(confidences=confidences):
                results = [self.result(confidence, score=index)
                           for index, confidence in enumerate(confidences)]
                with patch.object(cli.auto_module, "auto_solve", side_effect=results) as solve:
                    _, output = self.session("CIPHERTEXT\n\n\n", "--no-copy")
                self.assertEqual([call.kwargs["effort"] for call in solve.call_args_list],
                                 list(cli.EFFORT_LADDER[:len(confidences)]))
                self.assertEqual(output.count("Searching harder"), len(confidences) - 1)

    def test_escalation_shares_one_deadline(self) -> None:
        now = [100.0]
        budgets = []

        def solve(source, **options):
            budgets.append(options["max_time"])
            now[0] += 4.0
            return self.result("weak")

        with patch.object(cli.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(cli.auto_module, "auto_solve", side_effect=solve):
            _, output = self.session("CIPHERTEXT\n\n\n", "--max-time", "6")
        self.assertEqual(budgets, [2.0, 1.0])
        self.assertIn("Overall time limit reached", output)
        self.assertNotIn("(deep)", output)

    def test_a_new_message_gets_a_fresh_budget(self) -> None:
        now = [0.0]

        def solve(source, **options):
            now[0] += 1.0
            return self.result()

        with patch.object(cli.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(cli.auto_module, "auto_solve", side_effect=solve) as search:
            self.session("FIRSTCIPHERTEXTMESSAGELINE\n\n"
                         "SECONDCIPHERTEXTMESSAGELINE\n\n\n", "--no-copy")
        self.assertEqual([call.kwargs["max_time"] for call in search.call_args_list],
                         [40.0, 40.0])


class TestSymbolBudgets(unittest.TestCase):
    def test_zero_remaining_budget_starts_no_symbol_solver(self) -> None:
        args = argparse.Namespace(max_time=0.0, seed=1)
        with patch.object(cli.encodings, "solve") as solve:
            result = cli._solve_symbol_stream("12345" * 20, args)
        self.assertIsNone(result)
        solve.assert_not_called()

    def test_polybius_passes_its_budget_to_the_climb(self) -> None:
        ciphertext = polybius.encrypt("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 2)
        with patch.object(substitution, "solve", return_value=CandidateSet()) as solve:
            polybius.solve_unknown_square(ciphertext, time_budget=0.25, seed=1)
        self.assertEqual(solve.call_args.kwargs["time_budget"], 0.25)

    def test_crossed_stops_annealing_at_the_deadline(self) -> None:
        from random import Random

        scorer = Mock()
        scorer.score_values.return_value = 0
        with patch.object(crossed.time, "monotonic", return_value=10):
            reading = crossed._anneal([0, 1], [1, 0], scorer, 10, 1000,
                                      Random(1), deadline=10)
        self.assertEqual(scorer.score_values.call_count, 1)
        self.assertEqual(len(reading.plaintext), 4)


if __name__ == "__main__":
    unittest.main()
