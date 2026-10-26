"""Trifid known answers, round trips, search calibration and routing."""

import argparse
import itertools
import random
import time
import unittest
from unittest.mock import patch

from cipher_tool import auto, cli, trifid
from cipher_tool.candidates import CandidateSet
from cipher_tool.normalize import letters_only, normalize
from cipher_tool.scoring import load_corpus_text


class TestTrifidCipher(unittest.TestCase):
    """Independent coordinate vectors and all supported period conventions."""

    def test_known_answer(self) -> None:
        """ABC's coordinates 000/001/002 transpose to 000000012 = AAF."""
        self.assertEqual(trifid.encrypt("ABCDE", period=3), "AAFBK")
        self.assertEqual(trifid.decrypt("AAFBK", period=3), "ABCDE")

    def test_round_trips(self) -> None:
        """Ragged blocks, whole-message blocks, J and both fillers survive."""
        for filler in ("+", "."):
            cube = trifid.trifid_cube("A KEY WITH REPEATS", filler=filler)
            plain = "JAZZANDQUICKBROWNFOXES" + filler + "END"
            for period in (1, 3, 5, 7, 20, 100, 0, None):
                with self.subTest(filler=filler, period=period):
                    cipher = trifid.encrypt(plain, cube, period)
                    self.assertEqual(trifid.decrypt(cipher, cube, period), plain)
                    stream = trifid.letters_to_trits(cipher, cube)
                    self.assertEqual(trifid.trits_to_letters(stream, cube), cipher)
                    self.assertEqual(trifid.decrypt_trits(stream, cube, period), plain)

    def test_explicit_cube_string(self) -> None:
        """A full cube string preserves its arrangement and custom filler."""
        cube = trifid.trifid_cube("TEMPEST", filler=".")
        self.assertEqual(trifid.encrypt("JAZZ.", cube.letters, 3),
                         trifid.encrypt("JAZZ.", cube, 3))

    def test_empty_input(self) -> None:
        """Empty messages still validate periods and return empty text."""
        self.assertEqual(trifid.encrypt("", period=0), "")
        self.assertEqual(trifid.decrypt("", period=None), "")

    def test_all_symbol_orders(self) -> None:
        """The raw helper respects each of the six explicit symbol mappings."""
        cube = trifid.trifid_cube("TEMPEST")
        plain = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG+"
        stream = trifid.letters_to_trits(trifid.encrypt(plain, cube, 5), cube)
        for order in itertools.permutations("|/\\"):
            symbols = "".join(order)
            raw = stream.translate(str.maketrans("012", symbols))
            self.assertEqual(trifid.decrypt_trits(raw, cube, 5, symbols=symbols), plain)

    def test_bad_parameters(self) -> None:
        """Malformed cubes, periods and streams fail explicitly."""
        for value in ("A" * 27, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "ABCDEFGHIJKLMNOPQRSTUVWXYZ1"):
            with self.assertRaises(ValueError):
                trifid.TrifidCube(value)
        for filler in ("", "++", "A", "1", " "):
            with self.assertRaises(ValueError):
                trifid.trifid_cube(filler=filler)
        with self.assertRaises(ValueError):
            trifid.encrypt("ABC", period=-1)
        for stream in ("01", "012X00"):
            with self.assertRaises(ValueError):
                trifid.trits_to_letters(stream)
        with self.assertRaises(ValueError):
            trifid.decrypt_trits("012", symbols="001")
        for options in ({"iterations": 0}, {"restarts": 0}, {"time_budget": 0}):
            with self.assertRaises(ValueError):
                trifid.solve("ABC" * 30, **options)


class TestTrifidSearch(unittest.TestCase):
    """Fixed seeds recover long English and do not certify random trits."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build one 600-letter fixture using the toolkit's local prose."""
        cls.plain = letters_only(load_corpus_text())[1000:1600]
        cls.cube = trifid.trifid_cube("TEMPEST")
        cls.cipher = trifid.encrypt(cls.plain, cls.cube, 5)

    def test_letter_ciphertext_recovery(self) -> None:
        """Search all default periods without being supplied the keyword."""
        started = time.monotonic()
        found = trifid.solve(self.cipher, seed=42, time_budget=58.0)
        elapsed = time.monotonic() - started
        best = found.best()
        self.assertIsNotNone(best)
        self.assertEqual(best.plaintext, self.plain)
        self.assertEqual(best.diagnostics["period"], 5)
        self.assertEqual(set(best.diagnostics["periods_tested"]), {*range(3, 21), 0})
        self.assertLess(elapsed, 62.0)
        recovered = trifid.TrifidCube(best.diagnostics["cube"])
        self.assertEqual(trifid.encrypt(best.plaintext, recovered, 5), self.cipher)

    def test_raw_ciphertext_recovery(self) -> None:
        """The same message written in marks recovers through all six maps."""
        stream = trifid.letters_to_trits(self.cipher, self.cube)
        raw = stream.translate(str.maketrans("012", "|/\\"))
        started = time.monotonic()
        best = trifid.solve(raw, seed=42, time_budget=58.0).best()
        self.assertIsNotNone(best)
        self.assertEqual(best.plaintext, self.plain)
        self.assertEqual(best.diagnostics["symbol_mappings_tested"], 6)
        self.assertEqual(best.diagnostics["agreements"], 1)
        self.assertLess(time.monotonic() - started, 62.0)
        recovered = trifid.TrifidCube(best.diagnostics["cube"])
        self.assertEqual(trifid.decrypt_trits(
            raw, recovered, 5, symbols=best.diagnostics["symbols"]), self.plain)

    def test_random_trits_are_not_strong(self) -> None:
        """A long uniformly random ternary stream is a negative control."""
        rng = random.Random(912)
        raw = "".join(rng.choice("012") for _ in range(1800))
        found = trifid.solve(raw, seed=42, time_budget=20.0)
        self.assertTrue(found.ranked())
        self.assertTrue(all(candidate.confidence() != "strong" for candidate in found))

    def test_seed_reproducibility(self) -> None:
        """An unbudgeted fixed amount of work has deterministic results."""
        options = {"raw": True, "period": 5, "seed": 6, "restarts": 1,
                   "iterations": 80, "time_budget": None}
        stream = trifid.letters_to_trits(self.cipher, self.cube)
        first = trifid.solve(stream, **options).best()
        second = trifid.solve(stream, **options).best()
        self.assertEqual((first.key, first.score), (second.key, second.score))

    def test_budget_is_honoured(self) -> None:
        """A short clock budget terminates inside the screening pass."""
        started = time.monotonic()
        found = trifid.solve(self.cipher, seed=1, time_budget=0.02)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(found.ranked())
        self.assertTrue(found.best().diagnostics["time_budget_hit"])


class TestTrifidRouting(unittest.TestCase):
    """Recognition preserves symbol streams and leaves other families alone."""

    def test_recognition(self) -> None:
        """Complete triples or balanced symbol counts route to Trifid."""
        for stream in ("012", "012" * 40, "|/\\" * 40, "ABC" * 40,
                       "0" * 41 + "1" * 40 + "2" * 40,
                       "0" * 118 + "12"):
            self.assertTrue(trifid.recognise(stream))
            self.assertTrue(trifid.recognise(normalize(stream)))
        for stream in ("01" * 60, "12345" * 40, "ADFGVX" * 40,
                       "!@#$" * 40, "0" * 120 + "12"):
            self.assertFalse(trifid.recognise(stream))

    def test_auto_routes_before_other_symbol_solvers(self) -> None:
        """No ternary digits or marks are passed to the Polybius attack."""
        for stream in ("012" * 40, "|/\\" * 40, "ABC" * 40):
            with patch.object(trifid, "solve", return_value=CandidateSet()) as solve:
                with patch.object(auto.polybius, "solve_unknown_square") as other:
                    result = auto.auto_solve(stream, effort="fast", max_time=0.2, seed=9)
            solve.assert_called_once()
            other.assert_not_called()
            self.assertEqual(solve.call_args.kwargs["time_budget"], 0.2)
            self.assertEqual(result.stages[0].name, "Trifid (ternary stream)")

    def test_paste_routes_to_trifid(self) -> None:
        """The interactive symbol path also calls the new solver."""
        args = argparse.Namespace(max_time=0.2, seed=9)
        with patch.object(trifid, "solve", return_value=CandidateSet()) as solve:
            cli._solve_symbol_stream("|/\\" * 40, args)
        solve.assert_called_once()
        self.assertEqual(solve.call_args.kwargs["time_budget"], 0.2)

    def test_deep_includes_letter_ciphertext_attack(self) -> None:
        """Ordinary letter ciphertext gets a budgeted cube search at deep."""
        stages = auto.build_stages("deep", 3, 9)
        self.assertTrue(any(stage.run is trifid.solve for stage in stages))

    def test_other_mark_refusals_remain(self) -> None:
        """Unsupported mark counts keep the existing explanation."""
        for count in (2, 4, 7):
            text = "!@#$%^&"[:count] * 40
            self.assertIn("no solver in this toolkit reads", cli._render_letterless(text))
        self.assertIn("Polybius square", cli._render_letterless("!@#$%" * 40))
        self.assertIn("ADFGVX", cli._render_letterless("!@#$%^" * 40))
