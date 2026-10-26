"""Four-square geometry, ciphertext-only recovery and pipeline recognition."""

from __future__ import annotations

import random
import time
import unittest

from cipher_tool import foursquare
from cipher_tool.auto import auto_solve, build_stages, order_stages
from cipher_tool.normalize import letters_only, normalize
from cipher_tool.scoring import load_corpus_text
from cipher_tool.statistics import analyse


class TestFourSquare(unittest.TestCase):
    """The rectangle rule uses two standard and two independent keyed squares."""

    def test_hand_computed_rectangles(self) -> None:
        # EXAMP / LBCDF / GHIKN / OQRST / UVWYZ is top right;
        # KEYWO / RDABC / FGHIL / MNPQS / TUVXZ is bottom left.
        self.assertEqual(foursquare.encrypt("AAAFATABZZ", "EXAMPLE", "KEYWORD"),
                         "EKERMMXKZZ")
        self.assertEqual(foursquare.decrypt("EKERMMXKZZ", "EXAMPLE", "KEYWORD"),
                         "AAAFATABZZ")

    def test_round_trip(self) -> None:
        for text in ("", "A", "HELLO", "BALLOON", "Jolly jazz!", "FINALX"):
            with self.subTest(text=text):
                expected = letters_only(text).replace("J", "I")
                expected += "X" if len(expected) % 2 else ""
                cipher = foursquare.encrypt(text, "EXAMPLE", "KEYWORD")
                self.assertEqual(foursquare.decrypt(cipher, "EXAMPLE", "KEYWORD"),
                                 expected)

    def test_complete_squares_and_grouped_ciphertext(self) -> None:
        one = foursquare.ALPHABET[::-1]
        two = foursquare.ALPHABET[7:] + foursquare.ALPHABET[:7]
        cipher = foursquare.encrypt("Meet me at noon", one, two)
        grouped = " ".join(cipher[i:i + 5].lower()
                           for i in range(0, len(cipher), 5))
        self.assertEqual(foursquare.decrypt(grouped, one, two), "MEETMEATNOON")

    def test_key_preparation(self) -> None:
        square = foursquare.keyed_square("Jiggle!")
        self.assertTrue(square.startswith("IGLE"))
        self.assertEqual(set(square), set(foursquare.ALPHABET))
        self.assertEqual(len(square), 25)
        self.assertEqual(foursquare.keyed_square(), foursquare.ALPHABET)

    def test_invalid_ciphertext(self) -> None:
        for text in ("ABC", "AJ"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                foursquare.decrypt(text, "EXAMPLE", "KEYWORD")


class TestFourSquareSolver(unittest.TestCase):
    """A real annealing recovery and a control with no English to recover."""

    def test_recovers_500_letters_with_seed(self) -> None:
        plain = letters_only(load_corpus_text())[:500].replace("J", "I")
        cipher = foursquare.encrypt(plain, "EXAMPLE", "KEYWORD")
        started = time.monotonic()
        best = foursquare.solve(cipher, restarts=4, seed=1).best()
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 60.0)
        self.assertEqual(best.plaintext, plain)
        self.assertGreater(best.diagnostics["word_coverage"], 0.7)
        self.assertEqual(best.confidence(), "promising")
        one, two = best.diagnostics["square_one"], best.diagnostics["square_two"]
        self.assertEqual(set(one), set(foursquare.ALPHABET))
        self.assertEqual(set(two), set(foursquare.ALPHABET))
        self.assertEqual(foursquare.decrypt(cipher, one, two), best.plaintext)
        self.assertEqual(foursquare.encrypt(best.plaintext, one, two), cipher)
        self.assertIn(one, best.key)
        self.assertIn(two, best.key)

    def test_random_letters_are_not_strong(self) -> None:
        # Above the short-text cap: score and coverage must reject this too.
        rng = random.Random(37)
        noise = "".join(rng.choice(foursquare.ALPHABET) for _ in range(1000))
        best = foursquare.solve(noise, restarts=2, iterations=6000, seed=7).best()
        self.assertNotEqual(best.confidence(), "strong")
        self.assertLess(best.diagnostics["word_coverage"], 0.7)

    def test_seed_is_reproducible(self) -> None:
        cipher = foursquare.encrypt(letters_only(load_corpus_text())[:100],
                                    "MONARCHY", "ZEBRAS")
        first = foursquare.solve(cipher, seed=3, restarts=2, iterations=100).best()
        second = foursquare.solve(normalize(cipher), seed=3, restarts=2,
                                  iterations=100).best()
        self.assertEqual(first.key, second.key)
        self.assertEqual(first.plaintext, second.plaintext)

    def test_budget_is_reported(self) -> None:
        cipher = foursquare.encrypt("HELLOWORLD" * 10, "EXAMPLE", "KEYWORD")
        best = foursquare.solve(cipher, time_budget=0.0, seed=1).best()
        self.assertTrue(best.diagnostics["time_budget_hit"])
        self.assertEqual(best.diagnostics["moves_evaluated"], 0)
        self.assertNotEqual(best.confidence(), "strong")

    def test_unsupported_inputs_are_silent_in_the_pipeline(self) -> None:
        for text in ("", "ABC", "AJ" * 40):
            with self.subTest(text=text):
                self.assertEqual(len(foursquare.solve(text)), 0)
        self.assertEqual(len(foursquare.solve("AB" * 40, require_shape=True)), 0)

    def test_invalid_search_options(self) -> None:
        for options in ({"restarts": 0}, {"iterations": 0}, {"top": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                foursquare.solve("AB" * 40, **options)


class TestFourSquareRecognition(unittest.TestCase):
    """Doubles within pairs distinguish this shape from standard Playfair."""

    def test_four_square_shape(self) -> None:
        text = foursquare.ALPHABET * 2 + "AA"
        self.assertTrue(foursquare.recognises(text))
        families = [h.family for h in analyse(text).hypotheses]
        self.assertIn("Four-square (digraphic)", families)
        self.assertNotIn("Playfair", families)

    def test_every_shape_condition_matters(self) -> None:
        base = foursquare.ALPHABET * 2
        for text in (base, base + "ABBA", base + "AAJ", base + "AJ",
                     base.replace("Z", "A") + "AA"):
            with self.subTest(text=text):
                self.assertFalse(foursquare.recognises(text))
                self.assertNotIn("Four-square (digraphic)",
                                 [h.family for h in analyse(text).hypotheses])

    def test_pipeline_tries_recognised_ciphertext(self) -> None:
        stages = build_stages("normal", 5, 11)
        stage = next(stage for stage in stages if stage.name == "Four-square")
        self.assertTrue(stage.options["require_shape"])
        self.assertEqual(stage.options["seed"], 11)
        self.assertIs(stage.run, foursquare.solve)
        stage.options.update(restarts=1, iterations=10)
        text = foursquare.ALPHABET * 2 + "AA"
        ordered = order_stages(stages, analyse(text))
        self.assertLess(ordered.index(stage), stages.index(stage))
        result = auto_solve(text, effort="normal", stages=[stage])
        report = next(report for report in result.stages
                      if report.name == "Four-square")
        self.assertTrue(report.ran)
        self.assertGreater(report.candidates, 0)
        self.assertEqual(result.candidates.best().method, "Four-square")


if __name__ == "__main__":
    unittest.main()
