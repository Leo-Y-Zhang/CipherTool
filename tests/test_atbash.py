"""Tests for the Atbash cipher.

The hand-computed anchor: Atbash reflects the alphabet, so A becomes Z, T
becomes G (T is the 20th letter, so it maps to the 20th from the end),
C becomes X, K becomes P, D becomes W, W becomes D and N becomes M. Hence
ATTACKATDAWN becomes ZGGZXPZGWZDM, and the classic HELLO becomes SVOOL.
"""

from __future__ import annotations

import unittest

from cipher_tool.atbash import ATBASH_ALPHABET, decrypt, encrypt, solve
from cipher_tool.caesar import encrypt as caesar_encrypt
from cipher_tool.normalize import ALPHABET, letters_only, normalize
from cipher_tool.scoring import DATA_DIR


def corpus_letters(count: int = 400) -> str:
    """The first *count* letters of our expository corpus file."""
    text = (DATA_DIR / "corpus_04_expository.txt").read_text(encoding="utf-8")
    letters = letters_only(text)
    if len(letters) < count:  # pragma: no cover - guards a corrupt install
        raise AssertionError(
            f"corpus file holds only {len(letters)} letters, needed {count}"
        )
    return letters[:count]


class TestEncryptDecrypt(unittest.TestCase):
    def test_known_pair_worked_by_hand(self) -> None:
        self.assertEqual(encrypt("ATTACKATDAWN"), "ZGGZXPZGWZDM")
        self.assertEqual(decrypt("ZGGZXPZGWZDM"), "ATTACKATDAWN")

    def test_second_known_pair(self) -> None:
        self.assertEqual(encrypt("HELLO"), "SVOOL")
        self.assertEqual(encrypt("SVOOL"), "HELLO")

    def test_the_alphabet_maps_to_its_reverse(self) -> None:
        self.assertEqual(encrypt(ALPHABET), ALPHABET[::-1])
        self.assertEqual(ATBASH_ALPHABET, "ZYXWVUTSRQPONMLKJIHGFEDCBA")

    def test_middle_pair_swaps(self) -> None:
        # M and N sit either side of the mirror, so they swap with each other.
        self.assertEqual(encrypt("MN"), "NM")

    def test_encryption_is_its_own_inverse(self) -> None:
        for text in (
            "ATTACKATDAWN",
            "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG",
            "AZ",
            corpus_letters(200),
        ):
            with self.subTest(text=text[:20]):
                self.assertEqual(encrypt(encrypt(text)), text)
                self.assertEqual(decrypt(encrypt(text)), text)
                # decrypt and encrypt are the same transformation.
                self.assertEqual(decrypt(text), encrypt(text))

    def test_input_layout_does_not_matter(self) -> None:
        expected = "ZGGZXPZGWZDM"
        for variant in (
            "attack at dawn",
            "Attack at dawn!",
            "ATTAC KATDA WN",
            "ATTAC KATDA\nWN\n",
            "  attack,\tat --- dawn.  ",
            "a1t2t3a4c5k6a7t8d9a0w-n",
        ):
            with self.subTest(variant=variant):
                self.assertEqual(encrypt(variant), expected)

    def test_empty_input_is_not_an_error(self) -> None:
        self.assertEqual(encrypt(""), "")
        self.assertEqual(decrypt(""), "")
        self.assertEqual(encrypt("!!! 123 ???"), "")
        self.assertEqual(len(solve("")), 0)
        self.assertEqual(len(solve("!!! 123 ???")), 0)


class TestInvalidInput(unittest.TestCase):
    """Atbash has no key, so the only invalid input is the text itself."""

    def test_non_string_text_raises_value_error(self) -> None:
        for bad in (None, 12345, ["ABC"], b"ABC"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as caught:
                    encrypt(bad)  # type: ignore[arg-type]
                self.assertTrue(str(caught.exception).strip())
                self.assertIn("string", str(caught.exception).lower())

    def test_solve_rejects_the_wrong_kind_of_source(self) -> None:
        with self.assertRaises(ValueError) as caught:
            solve(None)  # type: ignore[arg-type]
        self.assertTrue(str(caught.exception).strip())


class TestSolve(unittest.TestCase):
    def test_recovers_the_plaintext(self) -> None:
        plaintext = corpus_letters(400)
        best = solve(encrypt(plaintext)).best()
        assert best is not None
        self.assertEqual(best.plaintext, plaintext)
        self.assertEqual(best.confidence(), "strong")

    def test_returns_exactly_one_candidate_because_there_is_no_key(self) -> None:
        candidates = solve(encrypt(corpus_letters(400)))
        self.assertEqual(len(candidates), 1)
        best = candidates.best()
        assert best is not None
        self.assertEqual(best.diagnostics["keys_tested"], 1)
        self.assertTrue(best.diagnostics["fixed_key"])
        self.assertIn("no key", best.diagnostics["note"])
        # With one candidate there is no margin to report, and the toolkit
        # must say None rather than invent a comforting number.
        self.assertIsNone(candidates.score_gap())

    def test_diagnostics_carry_the_independent_signals(self) -> None:
        best = solve(encrypt(corpus_letters(400))).best()
        assert best is not None
        self.assertIn("chi_squared", best.diagnostics)
        self.assertIn("ciphertext_ic", best.diagnostics)
        self.assertIn("normalised_score", best.diagnostics)
        self.assertIn("word_coverage", best.diagnostics)
        # Atbash only relabels letters, so ciphertext IC equals plaintext IC.
        self.assertAlmostEqual(best.diagnostics["ciphertext_ic"], 0.0655, delta=0.02)

    def test_display_pours_the_plaintext_back_into_the_layout(self) -> None:
        best = solve("ZGGZX PZGWZ DM").best()
        assert best is not None
        self.assertEqual(best.plaintext, "ATTACKATDAWN")
        self.assertEqual(best.display, "ATTAC KATDA WN")

    def test_accepts_normalized_text_as_well_as_a_string(self) -> None:
        ciphertext = encrypt(corpus_letters(400))
        from_string = solve(ciphertext).best()
        from_object = solve(normalize(ciphertext)).best()
        assert from_string is not None and from_object is not None
        self.assertEqual(from_string.plaintext, from_object.plaintext)

    def test_unknown_options_are_recorded_not_swallowed(self) -> None:
        best = solve(encrypt(corpus_letters(400)), seed=7).best()
        assert best is not None
        self.assertEqual(best.diagnostics["options_ignored"], "seed")


class TestHonestFailure(unittest.TestCase):
    """What the solver does when the ciphertext is not Atbash."""

    def test_a_caesar_ciphertext_is_not_reported_as_solved(self) -> None:
        # Nothing can be searched here: if the text is not Atbash, the single
        # reading is wrong and the only honest thing to do is say the reading
        # does not look like English.
        plaintext = corpus_letters(400)
        best = solve(caesar_encrypt(plaintext, 3)).best()
        assert best is not None
        self.assertNotEqual(best.plaintext, plaintext)
        self.assertIn(best.confidence(), ("weak", "unlikely"))
        self.assertLess(best.diagnostics["normalised_score"], -1.8)
        self.assertLess(best.diagnostics["word_coverage"], 0.35)

    def test_polyalphabetic_text_gets_an_index_of_coincidence_warning(self) -> None:
        # Atbash cannot change IC. If the ciphertext IC is nowhere near
        # English then no monoalphabetic cipher explains the text, and the
        # solver should say so before the operator reads the plaintext.
        plaintext = corpus_letters(400)
        key = "LEMON"
        ciphertext = "".join(
            chr(65 + (ord(letter) - 65 + ord(key[index % len(key)]) - 65) % 26)
            for index, letter in enumerate(plaintext)
        )
        best = solve(ciphertext).best()
        assert best is not None
        self.assertIn("warning", best.diagnostics)
        self.assertIn("IC", best.diagnostics["warning"])
        self.assertIn(best.confidence(), ("weak", "unlikely"))

    def test_ordinary_monoalphabetic_text_gets_no_ic_warning(self) -> None:
        # The mirror image of the test above: the warning must be evidence,
        # not decoration that fires on everything.
        best = solve(caesar_encrypt(corpus_letters(400), 3)).best()
        assert best is not None
        self.assertNotIn("warning", best.diagnostics)


if __name__ == "__main__":
    unittest.main()
