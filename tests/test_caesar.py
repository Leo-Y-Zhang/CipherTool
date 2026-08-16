"""Tests for the Caesar shift cipher and its exhaustive attack.

The hand-computed triple at the top is the anchor: A shifted three places is
D, T is W, C is F, K is N, D is G, W is Z, N is Q, so ATTACKATDAWN becomes
DWWDFNDWGDZQ. Everything else in the module is checked against behaviour that
follows from that one worked example.
"""

from __future__ import annotations

import unittest

from cipher_tool.caesar import (
    all_shifts,
    best_shift_by_chi_squared,
    chi_squared_by_shift,
    decrypt,
    encrypt,
    shifted_alphabet,
    solve,
)
from cipher_tool.normalize import ALPHABET, letters_only, normalize
from cipher_tool.scoring import DATA_DIR
from cipher_tool.statistics import chi_squared_english

#: A general monoalphabetic substitution that is NOT a shift: the letters of a
#: keyboard, top row first. Used to check that the solver admits defeat rather
#: than confidently returning the least-bad of 26 wrong answers.
KEYBOARD_ALPHABET = "QWERTYUIOPASDFGHJKLZXCVBNM"


def corpus_letters(count: int = 400) -> str:
    """The first *count* letters of our expository corpus file.

    The statistical parts of the attack need a few hundred letters before
    letter frequencies mean anything, so tests that exercise them use real
    prose rather than a toy phrase.
    """
    text = (DATA_DIR / "corpus_04_expository.txt").read_text(encoding="utf-8")
    letters = letters_only(text)
    if len(letters) < count:  # pragma: no cover - guards a corrupt install
        raise AssertionError(
            f"corpus file holds only {len(letters)} letters, needed {count}"
        )
    return letters[:count]


class TestEncryptDecrypt(unittest.TestCase):
    def test_known_pair_worked_by_hand(self) -> None:
        self.assertEqual(encrypt("ATTACKATDAWN", 3), "DWWDFNDWGDZQ")
        self.assertEqual(decrypt("DWWDFNDWGDZQ", 3), "ATTACKATDAWN")

    def test_second_known_pair_wraps_round_z(self) -> None:
        # Shift 25: A->Z, T->S, C->B, K->J. This is the wrap-around case.
        self.assertEqual(encrypt("ATTACK", 25), "ZSSZBJ")
        self.assertEqual(decrypt("ZSSZBJ", 25), "ATTACK")

    def test_shift_zero_changes_nothing(self) -> None:
        self.assertEqual(encrypt("ATTACKATDAWN", 0), "ATTACKATDAWN")

    def test_round_trip_for_every_shift(self) -> None:
        plaintext = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"
        for shift in range(26):
            with self.subTest(shift=shift):
                self.assertEqual(decrypt(encrypt(plaintext, shift), shift), plaintext)

    def test_negative_and_large_shifts_reduce_modulo_26(self) -> None:
        plaintext = "MEETMEATMIDNIGHT"
        self.assertEqual(encrypt(plaintext, -3), encrypt(plaintext, 23))
        self.assertEqual(encrypt(plaintext, 29), encrypt(plaintext, 3))
        self.assertEqual(decrypt(encrypt(plaintext, -100), -100), plaintext)

    def test_shifted_alphabet(self) -> None:
        self.assertEqual(shifted_alphabet(0), ALPHABET)
        self.assertEqual(shifted_alphabet(3), "DEFGHIJKLMNOPQRSTUVWXYZABC")

    def test_input_layout_does_not_matter(self) -> None:
        expected = encrypt("ATTACKATDAWN", 3)
        for variant in (
            "attack at dawn",
            "Attack at dawn!",
            "ATTAC KATDA WN",
            "ATTAC KATDA\nWN\n",
            "  attack,\tat --- dawn.  ",
            "a1t2t3a4c5k6a7t8d9a0w-n",
        ):
            with self.subTest(variant=variant):
                self.assertEqual(encrypt(variant, 3), expected)
                self.assertEqual(decrypt(expected, 3), "ATTACKATDAWN")

    def test_empty_input_is_not_an_error(self) -> None:
        self.assertEqual(encrypt("", 5), "")
        self.assertEqual(decrypt("", 5), "")
        self.assertEqual(encrypt("12345 !?", 5), "")
        self.assertEqual(all_shifts(""), [(shift, "") for shift in range(26)])
        self.assertEqual(len(solve("")), 0)
        self.assertEqual(len(solve("12345 !?")), 0)


class TestInvalidInput(unittest.TestCase):
    def test_non_integer_shift_raises_value_error(self) -> None:
        for bad in ("3", 3.0, None, [3], 3 + 0j):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as caught:
                    encrypt("ATTACK", bad)  # type: ignore[arg-type]
                self.assertTrue(str(caught.exception).strip())
                self.assertIn("shift", str(caught.exception).lower())

    def test_boolean_shift_is_rejected(self) -> None:
        # bool is a subclass of int, so this would silently mean "shift by 1".
        with self.assertRaises(ValueError) as caught:
            encrypt("ATTACK", True)  # type: ignore[arg-type]
        self.assertTrue(str(caught.exception).strip())

    def test_non_string_text_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as caught:
            encrypt(12345, 3)  # type: ignore[arg-type]
        self.assertTrue(str(caught.exception).strip())

    def test_solve_rejects_the_wrong_kind_of_source(self) -> None:
        with self.assertRaises(ValueError) as caught:
            solve(12345)  # type: ignore[arg-type]
        self.assertTrue(str(caught.exception).strip())


class TestAllShifts(unittest.TestCase):
    def test_returns_all_twenty_six_readings(self) -> None:
        ciphertext = encrypt("ATTACKATDAWN", 3)
        readings = all_shifts(ciphertext)
        self.assertEqual(len(readings), 26)
        self.assertEqual([shift for shift, _ in readings], list(range(26)))
        self.assertEqual(len({text for _, text in readings}), 26)

    def test_entry_s_is_the_decryption_by_s(self) -> None:
        ciphertext = encrypt("MEETMEATMIDNIGHT", 11)
        for shift, plaintext in all_shifts(ciphertext):
            with self.subTest(shift=shift):
                self.assertEqual(plaintext, decrypt(ciphertext, shift))
        self.assertEqual(dict(all_shifts(ciphertext))[11], "MEETMEATMIDNIGHT")


class TestChiSquared(unittest.TestCase):
    def test_rotation_shortcut_agrees_with_the_reference_measure(self) -> None:
        """The 26 rotated histograms must equal 26 real decryptions.

        chi_squared_by_shift rotates one count vector instead of deciphering
        the text 26 times. That is only legitimate if it gives exactly the
        same numbers as measuring each decryption directly.
        """
        ciphertext = encrypt(corpus_letters(400), 7)
        fast = dict(chi_squared_by_shift(ciphertext))
        for shift in range(26):
            with self.subTest(shift=shift):
                self.assertAlmostEqual(
                    fast[shift], chi_squared_english(decrypt(ciphertext, shift)),
                    places=12,
                )

    def test_empty_text_is_undefined_not_a_perfect_fit(self) -> None:
        values = chi_squared_by_shift("")
        self.assertEqual(len(values), 26)
        self.assertTrue(all(value == float("inf") for _, value in values))

    def test_frequency_fitting_finds_the_shift(self) -> None:
        plaintext = corpus_letters(400)
        for shift in (0, 1, 7, 13, 25):
            with self.subTest(shift=shift):
                self.assertEqual(
                    best_shift_by_chi_squared(encrypt(plaintext, shift)), shift
                )

    def test_frequency_fitting_ignores_layout(self) -> None:
        ciphertext = encrypt(corpus_letters(400), 9)
        spaced = " ".join(ciphertext[i:i + 5] for i in range(0, len(ciphertext), 5))
        self.assertEqual(best_shift_by_chi_squared(spaced.lower()), 9)


class TestSolve(unittest.TestCase):
    def test_recovers_the_plaintext_of_a_known_key(self) -> None:
        plaintext = corpus_letters(400)
        candidates = solve(encrypt(plaintext, 7))
        best = candidates.best()
        assert best is not None
        self.assertEqual(best.plaintext, plaintext)
        self.assertEqual(best.key, "shift=7")
        self.assertEqual(best.confidence(), "strong")

    def test_reports_every_shift_not_just_the_winner(self) -> None:
        candidates = solve(encrypt(corpus_letters(400), 7))
        self.assertEqual(len(candidates), 26)
        self.assertEqual(
            {candidate.key for candidate in candidates},
            {f"shift={shift}" for shift in range(26)},
        )
        # The winner should stand clearly apart from the field.
        gap = candidates.score_gap()
        assert gap is not None
        self.assertGreater(gap, 1.0)

    def test_diagnostics_carry_both_measures(self) -> None:
        candidates = solve(encrypt(corpus_letters(400), 7))
        best = candidates.best()
        assert best is not None
        diagnostics = best.diagnostics
        self.assertEqual(diagnostics["rank_by_chi2"], 1)
        self.assertEqual(diagnostics["rank_by_ngram"], 1)
        self.assertEqual(diagnostics["chi2_best_shift"], 7)
        self.assertEqual(diagnostics["ngram_best_shift"], 7)
        self.assertTrue(diagnostics["measures_agree"])
        self.assertEqual(diagnostics["shifts_tested"], 26)
        self.assertIn("word_coverage", diagnostics)
        self.assertIn("normalised_score", diagnostics)
        # Ranks must be a genuine permutation of 1..26 on both measures.
        for field in ("rank_by_chi2", "rank_by_ngram"):
            with self.subTest(field=field):
                self.assertEqual(
                    sorted(c.diagnostics[field] for c in candidates),
                    list(range(1, 27)),
                )

    def test_display_pours_the_plaintext_back_into_the_layout(self) -> None:
        candidates = solve("DWWDF NDWGD ZQ")
        chosen = [c for c in candidates if c.key == "shift=3"][0]
        self.assertEqual(chosen.plaintext, "ATTACKATDAWN")
        self.assertEqual(chosen.display, "ATTAC KATDA WN")

    def test_accepts_normalized_text_as_well_as_a_string(self) -> None:
        ciphertext = encrypt(corpus_letters(400), 4)
        from_string = solve(ciphertext).best()
        from_object = solve(normalize(ciphertext)).best()
        assert from_string is not None and from_object is not None
        self.assertEqual(from_string.plaintext, from_object.plaintext)

    def test_unknown_options_are_recorded_not_swallowed(self) -> None:
        best = solve(encrypt(corpus_letters(400), 7), keylength=4).best()
        assert best is not None
        self.assertEqual(best.diagnostics["options_ignored"], "keylength")


class TestHonestFailure(unittest.TestCase):
    """What the solver does when the ciphertext is not a Caesar shift."""

    def test_a_general_substitution_is_not_reported_as_solved(self) -> None:
        # A keyboard-order substitution alphabet is monoalphabetic but is not
        # a rotation, so no shift can undo it. Brute force is exhaustive here,
        # which means "the best of 26" is still wrong -- and the report must
        # say so rather than dressing up the least-bad answer.
        plaintext = corpus_letters(400)
        ciphertext = plaintext.translate(str.maketrans(ALPHABET, KEYBOARD_ALPHABET))
        best = solve(ciphertext).best()
        assert best is not None
        self.assertNotEqual(best.plaintext, plaintext)
        self.assertIn(best.confidence(), ("weak", "unlikely"))
        self.assertLess(best.diagnostics["normalised_score"], -1.8)
        self.assertLess(best.diagnostics["word_coverage"], 0.35)

    def test_wrong_shifts_are_labelled_unlikely(self) -> None:
        candidates = solve(encrypt(corpus_letters(400), 7))
        for candidate in candidates:
            if candidate.key == "shift=7":
                continue
            with self.subTest(key=candidate.key):
                self.assertIn(candidate.confidence(), ("weak", "unlikely"))

    def test_polyalphabetic_text_makes_the_two_measures_disagree(self) -> None:
        # A Vigenere ciphertext has no correct shift at all. The two
        # independent measures then have no reason to agree, and the
        # disagreement is the warning we want the operator to see.
        plaintext = corpus_letters(400)
        key = "LEMON"
        ciphertext = "".join(
            chr(65 + (ord(letter) - 65 + ord(key[index % len(key)]) - 65) % 26)
            for index, letter in enumerate(plaintext)
        )
        best = solve(ciphertext).best()
        assert best is not None
        self.assertIn(best.confidence(), ("weak", "unlikely"))
        self.assertFalse(best.diagnostics["measures_agree"])


if __name__ == "__main__":
    unittest.main()
