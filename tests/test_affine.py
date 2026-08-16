"""Tests for the affine cipher E(x) = (a*x + b) mod 26 and its brute force.

The hand-computed anchor uses a = 5, b = 8. A is 0, so 5*0 + 8 = 8 = I.
T is 19, so 5*19 + 8 = 103, and 103 - 78 = 25 = Z. C is 2, so 5*2 + 8 = 18 = S.
K is 10, so 58 - 52 = 6 = G. D is 3, so 23 = X. W is 22, so 118 - 104 = 14 = O.
N is 13, so 73 - 52 = 21 = V. Hence ATTACKATDAWN becomes IZZISGIZXIOV.
"""

from __future__ import annotations

import unittest

from cipher_tool.affine import (
    VALID_MULTIPLIERS,
    cipher_alphabet,
    decrypt,
    describe_key,
    encrypt,
    extended_gcd,
    modular_inverse,
    solve,
    valid_multipliers,
)
from cipher_tool.atbash import encrypt as atbash_encrypt
from cipher_tool.caesar import encrypt as caesar_encrypt
from cipher_tool.normalize import ALPHABET, letters_only, normalize
from cipher_tool.scoring import DATA_DIR

#: A general monoalphabetic substitution that is not affine: keyboard order.
KEYBOARD_ALPHABET = "QWERTYUIOPASDFGHJKLZXCVBNM"


def corpus_letters(count: int = 400) -> str:
    """The first *count* letters of our expository corpus file."""
    text = (DATA_DIR / "corpus_04_expository.txt").read_text(encoding="utf-8")
    letters = letters_only(text)
    if len(letters) < count:  # pragma: no cover - guards a corrupt install
        raise AssertionError(
            f"corpus file holds only {len(letters)} letters, needed {count}"
        )
    return letters[:count]


class TestModularArithmetic(unittest.TestCase):
    def test_extended_gcd_identity_holds(self) -> None:
        for a in range(0, 40):
            for b in (1, 2, 13, 26, 27):
                with self.subTest(a=a, b=b):
                    divisor, s, t = extended_gcd(a, b)
                    self.assertEqual(a * s + b * t, divisor)

    def test_inverse_is_a_genuine_inverse(self) -> None:
        for a in VALID_MULTIPLIERS:
            with self.subTest(a=a):
                self.assertEqual((a * modular_inverse(a, 26)) % 26, 1)

    def test_inverse_agrees_with_the_language_built_in(self) -> None:
        """Cross-check our extended Euclid against Python's own pow().

        The toolkit computes the inverse itself on purpose, so the built-in is
        used here only as an independent oracle in the test.
        """
        for modulus in (26, 25, 97):
            for a in range(1, modulus):
                if extended_gcd(a, modulus)[0] != 1:
                    continue
                with self.subTest(a=a, modulus=modulus):
                    self.assertEqual(
                        modular_inverse(a, modulus), pow(a, -1, modulus)
                    )

    def test_inverse_of_a_non_coprime_value_raises(self) -> None:
        for a in (0, 2, 4, 13, 26):
            with self.subTest(a=a):
                with self.assertRaises(ValueError) as caught:
                    modular_inverse(a, 26)
                self.assertTrue(str(caught.exception).strip())
                self.assertIn("inverse", str(caught.exception).lower())

    def test_bad_modulus_raises(self) -> None:
        for modulus in (0, 1, -26, 26.0, "26"):
            with self.subTest(modulus=modulus):
                with self.assertRaises(ValueError) as caught:
                    modular_inverse(5, modulus)  # type: ignore[arg-type]
                self.assertTrue(str(caught.exception).strip())

    def test_the_twelve_usable_multipliers(self) -> None:
        expected = (1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25)
        self.assertEqual(valid_multipliers(), expected)
        self.assertEqual(len(valid_multipliers()), 12)
        # 26 = 2 * 13, so the unusable multipliers are the even ones and 13.
        unusable = set(range(26)) - set(expected)
        self.assertEqual(unusable, {0, 13} | {a for a in range(26) if a % 2 == 0})


class TestEncryptDecrypt(unittest.TestCase):
    def test_known_pair_worked_by_hand(self) -> None:
        self.assertEqual(encrypt("ATTACKATDAWN", 5, 8), "IZZISGIZXIOV")
        self.assertEqual(decrypt("IZZISGIZXIOV", 5, 8), "ATTACKATDAWN")

    def test_second_known_pair(self) -> None:
        # The textbook example: a=5, b=8 sends AFFINECIPHER to IHHWVCSWFRCP.
        self.assertEqual(encrypt("AFFINECIPHER", 5, 8), "IHHWVCSWFRCP")
        self.assertEqual(decrypt("IHHWVCSWFRCP", 5, 8), "AFFINECIPHER")

    def test_round_trip_over_the_whole_key_space(self) -> None:
        plaintext = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG"
        for a in VALID_MULTIPLIERS:
            for b in range(26):
                with self.subTest(a=a, b=b):
                    self.assertEqual(decrypt(encrypt(plaintext, a, b), a, b), plaintext)

    def test_every_key_gives_a_bijective_alphabet(self) -> None:
        # This is the whole reason gcd(a, 26) must be 1: no two plaintext
        # letters may share a ciphertext letter.
        for a in VALID_MULTIPLIERS:
            for b in (0, 1, 8, 25):
                with self.subTest(a=a, b=b):
                    self.assertEqual(len(set(cipher_alphabet(a, b))), 26)

    def test_a_equals_one_is_caesar(self) -> None:
        plaintext = corpus_letters(200)
        for b in (0, 3, 17, 25):
            with self.subTest(b=b):
                self.assertEqual(encrypt(plaintext, 1, b), caesar_encrypt(plaintext, b))
        self.assertIn("Caesar", describe_key(1, 3))

    def test_a_twentyfive_b_twentyfive_is_atbash(self) -> None:
        plaintext = corpus_letters(200)
        self.assertEqual(encrypt(plaintext, 25, 25), atbash_encrypt(plaintext))
        self.assertIn("Atbash", describe_key(25, 25))
        self.assertEqual(describe_key(5, 8), "")

    def test_keys_are_reduced_modulo_26(self) -> None:
        plaintext = "ATTACKATDAWN"
        self.assertEqual(encrypt(plaintext, 31, 34), encrypt(plaintext, 5, 8))
        self.assertEqual(encrypt(plaintext, -21, -18), encrypt(plaintext, 5, 8))

    def test_input_layout_does_not_matter(self) -> None:
        expected = "IZZISGIZXIOV"
        for variant in (
            "attack at dawn",
            "Attack at dawn!",
            "ATTAC KATDA WN",
            "ATTAC KATDA\nWN\n",
            "  attack,\tat --- dawn.  ",
            "a1t2t3a4c5k6a7t8d9a0w-n",
        ):
            with self.subTest(variant=variant):
                self.assertEqual(encrypt(variant, 5, 8), expected)

    def test_empty_input_is_not_an_error(self) -> None:
        self.assertEqual(encrypt("", 5, 8), "")
        self.assertEqual(decrypt("", 5, 8), "")
        self.assertEqual(encrypt("1234 !?", 5, 8), "")
        self.assertEqual(len(solve("")), 0)
        self.assertEqual(len(solve("1234 !?")), 0)


class TestInvalidKeys(unittest.TestCase):
    def test_multiplier_sharing_a_factor_with_26_raises(self) -> None:
        for a in (0, 2, 4, 6, 13, 26, 52):
            with self.subTest(a=a):
                with self.assertRaises(ValueError) as caught:
                    encrypt("ATTACK", a, 8)
                message = str(caught.exception)
                self.assertTrue(message.strip())
                # The message must tell the operator what to use instead.
                self.assertIn("gcd", message)
                self.assertIn("3, 5, 7", message)

    def test_decrypt_rejects_the_same_bad_multipliers(self) -> None:
        with self.assertRaises(ValueError) as caught:
            decrypt("IZZISGIZXIOV", 13, 8)
        self.assertTrue(str(caught.exception).strip())

    def test_non_integer_key_parts_raise(self) -> None:
        for a, b in ((5.0, 8), ("5", 8), (5, "8"), (5, None), (True, 8), (5, False)):
            with self.subTest(a=a, b=b):
                with self.assertRaises(ValueError) as caught:
                    encrypt("ATTACK", a, b)  # type: ignore[arg-type]
                self.assertTrue(str(caught.exception).strip())

    def test_non_string_text_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            encrypt(12345, 5, 8)  # type: ignore[arg-type]
        self.assertTrue(str(caught.exception).strip())

    def test_solve_rejects_the_wrong_kind_of_source(self) -> None:
        with self.assertRaises(ValueError) as caught:
            solve(12345)  # type: ignore[arg-type]
        self.assertTrue(str(caught.exception).strip())

    def test_negative_time_budget_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            solve("ATTACKATDAWN", time_budget=-1.0)
        self.assertTrue(str(caught.exception).strip())


class TestSolve(unittest.TestCase):
    """Solving 400 letters is the expensive part, so it is done once."""

    plaintext: str
    ciphertext: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.plaintext = corpus_letters(400)
        cls.ciphertext = encrypt(cls.plaintext, 7, 12)
        cls.candidates = solve(cls.ciphertext)

    def test_recovers_the_key_and_the_plaintext(self) -> None:
        best = self.candidates.best()
        assert best is not None
        self.assertEqual(best.plaintext, self.plaintext)
        self.assertEqual(best.key, "a=7 b=12")
        self.assertEqual(best.confidence(), "strong")
        self.assertEqual(best.diagnostics["a"], 7)
        self.assertEqual(best.diagnostics["b"], 12)
        self.assertEqual(best.diagnostics["a_inverse"], modular_inverse(7, 26))

    def test_the_whole_key_space_is_reported(self) -> None:
        self.assertEqual(len(self.candidates), 312)
        self.assertEqual(
            {candidate.key for candidate in self.candidates},
            {f"a={a} b={b}" for a in VALID_MULTIPLIERS for b in range(26)},
        )
        gap = self.candidates.score_gap()
        assert gap is not None
        self.assertGreater(gap, 1.0)

    def test_diagnostics_carry_both_measures(self) -> None:
        best = self.candidates.best()
        assert best is not None
        self.assertEqual(best.diagnostics["rank_by_chi2"], 1)
        self.assertEqual(best.diagnostics["rank_by_ngram"], 1)
        self.assertEqual(best.diagnostics["chi2_best_key"], "a=7 b=12")
        self.assertTrue(best.diagnostics["measures_agree"])
        self.assertEqual(best.diagnostics["keys_tested"], 312)
        self.assertEqual(best.diagnostics["keys_possible"], 312)
        self.assertNotIn("time_budget_hit", best.diagnostics)
        for field in ("rank_by_chi2", "rank_by_ngram"):
            with self.subTest(field=field):
                self.assertEqual(
                    sorted(c.diagnostics[field] for c in self.candidates),
                    list(range(1, 313)),
                )

    def test_special_cases_are_named_in_the_diagnostics(self) -> None:
        by_key = {c.key: c for c in self.candidates}
        self.assertIn("Caesar", by_key["a=1 b=3"].diagnostics["equivalent_to"])
        self.assertIn("Atbash", by_key["a=25 b=25"].diagnostics["equivalent_to"])
        self.assertNotIn("equivalent_to", by_key["a=7 b=12"].diagnostics)

    def test_display_pours_the_plaintext_back_into_the_layout(self) -> None:
        best = [c for c in solve("IZZIS GIZXI OV") if c.key == "a=5 b=8"][0]
        self.assertEqual(best.plaintext, "ATTACKATDAWN")
        self.assertEqual(best.display, "ATTAC KATDA WN")

    def test_accepts_normalized_text_as_well_as_a_string(self) -> None:
        best = solve(normalize(self.ciphertext)).best()
        assert best is not None
        self.assertEqual(best.plaintext, self.plaintext)

    def test_unknown_options_are_recorded_not_swallowed(self) -> None:
        best = solve("ATTACKATDAWN", keylength=4).best()
        assert best is not None
        self.assertEqual(best.diagnostics["options_ignored"], "keylength")


class TestHonestFailure(unittest.TestCase):
    """What the solver does when brute force cannot help."""

    def test_a_general_substitution_is_not_reported_as_solved(self) -> None:
        # The search is exhaustive, so failing here is not a search failure --
        # it means the cipher is not affine at all. The report must not
        # disguise the least-bad of 312 wrong answers as a solution.
        plaintext = corpus_letters(400)
        ciphertext = plaintext.translate(str.maketrans(ALPHABET, KEYBOARD_ALPHABET))
        best = solve(ciphertext).best()
        assert best is not None
        self.assertNotEqual(best.plaintext, plaintext)
        self.assertIn(best.confidence(), ("weak", "unlikely"))
        self.assertLess(best.diagnostics["normalised_score"], -1.8)
        self.assertLess(best.diagnostics["word_coverage"], 0.35)

    def test_an_exhausted_time_budget_is_admitted(self) -> None:
        # A budget of zero cuts the search after the first key. The tool must
        # report that it only tried one key rather than implying the key space
        # was covered.
        candidates = solve(encrypt(corpus_letters(300), 7, 12), time_budget=0.0)
        best = candidates.best()
        assert best is not None
        self.assertTrue(best.diagnostics["time_budget_hit"])
        self.assertEqual(best.diagnostics["keys_tested"], 1)
        self.assertEqual(best.diagnostics["keys_possible"], 312)
        self.assertEqual(len(candidates), 1)
        # And the one key it managed is the wrong one, honestly labelled.
        self.assertIn(best.confidence(), ("weak", "unlikely"))

    def test_a_generous_budget_covers_the_whole_key_space(self) -> None:
        candidates = solve(encrypt(corpus_letters(300), 7, 12), time_budget=120.0)
        best = candidates.best()
        assert best is not None
        self.assertEqual(len(candidates), 312)
        self.assertNotIn("time_budget_hit", best.diagnostics)
        self.assertEqual(best.key, "a=7 b=12")


if __name__ == "__main__":
    unittest.main()
