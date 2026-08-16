"""Tests for the Hill cipher module.

The arithmetic in the "known answer" tests below was worked out by hand
before the code was written; the workings are in the comments so that a
reader can check them without trusting the implementation. Where a test
depends on statistics rather than on exact arithmetic it uses real English
prose from the project corpus, never a short toy phrase.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from cipher_tool.candidates import CandidateSet
from cipher_tool.normalize import letters_only, normalize
from cipher_tool.hill import (
    NotInvertibleError,
    adjugate,
    brute_force_feasible,
    decrypt,
    describe_matrix,
    determinant,
    encrypt,
    extended_gcd,
    identity_matrix,
    invertible_matrix_count,
    is_invertible,
    key_from_string,
    known_plaintext_attack,
    matrix_inverse,
    matrix_multiply,
    matrix_to_string,
    matrix_vector,
    modular_inverse,
    search_space_size,
    solve,
    transpose,
    validate_matrix,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

#: HILL -> H=7, I=8, L=11, L=11 laid out row by row.
HILL = [[7, 8], [11, 11]]

#: GYBNQKURP -> the standard 3 x 3 worked example; determinant 441 = 25 mod 26.
GYBNQKURP = [[6, 24, 1], [13, 16, 10], [20, 17, 15]]

#: A 4 x 4 key. Most English phrases of sixteen letters turn out to have an
#: even determinant and so cannot be used at all -- a good illustration of
#: how restrictive the gcd(det, 26) = 1 rule is. This one has determinant 19.
KEY_4X4 = key_from_string("ATTACKATDAWNXYZQ")

#: An invertible key that does NOT spell a plausible English keyword, used to
#: prove the exhaustive search really is exhaustive rather than merely lucky
#: about its ordering. det = 3*7 - 2*5 = 11, and gcd(11, 26) = 1.
NON_KEYWORD = [[3, 2], [5, 7]]

DATA_DIR = Path(__file__).resolve().parents[1] / "src" / "cipher_tool" / "data"


def corpus_letters(count: int) -> str:
    """The first *count* letters of the expository corpus file."""
    text = (DATA_DIR / "corpus_04_expository.txt").read_text(encoding="utf-8")
    letters = letters_only(text)
    if len(letters) < count:  # pragma: no cover - the corpus is far longer
        raise AssertionError(f"corpus has only {len(letters)} letters")
    return letters[:count]


# ---------------------------------------------------------------------------
# Modular arithmetic
# ---------------------------------------------------------------------------


class TestModularArithmetic(unittest.TestCase):
    def test_extended_gcd_known_triple(self) -> None:
        # 240 * (-9) + 46 * 47 = -2160 + 2162 = 2 = gcd(240, 46).
        self.assertEqual(extended_gcd(240, 46), (2, -9, 47))
        gcd_value, x, y = extended_gcd(240, 46)
        self.assertEqual(240 * x + 46 * y, gcd_value)

    def test_modular_inverse_known_values(self) -> None:
        # 15 * 7 = 105 = 4*26 + 1.
        self.assertEqual(modular_inverse(15, 26), 7)
        # 3 * 9 = 27 = 26 + 1.
        self.assertEqual(modular_inverse(3, 26), 9)
        # 25 * 25 = 625 = 24*26 + 1, so 25 is its own inverse.
        self.assertEqual(modular_inverse(25, 26), 25)

    def test_modular_inverse_covers_every_unit(self) -> None:
        units = [value for value in range(26) if value % 2 and value % 13]
        self.assertEqual(len(units), 12)
        for value in units:
            self.assertEqual((value * modular_inverse(value, 26)) % 26, 1)

    def test_modular_inverse_rejects_even_and_thirteen(self) -> None:
        for value in (0, 2, 13, 24):
            with self.assertRaises(ValueError) as caught:
                modular_inverse(value, 26)
            self.assertTrue(str(caught.exception).strip())
            self.assertIn("no inverse", str(caught.exception))


# ---------------------------------------------------------------------------
# Matrix arithmetic
# ---------------------------------------------------------------------------


class TestMatrixArithmetic(unittest.TestCase):
    def test_matrix_multiply_by_hand(self) -> None:
        # [[1,2],[3,4]] [[5,6],[7,8]] = [[19,22],[43,50]] over the integers;
        # 43 - 26 = 17 and 50 - 26 = 24.
        product = matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        self.assertEqual(product, [[19, 22], [17, 24]])

    def test_matrix_multiply_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            matrix_multiply([[1, 2, 3]], [[1, 2], [3, 4]])
        self.assertIn("shape mismatch", str(caught.exception))

    def test_matrix_vector_by_hand(self) -> None:
        # HILL applied to AT = (0, 19):
        #   7*0 + 8*19 = 152 = 5*26 + 22 -> 22 (W)
        #  11*0 + 11*19 = 209 = 8*26 + 1 -> 1  (B)
        self.assertEqual(matrix_vector(HILL, [0, 19]), [22, 1])

    def test_matrix_vector_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            matrix_vector(HILL, [1, 2, 3])
        self.assertIn("shape mismatch", str(caught.exception))

    def test_determinant_2x2_by_hand(self) -> None:
        # 7*11 - 8*11 = 77 - 88 = -11, and -11 + 26 = 15.
        self.assertEqual(determinant(HILL, None), -11)
        self.assertEqual(determinant(HILL), 15)

    def test_determinant_3x3_by_cofactor_expansion(self) -> None:
        #  6*(16*15 - 10*17) - 24*(13*15 - 10*20) + 1*(13*17 - 16*20)
        #  = 6*70 - 24*(-5) + (-99) = 420 + 120 - 99 = 441
        #  441 = 16*26 + 25.
        self.assertEqual(determinant(GYBNQKURP, None), 441)
        self.assertEqual(determinant(GYBNQKURP), 25)

    def test_determinant_4x4_triangular(self) -> None:
        # An upper triangular determinant is the product of the diagonal:
        # 2 * 3 * 4 * 5 = 120, and 120 - 4*26 = 16.
        triangular = [[2, 1, 1, 1], [0, 3, 1, 1], [0, 0, 4, 1], [0, 0, 0, 5]]
        self.assertEqual(determinant(triangular, None), 120)
        self.assertEqual(determinant(triangular), 16)

    def test_determinant_of_identity_is_one(self) -> None:
        for size in (1, 2, 3, 4):
            self.assertEqual(determinant(identity_matrix(size)), 1)

    def test_determinant_rejects_non_square(self) -> None:
        with self.assertRaises(ValueError) as caught:
            determinant([[1, 2, 3], [4, 5, 6]])
        self.assertIn("square", str(caught.exception))

    def test_transpose(self) -> None:
        self.assertEqual(transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]])

    def test_matrix_inverse_by_hand(self) -> None:
        # det(HILL) = 15, and 15^-1 = 7 mod 26.
        # adj = [[11, -8], [-11, 7]] = [[11, 18], [15, 7]] mod 26.
        # 7 * adj = [[77, 126], [105, 49]] = [[25, 22], [1, 23]] mod 26.
        self.assertEqual(matrix_inverse(HILL), [[25, 22], [1, 23]])

    def test_inverse_times_matrix_is_the_identity(self) -> None:
        for matrix in (HILL, NON_KEYWORD, GYBNQKURP):
            size = len(matrix)
            inverse = matrix_inverse(matrix)
            self.assertEqual(matrix_multiply(inverse, matrix), identity_matrix(size))
            self.assertEqual(matrix_multiply(matrix, inverse), identity_matrix(size))

    def test_adjugate_identity_holds_even_when_singular(self) -> None:
        # adj(M) M = det(M) I is a ring identity, true whether or not the
        # determinant happens to be invertible. [[1,2],[3,4]] has det 24.
        singular = [[1, 2], [3, 4]]
        det_value = determinant(singular)
        self.assertEqual(det_value, 24)
        product = matrix_multiply(adjugate(singular), singular)
        self.assertEqual(product, [[det_value, 0], [0, det_value]])

    def test_not_invertible_error_names_the_determinant(self) -> None:
        with self.assertRaises(NotInvertibleError) as caught:
            matrix_inverse([[1, 2], [3, 4]])
        error = caught.exception
        self.assertEqual(error.determinant, 24)
        self.assertEqual(error.common_factor, 2)
        message = str(error)
        self.assertTrue(message.strip())
        self.assertIn("24", message)
        self.assertIn("26 = 2 x 13", message)
        # A ValueError subclass, so broad handlers still catch it.
        self.assertIsInstance(error, ValueError)

    def test_determinant_divisible_by_thirteen_is_rejected(self) -> None:
        with self.assertRaises(NotInvertibleError) as caught:
            matrix_inverse([[1, 0], [0, 13]])
        self.assertEqual(caught.exception.determinant, 13)
        self.assertEqual(caught.exception.common_factor, 13)

    def test_is_invertible_matches_the_gcd_rule(self) -> None:
        self.assertTrue(is_invertible(HILL))
        self.assertTrue(is_invertible(GYBNQKURP))
        self.assertFalse(is_invertible([[1, 2], [3, 4]]))
        self.assertFalse(is_invertible([[1, 0], [0, 13]]))

    def test_validate_matrix_rejects_bad_shapes_and_types(self) -> None:
        for bad in ([[1, 2, 3], [4, 5, 6]], [], [[1, "a"], [2, 3]], "HILL"):
            with self.assertRaises(ValueError) as caught:
                validate_matrix(bad)  # type: ignore[arg-type]
            self.assertTrue(str(caught.exception).strip())

    def test_validate_matrix_reduces_entries(self) -> None:
        self.assertEqual(validate_matrix([[27, -1], [52, 30]]), [[1, 25], [0, 4]])


class TestKeySpaceCounts(unittest.TestCase):
    def test_invertible_2x2_count(self) -> None:
        # |GL(2, Z/26)| = |GL(2, Z/2)| * |GL(2, Z/13)|
        #               = (4-1)(4-2) * (169-1)(169-13)
        #               = 6 * 26208 = 157248.
        self.assertEqual(invertible_matrix_count(2), 157248)
        self.assertEqual(search_space_size(2), 456976)

    def test_counted_formula_agrees_with_brute_enumeration(self) -> None:
        counted = sum(
            1
            for a in range(26)
            for b in range(26)
            for c in range(26)
            for d in range(26)
            if is_invertible([[a, b], [c, d]])
        )
        self.assertEqual(counted, invertible_matrix_count(2))

    def test_brute_force_only_claimed_for_2x2(self) -> None:
        self.assertTrue(brute_force_feasible(2))
        self.assertFalse(brute_force_feasible(3))
        self.assertFalse(brute_force_feasible(4))
        self.assertEqual(search_space_size(3), 26**9)


# ---------------------------------------------------------------------------
# Keys as text
# ---------------------------------------------------------------------------


class TestKeyFromString(unittest.TestCase):
    def test_four_letter_key(self) -> None:
        self.assertEqual(key_from_string("HILL"), HILL)

    def test_nine_letter_key(self) -> None:
        self.assertEqual(key_from_string("GYBNQKURP"), GYBNQKURP)

    def test_key_is_normalised_like_ciphertext(self) -> None:
        for written in ("hill", "H I L L", "H.i.L,l!", "hIlL\n"):
            self.assertEqual(key_from_string(written), HILL)

    def test_matrix_to_string_round_trip(self) -> None:
        self.assertEqual(matrix_to_string(HILL), "HILL")
        self.assertEqual(matrix_to_string(GYBNQKURP), "GYBNQKURP")
        self.assertEqual(key_from_string(matrix_to_string(NON_KEYWORD)), NON_KEYWORD)

    def test_describe_matrix_is_copy_pasteable(self) -> None:
        self.assertEqual(describe_matrix(HILL), "key=HILL matrix=[[7,8],[11,11]]")

    def test_non_square_length_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            key_from_string("HILLS")
        message = str(caught.exception)
        self.assertTrue(message.strip())
        self.assertIn("perfect square", message)
        self.assertIn("4", message)
        self.assertIn("9", message)

    def test_empty_key_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            key_from_string("!!! 123 !!!")
        self.assertIn("no letters", str(caught.exception))

    def test_single_letter_key_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            key_from_string("A")
        self.assertIn("1 x 1", str(caught.exception))


# ---------------------------------------------------------------------------
# Encryption and decryption
# ---------------------------------------------------------------------------


class TestEncryptDecrypt(unittest.TestCase):
    def test_known_pair_2x2(self) -> None:
        # Worked by hand with K = HILL = [[7,8],[11,11]]:
        #   AT (0,19)  -> (152, 209) mod 26 = (22, 1)  = WB
        #   TA (19,0)  -> (133, 209) mod 26 = (3, 1)   = DB
        #   CK (2,10)  -> (94, 132)  mod 26 = (16, 2)  = QC
        #   AT         -> WB
        #   DA (3,0)   -> (21, 33)   mod 26 = (21, 7)  = VH
        #   WN (22,13) -> (258, 385) mod 26 = (24, 21) = YV
        self.assertEqual(encrypt("ATTACKATDAWN", HILL), "WBDBQCWBVHYV")

    def test_known_pair_3x3(self) -> None:
        # K = GYBNQKURP.
        #   ACT (0,2,19)  -> (67, 222, 319) mod 26 = (15, 14, 7)  = POH
        #   CAT (2,0,19)  -> (31, 216, 325) mod 26 = (5, 8, 13)   = FIN
        self.assertEqual(encrypt("ACTCAT", GYBNQKURP), "POHFIN")

    def test_known_pair_with_padding(self) -> None:
        # HELLO is five letters, so an X is appended before enciphering:
        #   HE (7,4)   -> (81, 121)  mod 26 = (3, 17)  = DR
        #   LL (11,11) -> (165, 242) mod 26 = (9, 8)   = JI
        #   OX (14,23) -> (282, 407) mod 26 = (22, 17) = WR
        self.assertEqual(encrypt("HELLO", HILL), "DRJIWR")
        # Decryption cannot tell the filler from real plaintext, and says so
        # by simply handing it back.
        self.assertEqual(decrypt("DRJIWR", HILL), "HELLOX")

    def test_alternative_filler(self) -> None:
        self.assertEqual(decrypt(encrypt("HELLO", HILL, filler="Q"), HILL), "HELLOQ")

    def test_round_trip_several_keys(self) -> None:
        message = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGANDAWAYITWENTAGAINXY"
        for matrix in (HILL, NON_KEYWORD, [[1, 0], [0, 1]], [[9, 4], [5, 7]]):
            size = len(matrix)
            trimmed = message[: len(message) - len(message) % size]
            self.assertEqual(decrypt(encrypt(trimmed, matrix), matrix), trimmed)

    def test_round_trip_3x3_and_4x4(self) -> None:
        message = corpus_letters(240)
        for matrix in (GYBNQKURP, KEY_4X4):
            size = len(matrix)
            trimmed = message[: len(message) - len(message) % size]
            self.assertEqual(decrypt(encrypt(trimmed, matrix), matrix), trimmed)

    def test_input_robustness(self) -> None:
        expected = encrypt("ATTACKATDAWN", HILL)
        variants = [
            "attack at dawn",
            "Attack, at dawn!",
            "ATTAC KATDA WN",
            "ATTAC KATDA\nWN\n",
            "  a t t a c k a t d a w n  ",
            "ATTACK-AT-DAWN (1917)",
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertEqual(encrypt(variant, HILL), expected)
                self.assertEqual(decrypt(expected, HILL), "ATTACKATDAWN")

    def test_empty_input_does_not_crash(self) -> None:
        self.assertEqual(encrypt("", HILL), "")
        self.assertEqual(decrypt("", HILL), "")
        self.assertEqual(encrypt("1234 !!", HILL), "")
        self.assertEqual(decrypt("1234 !!", HILL), "")

    def test_decrypt_rejects_a_partial_final_block(self) -> None:
        with self.assertRaises(ValueError) as caught:
            decrypt("WBDBQ", HILL)
        message = str(caught.exception)
        self.assertTrue(message.strip())
        self.assertIn("multiple", message)
        self.assertIn("block size 2", message)

    def test_encrypt_refuses_an_uninvertible_key(self) -> None:
        with self.assertRaises(NotInvertibleError) as caught:
            encrypt("ATTACKATDAWN", [[1, 2], [3, 4]])
        self.assertIn("could never be deciphered", str(caught.exception))

    def test_decrypt_refuses_an_uninvertible_key(self) -> None:
        with self.assertRaises(NotInvertibleError) as caught:
            decrypt("WBDBQCWBVHYV", [[2, 4], [6, 8]])
        self.assertTrue(str(caught.exception).strip())

    def test_bad_filler_raises(self) -> None:
        for filler in ("", "XY", "1", " "):
            with self.assertRaises(ValueError) as caught:
                encrypt("HELLO", HILL, filler=filler)
            self.assertIn("single A-Z letter", str(caught.exception))

    def test_non_square_matrix_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            encrypt("HELLO", [[1, 2, 3], [4, 5, 6]])
        self.assertIn("square", str(caught.exception))


# ---------------------------------------------------------------------------
# The known-plaintext attack
# ---------------------------------------------------------------------------


class TestKnownPlaintextAttack(unittest.TestCase):
    def test_recovers_a_2x2_key_from_four_letters(self) -> None:
        plain = "ATTACKATDAWN"
        result = known_plaintext_attack(encrypt(plain, HILL), plain, 2)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.matrix, HILL)
        self.assertTrue(result.verified)
        self.assertEqual(result.blocks_available, 6)
        self.assertIn("HILL", result.describe())

    def test_recovers_a_3x3_key(self) -> None:
        plain = corpus_letters(60)
        result = known_plaintext_attack(encrypt(plain, GYBNQKURP), plain, 3)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.matrix, GYBNQKURP)

    def test_tries_further_offsets_when_the_first_window_is_singular(self) -> None:
        # Blocks AB, AB, HI, LL. The first window is [AB, AB], whose matrix
        # P = [[0,0],[1,1]] has determinant 0, so it determines nothing.
        # The second window [AB, HI] gives P = [[0,7],[1,8]] with determinant
        # -7 = 19 mod 26, which is coprime to 26 and does determine the key.
        plain = "ABABHILLWORDSXYZ"
        result = known_plaintext_attack(encrypt(plain, HILL), plain, 2)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.matrix, HILL)
        self.assertEqual(result.blocks_used, (1, 2))
        self.assertEqual(result.attempts[0].blocks, (0, 1))
        self.assertEqual(result.attempts[0].determinant, 0)
        self.assertIn("singular", result.attempts[0].outcome)

    def test_misaligned_crib_is_reported_not_answered(self) -> None:
        # THIS IS THE FAILURE MODE THAT MATTERS. Shifting the matched texts by
        # a single letter breaks the block alignment. Plenty of block pairs
        # still give an invertible P, so the algebra happily produces keys --
        # they are just wrong. The verification step must catch that and the
        # attack must report failure rather than a confident answer.
        plain = corpus_letters(80)
        cipher = encrypt(plain, HILL)
        result = known_plaintext_attack(cipher[1:41], plain[1:41], 2)
        self.assertFalse(result.succeeded)
        self.assertIsNone(result.matrix)
        self.assertIsNone(result.blocks_used)
        self.assertTrue(result.reason.strip())
        self.assertIn("wrong offset", result.reason)
        self.assertTrue(result.attempts)
        # At least one attempt produced a well-formed key that failed to
        # reproduce the known text: that is exactly what verification is for.
        self.assertTrue(
            any("failed verification" in result.reason for _ in (0,))
            and any("not aligned" in attempt.outcome for attempt in result.attempts)
        )

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            known_plaintext_attack("ABCDEF", "ABCD", 2)
        self.assertIn("same length", str(caught.exception))

    def test_too_few_blocks_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            known_plaintext_attack("WB", "AT", 2)
        message = str(caught.exception)
        self.assertIn("at least 2 matched blocks", message)
        self.assertIn("4 letters", message)

    def test_bad_size_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            known_plaintext_attack("ABCD", "ABCD", 1)
        self.assertIn("at least 2", str(caught.exception))


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


class TestSolveExhaustive(unittest.TestCase):
    """The 2 x 2 exhaustive search. Each of these runs the real search."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.plain = corpus_letters(400)
        cls.cipher = encrypt(cls.plain, HILL)

    def test_recovers_a_keyword_key_from_ciphertext(self) -> None:
        result = solve(self.cipher, top=5)
        best = result.best()
        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best.plaintext, self.plain)
        self.assertIn("key=HILL", best.key)
        self.assertEqual(best.confidence(), "strong")
        self.assertTrue(best.diagnostics["search_complete"])
        self.assertEqual(best.diagnostics["key_space_covered"], 1.0)
        self.assertEqual(best.diagnostics["keys_scored"], invertible_matrix_count(2))
        self.assertNotIn("time_budget_hit", best.diagnostics)
        # A ranked list, not one answer.
        self.assertEqual(len(result), 5)
        self.assertIsNotNone(result.score_gap())

    def test_stage_one_digraph_scoring_alone_picks_the_right_key(self) -> None:
        # With shortlist=1 only the single best key from the cheap digraph
        # pass is rescored, so this pins down the hot loop itself rather than
        # letting the order-3 rescoring cover for a mistake in it. It also
        # shows 60 blocks (120 letters) are enough for stage 1 to decide.
        result = solve(self.cipher, top=1, shortlist=1, sample_blocks=60)
        best = result.best()
        assert best is not None
        self.assertEqual(best.diagnostics["stage2_rescored"], 1)
        self.assertEqual(best.diagnostics["stage1_blocks_scored"], 60)
        self.assertEqual(best.plaintext, self.plain)

    def test_recovers_a_key_that_does_not_read_as_a_keyword(self) -> None:
        # The search visits keyword-shaped matrices first. A completed sweep
        # must still find a key whose rows spell nothing.
        plain = corpus_letters(360)
        cipher = encrypt(plain, NON_KEYWORD)
        best = solve(cipher, top=3).best()
        assert best is not None
        self.assertEqual(best.plaintext, plain)
        self.assertIn(f"key={matrix_to_string(NON_KEYWORD)}", best.key)

    def test_accepts_normalized_text_and_relays_out_the_layout(self) -> None:
        grouped = " ".join(
            self.cipher[i : i + 5] for i in range(0, len(self.cipher), 5)
        )
        normalized = normalize(grouped)
        best = solve(normalized, top=1).best()
        assert best is not None
        self.assertEqual(best.plaintext, self.plain)
        self.assertIsNotNone(best.display)
        assert best.display is not None
        self.assertEqual(len(best.display), len(grouped))
        self.assertEqual(letters_only(best.display), self.plain)

    def test_time_budget_stops_cleanly_and_says_so(self) -> None:
        # OBSERVED FAILURE MODE: a search that did not finish must not be
        # presented as one that did.
        result = solve(self.cipher, top=2, time_budget=0.25)
        best = result.best()
        assert best is not None
        self.assertTrue(best.diagnostics["time_budget_hit"])
        self.assertFalse(best.diagnostics["search_complete"])
        self.assertLess(best.diagnostics["key_space_covered"], 1.0)
        self.assertLess(best.diagnostics["keys_scored"], invertible_matrix_count(2))
        self.assertTrue(
            any("stopped the search" in note for note in result.notes),
            result.notes,
        )
        self.assertTrue(any("biased" in note for note in result.notes))

    def test_trailing_letters_are_reported_not_hidden(self) -> None:
        result = solve(self.cipher + "Q", top=1)
        best = result.best()
        assert best is not None
        self.assertEqual(best.diagnostics["trailing_letters_ignored"], 1)
        self.assertEqual(best.diagnostics["letters_used"], 400)
        # relayout would be a lie when a letter was dropped, so there is none.
        self.assertIsNone(best.display)
        self.assertTrue(any("ignored" in note for note in result.notes))


class TestSolveOtherRoutes(unittest.TestCase):
    def test_supplied_key_is_scored_not_endorsed(self) -> None:
        plain = corpus_letters(200)
        cipher = encrypt(plain, HILL)
        result = solve(cipher, key="HILL")
        best = result.best()
        assert best is not None
        self.assertEqual(best.plaintext, plain)
        self.assertEqual(best.diagnostics["search"], "none -- the key was supplied, not found")
        self.assertTrue(any("not evidence" in note for note in result.notes))

    def test_supplied_matrix_matches_supplied_key(self) -> None:
        plain = corpus_letters(200)
        cipher = encrypt(plain, HILL)
        by_key = solve(cipher, key="HILL").best()
        by_matrix = solve(cipher, matrix=HILL).best()
        assert by_key is not None and by_matrix is not None
        self.assertEqual(by_key.plaintext, by_matrix.plaintext)

    def test_a_wrong_key_is_not_dressed_up_as_a_solution(self) -> None:
        # OBSERVED FAILURE MODE: BEAR is invertible (det 17) but wrong, so the
        # decryption is nonsense. The tool must say so rather than presenting
        # the only candidate it has as an answer.
        plain = corpus_letters(300)
        cipher = encrypt(plain, HILL)
        best = solve(cipher, key="BEAR").best()
        assert best is not None
        self.assertNotEqual(best.plaintext, plain)
        self.assertIn(best.confidence(), {"weak", "unlikely"})
        self.assertLess(best.diagnostics["word_coverage"], 0.35)
        self.assertLess(best.diagnostics["normalised_score"], -1.8)

    def test_key_and_matrix_together_raise(self) -> None:
        with self.assertRaises(ValueError) as caught:
            solve("ABCD", key="HILL", matrix=HILL)
        self.assertIn("not both", str(caught.exception))

    def test_size_contradicting_the_key_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            solve("ABCD", key="HILL", size=3)
        self.assertIn("contradicts", str(caught.exception))

    def test_crib_solves_a_3x3_key(self) -> None:
        plain = corpus_letters(300)
        cipher = encrypt(plain, GYBNQKURP)
        result = solve(cipher, size=3, crib=plain[:12])
        best = result.best()
        assert best is not None
        self.assertEqual(best.plaintext, plain)
        self.assertIn("key=GYBNQKURP", best.key)
        # Which three blocks were used depends on which give an invertible P;
        # what matters is that exactly three were needed and it verified.
        self.assertEqual(len(best.diagnostics["crib_blocks_used"]), 3)
        self.assertEqual(best.diagnostics["crib_verified_against_blocks"], 4)
        self.assertTrue(any("solved for, not searched for" in n for n in result.notes))

    def test_crib_at_a_later_block_boundary(self) -> None:
        plain = corpus_letters(200)
        cipher = encrypt(plain, HILL)
        best = solve(cipher, crib=plain[40:60], crib_at=40).best()
        assert best is not None
        self.assertEqual(best.plaintext, plain)

    def test_crib_off_a_block_boundary_raises(self) -> None:
        plain = corpus_letters(200)
        cipher = encrypt(plain, HILL)
        with self.assertRaises(ValueError) as caught:
            solve(cipher, crib=plain[41:61], crib_at=41)
        message = str(caught.exception)
        self.assertIn("block boundary", message)
        self.assertIn("40", message)
        self.assertIn("42", message)

    def test_a_wrong_crib_on_a_3x3_yields_no_candidates(self) -> None:
        # OBSERVED FAILURE MODE: the crib is the only attack available for a
        # 3 x 3 key. When it is wrong there is nothing left to offer, and the
        # honest output is an empty candidate list with an explanation -- not
        # a plausible-looking guess.
        plain = corpus_letters(300)
        cipher = encrypt(plain, GYBNQKURP)
        result = solve(cipher, size=3, crib="THISISNOTTHEPLAINTEXTATALL")
        self.assertEqual(len(result), 0)
        self.assertIsNone(result.best())
        self.assertTrue(any("attack failed" in note for note in result.notes))
        self.assertTrue(any("only available attack" in note for note in result.notes))


class TestSolveRefusals(unittest.TestCase):
    def test_3x3_without_a_key_or_crib_refuses_to_search(self) -> None:
        # OBSERVED FAILURE MODE, and the most important one in this module:
        # 26^9 matrices cannot be searched, so nothing is returned and the
        # reason is stated plainly.
        plain = corpus_letters(300)
        cipher = encrypt(plain, GYBNQKURP)
        result = solve(cipher, size=3)
        self.assertIsInstance(result, CandidateSet)
        self.assertEqual(len(result), 0)
        self.assertIsNone(result.best())
        joined = " ".join(result.notes)
        self.assertIn("5,429,503,678,976", joined)
        self.assertIn("impossible", joined)
        self.assertIn("crib", joined)

    def test_4x4_without_a_key_or_crib_refuses_to_search(self) -> None:
        plain = corpus_letters(320)
        cipher = encrypt(plain, KEY_4X4)
        result = solve(cipher, size=4)
        self.assertEqual(len(result), 0)
        self.assertTrue(any("impossible" in note for note in result.notes))

    def test_empty_input_returns_an_empty_set_with_a_reason(self) -> None:
        for text in ("", "   ", "1234!!"):
            result = solve(text)
            self.assertEqual(len(result), 0)
            self.assertTrue(result.notes)
            self.assertIn("no letters", result.notes[0])

    def test_text_shorter_than_one_block_returns_nothing(self) -> None:
        result = solve("A", size=2)
        self.assertEqual(len(result), 0)
        self.assertTrue(any("whole block" in note for note in result.notes))

    def test_block_size_below_two_raises(self) -> None:
        with self.assertRaises(ValueError) as caught:
            solve("ABCDEF", size=1)
        self.assertIn("at least 2", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
