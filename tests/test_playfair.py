"""Tests for the Playfair cipher, its digraph rules and its annealing attack.

The known-answer vectors were worked out by hand from the squares printed in
the test bodies, one digraph at a time, before any code was run.
"""

from __future__ import annotations

import unittest

from cipher_tool.playfair import (
    DEFAULT_ITERATIONS,
    STANDARD_OMISSIONS,
    PlayfairSquare,
    as_square,
    canonical_square,
    check_filler,
    decrypt,
    encrypt,
    plain_square,
    playfair_square,
    prepare_digraphs,
    prepare_text,
    rotate_square,
    solve,
    square_from_letters,
    validate_ciphertext,
)
from cipher_tool.scoring import DATA_DIR

# The classic worked example.
#
#   P L A Y F
#   I R E X M
#   B C D G H
#   K N O Q S
#   T U V W Z
#
# HIDETHEGOLDINTHETREESTUMP prepares to
#   HI DE TH EG OL DI NT HE TR EX ES TU MP
# and each pair enciphers as:
#   HI  H(2,4) I(1,0) rectangle -> (2,0)(1,4) = BM
#   DE  same column 2           -> (3,2)(2,2) = OD
#   TH  T(4,0) H(2,4) rectangle -> (4,4)(2,0) = ZB
#   EG  E(1,2) G(2,3) rectangle -> (1,3)(2,2) = XD
#   OL  O(3,2) L(0,1) rectangle -> (3,1)(0,2) = NA
#   DI  D(2,2) I(1,0) rectangle -> (2,0)(1,2) = BE
#   NT  N(3,1) T(4,0) rectangle -> (3,0)(4,1) = KU
#   HE  H(2,4) E(1,2) rectangle -> (2,2)(1,4) = DM
#   TR  T(4,0) R(1,1) rectangle -> (4,1)(1,0) = UI
#   EX  same row 1              -> (1,3)(1,4) = XM
#   ES  E(1,2) S(3,4) rectangle -> (1,4)(3,2) = MO
#   TU  same row 4              -> (4,1)(4,2) = UV
#   MP  M(1,4) P(0,0) rectangle -> (1,0)(0,4) = IF
EXAMPLE_KEY = "PLAYFAIR EXAMPLE"
EXAMPLE_PLAIN = "HIDE THE GOLD IN THE TREE STUMP"
EXAMPLE_PREPARED = "HIDETHEGOLDINTHETREXESTUMP"
EXAMPLE_CIPHER = "BMODZBXDNABEKUDMUIXMMOUVIF"

# The other standard square, used for the rule-by-rule tests.
#
#   M O N A R
#   C H Y B D
#   E F G I K
#   L P Q S T
#   U V W X Z
MONARCHY = "MONARCHY"


def corpus_letters(count: int, offset: int = 0) -> str:
    """A slice of the project's own expository prose, letters only."""
    text = (DATA_DIR / "corpus_04_expository.txt").read_text(encoding="utf-8")
    clean = "".join(ch for ch in text.upper() if "A" <= ch <= "Z")
    if len(clean) < offset + count:
        raise AssertionError("corpus file is shorter than this test expects")
    return clean[offset : offset + count]


class TestSquare(unittest.TestCase):
    def test_keyword_square_is_built_in_reading_order(self) -> None:
        square = playfair_square(EXAMPLE_KEY)
        self.assertEqual(square.letters, "PLAYFIREXMBCDGHKNOQSTUVWZ")
        self.assertEqual(
            square.rows(), ("PLAYF", "IREXM", "BCDGH", "KNOQS", "TUVWZ")
        )
        self.assertEqual(square.omitted, "J")
        self.assertEqual(square.folded_onto, "I")

    def test_monarchy_square(self) -> None:
        self.assertEqual(
            playfair_square(MONARCHY).rows(),
            ("MONAR", "CHYBD", "EFGIK", "LPQST", "UVWXZ"),
        )

    def test_keyword_j_is_merged_before_de_duplication(self) -> None:
        # JAM starts with J, which is I; the square must not contain J and
        # must not start with two I-like letters.
        self.assertEqual(playfair_square("JAM").letters[:3], "IAM")
        self.assertNotIn("J", playfair_square("JAM").letters)

    def test_plain_square_is_the_alphabet_without_j(self) -> None:
        self.assertEqual(plain_square().letters, "ABCDEFGHIKLMNOPQRSTUVWXYZ")

    def test_dropping_q_instead_of_j_keeps_j(self) -> None:
        square = playfair_square("ORANGES", omit="Q")
        self.assertNotIn("Q", square.letters)
        self.assertIn("J", square.letters)
        self.assertIsNone(square.folded_onto)

    def test_position_and_at_agree(self) -> None:
        square = playfair_square(MONARCHY)
        self.assertEqual(square.position("M"), (0, 0))
        self.assertEqual(square.position("Z"), (4, 4))
        self.assertEqual(square.at(2, 3), "I")
        # `at` wraps, which is what the row and column rules rely on.
        self.assertEqual(square.at(0, 5), square.at(0, 0))
        self.assertEqual(square.at(5, 0), square.at(0, 0))

    def test_fold_maps_j_onto_i(self) -> None:
        self.assertEqual(plain_square().fold("Jam jar!"), "IAMIAR")

    def test_fold_deletes_a_dropped_letter(self) -> None:
        self.assertEqual(plain_square(omit="Q").fold("QUEUE"), "UEUE")

    def test_as_square_accepts_both_forms(self) -> None:
        square = playfair_square(MONARCHY)
        self.assertIs(as_square(square), square)
        self.assertEqual(as_square(MONARCHY).letters, square.letters)


class TestSquareEquivalence(unittest.TestCase):
    """Cyclic rotations of a square encipher identically."""

    def test_rotation_does_not_change_the_ciphertext(self) -> None:
        square = playfair_square(MONARCHY)
        message = "THE GOLD IS BURIED UNDER THE OLD OAK TREE"
        expected = encrypt(message, square)
        for rows in range(5):
            for columns in range(5):
                rotated = rotate_square(square, rows=rows, columns=columns)
                self.assertEqual(
                    encrypt(message, rotated),
                    expected,
                    f"rotation by {rows} rows and {columns} columns changed "
                    "the ciphertext",
                )

    def test_rotation_moves_rows_and_columns(self) -> None:
        #   A B C D E        F G H I K        B C D E A
        #   F G H I K        L M N O P        G H I K F
        #   L M N O P  -1->  Q R S T U  -1->  M N O P L   (rows, then columns)
        #   Q R S T U        V W X Y Z        R S T U Q
        #   V W X Y Z        A B C D E        W X Y Z V
        square = plain_square()
        self.assertEqual(
            rotate_square(square, rows=1).rows(),
            ("FGHIK", "LMNOP", "QRSTU", "VWXYZ", "ABCDE"),
        )
        self.assertEqual(
            rotate_square(square, columns=1).rows(),
            ("BCDEA", "GHIKF", "MNOPL", "RSTUQ", "WXYZV"),
        )
        self.assertEqual(
            rotate_square(square, rows=5, columns=5).letters, square.letters
        )

    def test_canonical_form_is_shared_by_equivalent_squares(self) -> None:
        square = playfair_square(MONARCHY)
        rotated = rotate_square(square, rows=3, columns=2)
        self.assertNotEqual(square.letters, rotated.letters)
        self.assertEqual(
            canonical_square(square).letters, canonical_square(rotated).letters
        )
        self.assertEqual(canonical_square(square).letters[0], "A")

    def test_canonical_form_undoes_a_row_rotation(self) -> None:
        # A sits in the top row of the MONARCHY square, so rotate it away
        # first: canonicalising has to move whole rows to put it back.
        square = rotate_square(playfair_square(MONARCHY), rows=2, columns=1)
        self.assertNotEqual(square.position("A")[0], 0)
        self.assertEqual(canonical_square(square).position("A"), (0, 0))


class TestDigraphRules(unittest.TestCase):
    """Each of the three rules, and each wrap, tested on its own."""

    def test_same_row(self) -> None:
        # H(1,1) and Y(1,2) both step right: Y(1,2) and B(1,3).
        self.assertEqual(encrypt("HY", MONARCHY), "YB")
        self.assertEqual(decrypt("YB", MONARCHY), "HY")

    def test_same_row_wraps(self) -> None:
        # D is the last cell of row 1, so it wraps round to C.
        self.assertEqual(encrypt("DC", MONARCHY), "CH")
        self.assertEqual(decrypt("CH", MONARCHY), "DC")

    def test_same_column(self) -> None:
        # O(0,1) and H(1,1) both step down: H(1,1) and F(2,1).
        self.assertEqual(encrypt("OH", MONARCHY), "HF")
        self.assertEqual(decrypt("HF", MONARCHY), "OH")

    def test_same_column_wraps(self) -> None:
        # V is the bottom of column 1, so it wraps round to O.
        self.assertEqual(encrypt("VO", MONARCHY), "OH")
        self.assertEqual(decrypt("OH", MONARCHY), "VO")

    def test_rectangle(self) -> None:
        # M(0,0) and B(1,3) swap columns: A(0,3) and C(1,0).
        self.assertEqual(encrypt("MB", MONARCHY), "AC")
        self.assertEqual(decrypt("AC", MONARCHY), "MB")

    def test_rectangle_rule_is_its_own_inverse(self) -> None:
        square = playfair_square(MONARCHY)
        # Encrypting the ciphertext of a rectangle pair returns the plaintext.
        self.assertEqual(encrypt(encrypt("MB", square), square), "MB")

    def test_reversing_a_digraph_reverses_the_ciphertext(self) -> None:
        self.assertEqual(encrypt("BM", MONARCHY), "CA")
        self.assertEqual(encrypt("YH", MONARCHY), "BY")


class TestPreparation(unittest.TestCase):
    def test_doubled_letter_is_split(self) -> None:
        self.assertEqual(prepare_text("BALLOON"), "BALXLOON")
        self.assertEqual(
            prepare_digraphs("HELLO"), ["HE", "LX", "LO"]
        )

    def test_odd_final_letter_is_padded(self) -> None:
        self.assertEqual(prepare_text("ODD"), "ODDX")
        self.assertEqual(prepare_text("A"), "AX")

    def test_doubled_filler_uses_the_alternative(self) -> None:
        # The classic bug: splitting XX with X gives XX again.
        self.assertEqual(prepare_text("XX"), "XQXQ")
        self.assertEqual(prepare_digraphs("XX"), ["XQ", "XQ"])

    def test_lone_trailing_filler_uses_the_alternative(self) -> None:
        self.assertEqual(prepare_text("AXX"), "AXXQ")
        self.assertEqual(prepare_text("X"), "XQ")

    def test_double_across_a_pair_boundary_needs_no_filler(self) -> None:
        # The two Xs are already in different digraphs.
        self.assertEqual(prepare_text("AXXB"), "AXXB")

    def test_custom_filler(self) -> None:
        self.assertEqual(prepare_text("BALLOON", filler="Z"), "BALZLOON")
        self.assertEqual(
            prepare_text("ZZ", filler="Z", alternative="Q"), "ZQZQ"
        )

    def test_j_is_merged(self) -> None:
        self.assertEqual(prepare_text("JAM"), "IAMX")

    def test_empty_text(self) -> None:
        self.assertEqual(prepare_digraphs(""), [])
        self.assertEqual(prepare_text("   .!?  "), "")


class TestKnownAnswer(unittest.TestCase):
    def test_hand_computed_triple(self) -> None:
        self.assertEqual(encrypt(EXAMPLE_PLAIN, EXAMPLE_KEY), EXAMPLE_CIPHER)

    def test_hand_computed_decryption(self) -> None:
        self.assertEqual(decrypt(EXAMPLE_CIPHER, EXAMPLE_KEY), EXAMPLE_PREPARED)

    def test_balloon(self) -> None:
        # BA LX LO ON with the MONARCHY square:
        #   BA same column 3 -> I(2,3) B(1,3) = IB
        #   LX rectangle     -> S(3,3) U(4,0) = SU
        #   LO rectangle     -> P(3,1) M(0,0) = PM
        #   ON same row 0    -> N(0,2) A(0,3) = NA
        self.assertEqual(encrypt("BALLOON", MONARCHY), "IBSUPMNA")


class TestRoundTrip(unittest.TestCase):
    def test_round_trip_for_several_keys(self) -> None:
        message = "MEET ME BY THE OLD BRIDGE AT MIDNIGHT AND BRING THE MAPS"
        for key in ("MONARCHY", "PLAYFAIR EXAMPLE", "Z", "CIPHERCHALLENGE"):
            with self.subTest(key=key):
                cipher = encrypt(message, key)
                self.assertEqual(decrypt(cipher, key), prepare_text(message))

    def test_round_trip_with_a_custom_filler(self) -> None:
        message = "ALL HANDS TO THE PUMPS"
        cipher = encrypt(message, MONARCHY, filler="Z")
        self.assertEqual(
            decrypt(cipher, MONARCHY), prepare_text(message, filler="Z")
        )

    def test_round_trip_when_q_is_dropped(self) -> None:
        square = playfair_square("ORANGES", omit="Q")
        message = "THE BROWN FOX JUMPS OVER THE LAZY DOG"
        cipher = encrypt(message, square, alternative="Z")
        self.assertEqual(
            decrypt(cipher, square),
            prepare_text(message, square=square, alternative="Z"),
        )

    def test_ciphertext_is_always_an_even_number_of_letters(self) -> None:
        for message in ("A", "AB", "ABC", "HELLO", "BALLOON", "XX"):
            with self.subTest(message=message):
                self.assertEqual(len(encrypt(message, MONARCHY)) % 2, 0)


class TestInputRobustness(unittest.TestCase):
    def test_layout_does_not_change_the_result(self) -> None:
        variants = [
            "HIDETHEGOLDINTHETREESTUMP",
            "hide the gold in the tree stump",
            "Hide the gold, in the tree stump!",
            "HIDET HEGOL DINTH ETREE STUMP",
            "HIDET HEGOL\nDINTH ETREE\nSTUMP",
            "  hide\tthe\ngold in the tree stump  ",
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertEqual(encrypt(variant, EXAMPLE_KEY), EXAMPLE_CIPHER)

    def test_ciphertext_layout_does_not_change_decryption(self) -> None:
        grouped = "BMODZ BXDNA BEKUD MUIXM\nMOUVI F"
        self.assertEqual(decrypt(grouped, EXAMPLE_KEY), EXAMPLE_PREPARED)

    def test_keyword_layout_does_not_change_the_square(self) -> None:
        self.assertEqual(
            playfair_square("playfair example!").letters,
            playfair_square(EXAMPLE_KEY).letters,
        )


class TestEmptyInput(unittest.TestCase):
    def test_encrypt_and_decrypt_return_empty(self) -> None:
        self.assertEqual(encrypt("", MONARCHY), "")
        self.assertEqual(decrypt("", MONARCHY), "")
        self.assertEqual(encrypt("!!! 123 ???", MONARCHY), "")

    def test_solve_returns_an_empty_candidate_set(self) -> None:
        found = solve("")
        self.assertEqual(len(found), 0)
        self.assertIsNone(found.best())

    def test_validate_reports_the_emptiness(self) -> None:
        problems = validate_ciphertext("", plain_square())
        self.assertEqual(len(problems), 1)
        self.assertIn("no letters", problems[0])


class TestInvalidInput(unittest.TestCase):
    """Every rejection must say what is wrong and what to do about it."""

    def assertRaisesWithMessage(self, call, *args, **kwargs) -> str:
        with self.assertRaises(ValueError) as caught:
            call(*args, **kwargs)
        message = str(caught.exception)
        self.assertTrue(message.strip(), "ValueError carried no message")
        return message

    def test_keyword_without_letters(self) -> None:
        message = self.assertRaisesWithMessage(playfair_square, "123 ...")
        self.assertIn("keyword", message)

    def test_empty_keyword(self) -> None:
        self.assertRaisesWithMessage(playfair_square, "")

    def test_keyword_made_only_of_the_dropped_letter(self) -> None:
        message = self.assertRaisesWithMessage(
            playfair_square, "QQQ", omit="Q"
        )
        self.assertIn("omitted", message)

    def test_omit_must_be_one_letter(self) -> None:
        message = self.assertRaisesWithMessage(
            playfair_square, "MONARCHY", omit="JQ"
        )
        self.assertIn("one letter", message)

    def test_square_needs_twenty_five_letters(self) -> None:
        message = self.assertRaisesWithMessage(square_from_letters, "ABCDE")
        self.assertIn("25", message)

    def test_square_may_not_repeat_a_letter(self) -> None:
        message = self.assertRaisesWithMessage(
            square_from_letters, "AACDEFGHIKLMNOPQRSTUVWXYZ"
        )
        self.assertIn("repeat", message)

    def test_square_may_not_contain_the_omitted_letter(self) -> None:
        message = self.assertRaisesWithMessage(
            square_from_letters, "ABCDEFGHIJKLMNOPQRSTUVWXY"
        )
        self.assertIn("omitted", message)

    def test_filler_must_be_one_letter(self) -> None:
        message = self.assertRaisesWithMessage(
            encrypt, "HELLO", MONARCHY, filler="XY"
        )
        self.assertIn("single letter", message)

    def test_filler_and_alternative_must_differ(self) -> None:
        message = self.assertRaisesWithMessage(
            encrypt, "HELLO", MONARCHY, filler="X", alternative="X"
        )
        self.assertIn("must differ", message)

    def test_filler_must_be_in_the_square(self) -> None:
        square = playfair_square("ORANGES", omit="Q")
        message = self.assertRaisesWithMessage(encrypt, "HELLO", square)
        self.assertIn("not in the square", message)

    def test_odd_length_ciphertext(self) -> None:
        message = self.assertRaisesWithMessage(decrypt, "ABCDE", MONARCHY)
        self.assertIn("odd number", message)

    def test_ciphertext_containing_the_merged_letter(self) -> None:
        message = self.assertRaisesWithMessage(decrypt, "ABJC", MONARCHY)
        self.assertIn("J", message)

    def test_anchor_must_be_in_the_square(self) -> None:
        message = self.assertRaisesWithMessage(
            canonical_square, playfair_square(MONARCHY), anchor="J"
        )
        self.assertIn("not in this square", message)

    def test_key_of_the_wrong_type(self) -> None:
        self.assertRaisesWithMessage(as_square, 17)

    def test_negative_restarts(self) -> None:
        message = self.assertRaisesWithMessage(
            solve, EXAMPLE_CIPHER, restarts=-1
        )
        self.assertIn("negative", message)

    def test_negative_temperature(self) -> None:
        message = self.assertRaisesWithMessage(
            solve, EXAMPLE_CIPHER, temperature=-1.0
        )
        self.assertIn("temperature", message)

    def test_zero_restarts_switches_the_search_off(self) -> None:
        # The `auto` pipeline uses restarts=0 to mean "do not run this stage".
        found = solve(EXAMPLE_CIPHER, restarts=0)
        self.assertEqual(len(found), 0)

    def test_solve_rejects_impossible_ciphertext(self) -> None:
        message = self.assertRaisesWithMessage(solve, "ABCDE")
        self.assertIn("odd number", message)
        # A J in the ciphertext used to make the keyless search raise as well.
        # It no longer does -- the J says which letter the square omitted, and
        # the search reads it (see TestSolveChoosesTheOmittedLetter). It is
        # still an error against a square the CALLER asserted, because then it
        # contradicts a statement rather than answering an open question.
        message = self.assertRaisesWithMessage(solve, "ABJC", key=MONARCHY)
        self.assertIn("J", message)


class TestValidateCiphertext(unittest.TestCase):
    def test_genuine_ciphertext_has_no_problems(self) -> None:
        self.assertEqual(
            validate_ciphertext(EXAMPLE_CIPHER, playfair_square(EXAMPLE_KEY)),
            [],
        )

    def test_odd_length_is_reported(self) -> None:
        problems = validate_ciphertext("ABCDE", plain_square())
        self.assertTrue(any("odd number" in problem for problem in problems))

    def test_merged_letter_is_reported_with_positions(self) -> None:
        problems = validate_ciphertext("ABCJ", plain_square())
        self.assertTrue(any("positions" in problem for problem in problems))

    def test_doubled_digraph_is_reported(self) -> None:
        problems = validate_ciphertext("ABLLCD", plain_square())
        # Offset 2 is the start of a digraph, so LL is a doubled pair.
        self.assertTrue(
            any("doubled" in problem for problem in problems), problems
        )

    def test_doubled_letters_across_a_boundary_are_not_a_problem(self) -> None:
        # ALLB: the two Ls straddle the digraph boundary AL|LB, which Playfair
        # produces all the time.
        self.assertEqual(validate_ciphertext("ALLB", plain_square()), [])

    def test_dropped_letter_variant_reports_q(self) -> None:
        problems = validate_ciphertext("ABQC", plain_square(omit="Q"))
        self.assertTrue(any("drops Q" in problem for problem in problems))


class TestSolveWithSuppliedKey(unittest.TestCase):
    def test_known_key_recovers_the_plaintext(self) -> None:
        found = solve(EXAMPLE_CIPHER, key=EXAMPLE_KEY)
        best = found.best()
        self.assertIsNotNone(best)
        self.assertEqual(best.plaintext, EXAMPLE_PREPARED)
        self.assertEqual(best.key, f"key={EXAMPLE_KEY}")
        self.assertEqual(best.diagnostics["attack"], "supplied key (no search)")
        self.assertEqual(best.diagnostics["ciphertext_letters"], 26)
        self.assertEqual(best.diagnostics["digraphs"], 13)

    def test_the_right_key_beats_the_wrong_ones(self) -> None:
        plain = corpus_letters(400)
        cipher = encrypt(plain, "CIPHERCHALLENGE")
        found = solve(
            cipher, keys=["MONARCHY", "CIPHERCHALLENGE", "PLAYFAIR EXAMPLE"]
        )
        best = found.best()
        self.assertEqual(best.key, "key=CIPHERCHALLENGE")
        self.assertEqual(best.plaintext, prepare_text(plain))
        self.assertGreater(best.diagnostics["word_coverage"], 0.7)
        # And the margin over the runner-up is visible, not hidden.
        self.assertGreater(found.score_gap(), 0.5)

    def test_display_preserves_the_original_layout(self) -> None:
        grouped = "BMODZ BXDNA BEKUD MUIXM MOUVI F"
        best = solve(grouped, key=EXAMPLE_KEY).best()
        self.assertEqual(best.display, "HIDET HEGOL DINTH ETREX ESTUM P")


class TestSolveFailsHonestly(unittest.TestCase):
    """A wrong answer must look wrong. These tests watch it happen."""

    def test_wrong_key_is_not_reported_as_strong(self) -> None:
        plain = corpus_letters(400)
        cipher = encrypt(plain, "CIPHERCHALLENGE")
        best = solve(cipher, key="MONARCHY").best()
        self.assertNotEqual(best.plaintext, prepare_text(plain))
        self.assertLess(best.diagnostics["word_coverage"], 0.35)
        self.assertIn(best.confidence(), {"weak", "unlikely"})

    def test_short_ciphertext_is_labelled_hopeless(self) -> None:
        cipher = encrypt(corpus_letters(80), "CIPHERCHALLENGE")
        best = solve(cipher, seed=1, restarts=1, iterations=2000).best()
        self.assertIn("far too short", best.diagnostics["outlook"])
        self.assertEqual(best.diagnostics["ciphertext_letters"], len(cipher))
        self.assertNotEqual(best.confidence(), "strong")
        self.assertLess(best.diagnostics["word_coverage"], 0.7)

    def test_time_budget_is_respected_and_recorded(self) -> None:
        cipher = encrypt(corpus_letters(520), "CIPHERCHALLENGE")
        found = solve(cipher, seed=1, restarts=5, time_budget=0.25)
        best = found.best()
        self.assertTrue(best.diagnostics["time_budget_hit"])
        self.assertLess(best.diagnostics["moves_evaluated"], DEFAULT_ITERATIONS)
        self.assertNotEqual(best.confidence(), "strong")


class TestSolveByAnnealing(unittest.TestCase):
    """The statistical attack, on enough ciphertext for it to be fair."""

    def test_recovers_the_plaintext_from_520_letters(self) -> None:
        plain = corpus_letters(520)
        expected = prepare_text(plain)
        cipher = encrypt(plain, "CIPHERCHALLENGE")
        # Fillers make the ciphertext a little longer than the plaintext.
        self.assertEqual(len(cipher) % 2, 0)
        self.assertGreaterEqual(len(cipher), 520)

        found = solve(cipher, seed=1, restarts=5)
        best = found.best()

        self.assertGreater(
            best.diagnostics["word_coverage"],
            0.7,
            f"recovered text was: {best.plaintext[:120]}",
        )
        self.assertEqual(best.plaintext, expected)
        self.assertEqual(best.confidence(), "strong")
        self.assertEqual(best.diagnostics["ciphertext_letters"], len(cipher))
        self.assertIn("enough for this attack", best.diagnostics["outlook"])

    def test_recovered_square_is_equivalent_to_the_real_one(self) -> None:
        plain = corpus_letters(520)
        cipher = encrypt(plain, "CIPHERCHALLENGE")
        best = solve(cipher, seed=1, restarts=5).best()

        recovered = square_from_letters(best.key.split("=", 1)[1])
        truth = playfair_square("CIPHERCHALLENGE")
        # Rotations encipher identically, so compare canonical forms.
        self.assertEqual(
            canonical_square(recovered).letters, canonical_square(truth).letters
        )

    def test_the_search_is_reproducible(self) -> None:
        cipher = encrypt(corpus_letters(300), "CIPHERCHALLENGE")
        first = solve(cipher, seed=7, restarts=1, iterations=3000).best()
        second = solve(cipher, seed=7, restarts=1, iterations=3000).best()
        self.assertEqual(first.plaintext, second.plaintext)
        self.assertEqual(first.key, second.key)

    def test_different_seeds_take_different_paths(self) -> None:
        # Not a correctness requirement, but if two seeds gave identical
        # searches the seed would not be doing anything.
        cipher = encrypt(corpus_letters(300), "CIPHERCHALLENGE")
        first = solve(cipher, seed=1, restarts=1, iterations=3000).best()
        second = solve(cipher, seed=2, restarts=1, iterations=3000).best()
        self.assertNotEqual(first.key, second.key)


class TestSolveChoosesTheOmittedLetter(unittest.TestCase):
    """A J in the ciphertext is evidence about the square, not bad input.

    `cipher_tool playfair message.txt` used to end in a raw ValueError
    traceback for any ciphertext containing a J, because solve() refused to
    start unless the caller had already guessed the right omit= argument --
    an argument the command line does not even expose. A J proves the square
    did not merge I/J, so the keyless search now reads that and moves to the
    other standard omission, recording the swap on every candidate.
    """

    def q_square_ciphertext(self, count: int = 320) -> tuple[str, str]:
        """Ciphertext from a Q-dropping square, so it contains real Js."""
        plain = corpus_letters(count)
        square = playfair_square("CIPHERCHALLENGE", omit="Q")
        cipher = encrypt(plain, square, alternative="Z")
        self.assertIn("J", cipher, "this test needs a J in the ciphertext")
        expected = prepare_text(plain, square=square, alternative="Z")
        return cipher, expected

    def test_a_j_does_not_stop_the_keyless_search(self) -> None:
        cipher, _ = self.q_square_ciphertext()
        found = solve(cipher, seed=1, restarts=1, iterations=2000)
        self.assertGreater(len(found), 0)
        best = found.best()
        self.assertEqual(best.diagnostics["omitted_letter"], "Q")

    def test_the_swap_is_recorded_where_the_operator_will_read_it(self) -> None:
        cipher, _ = self.q_square_ciphertext()
        best = solve(cipher, seed=1, restarts=1, iterations=2000).best()
        note = best.diagnostics["omitted_letter_changed"]
        # It must say what it found, what it did about it, and that the
        # decision could still be wrong.
        self.assertIn("contains J", note)
        self.assertIn("drops Q", note)
        self.assertIn("not Playfair", note)

    def test_the_swap_lets_the_message_actually_be_read(self) -> None:
        # The point of the swap: this ciphertext is solvable, and refusing to
        # start meant nobody ever found that out.
        cipher, expected = self.q_square_ciphertext()
        best = solve(cipher, seed=1, restarts=3).best()
        self.assertEqual(best.plaintext, expected)
        self.assertEqual(best.confidence(), "strong")

    def test_ordinary_ciphertext_keeps_the_classic_merge(self) -> None:
        cipher = encrypt(corpus_letters(120), "CIPHERCHALLENGE")
        self.assertNotIn("J", cipher)
        best = solve(cipher, seed=1, restarts=1, iterations=2000).best()
        self.assertEqual(best.diagnostics["omitted_letter"], "J")
        self.assertNotIn("omitted_letter_changed", best.diagnostics)

    def test_a_supplied_key_is_not_quietly_re_keyed(self) -> None:
        # The operator has asserted a square. Contradicting evidence is worth
        # reporting; silently searching a different square is not.
        message = self.assertRaisesMessage(solve, "ABJDEFGH", key=MONARCHY)
        self.assertIn("merges I/J", message)

    def test_a_text_using_every_standard_omission_is_still_refused(self) -> None:
        # J and Q both present: no standard square can have produced this, and
        # inventing a third omission to search with would be manufacturing
        # evidence. Refusing, with the reason, is the honest answer.
        message = self.assertRaisesMessage(solve, "ABJQCDEF")
        for letter in STANDARD_OMISSIONS:
            self.assertIn(repr(letter), message)
        self.assertIn("not Playfair ciphertext", message)

    def assertRaisesMessage(self, call, *args, **kwargs) -> str:
        with self.assertRaises(ValueError) as caught:
            call(*args, **kwargs)
        return str(caught.exception)


class TestFillerValidation(unittest.TestCase):
    """A filler is either rejected or effective; it is never just ignored."""

    BAD_FILLERS = ("XY", "5", "QQQ", "", "  ")

    def test_check_filler_rejects_what_is_not_one_letter(self) -> None:
        for value in self.BAD_FILLERS:
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as caught:
                    check_filler(value)
                self.assertIn("single letter", str(caught.exception))

    def test_check_filler_cleans_what_it_accepts(self) -> None:
        self.assertEqual(check_filler(" x "), "X")
        self.assertEqual(check_filler("Z"), "Z")

    def test_check_filler_knows_which_argument_is_wrong(self) -> None:
        with self.assertRaises(ValueError) as caught:
            check_filler("XY", name="alternative filler")
        message = str(caught.exception)
        self.assertIn("alternative filler", message)
        self.assertIn("alternative=", message)

    def test_check_filler_rejects_a_letter_the_square_lacks(self) -> None:
        square = playfair_square("ORANGES", omit="Q")
        with self.assertRaises(ValueError) as caught:
            check_filler("Q", square=square)
        self.assertIn("not in the square", str(caught.exception))

    def test_every_entry_point_that_takes_a_filler_rejects_the_same_values(
        self,
    ) -> None:
        # The bug this guards: `--filler XY` was an error on one path and
        # silently discarded on another, so the operator could not tell
        # whether it had been understood.
        for value in self.BAD_FILLERS:
            for call, args in (
                (check_filler, (value,)),
                (prepare_digraphs, ("HELLO",)),
                (prepare_text, ("HELLO",)),
                (encrypt, ("HELLO", MONARCHY)),
            ):
                with self.subTest(value=value, call=call.__name__):
                    with self.assertRaises(ValueError):
                        if call is check_filler:
                            call(*args)
                        else:
                            call(*args, filler=value)

    def test_a_valid_filler_changes_the_ciphertext(self) -> None:
        # The other half of "never just ignored": a value that survives
        # validation has to do something.
        self.assertNotEqual(
            encrypt("BALLOON", MONARCHY, filler="Z"),
            encrypt("BALLOON", MONARCHY, filler="X"),
        )


class TestErrorsDoNotInventCommandLineFlags(unittest.TestCase):
    """Advice in an error message has to be true for whoever reads it.

    The J error used to say "try omit='Q'" and the filler error "pass
    alternative='Q'". Both are Python keyword arguments; a reader following
    them on the command line got "unrecognized arguments" instead.
    """

    def messages(self) -> list[str]:
        """Every rejection message that names a Python keyword argument."""
        found: list[str] = []
        for call, args, kwargs in (
            (playfair_square, ("MONARCHY",), {"omit": "JQ"}),
            (check_filler, ("XY",), {}),
            (check_filler, ("XY",), {"name": "alternative filler"}),
            (encrypt, ("HELLO", MONARCHY), {"filler": "X", "alternative": "X"}),
            (decrypt, ("ABJC", MONARCHY), {}),
            (
                playfair_square(MONARCHY).position,
                ("J",),
                {},
            ),
        ):
            with self.assertRaises(ValueError) as caught:
                call(*args, **kwargs)
            found.append(str(caught.exception))
        found.extend(validate_ciphertext("ABJC", plain_square()))
        return found

    def test_a_named_python_argument_is_labelled_as_one(self) -> None:
        for message in self.messages():
            with self.subTest(message=message[:60]):
                if not any(
                    token in message
                    for token in ("omit=", "filler=", "alternative=")
                ):
                    continue
                self.assertTrue(
                    "In Python" in message or "in Python" in message,
                    f"names a Python argument without saying so: {message}",
                )
                self.assertIn("library argument", message)

    def test_no_message_offers_a_flag_the_command_line_does_not_have(
        self,
    ) -> None:
        for message in self.messages():
            with self.subTest(message=message[:60]):
                for invented in ("--omit", "--alternative", "--fold"):
                    self.assertNotIn(invented, message)

    def test_advice_only_names_an_omission_the_ciphertext_allows(self) -> None:
        # Suggesting omit='Q' for a text that also contains a Q sends the
        # reader round a loop and implies the tool has an answer it lacks.
        allowed = validate_ciphertext("ABJC", plain_square())[0]
        self.assertIn("omit='Q'", allowed)

        blocked = validate_ciphertext("ABJQ", plain_square())[0]
        self.assertNotIn("omit='Q'", blocked)
        self.assertIn("not Playfair ciphertext", blocked)


class TestModuleIsOffline(unittest.TestCase):
    def test_no_networking_imports(self) -> None:
        source = (
            DATA_DIR.parent / "playfair.py"
        ).read_text(encoding="utf-8")
        for banned in (
            "import socket",
            "import urllib",
            "import http",
            "import requests",
            "subprocess",
            "webbrowser",
        ):
            self.assertNotIn(banned, source)


if __name__ == "__main__":
    unittest.main()
