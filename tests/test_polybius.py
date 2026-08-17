"""Tests for the Polybius square module.

The hand-computed cases below all use the standard square::

        1  2  3  4  5
     1  A  B  C  D  E
     2  F  G  H  I  K
     3  L  M  N  O  P
     4  Q  R  S  T  U
     5  V  W  X  Y  Z
"""

from __future__ import annotations

import unittest
from pathlib import Path

from cipher_tool.normalize import group_text, letters_only, normalize
from cipher_tool.polybius import (
    ADFGX_LABELS,
    LETTERS_AND_DIGITS,
    LETTERS_NO_J,
    LETTERS_NO_Q,
    PolybiusSquare,
    decrypt,
    encrypt,
    keyed_alphabet,
    solve,
    solve_unknown_square,
)
from cipher_tool.scoring import default_scorer

CORPUS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "cipher_tool"
    / "data"
    / "corpus_04_expository.txt"
)


def sample_text(count: int = 400) -> str:
    """A run of real English long enough for statistical attacks to work."""
    return letters_only(CORPUS.read_text(encoding="utf-8"))[:count]


class TestSquareConstruction(unittest.TestCase):
    def test_standard_grid_is_the_classical_layout(self) -> None:
        square = PolybiusSquare.standard()
        self.assertEqual(
            square.rows, ("ABCDE", "FGHIK", "LMNOP", "QRSTU", "VWXYZ")
        )
        self.assertEqual(square.size, 5)
        self.assertNotIn("J", square.symbols)
        self.assertTrue(square.is_lossy)

    def test_keyed_alphabet_is_hand_computable(self) -> None:
        # MONARCHY has no repeated letter; the rest of the 25-letter
        # alphabet follows in order, skipping the eight already used.
        self.assertEqual(
            keyed_alphabet("MONARCHY", LETTERS_NO_J),
            "MONARCHYBDEFGIKLPQSTUVWXZ",
        )
        square = PolybiusSquare.standard("MONARCHY")
        self.assertEqual(
            square.rows, ("MONAR", "CHYBD", "EFGIK", "LPQST", "UVWXZ")
        )

    def test_repeated_keyword_letters_are_used_once(self) -> None:
        self.assertEqual(
            keyed_alphabet("BEEKEEPER", LETTERS_NO_J)[:5], "BEKPR"
        )

    def test_keyword_j_folds_onto_i_before_the_square_is_built(self) -> None:
        square = PolybiusSquare.standard("JULY")
        self.assertEqual(square.symbols[:4], "IULY")

    def test_drop_q_square_keeps_i_and_j_apart(self) -> None:
        square = PolybiusSquare.without_q()
        self.assertEqual(square.symbols, LETTERS_NO_Q)
        self.assertIn("J", square.symbols)
        self.assertNotIn("Q", square.symbols)
        self.assertFalse(square.is_lossy)

    def test_six_by_six_holds_every_letter_and_digit(self) -> None:
        square = PolybiusSquare.six_by_six()
        self.assertEqual(square.size, 6)
        self.assertEqual(square.symbols, LETTERS_AND_DIGITS)
        self.assertFalse(square.is_lossy)

    def test_coordinates_and_letter_are_inverse(self) -> None:
        square = PolybiusSquare.standard()
        self.assertEqual(square.coordinates("A"), (0, 0))
        self.assertEqual(square.coordinates("K"), (1, 4))
        self.assertEqual(square.coordinates("Z"), (4, 4))
        self.assertEqual(square.letter(3, 0), "Q")
        self.assertEqual(square.letter(*square.coordinates("T")), "T")

    def test_merged_j_reports_the_cell_of_i(self) -> None:
        square = PolybiusSquare.standard()
        self.assertEqual(square.coordinates("J"), square.coordinates("I"))
        # And the loss is not recoverable: the cell decodes to I.
        self.assertEqual(square.letter(*square.coordinates("J")), "I")

    def test_transposed_square_swaps_rows_and_columns(self) -> None:
        square = PolybiusSquare.standard()
        flipped = square.transposed()
        self.assertEqual(flipped.rows[0], "AFLQV")
        self.assertEqual(flipped.coordinates("F"), (0, 1))
        self.assertEqual(square.coordinates("F"), (1, 0))

    def test_render_shows_the_grid_and_the_merge_warning(self) -> None:
        text = PolybiusSquare.standard().render()
        self.assertIn("A B C D E", text)
        self.assertIn("J=I", text)


class TestEncodeDecode(unittest.TestCase):
    def test_known_encoding_by_hand(self) -> None:
        # A=(1,1) T=(4,4) T=(4,4) A=(1,1) C=(1,3) K=(2,5)
        self.assertEqual(encrypt("ATTACK"), "114444111325")
        self.assertEqual(decrypt("114444111325"), "ATTACK")

    def test_known_encoding_with_adfgx_labels(self) -> None:
        square = PolybiusSquare.adfgx()
        self.assertEqual(square.row_labels, ADFGX_LABELS)
        self.assertEqual(square.encode("ATTACK"), "AAGGGGAAAFDX")
        self.assertEqual(square.decode("AAGGGGAAAFDX"), "ATTACK")

    def test_known_encoding_with_a_keyed_square(self) -> None:
        # MONARCHY square: H is row 2 column 2, I is row 3 column 4.
        square = PolybiusSquare.standard("MONARCHY")
        self.assertEqual(square.encode("HI"), "2234")

    def test_known_encoding_of_a_digit_in_the_six_by_six(self) -> None:
        # A is cell 0 -> (0,0) -> "11"; "1" is cell 27 -> (4,3) -> "54".
        self.assertEqual(PolybiusSquare.six_by_six().encode("A1"), "1154")

    def test_round_trip_over_several_squares(self) -> None:
        message = "MEETMEATTHEHARBOURATMIDNIGHT"
        squares = [
            PolybiusSquare.standard(),
            PolybiusSquare.standard("MONARCHY"),
            PolybiusSquare.without_q(),
            PolybiusSquare.without_q("SPHINX"),
            PolybiusSquare.six_by_six(),
            PolybiusSquare.adfgx("KEYWORD"),
            PolybiusSquare.adfgvx("CIPHER"),
        ]
        for square in squares:
            with self.subTest(square=square.name):
                self.assertEqual(square.decode(square.encode(message)), message)

    def test_round_trip_survives_the_whole_sample(self) -> None:
        square = PolybiusSquare.standard("HARBOUR")
        text = square.prepare(sample_text())
        self.assertEqual(square.decode(square.encode(text)), text)

    def test_layout_of_the_input_does_not_matter(self) -> None:
        clean = encrypt("ATTACKATDAWN")
        self.assertEqual(encrypt("attack at dawn"), clean)
        self.assertEqual(encrypt("Attack, at dawn!"), clean)
        self.assertEqual(encrypt("ATTAC KATDA WN"), clean)
        self.assertEqual(encrypt("ATTACKAT\nDAWN\n"), clean)

    def test_layout_of_the_ciphertext_does_not_matter(self) -> None:
        stream = encrypt("ATTACKATDAWN")
        self.assertEqual(decrypt(group_text(stream)), "ATTACKATDAWN")
        self.assertEqual(decrypt(stream + "\n\n"), "ATTACKATDAWN")
        self.assertEqual(decrypt("11 44 44 11 13 25"), "ATTACK")

    def test_merging_i_and_j_is_lossy_and_says_so(self) -> None:
        self.assertEqual(encrypt("JAM"), encrypt("IAM"))
        self.assertEqual(decrypt(encrypt("JAM")), "IAM")
        self.assertEqual(PolybiusSquare.standard().prepare("Jump!"), "IUMP")

    def test_empty_input_is_not_an_error(self) -> None:
        self.assertEqual(encrypt(""), "")
        self.assertEqual(decrypt(""), "")
        self.assertEqual(encrypt("!!! ..."), "")
        self.assertEqual(len(solve("")), 0)


class TestInvalidInput(unittest.TestCase):
    def assertRaisesWithMessage(self, callable_object) -> None:
        """Every refusal must explain itself, not just fail."""
        with self.assertRaises(ValueError) as caught:
            callable_object()
        self.assertTrue(str(caught.exception).strip())

    def test_odd_number_of_symbols_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            decrypt("1144441")
        message = str(caught.exception)
        self.assertIn("even", message)
        self.assertIn("7", message)

    def test_symbol_outside_the_labels_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            decrypt("1147")
        self.assertIn("'7'", str(caught.exception))
        # ...but the solver's permissive mode skips it deliberately.
        self.assertEqual(PolybiusSquare.standard().decode("11X44", strict=False), "AT")

    def test_grid_must_be_square_and_unique(self) -> None:
        self.assertRaisesWithMessage(lambda: PolybiusSquare("ABCDEFGHIJKL"))
        self.assertRaisesWithMessage(lambda: PolybiusSquare("AABBCCDDE"))

    def test_labels_must_match_the_grid_and_be_distinct(self) -> None:
        self.assertRaisesWithMessage(
            lambda: PolybiusSquare(LETTERS_NO_J, row_labels="1234")
        )
        self.assertRaisesWithMessage(
            lambda: PolybiusSquare(LETTERS_NO_J, row_labels="11234")
        )

    def test_merges_must_point_at_a_cell_that_exists(self) -> None:
        self.assertRaisesWithMessage(
            lambda: PolybiusSquare(LETTERS_NO_J, merges={"J": "J"})
        )
        self.assertRaisesWithMessage(
            lambda: PolybiusSquare(LETTERS_NO_J, merges={"A": "B"})
        )

    def test_keyword_without_usable_letters_is_rejected(self) -> None:
        self.assertRaisesWithMessage(lambda: PolybiusSquare.standard("1234"))
        self.assertRaisesWithMessage(lambda: PolybiusSquare.standard("  -- "))

    def test_letter_the_square_cannot_hold_is_refused_not_dropped(self) -> None:
        square = PolybiusSquare.without_q()
        with self.assertRaises(ValueError) as caught:
            square.encode("QUEEN")
        self.assertIn("Q", str(caught.exception))
        # The documented workaround, and the alternative convention.
        self.assertEqual(len(square.encode("KWEEN")), 10)
        merged = PolybiusSquare.without_q(merge_q_into="K")
        self.assertEqual(merged.encode("QUEEN"), merged.encode("KUEEN"))

    def test_coordinates_and_letter_reject_nonsense(self) -> None:
        square = PolybiusSquare.standard()
        self.assertRaisesWithMessage(lambda: square.coordinates("!"))
        self.assertRaisesWithMessage(lambda: square.coordinates("AB"))
        self.assertRaisesWithMessage(lambda: square.letter(5, 0))
        self.assertRaisesWithMessage(lambda: square.letter(0, -1))

    def test_bad_square_argument_is_rejected(self) -> None:
        self.assertRaisesWithMessage(lambda: encrypt("ATTACK", 5))

    def test_unusable_label_set_is_rejected(self) -> None:
        self.assertRaisesWithMessage(lambda: solve("1122", label_sets=["1"]))
        self.assertRaisesWithMessage(lambda: solve("1122", label_sets=["11234"]))

    def test_unusable_keyword_is_rejected_by_the_solver(self) -> None:
        self.assertRaisesWithMessage(lambda: solve("1122", keywords=["----"]))


class TestSolve(unittest.TestCase):
    def test_recovers_the_standard_square_from_a_long_message(self) -> None:
        square = PolybiusSquare.standard()
        expected = square.prepare(sample_text())
        result = solve(square.encode(expected))
        best = result.best()
        self.assertIsNotNone(best)
        self.assertEqual(best.plaintext, expected)
        self.assertEqual(best.confidence(), "strong")
        self.assertIn("labels=12345", best.key)

    def test_infers_adfgx_labels_from_the_ciphertext(self) -> None:
        square = PolybiusSquare.adfgx()
        expected = square.prepare(sample_text())
        best = solve(square.encode(expected)).best()
        self.assertEqual(best.plaintext, expected)
        self.assertEqual(best.diagnostics["labels"], "ADFGX")

    def test_recovers_a_six_by_six_square(self) -> None:
        square = PolybiusSquare.six_by_six()
        expected = square.prepare(sample_text())
        best = solve(square.encode(expected)).best()
        self.assertEqual(best.plaintext, expected)

    def test_recovers_a_keyed_square_when_the_keyword_is_offered(self) -> None:
        square = PolybiusSquare.standard("MONARCHY")
        expected = square.prepare(sample_text())
        result = solve(square.encode(expected), keywords=["ORCHESTRA", "MONARCHY"])
        best = result.best()
        self.assertEqual(best.plaintext, expected)
        self.assertIn("MONARCHY", best.key)
        self.assertEqual(best.confidence(), "strong")

    def test_a_column_major_square_is_found_by_the_transpose(self) -> None:
        # Writing the alphabet down the columns instead of along the rows is
        # the same square transposed, and it is a real convention.
        square = PolybiusSquare.standard().transposed()
        self.assertEqual(square.rows[0], "AFLQV")
        expected = square.prepare(sample_text())
        best = solve(square.encode(expected)).best()
        self.assertEqual(best.plaintext, expected)
        self.assertIn("transposed", best.key)
        # Switching the transpose off must lose it, not quietly find it anyway.
        limited = solve(square.encode(expected), try_transpose=False).best()
        self.assertNotEqual(limited.plaintext, expected)
        self.assertNotEqual(limited.confidence(), "strong")

    def test_label_permutations_find_labels_written_out_of_order(self) -> None:
        square = PolybiusSquare.standard(row_labels="53421")
        expected = square.prepare(sample_text(150))
        stream = square.encode(expected)
        # The default search assumes the labels are in their natural order.
        self.assertNotEqual(
            solve(stream, label_sets=["12345"]).best().plaintext, expected
        )
        best = solve(
            stream, label_sets=["12345"], label_permutations=True
        ).best()
        self.assertEqual(best.plaintext, expected)
        self.assertEqual(best.diagnostics["labels"], "53421")

    def test_forced_label_sets_are_obeyed_even_when_they_are_wrong(self) -> None:
        stream = PolybiusSquare.standard().encode(sample_text())
        self.assertEqual(solve(stream).best().diagnostics["labels"], "12345")
        # Told the labels are ADFGX, the solver must not quietly fall back on
        # what it can see: no symbol of the text is a label, so there is
        # nothing to decode and it says so by returning nothing.
        self.assertEqual(len(solve(stream, label_sets=["ADFGX"])), 0)

    def test_accepts_a_normalized_text_and_reads_its_original(self) -> None:
        # normalize() throws digits away, so a numeric Polybius stream has no
        # letters at all. The solver must work from .original or find nothing.
        square = PolybiusSquare.standard()
        expected = square.prepare(sample_text())
        source = normalize(group_text(square.encode(expected)))
        self.assertEqual(source.letters, "")
        self.assertEqual(solve(source).best().plaintext, expected)

    def test_top_limits_the_returned_set(self) -> None:
        stream = PolybiusSquare.standard().encode(sample_text())
        self.assertLessEqual(len(solve(stream, top=2)), 2)
        self.assertGreater(len(solve(stream, top=0)), 2)

    def test_diagnostics_record_the_search(self) -> None:
        stream = PolybiusSquare.standard().encode(sample_text())
        best = solve(stream).best()
        for field in ("square", "labels", "squares_tried", "word_coverage"):
            self.assertIn(field, best.diagnostics)

    def test_time_budget_is_honoured(self) -> None:
        stream = PolybiusSquare.standard().encode(sample_text())
        result = solve(stream, label_permutations=True, time_budget=0.0)
        self.assertGreater(len(result), 0)
        for candidate in result:
            self.assertTrue(candidate.diagnostics.get("time_budget_hit"))


class TestFailureModes(unittest.TestCase):
    """What the tool does when it is NOT looking at a Polybius stream."""

    def test_ordinary_english_produces_no_candidates_at_all(self) -> None:
        # Twenty-six distinct letters cannot be the labels of a 5x5 or 6x6
        # square, so there is nothing to decode. Reporting nothing is the
        # honest answer; inventing a decode would not be.
        self.assertEqual(len(solve(sample_text())), 0)

    def test_keyed_square_without_its_keyword_is_not_reported_as_solved(self) -> None:
        square = PolybiusSquare.standard("MONARCHY")
        truth = square.prepare(sample_text())
        result = solve(square.encode(truth))
        best = result.best()
        self.assertIsNotNone(best)
        self.assertNotEqual(best.plaintext, truth)
        self.assertNotEqual(best.confidence(), "strong")
        self.assertIn(best.confidence(), ("weak", "unlikely"))

    def test_odd_length_stream_is_flagged_rather_than_quietly_shifted(self) -> None:
        square = PolybiusSquare.standard()
        stream = square.encode(sample_text())[:-1]
        best = solve(stream).best()
        self.assertIn("odd_symbol_count", best.diagnostics)


class TestUnknownSquare(unittest.TestCase):
    """Recovering a KEYED square nobody supplied.

    Measured before this existed: `solve` on a keyed Polybius message with no
    keyword to hand returned `weak` and wrong, and handed the keyword it
    returned `strong` and right. Honest, and still a hole, because a
    competition does not give you the keyword.

    No hill climb is needed here, unlike Bifid. A Polybius stream is a
    monoalphabetic substitution written two symbols at a time, so mapping
    each distinct cell to a letter turns it into an ordinary substitution
    cipher that this toolkit already solves -- the same joint the ADFGVX
    attack cuts at. Measured: 0.9 seconds.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = default_scorer()
        cls.plaintext = PolybiusSquare.standard().prepare(sample_text(400))
        cls.ciphertext = encrypt(
            cls.plaintext, PolybiusSquare.standard("TEMPEST")
        )

    def test_it_recovers_a_keyed_square_nobody_supplied(self) -> None:
        found = solve_unknown_square(
            self.ciphertext, scorer=self.scorer, top=1, seed=1,
        )
        self.assertEqual(found.best().plaintext, self.plaintext)

    def test_it_reports_the_cells_so_the_answer_can_be_checked(self) -> None:
        best = solve_unknown_square(
            self.ciphertext, scorer=self.scorer, top=1, seed=1,
        ).best()
        self.assertIn("square", best.diagnostics)
        self.assertIn("=", best.diagnostics["square"])

    def test_letter_text_is_refused_rather_than_guessed_at(self) -> None:
        """The attack is meaningless on something that is not a stream."""
        found = solve_unknown_square(
            sample_text(200), scorer=self.scorer, top=1,
        )
        self.assertEqual(len(found), 0)

    def test_an_odd_number_of_symbols_is_refused(self) -> None:
        found = solve_unknown_square(
            self.ciphertext[:-1], scorer=self.scorer, top=1,
        )
        self.assertEqual(len(found), 0)

    def test_empty_input_is_not_a_crash(self) -> None:
        self.assertEqual(len(solve_unknown_square("", scorer=self.scorer)), 0)

    def test_the_pipeline_reaches_it(self) -> None:
        """A solver nothing calls is decoration."""
        from cipher_tool.auto import build_stages

        names = [stage.name for stage in build_stages("fast", 5, 1)]
        self.assertIn("Polybius (unknown square)", names)


if __name__ == "__main__":
    unittest.main()
