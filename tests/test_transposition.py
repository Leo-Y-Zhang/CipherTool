"""Tests for route/grid transposition and the transposition dispatcher.

Every "known pair" below was worked out on paper before the module existed.
The worked grid is ATTACKATDAWN written row by row into three rows of four:

        col   0 1 2 3
    row 0     A T T A
    row 1     C K A T
    row 2     D A W N

    columns               ACD TKA TAW ATN     -> ACDTKATAWATN
    boustrophedon rows    ATTA TAKC DAWN      -> ATTATAKCDAWN
    reverse                                   -> NWADTAKCATTA
    clockwise spiral      A T T A (top edge)
                          T N     (right edge, down)
                          W A D   (bottom edge, leftwards)
                          C       (left edge, up)
                          K A     (the middle row that is left)
                                              -> ATTATNWADCKA
    anti-diagonals        A | TC | TKD | AAA | TW | N
      (row+col constant,  -> ATCTKDAAATWN
       read downwards)

A ragged case, MEETMEATNO (ten letters) in three rows of four:

    row 0     M E E T
    row 1     M E A T
    row 2     N O . .

    columns   MMN EEO EA TT        -> MMNEEOEATT

which is exactly what a columnar transposition with an unshuffled key does,
and that agreement is asserted below against columnar.encrypt.
"""

from __future__ import annotations

import unittest

from cipher_tool import columnar, rail_fence, vigenere
from cipher_tool.normalize import letters_only
from cipher_tool.scoring import DATA_DIR, corpus_files, default_scorer
from cipher_tool.transposition import (
    FILL_ROUTES,
    METHOD,
    ROUTE_NAMES,
    ROUTES,
    decrypt,
    describe_routes,
    encrypt,
    grid_shapes,
    is_ragged,
    solve,
    solve_all,
    worst_window_score,
)

CORPUS = (DATA_DIR / "corpus_04_expository.txt").read_text(encoding="utf-8")

#: Long enough that the English model can tell the right answer from a near
#: miss. Statistical attacks on shorter texts are honestly unreliable.
SAMPLE = letters_only(CORPUS)[:360]


def scorer():
    """The shared scorer, so the suite builds the language model once."""
    return default_scorer()


class TestRoutesAreWellFormed(unittest.TestCase):
    """A route must visit every cell of the grid exactly once.

    This is the property everything else rests on: a route that skipped or
    repeated a cell would not corrupt the code visibly, it would produce a
    plausible-looking wrong plaintext.
    """

    def test_every_route_is_a_permutation_of_the_grid(self) -> None:
        for name in ROUTE_NAMES:
            route = ROUTES[name]
            for rows in range(1, 8):
                for cols in range(1, 8):
                    with self.subTest(route=name, rows=rows, cols=cols):
                        cells = route.cells(rows, cols)
                        self.assertEqual(len(cells), rows * cols)
                        self.assertEqual(len(set(cells)), rows * cols)
                        for row, col in cells:
                            self.assertTrue(0 <= row < rows)
                            self.assertTrue(0 <= col < cols)

    def test_there_are_eight_spirals(self) -> None:
        spirals = [name for name in ROUTE_NAMES if name.startswith("spiral")]
        self.assertEqual(len(spirals), 8, spirals)
        for corner in ("top_left", "top_right", "bottom_right", "bottom_left"):
            self.assertIn(f"spiral_cw_{corner}", ROUTES)
            self.assertIn(f"spiral_acw_{corner}", ROUTES)

    def test_the_spirals_really_differ(self) -> None:
        """Eight names would be worthless if they described one walk."""
        walks = {
            tuple(ROUTES[name].cells(4, 5))
            for name in ROUTE_NAMES
            if name.startswith("spiral")
        }
        self.assertEqual(len(walks), 8)

    def test_each_spiral_starts_at_its_corner_and_turns_its_way(self) -> None:
        """The named corner and sense must be the ones actually walked.

        Checked on the first two cells (which fix the corner and the direction
        of travel) and on the outer ring (which must be finished before the
        walk moves inwards). Those two together are what "clockwise spiral
        from the top right" means; the rest of the walk follows.
        """
        rows, cols = 4, 5
        expected_second = {
            "spiral_cw_top_left": (0, 1),
            "spiral_acw_top_left": (1, 0),
            "spiral_cw_top_right": (1, cols - 1),
            "spiral_acw_top_right": (0, cols - 2),
            "spiral_cw_bottom_right": (rows - 1, cols - 2),
            "spiral_acw_bottom_right": (rows - 2, cols - 1),
            "spiral_cw_bottom_left": (rows - 2, 0),
            "spiral_acw_bottom_left": (rows - 1, 1),
        }
        corners = {
            "top_left": (0, 0),
            "top_right": (0, cols - 1),
            "bottom_right": (rows - 1, cols - 1),
            "bottom_left": (rows - 1, 0),
        }
        ring = {
            (row, col)
            for row in range(rows)
            for col in range(cols)
            if row in (0, rows - 1) or col in (0, cols - 1)
        }
        for name, second in expected_second.items():
            with self.subTest(route=name):
                cells = ROUTES[name].cells(rows, cols)
                corner = name.split("_", 2)[2]
                self.assertEqual(cells[0], corners[corner])
                self.assertEqual(cells[1], second)
                self.assertEqual(set(cells[: len(ring)]), ring)

    def test_only_a_row_fill_is_declared_safe_on_a_ragged_grid(self) -> None:
        safe = [name for name in ROUTE_NAMES if ROUTES[name].ragged_safe]
        self.assertEqual(safe, ["rows"])

    def test_route_read_inverts_route_write(self) -> None:
        for name in ROUTE_NAMES:
            with self.subTest(route=name):
                route = ROUTES[name]
                grid = route.write("ATTACKATDAWN", 3, 4)
                self.assertEqual(route.read(grid), "ATTACKATDAWN")
                # A short text leaves blanks, which read() steps over.
                grid = route.write("ATTACKAT", 3, 4)
                self.assertEqual(route.read(grid), "ATTACKAT")

    def test_describe_routes_names_every_route(self) -> None:
        text = describe_routes()
        for name in ROUTE_NAMES:
            self.assertIn(name, text)


class TestKnownPairs(unittest.TestCase):
    """Hand-computed plaintext/key/ciphertext triples (see module docstring)."""

    PLAIN = "ATTACKATDAWN"

    def test_column_read(self) -> None:
        self.assertEqual(encrypt(self.PLAIN, "columns", 3, 4), "ACDTKATAWATN")

    def test_boustrophedon_read(self) -> None:
        self.assertEqual(
            encrypt(self.PLAIN, "boustrophedon_rows", 3, 4), "ATTATAKCDAWN"
        )

    def test_boustrophedon_column_read(self) -> None:
        # col 0 down A C D, col 1 up A K T, col 2 down T A W, col 3 up N T A
        self.assertEqual(
            encrypt(self.PLAIN, "boustrophedon_columns", 3, 4), "ACDAKTTAWNTA"
        )

    def test_clockwise_spiral_from_the_top_left(self) -> None:
        self.assertEqual(
            encrypt(self.PLAIN, "spiral_cw_top_left", 3, 4), "ATTATNWADCKA"
        )

    def test_anticlockwise_spiral_from_the_bottom_right(self) -> None:
        self.assertEqual(
            encrypt(self.PLAIN, "spiral_acw_bottom_right", 3, 4), "NTATTACDAWAK"
        )

    def test_reverse(self) -> None:
        self.assertEqual(encrypt(self.PLAIN, "reverse", 3, 4), "NWADTAKCATTA")
        self.assertEqual(encrypt(self.PLAIN, "reverse", 3, 4), self.PLAIN[::-1])

    def test_diagonal_read(self) -> None:
        self.assertEqual(encrypt(self.PLAIN, "diagonals", 3, 4), "ATCTKDAAATWN")

    def test_alternating_diagonal_read(self) -> None:
        self.assertEqual(
            encrypt(self.PLAIN, "diagonals_alternating", 3, 4), "ACTTKDAAATWN"
        )

    def test_column_fill_read_back_by_rows(self) -> None:
        self.assertEqual(
            encrypt(self.PLAIN, "rows", 3, 4, fill="columns"), "AAAATCTWTKDN"
        )

    def test_a_row_fill_read_by_rows_changes_nothing(self) -> None:
        """Sanity check on the machinery, not a cipher anyone would use."""
        self.assertEqual(encrypt(self.PLAIN, "rows", 3, 4), self.PLAIN)

    def test_ragged_column_read_agrees_with_columnar(self) -> None:
        """Cross-check against the other module that owns this arithmetic."""
        ragged = "MEETMEATNO"
        self.assertEqual(encrypt(ragged, "columns", cols=4), "MMNEEOEATT")
        self.assertEqual(
            encrypt(ragged, "columns", cols=4),
            columnar.encrypt(ragged, (0, 1, 2, 3)),
        )

    def test_a_derived_dimension_matches_an_explicit_one(self) -> None:
        self.assertEqual(
            encrypt(self.PLAIN, "columns", cols=4),
            encrypt(self.PLAIN, "columns", 3, 4),
        )
        self.assertEqual(
            encrypt(self.PLAIN, "columns", 3),
            encrypt(self.PLAIN, "columns", 3, 4),
        )


class TestRoundTrip(unittest.TestCase):
    def test_decrypt_undoes_encrypt_for_every_route(self) -> None:
        text = letters_only(CORPUS)[:120]
        for read_name in ROUTE_NAMES:
            for fill_name in FILL_ROUTES:
                for cols in (5, 8, 12):
                    with self.subTest(read=read_name, fill=fill_name, cols=cols):
                        cipher = encrypt(text, read_name, cols=cols, fill=fill_name)
                        self.assertEqual(len(cipher), len(text))
                        self.assertEqual(
                            decrypt(cipher, read_name, cols=cols, fill=fill_name),
                            text,
                        )

    def test_round_trip_on_ragged_grids(self) -> None:
        text = letters_only(CORPUS)[:97]  # prime, so nothing fits exactly
        for read_name in ROUTE_NAMES:
            for cols in (6, 9, 13):
                with self.subTest(read=read_name, cols=cols):
                    self.assertTrue(is_ragged(len(text), -(-97 // cols), cols))
                    cipher = encrypt(text, read_name, cols=cols)
                    self.assertEqual(len(cipher), len(text))
                    self.assertEqual(decrypt(cipher, read_name, cols=cols), text)

    def test_a_transposition_never_changes_the_letters(self) -> None:
        """The family's defining property, and the reason chi-squared works."""
        text = letters_only(CORPUS)[:200]
        for read_name in ROUTE_NAMES:
            with self.subTest(read=read_name):
                cipher = encrypt(text, read_name, cols=10)
                self.assertEqual(sorted(cipher), sorted(text))


class TestInputRobustness(unittest.TestCase):
    CLEAN = "ATTACKATDAWN"
    EXPECTED = "ACDTKATAWATN"

    def test_lowercase(self) -> None:
        self.assertEqual(encrypt("attackatdawn", "columns", 3, 4), self.EXPECTED)

    def test_spaces_and_punctuation(self) -> None:
        self.assertEqual(
            encrypt("Attack at dawn!", "columns", 3, 4), self.EXPECTED
        )

    def test_five_letter_groups_and_line_breaks(self) -> None:
        self.assertEqual(
            encrypt("ATTAC KATDA\nWN\n", "columns", 3, 4), self.EXPECTED
        )

    def test_decrypt_is_equally_forgiving(self) -> None:
        self.assertEqual(
            decrypt("acdtk atawa tn!", "columns", 3, 4), self.CLEAN
        )

    def test_route_names_tolerate_spacing_and_case(self) -> None:
        self.assertEqual(
            encrypt(self.CLEAN, "Boustrophedon-Rows", 3, 4),
            encrypt(self.CLEAN, "boustrophedon_rows", 3, 4),
        )


class TestEmptyInput(unittest.TestCase):
    def test_encrypt_and_decrypt_return_empty(self) -> None:
        self.assertEqual(encrypt("", "columns", 3, 4), "")
        self.assertEqual(decrypt("", "columns", 3, 4), "")
        self.assertEqual(encrypt("!!! 123", "spiral_cw_top_left", 3, 4), "")

    def test_solve_returns_no_candidates(self) -> None:
        self.assertEqual(len(solve("")), 0)
        self.assertEqual(len(solve("ABC")), 0)  # too small for any grid

    def test_solve_all_returns_no_candidates(self) -> None:
        self.assertEqual(len(solve_all("")), 0)


class TestGridShapes(unittest.TestCase):
    def test_exact_rectangles_are_the_factorisations(self) -> None:
        self.assertEqual(grid_shapes(12), [(2, 6), (3, 4), (4, 3), (6, 2)])

    def test_a_prime_length_has_no_exact_rectangle(self) -> None:
        """An honest empty answer: no rectangle of 293 cells has two sides."""
        self.assertEqual(grid_shapes(293), [])
        self.assertNotEqual(grid_shapes(293, allow_ragged=True), [])

    def test_sides_are_bounded(self) -> None:
        for rows, cols in grid_shapes(360, min_side=3, max_side=20):
            self.assertTrue(3 <= rows <= 20)
            self.assertTrue(3 <= cols <= 20)
            self.assertEqual(rows * cols, 360)

    def test_ragged_shapes_never_leave_a_whole_empty_row(self) -> None:
        length = 97
        for rows, cols in grid_shapes(length, allow_ragged=True):
            with self.subTest(rows=rows, cols=cols):
                self.assertGreaterEqual(rows * cols, length)
                self.assertLess((rows - 1) * cols, length)

    def test_ragged_includes_the_exact_shapes(self) -> None:
        exact = set(grid_shapes(120))
        ragged = set(grid_shapes(120, allow_ragged=True))
        self.assertTrue(exact)
        self.assertTrue(exact <= ragged)


class TestInvalidKeys(unittest.TestCase):
    """Bad input must raise ValueError with something a human can act on."""

    def assert_raises_explained(self, call, *fragments: str) -> None:
        """Run *call*, require a ValueError whose message says something."""
        with self.assertRaises(ValueError) as caught:
            call()
        message = str(caught.exception)
        self.assertTrue(message.strip(), "the error message was empty")
        for fragment in fragments:
            self.assertIn(fragment, message)

    def test_unknown_route_lists_the_known_ones(self) -> None:
        self.assert_raises_explained(
            lambda: encrypt("ATTACKATDAWN", "corkscrew", 3, 4),
            "corkscrew", "rows",
        )

    def test_no_dimension_at_all(self) -> None:
        self.assert_raises_explained(
            lambda: encrypt("ATTACKATDAWN", "columns"), "row count"
        )

    def test_grid_too_small_for_the_text(self) -> None:
        self.assert_raises_explained(
            lambda: encrypt("ATTACKATDAWN", "columns", 2, 4), "fewer"
        )

    def test_grid_with_a_wholly_empty_last_row(self) -> None:
        self.assert_raises_explained(
            lambda: encrypt("ATTACKATDAWN", "columns", 5, 4), "empty"
        )

    def test_a_ragged_grid_refuses_an_ambiguous_fill(self) -> None:
        """The refusal that matters: it declines to guess a convention."""
        self.assert_raises_explained(
            lambda: encrypt("MEETMEATNO", "rows", 3, 4, fill="columns"),
            "empty", "row-by-row", "unambiguous",
        )

    def test_fractional_dimensions_are_rejected_not_rounded(self) -> None:
        self.assert_raises_explained(
            lambda: encrypt("ATTACKATDAWN", "columns", 3.5), "integer"
        )

    def test_bad_side_bounds(self) -> None:
        self.assert_raises_explained(
            lambda: grid_shapes(12, 5, 3), "max_side"
        )

    def test_unknown_solver_option(self) -> None:
        self.assert_raises_explained(
            lambda: solve("ATTACKATDAWN", nonsense=1), "nonsense"
        )

    def test_time_budget_must_be_positive(self) -> None:
        self.assert_raises_explained(
            lambda: solve("ATTACKATDAWN", time_budget=0), "time_budget"
        )
        self.assert_raises_explained(
            lambda: solve_all("ATTACKATDAWN", time_budget=-1), "time_budget"
        )

    def test_empty_route_list(self) -> None:
        self.assert_raises_explained(
            lambda: solve(SAMPLE, routes=[]), "at least one"
        )


class TestSolver(unittest.TestCase):
    """The attack must recover the plaintext, not merely run."""

    def check_recovery(self, plain: str, read: str, cols: int,
                       fill: str = "rows") -> None:
        """Encipher *plain*, solve, and require the plaintext back."""
        cipher = encrypt(plain, read, cols=cols, fill=fill)
        found = solve(cipher, scorer=scorer(), top=5)
        best = found.best()
        self.assertIsNotNone(best)
        self.assertEqual(
            best.plaintext, plain,
            f"top candidate was {best.key}: {best.preview(60)}",
        )
        self.assertEqual(best.method, METHOD)
        self.assertEqual(best.confidence(), "strong")

    def test_recovers_a_spiral(self) -> None:
        self.check_recovery(SAMPLE, "spiral_cw_top_left", 15)

    def test_recovers_an_alternating_diagonal(self) -> None:
        self.check_recovery(SAMPLE, "diagonals_alternating", 12)

    def test_recovers_a_boustrophedon(self) -> None:
        self.check_recovery(SAMPLE, "boustrophedon_columns", 18)

    def test_recovers_a_ragged_grid(self) -> None:
        ragged = letters_only(CORPUS)[:317]  # prime: no exact rectangle exists
        self.check_recovery(ragged, "spiral_acw_bottom_right", 19)

    def test_recovers_a_fancy_fill_by_searching_both_directions(self) -> None:
        """A sender who wrote along a spiral and read along the rows.

        The four fill routes searched by default do not include a spiral, so
        this is only found because the search also tries every pair swapped:
        decrypting "filled along A, read along B" is the same permutation as
        encrypting "filled along B, read along A".
        """
        cipher = encrypt(SAMPLE, "rows", cols=15, fill="spiral_cw_top_left")
        found = solve(cipher, scorer=scorer(), top=5)
        self.assertEqual(found.best().plaintext, SAMPLE)

        limited = solve(cipher, scorer=scorer(), top=5, both_directions=False)
        self.assertNotEqual(limited.best().plaintext, SAMPLE)

    def test_diagnostics_name_the_grids_and_routes_tested(self) -> None:
        cipher = encrypt(SAMPLE, "diagonals", cols=15)
        best = solve(cipher, scorer=scorer(), top=3).best()
        evidence = best.diagnostics
        self.assertIn("24x15", evidence["grids_tested"])
        self.assertIn("diagonals", evidence["routes_tested"])
        self.assertIn("rows", evidence["fills_tested"])
        self.assertGreater(evidence["combinations_tested"], 100)
        self.assertEqual(evidence["length"], len(SAMPLE))
        self.assertEqual(evidence["grid"], "24 rows x 15 cols")
        self.assertEqual(evidence["read_route"], "diagonals")
        # A transposition leaves the letter frequencies alone, so the
        # ciphertext's own chi-squared is small. That is the family tell.
        self.assertLess(evidence["ciphertext_chi_squared"], 0.2)

    def test_ambiguous_ragged_fills_are_counted_not_hidden(self) -> None:
        cipher = encrypt(letters_only(CORPUS)[:317], "columns", cols=19)
        best = solve(cipher, scorer=scorer(), top=3).best()
        self.assertGreater(best.diagnostics["skipped_ambiguous_ragged_fills"], 0)

    def test_a_pinned_shape_searches_only_that_shape(self) -> None:
        cipher = encrypt(SAMPLE, "spiral_cw_top_left", cols=15)
        found = solve(cipher, scorer=scorer(), top=3, cols=15)
        self.assertEqual(found.best().plaintext, SAMPLE)
        self.assertEqual(
            found.best().diagnostics["grids_tested"].count(";"), 0
        )

    def test_time_budget_is_respected_and_reported(self) -> None:
        cipher = encrypt(letters_only(CORPUS)[:900], "spiral_cw_top_left",
                         cols=30)
        found = solve(cipher, scorer=scorer(), top=3, time_budget=0.05)
        for candidate in found.ranked():
            self.assertTrue(candidate.diagnostics.get("time_budget_hit"))

    def test_seed_is_accepted_and_the_search_is_deterministic(self) -> None:
        cipher = encrypt(SAMPLE, "diagonals", cols=15)
        first = solve(cipher, scorer=scorer(), top=3, seed=1)
        second = solve(cipher, scorer=scorer(), top=3, seed=999)
        self.assertEqual(
            [c.key for c in first.ranked()], [c.key for c in second.ranked()]
        )


class TestFailureModes(unittest.TestCase):
    """What the tool does when it cannot solve the text.

    A solver that always sounds confident is worse than useless in a
    competition, so these tests assert on the honesty of the report rather
    than on a correct answer.
    """

    def test_a_polyalphabetic_cipher_is_not_confidently_mis_solved(self) -> None:
        """Vigenere text is not a transposition and no route can rescue it.

        Its letter frequencies are flattened, so no rearrangement of them
        reads as English. The solver must say so rather than presenting its
        least-bad permutation as an answer.
        """
        cipher = vigenere.encrypt(SAMPLE, "LEMON")
        found = solve(cipher, scorer=scorer(), top=3)
        best = found.best()
        self.assertIsNotNone(best)
        self.assertIn(best.confidence(), {"weak", "unlikely"})
        self.assertLess(best.diagnostics["word_coverage"], 0.35)
        self.assertNotEqual(best.plaintext, SAMPLE)

    def test_a_route_outside_the_search_is_not_faked(self) -> None:
        """Ask for the wrong routes and the true plaintext must not appear."""
        cipher = encrypt(SAMPLE, "spiral_cw_top_left", cols=15)
        found = solve(
            cipher, scorer=scorer(), top=5,
            routes=["rows", "columns", "reverse"], both_directions=False,
        )
        self.assertNotIn(SAMPLE, [c.plaintext for c in found.ranked()])
        self.assertNotEqual(found.best().confidence(), "strong")
        # And the diagnostics say exactly what was looked at, so the gap is
        # visible rather than implied.
        self.assertEqual(
            found.best().diagnostics["routes_tested"], "rows,columns,reverse"
        )

    def test_block_shuffled_english_is_caught_by_the_window_score(self) -> None:
        """The near miss that the letter model alone cannot see.

        A route that is nearly right returns the plaintext in reordered
        blocks: real words throughout, a good overall score, and one bad join
        per block boundary. The overall per-letter score barely moves; the
        worst eight-letter window does. Measured distributions are in the
        module docstring -- the margin is real but not wide, which is why the
        assertion below is on the gap and not on an absolute threshold.
        """
        for size in (10, 15, 22, 30):
            with self.subTest(block=size):
                blocks = [SAMPLE[i:i + size] for i in range(0, len(SAMPLE), size)]
                shuffled = "".join(blocks[1::2] + blocks[0::2])
                self.assertNotEqual(shuffled, SAMPLE)
                # The overall score hardly notices: real words are intact.
                self.assertGreater(scorer().normalised(shuffled), -1.2)
                # The worst window does notice.
                self.assertLess(
                    worst_window_score(shuffled, scorer()),
                    worst_window_score(SAMPLE, scorer()) - 0.2,
                )

    def test_the_window_score_is_reported_as_evidence_not_used_for_ranking(
        self,
    ) -> None:
        cipher = encrypt(SAMPLE, "spiral_cw_top_left", cols=15)
        found = solve(cipher, scorer=scorer(), top=5)
        best = found.best()
        self.assertEqual(best.plaintext, SAMPLE)
        self.assertIn("worst_window_score", best.diagnostics)
        # The ranking is the scorer's, unchanged: candidates come back in
        # descending English score, whatever the window score says.
        ranked = found.ranked()
        self.assertEqual(
            [candidate.score for candidate in ranked],
            sorted((candidate.score for candidate in ranked), reverse=True),
        )
        self.assertAlmostEqual(
            best.diagnostics["worst_window_score"],
            worst_window_score(SAMPLE, scorer()),
        )

    def test_the_window_is_calibrated_to_the_documented_numbers(self) -> None:
        """Pin the default window to the table in the module docstring.

        The docstring quotes measured ranges for an eight-letter window, and
        those numbers are only true for that window: three letters drags real
        English down to about -1.8, twelve lifts it to about -1.0, and either
        change would leave the documented calibration quietly wrong. The
        bounds below sit outside the measured range of this fixed sample
        (-1.270 to -1.165) and inside the ranges every other window produces.

        Deterministic: fixed slices of the shipped corpus, no random choice.
        Re-measure if the corpus files change.
        """
        samples = []
        for path in corpus_files():
            letters = letters_only(path.read_text(encoding="utf-8"))
            for offset in (0, 1000, 2000):
                if len(letters) >= offset + 330:
                    samples.append(letters[offset:offset + 330])
        self.assertGreaterEqual(len(samples), 12)
        for index, text in enumerate(samples):
            with self.subTest(sample=index):
                worst = worst_window_score(text, scorer())
                self.assertGreater(worst, -1.35)
                self.assertLess(worst, -1.10)

    def test_the_window_score_survives_short_and_empty_text(self) -> None:
        self.assertEqual(worst_window_score("", scorer()), float("-inf"))
        self.assertLess(worst_window_score("THE", scorer()), 0.0)

    def test_the_wrong_grid_shape_gives_junk_not_a_plausible_lie(self) -> None:
        """The right route on the wrong shape still scores clearly worse.

        MEASURED on this sample: the true plaintext scores -0.74 per letter
        with 92 per cent word coverage, while the same spiral read out of a
        grid three columns too wide scores about -1.75 with 31 per cent. The
        near miss is not nothing -- a spiral's outer ring survives a small
        change of width -- which is exactly why the margin is asserted rather
        than assumed.
        """
        cipher = encrypt(SAMPLE, "spiral_cw_top_left", cols=15)
        truth = scorer().breakdown(SAMPLE)
        for cols in (13, 14, 16, 18):
            with self.subTest(cols=cols):
                wrong = decrypt(cipher, "spiral_cw_top_left", cols=cols)
                self.assertNotEqual(wrong, SAMPLE)
                report = scorer().breakdown(wrong)
                self.assertLess(
                    report.ngram_per_letter, truth.ngram_per_letter - 0.8
                )
                self.assertLess(report.word_coverage, truth.word_coverage / 2)


class TestDispatcher(unittest.TestCase):
    """solve_all must find whichever transposition family was actually used."""

    def check_family(self, cipher: str, method: str) -> None:
        """Solve *cipher* with the dispatcher and check the winning family."""
        found = solve_all(cipher, scorer=scorer(), top=5, max_key_length=6,
                          seed=1)
        best = found.best()
        self.assertIsNotNone(best)
        self.assertEqual(best.plaintext, SAMPLE, best.key)
        self.assertEqual(best.method, method)
        self.assertIn("rail fence", best.diagnostics["families_run"])
        self.assertIn("columnar", best.diagnostics["families_run"])
        self.assertIn("route/grid", best.diagnostics["families_run"])

    def test_finds_a_rail_fence(self) -> None:
        self.check_family(rail_fence.encrypt(SAMPLE, 6), "Rail fence")

    def test_finds_a_columnar(self) -> None:
        self.check_family(
            columnar.encrypt(SAMPLE, "ZEBRAS"), "Columnar transposition"
        )

    def test_finds_a_route(self) -> None:
        self.check_family(
            encrypt(SAMPLE, "spiral_cw_top_left", cols=15), METHOD
        )

    def test_all_three_families_contribute_candidates(self) -> None:
        """Nothing is dropped because another family scored better.

        ``top=0`` asks for the whole merged set, which is the only way to see
        the runners-up: with a small ``top`` the route search wins every slot
        on a route ciphertext, simply because it looked at far more
        arrangements and so has a better-looking pile of near misses.
        """
        found = solve_all(
            encrypt(SAMPLE, "diagonals", cols=15), scorer=scorer(), top=0,
            max_key_length=5, seed=1,
        )
        methods = {candidate.method for candidate in found.ranked()}
        self.assertEqual(
            methods, {"Rail fence", "Columnar transposition", METHOD}
        )
        self.assertEqual(found.best().plaintext, SAMPLE)

    def test_families_run_reports_the_real_contribution_of_each(self) -> None:
        """The counts in ``families_run`` must add up to the merged set.

        No family's candidates can collide with another's -- the candidate
        set keys on (method, plaintext) and each family has its own method --
        so with ``top=0``, which asks for everything, the merged size is
        exactly the sum of the three counts. That pins the note to what
        actually happened rather than to what was meant to happen.
        """
        found = solve_all(
            encrypt(SAMPLE, "diagonals", cols=15), scorer=scorer(), top=0,
            max_key_length=5, seed=1,
        )
        note = found.best().diagnostics["families_run"]
        counts = [int(part.split("(")[1].rstrip(")")) for part in note.split(", ")]
        self.assertEqual(len(counts), 3, note)
        self.assertEqual(sum(counts), len(found), note)
        self.assertTrue(all(count > 0 for count in counts), note)

    def test_a_family_that_found_nothing_says_so(self) -> None:
        """"Ran and found nothing" must not read the same as "was not run".

        The side bounds here admit no grid at all for this length, so the
        route search runs and legitimately returns nothing. The other two
        families are unaffected, and the note has to show the zero.
        """
        found = solve_all(
            encrypt(SAMPLE, "diagonals", cols=15), scorer=scorer(), top=0,
            max_key_length=5, seed=1, min_side=39, max_side=40,
        )
        note = found.best().diagnostics["families_run"]
        self.assertIn("route/grid(0)", note)
        self.assertNotIn("families_skipped_no_time", found.best().diagnostics)
        counts = [int(part.split("(")[1].rstrip(")")) for part in note.split(", ")]
        self.assertEqual(sum(counts), len(found), note)
        self.assertNotIn(
            METHOD, {candidate.method for candidate in found.ranked()}
        )

    def test_a_budget_too_small_to_search_returns_nothing_not_junk(self) -> None:
        found = solve_all(SAMPLE, scorer=scorer(), top=5, time_budget=1e-6)
        self.assertEqual(len(found), 0)

    def test_a_time_budget_is_shared_and_reported(self) -> None:
        found = solve_all(
            encrypt(SAMPLE, "diagonals", cols=15), scorer=scorer(), top=3,
            time_budget=0.3, seed=1,
        )
        self.assertTrue(found.ranked())
        self.assertTrue(
            any(c.diagnostics.get("time_budget_hit") for c in found.ranked())
        )

    def test_unknown_option_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            solve_all(SAMPLE, nonsense=True)


if __name__ == "__main__":
    unittest.main()
