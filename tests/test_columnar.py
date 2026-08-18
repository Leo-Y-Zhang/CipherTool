"""Tests for columnar transposition (ragged, complete and double).

The hand-computed cases were worked out on paper before the module existed.

Ragged, the classic six-column example::

    keyword  Z E B R A S      read order A(4) B(2) E(1) R(3) S(5) Z(0)
    column   0 1 2 3 4 5

             W E A R E D
             I S C O V E
             R E D F L E
             E A T O N C
             E                <- 25 letters, so the last row is short

    columns    0=WIREE 1=ESEA 2=ACDT 3=ROFO 4=EVLN 5=DEEC
    ciphertext EVLN ACDT ESEA ROFO DEEC WIREE

Ragged with a SHORT column read first, which is what catches an
implementation that slices the ciphertext into equal blocks::

    keyword  C B A D E      read order A(2) B(1) C(0) D(3) E(4)
    column   0 1 2 3 4

             A T T A C
             K A T D A
             W N              <- 12 letters, columns 0 and 1 are long

    columns    0=AKW 1=TAN 2=TT 3=AD 4=CA
    ciphertext TT TAN AKW AD CA
"""

from __future__ import annotations

import random
import time
import unittest
from dataclasses import replace

from cipher_tool.columnar import (
    column_lengths,
    decrypt,
    decrypt_double,
    encrypt,
    encrypt_double,
    grid_rows,
    key_order,
    keyword_from_order,
    plausible_column_counts,
    solve,
    solve_double,
)
from cipher_tool.normalize import group_text, letters_only, normalize
from cipher_tool.scoring import DATA_DIR, default_scorer

CORPUS = (DATA_DIR / "corpus_04_expository.txt").read_text(encoding="utf-8")
CLASSIC = "WEAREDISCOVEREDFLEEATONCE"


class TestKeyOrder(unittest.TestCase):
    def test_known_permutation(self) -> None:
        self.assertEqual(key_order("ZEBRAS"), (4, 2, 1, 3, 5, 0))

    def test_repeated_letters_break_ties_left_to_right(self) -> None:
        # The three A's are read in the order they appear in the keyword.
        self.assertEqual(key_order("BANANA"), (1, 3, 5, 0, 2, 4))
        self.assertEqual(key_order("AA"), (0, 1))

    def test_key_is_cleaned_like_ciphertext(self) -> None:
        self.assertEqual(key_order("Zebra's!"), key_order("ZEBRAS"))

    def test_keyword_from_order_round_trips(self) -> None:
        for keyword in ("ZEBRAS", "BANANA", "CAB", "MONARCHY", "DR"):
            with self.subTest(keyword=keyword):
                order = key_order(keyword)
                self.assertEqual(key_order(keyword_from_order(order)), order)
        self.assertEqual(keyword_from_order((4, 2, 1, 3, 5, 0)), "FCBDAE")


class TestGridArithmetic(unittest.TestCase):
    def test_column_lengths_ragged(self) -> None:
        # 25 letters over 6 columns: five rows, only the first column full.
        self.assertEqual(column_lengths(25, 6), [5, 4, 4, 4, 4, 4])
        self.assertEqual(column_lengths(12, 5), [3, 3, 2, 2, 2])
        self.assertEqual(grid_rows(25, 6), 5)

    def test_column_lengths_exact(self) -> None:
        self.assertEqual(column_lengths(24, 6), [4, 4, 4, 4, 4, 4])

    def test_column_lengths_total_the_text(self) -> None:
        for length in range(1, 60):
            for count in range(2, 10):
                with self.subTest(length=length, count=count):
                    self.assertEqual(sum(column_lengths(length, count)), length)

    def test_plausible_column_counts(self) -> None:
        self.assertEqual(plausible_column_counts(24), [2, 3, 4, 6, 8, 12, 24])
        self.assertEqual(plausible_column_counts(25), [5, 25])


class TestKnownPairs(unittest.TestCase):
    def test_known_pair(self) -> None:
        self.assertEqual(encrypt(CLASSIC, "ZEBRAS"), "EVLNACDTESEAROFODEECWIREE")

    def test_known_pair_decrypts(self) -> None:
        self.assertEqual(decrypt("EVLNACDTESEAROFODEECWIREE", "ZEBRAS"), CLASSIC)

    def test_known_pair_short_column_read_first(self) -> None:
        # The classic ragged bug: the first ciphertext block here is only two
        # letters long, so equal-sized slicing gets every column wrong.
        self.assertEqual(encrypt("ATTACKATDAWN", "CBADE"), "TTTANAKWADCA")
        self.assertEqual(decrypt("TTTANAKWADCA", "CBADE"), "ATTACKATDAWN")

    def test_known_pair_exact_rectangle(self) -> None:
        self.assertEqual(encrypt("ATTACKATDAWN", "CAB"), "TCTWTKDNAAAA")

    def test_known_pair_complete(self) -> None:
        # Padding 25 letters to 30 gives every column five letters.
        self.assertEqual(
            encrypt(CLASSIC, "ZEBRAS", complete=True),
            "EVLNXACDTXESEAXROFOXDEECXWIREE",
        )
        self.assertEqual(
            decrypt(
                "EVLNXACDTXESEAXROFOXDEECXWIREE",
                "ZEBRAS",
                complete=True,
                strip_filler=True,
            ),
            CLASSIC,
        )

    def test_explicit_permutation_matches_the_keyword(self) -> None:
        self.assertEqual(
            encrypt(CLASSIC, (4, 2, 1, 3, 5, 0)), encrypt(CLASSIC, "ZEBRAS")
        )


class TestRoundTrip(unittest.TestCase):
    def test_round_trip_many_keys_and_lengths(self) -> None:
        source = letters_only(CORPUS)
        for keyword in ("DR", "CAB", "ZEBRAS", "MONARCHY", "BANANA", "WATERFALL"):
            for length in (23, 24, 25, 61, 100, 137):
                plaintext = source[:length]
                with self.subTest(keyword=keyword, length=length):
                    self.assertEqual(
                        decrypt(encrypt(plaintext, keyword), keyword), plaintext
                    )

    def test_round_trip_complete(self) -> None:
        plaintext = letters_only(CORPUS)[:137]
        ciphertext = encrypt(plaintext, "ZEBRAS", complete=True)
        self.assertEqual(len(ciphertext) % 6, 0)
        recovered = decrypt(ciphertext, "ZEBRAS", complete=True)
        self.assertTrue(recovered.startswith(plaintext))
        self.assertEqual(recovered[len(plaintext) :], "X")
        self.assertEqual(
            decrypt(ciphertext, "ZEBRAS", complete=True, strip_filler=True),
            plaintext,
        )

    def test_round_trip_double(self) -> None:
        plaintext = letters_only(CORPUS)[:137]
        for first, second in (("ZEBRAS", "CAB"), ("DR", "MONARCHY")):
            with self.subTest(first=first, second=second):
                ciphertext = encrypt_double(plaintext, first, second)
                self.assertEqual(
                    decrypt_double(ciphertext, first, second), plaintext
                )

    def test_round_trip_double_complete(self) -> None:
        plaintext = letters_only(CORPUS)[:137]
        ciphertext = encrypt_double(
            plaintext, "ZEBRAS", "CABD", complete=True
        )
        # One padding to a multiple of lcm(6, 4) = 12 keeps both grids full.
        self.assertEqual(len(ciphertext) % 12, 0)
        self.assertEqual(
            decrypt_double(
                ciphertext, "ZEBRAS", "CABD", complete=True, strip_filler=True
            ),
            plaintext,
        )

    def test_double_is_not_the_same_as_single(self) -> None:
        plaintext = letters_only(CORPUS)[:100]
        self.assertNotEqual(
            encrypt_double(plaintext, "ZEBRAS", "CAB"),
            encrypt(encrypt(plaintext, "ZEBRAS"), "ZEBRAS"),
        )

    def test_encryption_only_moves_letters(self) -> None:
        plaintext = letters_only(CORPUS)[:137]
        self.assertEqual(sorted(encrypt(plaintext, "ZEBRAS")), sorted(plaintext))


class TestInputRobustness(unittest.TestCase):
    def test_layout_does_not_change_the_result(self) -> None:
        clean = encrypt("ATTACKATDAWN", "CBADE")
        for variant in (
            "attackatdawn",
            "Attack at dawn!",
            "ATTAC KATDA WN",
            "ATTAC\nKATDA\nWN",
            "  a t t a c k , a t   d a w n .  ",
        ):
            with self.subTest(variant=variant):
                self.assertEqual(encrypt(variant, "CBADE"), clean)

    def test_decrypt_ignores_layout(self) -> None:
        self.assertEqual(decrypt("tttan akwad ca", "CBADE"), "ATTACKATDAWN")


class TestEmptyInput(unittest.TestCase):
    def test_empty_encrypt_and_decrypt(self) -> None:
        self.assertEqual(encrypt("", "ZEBRAS"), "")
        self.assertEqual(decrypt("", "ZEBRAS"), "")
        self.assertEqual(encrypt("!!!", "ZEBRAS", complete=True), "")
        self.assertEqual(encrypt_double("", "ZEBRAS", "CAB"), "")

    def test_empty_solve(self) -> None:
        self.assertEqual(len(solve("", scorer=default_scorer())), 0)
        self.assertEqual(len(solve("ABC", scorer=default_scorer())), 0)


class TestInvalidKeys(unittest.TestCase):
    def _assert_explains(self, call, *args, **kwargs) -> str:
        with self.assertRaises(ValueError) as caught:
            call(*args, **kwargs)
        message = str(caught.exception)
        self.assertTrue(message.strip(), "ValueError must carry an explanation")
        return message

    def test_key_too_short(self) -> None:
        for key in ("A", "", "!!", "7"):
            with self.subTest(key=key):
                message = self._assert_explains(encrypt, CLASSIC, key)
                self.assertIn("least two", message)

    def test_permutation_must_be_a_permutation(self) -> None:
        message = self._assert_explains(encrypt, CLASSIC, (0, 1, 3))
        self.assertIn("permutation", message)
        self._assert_explains(encrypt, CLASSIC, (0, 0, 1))
        self._assert_explains(encrypt, CLASSIC, (1, 2, 3))

    def test_permutation_must_be_whole_numbers(self) -> None:
        message = self._assert_explains(encrypt, CLASSIC, (0.0, 1.0))
        self.assertIn("whole numbers", message)

    def test_filler_must_be_one_letter(self) -> None:
        for filler in ("XY", "", "1", "  "):
            with self.subTest(filler=filler):
                message = self._assert_explains(
                    encrypt, CLASSIC, "ZEBRAS", complete=True, filler=filler
                )
                self.assertIn("single letter", message)

    def test_unknown_solver_option(self) -> None:
        message = self._assert_explains(
            solve, CLASSIC, scorer=default_scorer(), keylength=6
        )
        self.assertIn("keylength", message)

    def test_none_options_mean_not_supplied(self) -> None:
        # A command line hands us None for arguments the user left out.
        plaintext = letters_only(CORPUS)[:120]
        result = solve(
            encrypt(plaintext, "CAB"),
            scorer=default_scorer(),
            top=2,
            key_length=None,
            max_key_length=None,
            time_budget=None,
            seed=None,
        )
        self.assertTrue(result)

    def test_bad_solver_option_value(self) -> None:
        message = self._assert_explains(
            solve, CLASSIC, scorer=default_scorer(), max_key_length="six"
        )
        self.assertIn("integer", message)


class TestSolver(unittest.TestCase):
    """The attack, on text long enough to be honestly solvable."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = default_scorer()
        cls.plaintext = letters_only(CORPUS)[:300]
        assert len(cls.plaintext) >= 300

    def test_recovers_plaintext_and_key(self) -> None:
        ciphertext = encrypt(self.plaintext, "ZEBRAS")
        result = solve(ciphertext, scorer=self.scorer, top=5)
        best = result.best()
        self.assertEqual(best.plaintext, self.plaintext)
        # Reported as the canonical keyword for the same permutation.
        self.assertEqual(key_order(best.key.split("=")[1].split(" ")[0]),
                         key_order("ZEBRAS"))
        self.assertEqual(best.confidence(), "strong")

    def test_recovers_several_key_lengths(self) -> None:
        for keyword in ("DR", "CAB", "CIPHER", "KEYWORD", "MONARCHY"):
            with self.subTest(keyword=keyword):
                ciphertext = encrypt(self.plaintext, keyword)
                best = solve(ciphertext, scorer=self.scorer, top=3).best()
                self.assertEqual(best.plaintext, self.plaintext)

    def test_recovers_a_ragged_grid(self) -> None:
        # 301 letters over 6 columns: rows of 51/50, so the ciphertext blocks
        # have two different lengths and the solver must work out which.
        plaintext = letters_only(CORPUS)[:301]
        ciphertext = encrypt(plaintext, "ZEBRAS")
        best = solve(ciphertext, scorer=self.scorer, top=3, key_length=6).best()
        self.assertEqual(best.plaintext, plaintext)
        self.assertEqual(best.diagnostics["ragged_columns"], 1)

    def test_greedy_search_for_a_long_key(self) -> None:
        # DEFAULT_MAX_EXHAUSTIVE is 9, so a 9-column key is now enumerated;
        # reaching the greedy path takes an explicit lower ceiling.
        ciphertext = encrypt(self.plaintext, "WATERFALL")
        best = solve(
            ciphertext, scorer=self.scorer, top=3, key_length=9, seed=11,
            max_exhaustive=8,
        ).best()
        self.assertEqual(best.plaintext, self.plaintext)
        self.assertIn("greedy", best.diagnostics["search"])

    def test_nine_columns_is_enumerated_by_default(self) -> None:
        """The default must enumerate every key length it also tries.

        Measured: at 9 columns exhaustive search found the true key on six
        samples out of six and greedy on five, for 0.28s against 0.02s. The
        accuracy is worth the quarter second, and the default says so.
        """
        ciphertext = encrypt(self.plaintext, "WATERFALL")
        best = solve(ciphertext, scorer=self.scorer, top=3, key_length=9,
                     seed=11).best()
        self.assertEqual(best.plaintext, self.plaintext)
        self.assertEqual(best.diagnostics["search"], "exhaustive")

    def test_seed_makes_the_greedy_search_reproducible(self) -> None:
        ciphertext = encrypt(self.plaintext, "WATERFALL")
        first = solve(ciphertext, scorer=self.scorer, key_length=9, seed=5)
        second = solve(ciphertext, scorer=self.scorer, key_length=9, seed=5)
        self.assertEqual(
            [candidate.key for candidate in first],
            [candidate.key for candidate in second],
        )

    def test_complete_option_restricts_to_divisors(self) -> None:
        ciphertext = encrypt(self.plaintext, "ZEBRAS", complete=True)
        result = solve(ciphertext, scorer=self.scorer, top=3, complete=True)
        best = result.best()
        self.assertTrue(best.plaintext.startswith(self.plaintext[:80]))
        grids = best.diagnostics["grids_tested"]
        # 300 = 2^2 x 3 x 5^2, so 7, 8 and 9 columns cannot make a rectangle.
        self.assertIn("6x50", grids)
        self.assertNotIn("7x", grids)

    def test_diagnostics_list_every_grid_tested(self) -> None:
        result = solve(
            encrypt(self.plaintext, "CIPHER"),
            scorer=self.scorer,
            top=2,
            max_key_length=6,
        )
        grids = result.best().diagnostics["grids_tested"]
        for count in range(2, 7):
            self.assertIn(f"{count}x", grids)
        self.assertEqual(result.best().diagnostics["length"], 300)
        self.assertIn("column_pair_score", result.best().diagnostics)

    def test_short_ciphertext_still_solvable(self) -> None:
        """Forty letters, where every scrap of evidence has to be used.

        These two cases are the regression test for the row-wrap term in the
        column-pair score: both are recovered with it and neither is without
        it (measured -- see ``adjacency_matrices``). They are calibration
        against the current corpus, so if the corpus ever changes and one of
        them starts failing, re-measure the wrap term before weakening this.
        """
        source = letters_only(CORPUS)
        for start, keyword in ((2300, "PLANETS"), (2700, "MONARCHY")):
            plaintext = source[start : start + 40]
            with self.subTest(start=start, keyword=keyword):
                ciphertext = encrypt(plaintext, keyword)
                best = solve(
                    ciphertext,
                    scorer=self.scorer,
                    top=1,
                    key_length=len(keyword),
                ).best()
                self.assertEqual(best.plaintext, plaintext)

    def test_time_budget_is_respected_and_reported(self) -> None:
        ciphertext = encrypt(self.plaintext, "MONARCHY")
        result = solve(ciphertext, scorer=self.scorer, top=3, time_budget=0.02)
        self.assertTrue(result)
        self.assertTrue(result.best().diagnostics.get("time_budget_hit"))

    def test_accepts_a_normalized_text(self) -> None:
        # The command line normalises before calling us, so both entry points
        # must behave identically.
        ciphertext = encrypt(self.plaintext, "CIPHER")
        grouped = normalize(group_text(ciphertext))
        from_text = solve(ciphertext, scorer=self.scorer, top=3)
        from_object = solve(grouped, scorer=self.scorer, top=3)
        self.assertEqual(from_object.best().plaintext, self.plaintext)
        self.assertEqual(
            [candidate.key for candidate in from_text],
            [candidate.key for candidate in from_object],
        )

    def test_display_is_not_the_original_layout(self) -> None:
        result = solve(
            encrypt(self.plaintext, "ZEBRAS"), scorer=self.scorer, top=1
        )
        best = result.best()
        self.assertIsNotNone(best.display)
        self.assertEqual(letters_only(best.display), best.plaintext)
        self.assertIn(" ", best.display)


class TestFailureModes(unittest.TestCase):
    """What the tool does when the text is NOT a single columnar."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = default_scorer()
        cls.plaintext = letters_only(CORPUS)[:300]

    def test_random_letters_are_not_reported_as_solved(self) -> None:
        rng = random.Random(7)
        noise = "".join(
            rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(300)
        )
        best = solve(noise, scorer=self.scorer, top=3, max_key_length=6).best()
        self.assertIsNotNone(best)
        self.assertIn(best.confidence(), ("weak", "unlikely"))
        self.assertLess(best.diagnostics["word_coverage"], 0.5)
        self.assertLess(best.diagnostics["normalised_score"], -2.0)

    def test_double_transposition_is_not_broken_by_the_single_solver(self) -> None:
        # Honesty check: two passes defeat this attack, and the tool must say
        # so with a low confidence rather than dressing up its best guess.
        ciphertext = encrypt_double(self.plaintext, "ZEBRAS", "MONARCHY")
        best = solve(ciphertext, scorer=self.scorer, top=3).best()
        self.assertNotEqual(best.plaintext, self.plaintext)
        self.assertNotEqual(best.confidence(), "strong")

    def test_wrong_key_gives_junk_not_plaintext(self) -> None:
        ciphertext = encrypt(self.plaintext, "ZEBRAS")
        wrong = decrypt(ciphertext, "SARBEZ")
        self.assertNotEqual(wrong, self.plaintext)
        self.assertLess(
            self.scorer.normalised(wrong), self.scorer.normalised(self.plaintext)
        )

    def test_complete_decryption_of_a_ragged_length_is_refused(self) -> None:
        # 25 letters cannot be a complete six-column rectangle. Saying so is
        # better than silently decrypting it as something it is not.
        with self.assertRaises(ValueError) as caught:
            decrypt("EVLNACDTESEAROFODEECWIREE", "ZEBRAS", complete=True)
        self.assertIn("whole number of rows", str(caught.exception))

    def test_impossible_solver_configuration_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            solve(
                encrypt(self.plaintext, "KEYWORD"),
                scorer=self.scorer,
                key_length=7,
                complete=True,
            )
        message = str(caught.exception)
        self.assertIn("impossible", message)
        self.assertIn("7", message)


class TestDoubleSolver(unittest.TestCase):
    """Attacking two passes of columnar transposition.

    Measured before this existed: `auto --deep` on a double-columnar message
    returned a `promising` reading that was WRONG -- confident and incorrect,
    the worst combination. The single-pass attack cannot work here and says
    so in its own docstring: after the second pass, letters that were
    neighbours in a plaintext row are no longer a fixed distance apart, so
    the column-pair statistics have nothing to lock onto.

    The search is therefore over both permutations at once, which is not
    exhaustive and must never claim to be.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = default_scorer()
        cls.plaintext = letters_only(CORPUS)[:400]

    # Most of these tests pin `restarts` and `iterations` well below the
    # module defaults. That is a deliberate trade: only ONE test here needs
    # the search to be reliable on a hard key pair, and running the full
    # twelve restarts in all of them cost eight minutes. Where the point is
    # plumbing -- reproducibility, budgets, diagnostics -- a short search
    # proves it just as well. The default restart count is justified by
    # measurement in the module, not by these tests.
    CHEAP = {"restarts": 3, "iterations": 4000}

    def test_recovers_a_short_double_key_pair(self) -> None:
        ciphertext = encrypt_double(self.plaintext, "KEYS", "CAB")
        result = solve_double(
            ciphertext, scorer=self.scorer, top=3,
            first_length=4, second_length=3, seed=1,
            restarts=6, iterations=15_000,
        )
        self.assertEqual(result.best().plaintext, self.plaintext)

    def test_recovers_a_realistic_seven_by_six(self) -> None:
        """Key lengths a competition actually uses."""
        ciphertext = encrypt_double(self.plaintext, "MONARCH", "BERLIN")
        result = solve_double(
            ciphertext, scorer=self.scorer, top=3,
            first_length=7, second_length=6, seed=1,
        )
        self.assertEqual(result.best().plaintext, self.plaintext)

    def test_it_finds_the_lengths_when_not_told_them(self) -> None:
        """A real user does not know the key lengths.

        Short keys on purpose. What is under test is the shape sweep -- that
        the solver tries length pairs and recognises the right one -- not how
        hard a key it can crack, which `test_recovers_a_realistic_seven_by_six`
        covers at full strength. Sweeping up to 4x4 at the default restart
        count took four minutes for the same assurance.
        """
        ciphertext = encrypt_double(self.plaintext, "KEY", "CA")
        result = solve_double(
            ciphertext, scorer=self.scorer, top=3,
            max_key_length=3, seed=1, restarts=6, iterations=15_000,
        )
        self.assertEqual(result.best().plaintext, self.plaintext)
        best = result.best()
        self.assertEqual(best.diagnostics["first_key_length"], 3)
        self.assertEqual(best.diagnostics["second_key_length"], 2)

    def test_the_key_is_reported_so_the_answer_can_be_checked(self) -> None:
        ciphertext = encrypt_double(self.plaintext, "KEYS", "CAB")
        best = solve_double(
            ciphertext, scorer=self.scorer, top=1,
            first_length=4, second_length=3, seed=1, **self.CHEAP,
        ).best()
        # The invariant is that the key it SHOWS you reproduces the answer it
        # SHOWS you. A reported key that does not regenerate the reported
        # plaintext is unfalsifiable, and a competition answer nobody can
        # check by hand is worthless whether or not it happens to be right.
        recovered = decrypt_double(
            ciphertext,
            best.diagnostics["first_permutation"],
            best.diagnostics["second_permutation"],
        )
        self.assertEqual(recovered, best.plaintext)

    def test_it_never_claims_the_search_was_exhaustive(self) -> None:
        """It is a randomised climb. Saying otherwise would be a lie."""
        ciphertext = encrypt_double(self.plaintext, "KEYS", "CAB")
        best = solve_double(
            ciphertext, scorer=self.scorer, top=1,
            first_length=4, second_length=3, seed=1, **self.CHEAP,
        ).best()
        self.assertIn("not exhaustive", best.diagnostics["search"])

    def test_the_seed_makes_a_run_reproducible(self) -> None:
        ciphertext = encrypt_double(self.plaintext, "MONARCH", "BERLIN")
        options = dict(scorer=self.scorer, top=1, first_length=7,
                       second_length=6, seed=17, **self.CHEAP)
        first = solve_double(ciphertext, **options).best()
        second = solve_double(ciphertext, **options).best()
        self.assertEqual(first.plaintext, second.plaintext)
        self.assertEqual(first.key, second.key)

    def test_a_time_budget_is_honoured_and_recorded(self) -> None:
        ciphertext = encrypt_double(self.plaintext, "MONARCH", "BERLIN")
        started = time.monotonic()
        result = solve_double(
            ciphertext, scorer=self.scorer, top=3,
            max_key_length=8, seed=1, time_budget=1.0,
        )
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 20.0, "the budget must actually stop it")
        if result:
            self.assertTrue(
                any(c.diagnostics.get("time_budget_hit")
                    for c in result.ranked())
            )

    def test_random_letters_are_not_reported_as_solved(self) -> None:
        """The honesty bar this whole toolkit is built around."""
        rng = random.Random(11)
        noise = "".join(
            rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(300)
        )
        best = solve_double(
            noise, scorer=self.scorer, top=1,
            first_length=4, second_length=3, seed=1, **self.CHEAP,
        ).best()
        if best is not None:
            self.assertNotEqual(best.confidence(), "strong")

    def test_empty_input_is_not_a_crash(self) -> None:
        self.assertEqual(len(solve_double("", scorer=self.scorer)), 0)

    def test_the_pipeline_reaches_it_at_deep_and_not_before(self) -> None:
        """A solver nothing calls is decoration.

        The measured failure that prompted all this was `auto --deep` giving
        a `promising` and WRONG reading of a double columnar message, so the
        fix is only real if the pipeline actually runs this.
        """
        from cipher_tool.auto import build_stages

        for effort in ("fast", "normal"):
            names = [stage.name for stage in build_stages(effort, 5, 1)]
            self.assertNotIn("double columnar", names,
                             f"too expensive to run at {effort}")
        deep = [stage for stage in build_stages("deep", 5, 1)
                if stage.name == "double columnar"]
        self.assertEqual(len(deep), 1)
        self.assertIsNotNone(
            deep[0].options.get("time_budget"),
            "unbounded, this stage would run for hours without max_time",
        )
        # A 7-letter keyword is ordinary in this competition. With the
        # ceiling at 6 the paste screen escalated all the way to deep on a
        # 7x6 message and still reported `weak` -- honest, and useless to
        # the person holding the ciphertext.
        self.assertGreaterEqual(
            deep[0].options.get("max_key_length", 0), 7,
            "the pipeline must reach ordinary keyword lengths",
        )

    def test_the_pipeline_actually_solves_one(self) -> None:
        """The stage's real settings, with the clock taken out of it.

        This test used to run the stage exactly as it ships, 120-second
        budget and all, and assert that a RANDOMISED search finished in
        time. That is a claim about the machine as much as about the
        toolkit, and it eventually failed on the slowest CI runner while
        passing on six others. MEASURED here: the search returns the right
        key in 34 seconds; at a 30-second budget it still solves; at 20 it
        returns no candidate at all, because a search cut short offers
        nothing rather than a guess.

        So the budget is overridden generously. Nothing else changes -- the
        restarts, the iterations and the key-length ceiling are the shipped
        ones -- and the stage still stops as soon as its restarts are done,
        so this costs no extra time on a machine that was passing anyway.
        That the SHIPPED stage carries a budget at all is asserted
        separately, just above.
        """
        from cipher_tool.auto import auto_solve, build_stages

        shipped = [s for s in build_stages("deep", 5, 1)
                   if s.name == "double columnar"][0]
        stage = [replace(shipped,
                         options={**shipped.options, "time_budget": 600.0})]
        ciphertext = encrypt_double(self.plaintext, "KEY", "CA")
        result = auto_solve(ciphertext, scorer=self.scorer, effort="deep",
                            top=3, seed=1, stages=stage)
        best = result.candidates.best()
        self.assertIsNotNone(best, "the stage produced no candidate at all")
        self.assertEqual(best.plaintext, self.plaintext)


if __name__ == "__main__":
    unittest.main()
