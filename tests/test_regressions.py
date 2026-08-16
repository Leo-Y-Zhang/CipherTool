"""Regression tests: bugs we actually hit, pinned so they cannot come back.

Each test names the bug and why it was dangerous. These are deliberately
written as independent re-derivations rather than as checks of internal
state, because the bugs they guard against were all cases of a fast path
quietly disagreeing with the slow, obviously-correct one.
"""

from __future__ import annotations

import random
import time
import unittest
from unittest import mock

from cipher_tool import bifid, encodings, playfair, polybius, substitution
from cipher_tool.normalize import ALPHABET, letters_only
from cipher_tool.scoring import corpus_files, default_scorer
from cipher_tool.substitution import _HillClimber


def sample(length: int = 400, offset: int = 1000) -> str:
    text = letters_only(corpus_files()[0].read_text(encoding="utf-8"))
    return text[offset:offset + length]


class TestIncrementalScoringMatchesFullRescore(unittest.TestCase):
    """The hill climber's incremental score must equal a full recomputation.

    The bug: when a swap was accepted, the cached per-window scores were not
    refreshed. Every later comparison then measured the new arrangement
    against stale numbers, so the climb silently optimised the wrong
    objective. Nothing crashed and nothing looked wrong -- the solver just
    quietly became worse, which is the most dangerous kind of bug in a tool
    whose whole job is to rank guesses.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = default_scorer()

    def test_running_total_equals_full_score_after_climbing(self) -> None:
        generator = random.Random(11)
        for trial in range(8):
            with self.subTest(trial=trial):
                alphabet = list(ALPHABET)
                generator.shuffle(alphabet)
                key = substitution.SubstitutionKey.from_alphabet(
                    "".join(alphabet)
                )
                ciphertext = substitution.encrypt(sample(400), key)

                climber = _HillClimber(
                    self.scorer.encode(ciphertext),
                    self.scorer.table(),
                    list(range(26)),
                )
                start = list(range(26))
                generator.shuffle(start)
                result, running_total, _ = climber.climb(start)

                self.assertAlmostEqual(
                    running_total,
                    climber.full_score(result),
                    places=6,
                    msg="incremental score drifted from the true score; the "
                        "window cache is not being kept in step",
                )

    def test_the_climb_actually_improves_on_its_starting_point(self) -> None:
        generator = random.Random(5)
        alphabet = list(ALPHABET)
        generator.shuffle(alphabet)
        key = substitution.SubstitutionKey.from_alphabet("".join(alphabet))
        ciphertext = substitution.encrypt(sample(400), key)

        climber = _HillClimber(self.scorer.encode(ciphertext),
                               self.scorer.table(), list(range(26)))
        start = list(range(26))
        generator.shuffle(start)
        before = climber.full_score(start)
        result, after, _ = climber.climb(start)
        self.assertGreater(after, before)

    def test_a_local_optimum_admits_no_improving_swap(self) -> None:
        """After the climb stops, no single swap may raise the score.

        This is the definition of a local optimum. A stale cache breaks it:
        the climber thinks it has finished while improving swaps remain.
        """
        generator = random.Random(17)
        alphabet = list(ALPHABET)
        generator.shuffle(alphabet)
        key = substitution.SubstitutionKey.from_alphabet("".join(alphabet))
        ciphertext = substitution.encrypt(sample(300), key)

        climber = _HillClimber(self.scorer.encode(ciphertext),
                               self.scorer.table(), list(range(26)))
        start = list(range(26))
        generator.shuffle(start)
        result, total, cut_short = climber.climb(start)
        self.assertFalse(cut_short)

        for first in range(26):
            for second in range(first + 1, 26):
                candidate = list(result)
                candidate[first], candidate[second] = (
                    candidate[second], candidate[first]
                )
                self.assertLessEqual(
                    climber.full_score(candidate), total + 1e-9,
                    f"swapping {first} and {second} still improves the score, "
                    "so the climb stopped early",
                )


class TestMorseLineBreaks(unittest.TestCase):
    """A newline in Morse is a word gap, not a letter gap.

    The bug: newlines were folded to a single space, so Morse transcribed
    one word per line decoded with every word run together.
    """

    def test_one_word_per_line(self) -> None:
        text = ".... . .-.. .-.. ---\n.-- --- .-. .-.. -.."
        self.assertEqual(encodings.decode_morse(text), "HELLO WORLD")

    def test_windows_line_endings(self) -> None:
        text = ".... . .-.. .-.. ---\r\n.-- --- .-. .-.. -.."
        self.assertEqual(encodings.decode_morse(text), "HELLO WORLD")

    def test_explicit_slash_still_works(self) -> None:
        text = ".... . .-.. .-.. --- / .-- --- .-. .-.. -.."
        self.assertEqual(encodings.decode_morse(text), "HELLO WORLD")

    def test_single_line_is_unaffected(self) -> None:
        self.assertEqual(encodings.decode_morse(".... . .-.. .-.. ---"),
                         "HELLO")


class TestBifidWorkedExample(unittest.TestCase):
    """The example in the module docstring must be what the code does.

    The bug: the docstring explained which plaintext letters the third
    ciphertext letter drew its coordinates from, and named the wrong ones.
    Documentation that contradicts the code is worse than none, because a
    teammate will trust it.
    """

    def test_attack_enciphers_to_dqbdqp(self) -> None:
        square = polybius.PolybiusSquare.standard()
        self.assertEqual(bifid.encrypt("ATTACK", square), "DQBDQP")

    def test_first_half_of_the_stream_is_rows_only(self) -> None:
        # The claim now made in the docstring: for an even-length block the
        # first half of the ciphertext is built purely from plaintext rows.
        square = polybius.PolybiusSquare.standard()
        plaintext = "ATTACK"
        rows, columns = [], []
        for letter in plaintext:
            row, column = square.coordinates(letter)
            rows.append(row)
            columns.append(column)
        stream = rows + columns
        rebuilt = "".join(
            square.letter(stream[i], stream[i + 1])
            for i in range(0, len(stream), 2)
        )
        self.assertEqual(rebuilt, bifid.encrypt(plaintext, square))
        # The third ciphertext letter uses stream positions 4 and 5, both of
        # which lie in the row half.
        self.assertLess(5, len(rows))


class TestKnownLimitsAreReportedHonestly(unittest.TestCase):
    """Things the toolkit cannot do, pinned so it keeps admitting them.

    Found by end-to-end verification rather than by unit tests. Each of these
    is a case where returning a confident wrong answer would be far worse
    than returning a hedged one.
    """

    def test_bifid_with_an_unknown_keyed_square_admits_defeat(self) -> None:
        from cipher_tool import bifid

        plaintext = sample(400)
        ciphertext = bifid.encrypt(plaintext, "TEMPEST", 7)

        # No keyword supplied: the square is unknown and the search cannot
        # find it. The tool must say so rather than present its best guess
        # as an answer.
        blind = bifid.solve(ciphertext, top=3).best()
        self.assertIsNotNone(blind)
        self.assertIn(blind.confidence(), {"weak", "unlikely"})
        self.assertNotIn(plaintext[:40], blind.plaintext)

        # Supply the keyword and it solves immediately, which shows the
        # failure above is about the unknown square, not about Bifid.
        informed = bifid.solve(ciphertext, top=3,
                               keywords=["TEMPEST", "HARBOUR"]).best()
        self.assertIn(plaintext[:40], informed.plaintext)
        self.assertEqual(informed.confidence(), "strong")

    def test_ciphertext_autokey_primer_is_not_determined_by_the_message(
        self,
    ) -> None:
        """Any primer gives the same plaintext after its own length.

        For ciphertext autokey the key is primer + ciphertext, so from
        position m onward the plaintext is forced by the ciphertext alone.
        The primer is therefore unrecoverable beyond what the English model
        can infer from m letters -- and the solver must say so rather than
        present its guessed primer as a recovered key.
        """
        from cipher_tool import autokey

        plaintext = sample(300)
        ciphertext = autokey.encrypt(plaintext, "OAK", mode="ciphertext")

        for primer in ("OAK", "DAX", "ZZZ"):
            decrypted = autokey.decrypt(ciphertext, primer, mode="ciphertext")
            self.assertEqual(decrypted[3:], plaintext[3:],
                             f"primer {primer} should still reveal the tail")

        best = autokey.solve(ciphertext, top=3, max_primer=4, seed=1).best()
        self.assertIsNotNone(best)
        self.assertIn("head_note", best.diagnostics,
                      "the solver must explain that the primer is a guess")
        self.assertIn("do not depend on the primer",
                      best.diagnostics["head_note"])

    def test_playfair_plaintext_keeps_its_fillers(self) -> None:
        """A Playfair decryption is not character-identical to the original.

        Doubled letters were split with a filler during encryption, and
        decryption cannot know which X was inserted and which was written.
        Anything comparing a Playfair result against the original must
        account for that -- our own verification script got this wrong first.
        """
        from cipher_tool import playfair

        plaintext = "THEENDOFITLOOKING"
        recovered = playfair.decrypt(
            playfair.encrypt(plaintext, "MONARCHY"), "MONARCHY"
        )
        self.assertNotEqual(recovered, plaintext)
        self.assertIn("X", recovered)
        self.assertEqual(recovered.replace("X", ""),
                         plaintext.replace("X", ""))


class TestCoarseClockDoesNotChangeTheAnswer(unittest.TestCase):
    """A time budget must mean the same thing on every machine.

    Windows resolves ``time.monotonic()`` to about 15.6 milliseconds, so two
    consecutive calls often return the same number. Two bugs fell out of that
    and only showed up on the Windows CI runner:

    * deadline checks used ``>`` rather than ``>=``, so a deadline that had
      exactly arrived was treated as not yet reached;
    * ``transposition.solve_all`` floored each family's slice at 50ms, which
      silently overrode a smaller budget the caller had asked for.

    Together they meant a near-zero budget produced a full set of candidates
    on a coarse clock and none on a fine one -- the same call giving
    different answers on different laptops. These tests quantise the clock to
    reproduce that locally.
    """

    GRANULARITY = 0.016

    def _coarse_clock(self):
        real = time.monotonic

        def coarse() -> float:
            return (real() // self.GRANULARITY) * self.GRANULARITY

        return mock.patch("time.monotonic", coarse)

    def test_autokey_zero_budget_searches_nothing(self) -> None:
        from cipher_tool.autokey import plaintext_autokey_encrypt
        from cipher_tool.autokey import solve as autokey_solve

        ciphertext = plaintext_autokey_encrypt(sample(420), "KEY")
        with self._coarse_clock():
            result = autokey_solve(ciphertext, time_budget=0.0)
        self.assertEqual(len(result), 0)

    def test_transposition_tiny_budget_searches_nothing(self) -> None:
        from cipher_tool.transposition import solve_all

        with self._coarse_clock():
            found = solve_all(sample(200), top=5, time_budget=1e-6)
        self.assertEqual(len(found), 0)

    def test_a_workable_budget_still_produces_candidates(self) -> None:
        """The guard must not have turned the budget into a blanket refusal."""
        from cipher_tool.transposition import solve_all

        with self._coarse_clock():
            found = solve_all(sample(200), top=3, time_budget=2.0, seed=1)
        self.assertGreater(len(found), 0)

    def test_skipped_families_are_reported_not_hidden(self) -> None:
        from cipher_tool.transposition import solve_all

        with self._coarse_clock():
            found = solve_all(sample(200), top=3, time_budget=0.05, seed=1)
        # Whatever ran or did not, the caller must be able to find out which.
        for candidate in found.ranked():
            self.assertTrue(
                "families_run" in candidate.diagnostics
                or "families_skipped_no_time" in candidate.diagnostics
            )


class TestNeverOverstatesWhatItKnows(unittest.TestCase):
    """The failure this whole toolkit exists to avoid.

    Found by adversarial first-run testing, not by the unit tests -- every
    module's own tests passed throughout.
    """

    def test_unencrypted_input_is_not_reported_as_a_solve(self) -> None:
        """Identity keys must not be sold as decryptions.

        The bug: given text that was never encrypted, every solver found its
        own do-nothing key -- Caesar shift 0, Vigenere key AAA, affine
        a=1 b=0 -- each scored 'strong' because the text really is English.
        The report then announced "CORROBORATION: Vigenere, Caesar shift,
        Affine independently produced the same plaintext. Agreement between
        unrelated attacks is strong evidence." Four attacks that decrypted
        nothing, presented as corroborating one another.
        """
        from cipher_tool.auto import auto_solve

        plaintext = sample(380, offset=0)
        result = auto_solve(plaintext, effort="fast", top=5, seed=1)

        self.assertTrue(
            result.candidates.looks_unencrypted(),
            "the pipeline should notice the input was never encrypted",
        )
        self.assertEqual(
            result.candidates.corroborations(), [],
            "identity keys must never be counted as agreeing evidence",
        )
        rendered = result.render(top=5)
        self.assertIn("DOES NOT APPEAR TO BE ENCRYPTED", rendered)
        self.assertNotIn("CORROBORATION", rendered)

    def test_genuine_agreement_is_still_reported(self) -> None:
        """The fix must not have silenced real corroboration.

        A Caesar is found by Caesar, affine and substitution alike, and that
        agreement is worth stating. Only identity results are excluded.
        """
        from cipher_tool import caesar
        from cipher_tool.auto import auto_solve

        ciphertext = caesar.encrypt(sample(400), 9)
        result = auto_solve(ciphertext, effort="fast", top=5, seed=1)
        self.assertFalse(result.candidates.looks_unencrypted())
        self.assertGreater(len(result.candidates.corroborations()), 1)
        self.assertIn("CORROBORATION", result.render(top=5))

    def test_random_letters_are_not_called_likely_anything(self) -> None:
        """The bug: 400 random letters gave 'Polyalphabetic (likely)'.

        An index of coincidence at the flat-random value is the absence of
        evidence, not evidence for a repeating key. The justification was
        also arithmetically wrong -- it said the value 'sits between random
        and English' for a value below random.
        """
        from cipher_tool.statistics import analyse

        for seed in (2, 11, 23):
            with self.subTest(seed=seed):
                generator = random.Random(seed)
                noise = "".join(
                    generator.choice(ALPHABET) for _ in range(500)
                )
                hypotheses = analyse(noise).hypotheses
                for hypothesis in hypotheses:
                    self.assertNotEqual(
                        hypothesis.confidence, "likely",
                        f"random letters must not make {hypothesis.family!r} "
                        "likely",
                    )
                    if "sits between random" in hypothesis.reason:
                        self.fail("claimed the IC sits between random and "
                                  "English without checking that it does")
                self.assertTrue(
                    any("not an English letter cipher" in h.family
                        for h in hypotheses),
                    "flat statistics should raise the possibility that this "
                    "is not a letter cipher at all",
                )

    def test_a_real_vigenere_is_still_called_likely(self) -> None:
        """The fix must not have made the report useless on real ciphertext."""
        from cipher_tool import vigenere
        from cipher_tool.statistics import analyse

        ciphertext = vigenere.encrypt(sample(900, offset=200), "ORCHID")
        families = [
            h.family for h in analyse(ciphertext).hypotheses
            if h.confidence == "likely"
        ]
        self.assertTrue(
            any("Polyalphabetic" in family for family in families),
            f"a real Vigenere should still read likely, got {families}",
        )


class TestCommandLineApiCalls(unittest.TestCase):
    """The CLI must call APIs that actually exist.

    The bug: three command paths called functions that were never defined
    (a ``PolybiusSquare.from_keyword`` classmethod that does not exist, and
    ``validate_ciphertext`` without its required square). Unit tests of the
    libraries all passed; only running the commands found it.
    """

    def test_polybius_square_factory_exists(self) -> None:
        square = polybius.PolybiusSquare.standard("TEMPEST")
        self.assertEqual(square.size, 5)
        self.assertFalse(hasattr(polybius.PolybiusSquare, "from_keyword"))

    def test_validate_ciphertext_needs_a_square(self) -> None:
        square = playfair.plain_square()
        self.assertEqual(playfair.validate_ciphertext("ABCD", square), [])
        with self.assertRaises(TypeError):
            playfair.validate_ciphertext("ABCD")


if __name__ == "__main__":
    unittest.main()
