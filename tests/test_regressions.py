"""Regression tests: bugs we actually hit, pinned so they cannot come back.

Each test names the bug and why it was dangerous. These are deliberately
written as independent re-derivations rather than as checks of internal
state, because the bugs they guard against were all cases of a fast path
quietly disagreeing with the slow, obviously-correct one.
"""

from __future__ import annotations

import random
import unittest

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
