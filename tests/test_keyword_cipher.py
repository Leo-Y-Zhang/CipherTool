"""Tests for the keyword substitution cipher and its attacks.

The hand-worked example used throughout is the keyword SECRET:

    reduced   SECRT
    remainder ABDFGHIJKLMNOPQUVWXYZ
    plain     A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
    cipher    S E C R T A B D F G H I J K L M N O P Q U V W X Y Z

so ATTACKATDAWN enciphers, letter by letter, to SQQSCHSQRSWK.
"""

from __future__ import annotations

import sys
import unittest

from cipher_tool.candidates import Candidate, CandidateSet
from cipher_tool.normalize import ALPHABET, letters_only, normalize
from cipher_tool.scoring import DATA_DIR, default_scorer
from cipher_tool.keyword_cipher import (
    KeywordRecovery,
    _alphabets_in,
    candidate_keywords,
    decrypt,
    decrypt_with_alphabet,
    deduplicate,
    describe_alphabet,
    encrypt,
    encrypt_with_alphabet,
    invert_alphabet,
    keyword_alphabet,
    solve,
    validate_alphabet,
)

SECRET_ALPHABET = "SECRTABDFGHIJKLMNOPQUVWXYZ"
SECRET_REVERSED = "SECRTZYXWVUQPONMLKJIHGFDBA"

#: A permutation with no long alphabetical run at any of the 52 alignments.
#: Recovery must report nothing at all for this one.
SCRAMBLED = "OPMDYQHWKCTEURBVFZSIGLJAXN"


def english(count: int = 400) -> str:
    """The first *count* letters of the expository corpus file."""
    text = (DATA_DIR / "corpus_04_expository.txt").read_text(encoding="utf-8")
    letters = letters_only(text)
    if len(letters) < count:  # pragma: no cover - corpus is far longer
        raise AssertionError(f"corpus has only {len(letters)} letters")
    return letters[:count]


# ---------------------------------------------------------------------------
# A stand-in for the companion substitution solver
#
# solve(hill_climb=True) calls cipher_tool.substitution. That module is a
# randomised hill climber: driving it here would make these tests slow, and
# would make a result about our own plumbing depend on somebody else's search
# succeeding. So we swap in a stub that answers in the documented format and
# put the real module back afterwards.
# ---------------------------------------------------------------------------


class _StubSubstitution:
    """Minimal stand-in exposing just the ``solve`` the keyword module uses."""

    def __init__(self, key_text: str) -> None:
        self.key_text = key_text
        self.calls: list[dict[str, object]] = []

    def solve(self, source, *, scorer=None, top=5, seed=None, time_budget=None):
        """Record the call and hand back one candidate in the real format."""
        self.calls.append({"top": top, "seed": seed, "time_budget": time_budget})
        return CandidateSet(
            [Candidate(method="Substitution", key=self.key_text, score=-1.0,
                       plaintext="STUB")]
        )


def _stub_substitution(key_text: str) -> _StubSubstitution:
    """Build a stub whose single candidate carries *key_text*."""
    return _StubSubstitution(key_text)


def _install_substitution(stub: _StubSubstitution | None):
    """Put *stub* where ``from . import substitution`` will find it.

    Passing ``None`` makes the import fail instead, which is how the missing
    module is simulated. Returns what was there before, for the cleanup.
    """
    import cipher_tool

    previous_attribute = getattr(cipher_tool, "substitution", None)
    previous_module = sys.modules.get("cipher_tool.substitution")
    if stub is None:
        if hasattr(cipher_tool, "substitution"):
            delattr(cipher_tool, "substitution")
        sys.modules["cipher_tool.substitution"] = None  # type: ignore[assignment]
    else:
        cipher_tool.substitution = stub  # type: ignore[attr-defined]
        sys.modules["cipher_tool.substitution"] = stub  # type: ignore[assignment]
    return previous_attribute, previous_module


def _restore_substitution(previous_attribute, previous_module) -> None:
    """Undo :func:`_install_substitution`."""
    import cipher_tool

    if previous_attribute is None:
        if hasattr(cipher_tool, "substitution"):
            delattr(cipher_tool, "substitution")
    else:
        cipher_tool.substitution = previous_attribute  # type: ignore[attr-defined]
    if previous_module is None:
        sys.modules.pop("cipher_tool.substitution", None)
    else:
        sys.modules["cipher_tool.substitution"] = previous_module


# ---------------------------------------------------------------------------
# Building the alphabet
# ---------------------------------------------------------------------------


class TestKeywordAlphabet(unittest.TestCase):
    def test_deduplicate(self) -> None:
        self.assertEqual(deduplicate("SECRET"), "SECRT")
        self.assertEqual(deduplicate("bookkeeper"), "BOKEPR")
        self.assertEqual(deduplicate("the quick brown!"), "THEQUICKBROWN")

    def test_known_alphabet(self) -> None:
        self.assertEqual(keyword_alphabet("SECRET"), SECRET_ALPHABET)

    def test_known_alphabet_reversed_remainder(self) -> None:
        self.assertEqual(
            keyword_alphabet("SECRET", reverse_remainder=True), SECRET_REVERSED
        )

    def test_known_alphabet_with_start_letter(self) -> None:
        # HELLO reduces to HELO. Written under C, the remainder ABCDFG...
        # runs on from position 6 and wraps round to fill A and B with Y, Z.
        self.assertEqual(keyword_alphabet("HELLO", "C"), "YZHELOABCDFGIJKMNPQRSTUVWX")

    def test_start_letter_only_rotates(self) -> None:
        """Every offset is a rotation of the textbook alphabet."""
        base = keyword_alphabet("PORTCULLIS")
        for offset, letter in enumerate(ALPHABET):
            expected = base if offset == 0 else base[-offset:] + base[:-offset]
            self.assertEqual(keyword_alphabet("PORTCULLIS", letter), expected, letter)

    def test_alphabets_are_permutations(self) -> None:
        for word in ("A", "SECRET", "ZEBRA", "THEQUICKBROWNFOX", "ZYXWVUTSRQPONM"):
            for start in ("A", "M", "Z"):
                for reverse in (False, True):
                    alphabet = keyword_alphabet(word, start, reverse)
                    self.assertEqual(sorted(alphabet), list(ALPHABET))

    def test_keyword_is_cleaned_before_use(self) -> None:
        self.assertEqual(keyword_alphabet("se-cret!"), keyword_alphabet("SECRET"))
        self.assertEqual(keyword_alphabet("SECRET", "a"), keyword_alphabet("SECRET"))

    def test_invert_alphabet_round_trips(self) -> None:
        alphabet = keyword_alphabet("LIGHTHOUSE", "F", True)
        self.assertEqual(invert_alphabet(invert_alphabet(alphabet)), alphabet)


# ---------------------------------------------------------------------------
# Encryption and decryption
# ---------------------------------------------------------------------------


class TestEncryptDecrypt(unittest.TestCase):
    def test_known_pair(self) -> None:
        self.assertEqual(encrypt("ATTACKATDAWN", "SECRET"), "SQQSCHSQRSWK")

    def test_known_pair_decrypts(self) -> None:
        self.assertEqual(decrypt("SQQSCHSQRSWK", "SECRET"), "ATTACKATDAWN")

    def test_known_pair_reversed_remainder(self) -> None:
        # Using SECRTZYXWVUQPONMLKJIHGFDBA: A->S, T->I, C->C, K->U.
        self.assertEqual(encrypt("ATTACK", "SECRET", reverse_remainder=True), "SIISCU")

    def test_known_pair_with_start_letter(self) -> None:
        # Using YZHELOABCDFGIJKMNPQRSTUVWX: H->B, E->L, L->G, O->K.
        self.assertEqual(encrypt("HELLO", "HELLO", "C"), "BLGGK")

    def test_round_trip(self) -> None:
        plaintext = english(200)
        for word in ("SECRET", "A", "LIGHTHOUSE", "ZEBRA", "QUARTZ"):
            for start in ("A", "K", "Z"):
                for reverse in (False, True):
                    ciphertext = encrypt(plaintext, word, start, reverse)
                    self.assertEqual(
                        decrypt(ciphertext, word, start, reverse), plaintext
                    )

    def test_input_robustness(self) -> None:
        """Layout, case and punctuation must not change the answer."""
        clean = encrypt("ATTACKATDAWN", "SECRET")
        for variant in (
            "attackatdawn",
            "Attack at dawn!",
            "ATTAC KATDA WN",
            "attac\nkatda\nwn",
            "a-t-t-a-c-k, a.t. d/a/w/n?",
            "  ATTACK   AT   DAWN  ",
        ):
            self.assertEqual(encrypt(variant, "SECRET"), clean, variant)

    def test_empty_input(self) -> None:
        self.assertEqual(encrypt("", "SECRET"), "")
        self.assertEqual(decrypt("", "SECRET"), "")
        self.assertEqual(encrypt("1234 !!", "SECRET"), "")

    def test_explicit_alphabet_helpers(self) -> None:
        self.assertEqual(
            encrypt_with_alphabet("ATTACKATDAWN", SECRET_ALPHABET), "SQQSCHSQRSWK"
        )
        self.assertEqual(
            decrypt_with_alphabet("SQQSCHSQRSWK", SECRET_ALPHABET), "ATTACKATDAWN"
        )


class TestValidation(unittest.TestCase):
    def assertRaisesWithMessage(self, call) -> str:
        with self.assertRaises(ValueError) as caught:
            call()
        message = str(caught.exception)
        self.assertTrue(message.strip(), "ValueError carried no explanation")
        return message

    def test_empty_keyword_rejected(self) -> None:
        message = self.assertRaisesWithMessage(lambda: keyword_alphabet(""))
        self.assertIn("at least one letter", message)

    def test_keyword_without_letters_rejected(self) -> None:
        message = self.assertRaisesWithMessage(lambda: encrypt("HELLO", "1234!"))
        self.assertIn("at least one letter", message)

    def test_bad_start_letter_rejected(self) -> None:
        self.assertRaisesWithMessage(lambda: keyword_alphabet("SECRET", "AB"))
        self.assertRaisesWithMessage(lambda: keyword_alphabet("SECRET", ""))
        message = self.assertRaisesWithMessage(lambda: decrypt("ABC", "SECRET", "7"))
        self.assertIn("start_letter", message)

    def test_short_alphabet_rejected(self) -> None:
        message = self.assertRaisesWithMessage(lambda: validate_alphabet("ABCDEF"))
        self.assertIn("26", message)

    def test_repeated_letter_in_alphabet_rejected(self) -> None:
        broken = SECRET_ALPHABET[:-1] + "A"  # 26 letters, two As, no Z
        self.assertEqual(len(broken), 26)
        message = self.assertRaisesWithMessage(
            lambda: decrypt_with_alphabet("ABC", broken)
        )
        self.assertIn("exactly once", message)
        self.assertIn("missing Z", message)
        self.assertIn("repeated A", message)

    def test_solve_option_validation(self) -> None:
        self.assertRaisesWithMessage(lambda: solve("HELLO", top=0))
        self.assertRaisesWithMessage(lambda: solve("HELLO", time_budget=0))
        self.assertRaisesWithMessage(lambda: solve("HELLO", start_letters="!"))
        self.assertRaisesWithMessage(lambda: solve("HELLO", words=["!!", "  "]))

    def test_candidate_keywords_rejects_non_alphabet(self) -> None:
        self.assertRaisesWithMessage(lambda: candidate_keywords("SHORT"))
        self.assertRaisesWithMessage(
            lambda: candidate_keywords(SECRET_ALPHABET, minimum_tail=0)
        )


# ---------------------------------------------------------------------------
# Working backwards from an alphabet to its keyword
# ---------------------------------------------------------------------------


class TestKeywordRecovery(unittest.TestCase):
    def test_recovers_secret(self) -> None:
        results = candidate_keywords(SECRET_ALPHABET)
        best = results[0]
        self.assertEqual(best.keyword, "SECRT")
        self.assertEqual(best.start_letter, "A")
        self.assertFalse(best.reverse_remainder)
        self.assertIn("SECRET", best.lexicon_words)
        self.assertEqual(best.key_string(), "keyword=SECRET")
        self.assertEqual(best.tail_length, 21)
        self.assertLess(best.chance_probability, 1e-18)

    def test_recovers_offset_and_reversed_variants(self) -> None:
        for word, start, reverse in (
            ("LIGHTHOUSE", "A", False),
            ("PORTCULLIS", "M", True),
            ("MOON", "Q", False),
            ("SECRET", "R", True),
        ):
            alphabet = keyword_alphabet(word, start, reverse)
            results = candidate_keywords(alphabet, limit=None)
            wanted = (deduplicate(word), start, reverse)
            found = {(r.keyword, r.start_letter, r.reverse_remainder) for r in results}
            self.assertIn(wanted, found, f"{word}/{start}/{reverse}")

    def test_every_suggestion_rebuilds_the_alphabet(self) -> None:
        """A suggestion that does not regenerate the alphabet is worthless."""
        for word, start, reverse in (("SECRET", "A", False), ("HARBOUR", "T", True)):
            alphabet = keyword_alphabet(word, start, reverse)
            results = candidate_keywords(alphabet, limit=None)
            self.assertTrue(results)
            for result in results:
                self.assertIsInstance(result, KeywordRecovery)
                self.assertEqual(
                    keyword_alphabet(
                        result.keyword, result.start_letter, result.reverse_remainder
                    ),
                    alphabet,
                    result.describe(),
                )

    def test_scrambled_alphabet_reports_nothing(self) -> None:
        """FAILURE MODE: an alphabet with no keyword structure gets no answer.

        This is the honest half of the feature. A random permutation is not a
        keyword alphabet, and the tool must say so rather than inventing a
        stem from the first few letters.
        """
        self.assertEqual(sorted(SCRAMBLED), list(ALPHABET))
        self.assertEqual(candidate_keywords(SCRAMBLED), [])
        self.assertIn("no keyword structure", describe_alphabet(SCRAMBLED))

    def test_evidence_threshold_is_enforced(self) -> None:
        """Demanding an impossibly long tail must silence the method."""
        self.assertEqual(
            candidate_keywords(SECRET_ALPHABET, minimum_tail=26, extend=0), []
        )

    def test_zebra_is_read_back_correctly(self) -> None:
        """ZEBRA's A is swallowed by the ordered tail, so readings compete.

        This test used to assert that the rotated reading "EBR start=B"
        ranked first, and called it a documented limitation. It no longer
        does: with a larger lexicon the recovery prefers a reading that is a
        real English word, so ZEBRA comes back as ZEBRA. The alternatives are
        still offered, because the ciphertext genuinely cannot tell them
        apart, and every one of them must rebuild the identical alphabet.
        """
        alphabet = keyword_alphabet("ZEBRA")
        results = candidate_keywords(alphabet, limit=None)
        self.assertEqual(results[0].keyword, "ZEBRA")
        stems = {r.keyword for r in results}
        self.assertIn("ZEBR", stems)
        self.assertIn("EBR", stems)
        for result in results:
            self.assertEqual(
                keyword_alphabet(
                    result.keyword, result.start_letter, result.reverse_remainder
                ),
                alphabet,
            )

    def test_rotations_rank_below_plain_readings(self) -> None:
        results = candidate_keywords(SECRET_ALPHABET, limit=None)
        first_rotation = next(
            index for index, r in enumerate(results) if r.rotated_duplicate
        )
        last_plain = max(
            index for index, r in enumerate(results) if not r.rotated_duplicate
        )
        self.assertGreater(first_rotation, last_plain)

    def test_dictionary_ranking_decides_between_readings(self) -> None:
        """The dictionary, not the tail length, must pick between readings.

        FORMULA ends in A and its remainder starts at B, so the ordered tail
        swallows the final A and the longest-tail reading is the non-word
        FORMUL. Only the lexicon can tell that the letter belongs to the
        keyword after all.
        """
        alphabet = keyword_alphabet("FORMULA")
        with_dictionary = candidate_keywords(alphabet)[0]
        without = candidate_keywords(alphabet, lexicon=())[0]
        self.assertEqual(with_dictionary.keyword, "FORMULA")
        self.assertEqual(with_dictionary.lexicon_words, ("FORMULA",))
        self.assertEqual(without.keyword, "FORMUL")
        self.assertGreater(without.tail_length, with_dictionary.tail_length)

    def test_lexicon_can_be_switched_off(self) -> None:
        results = candidate_keywords(SECRET_ALPHABET, lexicon=())
        self.assertEqual(results[0].lexicon_words, ())
        self.assertEqual(results[0].key_string(), "keyword=SECRT")


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------


class TestSolve(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plain = english(400)
        cls.scorer = default_scorer()

    def test_recovers_keyword_from_ciphertext(self) -> None:
        ciphertext = encrypt(self.plain, "LIGHTHOUSE")
        best = solve(ciphertext, scorer=self.scorer).best()
        self.assertIsNotNone(best)
        self.assertEqual(best.plaintext, self.plain)
        self.assertEqual(best.key, "keyword=LIGHTHOUSE")
        self.assertEqual(best.confidence(), "strong")

    def test_recovers_reversed_remainder_keyword(self) -> None:
        ciphertext = encrypt(self.plain, "HARBOUR", reverse_remainder=True)
        best = solve(ciphertext, scorer=self.scorer).best()
        self.assertEqual(best.plaintext, self.plain)
        self.assertEqual(best.key, "keyword=HARBOUR remainder=reverse")
        self.assertEqual(best.diagnostics["remainder"], "reverse")

    def test_recovers_offset_keyword_when_sweeping_start_letters(self) -> None:
        ciphertext = encrypt(self.plain, "MOON", "Q")
        best = solve(
            ciphertext,
            scorer=self.scorer,
            words=["MOON", "CASTLE", "RIVER"],
            start_letters=ALPHABET,
        ).best()
        self.assertEqual(best.plaintext, self.plain)
        self.assertEqual(best.key, "keyword=MOON start=Q")

    def test_accepts_normalized_text_and_relayouts_display(self) -> None:
        source = normalize("SQQSC HSQRS WK\nSQQSC HSQRS WK")
        best = solve(source, scorer=self.scorer, words=["SECRET"]).best()
        self.assertEqual(best.plaintext, "ATTACKATDAWNATTACKATDAWN")
        self.assertEqual(best.display, "ATTAC KATDA WN\nATTAC KATDA WN")

    def test_layout_does_not_change_the_solve(self) -> None:
        ciphertext = encrypt(self.plain, "LIGHTHOUSE")
        grouped = " ".join(ciphertext[i : i + 5] for i in range(0, len(ciphertext), 5))
        best = solve(grouped.lower(), scorer=self.scorer).best()
        self.assertEqual(best.plaintext, self.plain)

    def test_empty_input(self) -> None:
        self.assertEqual(len(solve("", scorer=self.scorer)), 0)
        self.assertIsNone(solve("  12 34 ", scorer=self.scorer).best())

    def test_diagnostics_carry_the_evidence(self) -> None:
        ciphertext = encrypt(self.plain, "LIGHTHOUSE")
        best = solve(ciphertext, scorer=self.scorer).best()
        diagnostics = best.diagnostics
        self.assertEqual(diagnostics["alphabet"], keyword_alphabet("LIGHTHOUSE"))
        self.assertEqual(diagnostics["source"], "dictionary search")
        self.assertGreater(diagnostics["keywords_tried"], 3000)
        self.assertGreater(diagnostics["word_coverage"], 0.7)
        self.assertAlmostEqual(diagnostics["ciphertext_ic"], 0.068, delta=0.02)
        self.assertNotIn("time_budget_hit", diagnostics)

    def test_supplied_alphabet_is_analysed_for_a_keyword(self) -> None:
        ciphertext = encrypt(self.plain, "PORTCULLIS")
        found = solve(
            ciphertext,
            scorer=self.scorer,
            words=["NOTTHEKEY"],
            alphabets=[keyword_alphabet("PORTCULLIS")],
        )
        best = found.best()
        self.assertEqual(best.plaintext, self.plain)
        self.assertEqual(best.key, "keyword=PORTCULIS")
        self.assertEqual(best.diagnostics["source"], "supplied alphabet")
        self.assertIn("ordered tail", best.diagnostics["keyword_analysis"])

    def test_time_budget_is_respected_and_recorded(self) -> None:
        ciphertext = encrypt(self.plain, "LIGHTHOUSE")
        found = solve(ciphertext, scorer=self.scorer, time_budget=0.05)
        self.assertTrue(found)
        best = found.best()
        self.assertTrue(best.diagnostics["time_budget_hit"])
        self.assertLess(best.diagnostics["keywords_tried"], 7000)

    # -- failure modes -----------------------------------------------------

    def test_wrong_wordlist_is_reported_as_weak(self) -> None:
        """FAILURE MODE: the keyword is not in the list, so nothing is found.

        The solver still returns its best guess -- that is what a ranked
        candidate list is for -- but it must not dress the guess up. Both
        signals have to say no: near-random letter statistics and almost no
        dictionary coverage.
        """
        ciphertext = encrypt(self.plain, "LIGHTHOUSE")
        best = solve(
            ciphertext, scorer=self.scorer, words=["CASTLE", "RIVER", "MOUNTAIN"]
        ).best()
        self.assertNotEqual(best.plaintext, self.plain)
        self.assertIn(best.confidence(), {"weak", "unlikely"})
        self.assertLess(best.diagnostics["normalised_score"], -1.8)
        self.assertLess(best.diagnostics["word_coverage"], 0.4)

    def test_non_keyword_alphabet_is_not_claimed_solved(self) -> None:
        """FAILURE MODE: the cipher was not a keyword cipher at all.

        A plain scrambled substitution alphabet cannot be built from any
        keyword, so every trial is wrong. The best of a bad set must still
        read as weak, and asking for the keyword must return nothing.
        """
        ciphertext = encrypt_with_alphabet(self.plain, SCRAMBLED)
        found = solve(ciphertext, scorer=self.scorer)
        best = found.best()
        self.assertNotEqual(best.plaintext, self.plain)
        self.assertIn(best.confidence(), {"weak", "unlikely"})
        self.assertLess(best.diagnostics["normalised_score"], -1.8)
        for candidate in found.top(5):
            self.assertNotEqual(candidate.confidence(), "strong")
        self.assertEqual(candidate_keywords(SCRAMBLED), [])

    def test_alphabets_are_read_out_of_another_solvers_candidate(self) -> None:
        """The hill climber reports one direction of the table; we read both.

        ``substitution.solve`` writes its key as ``key=<26 letters>`` and that
        key deciphers rather than enciphers, so keyword structure may be in
        the alphabet or in its inverse. This uses a stand-in candidate in
        exactly that format, so it tests our parsing rather than the other
        module's behaviour.
        """
        alphabet = keyword_alphabet("LIGHTHOUSE")
        deciphering = invert_alphabet(alphabet)
        candidate = Candidate(
            method="Substitution",
            key=f"key={deciphering}",
            score=-1.0,
            plaintext="ANYTHING",
        )
        found = _alphabets_in(candidate)
        self.assertIn(deciphering, found)
        self.assertIn(alphabet, found, "the inverse orientation was not tried")
        self.assertEqual(
            candidate_keywords(alphabet)[0].key_string(), "keyword=LIGHTHOUSE"
        )

    def test_junk_key_strings_yield_no_alphabets(self) -> None:
        """FAILURE MODE: text that is not a permutation must be ignored."""
        for key in ("shift=3", "key=LEMON", "rails=4", "key=" + "A" * 26):
            candidate = Candidate(method="X", key=key, score=0.0, plaintext="")
            self.assertEqual(_alphabets_in(candidate), [], key)

    def test_hill_climb_output_is_analysed_for_keywords(self) -> None:
        """The hill climber's keys must arrive labelled and keyword-analysed.

        A stand-in stands in for ``substitution.solve``. Driving the real
        climber here would make this test randomised, slow, and dependent on
        another module's behaviour for a result about *our* wiring; the stub
        returns a key in the exact format the real solver emits.
        """
        alphabet = keyword_alphabet("PORTCULLIS")
        stub = _stub_substitution(f"key={invert_alphabet(alphabet)}")
        self.addCleanup(_restore_substitution, *_install_substitution(stub))

        ciphertext = encrypt(self.plain, "PORTCULLIS")
        found = solve(
            ciphertext, scorer=self.scorer, words=["NOTTHEKEY"], hill_climb=True,
            hill_climb_top=2, seed=11,
        )
        climbed = [
            candidate
            for candidate in found
            if candidate.diagnostics.get("source") == "substitution hill climb"
        ]
        self.assertTrue(climbed, "no candidates came back from the hill climber")
        best = max(climbed, key=lambda candidate: candidate.score)
        self.assertEqual(best.plaintext, self.plain)
        self.assertEqual(best.key, "keyword=PORTCULIS")
        self.assertIn("ordered tail", best.diagnostics["keyword_analysis"])
        self.assertEqual(stub.calls[0]["seed"], 11)

    def test_hill_climb_without_the_substitution_module_explains_itself(self) -> None:
        """FAILURE MODE: the option is asked for but the module is absent.

        Silently returning dictionary-only results would be a lie about what
        was searched, so the request must fail loudly and say what to do.
        """
        self.addCleanup(_restore_substitution, *_install_substitution(None))
        with self.assertRaises(ImportError) as caught:
            solve("SQQSCHSQRSWK", scorer=self.scorer, words=["SECRET"],
                  hill_climb=True)
        self.assertIn("cipher_tool.substitution", str(caught.exception))
        self.assertIn("without hill_climb", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
