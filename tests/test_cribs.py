"""Tests for the crib engine.

Hand computations used as anchors, worked out on paper and re-checked here in
the comments so a reader can verify them without running anything:

*Substitution.* Ciphertext XPQQZY, crib MEET. Offset 0 puts XPQQ under MEET,
which needs X=M, P=E and then Q=E as well -- two cipher letters meaning the
same plain letter, impossible for a bijective alphabet. Offset 1 puts PQQZ
under MEET: P=M, Q=E, Q=E again (consistent), Z=T. That fits, and fixes three
letters. Offset 2 puts QQZY under MEET, needing Q=M and Q=E at once, which is
impossible. So exactly one placement, at offset 1.

*Caesar.* A=0 so A+3=D; T=19 so 19+3=22=W; C=2 so C+3=F; K=10 so K+3=N.
ATTACK enciphers to DWWDFN under shift 3, and ATTACKATDAWN enciphers to
DWWDFNDWGDZQ. DAWN gives D+3=G, A+3=D, W+3=Z, N+3=Q, i.e. GDZQ, which sits at
offset 8 of that ciphertext.

*Affine with a=5, b=8.* A=0 -> 8 = I. T=19 -> 5*19+8 = 103 = 103-78 = 25 = Z.
C=2 -> 18 = S. K=10 -> 58-52 = 6 = G. So ATTACK enciphers to IZZISG and
ATTACKATDAWN to IZZISGIZXIOV.

*Vigenere.* T=19, K=10, 19+10 = 29 = 3 = D. H=7, E=4, 11 = L. E=4, Y=24,
28 = 2 = C. So THE under the key KEY is DLC, and subtracting THE from DLC
returns the key letters KEY. In the ciphertext XXDLCXX the crib THE at offset
2 therefore needs the key letters K, E, Y at text positions 2, 3, 4. With an
assumed key length of 3 those are key positions 2, 0, 1, so the key reads EYK.

The statistical tests use 400 letters of our own expository corpus, which is
the shortest text on which a crib attack is worth believing at all.
"""

from __future__ import annotations

import unittest

from cipher_tool import cribs
from cipher_tool.affine import encrypt as affine_encrypt
from cipher_tool.caesar import encrypt as caesar_encrypt
from cipher_tool.normalize import ALPHABET, letters_only, normalize
from cipher_tool.scoring import DATA_DIR
from cipher_tool.substitution import SubstitutionKey
from cipher_tool.substitution import encrypt as substitution_encrypt
from cipher_tool.vigenere import decrypt as vigenere_decrypt
from cipher_tool.vigenere import encrypt as vigenere_encrypt

#: A general monoalphabetic substitution that is not a shift: keyboard order,
#: read as a decryption alphabet (cipher A stands for plain Q, and so on).
KEYBOARD_ALPHABET = "QWERTYUIOPASDFGHJKLZXCVBNM"

#: The same alphabet as a cipher -> plain mapping, for checking recovered keys.
KEYBOARD_MAPPING = dict(zip(ALPHABET, KEYBOARD_ALPHABET))


def corpus_letters(count: int = 400) -> str:
    """The first *count* letters of our expository corpus file."""
    text = (DATA_DIR / "corpus_04_expository.txt").read_text(encoding="utf-8")
    letters = letters_only(text)
    if len(letters) < count:  # pragma: no cover - guards a corrupt install
        raise AssertionError(
            f"corpus file holds only {len(letters)} letters, needed {count}"
        )
    return letters[:count]


class TestHandComputedAnchors(unittest.TestCase):
    """Exact expected values, computed on paper (see the module docstring)."""

    def test_substitution_placement_is_the_single_bijective_offset(self) -> None:
        placements = cribs.substitution_placements("XPQQZY", "MEET")
        self.assertEqual(len(placements), 1)
        placement = placements[0]
        self.assertEqual(placement.position, 1)
        self.assertEqual(placement.window, "PQQZ")
        self.assertEqual(placement.mapping, {"P": "M", "Q": "E", "Z": "T"})
        self.assertEqual(placement.fixes, 3)
        self.assertEqual(placement.fixed_points, ())

    def test_the_two_rejected_offsets_are_rejected_for_the_stated_reason(self) -> None:
        """Offsets 0 and 2 die on the bijection rule, not on the shape rule.

        Both windows have the right length and both would give a consistent
        cipher->plain map read one way; what kills them is two cipher letters
        being forced onto the same plain letter.
        """
        self.assertEqual(
            [p.position for p in cribs.substitution_placements("XPQQZY", "MEET")],
            [1],
        )

    def test_caesar_anchor(self) -> None:
        self.assertEqual(caesar_encrypt("ATTACK", 3), "DWWDFN")
        placements = cribs.caesar_placements("DWWDFNDWGDZQ", "ATTACK")
        self.assertEqual(len(placements), 1)
        self.assertEqual(placements[0].shift, 3)
        self.assertEqual(placements[0].positions, (0,))
        self.assertEqual(placements[0].key, "shift=3")

    def test_caesar_anchor_finds_a_crib_that_is_not_at_the_start(self) -> None:
        placements = cribs.caesar_placements("DWWDFNDWGDZQ", "DAWN")
        self.assertEqual([(p.shift, p.positions) for p in placements], [(3, (8,))])

    def test_overlapping_occurrences_are_all_reported(self) -> None:
        """THAT overlaps itself in THATHAT, at offsets 0 and 3.

        A search that skipped past each match would report only offset 0 and
        quietly lose a real possibility.
        """
        placements = cribs.caesar_placements("THATHAT", "THAT")
        self.assertEqual([(p.shift, p.positions) for p in placements], [(0, (0, 3))])

    def test_affine_anchor(self) -> None:
        self.assertEqual(affine_encrypt("ATTACK", 5, 8), "IZZISG")
        placements = cribs.affine_placements("IZZISGIZXIOV", "ATTACK")
        self.assertEqual([(p.a, p.b, p.positions) for p in placements], [(5, 8, (0,))])
        self.assertEqual(placements[0].key, "a=5 b=8")

    def test_vigenere_anchor_subtracts_the_crib_to_reveal_the_key(self) -> None:
        self.assertEqual(vigenere_encrypt("THE", "KEY"), "DLC")
        fragments = cribs.vigenere_placements("XXDLCXX", "THE")
        by_position = {f.position: f.fragment for f in fragments}
        self.assertEqual(by_position[2], "KEY")
        self.assertEqual(len(fragments), 5)  # 7 letters, 3-letter crib

    def test_vigenere_anchor_rotates_the_fragment_into_key_order(self) -> None:
        fragments = cribs.vigenere_placements("XXDLCXX", "THE", 3)
        at_two = [f for f in fragments if f.position == 2][0]
        self.assertEqual(at_two.fragment, "KEY")
        self.assertEqual(at_two.partial_key, "EYK")
        self.assertEqual(at_two.fixes, 3)
        # The rotation is not cosmetic: this key really does put THE there.
        self.assertEqual(vigenere_decrypt("XXDLCXX", "EYK")[2:5], "THE")


class TestPlacementsAlwaysContainTheTruth(unittest.TestCase):
    """The equivalent of a round trip: encrypt, then look for the real answer."""

    def test_caesar_reports_the_shift_that_was_used(self) -> None:
        plaintext = "MEETMEATTHEHARBOURATMIDNIGHTANDBRINGTHEPAPERS"
        for shift in (1, 4, 7, 13, 25):
            with self.subTest(shift=shift):
                ciphertext = caesar_encrypt(plaintext, shift)
                found = cribs.caesar_placements(ciphertext, "HARBOUR")
                self.assertIn(shift, [p.shift for p in found])
                for placement in found:
                    if placement.shift == shift:
                        self.assertEqual(placement.positions, (11,))

    def test_affine_reports_the_key_that_was_used(self) -> None:
        plaintext = "MEETMEATTHEHARBOURATMIDNIGHTANDBRINGTHEPAPERS"
        for a, b in ((5, 8), (11, 0), (25, 25), (7, 19)):
            with self.subTest(a=a, b=b):
                ciphertext = affine_encrypt(plaintext, a, b)
                found = cribs.affine_placements(ciphertext, "HARBOUR")
                self.assertIn((a, b), [(p.a, p.b) for p in found])

    def test_substitution_reports_the_true_offset_and_a_correct_partial_key(self) -> None:
        plaintext = corpus_letters()
        ciphertext = substitution_encrypt(plaintext, KEYBOARD_ALPHABET)
        crib = "UNDERSTOOD"
        true_offset = plaintext.find(crib)
        self.assertNotEqual(true_offset, -1)

        found = cribs.substitution_placements(ciphertext, crib)
        positions = [placement.position for placement in found]
        self.assertIn(true_offset, positions)

        placement = [p for p in found if p.position == true_offset][0]
        for cipher, plain in placement.mapping.items():
            self.assertEqual(KEYBOARD_MAPPING[cipher], plain)

    def test_vigenere_reports_the_true_key_at_the_true_offset(self) -> None:
        plaintext = corpus_letters()
        ciphertext = vigenere_encrypt(plaintext, "LEMON")
        crib = "QUESTION"
        true_offset = plaintext.find(crib)
        self.assertNotEqual(true_offset, -1)

        fragments = cribs.vigenere_placements(ciphertext, crib, 5)
        at_true = [f for f in fragments if f.position == true_offset][0]
        self.assertTrue(at_true.consistent)
        self.assertEqual(at_true.partial_key, "LEMON")
        self.assertEqual(vigenere_decrypt(ciphertext, "LEMON"), plaintext)


class TestPruningIsWorthHaving(unittest.TestCase):
    """Measured pruning, so a regression that stops pruning is visible."""

    def test_a_repetitive_crib_leaves_almost_nothing(self) -> None:
        """UNDERSTOOD has two repeated letters and survives at one offset only.

        This is the whole value of the bijection test. If this number ever
        climbs, mapping_from_pair has stopped rejecting many-to-one maps.
        """
        plaintext = corpus_letters()
        ciphertext = substitution_encrypt(plaintext, KEYBOARD_ALPHABET)
        found = cribs.substitution_placements(ciphertext, "UNDERSTOOD")
        self.assertEqual([p.position for p in found], [plaintext.find("UNDERSTOOD")])

    def test_a_known_key_length_prunes_the_vigenere_offsets_hard(self) -> None:
        plaintext = corpus_letters()
        ciphertext = vigenere_encrypt(plaintext, "LEMON")
        fragments = cribs.vigenere_placements(ciphertext, "QUESTION", 5)
        surviving = cribs.consistent_fragments(fragments)
        self.assertLess(len(surviving), len(fragments) / 20)
        self.assertEqual(
            sorted({fragment.partial_key for fragment in surviving}), ["LEMON"]
        )

    def test_consistency_prunes_nothing_when_the_crib_is_no_longer_than_the_key(self) -> None:
        """An honest limitation, asserted so nobody assumes otherwise.

        A fragment can only contradict itself where it wraps round the key, so
        a crib no longer than the key never wraps and every offset survives.
        The pruning in the previous test comes from the crib being longer than
        the key, not from the key length as such.
        """
        ciphertext = vigenere_encrypt(corpus_letters(), "LEMON")
        fragments = cribs.vigenere_placements(ciphertext, "ZEBRA", 5)
        self.assertTrue(all(fragment.consistent for fragment in fragments))
        self.assertEqual(len(cribs.consistent_fragments(fragments)), len(fragments))


class TestKeyLengthEvidence(unittest.TestCase):
    def test_a_repeated_fragment_votes_for_the_divisors_of_its_spacing(self) -> None:
        """MEETING twice, ten letters apart, under a five-letter key.

        Ten is a multiple of five, so the key is in the same phase both times
        and the two ciphertext windows come out identical -- which is exactly
        when the two key fragments come out identical. The spacing then votes
        for its own divisors, 5 among them.
        """
        plaintext = "MEETING" + "ABC" + "MEETING" + "XYZQRS"
        ciphertext = vigenere_encrypt(plaintext, "LEMON")
        fragments = cribs.vigenere_placements(ciphertext, "MEETING")
        repeats = cribs.repeated_fragments(fragments)
        self.assertIn((0, 10), list(repeats.values()))

        votes = cribs.key_length_votes(fragments)
        self.assertIn(5, votes)
        self.assertIn(2, votes)
        self.assertNotIn(1, votes)  # 1 divides everything and says nothing

    def test_word_key_fragments_flags_a_fragment_that_reads_as_a_word(self) -> None:
        ciphertext = vigenere_encrypt(corpus_letters(), "LEMON")
        fragments = cribs.vigenere_placements(ciphertext, "QUESTION", 5)
        hits = cribs.word_key_fragments(
            cribs.consistent_fragments(fragments), lexicon={"LEMON"}
        )
        self.assertEqual(
            sorted({fragment.partial_key for fragment in hits}), ["LEMON"]
        )

    def test_word_key_fragments_ignores_fragments_that_are_not_words(self) -> None:
        fragments = cribs.vigenere_placements("XXDLCXX", "THE", 3)
        self.assertEqual(cribs.word_key_fragments(fragments, lexicon={"LEMON"}), [])


class TestTransposition(unittest.TestCase):
    def test_a_missing_letter_rules_the_whole_family_out(self) -> None:
        """The strongest negative available here, and it is exact.

        The ciphertext below is the letters of MEETMEATNOON rearranged, so it
        cannot contain HARBOUR: there is no R in it at all, and a transposition
        cannot invent one.
        """
        help_report = cribs.transposition_crib_help("NOONMEETMEAT", "HARBOUR")
        self.assertFalse(help_report.possible)
        self.assertIn("R", help_report.missing)
        self.assertIn("RULED OUT", help_report.render())

    def test_multiplicity_matters_not_just_presence(self) -> None:
        """One E present, two needed: still ruled out."""
        help_report = cribs.transposition_crib_help("MEATNO", "MEET")
        self.assertFalse(help_report.possible)
        self.assertEqual(help_report.missing, {"E": 1})
        self.assertEqual(help_report.needed["E"], 2)
        self.assertEqual(help_report.available["E"], 1)

    def test_available_letters_are_reported_with_their_positions(self) -> None:
        help_report = cribs.transposition_crib_help("MEETMEATNOON", "MEET")
        self.assertTrue(help_report.possible)
        self.assertEqual(help_report.positions["M"], (0, 4))
        self.assertEqual(help_report.positions["T"], (3, 7))
        self.assertEqual(help_report.positions["E"], (1, 2, 5))
        self.assertEqual(help_report.rarest_letters(1), ("M",))

    def test_the_report_says_the_positive_result_is_weak(self) -> None:
        text = cribs.transposition_crib_help("MEETMEATNOON", "MEET").render()
        self.assertIn("weak", text.lower())
        self.assertIn("not ruled out", text.lower())


class TestInputRobustness(unittest.TestCase):
    """Layout of the input must never change the answer."""

    CLEAN = "DWWDFNDWGDZQ"
    MESSY = "dwwdf ndwgd zq"
    PUNCTUATED = "Dwwd, fnDw! gdzq?"
    GROUPED = "DWWDF NDWGD ZQ"
    MULTILINE = "DWWDF\nNDWGD\nZQ\n"

    def test_all_layouts_give_the_same_substitution_placements(self) -> None:
        expected = [p.position for p in cribs.substitution_placements(self.CLEAN, "MEET")]
        for variant in (self.MESSY, self.PUNCTUATED, self.GROUPED, self.MULTILINE):
            with self.subTest(variant=variant):
                found = cribs.substitution_placements(variant, "MEET")
                self.assertEqual([p.position for p in found], expected)

    def test_all_layouts_give_the_same_caesar_placements(self) -> None:
        for variant in (self.MESSY, self.PUNCTUATED, self.GROUPED, self.MULTILINE):
            with self.subTest(variant=variant):
                found = cribs.caesar_placements(variant, "ATTACK")
                self.assertEqual([(p.shift, p.positions) for p in found], [(3, (0,))])

    def test_the_crib_itself_is_cleaned_the_same_way(self) -> None:
        for spelling in ("attack", "Att ack", "a-t-t-a-c-k", "ATTACK "):
            with self.subTest(spelling=spelling):
                found = cribs.caesar_placements(self.CLEAN, spelling)
                self.assertEqual([(p.shift, p.positions) for p in found], [(3, (0,))])

    def test_a_normalized_text_is_accepted_as_well_as_a_string(self) -> None:
        normalized = normalize(self.GROUPED)
        found = cribs.substitution_placements(normalized, "MEET")
        self.assertEqual(
            [p.position for p in found],
            [p.position for p in cribs.substitution_placements(self.CLEAN, "MEET")],
        )

    def test_positions_are_letter_positions_not_string_positions(self) -> None:
        """The grouped input has spaces before offset 11; the answer must not.

        DAWN sits at letter 8 of DWWDFNDWGDZQ. In "DWWDF NDWGD ZQ" that letter
        is at string index 9, and reporting 9 would be a bug that only shows up
        on real competition ciphertext.
        """
        found = cribs.caesar_placements("DWWDF NDWGD ZQ", "DAWN")
        self.assertEqual([(p.shift, p.positions) for p in found], [(3, (8,))])


class TestEmptyAndOversizedInput(unittest.TestCase):
    def test_empty_ciphertext_does_not_crash_anything(self) -> None:
        self.assertEqual(cribs.substitution_placements("", "THE"), [])
        self.assertEqual(cribs.caesar_placements("", "THE"), [])
        self.assertEqual(cribs.affine_placements("", "THE"), [])
        self.assertEqual(cribs.vigenere_placements("", "THE"), [])
        self.assertFalse(cribs.transposition_crib_help("", "THE").possible)

    def test_a_crib_longer_than_the_text_reports_nothing_tested(self) -> None:
        """Not the same as "ruled out", and the report must not say it is."""
        report = cribs.test_crib("SHORT", "IMMEDIATELY")
        self.assertEqual(report.offsets_tested, 0)
        rendered = report.render()
        self.assertIn("nothing is ruled out", rendered)
        self.assertIn("does not fit", report.summary())

    def test_the_empty_report_still_renders(self) -> None:
        rendered = cribs.test_crib("", "THE").render()
        self.assertIn("Crib test", rendered)
        self.assertIn("POSSIBILITY", rendered)

    def test_suggestions_for_a_text_too_short_for_any_crib(self) -> None:
        self.assertEqual(cribs.suggest_cribs("AB"), [])
        self.assertIn("none", cribs.describe_suggestions("AB"))


class TestInvalidInputRaises(unittest.TestCase):
    """Every bad input must explain itself rather than fail silently."""

    def assert_explains(self, error: Exception) -> None:
        self.assertTrue(str(error).strip(), "the error message must not be empty")

    def test_an_empty_crib_is_refused(self) -> None:
        for bad in ("", "   ", "1234", "!!"):
            with self.subTest(crib=bad):
                with self.assertRaises(ValueError) as caught:
                    cribs.substitution_placements("ABCDEF", bad)
                self.assert_explains(caught.exception)
                self.assertIn("crib", str(caught.exception))

    def test_a_non_string_crib_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            cribs.caesar_placements("ABCDEF", 42)  # type: ignore[arg-type]
        self.assert_explains(caught.exception)

    def test_a_non_text_ciphertext_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            cribs.substitution_placements(["A", "B"], "THE")  # type: ignore[arg-type]
        self.assert_explains(caught.exception)

    def test_a_bad_key_length_is_refused(self) -> None:
        for bad in (0, -3, "5", 2.5, True):
            with self.subTest(key_length=bad):
                with self.assertRaises(ValueError) as caught:
                    cribs.vigenere_placements(
                        "ABCDEFGH", "THE", bad  # type: ignore[arg-type]
                    )
                self.assert_explains(caught.exception)
                self.assertIn("key_length", str(caught.exception))

    def test_an_unknown_method_name_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            cribs.test_crib("ABCDEFGH", "THE", methods=["substitution", "enigma"])
        message = str(caught.exception)
        self.assert_explains(caught.exception)
        self.assertIn("enigma", message)
        self.assertIn("vigenere", message)  # it lists what IS valid

    def test_a_single_string_of_methods_is_refused(self) -> None:
        """"substitution" would iterate as 's', 'u', 'b', ... and test nothing."""
        with self.assertRaises(ValueError) as caught:
            cribs.test_crib("ABCDEFGH", "THE", methods="substitution")
        self.assert_explains(caught.exception)

    def test_a_bad_known_key_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            cribs.substitution_placements(
                "ABCDEF", "THE", known="QE XT"  # type: ignore[arg-type]
            )
        self.assert_explains(caught.exception)


class TestFailureModesAreReportedHonestly(unittest.TestCase):
    """Things that should NOT solve, and the tool saying so."""

    def test_a_polyalphabetic_ciphertext_rules_out_caesar_and_affine(self) -> None:
        """The crib IS in the plaintext, and Caesar still finds nothing.

        That is the correct answer, and the interesting one: a tool that
        reported a "best" shift here would be inventing evidence.
        """
        plaintext = corpus_letters()
        ciphertext = vigenere_encrypt(plaintext, "LEMON")
        self.assertIn("QUESTION", plaintext)
        self.assertEqual(cribs.caesar_placements(ciphertext, "QUESTION"), [])
        self.assertEqual(cribs.affine_placements(ciphertext, "QUESTION"), [])

        report = cribs.test_crib(ciphertext, "QUESTION", key_length=5)
        self.assertNotIn("caesar", report.possible_methods())
        self.assertNotIn("affine", report.possible_methods())
        self.assertIn("vigenere", report.possible_methods())

    def test_a_crib_that_cannot_fit_gives_an_empty_list_not_a_guess(self) -> None:
        """AA needs a doubled letter; this ciphertext has none."""
        self.assertEqual(cribs.substitution_placements("ABCDEFG", "AA"), [])
        report = cribs.test_crib("ABCDEFG", "AA", methods=["substitution"])
        self.assertIn("No offset survives", report.render())

    def test_a_wrong_partial_key_removes_the_true_placement(self) -> None:
        """Contradicting the truth must empty the list, not bend around it."""
        plaintext = corpus_letters()
        ciphertext = substitution_encrypt(plaintext, KEYBOARD_ALPHABET)
        offset = plaintext.find("UNDERSTOOD")
        wrong = SubstitutionKey({ciphertext[offset]: "Z"})
        self.assertNotEqual(KEYBOARD_MAPPING[ciphertext[offset]], "Z")
        self.assertEqual(
            cribs.substitution_placements(ciphertext, "UNDERSTOOD", known=wrong), []
        )

    def test_a_false_crib_never_produces_the_real_key(self) -> None:
        """ZEBRA is not in the plaintext. Every offset "fits", none is right.

        This is the honest shape of a weak crib: 396 possibilities reported,
        no claim made about any of them, and the true key nowhere among them.
        """
        plaintext = corpus_letters()
        ciphertext = vigenere_encrypt(plaintext, "LEMON")
        self.assertNotIn("ZEBRA", plaintext)

        fragments = cribs.consistent_fragments(
            cribs.vigenere_placements(ciphertext, "ZEBRA", 5)
        )
        self.assertGreater(len(fragments), 100)
        self.assertNotIn("LEMON", {fragment.partial_key for fragment in fragments})

        report = cribs.test_crib(ciphertext, "ZEBRA", key_length=5)
        self.assertIn("Possible is not probable", report.summary())

    def test_no_fixed_points_can_discard_the_truth(self) -> None:
        """The documented danger of the flag, observed rather than asserted.

        In XPQQTY the crib MEET fits at offset 1 with T standing for T. That is
        perfectly legal for a pencil-and-paper substitution, and switching the
        Enigma-style rule on throws the true placement away.
        """
        allowed = cribs.substitution_placements("XPQQTY", "MEET")
        self.assertEqual([p.position for p in allowed], [1])
        self.assertEqual(allowed[0].fixed_points, ("T",))

        forbidden = cribs.substitution_placements(
            "XPQQTY", "MEET", no_fixed_points=True
        )
        self.assertEqual(forbidden, [])

    def test_the_report_marks_a_fixed_point_rather_than_hiding_it(self) -> None:
        rendered = cribs.test_crib("XPQQTY", "MEET", methods=["substitution"]).render()
        self.assertIn("stand for", rendered)


class TestReportStructure(unittest.TestCase):
    def test_untested_is_not_the_same_as_nothing_found(self) -> None:
        report = cribs.test_crib("DWWDFNDWGDZQ", "ATTACK", methods=["caesar"])
        self.assertEqual(report.methods, ("caesar",))
        self.assertIsNone(report.substitution)   # never tested
        self.assertIsNone(report.transposition)  # never tested
        self.assertTrue(report.caesar)           # tested, and found something

    def test_methods_are_reported_in_the_module_order(self) -> None:
        report = cribs.test_crib(
            "DWWDFNDWGDZQ", "ATTACK", methods=["vigenere", "caesar"]
        )
        self.assertEqual(report.methods, ("caesar", "vigenere"))

    def test_no_methods_means_all_methods(self) -> None:
        for argument in (None, []):
            with self.subTest(methods=argument):
                report = cribs.test_crib("DWWDFNDWGDZQ", "AT", methods=argument)
                self.assertEqual(report.methods, cribs.METHODS)

    def test_the_report_never_claims_a_solution(self) -> None:
        rendered = cribs.test_crib("DWWDFNDWGDZQ", "ATTACK").render()
        for overclaim in ("solved", "the answer is", "definitely", "certainly"):
            self.assertNotIn(overclaim, rendered.lower())
        self.assertIn("POSSIBILITY", rendered)

    def test_a_very_short_crib_is_flagged_as_almost_useless(self) -> None:
        rendered = cribs.test_crib("DWWDFNDWGDZQ", "AT").render()
        self.assertIn("WARNING", rendered)

    def test_possible_methods_lists_only_families_still_standing(self) -> None:
        ciphertext = caesar_encrypt(
            "MEETMEATTHEHARBOURATMIDNIGHTANDBRINGTHEPAPERS", 7
        )
        report = cribs.test_crib(ciphertext, "HARBOUR")
        self.assertIn("caesar", report.possible_methods())
        # No R in this ciphertext, so no rearrangement of it holds HARBOUR.
        self.assertNotIn("transposition", report.possible_methods())

    def test_a_placement_can_be_turned_into_a_usable_key(self) -> None:
        placement = cribs.substitution_placements("XPQQZY", "MEET")[0]
        key = placement.key()
        self.assertIsInstance(key, SubstitutionKey)
        self.assertEqual(key.get("Q"), "E")
        self.assertFalse(key.is_complete)
        self.assertIn("offset 1", placement.describe())


class TestSuggestions(unittest.TestCase):
    def test_suggestions_are_stock_cribs_that_fit(self) -> None:
        words = cribs.suggest_cribs(corpus_letters())
        self.assertIn("MESSAGE", words)
        self.assertIn("THE", words)
        self.assertTrue(all(len(word) >= 3 for word in words))

    def test_suggestions_are_longest_first(self) -> None:
        lengths = [len(word) for word in cribs.suggest_cribs(corpus_letters())]
        self.assertEqual(lengths, sorted(lengths, reverse=True))

    def test_a_short_text_drops_the_long_suggestions(self) -> None:
        words = cribs.suggest_cribs("A" * 12)
        self.assertIn("THE", words)
        self.assertNotIn("IMMEDIATELY", words)

    def test_suggestions_accept_a_statistics_object(self) -> None:
        from cipher_tool.statistics import analyse

        stats = analyse(corpus_letters())
        self.assertEqual(cribs.suggest_cribs(stats), cribs.suggest_cribs(corpus_letters()))

    def test_suggestions_are_labelled_as_guesses(self) -> None:
        described = cribs.describe_suggestions(corpus_letters())
        self.assertIn("GUESSES", described)
        self.assertIn("not evidence", described)


if __name__ == "__main__":
    unittest.main()
