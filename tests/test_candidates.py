"""Tests for candidate management and the honesty of the reports."""

from __future__ import annotations

import unittest

from cipher_tool.candidates import (
    Candidate,
    CandidateSet,
    looks_degenerate,
    render_candidate,
    render_candidates,
)

#: A stand-in plaintext for tests about the LABEL rather than the content.
#: It has to spread its letters realistically: `confidence()` refuses to call
#: a reading made of one repeated letter better than `weak`, so the "X" * 40
#: placeholder these tests used to carry was measuring the degeneracy guard
#: instead of the band under test.
SPREAD = ("THEQUICKBROWNFOXJUMPSOVERTHELAZYDOGWHILE"
          "SEVERALMORECLAUSESFILLOUTTHESENTENCE")


def make(method: str = "Caesar", key: str = "shift=3", score: float = -100.0,
         plaintext: str = "ATTACKATDAWN", **diagnostics) -> Candidate:
    return Candidate(method=method, key=key, score=score, plaintext=plaintext,
                     diagnostics=dict(diagnostics))


class TestCandidate(unittest.TestCase):
    def test_normalised_score(self) -> None:
        candidate = make(score=-24.0, plaintext="ABCDEFGHIJKL")
        self.assertAlmostEqual(candidate.normalised_score, -2.0)

    def test_normalised_score_of_empty_plaintext(self) -> None:
        self.assertEqual(make(plaintext="").normalised_score, float("-inf"))

    def test_preview_truncates(self) -> None:
        candidate = make(plaintext="A" * 200)
        self.assertEqual(len(candidate.preview(40)), 40)
        self.assertTrue(candidate.preview(40).endswith("..."))

    def test_preview_collapses_whitespace(self) -> None:
        candidate = make(plaintext="AB")
        candidate.display = "AB   CD\n\nEF"
        self.assertEqual(candidate.preview(), "AB CD EF")


class TestConfidence(unittest.TestCase):
    def test_strong_needs_both_signals(self) -> None:
        candidate = make(normalised_score=-0.9, word_coverage=0.85)
        self.assertEqual(candidate.confidence(), "strong")

    def test_good_ngrams_but_poor_coverage_is_not_strong(self) -> None:
        # This is the near-miss case: English-looking letter statistics that
        # do not actually cut into words. It must not be labelled strong.
        candidate = make(normalised_score=-0.9, word_coverage=0.20)
        self.assertNotEqual(candidate.confidence(), "strong")

    def test_promising_band(self) -> None:
        candidate = make(normalised_score=-1.4, word_coverage=0.50)
        self.assertEqual(candidate.confidence(), "promising")

    def test_unlikely(self) -> None:
        candidate = make(normalised_score=-2.8, word_coverage=0.05)
        self.assertEqual(candidate.confidence(), "unlikely")

    def test_missing_coverage_is_capped_below_strong(self) -> None:
        # Without the second signal we cannot justify "strong", however good
        # the n-gram score looks.
        candidate = make(normalised_score=-0.5)
        self.assertEqual(candidate.confidence(), "promising")

    def test_never_reports_solved(self) -> None:
        candidate = make(normalised_score=-0.5, word_coverage=1.0)
        self.assertNotIn("solved", candidate.confidence().lower())


class TestCandidateSet(unittest.TestCase):
    def test_ranked_best_first(self) -> None:
        candidates = CandidateSet([
            make(key="a", score=-300.0, plaintext="AAA"),
            make(key="b", score=-100.0, plaintext="BBB"),
            make(key="c", score=-200.0, plaintext="CCC"),
        ])
        self.assertEqual([c.key for c in candidates.ranked()], ["b", "c", "a"])
        self.assertEqual(candidates.best().key, "b")

    def test_duplicates_merge_and_count_agreements(self) -> None:
        candidates = CandidateSet()
        for _ in range(4):
            candidates.add(make(plaintext="SAME"))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates.best().diagnostics["agreements"], 4)

    def test_duplicate_keeps_the_higher_score(self) -> None:
        candidates = CandidateSet()
        candidates.add(make(key="low", score=-500.0, plaintext="SAME"))
        candidates.add(make(key="high", score=-100.0, plaintext="SAME"))
        best = candidates.best()
        self.assertEqual(best.score, -100.0)
        self.assertEqual(best.key, "high")
        self.assertEqual(best.diagnostics["agreements"], 2)

    def test_same_plaintext_from_different_methods_is_not_a_duplicate(self) -> None:
        candidates = CandidateSet([
            make(method="Caesar", plaintext="SAME"),
            make(method="Affine", plaintext="SAME"),
        ])
        self.assertEqual(len(candidates), 2)

    def test_empty_set(self) -> None:
        candidates = CandidateSet()
        self.assertFalse(candidates)
        self.assertIsNone(candidates.best())
        self.assertEqual(candidates.top(5), [])
        self.assertIsNone(candidates.score_gap())

    def test_top_handles_non_positive_counts(self) -> None:
        candidates = CandidateSet([make()])
        self.assertEqual(candidates.top(0), [])
        self.assertEqual(candidates.top(-1), [])

    def test_score_gap(self) -> None:
        candidates = CandidateSet([
            make(key="a", score=-10.0, plaintext="ABCDEFGHIJ"),
            make(key="b", score=-20.0, plaintext="KLMNOPQRST"),
        ])
        self.assertAlmostEqual(candidates.score_gap(), 1.0)


class TestRendering(unittest.TestCase):
    def test_single_candidate_shows_the_required_fields(self) -> None:
        text = render_candidate(make(normalised_score=-0.9, word_coverage=0.9), 1)
        for field in ("Candidate 1", "Method:", "Key/config:", "Score:",
                      "Confidence:", "Plaintext:"):
            self.assertIn(field, text)

    def test_confidence_is_labelled_a_heuristic(self) -> None:
        self.assertIn("[heuristic]", render_candidate(make()))

    def test_list_always_carries_the_uncertainty_note(self) -> None:
        text = render_candidates([make(), make(plaintext="OTHER")])
        self.assertIn("ranked guesses", text)
        self.assertIn("Read the plaintext", text)

    def test_empty_list_says_so(self) -> None:
        self.assertIn("No candidates", render_candidates([]))

    def test_top_limits_output(self) -> None:
        many = [make(plaintext=f"PLAIN{i}") for i in range(10)]
        text = render_candidates(many, top=3)
        self.assertIn("Candidate 3", text)
        self.assertNotIn("Candidate 4", text)

    def test_full_text_mode_prints_everything(self) -> None:
        candidate = make(plaintext="Q" * 300)
        text = render_candidates([candidate], full_text=True)
        self.assertEqual(text.count("Q"), 300)

    def test_diagnostics_are_shown_as_evidence(self) -> None:
        text = render_candidate(make(restarts=25, mean_ic=0.0667))
        self.assertIn("Evidence:", text)
        self.assertIn("restarts: 25", text)
        self.assertIn("0.0667", text)


class TestPartlyEnglishIsNotCalledFailure(unittest.TestCase):
    """A correct solve whose message embeds non-prose must not read as weak.

    From the 2017 challenge 5A, which the toolkit decrypted PERFECTLY and
    then reported as weak. The message carries a steganographic frieze --
    1,500 letters of black and white tiles, enciphered with the words -- so
    scored whole the right answer is -2.070 per letter with 36 per cent word
    coverage. Windowed, three quarters of the prose is plainly English.

    The label is raised no further than `promising`, and deliberately: part
    of the message genuinely is not English, so `strong` would overstate it.
    But telling somebody their correct answer failed is worse than either.
    """

    def test_a_partly_english_reading_is_promoted_to_promising(self) -> None:
        candidate = make(plaintext=SPREAD, score=-2.07 * len(SPREAD),
                         normalised_score=-2.07, word_coverage=0.36,
                         english_fraction=0.40)
        self.assertEqual(candidate.confidence(), "promising")

    def test_it_is_not_promoted_all_the_way_to_strong(self) -> None:
        candidate = make(plaintext=SPREAD, score=-2.07 * len(SPREAD),
                         normalised_score=-2.07, word_coverage=0.36,
                         english_fraction=0.95)
        self.assertNotEqual(candidate.confidence(), "strong")

    def test_noise_with_no_english_portion_stays_where_it_was(self) -> None:
        candidate = make(plaintext=SPREAD, score=-3.0 * len(SPREAD),
                         normalised_score=-3.0, word_coverage=0.05,
                         english_fraction=0.0)
        self.assertIn(candidate.confidence(), {"weak", "unlikely"})

    def test_a_strong_reading_is_never_weakened_by_this(self) -> None:
        candidate = make(plaintext=SPREAD, score=-0.8 * len(SPREAD),
                         normalised_score=-0.8, word_coverage=0.9,
                         english_fraction=1.0)
        self.assertEqual(candidate.confidence(), "strong")


class TestDegenerateReadingsAreNotSold(unittest.TestCase):
    """A reading made of one short word repeated is not an answer at all.

    Both existing signals read the plaintext and neither counts how many
    DIFFERENT letters it uses, so `ANDANDAND...` scores -0.637 per letter
    with word coverage 1.000 -- better than genuine English at -0.710 -- and
    was labelled `strong`. A search with more key freedom than ciphertext
    collapses onto exactly this shape, so the scorer was rewarding the
    collapse instead of catching it.
    """

    def test_a_repeated_word_is_not_strong(self) -> None:
        candidate = make(plaintext="AND" * 209, score=-0.637 * 627,
                         normalised_score=-0.637, word_coverage=1.0)
        self.assertIn(candidate.confidence(), {"weak", "unlikely"})

    def test_two_letters_alternating_is_not_strong(self) -> None:
        candidate = make(plaintext="ID" * 313, score=-0.756 * 626,
                         normalised_score=-0.756, word_coverage=0.748)
        self.assertIn(candidate.confidence(), {"weak", "unlikely"})

    def test_the_partial_prose_promotion_cannot_rescue_it(self) -> None:
        # english_fraction promotes weak -> promising, and a degenerate text
        # passes a window-by-window prose test perfectly. The guard is applied
        # after that promotion for exactly this reason.
        candidate = make(plaintext="THE" * 209, score=-2.5 * 627,
                         normalised_score=-2.5, word_coverage=0.30,
                         english_fraction=1.0)
        self.assertIn(candidate.confidence(), {"weak", "unlikely"})

    def test_a_short_real_plaintext_is_never_rejected(self) -> None:
        # ATTACKATDAWN is a legitimate plaintext whose two commonest letters
        # are 58 per cent of it. Short text is repetitive by nature.
        candidate = make(plaintext="ATTACKATDAWN", score=-0.9 * 12,
                         normalised_score=-0.9, word_coverage=0.85)
        self.assertEqual(candidate.confidence(), "strong")

    def test_genuine_english_is_untouched(self) -> None:
        candidate = make(plaintext=SPREAD, score=-0.9 * len(SPREAD),
                         normalised_score=-0.9, word_coverage=0.85)
        self.assertEqual(candidate.confidence(), "strong")

    def test_it_reports_a_reason(self) -> None:
        self.assertIsNone(looks_degenerate(SPREAD))
        self.assertIn("commonest", looks_degenerate("AND" * 209))
        self.assertIn("different letters",
                      looks_degenerate(("ABCDEFGHIJKLMNO" * 20)[:300]))


if __name__ == "__main__":
    unittest.main()
