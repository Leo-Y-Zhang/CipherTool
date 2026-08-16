"""Tests for the local English scoring model.

The important tests here are the calibration ones. The thresholds in
``candidates.py`` are derived from these measurements, so if the corpus
changes and these fail, the thresholds need revisiting -- that is the point.
"""

from __future__ import annotations

import random
import unittest

from cipher_tool.normalize import ALPHABET, letters_only
from cipher_tool.scoring import (
    EnglishScorer,
    corpus_files,
    default_scorer,
    load_corpus_text,
    normalised_score,
    score_text,
)


def _shift(text: str, amount: int) -> str:
    return "".join(ALPHABET[(ord(c) - 65 + amount) % 26] for c in text)


class TestCorpus(unittest.TestCase):
    def test_corpus_files_exist(self) -> None:
        files = corpus_files()
        self.assertGreaterEqual(len(files), 1)
        for path in files:
            self.assertGreater(path.stat().st_size, 1000, f"{path} is tiny")

    def test_corpus_is_pure_ascii(self) -> None:
        # Non-ASCII in the corpus would fold unpredictably into the model.
        text = load_corpus_text()
        offenders = {char for char in text if ord(char) > 127}
        self.assertEqual(offenders, set(), f"non-ASCII in corpus: {offenders}")

    def test_corpus_is_large_enough(self) -> None:
        self.assertGreater(len(letters_only(load_corpus_text())), 50_000)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = default_scorer()

    def test_table_has_the_right_shape(self) -> None:
        table = self.scorer.table()
        self.assertEqual(len(table), 26**4)
        self.assertTrue(all(value < 0 for value in table[:1000]))

    def test_score_matches_a_naive_recomputation(self) -> None:
        # score_values slides a window with modular arithmetic instead of
        # re-slicing. This is exactly where an off-by-one would hide, and a
        # silently wrong scorer would poison every solver in the toolkit.
        scorer = self.scorer
        table = scorer.table()
        generator = random.Random(11)
        for _ in range(150):
            values = [generator.randrange(26)
                      for _ in range(generator.randint(0, 40))]
            if not values:
                self.assertEqual(scorer.score_values(values), 0.0)
                continue
            expected = scorer._log1[values[0]]
            if len(values) > 1:
                expected += scorer._log2[values[0] * 26 + values[1]]
            if len(values) > 2:
                expected += scorer._log3[
                    (values[0] * 26 + values[1]) * 26 + values[2]
                ]
            for i in range(3, len(values)):
                a, b, c, d = values[i - 3], values[i - 2], values[i - 1], values[i]
                expected += table[((a * 26 + b) * 26 + c) * 26 + d]
            self.assertAlmostEqual(scorer.score_values(values), expected, places=9)

    def test_english_beats_random(self) -> None:
        english = "THEREISNOTHINGSOFATALTOCHARACTERASHALFFINISHEDTASKS"
        generator = random.Random(3)
        noise = "".join(generator.choice(ALPHABET) for _ in range(len(english)))
        self.assertGreater(score_text(english), score_text(noise))

    def test_ignores_case_and_punctuation(self) -> None:
        self.assertAlmostEqual(
            score_text("Attack at dawn!"), score_text("ATTACKATDAWN"), places=9
        )

    def test_empty_text(self) -> None:
        self.assertEqual(score_text(""), 0.0)
        self.assertEqual(normalised_score(""), float("-inf"))

    def test_rejects_a_corpus_that_is_too_small(self) -> None:
        with self.assertRaises(ValueError) as context:
            EnglishScorer("hello world")
        self.assertIn("too small", str(context.exception))


class TestWordCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = default_scorer()

    def test_english_covers_well(self) -> None:
        text = "THEMESSAGEWASSENTFROMTHEHARBOURBEFOREMIDNIGHT"
        self.assertGreater(self.scorer.word_coverage(text), 0.8)

    def test_random_covers_badly(self) -> None:
        generator = random.Random(5)
        noise = "".join(generator.choice(ALPHABET) for _ in range(300))
        self.assertLess(self.scorer.word_coverage(noise), 0.25)

    def test_empty(self) -> None:
        self.assertEqual(self.scorer.word_coverage(""), 0.0)

    def test_find_words_reports_real_words(self) -> None:
        found = self.scorer.find_words("XXTHROUGHXXMESSAGEXX", minimum_length=5)
        self.assertIn("THROUGH", found)
        self.assertIn("MESSAGE", found)

    def test_lexicon_excludes_stray_single_letters(self) -> None:
        lexicon = self.scorer.lexicon
        self.assertIn("A", lexicon)
        self.assertIn("I", lexicon)
        for letter in "BCDFGHJKLMNOPQRSTUVWXYZ":
            self.assertNotIn(letter, lexicon)


class TestCalibrationSeparation(unittest.TestCase):
    """The measurements the confidence thresholds are built on.

    Trains on all corpus files but the last and tests on the last, so the
    model has genuinely never seen the text it is judging.
    """

    @classmethod
    def setUpClass(cls) -> None:
        files = corpus_files()
        training = "\n".join(
            path.read_text(encoding="utf-8") for path in files[:-1]
        )
        cls.scorer = EnglishScorer(training)
        cls.held_out = letters_only(files[-1].read_text(encoding="utf-8"))

    def _samples(self, size: int = 300, count: int = 12) -> list[str]:
        return [
            self.held_out[i * size : (i + 1) * size] for i in range(count)
        ]

    def test_held_out_english_scores_above_the_strong_threshold(self) -> None:
        from cipher_tool.candidates import _STRONG_NGRAM

        for sample in self._samples():
            self.assertGreater(
                self.scorer.normalised(sample),
                _STRONG_NGRAM,
                "unseen English should clear the 'strong' n-gram threshold",
            )

    def test_wrong_decryptions_score_below_the_weak_threshold(self) -> None:
        from cipher_tool.candidates import _WEAK_NGRAM

        for sample in self._samples():
            self.assertLess(
                self.scorer.normalised(_shift(sample, 7)),
                _WEAK_NGRAM,
                "a wrong Caesar shift must not reach even 'weak'",
            )

    def test_a_near_miss_key_lands_between_the_two(self) -> None:
        # Swap two letters of the identity key: mostly-right plaintext.
        from cipher_tool.candidates import _STRONG_NGRAM, _WEAK_NGRAM

        table = list(ALPHABET)
        table[4], table[19] = table[19], table[4]  # E <-> T
        mapping = dict(zip(ALPHABET, table))
        scores = [
            self.scorer.normalised("".join(mapping[c] for c in sample))
            for sample in self._samples()
        ]
        average = sum(scores) / len(scores)
        self.assertLess(average, _STRONG_NGRAM)
        self.assertGreater(average, _WEAK_NGRAM)

    def test_coverage_separates_english_from_noise(self) -> None:
        english = [self.scorer.word_coverage(s) for s in self._samples()]
        wrong = [
            self.scorer.word_coverage(_shift(s, 7)) for s in self._samples()
        ]
        self.assertGreater(min(english), 0.6)
        self.assertLess(max(wrong), 0.35)


if __name__ == "__main__":
    unittest.main()
