"""Tests for the block-permutation detector.

The two failures worth pinning are opposite ones, and only one of them is
obvious. Missing a real scramble costs a challenge. **Firing on a message that
was never scrambled is worse**, because it hands the pipeline a mangled text
and every later stage then searches a message the setter never wrote.

That second failure is not hypothetical: the first version of ``detect`` fired
on three unscrambled ciphers, because it allowed the IDENTITY permutation to
win. Undoing nothing preserves the natural bigram structure, which beats a
shuffled control on any English-derived text. ``test_identity_is_not_a_finding``
is that defect, pinned.
"""

from __future__ import annotations

import random
import unittest

from cipher_tool import unscramble

PLAIN = (
    "MYDEARBABBAGEIAMWRITINGTOTHANKYOUFORFORWARDINGTHEFINALDESIGNSFORYOUR"
    "PROTOTYPEIHAVEMANAGEDTOINTEGRATETHEDEVICEWITHTHEMACHINESATSOMEOFOUR"
    "TELEGRAPHSTATIONSANDHAVEBEENRUNNINGITONSAMPLETEXTINTERCEPTEDFROMTHE"
    "TRAFFICPASSINGTHROUGHTHOSEOFFICESINGENERALMYENGINEERSHAVEBEENHIGHLY"
    "IMPRESSEDBYYOURMECHANISMTHEYWEREASTONISHEDATTHESPEEDWITHWHICHITWAS"
    "ABLETOPARSETHETRAFFICANDTOCORRECTLYFLAGMESSAGESTHATDESERVEORREQUIRE"
    "GREATERATTENTIONCURIOUSLYHOWEVERONEOFTHETELEGRAMSHASRESISTEDATTACK"
    "BOTHBYYOURCOMPUTERANDBYMYCRYPTOLOGISTSDESPITETHEIRBESTEFFORTSTHEY"
    "WEREUNABLETODETERMINETHENATUREOFTHETEXTITCONTAINS"
) * 2


def substitute(text: str, seed: int = 7) -> str:
    """A monoalphabetic substitution -- must be invisible to the detector."""
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    shuffled = letters[:]
    random.Random(seed).shuffle(shuffled)
    return text.translate(str.maketrans("".join(letters), "".join(shuffled)))


def scramble(text: str, permutation: tuple[int, ...]) -> str:
    """Apply a block permutation -- the thing the detector must find."""
    width = len(permutation)
    out = []
    for start in range(0, len(text) - width + 1, width):
        block = text[start:start + width]
        out.append("".join(block[source] for source in permutation))
    return "".join(out)


class DetectScramble(unittest.TestCase):
    def test_finds_a_period_four_scramble_under_a_substitution(self) -> None:
        """The 2025 6A shape: substitute, then shuffle in blocks of four."""
        ciphertext = scramble(substitute(PLAIN), (0, 3, 2, 1))
        found = unscramble.detect(ciphertext)
        self.assertIsNotNone(found, "the scramble was missed entirely")
        assert found is not None
        self.assertEqual(found.width, 4)
        self.assertEqual(
            found.text[:40], substitute(PLAIN)[:40],
            "detected, but the recovered order is wrong",
        )

    def test_recovers_several_widths(self) -> None:
        for permutation in ((1, 0), (0, 2, 1), (2, 0, 3, 1)):
            with self.subTest(permutation=permutation):
                ciphertext = scramble(substitute(PLAIN), permutation)
                found = unscramble.detect(ciphertext)
                self.assertIsNotNone(found)
                assert found is not None
                self.assertEqual(found.text[:40], substitute(PLAIN)[:40])


class StaysSilent(unittest.TestCase):
    """The dangerous direction: firing on something that is not scrambled."""

    def test_identity_is_not_a_finding(self) -> None:
        """A plain substitution is NOT scrambled, so it must stay silent.

        Pins the real defect: allowing the identity permutation let the
        detector "discover" that it had undone nothing, and report it.
        """
        self.assertIsNone(unscramble.detect(substitute(PLAIN)))

    def test_silent_on_ordinary_english(self) -> None:
        self.assertIsNone(unscramble.detect(PLAIN))

    def test_silent_on_random_letters(self) -> None:
        """Noise has no order to restore, so there is nothing to find."""
        letters = list(PLAIN)
        random.Random(1).shuffle(letters)
        self.assertIsNone(unscramble.detect("".join(letters)))

    def test_refuses_below_the_minimum_length(self) -> None:
        self.assertIsNone(unscramble.detect("MYDEARBABBAGE" * 3))


class Statistic(unittest.TestCase):
    def test_concentration_ignores_letter_names(self) -> None:
        """Substitution-invariance is the load-bearing assumption.

        It is what lets the transposition be stripped off before the cipher
        underneath is known. If this ever fails, the whole method fails.
        """
        self.assertAlmostEqual(
            unscramble.concentration(PLAIN),
            unscramble.concentration(substitute(PLAIN)),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
