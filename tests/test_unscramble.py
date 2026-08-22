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

import pytest

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
    table = str.maketrans("".join(letters), "".join(shuffled))
    return text.translate(table)


def scramble(text: str, permutation: tuple[int, ...]) -> str:
    """Apply a block permutation -- the thing the detector must find."""
    width = len(permutation)
    out = []
    for start in range(0, len(text) - width + 1, width):
        block = text[start:start + width]
        out.append("".join(block[source] for source in permutation))
    return "".join(out)


def test_finds_a_period_four_scramble_under_a_substitution() -> None:
    """The 2025 6A shape: substitute, then shuffle in blocks of four."""
    ciphertext = scramble(substitute(PLAIN), (0, 3, 2, 1))
    found = unscramble.detect(ciphertext)
    assert found is not None, "the scramble was missed entirely"
    assert found.width == 4
    assert found.text[:40] == substitute(PLAIN)[:40], (
        "detected, but the recovered order is wrong"
    )


def test_identity_is_not_a_finding() -> None:
    """A plain substitution is NOT scrambled, so the detector must stay silent.

    Pins the real defect: allowing the identity permutation let the detector
    "discover" that it had undone nothing, and report that as a scramble.
    """
    assert unscramble.detect(substitute(PLAIN)) is None


def test_silent_on_ordinary_english() -> None:
    """Unenciphered text is not scrambled either."""
    assert unscramble.detect(PLAIN) is None


def test_silent_on_random_letters() -> None:
    """Noise has no order to restore, so there is nothing to find in it."""
    letters = list(PLAIN)
    random.Random(1).shuffle(letters)
    assert unscramble.detect("".join(letters)) is None


def test_refuses_below_the_minimum_length() -> None:
    """Too few bigrams for the statistic to mean anything."""
    assert unscramble.detect("MYDEARBABBAGE" * 3) is None


@pytest.mark.parametrize("permutation", [(1, 0), (0, 2, 1), (2, 0, 3, 1)])
def test_recovers_several_widths(permutation: tuple[int, ...]) -> None:
    ciphertext = scramble(substitute(PLAIN), permutation)
    found = unscramble.detect(ciphertext)
    assert found is not None
    assert found.text[:40] == substitute(PLAIN)[:40]


def test_concentration_ignores_letter_names() -> None:
    """The statistic must be substitution-invariant, or the whole method fails.

    This is the load-bearing assumption: it is what lets the transposition be
    stripped off before the cipher underneath is known.
    """
    assert unscramble.concentration(PLAIN) == pytest.approx(
        unscramble.concentration(substitute(PLAIN))
    )
