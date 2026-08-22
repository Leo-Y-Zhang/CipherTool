"""A letterless paste must not be treated as an empty one.

``auto_solve`` used to stop at ``normalized.is_empty`` with "the input contains
no letters, so nothing was attempted", and return before a single stage ran.
Polybius, Nihilist and straddling-checkerboard ciphertexts are ALL DIGITS, so
that early exit hid a whole family: 2023 challenge 8B, 2,714 digits over five
symbols, was refused in 0.0 seconds by a toolkit that already contained the
solver that cracks it in seven.

These tests pin the two halves of that fix, and the second one is the half that
was wrong on the first attempt.
"""

from __future__ import annotations

from cipher_tool import auto, polybius

PLAIN = (
    "MYINVESTIGATIONSOFTHEINCIDENTHAVEBEENMOREFRUITFULTHANICOULDHAVEHOPED"
    "PINKERTONSHAVEPROVIDEDCLEAREVIDENCETHATTHEREWASACONSPIRACYTOSTRIKE"
    "ATTHEHEARTOFGOVERNMENTANDTHATTHEPLOTREACHEDFURTHERTHANWEFEARED"
    "IHAVESETOUTBELOWWHATWEKNOWANDWHATWEONLYSUSPECTSOTHATYOUMAYJUDGE"
    "FORYOURSELFWHETHERTOBRINGTHISBEFORETHECABINET"
) * 3


def as_digits(text: str) -> str:
    """Encrypt through the classical numeric 5x5 square."""
    return polybius.encrypt(text)


def test_a_pure_digit_paste_is_solved_not_refused() -> None:
    """The 2023 8B shape: no letters at all, and a real message underneath."""
    result = auto.auto_solve(as_digits(PLAIN), effort="fast", top=3, seed=1)
    best = result.candidates.best()
    assert best is not None, "a letterless paste was refused outright"
    assert "MYINVESTIGATIONS" in best.plaintext
    assert any(stage.ran for stage in result.stages), (
        "no stage ran at all -- the early exit is back"
    )


def test_an_odd_digit_count_still_solves() -> None:
    """One stray digit must not cost the whole message.

    The first version of the fix required an EVEN count and threw away two more
    real challenges -- 2024 7B (2,315 digits) and 8B (3,025) -- because
    hand-copied competition material routinely gains or loses a symbol. The
    final unpaired digit is dropped; a leading one would misalign every pair
    after it.
    """
    result = auto.auto_solve(as_digits(PLAIN) + "3", effort="fast", top=3,
                             seed=1)
    best = result.candidates.best()
    assert best is not None, "an odd digit count refused the whole message"
    assert "MYINVESTIGATIONS" in best.plaintext


def test_the_oddity_is_reported_not_hidden() -> None:
    ran = [s for s in auto.auto_solve(as_digits(PLAIN) + "3", effort="fast",
                                      top=1, seed=1).stages if s.ran]
    assert any("odd count" in stage.note for stage in ran), (
        "a digit was dropped and the report did not say so"
    )


def test_a_genuinely_empty_paste_is_still_refused() -> None:
    """The early exit must survive for the case it was written for."""
    result = auto.auto_solve("....  ----  ....", effort="fast", top=1, seed=1)
    assert result.candidates.best() is None
    assert not any(stage.ran for stage in result.stages)


def test_a_short_digit_run_is_not_treated_as_a_cipher() -> None:
    """A date or a reference number is not a Polybius ciphertext."""
    result = auto.auto_solve("1234 5678 2345", effort="fast", top=1, seed=1)
    assert result.candidates.best() is None
