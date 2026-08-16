"""Tests for cipher_tool.encodings.

Two things are being tested here and the second matters more than the first.

The easy half is that each notation decodes correctly: known hand-computed
triples, round trips, and clear errors for malformed input.

The hard half is that the module does *not* claim a notation it has not got
evidence for. A tool that announces "this is hexadecimal" about a piece of
ciphertext sends its user off to analyse rubbish, so the false-positive cases
are tested explicitly: English prose, uppercase ciphertext trimmed to a
Base64-shaped length, and hexadecimal-looking words such as DEADBEEF must all
be reported as nothing at all.
"""

from __future__ import annotations

import unittest

from cipher_tool.encodings import (
    MORSE_TABLE,
    NOT_ENCRYPTION,
    EncodingGuess,
    decode_base64,
    decode_binary,
    decode_decimal,
    decode_hex,
    decode_morse,
    describe_guesses,
    encode_base64,
    encode_binary,
    encode_decimal,
    encode_hex,
    encode_morse,
    identify,
    is_base64,
    is_binary,
    is_decimal,
    is_hex,
    is_morse,
    solve,
)
from cipher_tool.normalize import letters_only, normalize
from cipher_tool.scoring import DATA_DIR


def corpus_sample(count: int = 400) -> str:
    """An ASCII slice of the expository corpus, used as "real English"."""
    raw = (DATA_DIR / "corpus_04_expository.txt").read_text(encoding="utf-8")
    ascii_only = "".join(ch for ch in raw if ch.isascii())
    sample = ascii_only[:count]
    if len(sample) < count:
        raise AssertionError("corpus file is shorter than the test needs")
    return sample


class TestHex(unittest.TestCase):
    def test_known_pair(self) -> None:
        # H=0x48 E=0x45 L=0x4C L=0x4C O=0x4F, worked out from the ASCII table.
        self.assertEqual(encode_hex("HELLO"), "48454C4C4F")
        self.assertEqual(decode_hex("48454C4C4F"), "HELLO")

    def test_layout_does_not_matter(self) -> None:
        expected = "HELLO"
        for variant in (
            "48454c4c4f",
            "48 45 4C 4C 4F",
            "48:45:4c:4c:4f",
            "48-45-4C-4C-4F",
            "0x48 0x45 0x4C 0x4C 0x4F",
            r"\x48\x45\x4C\x4C\x4F",
            "4845\n4C4C\n4F",
            "  48454C4C4F  ",
        ):
            with self.subTest(variant=variant):
                self.assertEqual(decode_hex(variant), expected)

    def test_round_trip(self) -> None:
        for text in ("A", "ATTACK AT DAWN", "Mixed Case 123!", "?"):
            with self.subTest(text=text):
                self.assertEqual(decode_hex(encode_hex(text)), text)

    def test_empty_input(self) -> None:
        self.assertEqual(decode_hex(""), "")
        self.assertEqual(encode_hex(""), "")
        self.assertFalse(is_hex(""))

    def test_invalid_input_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            decode_hex("48GG")
        self.assertTrue(str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            decode_hex("48454C4C4")  # odd number of digits
        self.assertIn("even", str(ctx.exception))
        with self.assertRaises(ValueError):
            encode_hex("caf" + chr(233))  # not ASCII: no byte value

    def test_detector_needs_a_printable_decode(self) -> None:
        # The failure mode: DEADBEEF is legal hexadecimal and decodes to four
        # high-bit bytes. Parsing is not evidence, so the tool must say no.
        self.assertFalse(is_hex("DEADBEEF"))
        self.assertEqual(identify("DEADBEEF"), [])
        self.assertTrue(is_hex("48454C4C4F"))


class TestBinary(unittest.TestCase):
    def test_known_pair(self) -> None:
        # H = 72 = 01001000, I = 73 = 01001001.
        self.assertEqual(encode_binary("HI"), "01001000 01001001")
        self.assertEqual(decode_binary("01001000 01001001"), "HI")

    def test_seven_bit_groups(self) -> None:
        self.assertEqual(decode_binary("1001000 1001001"), "HI")
        self.assertEqual(encode_binary("HI", bits=7), "1001000 1001001")

    def test_grouping_decides_the_width_when_both_widths_fit(self) -> None:
        # Eight characters at 7 bits is 56 bits, which is also seven bytes at
        # 8 bits, so the length says nothing. Worse, this particular message
        # is fully printable under BOTH readings (" PATIENT" at 7 bits,
        # "AB\rI1gT" at 8), so the printable-decode tie-break says nothing
        # either. The transcription's own grouping is the only evidence left,
        # and it is what the decoder must use.
        seven = encode_binary(" PATIENT", bits=7)
        self.assertEqual(seven.count(" "), 7)
        self.assertEqual(decode_binary(seven, bits=8), "AB\rI1gT")
        self.assertEqual(decode_binary(seven), " PATIENT")

    def test_layout_does_not_matter(self) -> None:
        for variant in (
            "0100100001001001",
            "01001000 01001001",
            "01001000,01001001",
            "01001000\n01001001",
            "  01001000   01001001  ",
        ):
            with self.subTest(variant=variant):
                self.assertEqual(decode_binary(variant), "HI")

    def test_round_trip(self) -> None:
        for text in ("HELLO WORLD", "a", "Numbers 42!"):
            for bits in (7, 8):
                with self.subTest(text=text, bits=bits):
                    self.assertEqual(
                        decode_binary(encode_binary(text, bits=bits), bits=bits), text
                    )

    def test_empty_input(self) -> None:
        self.assertEqual(decode_binary(""), "")
        self.assertFalse(is_binary(""))

    def test_invalid_input_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            decode_binary("01001002")
        self.assertTrue(str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            decode_binary("010")  # not a whole number of characters
        self.assertIn("multiple", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            decode_binary("01001000", bits=6)
        self.assertIn("7 or 8", str(ctx.exception))

    def test_detector_needs_a_printable_decode(self) -> None:
        self.assertTrue(is_binary("01001000 01001001"))
        # Eight bytes of 1s decode to 0xFF, which is not text.
        self.assertFalse(is_binary("11111111 11111111 11111111"))


class TestDecimal(unittest.TestCase):
    def test_known_pair(self) -> None:
        self.assertEqual(encode_decimal("HELLO"), "72 69 76 76 79")
        self.assertEqual(decode_decimal("72 69 76 76 79"), "HELLO")

    def test_layout_does_not_matter(self) -> None:
        for variant in (
            "72 69 76 76 79",
            "72,69,76,76,79",
            "72, 69, 76, 76, 79",
            "72;69;76;76;79",
            " 72\n69\n76 76 79 ",
        ):
            with self.subTest(variant=variant):
                self.assertEqual(decode_decimal(variant), "HELLO")

    def test_round_trip(self) -> None:
        for text in ("ATTACK AT DAWN", "z", "Punctuation: yes!"):
            with self.subTest(text=text):
                self.assertEqual(decode_decimal(encode_decimal(text)), text)

    def test_empty_input(self) -> None:
        self.assertEqual(decode_decimal(""), "")
        self.assertFalse(is_decimal(""))

    def test_invalid_input_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            decode_decimal("72 sixty-nine 76")
        self.assertTrue(str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            decode_decimal("72 300 76")
        self.assertIn("0-255", str(ctx.exception))

    def test_letter_numbering_is_not_claimed(self) -> None:
        # 8-5-12-12-15 is A1Z26 for HELLO, not ASCII. As ASCII it is control
        # characters, so the module reports nothing rather than guessing at a
        # scheme it does not implement.
        self.assertFalse(is_decimal("8 5 12 12 15"))
        self.assertEqual(
            [guess.format for guess in identify("8 5 12 12 15")],
            [],
        )


class TestBase64(unittest.TestCase):
    def test_known_pair(self) -> None:
        self.assertEqual(encode_base64("Man"), "TWFu")
        self.assertEqual(decode_base64("TWFu"), "Man")
        # Hand-computed: 01001000 01000101 01001100 -> 18 4 21 12 -> SEVM, and
        # so on to the two-byte tail LD -> TEQ= .
        self.assertEqual(encode_base64("HELLO WORLD"), "SEVMTE8gV09STEQ=")
        self.assertEqual(decode_base64("SEVMTE8gV09STEQ="), "HELLO WORLD")

    def test_line_breaks_are_tolerated(self) -> None:
        wrapped = "SEVMTE8g\nV09STEQ="
        self.assertEqual(decode_base64(wrapped), "HELLO WORLD")

    def test_round_trip(self) -> None:
        for text in ("A", "AB", "ABC", "The quick brown fox.", "12345678"):
            with self.subTest(text=text):
                self.assertEqual(decode_base64(encode_base64(text)), text)

    def test_empty_input(self) -> None:
        self.assertEqual(decode_base64(""), "")
        self.assertEqual(encode_base64(""), "")
        self.assertFalse(is_base64(""))

    def test_invalid_input_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            decode_base64("TWF")  # length not a multiple of four
        self.assertIn("multiple of 4", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            decode_base64("TW*u")
        self.assertTrue(str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            decode_base64("TWFu====")
        self.assertTrue(str(ctx.exception))

    def test_detects_real_base64(self) -> None:
        encoded = encode_base64("MEET ME AT THE OLD BRIDGE AT MIDNIGHT")
        self.assertTrue(is_base64(encoded))

    def test_uppercase_ciphertext_is_never_called_base64(self) -> None:
        # The documented limitation, tested rather than assumed, on the worst
        # case we could find. "STBEPERS" is eight consecutive letters of the
        # expository corpus; it is legal Base64, its length is a multiple of
        # four, and it decodes to six bytes that are ALL printable ASCII
        # ("I0D<DR"). Every other test in the module passes it. Only the
        # "a plain uppercase ciphertext cannot be told from Base64" rule
        # stops the tool announcing an encoding here, so if that rule is ever
        # removed this test is what notices.
        self.assertEqual(decode_base64("STBEPERS"), "I0D<DR")
        self.assertFalse(is_base64("STBEPERS"))
        self.assertEqual(identify("STBEPERS"), [])


class TestMorse(unittest.TestCase):
    def test_known_pair(self) -> None:
        self.assertEqual(encode_morse("SOS"), "... --- ...")
        self.assertEqual(decode_morse("... --- ..."), "SOS")
        self.assertEqual(
            encode_morse("HELLO WORLD"),
            ".... . .-.. .-.. --- / .-- --- .-. .-.. -..",
        )
        self.assertEqual(
            decode_morse(".... . .-.. .-.. --- / .-- --- .-. .-.. -.."),
            "HELLO WORLD",
        )

    def test_underscore_dashes_and_spaced_words(self) -> None:
        self.assertEqual(decode_morse("... ___ ..."), "SOS")
        self.assertEqual(decode_morse(".... .   .-- . .-.. .-.."), "HE WELL")

    def test_layout_does_not_matter(self) -> None:
        for variant in (
            "... --- ...",
            "...  ---  ...",
            "...\n---\n...",
            "   ... --- ...   ",
        ):
            with self.subTest(variant=variant):
                self.assertEqual(decode_morse(variant).replace(" ", ""), "SOS")

    def test_round_trip(self) -> None:
        for text in ("ATTACK AT DAWN", "SEND 500 MEN", "WHO GOES THERE?"):
            with self.subTest(text=text):
                self.assertEqual(decode_morse(encode_morse(text)), text)

    def test_table_is_one_to_one(self) -> None:
        codes = list(MORSE_TABLE.values())
        self.assertEqual(len(codes), len(set(codes)))
        self.assertEqual(len(MORSE_TABLE), 54)

    def test_empty_input(self) -> None:
        self.assertEqual(decode_morse(""), "")
        self.assertEqual(encode_morse(""), "")
        self.assertFalse(is_morse(""))

    def test_invalid_input_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            decode_morse("........")  # eight dots is not a Morse symbol
        self.assertIn("not a Morse symbol", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            decode_morse(".- X -.")
        self.assertTrue(str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            encode_morse("100%")
        self.assertIn("%", str(ctx.exception))

    def test_unsolvable_morse_is_reported_as_nothing(self) -> None:
        self.assertFalse(is_morse("........ ........"))
        self.assertEqual(identify("........ ........"), [])


class TestIdentify(unittest.TestCase):
    def test_each_format_is_found(self) -> None:
        message = "MEET AT THE OLD BRIDGE"
        for encoded, expected in (
            (encode_hex(message, separator=" "), "hexadecimal"),
            (encode_binary(message), "binary"),
            (encode_decimal(message), "decimal ASCII"),
            (encode_base64(message), "Base64"),
            (encode_morse(message), "Morse code"),
        ):
            with self.subTest(expected=expected):
                formats = [guess.format for guess in identify(encoded)]
                self.assertIn(expected, formats)

    def test_reasons_and_confidence_are_populated(self) -> None:
        guesses = identify(encode_hex("MEET AT THE OLD BRIDGE"))
        self.assertTrue(guesses)
        for guess in guesses:
            self.assertIn(guess.confidence, ("possible", "likely"))
            self.assertTrue(guess.reason.strip())
            self.assertIn("NOT encryption", guess.describe())

    def test_english_prose_is_not_an_encoding(self) -> None:
        prose = corpus_sample(400)
        self.assertEqual(identify(prose), [])

    def test_uppercase_ciphertext_is_not_an_encoding(self) -> None:
        # Trimmed to a multiple of four so the Base64 shape test cannot save
        # us; this has to be rejected on the evidence, not on the length.
        letters = letters_only(corpus_sample(600))
        trimmed = letters[: (len(letters) // 4) * 4]
        self.assertGreaterEqual(len(trimmed), 300)
        self.assertEqual(identify(trimmed), [])
        self.assertFalse(is_hex(trimmed))
        self.assertFalse(is_binary(trimmed))
        self.assertFalse(is_base64(trimmed))
        self.assertFalse(is_decimal(trimmed))
        self.assertFalse(is_morse(trimmed))

    def test_lowercase_english_without_punctuation_is_not_base64(self) -> None:
        # 40 characters, a multiple of four, every character in the Base64
        # alphabet, and lowercase letters present. Only the decode test can
        # reject this one.
        self.assertFalse(is_base64("Thequickbrownfoxjumpsoverthelazydogagain"))

    def test_ambiguity_is_reported_not_hidden(self) -> None:
        # "72 69 76 76 79" is decimal ASCII for HELLO, and with the spaces
        # removed it is also valid hexadecimal for "rivvy". Both are true.
        formats = [guess.format for guess in identify("72 69 76 76 79")]
        self.assertIn("decimal ASCII", formats)
        self.assertIn("hexadecimal", formats)

    def test_empty_and_blank_input(self) -> None:
        self.assertEqual(identify(""), [])
        self.assertEqual(identify("   \n  "), [])

    def test_describe_guesses_says_nothing_found(self) -> None:
        self.assertIn("No encoding", describe_guesses([]))

    def test_guess_rejects_a_bad_confidence_label(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            EncodingGuess(format="hexadecimal", confidence="certain", reason="x")
        self.assertIn("confidence", str(ctx.exception))


class TestSolve(unittest.TestCase):
    def test_recovers_a_long_base64_message(self) -> None:
        sample = corpus_sample(400)
        candidates = solve(encode_base64(sample))
        best = candidates.best()
        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best.plaintext, sample)
        self.assertIn("Base64", best.method)

    def test_recovers_a_long_hex_message(self) -> None:
        sample = corpus_sample(400)
        best = solve(encode_hex(sample, separator=" ")).best()
        assert best is not None
        self.assertEqual(best.plaintext, sample)

    def test_recovers_a_long_binary_message(self) -> None:
        sample = corpus_sample(400)
        best = solve(encode_binary(sample)).best()
        assert best is not None
        self.assertEqual(best.plaintext, sample)

    def test_recovers_morse(self) -> None:
        message = "MEET ME AT THE OLD BRIDGE AT MIDNIGHT AND BRING THE PAPERS"
        best = solve(encode_morse(message)).best()
        assert best is not None
        self.assertEqual(best.plaintext, message)

    def test_every_candidate_says_it_is_not_encryption(self) -> None:
        candidates = solve(encode_decimal("MEET AT THE OLD BRIDGE"))
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertIn("NOT encryption", candidate.method)
            self.assertTrue(candidate.diagnostics["not_encryption"])
            self.assertEqual(candidate.diagnostics["warning"], NOT_ENCRYPTION)
            self.assertIn("normalised_score", candidate.diagnostics)
            self.assertIn("word_coverage", candidate.diagnostics)
            self.assertIsNone(candidate.display)

    def test_english_reading_outranks_the_accidental_one(self) -> None:
        # "72 69 76 76 79" decodes as decimal ASCII to HELLO and, with the
        # spaces removed, as hexadecimal to "rivvy". Both readings are
        # returned; the one that produces English wins on score rather than on
        # the order the detectors happen to run in.
        candidates = solve("72 69 76 76 79")
        methods = [candidate.method for candidate in candidates]
        self.assertGreaterEqual(len(methods), 2)
        best = candidates.best()
        assert best is not None
        self.assertEqual(best.plaintext, "HELLO")
        self.assertIn("decimal ASCII", best.method)

    def test_ciphertext_produces_no_candidates(self) -> None:
        # The honest failure: a monoalphabetic ciphertext is not an encoding,
        # and the module returns nothing rather than a confident decode.
        letters = letters_only(corpus_sample(600))
        candidates = solve(letters[: (len(letters) // 4) * 4])
        self.assertEqual(len(candidates), 0)
        self.assertIsNone(candidates.best())

    def test_empty_input(self) -> None:
        self.assertEqual(len(solve("")), 0)

    def test_accepts_normalized_text(self) -> None:
        encoded = encode_hex("MEET AT THE OLD BRIDGE", separator=" ")
        from_string = solve(encoded).best()
        from_normalized = solve(normalize(encoded)).best()
        assert from_string is not None and from_normalized is not None
        self.assertEqual(from_string.plaintext, from_normalized.plaintext)
        self.assertEqual(from_string.plaintext, "MEET AT THE OLD BRIDGE")

    def test_top_limits_the_candidate_list(self) -> None:
        candidates = solve("72 69 76 76 79", top=1)
        self.assertEqual(len(candidates), 1)

    def test_unknown_option_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            solve("48454C4C4F", rails=4)
        self.assertIn("rails", str(ctx.exception))

    def test_include_possible_filter(self) -> None:
        strict = solve(encode_hex("MEET AT THE OLD BRIDGE"), include_possible=False)
        for candidate in strict:
            self.assertEqual(candidate.diagnostics["detector_confidence"], "likely")


if __name__ == "__main__":
    unittest.main()
