"""Command line interface for the toolkit.

    cipher_tool analyse message.txt
    cipher_tool auto message.txt --normal
    cipher_tool vigenere message.txt --key-length 7
    cipher_tool crib message.txt "THE"
    cipher_tool shell

Two conventions run through every command.

**The original input is never destroyed.** Commands read a file (or
``--text``, or standard input with ``-``) and keep the original alongside the
letters-only view. ``show`` prints both.

**Supplying a key means "do this"; omitting it means "search".** So
``cipher_tool caesar message.txt --shift 3`` decrypts with that shift, while
``cipher_tool caesar message.txt`` ranks all twenty-six. Add ``--encrypt`` to
run a cipher forwards instead of backwards.

Every solve prints ranked candidates with their evidence, and ends with the
disclaimer in ``cipher_tool/__init__.py``. Nothing here submits anything
anywhere; see RULES_COMPLIANCE.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from . import (
    DISCLAIMER,
    __version__,
    affine,
    atbash,
    auto as auto_module,
    autokey,
    beaufort,
    bifid,
    caesar,
    columnar,
    context as context_module,
    cribs,
    encodings,
    hill,
    keyword_cipher,
    playfair,
    polybius,
    rail_fence,
    substitution,
    transposition,
    vigenere,
)
from .candidates import Candidate, CandidateSet, render_candidates
from .normalize import group_text, normalize, NormalizedText
from .scoring import annotate, default_scorer
from .statistics import analyse, render_report

PROGRAM = "cipher_tool"


# ---------------------------------------------------------------------------
# Input and output
# ---------------------------------------------------------------------------


class InputError(Exception):
    """Raised when the ciphertext cannot be read. Reported, never guessed at."""


def read_source(args: argparse.Namespace) -> tuple[str, str | None]:
    """Return ``(text, path)`` for the ciphertext the user named.

    Accepts a file path, ``-`` for standard input, or ``--text``. A path that
    does not exist is an error with a suggestion, not a silent fallback to
    treating the argument as literal ciphertext -- guessing there would turn
    a typo into a confusing solve of the filename.
    """
    if getattr(args, "text", None):
        return args.text, None

    target = getattr(args, "input", None)
    if target in (None, ""):
        raise InputError(
            "No ciphertext given. Pass a file, or --text \"HEALI OPASD\", "
            "or - to read standard input."
        )
    if target == "-":
        return sys.stdin.read(), None

    path = Path(target)
    if not path.exists():
        raise InputError(
            f"No such file: {target}\n"
            f"  If you meant to pass the ciphertext itself, use "
            f"--text \"{target[:30]}...\"."
        )
    if path.is_dir():
        raise InputError(f"{target} is a directory, not a ciphertext file.")
    try:
        return path.read_text(encoding="utf-8"), str(path)
    except UnicodeDecodeError as error:
        raise InputError(
            f"{target} is not readable as UTF-8 text ({error}). "
            "If it is a binary file, this toolkit cannot read it."
        ) from error


def emit(text: str, args: argparse.Namespace) -> None:
    """Print to the terminal, or write to ``--output`` if one was given."""
    output = getattr(args, "output", None)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
        print(f"Written to {output}")
    else:
        print(text)


def finish(args: argparse.Namespace) -> None:
    """Print the standing disclaimer unless the user asked for quiet."""
    if not getattr(args, "quiet", False):
        print()
        print("-" * 72)
        print(DISCLAIMER)


def show_candidates(
    found: CandidateSet, args: argparse.Namespace, title: str
) -> None:
    """Render a solver's output, including anything it wants to warn about."""
    notes = getattr(found, "notes", ())
    body = render_candidates(
        found.ranked(), top=args.top, full_text=args.full, title=title
    )
    if notes:
        body = body + "\n\nNotes from the search:\n" + "\n".join(
            f"  - {note}" for note in notes
        )
    agreeing = found.corroborations()
    if len(agreeing) > 1:
        body += (
            "\n\nCORROBORATION: "
            + ", ".join(agreeing)
            + " independently produced the same plaintext."
        )
    gap = found.score_gap()
    if gap is not None and gap < 0.10:
        body += (
            f"\n\nWARNING: the best candidate leads the best COMPETING "
            f"reading by only {gap:.3f} per letter. The search has not "
            "singled one out -- read both."
        )
    emit(body, args)


def single_candidate(
    method: str, key: str, plaintext: str, normalized: NormalizedText,
    diagnostics: dict[str, Any] | None = None,
) -> CandidateSet:
    """Wrap one supplied-key decryption as a candidate, scored honestly.

    A key the user supplied still gets scored and labelled. If they typed the
    wrong key, the report should say the result does not look like English
    rather than presenting it as an answer.
    """
    scorer = default_scorer()
    notes = dict(diagnostics or {})
    annotate(notes, plaintext, scorer)
    candidate = Candidate(
        method=method,
        key=key,
        score=scorer.score(plaintext),
        plaintext=plaintext,
        diagnostics=notes,
        display=(normalized.relayout(plaintext)
                 if len(plaintext) == len(normalized.letters) else None),
    )
    return CandidateSet([candidate])


# ---------------------------------------------------------------------------
# Shared argument groups
# ---------------------------------------------------------------------------


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    """The ciphertext source, shared by nearly every command."""
    parser.add_argument("input", nargs="?",
                        help="ciphertext file, or - for standard input")
    parser.add_argument("--text", metavar="STRING",
                        help="ciphertext given directly instead of a file")


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Output shaping, shared by every command that produces a report."""
    parser.add_argument("--top", type=int, default=5, metavar="N",
                        help="how many candidates to show (default 5)")
    parser.add_argument("--full", action="store_true",
                        help="print whole plaintexts instead of a preview")
    parser.add_argument("--output", metavar="FILE",
                        help="write the report to a file instead of stdout")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress the closing disclaimer")


def add_search_arguments(parser: argparse.ArgumentParser) -> None:
    """Controls for how hard and how long a search may work."""
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="random seed, to make a search reproducible")
    parser.add_argument("--max-time", type=float, default=None, metavar="S",
                        dest="max_time",
                        help="stop searching after this many seconds")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def command_show(args: argparse.Namespace) -> int:
    """Print the original and the normalised ciphertext side by side."""
    text, path = read_source(args)
    normalized = normalize(text)

    lines = ["ORIGINAL INPUT", "=" * 72]
    lines.append(normalized.original.rstrip("\n") or "(empty)")
    lines.append("")
    lines.append(f"NORMALISED ({normalized.length} letters, grouped in "
                 f"{args.group}s for reading only)")
    lines.append("=" * 72)
    lines.append(group_text(normalized.letters, args.group, 10) or "(no letters)")
    lines.append("")
    lines.append("CONTINUOUS")
    lines.append("=" * 72)
    lines.append(normalized.letters or "(no letters)")

    uniform = normalized.uniform_group_length()
    # One group tells us nothing -- the note is only worth making when the
    # input really was laid out in several equal blocks.
    if uniform is not None and len(normalized.groups) > 1:
        lines.append("")
        lines.append(
            f"NOTE: every whitespace group in the input is {uniform} "
            "characters long. That is transcription formatting. It is NOT "
            "evidence about where plaintext words begin and end, and nothing "
            "in this toolkit treats it as such."
        )
    emit("\n".join(lines), args)
    finish(args)
    return 0


def command_analyse(args: argparse.Namespace) -> int:
    """Measure the ciphertext and suggest families to try."""
    text, _ = read_source(args)
    emit(render_report(analyse(text)), args)
    finish(args)
    return 0


def command_model(args: argparse.Namespace) -> int:
    """Describe the English scoring model and where it came from."""
    emit(default_scorer().describe_model(), args)
    finish(args)
    return 0


def command_caesar(args: argparse.Namespace) -> int:
    """Caesar shift: decrypt with a shift, or rank all twenty-six."""
    text, _ = read_source(args)
    normalized = normalize(text)
    if args.shift is not None:
        if args.encrypt:
            result = caesar.encrypt(normalized.letters, args.shift)
            emit(_encrypted_report("Caesar", f"shift={args.shift}", result,
                                   args), args)
        else:
            plaintext = caesar.decrypt(normalized.letters, args.shift)
            show_candidates(
                single_candidate("Caesar", f"shift={args.shift}", plaintext,
                                 normalized),
                args, "Caesar with the shift you supplied")
        finish(args)
        return 0

    found = caesar.solve(normalized, top=max(args.top, 26) if args.all
                         else args.top)
    show_candidates(found, args,
                    "Caesar -- all 26 shifts, ranked" if args.all
                    else "Caesar -- best shifts")
    finish(args)
    return 0


def command_atbash(args: argparse.Namespace) -> int:
    """Atbash: reverse the alphabet. A fixed key, so there is nothing to search."""
    text, _ = read_source(args)
    normalized = normalize(text)
    if args.encrypt:
        emit(_encrypted_report("Atbash", "fixed", atbash.encrypt(
            normalized.letters), args), args)
    else:
        show_candidates(atbash.solve(normalized, top=1), args, "Atbash")
    finish(args)
    return 0


def command_affine(args: argparse.Namespace) -> int:
    """Affine cipher E(x) = (a*x + b) mod 26."""
    text, _ = read_source(args)
    normalized = normalize(text)
    if args.a is not None or args.b is not None:
        if args.a is None or args.b is None:
            raise InputError("Give both -a and -b, or neither.")
        try:
            if args.encrypt:
                result = affine.encrypt(normalized.letters, args.a, args.b)
                emit(_encrypted_report("affine", f"a={args.a} b={args.b}",
                                       result, args), args)
            else:
                plaintext = affine.decrypt(normalized.letters, args.a, args.b)
                show_candidates(
                    single_candidate("affine", f"a={args.a} b={args.b}",
                                     plaintext, normalized),
                    args, "Affine with the key you supplied")
        except ValueError as error:
            raise InputError(str(error)) from error
        finish(args)
        return 0

    show_candidates(affine.solve(normalized, top=args.top), args,
                    "Affine -- all 312 valid keys, ranked")
    finish(args)
    return 0


def command_substitution(args: argparse.Namespace) -> int:
    """General monoalphabetic substitution: hill climbing with restarts."""
    text, _ = read_source(args)
    normalized = normalize(text)

    fixed = _parse_mapping(args.map) if args.map else None

    if args.key:
        key = (substitution.SubstitutionKey.from_alphabet(args.key)
               if len(args.key.replace(" ", "")) == 26
               else substitution.SubstitutionKey.from_string(args.key))
        if args.encrypt:
            emit(_encrypted_report("substitution", str(args.key),
                                   substitution.encrypt(normalized.letters,
                                                        key), args), args)
        else:
            plaintext = substitution.decrypt(normalized.letters, key)
            show_candidates(
                single_candidate("substitution", args.key, plaintext,
                                 normalized,
                                 {"key_complete": key.is_complete}),
                args, "Substitution with the key you supplied")
        finish(args)
        return 0

    if args.words:
        wordlist = [word for word in args.words.replace(",", " ").split()]
        matches = substitution.analyse_words(wordlist, known=fixed)
        lines = ["Word pattern analysis", "=" * 72,
                 "Each ciphertext word is matched against English words with "
                 "the same repeated-letter shape.",
                 "These are candidates only.", ""]
        for word, options in matches.items():
            lines.append(f"  {word}  ({len(options)} match(es))")
            lines.append(f"    {', '.join(options[:12]) or '(none)'}")
        emit("\n".join(lines), args)
        finish(args)
        return 0

    found = substitution.solve(
        normalized, top=args.top, restarts=args.restarts, seed=args.seed,
        time_budget=args.max_time, fixed=fixed,
    )
    show_candidates(found, args, "Monoalphabetic substitution")
    finish(args)
    return 0


def command_keyword(args: argparse.Namespace) -> int:
    """Keyword-alphabet substitution."""
    text, _ = read_source(args)
    normalized = normalize(text)
    if args.key:
        try:
            if args.encrypt:
                result = keyword_cipher.encrypt(
                    normalized.letters, args.key,
                    reverse_remainder=args.reverse)
                emit(_encrypted_report("keyword", args.key, result, args), args)
            else:
                plaintext = keyword_cipher.decrypt(
                    normalized.letters, args.key,
                    reverse_remainder=args.reverse)
                show_candidates(
                    single_candidate("keyword substitution",
                                     f"keyword={args.key}", plaintext,
                                     normalized),
                    args, "Keyword substitution with the key you supplied")
        except ValueError as error:
            raise InputError(str(error)) from error
        finish(args)
        return 0

    words = ([w for w in args.words.replace(",", " ").split()]
             if args.words else None)
    found = keyword_cipher.solve(normalized, top=args.top, words=words,
                                 seed=args.seed, time_budget=args.max_time)
    show_candidates(found, args, "Keyword substitution")
    finish(args)
    return 0


def command_vigenere(args: argparse.Namespace) -> int:
    """Vigenere, with key-length evidence from three independent methods."""
    text, _ = read_source(args)
    normalized = normalize(text)

    if args.evidence:
        evidence = vigenere.estimate_key_lengths(
            normalized, max_key_length=args.max_key_length)
        emit(vigenere.describe_key_lengths(evidence), args)
        finish(args)
        return 0

    if args.key and args.encrypt:
        emit(_encrypted_report("Vigenere", f"key={args.key}",
                               vigenere.encrypt(normalized.letters, args.key),
                               args), args)
        finish(args)
        return 0

    found = vigenere.solve(
        normalized, top=args.top, key=args.key, key_length=args.key_length,
        max_key_length=args.max_key_length, time_budget=args.max_time,
    )
    show_candidates(found, args, "Vigenere")
    finish(args)
    return 0


def command_beaufort(args: argparse.Namespace) -> int:
    """Beaufort and variant Beaufort."""
    text, _ = read_source(args)
    normalized = normalize(text)
    if args.key:
        try:
            if args.encrypt:
                result = beaufort.encrypt(normalized.letters, args.key,
                                          variant=args.variant)
                emit(_encrypted_report(
                    "variant Beaufort" if args.variant else "Beaufort",
                    f"key={args.key}", result, args), args)
            else:
                plaintext = beaufort.decrypt(normalized.letters, args.key,
                                             variant=args.variant)
                show_candidates(
                    single_candidate(
                        "variant Beaufort" if args.variant else "Beaufort",
                        f"key={args.key}", plaintext, normalized),
                    args, "Beaufort with the key you supplied")
        except ValueError as error:
            raise InputError(str(error)) from error
        finish(args)
        return 0

    # Without --variant both forms are searched and reported together, which
    # is what you want when you do not know which was used. With --variant
    # the user has told us; silently searching both anyway would ignore them.
    variants = ("variant",) if args.variant else ("beaufort", "variant")
    found = beaufort.solve(normalized, top=args.top,
                           key_length=args.key_length,
                           max_key_length=args.max_key_length,
                           variants=variants,
                           time_budget=args.max_time)
    show_candidates(found, args,
                    "Variant Beaufort" if args.variant
                    else "Beaufort and variant Beaufort")
    finish(args)
    return 0


def command_autokey(args: argparse.Namespace) -> int:
    """Autokey, plaintext and ciphertext variants."""
    text, _ = read_source(args)
    normalized = normalize(text)
    if args.primer:
        try:
            if args.encrypt:
                result = autokey.encrypt(normalized.letters, args.primer,
                                         mode=args.mode)
                emit(_encrypted_report(f"{args.mode} autokey",
                                       f"primer={args.primer}", result, args),
                     args)
            else:
                plaintext = autokey.decrypt(normalized.letters, args.primer,
                                            mode=args.mode)
                show_candidates(
                    single_candidate(f"{args.mode} autokey",
                                     f"primer={args.primer}", plaintext,
                                     normalized),
                    args, "Autokey with the primer you supplied")
        except ValueError as error:
            raise InputError(str(error)) from error
        finish(args)
        return 0

    found = autokey.solve(normalized, top=args.top, max_primer=args.max_primer,
                          seed=args.seed, time_budget=args.max_time)
    show_candidates(found, args, "Autokey")
    finish(args)
    return 0


def command_railfence(args: argparse.Namespace) -> int:
    """Rail fence (zigzag) transposition."""
    text, _ = read_source(args)
    normalized = normalize(text)
    if args.rails is not None:
        try:
            if args.encrypt:
                result = rail_fence.encrypt(normalized.letters, args.rails,
                                            args.offset)
                emit(_encrypted_report("rail fence", f"rails={args.rails}",
                                       result, args), args)
            else:
                plaintext = rail_fence.decrypt(normalized.letters, args.rails,
                                               args.offset)
                show_candidates(
                    single_candidate("rail fence",
                                     f"rails={args.rails} offset={args.offset}",
                                     plaintext, normalized),
                    args, "Rail fence with the settings you supplied")
        except ValueError as error:
            raise InputError(str(error)) from error
        finish(args)
        return 0

    show_candidates(rail_fence.solve(normalized, top=args.top), args,
                    "Rail fence -- every rail count tried")
    finish(args)
    return 0


def command_columnar(args: argparse.Namespace) -> int:
    """Columnar transposition."""
    text, _ = read_source(args)
    normalized = normalize(text)
    if args.key:
        try:
            if args.encrypt:
                result = columnar.encrypt(normalized.letters, args.key,
                                          complete=args.complete)
                emit(_encrypted_report("columnar", f"key={args.key}", result,
                                       args), args)
            else:
                plaintext = columnar.decrypt(normalized.letters, args.key,
                                             complete=args.complete)
                show_candidates(
                    single_candidate("columnar transposition",
                                     f"key={args.key}", plaintext, normalized),
                    args, "Columnar with the key you supplied")
        except ValueError as error:
            raise InputError(str(error)) from error
        finish(args)
        return 0

    found = columnar.solve(normalized, top=args.top,
                           key_length=args.key_length,
                           max_key_length=args.max_key_length,
                           seed=args.seed, time_budget=args.max_time)
    show_candidates(found, args, "Columnar transposition")
    finish(args)
    return 0


def command_transposition(args: argparse.Namespace) -> int:
    """Every transposition family at once: rail fence, columnar, route/grid."""
    text, _ = read_source(args)
    normalized = normalize(text)
    if args.routes:
        emit(transposition.describe_routes(), args)
        finish(args)
        return 0
    found = transposition.solve_all(
        normalized, top=args.top, max_key_length=args.max_key_length,
        seed=args.seed, time_budget=args.max_time,
    )
    show_candidates(found, args, "Transposition (all families)")
    finish(args)
    return 0


def command_polybius(args: argparse.Namespace) -> int:
    """Polybius square coordinate encoding."""
    text, _ = read_source(args)
    normalized = normalize(text)
    if args.key or args.square:
        square = (
            polybius.PolybiusSquare(args.square)
            if args.square
            else polybius.PolybiusSquare.standard(args.key)
        )
        if args.encrypt:
            emit(_encrypted_report("Polybius", f"key={args.key or ''}",
                                   polybius.encrypt(normalized.letters, square),
                                   args), args)
        else:
            try:
                plaintext = polybius.decrypt(text, square)
            except ValueError as error:
                raise InputError(str(error)) from error
            show_candidates(
                single_candidate("Polybius", f"key={args.key or ''}",
                                 plaintext, normalized),
                args, "Polybius with the square you supplied")
        finish(args)
        return 0
    keywords = ([w for w in args.words.replace(",", " ").split()]
                if args.words else None)
    show_candidates(polybius.solve(text, top=args.top, keywords=keywords),
                    args, "Polybius")
    finish(args)
    return 0


def command_bifid(args: argparse.Namespace) -> int:
    """Bifid: Polybius coordinates, fractionated and re-read."""
    text, _ = read_source(args)
    normalized = normalize(text)
    if args.key is not None or args.period is not None:
        square = (polybius.PolybiusSquare.standard(args.key)
                  if args.key else None)
        try:
            if args.encrypt:
                result = bifid.encrypt(normalized.letters, square, args.period)
                emit(_encrypted_report(
                    "Bifid", f"key={args.key or 'standard'} "
                             f"period={args.period}", result, args), args)
            else:
                plaintext = bifid.decrypt(normalized.letters, square,
                                          args.period)
                show_candidates(
                    single_candidate("Bifid",
                                     f"key={args.key or 'standard'} "
                                     f"period={args.period}", plaintext,
                                     normalized),
                    args, "Bifid with the settings you supplied")
        except ValueError as error:
            raise InputError(str(error)) from error
        finish(args)
        return 0

    keywords = ([w for w in args.words.replace(",", " ").split()]
                if args.words else None)
    found = bifid.solve(normalized, top=args.top, keywords=keywords,
                        max_period=args.max_period,
                        time_budget=args.max_time)
    show_candidates(found, args, "Bifid")
    finish(args)
    return 0


def command_playfair(args: argparse.Namespace) -> int:
    """Playfair digraph substitution on a keyed 5x5 square."""
    text, _ = read_source(args)
    normalized = normalize(text)

    if args.check:
        # Formatting validation barely depends on which square was used --
        # only on which letter the square omits -- so an unkeyed square is
        # the right thing to check against when no key is known.
        square = (playfair.playfair_square(args.key) if args.key
                  else playfair.plain_square())
        problems = playfair.validate_ciphertext(normalized.letters, square)
        if problems:
            emit("Playfair validation problems:\n" + "\n".join(
                f"  - {problem}" for problem in problems), args)
        else:
            emit("No Playfair formatting problems found. That does not mean "
                 "the cipher IS Playfair -- only that the text could be.", args)
        finish(args)
        return 0

    if args.key:
        try:
            if args.encrypt:
                result = playfair.encrypt(normalized.letters, args.key,
                                          filler=args.filler)
                emit(_encrypted_report("Playfair", f"key={args.key}", result,
                                       args), args)
            else:
                plaintext = playfair.decrypt(normalized.letters, args.key)
                show_candidates(
                    single_candidate("Playfair", f"key={args.key}", plaintext,
                                     normalized),
                    args, "Playfair with the key you supplied")
        except ValueError as error:
            raise InputError(str(error)) from error
        finish(args)
        return 0

    found = playfair.solve(normalized, top=args.top, restarts=args.restarts,
                           seed=args.seed, time_budget=args.max_time)
    show_candidates(found, args, "Playfair")
    finish(args)
    return 0


def command_hill(args: argparse.Namespace) -> int:
    """Hill cipher over small matrices, with pure-Python modular algebra."""
    text, _ = read_source(args)
    normalized = normalize(text)

    matrix = None
    if args.matrix:
        matrix = _parse_matrix(args.matrix)
    elif args.key:
        try:
            matrix = hill.key_from_string(args.key)
        except ValueError as error:
            raise InputError(str(error)) from error

    if matrix is not None:
        try:
            if args.encrypt:
                result = hill.encrypt(normalized.letters, matrix)
                emit(_encrypted_report("Hill",
                                       hill.matrix_to_string(matrix), result,
                                       args), args)
            else:
                plaintext = hill.decrypt(normalized.letters, matrix)
                show_candidates(
                    single_candidate("Hill", hill.matrix_to_string(matrix),
                                     plaintext, normalized,
                                     {"determinant":
                                      hill.determinant(matrix)}),
                    args, "Hill with the matrix you supplied")
        except hill.NotInvertibleError as error:
            raise InputError(str(error)) from error
        except ValueError as error:
            raise InputError(str(error)) from error
        finish(args)
        return 0

    found = hill.solve(normalized, top=args.top, size=args.size,
                       crib=args.crib, crib_at=args.crib_at,
                       time_budget=args.max_time)
    show_candidates(found, args, "Hill")
    finish(args)
    return 0


def command_encodings(args: argparse.Namespace) -> int:
    """Identify and decode non-encryption formats.

    Prints the evidence for each guess, then the same guesses ranked by how
    much their decoding looks like English. The ranking matters: several
    formats can parse the same text (``72 69 76 76 79`` is valid hexadecimal
    *and* valid decimal ASCII), and only one of them produces a word.
    """
    text, _ = read_source(args)
    guesses = encodings.identify(text)
    emit(encodings.describe_guesses(guesses), args)
    found = encodings.solve(text, top=args.top)
    if found:
        show_candidates(found, args, "Decodings, ranked by English score")
    finish(args)
    return 0


def command_crib(args: argparse.Namespace) -> int:
    """Test where a guessed piece of plaintext could sit."""
    text, path = read_source(args)
    words = list(args.crib)
    # `crib` takes an optional file positional AND one or more crib
    # positionals, so with --text the first crib word lands in `input`
    # instead. Rescue it rather than reporting a confusing "no crib given".
    if getattr(args, "text", None) and args.input and not words:
        words = [args.input]
        path = None
    if path and args.use_context:
        notes = context_module.ContextNotes.load(path)
        words = context_module.merge_cribs(notes, extra=words)
    if not words:
        raise InputError(
            "Give at least one crib, e.g. cipher_tool crib message.txt \"THE\""
        )
    reports = [
        cribs.test_crib(text, word, methods=args.methods) for word in words
    ]
    emit("\n\n".join(report.render() for report in reports), args)
    finish(args)
    return 0


def command_context(args: argparse.Namespace) -> int:
    """Record and show the team's story notes for a ciphertext."""
    _, path = read_source(args)
    if path is None:
        raise InputError(
            "Context notes are stored beside a ciphertext FILE, so this "
            "command needs a file rather than --text."
        )
    notes = context_module.ContextNotes.load(path)

    changed = False
    try:
        for assignment in args.add or []:
            field_name, value = context_module.parse_assignment(assignment)
            notes.add(field_name, value)
            changed = True
        for field_name in args.clear or []:
            if field_name not in {name for name, _ in context_module.FIELDS}:
                raise ValueError(f"Unknown context field {field_name!r}.")
            setattr(notes, field_name, [])
            changed = True
    except ValueError as error:
        # A mistyped field name is a user error, so report it as one rather
        # than letting it escape as a traceback.
        raise InputError(str(error)) from error

    if changed:
        target = notes.save(path)
        print(f"Saved to {target}")
    emit(notes.render(path), args)
    finish(args)
    return 0


def command_auto(args: argparse.Namespace) -> int:
    """Run the whole pipeline: cheap diagnostics first, then the searches."""
    text, path = read_source(args)
    effort = args.effort
    result = auto_module.auto_solve(
        text, effort=effort, top=args.top, seed=args.seed,
        max_time=args.max_time,
    )
    emit(result.render(top=args.top, full_text=args.full,
                       show_stats=args.stats), args)
    finish(args)
    return 0


def command_shell(args: argparse.Namespace) -> int:
    """Interactive session: load a ciphertext once, then run commands on it."""
    return run_shell(args)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encrypted_report(method: str, key: str, ciphertext: str,
                      args: argparse.Namespace) -> str:
    """Format the output of running a cipher forwards."""
    return (
        f"{method} encryption\n"
        + "=" * 72 + "\n"
        + f"Key/config : {key}\n"
        + f"Length     : {len(ciphertext)} letters\n\n"
        + group_text(ciphertext, args.group if hasattr(args, "group") else 5,
                     10)
        + "\n\nContinuous:\n" + ciphertext
    )


def _parse_mapping(entries: Sequence[str]) -> "substitution.SubstitutionKey":
    """Turn ``--map Q=E --map X=T`` into a partial substitution key.

    Accepts commas inside one argument too, so ``--map "Q=E,X=T"`` works.
    Inconsistent mappings are rejected by ``SubstitutionKey`` itself, with a
    message naming the conflicting letters.
    """
    joined = " ".join(entries).replace(",", " ")
    try:
        return substitution.SubstitutionKey.from_string(joined)
    except ValueError as error:
        raise InputError(
            f"Could not read the mapping {joined!r}: {error}\n"
            "  Write it as pairs of letters, cipher first, e.g. "
            "--map Q=E --map X=T"
        ) from error


def _parse_matrix(text: str) -> list[list[int]]:
    """Parse ``"3,3,2,5"`` or ``"3 3; 2 5"`` into a square matrix."""
    numbers: list[int] = []
    for token in text.replace(";", " ").replace(",", " ").split():
        try:
            numbers.append(int(token))
        except ValueError as error:
            raise InputError(
                f"{token!r} is not a whole number. Give the matrix as "
                "comma-separated numbers read across the rows, e.g. "
                "--matrix 3,3,2,5"
            ) from error
    size = int(round(len(numbers) ** 0.5))
    if size * size != len(numbers) or size < 2:
        raise InputError(
            f"A Hill matrix must be square: {len(numbers)} numbers cannot "
            "form one. Give 4 numbers for 2x2, or 9 for 3x3."
        )
    return [numbers[row * size:(row + 1) * size] for row in range(size)]


# ---------------------------------------------------------------------------
# Interactive shell
# ---------------------------------------------------------------------------

SHELL_HELP = """
Interactive commands
--------------------
  load <file>          load a ciphertext from a file
  text <ciphertext>    type or paste a ciphertext directly
  show                 print the original and the normalised text
  analyse              measure it and suggest families to try
  auto [fast|normal|deep]
                       run the whole pipeline
  <command> [options]  any solver command, e.g.
                         caesar
                         vigenere --key-length 7
                         substitution --restarts 40
                         crib THE
  top <n>              how many candidates to show (currently {top})
  help                 this message
  quit                 leave

The loaded ciphertext is reused, so you do not retype it.
"""


def run_shell(args: argparse.Namespace) -> int:
    """A small REPL for working one ciphertext over several commands."""
    print(f"{PROGRAM} {__version__} interactive shell. Type 'help' or 'quit'.")
    state: dict[str, Any] = {"text": None, "path": None, "top": args.top}

    if getattr(args, "input", None):
        try:
            state["text"], state["path"] = read_source(args)
            print(f"Loaded {state['path']} "
                  f"({normalize(state['text']).length} letters).")
        except InputError as error:
            print(f"error: {error}")

    parser = build_parser()

    while True:
        try:
            line = input("cipher> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        word, _, rest = line.partition(" ")
        word = word.lower()

        if word in {"quit", "exit", "q"}:
            break
        if word in {"help", "?"}:
            print(SHELL_HELP.format(top=state["top"]))
            continue
        if word == "load":
            path = Path(rest.strip().strip('"'))
            if not path.exists():
                print(f"error: no such file: {path}")
                continue
            state["text"] = path.read_text(encoding="utf-8")
            state["path"] = str(path)
            print(f"Loaded {path} ({normalize(state['text']).length} letters).")
            continue
        if word == "text":
            state["text"] = rest
            state["path"] = None
            print(f"Loaded {normalize(rest).length} letters.")
            continue
        if word == "top":
            try:
                state["top"] = max(1, int(rest))
                print(f"Showing up to {state['top']} candidates.")
            except ValueError:
                print("error: top needs a number, e.g. 'top 10'")
            continue

        if state["text"] is None:
            print("error: nothing loaded. Use 'load <file>' or 'text <...>'.")
            continue

        # Re-use the real parser so the shell and the command line behave
        # identically -- one implementation, not two.
        tokens = [word] + _split_arguments(rest)
        if word == "auto" and rest.strip() in {"fast", "normal", "deep"}:
            tokens = ["auto", f"--{rest.strip()}"]
        try:
            parsed = parser.parse_args(tokens)
        except SystemExit:
            print("error: could not read that command. Type 'help'.")
            continue

        parsed.text = state["text"]
        parsed.input = state["path"]
        parsed.top = state["top"]
        parsed.quiet = True
        try:
            parsed.handler(parsed)
        except InputError as error:
            print(f"error: {error}")
        except Exception as error:  # keep the shell alive
            print(f"error: {type(error).__name__}: {error}")

    print("Bye.")
    return 0


def _split_arguments(text: str) -> list[str]:
    """Split a shell line on whitespace, honouring double quotes."""
    tokens: list[str] = []
    current: list[str] = []
    in_quotes = False
    for char in text:
        if char == '"':
            in_quotes = not in_quotes
        elif char.isspace() and not in_quotes:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the full command line parser."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Offline classical cryptanalysis toolkit, written from scratch. "
            "Produces ranked candidate plaintexts locally; it never submits "
            "anything anywhere."
        ),
        epilog=DISCLAIMER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                        version=f"{PROGRAM} {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    def add(name: str, handler: Callable[[argparse.Namespace], int],
            help_text: str, *, aliases: Sequence[str] = (),
            search: bool = False) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text, aliases=list(aliases),
                                    description=help_text)
        add_input_arguments(sub)
        add_output_arguments(sub)
        if search:
            add_search_arguments(sub)
        sub.set_defaults(handler=handler, group=5)
        return sub

    # -- inspection --------------------------------------------------------
    show = add("show", command_show, "print the original and normalised text")
    show.add_argument("--group", type=int, default=5, metavar="N",
                      help="group size for display only (default 5)")

    add("analyse", command_analyse,
        "measure the ciphertext and suggest what to try",
        aliases=["analyze"])

    model = subparsers.add_parser(
        "model", help="describe the English scoring model and its provenance")
    add_output_arguments(model)
    model.set_defaults(handler=command_model)

    # -- monoalphabetic ----------------------------------------------------
    caesar_parser = add("caesar", command_caesar,
                        "Caesar shift: decrypt with --shift, or rank all 26")
    caesar_parser.add_argument("--shift", type=int, metavar="N")
    caesar_parser.add_argument("--encrypt", action="store_true")
    caesar_parser.add_argument("--all", action="store_true",
                               help="show all 26 shifts, not just the best")

    atbash_parser = add("atbash", command_atbash, "Atbash (reverse alphabet)")
    atbash_parser.add_argument("--encrypt", action="store_true")

    affine_parser = add("affine", command_affine,
                        "affine cipher E(x) = (a*x + b) mod 26")
    affine_parser.add_argument("-a", type=int, metavar="A")
    affine_parser.add_argument("-b", type=int, metavar="B")
    affine_parser.add_argument("--encrypt", action="store_true")

    sub_parser = add("substitution", command_substitution,
                     "general monoalphabetic substitution (hill climbing)",
                     search=True)
    sub_parser.add_argument("--key", metavar="KEY",
                            help="a 26-letter alphabet, or pairs like 'QE XT'")
    sub_parser.add_argument("--map", action="append", metavar="C=P",
                            help="fix a mapping during the search; repeatable")
    sub_parser.add_argument("--restarts", type=int, default=25, metavar="N")
    sub_parser.add_argument("--words", metavar="LIST",
                            help="ciphertext words to pattern-match "
                                 "(only if word divisions are genuine)")
    sub_parser.add_argument("--encrypt", action="store_true")

    keyword_parser = add("keyword", command_keyword,
                         "keyword-alphabet substitution", search=True)
    keyword_parser.add_argument("--key", metavar="KEYWORD")
    keyword_parser.add_argument("--words", metavar="LIST",
                                help="candidate keywords to try")
    keyword_parser.add_argument("--reverse", action="store_true",
                                help="fill the rest of the alphabet backwards")
    keyword_parser.add_argument("--encrypt", action="store_true")

    # -- polyalphabetic ----------------------------------------------------
    vigenere_parser = add("vigenere", command_vigenere,
                          "Vigenere, with Kasiski and IC key-length evidence",
                          search=True)
    vigenere_parser.add_argument("--key", metavar="KEY")
    vigenere_parser.add_argument("--key-length", type=int, metavar="N",
                                 dest="key_length")
    vigenere_parser.add_argument("--max-key-length", type=int, default=20,
                                 metavar="N", dest="max_key_length")
    vigenere_parser.add_argument("--evidence", action="store_true",
                                 help="show the key-length evidence only")
    vigenere_parser.add_argument("--encrypt", action="store_true")

    beaufort_parser = add("beaufort", command_beaufort,
                          "Beaufort and variant Beaufort", search=True)
    beaufort_parser.add_argument("--key", metavar="KEY")
    beaufort_parser.add_argument("--key-length", type=int, metavar="N",
                                 dest="key_length")
    beaufort_parser.add_argument("--max-key-length", type=int, default=20,
                                 metavar="N", dest="max_key_length")
    beaufort_parser.add_argument("--variant", action="store_true",
                                 help="use variant Beaufort rather than plain")
    beaufort_parser.add_argument("--encrypt", action="store_true")

    autokey_parser = add("autokey", command_autokey,
                         "autokey, plaintext and ciphertext variants",
                         search=True)
    autokey_parser.add_argument("--primer", metavar="KEY")
    autokey_parser.add_argument("--mode", choices=("plaintext", "ciphertext"),
                                default="plaintext")
    autokey_parser.add_argument("--max-primer", type=int, default=8,
                                metavar="N", dest="max_primer",
                                help="longest primer to search (default 8)")
    autokey_parser.add_argument("--encrypt", action="store_true")

    # -- transposition -----------------------------------------------------
    rail_parser = add("railfence", command_railfence, "rail fence transposition")
    rail_parser.add_argument("--rails", type=int, metavar="N")
    rail_parser.add_argument("--offset", type=int, default=0, metavar="N")
    rail_parser.add_argument("--encrypt", action="store_true")

    columnar_parser = add("columnar", command_columnar,
                          "columnar transposition", search=True)
    columnar_parser.add_argument("--key", metavar="KEYWORD")
    columnar_parser.add_argument("--key-length", type=int, metavar="N",
                                 dest="key_length")
    columnar_parser.add_argument("--max-key-length", type=int, default=8,
                                 metavar="N", dest="max_key_length")
    columnar_parser.add_argument("--complete", action="store_true",
                                 help="assume a padded complete rectangle")
    columnar_parser.add_argument("--encrypt", action="store_true")

    transposition_parser = add("transposition", command_transposition,
                               "every transposition family at once",
                               search=True)
    transposition_parser.add_argument("--max-key-length", type=int, default=8,
                                      metavar="N", dest="max_key_length")
    transposition_parser.add_argument("--routes", action="store_true",
                                      help="list the routes that can be tested")

    # -- digraphic and fractionating --------------------------------------
    polybius_parser = add("polybius", command_polybius, "Polybius square")
    polybius_parser.add_argument("--key", metavar="KEYWORD")
    polybius_parser.add_argument("--square", metavar="LETTERS",
                                 help="an explicit 25- or 36-letter square")
    polybius_parser.add_argument("--words", metavar="LIST",
                                 help="candidate keywords to try")
    polybius_parser.add_argument("--encrypt", action="store_true")

    bifid_parser = add("bifid", command_bifid, "Bifid", search=True)
    bifid_parser.add_argument("--key", metavar="KEYWORD")
    bifid_parser.add_argument("--period", type=int, metavar="N")
    bifid_parser.add_argument("--max-period", type=int, default=15,
                              metavar="N", dest="max_period")
    bifid_parser.add_argument("--words", metavar="LIST")
    bifid_parser.add_argument("--encrypt", action="store_true")

    playfair_parser = add("playfair", command_playfair, "Playfair",
                          search=True)
    playfair_parser.add_argument("--key", metavar="KEYWORD")
    playfair_parser.add_argument("--filler", default="X", metavar="LETTER")
    playfair_parser.add_argument("--restarts", type=int,
                                 default=playfair.DEFAULT_RESTARTS,
                                 metavar="N",
                                 help="hill-climb restarts (default "
                                      f"{playfair.DEFAULT_RESTARTS})")
    playfair_parser.add_argument("--check", action="store_true",
                                 help="report formatting problems only")
    playfair_parser.add_argument("--encrypt", action="store_true")

    hill_parser = add("hill", command_hill, "Hill cipher (2x2 and 3x3)",
                      search=True)
    hill_parser.add_argument("--matrix", metavar="NUMBERS",
                             help="matrix read across the rows, e.g. 3,3,2,5")
    hill_parser.add_argument("--key", metavar="LETTERS",
                             help="matrix given as letters, e.g. HILL")
    hill_parser.add_argument("--size", type=int, default=None, metavar="N")
    hill_parser.add_argument("--crib", metavar="TEXT",
                             help="known plaintext, for the known-plaintext "
                                  "attack")
    hill_parser.add_argument("--crib-at", type=int, default=0, metavar="N",
                             dest="crib_at")
    hill_parser.add_argument("--encrypt", action="store_true")

    # -- other -------------------------------------------------------------
    add("encodings", command_encodings,
        "identify hex, binary, decimal, Base64 or Morse (NOT encryption)")

    crib_parser = add("crib", command_crib,
                      "test where a guessed piece of plaintext could sit")
    crib_parser.add_argument("crib", nargs="*", metavar="TEXT")
    crib_parser.add_argument("--methods", nargs="*", default=None,
                             metavar="NAME",
                             help="limit to certain cipher families")
    crib_parser.add_argument("--no-context", action="store_false",
                             dest="use_context",
                             help="ignore saved story context")

    context_parser = add("context", command_context,
                         "record the team's story notes for a ciphertext")
    context_parser.add_argument("--add", action="append", metavar="FIELD=VALUE")
    context_parser.add_argument("--clear", action="append", metavar="FIELD")

    auto_parser = add("auto", command_auto,
                      "run the whole pipeline, cheap checks first",
                      aliases=["solve"], search=True)
    effort_group = auto_parser.add_mutually_exclusive_group()
    effort_group.add_argument("--fast", dest="effort", action="store_const",
                              const="fast")
    effort_group.add_argument("--normal", dest="effort", action="store_const",
                              const="normal")
    effort_group.add_argument("--deep", dest="effort", action="store_const",
                              const="deep")
    auto_parser.add_argument("--stats", action="store_true",
                             help="include the full analyse report")
    auto_parser.set_defaults(effort="normal")

    shell_parser = subparsers.add_parser(
        "shell", help="interactive session over one ciphertext")
    add_input_arguments(shell_parser)
    add_output_arguments(shell_parser)
    add_search_arguments(shell_parser)
    shell_parser.set_defaults(handler=command_shell, group=5)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 1
    try:
        return args.handler(args) or 0
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
