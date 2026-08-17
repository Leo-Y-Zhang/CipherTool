"""ADFGVX and ADFGX: fractionation followed by transposition.

The German army cipher of 1918, and the one classical cipher in this toolkit
that is genuinely two ciphers stacked. Every plaintext letter is looked up in
a keyed Polybius square and replaced by its ROW and COLUMN labels, so the
message doubles in length and ends up written in six letters -- A, D, F, G, V
and X, chosen because they are far apart in Morse and hard to mishear. The
symbol stream is then columnar-transposed under a second keyword.

That combination is what made it strong. The fractionation alone is a
monoalphabetic substitution and falls to letter frequencies; the
transposition alone leaves the letters untouched and falls to column-pair
statistics. Stacked, each hides the other: the transposition tears every cell
in half and scatters the halves, so frequency analysis sees nothing, and the
fractionation destroys the letter statistics that a transposition attack
needs.

Attacking it
------------
Undo the transposition FIRST, and score a candidate column order by the index
of coincidence of the symbols *paired up two at a time*. When the order is
right, the pairs are the original square cells, so their distribution is the
distribution of English letters -- lumpy, IC near 0.067. When it is wrong,
the pairs straddle cell boundaries and average out towards flat, IC near
0.040 for a 36-cell square. MEASURED on a 600-letter message: 0.0678 against
0.0398. That is a wide, cheap signal, and crucially it needs no knowledge of
the square at all.

It has one blind spot, and it must be handled rather than hoped away: the
index of coincidence is a property of the multiset of pairs, so it cannot see
their ORDER. Several column orders produce the same pairs in different
sequences and score identically. Measured on that message, twelve orders tied
at the maximum -- and the true one was among them, with a clear gap of 0.0124
to the next distinct value. So the sweep does not pick a winner; it hands the
whole tied set on.

Breaking the tie is then free, because what is left IS a monoalphabetic
substitution: map each distinct cell to a letter and the message becomes an
ordinary substitution cipher, which this toolkit already solves well. The
order that yields the most English-looking plaintext wins, and the mapping it
found reconstructs the square.

So no new cryptanalysis is invented here. The work is in cutting the problem
at the joint: an IC sweep that ignores the square, then the existing
substitution climber that ignores the transposition.
"""

from __future__ import annotations

import itertools
import time
from collections import Counter
from typing import Any, Sequence

from . import columnar, polybius, substitution
from .candidates import Candidate, CandidateSet
from .normalize import NormalizedText, group_text, normalize
from .polybius import ADFGVX_LABELS, ADFGX_LABELS
from .scoring import EnglishScorer, annotate, default_scorer

#: Longest transposition key the sweep tries when not told one. Every length
#: up to this is enumerated exhaustively, so the cost is the factorial: 8! is
#: 40,320 cheap scorings, 9! is 363,000 and 10! is 3.6 million. Eight covers
#: the keyword lengths a competition actually uses.
DEFAULT_MAX_KEY_LENGTH = 8

#: Shortest transposition key worth trying. A one-column transposition is the
#: identity and a two-column one barely disturbs the stream.
MIN_KEY_LENGTH = 2

#: How many of the tied column orders are handed to the substitution stage.
#: The tie is exact and its size is a property of the cipher rather than of
#: the message -- twelve on the measured example -- so this is a safety net
#: against a pathological case, not a tuning knob.
DEFAULT_TIE_LIMIT = 24

#: Restarts for the substitution climb used to break the tie. Deliberately
#: modest: it runs once per tied order, and a wrong order cannot be rescued
#: by more restarts because there is no English in it to find.
DEFAULT_SUBSTITUTION_RESTARTS = 12

#: How close to the best index of coincidence a column order must be to count
#: as tied. Small, because the gap to the next distinct value is large.
IC_TIE_EPSILON = 1e-9

METHOD = "ADFGVX"

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def digraph_ic(symbols: str) -> float:
    """Index of coincidence over the symbols paired up two at a time.

    This is the whole signal the transposition sweep runs on. Pairs are taken
    at even offsets: after the right column order is restored, position 0 and
    1 are one cell, 2 and 3 the next, and so on.
    """
    pairs = [symbols[index:index + 2]
             for index in range(0, len(symbols) - 1, 2)]
    total = len(pairs)
    if total < 2:
        return 0.0
    counts = Counter(pairs)
    return sum(n * (n - 1) for n in counts.values()) / (total * (total - 1))


def looks_like(text: str | NormalizedText) -> str | None:
    """``"ADFGVX"``, ``"ADFGX"`` or ``None``.

    Recognition is unusually reliable for this cipher and worth doing before
    anything expensive: the ciphertext is written in five or six specific
    letters and nothing else, and its length is always even because every
    plaintext letter became exactly two symbols.
    """
    normalized = text if isinstance(text, NormalizedText) else normalize(text)
    letters = normalized.letters
    if len(letters) < 4 or len(letters) % 2:
        return None
    used = set(letters)
    if used <= set(ADFGVX_LABELS) and len(used) > len(ADFGX_LABELS):
        return "ADFGVX"
    if used <= set(ADFGX_LABELS):
        return "ADFGX"
    return None


def _square_for(keyword: str | None, labels: str) -> polybius.PolybiusSquare:
    if len(labels) == len(ADFGX_LABELS):
        return polybius.PolybiusSquare.standard(keyword, row_labels=labels)
    return polybius.PolybiusSquare.six_by_six(keyword, row_labels=labels)


def encrypt(
    text: str,
    square_keyword: str | None,
    transposition_key: str | Sequence[int],
    *,
    labels: str = ADFGVX_LABELS,
) -> str:
    """Fractionate through the keyed square, then transpose the symbols."""
    square = _square_for(square_keyword, labels)
    return columnar.encrypt(polybius.encrypt(text, square), transposition_key)


def decrypt(
    text: str,
    square_keyword: str | None,
    transposition_key: str | Sequence[int],
    *,
    labels: str = ADFGVX_LABELS,
) -> str:
    """Exact inverse of :func:`encrypt`: undo the transposition first."""
    square = _square_for(square_keyword, labels)
    return polybius.decrypt(columnar.decrypt(text, transposition_key), square)


def _tied_orders(
    letters: str,
    lengths: Sequence[int],
    tie_limit: int,
    deadline: float | None,
) -> tuple[list[tuple[int, tuple[int, ...]]], list[int], bool]:
    """Column orders whose paired-symbol IC ties for the best seen.

    Returns the tied orders with their key lengths, the lengths actually
    tried, and whether a time budget cut the sweep short.
    """
    best_ic = float("-inf")
    tied: list[tuple[int, tuple[int, ...]]] = []
    tried: list[int] = []
    budget_hit = False

    for count in lengths:
        if deadline is not None and time.monotonic() > deadline:
            budget_hit = True
            break
        if count > len(letters):
            continue
        tried.append(count)
        for order in itertools.permutations(range(count)):
            value = digraph_ic(columnar.decrypt(letters, order))
            if value > best_ic + IC_TIE_EPSILON:
                best_ic = value
                tied = [(count, order)]
            elif value > best_ic - IC_TIE_EPSILON and len(tied) < tie_limit:
                tied.append((count, order))

    return tied, tried, budget_hit


def _render_square(pairs: Sequence[str], plaintext: str, labels: str) -> str:
    """The recovered square, as ``AD=T`` cells joined by spaces.

    Reconstructed by alignment rather than from the substitution key: cell i
    of the stream produced letter i of the plaintext, so reading the two
    together IS the square. Reported because the transposition key alone does
    not let anybody re-read the message by hand, and an answer nobody can
    check is worth very little in a competition.
    """
    cells: dict[str, str] = {}
    for pair, letter in zip(pairs, plaintext):
        cells.setdefault(pair, letter)
    return " ".join(f"{pair}={letter}" for pair, letter in sorted(cells.items()))


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    **options: Any,
) -> CandidateSet:
    """Attack an ADFGVX (or ADFGX) message. Ranked candidates only.

    Returns nothing at all for text that is not written in the five or six
    labels, which is the honest answer rather than a best guess: this attack
    has no meaning applied to anything else.

    Options
    -------
    key_length:
        Attack this transposition key length only.
    max_key_length:
        Otherwise try every length from 2 up to this
        (default :data:`DEFAULT_MAX_KEY_LENGTH`).
    restarts:
        Restarts for the substitution climb that breaks the IC tie.
    seed:
        Makes the substitution stage reproducible.
    time_budget:
        Seconds. The sweep stops cleanly and records ``time_budget_hit``.
    """
    engine = scorer or default_scorer()
    text = source if isinstance(source, NormalizedText) else normalize(source)
    letters = text.letters

    results = CandidateSet(source_letters=letters)
    variant = looks_like(text)
    if variant is None:
        return results

    key_length = options.pop("key_length", None)
    max_key_length = int(options.pop("max_key_length", DEFAULT_MAX_KEY_LENGTH))
    restarts = int(options.pop("restarts", DEFAULT_SUBSTITUTION_RESTARTS))
    tie_limit = int(options.pop("tie_limit", DEFAULT_TIE_LIMIT))
    seed = options.pop("seed", None)
    time_budget = options.pop("time_budget", None)
    if options:
        raise ValueError(
            f"unknown option(s) for ADFGVX: {', '.join(sorted(options))}"
        )

    lengths = ([int(key_length)] if key_length is not None
               else list(range(MIN_KEY_LENGTH, max_key_length + 1)))
    deadline = (time.monotonic() + float(time_budget)
                if time_budget is not None else None)

    tied, tried, budget_hit = _tied_orders(letters, lengths, tie_limit,
                                           deadline)

    for count, order in tied:
        if deadline is not None and time.monotonic() > deadline:
            budget_hit = True
            break
        stream = columnar.decrypt(letters, order)
        pairs = [stream[index:index + 2]
                 for index in range(0, len(stream) - 1, 2)]
        distinct = sorted(set(pairs))
        if len(distinct) > len(_LETTERS):
            # More cells in use than there are letters to map them to. A
            # competition plaintext is letters only, so this means the guess
            # is wrong rather than that the message contains digits.
            continue
        alphabet = {pair: _LETTERS[index]
                    for index, pair in enumerate(distinct)}
        mapped = "".join(alphabet[pair] for pair in pairs)

        found = substitution.solve(mapped, scorer=engine, top=1,
                                   restarts=restarts, seed=seed)
        best = found.best()
        if best is None:
            continue

        diagnostics: dict[str, Any] = {
            "variant": variant,
            "transposition_key_length": count,
            "transposition_permutation": order,
            "digraph_ic": round(digraph_ic(stream), 5),
            "cells_used": len(distinct),
            "key_lengths_tried": ",".join(str(value) for value in tried),
            "tied_orders": len(tied),
            "square": _render_square(pairs, best.plaintext, ADFGVX_LABELS),
            "search": (
                "exhaustive over column orders, then a substitution climb "
                "per tied order"
            ),
        }
        annotate(diagnostics, best.plaintext, engine)
        keyword = columnar.keyword_from_order(order)
        results.add(
            Candidate(
                method=f"{variant} (fractionation + transposition)",
                key=f"transposition={keyword or order} ({count})",
                score=best.score,
                plaintext=best.plaintext,
                diagnostics=diagnostics,
                # Both stages move letters around, so position i of the
                # plaintext has nothing to do with position i of the input.
                display=group_text(best.plaintext),
            )
        )

    for candidate in results.ranked():
        if budget_hit:
            candidate.diagnostics["time_budget_hit"] = True

    if top is not None and top > 0:
        return CandidateSet(results.top(top), source_letters=letters)
    return results
