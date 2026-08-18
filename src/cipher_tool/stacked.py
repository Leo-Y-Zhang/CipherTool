"""Two ciphers piled up: a polyalphabetic, then a transposition over it.

The problem
-----------
Every other solver in this toolkit attacks ONE cipher. Stack two and each of
them fails, and fails in the most misleading way available: the correct
intermediate answer is not English, so the scorer that is supposed to
recognise success rejects it.

That is not hypothetical. The 2017 National Cipher Challenge, challenge 7B,
is a Vigenere under the key SCYTALE followed by a six-column transposition.
Attacked as a Vigenere it reaches `weak`; attacked as a transposition it
reaches `weak`. An earlier attempt on it recovered the key correctly, read
the resulting gibberish, and concluded the alphabets must be mixed -- filing
a second cipher as evidence for a harder version of the first. The key had
been right all along.

With this module the same 3,583 letters come back at `strong` in under three
seconds, matching the published answer exactly, letter for letter.

**A perfect key that reads as nonsense means another layer, not a wrong
key.** That is the whole lesson, and it is why this module exists.

Cutting it at the joint
-----------------------
The same move as ADFGVX: find a statistic that survives the OUTER layer, use
it to strip the inner one, and hand what is left to a solver that already
works.

A columnar transposition reads out one whole column at a time, so each
column arrives in the ciphertext as a CONTIGUOUS run. If a periodic
polyalphabetic was applied before the transposition, then inside any one of
those runs the key still advances with its own period. So:

    split the ciphertext into `width` contiguous blocks, and measure the
    mean index of coincidence of the cosets at spacing `period` WITHIN each
    block.

When both numbers are right, every coset is one monoalphabetic image of
English and the mean sits near 0.066. When either is wrong, cosets straddle
alphabets and it falls towards 0.038. MEASURED on the real 3,583-letter 7B
ciphertext: 0.0661 at width 6, period 7, against a worst case of 0.0404 --
and it needs no knowledge of the key, the alphabet or the column order.

Two guards, both learned by measuring rather than by reasoning:

* **Plain English scores highly at every width and period**, because English
  cosets have an English index of coincidence whatever you do to them.
  Measured on 3,500 letters of prose, the WORST setting still scored 0.0649,
  so the detector cannot tell prose from a peeled stack. The whole-message
  index of coincidence is checked first, but that is an EARLY EXIT rather
  than the safety net: measured over sixteen English-shaped texts, every one
  was already refused further down, because the smallest tied shape for
  English is a width of 1 -- no transposition, and nothing here to attack.

* **Multiples of the true shape peak too.** Width 12 scored 0.0651 against
  width 6's 0.0661 on the same message, because splitting each real column
  in half leaves the key phase intact inside each half. Taking the highest
  score alone would report width 12. The smallest shape within a whisker of
  the best is the answer, exactly as with a Vigenere key length.

The ragged grid, and a search that did not pay for itself
--------------------------------------------------------
When the message is not a whole number of rows, some columns hold one more
letter than the others -- and those are the first few columns of the GRID,
which arrive in the ciphertext in KEY order. So which of the contiguous
blocks are the long ones depends on the key, which is exactly what is not
known yet. MEASURED on a 4,000-letter message under ZEBRAS, the blocks run
666, 667, 667, 667, 666, 667 while the obvious guess gives 667, 667, 667,
667, 666, 666.

Searching every possible assignment was built, and then measured against
simply guessing, over the same 100 stacked messages: 61 readings of 99 per
cent or better against 62, 42 exact against 40, and no confident-and-wrong
answer either way. It made no difference that could be told from noise, so
the guess is what ships and the search is gone. What it leaves behind is a
handful of letters at block boundaries -- three in 2,000 on the measured
case -- which is why an answer here is worth re-reading at the joins.

Once the shape is known the polyalphabetic layer comes off without a search.
Each block is lined up against the first by cross-correlating its coset
letter counts; the aligned cosets are pooled, which is what makes this
robust -- seven cosets of five hundred letters instead of forty-two of
eighty; the pooled cosets are lined up with each other the same way; and one
chi-squared against English fixes the single remaining absolute shift. What
falls out is the transposed plaintext, letter for letter, and the existing
columnar solver finishes the job with the width already pinned.

Reading the key that comes back
-------------------------------
The key is reported as it runs along a COLUMN, and that is not the order the
setter wrote it in. The polyalphabetic was applied along the plaintext, but
going down a column of the transposition steps the plaintext index by
`width`, so the key returns sampled with stride ``width mod period``, from an
unknown starting point.

On the real 2017 7B message that stride is 6 mod 7, which is -1: the key
comes back as SELATYC, and SCYTALE read backwards is ELATYCS, of which
SELATYC is a rotation. The letters are right and the reading order is a
decimation. Anyone reporting a key to a competition should undo that before
writing it down; the plaintext, which is what this module actually claims,
is unaffected either way.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Sequence

from . import columnar
from .candidates import Candidate, CandidateSet
from .normalize import NormalizedText, group_text, normalize
from .scoring import EnglishScorer, annotate, default_scorer

#: Fewest letters worth attempting. The sweep needs `width * period` cosets
#: each holding enough letters for an index of coincidence to mean anything,
#: so short messages cannot support it. MEASURED: a 600-letter message built
#: from a 7-letter key and a 6-column transposition was not detectable at all
#: and the solver correctly reported `unlikely`; the same construction at
#: 1,200 letters was detected and solved exactly.
MIN_LETTERS = 900

#: A coset shorter than this tells you nothing, so it is left out of the mean
#: rather than allowed to add noise to it.
MIN_COSET = 20

#: Above this, the whole message already has English letter statistics, so
#: there is no polyalphabetic layer to peel and this attack does not apply.
#: Halfway between a polyalphabetic 0.042 and an English 0.066.
MAX_SOURCE_IC = 0.054

#: The mean within-block coset IC a real shape has to reach. Well below
#: English, because the cosets are short and noisy, and well above the 0.040
#: a wrong shape produces.
MIN_SIGNAL = 0.058

#: How close to the best score another shape must be to count as tied. Set
#: from the measured gap: on the real message width 12 scored 0.0651 against
#: width 6's 0.0661, and the smallest of the tied shapes is the answer.
NEAR_BEST = 0.004

# MEASURED AND REJECTED, so that nobody spends the evening on it twice.
# When the grid is ragged the coset statistic cannot always separate layouts
# that differ by one letter at a block boundary, and the wrong choice costs
# a handful of letters in the answer. Peeling the best THREE layouts in full
# and letting the English scorer pick between them was tried over the same
# 100 stacked messages: exact answers went from 42 to 44, and the sweep went
# from 198 seconds to 531. Two more exact readings for two and a half times
# the runtime is not a trade worth making, and it moved the number that
# actually matters -- readings of 99 percent or better, and confident
# answers that are wrong -- not at all.
DEFAULT_MAX_WIDTH = 12
DEFAULT_MAX_PERIOD = 12

METHOD = "Polyalphabetic, then transposition"

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: Letter frequencies of English, percent, A to Z. Used only to fix the one
#: absolute shift left at the end, where the text really is English-shaped.
_ENGLISH = (8.17, 1.49, 2.78, 4.25, 12.70, 2.23, 2.02, 6.09, 6.97, 0.15,
            0.77, 4.03, 2.41, 6.75, 7.51, 1.93, 0.10, 5.99, 6.33, 9.06,
            2.76, 0.98, 2.36, 0.15, 1.97, 0.07)


def index_of_coincidence(text: str) -> float:
    """Probability that two letters drawn from *text* are the same."""
    total = len(text)
    if total < 2:
        return 0.0
    counts = Counter(text)
    return sum(n * (n - 1) for n in counts.values()) / (total * (total - 1))


def split_blocks(letters: str, width: int) -> list[str]:
    """Cut the ciphertext into *width* contiguous blocks.

    These are the columns a columnar transposition wrote out, in whatever
    order it wrote them. The order does not matter here, because the
    statistic is computed inside each block -- but the LENGTHS do, and they
    are not obvious.

    When the message is not a whole number of rows, some grid columns hold
    one more letter than the others, and those are the first few columns of
    the GRID. They arrive in the ciphertext in KEY order, so which of the
    contiguous blocks are the long ones depends on the key, which is exactly
    what is not known yet. MEASURED on a 4,000-letter message under the key
    ZEBRAS: the ciphertext blocks run 666, 667, 667, 667, 666, 667, while
    assuming the long ones come first gives 667, 667, 667, 667, 666, 666.

    The obvious guess is used: the long blocks come first. It is right only
    when the key happens to read the long columns first, and searching the
    alternatives was measured and found not to pay -- see the module
    docstring.
    """
    total = len(letters)
    size, extra = divmod(total, width)
    chosen = set(range(extra))
    blocks: list[str] = []
    at = 0
    for index in range(width):
        length = size + (1 if index in chosen else 0)
        blocks.append(letters[at:at + length])
        at += length
    return blocks


def block_signal(letters: str, width: int, period: int) -> float:
    """Mean index of coincidence of the cosets INSIDE each block.

    This is the whole detector. It asks one question -- "if I cut this into
    `width` pieces, does a key of `period` letters run through each piece?"
    -- and answers it without knowing the key or the column order.
    """
    values: list[float] = []
    for block in split_blocks(letters, width):
        for offset in range(period):
            coset = block[offset::period]
            if len(coset) >= MIN_COSET:
                values.append(index_of_coincidence(coset))
    return sum(values) / len(values) if values else 0.0


def detect(
    letters: str,
    *,
    max_width: int = DEFAULT_MAX_WIDTH,
    max_period: int = DEFAULT_MAX_PERIOD,
) -> tuple[int, int, float] | None:
    """The smallest shape that carries the signal, or ``None``.

    Returns ``(width, period, signal)``.

    Smallest, not best: multiples of the true shape score almost as highly,
    because splitting a real column in half leaves the key phase intact
    inside each half. MEASURED over 120 constructions, the highest-scoring
    shape was NOT the true one in 73 of them.
    """
    scan = [
        (block_signal(letters, width, period), width, period)
        for width in range(1, max_width + 1)
        for period in range(2, max_period + 1)
    ]
    if not scan:
        return None
    best = max(value for value, _, _ in scan)
    if best < MIN_SIGNAL:
        return None
    tied = sorted((width, period) for value, width, period in scan
                  if value >= best - NEAR_BEST)
    width, period = tied[0]
    return width, period, best


def _counts(text: str) -> list[int]:
    counter = Counter(text)
    return [counter.get(letter, 0) for letter in _ALPHABET]


def _best_rotation(reference: Sequence[int], other: Sequence[int]) -> int:
    """The rotation of *other* that lines its shape up with *reference*."""
    return max(
        range(26),
        key=lambda turn: sum(reference[i] * other[(i + turn) % 26]
                             for i in range(26)),
    )


def peel(letters: str, width: int, period: int) -> tuple[str, str]:
    """Strip the polyalphabetic layer. Returns ``(text, key)``.

    Nothing here is a search. The blocks are lined up with each other, the
    aligned cosets are pooled and lined up with each other, and one
    chi-squared fixes the single absolute shift that is left. Pooling is what
    makes it reliable: on the real message it turns forty-two cosets of
    eighty letters into seven of five hundred.
    """
    blocks = split_blocks(letters, width)
    profiles = [[_counts(block[offset::period]) for offset in range(period)]
                for block in blocks]

    # Line every block up against the first: which shift of its cosets makes
    # its alphabets agree with block 0's?
    deltas = [0]
    for index in range(1, width):
        deltas.append(max(
            range(period),
            key=lambda turn: sum(
                sum(a * b for a, b in zip(profiles[0][offset],
                                          profiles[index][(offset + turn)
                                                          % period]))
                for offset in range(period)
            ),
        ))

    pooled: list[list[int]] = []
    for offset in range(period):
        total = [0] * 26
        for index in range(width):
            for letter, count in enumerate(
                profiles[index][(offset + deltas[index]) % period]
            ):
                total[letter] += count
        pooled.append(total)

    rotations = [_best_rotation(pooled[0], pooled[offset])
                 for offset in range(period)]

    stripped: list[str] = []
    for index, block in enumerate(blocks):
        for position, letter in enumerate(block):
            coset = (position % period - deltas[index]) % period
            stripped.append(
                _ALPHABET[(ord(letter) - 65 - rotations[coset]) % 26]
            )
    text = "".join(stripped)

    counts = _counts(text)
    total = len(text)

    def chi_squared(shift: int) -> float:
        return sum(
            (counts[(i + shift) % 26] - _ENGLISH[i] * total / 100) ** 2
            / (_ENGLISH[i] * total / 100)
            for i in range(26)
        )

    shift = min(range(26), key=chi_squared)
    key = "".join(_ALPHABET[(rotations[offset] + shift) % 26]
                  for offset in range(period))
    return (
        "".join(_ALPHABET[(ord(letter) - 65 - shift) % 26] for letter in text),
        key,
    )


def solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    **options: Any,
) -> CandidateSet:
    """Attack a polyalphabetic with a transposition laid over it.

    Returns nothing at all -- not a best guess -- for text this attack has no
    business on: too short, already English-shaped, or carrying no periodic
    signal inside contiguous blocks.

    Options
    -------
    width:
        Pin the number of columns instead of detecting it.
    period:
        Pin the polyalphabetic key length.
    max_width, max_period:
        Bounds for the sweep.
    time_budget:
        Seconds; the sweep stops cleanly and candidates record
        ``time_budget_hit``.
    seed:
        Forwarded to the columnar search that finishes the job.
    """
    engine = scorer if scorer is not None else default_scorer()
    text = normalize(source) if isinstance(source, str) else source
    letters = text.letters

    width = options.pop("width", None)
    period = options.pop("period", None)
    max_width = int(options.pop("max_width", DEFAULT_MAX_WIDTH))
    max_period = int(options.pop("max_period", DEFAULT_MAX_PERIOD))
    seed = options.pop("seed", None)
    time_budget = options.pop("time_budget", None)
    if options:
        raise ValueError(
            "unknown option(s) for stacked.solve: "
            + ", ".join(sorted(str(name) for name in options))
        )
    if time_budget is not None and time_budget <= 0:
        raise ValueError(f"time_budget must be positive, got {time_budget}")

    results = CandidateSet(source_letters=letters)
    if len(letters) < MIN_LETTERS:
        return results
    # English letter statistics mean there is no polyalphabetic layer to
    # peel. This is an EARLY EXIT, not the safety net -- and saying so
    # matters, because deleting it changes no answer and a comment claiming
    # otherwise would be the kind of thing nobody re-checks. MEASURED on
    # sixteen English-shaped texts (prose, columnar, block permutation and
    # rail fence, 1,000 to 5,000 letters): every one of them was already
    # refused further down, because `detect` returns a width of 1 for text
    # whose cosets look English at every shape. What this line actually buys
    # is skipping 132 passes over the message, on a stage that runs on every
    # `normal` solve.
    if index_of_coincidence(letters) > MAX_SOURCE_IC:
        return results

    deadline = (time.monotonic() + float(time_budget)
                if time_budget is not None else None)

    if width is not None and period is not None:
        shape: tuple[int, int, float] | None = (
            int(width), int(period),
            block_signal(letters, int(width), int(period)),
        )
    else:
        shape = detect(letters, max_width=max_width, max_period=max_period)
    if shape is None:
        return results

    found_width, found_period, signal = shape
    # Width 1 means no transposition at all: an ordinary periodic
    # polyalphabetic, which the Vigenere and Beaufort solvers already do
    # properly. Answering here would duplicate them and claim a stack that
    # is not there.
    if found_width < 2:
        return results

    stripped, key = peel(letters, found_width, found_period)

    plans: list[dict[str, Any]] = [
        {"key_length": found_width, "seed": seed},
        {"max_key_length": max(9, found_width), "seed": seed},
    ]
    for plan in plans:
        if deadline is not None and time.monotonic() > deadline:
            break
        inner = columnar.solve(stripped, scorer=engine, top=3, **plan)
        for candidate in inner.ranked():
            diagnostics: dict[str, Any] = {
                "columns": found_width,
                "polyalphabetic_period": found_period,
                "polyalphabetic_key": key,
                "block_coset_ic": round(signal, 5),
                "transposition_key": candidate.key,
                "search": (
                    "index of coincidence inside contiguous blocks, then a "
                    "columnar search on what was left"
                ),
            }
            annotate(diagnostics, candidate.plaintext, engine)
            results.add(
                Candidate(
                    method=METHOD,
                    key=f"key={key} ({found_period}) then {candidate.key}",
                    score=candidate.score,
                    plaintext=candidate.plaintext,
                    diagnostics=diagnostics,
                    # Both layers move or replace letters, so the input
                    # spacing means nothing here.
                    display=group_text(candidate.plaintext),
                )
            )

    if deadline is not None and time.monotonic() > deadline:
        for candidate in results.ranked():
            candidate.diagnostics["time_budget_hit"] = True

    if top is not None and top > 0:
        return CandidateSet(results.top(top), source_letters=letters)
    return results
