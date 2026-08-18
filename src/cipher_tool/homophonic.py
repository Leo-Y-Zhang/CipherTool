"""Homophonic substitution: many symbols, twenty-six letters.

The cipher
----------
A homophonic substitution gives the commonest letters several symbols each, so
that the flat frequency profile a solver looks for is never there. Fifty-two
symbols onto twenty-six letters -- two apiece, a playing card per letter-half
-- is the classic shape, and it is what a deck-of-cards transcription turns
out to be.

``substitution.py`` cannot read it: its whole model is one cipher letter to
one plain letter. ``polybius.py`` cannot read it either, for the reason its
own docstring gives -- above twenty-six cells the mapping stops being a
bijection and becomes a different cipher.

The attack, and why the constraint IS the attack
------------------------------------------------
Simulated annealing over an assignment of symbols to letters, with two rules:

* the number of symbols standing for each letter is **fixed** before the
  search starts (the slot multiset), and
* the only move is a **swap** of the letters on two symbols, which preserves
  that multiset by construction.

MEASURED 2026-08-18, with the constraint: 0.983 exact letters on a 52-symbol
cipher from 400 units, and 0.901 on a 36-symbol one from 313 (at twelve
restarts; six gives 0.856).

What the constraint is FOR is not what it looks like. On genuine homophonic
ciphertext the unconstrained search does no harm at all -- at 52/400, 52/600,
36/313, 40/320 and 52/340 it matched or beat the constrained one, reaching
1.000 exact letters. The constraint earns its keep on a stream that is NOT
homophonic, which is the case that matters, because that is when a search
manufactures a reading of something it cannot read. On 600 units of uniformly
random cards, with no plaintext to find:

    constrained     26 letters used, -2.179 per letter, labelled `weak`
    UNCONSTRAINED    8 letters used, -1.248 per letter, labelled `promising`
                    "SEEITSSESINILETSTATSSITSSITSSSTSEETSISASSITSLESS..."

Under a quadgram model a five-letter language scores better per letter than
English when nothing forbids it, and the free search walks off into IS, IT,
AT, AN and AS and stays there -- scoring nearly a whole log unit per letter
BETTER than the honest reading. Score alone cannot tell those two apart. A key
that must spend a symbol on Z cannot become a key that only writes AEILNRST,
and that is the entire defence.

``constrain_slots=False`` exists so that ``tests/test_homophonic.py`` can
watch that collapse happen. It changes the move set, not a constant.

Honest limits
-------------
* The search is annealing, not exhaustive. It says so in its diagnostics.
* Below :data:`MIN_UNITS_PER_SYMBOL` units per symbol it refuses outright, and
  below :data:`CONFIDENT_UNITS_PER_SYMBOL` it caps its own confidence at
  ``promising``. The two measured recoveries sat at 9.1 and 9.5 units per
  observed symbol; a search with more key freedom than the ciphertext can pay
  for produces a fluent sentence that is not the plaintext.
* **A fixed multiset that is WRONG cannot be climbed out of**, because no swap
  can change it. MEASURED on a 32-symbol cipher of which only 28 symbols
  occur in 260 units: 0.065, 0.000 and 0.850 exact letters at three seeds,
  and the 0.000 run was labelled `promising`. Just above
  :data:`MIN_SYMBOLS` the assumed multiset is most likely to be wrong and the
  family is least distinguishable from a plain substitution. The
  units-per-symbol gates do not see this, and nothing here pretends they do.
* It does not repair a broken transcription. If the paired-symbol recogniser
  reports that alternation breaks, every unit past the break is a fiction and
  this module refuses rather than reading one.
"""

from __future__ import annotations

import math
import random
import time
from typing import Iterable, Mapping, Sequence

from . import paired
from .candidates import Candidate, CandidateSet
from .normalize import ALPHABET, ALPHABET_SIZE, NormalizedText, normalize
from .reference import ENGLISH_FREQUENCY_ORDER, ENGLISH_LETTER_PROBABILITY
from .scoring import EnglishScorer, annotate, default_scorer

METHOD = "Homophonic substitution (annealing over a fixed letter-slot multiset)"

#: Independent runs from different starting keys. Half are seeded from the
#: symbol frequencies, half are random -- see :func:`_starting_key`.
DEFAULT_RESTARTS = 6

#: Moves per restart. Geometric cooling is spread over exactly this many.
DEFAULT_ITERATIONS = 60_000

#: Which slot multiset to build when the caller does not say. Uniform, because
#: it is what the measured recoveries used and because a frequency-shaped
#: multiset assumes the plaintext's letter profile before the search has seen
#: any of it.
DEFAULT_SLOT_MODEL = "uniform"

#: Below this many units the search has more freedom than the text can pay
#: for whatever the symbol count is.
MIN_UNITS = 200

#: Refuse below this many units per symbol. The two measured successes were at
#: 9.1 and 9.5 units per OBSERVED symbol; this is the floor below which the
#: answer means nothing.
MIN_UNITS_PER_SYMBOL = 6.0

#: Below this, the label is capped at `promising` however well it scores.
CONFIDENT_UNITS_PER_SYMBOL = 8.0

#: Below this word coverage the label is capped at `weak`, regardless of how
#: much ciphertext there was.
#:
#: Units-per-symbol measures whether the ATTACK had enough evidence; it says
#: nothing about whether this particular RUN found the key. An unlucky seed
#: fails on ciphertext that is comfortably long enough, and the failure is
#: invisible to every other signal, because a mis-keyed homophonic decrypt is
#: English-shaped window by window BY CONSTRUCTION -- every symbol still maps
#: to a real letter. That trips the partial-prose promotion in
#: ``candidates.py``, which lifts weak to `promising`.
#:
#: MEASURED over 24 runs at 400, 626 and 820 units, eight seeds each, at the
#: paste screen's own restart count. Readings that recovered 90 per cent or
#: more of the letters scored word coverage 0.833 to 0.955; readings that
#: recovered under half scored 0.307 to 0.383. Nothing landed in between, and
#: this floor sits in the middle of that empty band.
RELIABLE_WORD_COVERAGE = 0.60

#: Twenty-six or fewer symbols is a monoalphabetic substitution, and
#: ``substitution.py`` breaks those better than this ever will.
MIN_SYMBOLS = 27

#: Above this it is not a homophonic key; it is a codebook, and a different
#: problem.
MAX_SYMBOLS = 200

#: Quadgram window, matching the model in ``scoring.py``.
_WINDOW = 4

#: Geometric cooling bounds, in the units the scorer works in (log10
#: probability per window). Tuned against the two measured recoveries.
_TEMPERATURE_START = 1.5
_TEMPERATURE_END = 0.06

#: Reading the clock is a syscall, so do it once per this many moves.
_CLOCK_EVERY = 256

#: The tokens :func:`random_key` builds a key from: a playing-card deck, which
#: is the inventory this family actually turns up in.
_DECK_RANKS = "23456789XJQKA"
_DECK_SUITS = "CDHS"


class HomophonicResult(CandidateSet):
    """A :class:`CandidateSet` that also carries notes about the search.

    Every refusal in this module returns an empty one of these with the reason
    in ``notes``. ``auto_solve`` already surfaces ``.notes`` in its stage
    report, so a refusal is visible rather than looking like a stage that
    found nothing.
    """

    def __init__(
        self,
        candidates: Iterable[Candidate] = (),
        notes: Iterable[str] = (),
    ) -> None:
        super().__init__(candidates)
        self.notes: tuple[str, ...] = tuple(notes)


def slot_multiset(symbol_count: int, *, model: str = DEFAULT_SLOT_MODEL
                  ) -> dict[str, int]:
    """How many symbols stand for each letter. The values sum to *symbol_count*.

    ``"uniform"``
        ``symbol_count // 26`` each, with the remainder given to the commonest
        letters. For 52 that is exactly two apiece.
    ``"frequency"``
        One each, and the rest shared out in proportion to English letter
        frequency by largest remainder.
    """
    if symbol_count < ALPHABET_SIZE:
        raise ValueError(
            f"a homophonic key needs at least {ALPHABET_SIZE} symbols, one "
            f"per letter; got {symbol_count}"
        )
    if model == "uniform":
        base, remainder = divmod(symbol_count, ALPHABET_SIZE)
        slots = {letter: base for letter in ALPHABET}
        for letter in ENGLISH_FREQUENCY_ORDER[:remainder]:
            slots[letter] += 1
        return slots
    if model == "frequency":
        slots = {letter: 1 for letter in ALPHABET}
        spare = symbol_count - ALPHABET_SIZE
        shares = {
            letter: ENGLISH_LETTER_PROBABILITY[letter] * spare
            for letter in ALPHABET
        }
        for letter in ALPHABET:
            slots[letter] += int(shares[letter])
        # Largest remainder, ties broken by frequency order so the result does
        # not depend on dictionary iteration order.
        left = symbol_count - sum(slots.values())
        ordered = sorted(
            ALPHABET,
            key=lambda letter: (
                -(shares[letter] - int(shares[letter])),
                ENGLISH_FREQUENCY_ORDER.index(letter),
            ),
        )
        for letter in ordered[:left]:
            slots[letter] += 1
        return slots
    raise ValueError(
        f"unknown slot model {model!r}; choose 'uniform' or 'frequency'"
    )


def deck_tokens(count: int) -> list[str]:
    """*count* distinct rank-and-suit tokens, in a fixed order.

    A deck is the inventory this cipher family arrives in, and using it here
    keeps the test ciphertexts the same shape as the real material.
    """
    tokens = [rank + suit for rank in _DECK_RANKS for suit in _DECK_SUITS]
    if count > len(tokens):
        raise ValueError(
            f"a deck has {len(tokens)} cards; {count} tokens were asked for. "
            "Supply your own tokens for a larger inventory."
        )
    return tokens[:count]


def random_key(
    symbol_count: int,
    *,
    model: str = DEFAULT_SLOT_MODEL,
    seed: int | None = None,
) -> dict[str, tuple[str, ...]]:
    """A random homophonic key: letter -> the tokens that stand for it."""
    slots = slot_multiset(symbol_count, model=model)
    tokens = deck_tokens(symbol_count)
    generator = random.Random(seed)
    generator.shuffle(tokens)
    key: dict[str, tuple[str, ...]] = {}
    cursor = 0
    for letter in ALPHABET:
        take = slots[letter]
        key[letter] = tuple(tokens[cursor:cursor + take])
        cursor += take
    return key


def encrypt(
    plaintext: str,
    key: Mapping[str, Sequence[str]],
    *,
    seed: int | None = None,
) -> list[str]:
    """Encipher *plaintext*, choosing among each letter's tokens at random."""
    generator = random.Random(seed)
    letters = "".join(
        character for character in plaintext.upper() if character in ALPHABET
    )
    units: list[str] = []
    for letter in letters:
        options = key.get(letter)
        if not options:
            raise ValueError(f"the key has no symbol for {letter!r}")
        units.append(generator.choice(list(options)))
    return units


def decrypt(units: Sequence[str], mapping: Mapping[str, str]) -> str:
    """Decipher *units* through a token-to-letter *mapping*."""
    missing = sorted({unit for unit in units if unit not in mapping})
    if missing:
        raise ValueError(
            f"the mapping does not cover {len(missing)} token(s): "
            f"{', '.join(missing[:5])}"
        )
    return "".join(mapping[unit] for unit in units)


def solve(
    source: str | NormalizedText | Sequence[str],
    *,
    scorer: EnglishScorer | None = None,
    top: int = 5,
    **options: object,
) -> HomophonicResult:
    """Attack a homophonic substitution, or say why it will not try.

    *source* may be raw text, a :class:`NormalizedText`, or a sequence of
    tokens. A sequence is taken as the unit stream exactly as given: the
    caller has already decided the tokenisation and this will not second-guess
    it.

    Options
    -------
    units:
        An explicit unit list, overriding whatever *source* would give.
    unit_size:
        1 or 2 symbols per unit. The default is 2 when the paired-symbol
        recogniser sees two alternating alphabets, and 1 otherwise.
    restarts, iterations, seed:
        Search size and reproducibility.
    slots:
        ``"uniform"`` or ``"frequency"``.
    constrain_slots:
        Default True. See the module docstring; False exists for one test.
    time_budget:
        Seconds. Accepted and honoured, and NEVER a reason to raise --
        ``auto_solve`` retries only on ``TypeError``, so a solver that
        rejects a budget is dropped from every timed run without a word.
    """
    engine = scorer or default_scorer()
    supplied_units = options.pop("units", None)
    unit_size = options.pop("unit_size", None)
    restarts = int(options.pop("restarts", DEFAULT_RESTARTS) or 1)
    iterations = int(options.pop("iterations", DEFAULT_ITERATIONS))
    seed = options.pop("seed", None)
    slot_model = str(options.pop("slots", DEFAULT_SLOT_MODEL))
    constrain_slots = bool(options.pop("constrain_slots", True))
    time_budget = options.pop("time_budget", None)
    if options:
        raise ValueError(
            "unknown option(s) for the homophonic search: "
            f"{', '.join(sorted(str(name) for name in options))}"
        )
    if time_budget is not None and float(time_budget) <= 0:
        return HomophonicResult(
            notes=("no time was left for the homophonic search",)
        )

    if supplied_units is not None:
        units = [str(unit) for unit in supplied_units]
        unit_size = int(unit_size) if unit_size else _token_size(units)
    else:
        units, unit_size, refusal = _units_from(source, unit_size)
        if refusal:
            return HomophonicResult(notes=(refusal,))

    inventory = sorted(set(units))
    symbols = len(inventory)
    if symbols < MIN_SYMBOLS:
        return HomophonicResult(notes=(
            f"only {symbols} distinct symbols, so this is a monoalphabetic "
            "substitution rather than a homophonic one; the substitution "
            "solver does that job and does it better",
        ))
    if symbols > MAX_SYMBOLS:
        return HomophonicResult(notes=(
            f"{symbols} distinct symbols is more than {MAX_SYMBOLS}, which is "
            "a codebook rather than a homophonic key",
        ))
    if len(units) < MIN_UNITS:
        return HomophonicResult(notes=(
            f"only {len(units)} units; a homophonic search needs at least "
            f"{MIN_UNITS} before its answer means anything",
        ))
    per_symbol = len(units) / symbols
    if per_symbol < MIN_UNITS_PER_SYMBOL:
        return HomophonicResult(notes=(
            f"{per_symbol:.1f} units per symbol, below the floor of "
            f"{MIN_UNITS_PER_SYMBOL:.0f}; the search has more key freedom "
            "than this much ciphertext can pay for and would return a fluent "
            "sentence that is not the plaintext",
        ))

    slots = slot_multiset(symbols, model=slot_model)
    index_of = {symbol: index for index, symbol in enumerate(inventory)}
    values = [index_of[unit] for unit in units]

    annealer = _Annealer(values, engine.table(), symbols)
    if not annealer.searchable:
        return HomophonicResult(notes=(
            "the unit stream is shorter than one scoring window, so there is "
            "nothing to search",
        ))

    generator = random.Random(seed)
    deadline = (
        time.monotonic() + float(time_budget) if time_budget is not None
        else None
    )
    best_key: list[int] | None = None
    best_total = float("-inf")
    audit_gap = 0.0
    cut_short = False
    runs = 0
    for attempt in range(restarts):
        start = _starting_key(
            annealer.counts, slots, generator, frequency_seeded=not attempt % 2
        )
        key, total, stopped = annealer.run(
            start, iterations, generator,
            deadline=deadline, constrain_slots=constrain_slots,
        )
        runs += 1
        cut_short = cut_short or stopped
        audit_gap = max(audit_gap, abs(total - annealer.full_score(key)))
        if total > best_total:
            best_total, best_key = total, list(key)
        if stopped:
            break

    if best_key is None:  # unreachable while restarts >= 1; not assumed
        return HomophonicResult(notes=("the search made no attempt",))

    plaintext = "".join(ALPHABET[best_key[value]] for value in values)
    diagnostics: dict[str, object] = {
        "symbols": symbols,
        "units": len(units),
        "units_per_symbol": round(per_symbol, 2),
        "restarts": runs,
        "iterations": iterations,
        "slot_model": slot_model,
        "unit_size": unit_size,
        "search": (
            f"annealing, not exhaustive: {runs} restart(s) of {iterations} "
            "moves over a fixed letter-slot multiset"
        ),
        "score_audit_gap": round(audit_gap, 9),
    }
    if not constrain_slots:
        diagnostics["slot_constraint"] = (
            "DISABLED -- any symbol may take any letter. This search is known "
            "to collapse onto a handful of letters and its output is not a "
            "reading of anything"
        )
    if cut_short:
        diagnostics["time_budget_hit"] = True
    if per_symbol < CONFIDENT_UNITS_PER_SYMBOL:
        diagnostics["confidence_cap"] = "promising"
        diagnostics["thin_ciphertext"] = (
            f"{per_symbol:.1f} units per symbol, below the "
            f"{CONFIDENT_UNITS_PER_SYMBOL:.0f} at which this attack was "
            "measured working"
        )
    annotate(diagnostics, plaintext, engine)

    # `annotate` is what measures word coverage, so this has to follow it.
    # A run that simply did not find the key is capped here; the cap above
    # only knows whether there was enough ciphertext to try.
    coverage = diagnostics.get("word_coverage")
    if coverage is not None and coverage < RELIABLE_WORD_COVERAGE:
        diagnostics["confidence_cap"] = "weak"
        diagnostics["low_word_coverage"] = (
            f"{coverage:.2f} word coverage, below the "
            f"{RELIABLE_WORD_COVERAGE:.2f} that separated a solved reading "
            "from a mis-keyed one when this attack was measured"
        )

    found = HomophonicResult([
        Candidate(
            method=METHOD,
            key=f"{symbols} symbols onto 26 letters, {slot_model} slots",
            score=engine.score(plaintext),
            plaintext=plaintext,
            diagnostics=diagnostics,
        )
    ])
    if top is not None and top > 0:
        return HomophonicResult(found.top(top), notes=found.notes)
    return found


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _units_from(
    source: str | NormalizedText | Sequence[str],
    unit_size: object,
) -> tuple[list[str], int, str]:
    """Turn *source* into a unit stream, or explain why it cannot be one.

    A sequence of tokens is taken verbatim. Text is reduced to its symbol
    stream and then cut into units of one or two symbols, and the choice
    between those two is the recogniser's, not a guess.

    An explicit *unit_size* overrides the recogniser entirely, including its
    refusal on a broken alternation. That is deliberate: somebody who has read
    the break index and decided to pair anyway has made a decision the toolkit
    should not make for them, and should not be blocked from making.
    """
    if isinstance(source, NormalizedText):
        # A hand-built NormalizedText has no symbol view -- that is the NOT
        # MEASURED case -- so fall back to the letters it does have.
        stream = source.symbols or source.letters
    elif isinstance(source, str):
        stream = normalize(source).symbols
    else:
        tokens = [str(token) for token in source]
        return tokens, _token_size(tokens), ""

    size = int(unit_size) if unit_size else 0
    if not size:
        report = paired.recognise(stream)
        if report.detected and not report.levels[0].clean:
            level = report.levels[0]
            return [], 0, (
                "the symbols alternate between two alphabets but alternation "
                f"breaks at symbol {level.repairable_at}, so every pair past "
                "that point is a fiction; repair the transcription first -- "
                "this will not invent a symbol to make the count work"
            )
        size = 2 if report.detected else 1

    if size == 1:
        return list(stream), 1, ""
    return [
        stream[index:index + size]
        for index in range(0, len(stream) - size + 1, size)
    ], size, ""


def _token_size(units: Sequence[str]) -> int:
    """Symbols per unit when every unit is the same length, else 0.

    Zero means "as supplied": the caller tokenised the stream themselves and
    the diagnostics should not claim a size that was never chosen.
    """
    sizes = {len(unit) for unit in units}
    return sizes.pop() if len(sizes) == 1 else 0


def _starting_key(
    counts: Sequence[int],
    slots: Mapping[str, int],
    generator: random.Random,
    *,
    frequency_seeded: bool,
) -> list[int]:
    """One starting assignment that realises the slot multiset exactly.

    Half the restarts start from the symbol frequencies -- the commonest
    symbol gets the slot with the highest expected share, and so on -- and
    half start from a shuffle, because a seeded start that is wrong in a
    structured way is harder to climb out of than one that is wrong at random.
    """
    letters: list[int] = []
    for letter in ALPHABET:
        letters.extend([ALPHABET.index(letter)] * slots[letter])

    if not frequency_seeded:
        generator.shuffle(letters)
        return letters

    # Expected share of ONE slot, not of the letter: E with two symbols is
    # about as common per symbol as A with one.
    def share(letter_index: int) -> float:
        letter = ALPHABET[letter_index]
        return ENGLISH_LETTER_PROBABILITY[letter] / slots[letter]

    letters.sort(key=share, reverse=True)
    # Jitter, so the six restarts are not six copies of one deterministic
    # guess: the seeded ranking is a good start, not a known answer.
    order = sorted(
        range(len(counts)),
        key=lambda symbol: -counts[symbol] * generator.uniform(0.75, 1.25),
    )
    key = [0] * len(counts)
    for symbol, letter_index in zip(order, letters):
        key[symbol] = letter_index
    return key


class _Annealer:
    """One unit stream, prepared once, annealed many times.

    Incremental scoring, re-derived over symbols from the trick documented in
    ``substitution._HillClimber``: swapping the letters on symbols X and Y
    changes the decryption only where an X or a Y stands, and a position i can
    only affect the four-letter windows starting in ``[i - 3, i]``. So a move
    costs the size of the union of the two symbols' window sets, not a rescore
    of the whole message.

    The running total is maintained by addition of deltas, which is the
    classic place for a silent bug. :meth:`full_score` recomputes it from
    scratch, ``solve`` reports the largest disagreement it saw as
    ``score_audit_gap`` on every real run, and a test asserts that gap is
    zero.
    """

    def __init__(
        self,
        values: Sequence[int],
        table: Sequence[float],
        symbol_count: int,
    ) -> None:
        self.values = list(values)
        self.table = table
        self.symbol_count = symbol_count
        self.count = len(self.values)
        self.last_window = self.count - _WINDOW
        self.searchable = self.last_window >= 0

        self.counts = [0] * symbol_count
        positions: list[list[int]] = [[] for _ in range(symbol_count)]
        for index, value in enumerate(self.values):
            positions[value].append(index)
            self.counts[value] += 1
        self.positions_of = positions

        touched: list[set[int]] = [set() for _ in range(symbol_count)]
        if self.searchable:
            for index, value in enumerate(self.values):
                low = max(0, index - (_WINDOW - 1))
                high = min(index, self.last_window)
                touched[value].update(range(low, high + 1))
        self.touched = [frozenset(entry) for entry in touched]

    def full_score(self, letter_of: Sequence[int]) -> float:
        """Independent recomputation of the objective, used to audit a run."""
        table = self.table
        decoded = [letter_of[value] for value in self.values]
        total = 0.0
        for start in range(self.last_window + 1):
            total += table[
                ((decoded[start] * 26 + decoded[start + 1]) * 26
                 + decoded[start + 2]) * 26 + decoded[start + 3]
            ]
        return total

    def run(
        self,
        letter_of: Sequence[int],
        iterations: int,
        generator: random.Random,
        *,
        deadline: float | None = None,
        constrain_slots: bool = True,
    ) -> tuple[list[int], float, bool]:
        """Anneal from *letter_of*. Returns the best key, its score, and
        whether the clock stopped it rather than the iteration count."""
        table = self.table
        positions_of = self.positions_of
        touched = self.touched
        current = list(letter_of)
        decoded = [current[value] for value in self.values]

        window_count = self.last_window + 1
        window_score_of = [0.0] * window_count
        for start in range(window_count):
            window_score_of[start] = table[
                ((decoded[start] * 26 + decoded[start + 1]) * 26
                 + decoded[start + 2]) * 26 + decoded[start + 3]
            ]
        total = sum(window_score_of)
        best_total, best_key = total, list(current)

        symbol_count = self.symbol_count
        cooling = (_TEMPERATURE_END / _TEMPERATURE_START) ** (
            1.0 / max(iterations, 1)
        )
        temperature = _TEMPERATURE_START
        cut_short = False

        for step in range(iterations):
            if (
                deadline is not None
                and not step % _CLOCK_EVERY
                and time.monotonic() >= deadline
            ):
                cut_short = True
                break
            temperature *= cooling

            first = generator.randrange(symbol_count)
            if constrain_slots:
                second = generator.randrange(symbol_count)
                if current[first] == current[second]:
                    continue
                windows = touched[first] | touched[second]
                letter_first, letter_second = current[first], current[second]
            else:
                # The mutation: any symbol may take any letter, so the slot
                # multiset is no longer preserved. See the module docstring.
                second = first
                letter_first = current[first]
                letter_second = generator.randrange(ALPHABET_SIZE)
                if letter_first == letter_second:
                    continue
                windows = touched[first]
            if not windows:
                continue

            before = sum(map(window_score_of.__getitem__, windows))
            for index in positions_of[first]:
                decoded[index] = letter_second
            if second != first:
                for index in positions_of[second]:
                    decoded[index] = letter_first

            after = 0.0
            for start in windows:
                after += table[
                    ((decoded[start] * 26 + decoded[start + 1]) * 26
                     + decoded[start + 2]) * 26 + decoded[start + 3]
                ]

            delta = after - before
            if delta > 0 or generator.random() < math.exp(delta / temperature):
                current[first] = letter_second
                if second != first:
                    current[second] = letter_first
                for start in windows:
                    window_score_of[start] = table[
                        ((decoded[start] * 26 + decoded[start + 1]) * 26
                         + decoded[start + 2]) * 26 + decoded[start + 3]
                    ]
                total += delta
                if total > best_total:
                    best_total, best_key = total, list(current)
            else:
                for index in positions_of[first]:
                    decoded[index] = letter_first
                if second != first:
                    for index in positions_of[second]:
                        decoded[index] = letter_second

        return best_key, best_total, cut_short
