# TDD -- Symbol-stream honesty: inventory, paired-symbol recogniser, homophonic family

**Status:** draft
**Date:** 2026-08-18 | **PRD:** `docs/PRD_symbol_stream_honesty.md` | **Repo:** CipherTool

## Approach

Four pieces, each independently revertible, in dependency order.

1. **`normalize.py` learns to say what it dropped.** `NormalizedText` gains an
   `Inventory` (letters / digits / other / spaces), an ordered alphanumeric
   `symbols` view and its position map. Every existing field and method is
   untouched, so nothing downstream changes until something asks.
2. **`paired.py`, a recogniser, not a solver.** Given a symbol stream it decides
   whether two disjoint symbol classes strictly alternate, reports the class
   sizes, the implied cell count, where the alternation breaks, and a plain
   description; it names a 13-rank x 4-suit inventory as a playing-card deck.
   It produces no plaintext and therefore is not an auto stage; `auto_solve`
   attaches its report to `AutoResult` and the CLI prints it.
3. **`homophonic.py`, a new cipher family.** A symbol stream with more than 26
   distinct symbols mapped onto 26 letters, attacked by simulated annealing over
   a FIXED letter-slot multiset with swap-only moves. Module, CLI subcommand,
   auto stage, tests, docs.
4. **`cli.py` routing.** The paste screen always prints the true inventory;
   when non-letters are material it takes the symbol-capable path; when nothing
   solves it, it refuses with a reason naming what was recognised. Letters-only
   candidates over a digit-bearing stream are capped at `weak` inside
   `auto_solve`, using the existing `confidence_cap` mechanism, so the cap holds
   for the `auto` command and the library too -- not only for one screen.

The load-bearing decision: **the letters-only pipeline is not made cleverer, it
is made unable to claim.** The cap is applied at the stage that read the letters
view, and the paste screen stops escalating when the stream is material, so a
card cipher can no longer spend three minutes climbing to a confident wrong
answer.

## Data model

No database, no migrations, no persisted state -- this is a local CLI over text.
The equivalent risk here is the in-memory contract every module shares:
`NormalizedText`. Changes to it are additive, with defaults, appended last, so
that every existing construction (positional or keyword) keeps working.

`src/cipher_tool/normalize.py`

```python
DIGITS = "0123456789"

@dataclass(frozen=True)
class Inventory:
    letters: int = 0     # A-Z after accent folding
    digits: int = 0      # 0-9 (ASCII, after NFKD folding)
    other: int = 0       # non-space, non-alphanumeric: punctuation, symbols
    spaces: int = 0      # whitespace; layout, never content

    @property
    def symbols(self) -> int: ...        # letters + digits
    @property
    def total(self) -> int: ...          # letters + digits + other + spaces
    @property
    def digit_fraction(self) -> float:   # digits / symbols, 0.0 when symbols == 0
    def describe(self) -> str:
        "1251 symbols: 891 letters and 360 digits"   # omits a class that is 0
```

`NormalizedText` gains, appended after `groups`:

| field | type | default | null/legacy case |
|---|---|---|---|
| `symbols` | `str` | `""` | a `NormalizedText` built by hand (tests, and `stacked.py` style call sites) has no symbols view; every consumer must treat `""` as "not measured", never as "no symbols" |
| `symbol_positions` | `tuple[int, ...]` | `()` | same |
| `inventory` | `Inventory` | `Inventory()` | all-zero inventory means NOT MEASURED. `_non_letters_are_material()` returns False on it, so the legacy case fails closed onto today's behaviour |

**This is the case that reaches production.** Anything constructing a
`NormalizedText` without going through `normalize()` gets zeroes, and every
predicate over the inventory must be written so that zeroes mean "behave exactly
as the tool does today". Grep for `NormalizedText(` before writing the
predicates.

New derived members on `NormalizedText`:

```python
@property
def has_symbols(self) -> bool          # bool(self.symbols)
@property
def digit_fraction(self) -> float      # self.inventory.digit_fraction
def describe_input(self) -> str        # self.inventory.describe()
```

Invariants, asserted by tests:

* `len(symbols) == inventory.letters + inventory.digits`
* `"".join(c for c in symbols if not c.isdigit()) == letters`
* `len(symbol_positions) == len(symbols)`, and every index points at the
  character of `original` that produced it
* `original`, `letters`, `positions`, `groups` are byte-identical to the values
  produced before this change, for every input in the golden corpus

## Interfaces

### `normalize.py`

```python
def symbols_only(text: str) -> str
    """Uppercase A-Z0-9 of *text*, in order, nothing else."""
def inventory_of(text: str) -> Inventory
```

`normalize()` computes the symbols view in the SAME single pass that already
walks the original character by character, folding one character at a time. It
must not call `fold_to_ascii` twice over the whole string; the position map
depends on that walk being exact.

### `paired.py` (new)

```python
MINIMUM_SYMBOLS = 40          # below this, alternation is chance
RANKS = frozenset("23456789TXJQKA")
SUITS = frozenset("CDHS")

@dataclass(frozen=True)
class Alternation:
    """One level of strict alternation over a token stream."""
    first_class: tuple[str, ...]      # distinct tokens at even positions
    second_class: tuple[str, ...]     # distinct tokens at odd positions
    tokens: int                       # length of the stream at this level
    units: int                        # tokens // 2, the implied cell count
    distinct_units: int               # distinct cells actually observed
    cells_available: int              # len(first_class) * len(second_class)
    breaks: tuple[int, ...]           # indices where alternation fails
    clean: bool                       # no breaks at all
    repairable_at: int | None         # single index where one inserted or
                                      # deleted token would restore alternation

@dataclass(frozen=True)
class PairedReport:
    detected: bool
    reason: str                       # why not, when detected is False
    levels: tuple[Alternation, ...]   # level 0 over symbols, level 1 over cells
    inventory_name: str | None        # "playing-card deck", or None
    shuffle_control: float | None     # fraction of shuffles that alternate
    description: str                  # plain English; always safe to print

def recognise(stream: str | Sequence[str], *, max_levels: int = 2,
              minimum: int = MINIMUM_SYMBOLS, control_trials: int = 200,
              seed: int = 0) -> PairedReport
def cells(stream: Sequence[str]) -> list[str]     # pair up; drops a lone tail
def describe(report: PairedReport) -> str          # the printable block
```

Detection rule at one level, over tokens `T`:

1. `len(T) >= minimum`, else `detected=False`, reason names the length.
2. `A = {T[i] for even i}`, `B = {T[i] for odd i}`. Detected iff `A & B` is
   empty **and** `len(A) >= 2` **and** `len(B) >= 2`.
3. `len(A) == 1` or `len(B) == 1` is a SEPARATOR, not a paired alphabet:
   `detected=False`, reason says so. (`A1B1C1D1` must not be sold as structure.)
4. If `A & B` is non-empty: compute the longest clean prefix `P` and longest
   clean suffix `S` under their own parities. If `len(P) + len(S) >= len(T) - 1`
   the fault is a single slip: `detected=True`, `clean=False`,
   `breaks=(len(P),)`, `repairable_at=len(P)`, and the description says one
   token appears to be missing or extra at that index. Otherwise
   `detected=False` and the reason gives the number of shared symbols.
   **The stream is never edited.**
5. Naming: if one class is exactly `SUITS` and the other has >= 10 distinct
   tokens all within `RANKS`, `inventory_name = "playing-card deck"` and the
   description says "13 ranks x 4 suits = a 52-card deck; each pair of symbols
   is one card". Otherwise the description says "a {a} x {b} paired alphabet
   ({a*b} cells)" and names nothing.
6. Level 1 applies the same test to `cells(T)`. When it detects, the description
   adds: the cells themselves alternate between two disjoint sets of {n1} and
   {n2}, so the unit is TWO cells (four symbols) and each unit carries more than
   one letter -- a digraph or a codebook entry; the toolkit cannot decide which
   from structure alone.

   > **NARROWED IN IMPLEMENTATION, 2026-08-18.** Level 1 is only looked for
   > when level 0 is `clean`. Past a slip the pairing is off by one, so every
   > cell after it is one the module invented, and level 1 is a claim about
   > the cells. MEASURED: on a two-level stream with a single symbol deleted
   > it still "detected", over a cell stream that was half fabricated. The
   > same reasoning takes `distinct_units` down to the clean prefix -- counted
   > over the whole of a real 52-card stream with one symbol missing it came
   > to 95 distinct cards out of a possible 52, and the report said so.
7. `shuffle_control`: shuffle the same multiset `control_trials` times with a
   seeded `random.Random(seed)` and report the fraction that alternate cleanly.
   Expected 0.0. This is what stops "detected" from being an artefact of the
   inventory, and it is cheap: O(trials * n).

Cost: O(n) per level plus O(trials * n) for the control. No quadratic scan --
the prefix/suffix construction gives the single-slip answer in one pass.

### `homophonic.py` (new)

```python
METHOD = "Homophonic substitution (annealing over a fixed letter-slot multiset)"
DEFAULT_RESTARTS = 6
DEFAULT_ITERATIONS = 60_000
MIN_UNITS = 200                 # below this the search has more freedom than text
MIN_UNITS_PER_SYMBOL = 6.0      # refuse below; measured successes were 7.7 and 8.7
CONFIDENT_UNITS_PER_SYMBOL = 8.0  # below this, cap the label at `promising`
MAX_SYMBOLS = 200               # above this it is not a homophonic key

def slot_multiset(symbol_count: int, *, model: str = "uniform") -> dict[str, int]
    """Letter -> how many symbols stand for it. Sums to *symbol_count*.

    "uniform"   -- symbol_count // 26 each, remainder to the commonest letters.
                   For 52 that is exactly two per letter.
    "frequency" -- proportional to English letter frequency, minimum one each.
    """
def random_key(symbol_count: int, *, model: str = "uniform",
               seed: int | None = None) -> dict[str, tuple[str, ...]]
def encrypt(plaintext: str, key: Mapping[str, Sequence[str]], *,
            seed: int | None = None) -> list[str]
def decrypt(units: Sequence[str], mapping: Mapping[str, str]) -> str
def solve(source: str | NormalizedText | Sequence[str], *,
          scorer: EnglishScorer | None = None, top: int = 5,
          **options: object) -> CandidateSet
```

`solve` options: `units` (explicit token list), `unit_size` (1 or 2 symbols per
token; default: 2 when `paired.recognise` reports a clean level-0 alternation,
else 1), `restarts`, `iterations`, `seed`, `time_budget`, `slots`
(`"uniform"` | `"frequency"`), `constrain_slots` (default True).

Contract, and the traps it is written against:

* Accepts `time_budget` and NEVER raises on it. `time_budget <= 0` returns an
  empty `CandidateSet`, exactly as `polybius.solve_unknown_square` does. A stage
  that raised `ValueError` on an unexpected `time_budget` was silently dropped
  from every real run, because `auto_solve` retries only on `TypeError`.
* Refuses cheaply, before any work: fewer than 27 distinct symbols (that is
  `substitution.py`'s job and it does it better), more than `MAX_SYMBOLS`,
  fewer than `MIN_UNITS` units, or `units < MIN_UNITS_PER_SYMBOL * symbols`.
  Every refusal returns an empty `CandidateSet` and attaches a `notes` tuple
  explaining, the way `hill.py` does; `auto_solve` already surfaces `.notes`.
* Sets `diagnostics["confidence_cap"] = "promising"` when
  `units < CONFIDENT_UNITS_PER_SYMBOL * symbols`. The search has more key
  freedom than the ciphertext can pay for and must not be able to say `strong`.
* Diagnostics always carry: `symbols`, `units`, `units_per_symbol`, `restarts`,
  `iterations`, `slot_model`, `unit_size`, and
  `"search": "annealing, not exhaustive"`.
* Calls `scoring.annotate()` so `Candidate.confidence()` has both signals.

The search, and why each part is there:

* State: `letter_of[symbol_index]`, an assignment that realises the slot
  multiset exactly.
* Move: pick two symbol indices whose letters differ and **swap** them. Swapping
  preserves the multiset by construction. This is the whole algorithm.
  MEASURED: with the constraint, a 52-symbol two-per-letter cipher is recovered
  at 100 per cent exact letters from 400 units, and a 36-symbol one at 100 per
  cent from 313. UNCONSTRAINED -- letting any symbol take any letter -- the
  search collapses onto a degenerate five-letter text (IS/IT/AT/AN/AS), because
  a five-letter language scores better per letter than English under a quadgram
  model when nothing forbids it.

  > **CORRECTED BY MEASUREMENT, 2026-08-18 (coder).** The recoveries are 0.983
  > and 0.901 (twelve restarts; six gives 0.856), not 1.00. And the collapse
  > does **not** happen on genuine homophonic ciphertext: at 52/400, 52/600,
  > 36/313, 40/320 and 52/340 the unconstrained search matched or beat the
  > constrained one, reaching 1.000. It collapses on a stream that is NOT
  > homophonic, which is the case that matters. On 600 units of uniformly
  > random cards it produced eight distinct letters at -1.248 per letter and
  > `promising` -- `SEEITSSESINILETSTATSSITSSITSSSTSEETSISASSITSLESS...` --
  > against the constrained search's 26 letters at -2.179 and `weak`. The
  > design's conclusion stands; the regime it was attributed to does not, and
  > the test asserts the measured one. See ALGORITHMS.md.
* `constrain_slots=False` exists solely so a test can observe that collapse.
  It is a real mutation: it changes the move set, not a constant.
* Objective: the same order-3 quadgram window score the substitution solver
  optimises, via the PUBLIC scoring API (`engine.encode`, `engine.table()`).
  No import of `substitution.py` privates.
* Incremental scoring: a swap changes the decryption only at positions holding
  those two symbols, so only the windows those positions touch are rescored --
  the trick documented in `substitution._HillClimber`, re-derived over symbols.
  A `full_score()` recomputation audits the running total, and a test asserts
  the two agree; a delta-maintained total is the classic place for a silent bug.
* Annealing: geometric cooling from `T0` to `T_end` over `iterations`, accept
  worse moves with probability `exp(delta / T)`, `random.Random(seed)` for
  reproducibility, `restarts` independent runs, best by recomputed full score.
* Clock: check `time.monotonic()` once every 256 moves, not every move.
  `diagnostics["time_budget_hit"]` when it stops early -- `auto.py` already
  prints "search cut short by the clock" from that flag.

### `auto.py`

Additive only:

```python
@dataclass(frozen=True)
class Stage:
    ...
    reads: str = "letters"      # or "symbols"
```

`reads` is what makes the confidence cap correct. Stages that read the symbol
stream (`encodings`, `Polybius (unknown square)`, `homophonic`) are marked
`reads="symbols"`; everything else keeps the default and is therefore capped
when the stream it did not see was material.

```python
@dataclass
class AutoResult:
    ...
    structure: paired.PairedReport | None = None
```

In `auto_solve`, after the stage loop and before `_add_reversed_readings`:

```python
if normalized.inventory.symbols and _non_letters_are_material(normalized):
    for candidate in result.candidates.ranked():
        if candidate.diagnostics.get("reads") != "symbols":
            candidate.diagnostics["confidence_cap"] = "weak"
            candidate.diagnostics["discarded_symbols"] = (
                f"{normalized.inventory.digits} of "
                f"{normalized.inventory.symbols} symbols in this message are "
                "digits and were not part of this reading")
```

Each stage stamps `diagnostics["reads"] = stage.reads` on the candidates it
produced (done once in the loop, not in every solver).

New stage, added to `build_stages`:

```python
Stage("homophonic", "homophonic", "fast", 8.0, homophonic.solve,
      {"top": top, "restarts": (2, 6, 12)[scale], "seed": seed},
      reads="symbols"),
```

`"fast"` because it refuses in microseconds on anything with 26 or fewer
distinct symbols -- the same argument `auto.py` already makes for ADFGVX and for
the Polybius square search -- and because the paste screen must be able to
reach it without escalating twice.

`auto_solve` also sets `result.structure = paired.recognise(normalized.symbols)`
when `len(normalized.symbols) >= paired.MINIMUM_SYMBOLS`, and `AutoResult.render`
prints the description when `structure is not None and structure.detected`.

### `cli.py`

```python
MATERIAL_DIGIT_FRACTION = 0.05
MATERIAL_DIGIT_COUNT = 10

def _non_letters_are_material(normalized: NormalizedText) -> bool
    """True when the letters-only view is not a fair reading of the paste.

    False for a zeroed (hand-built) inventory, so the legacy case behaves
    exactly as the tool does today. MEASURED 2026-08-18: all 40 official
    archive ciphertexts have digit fraction 0.0000, so no positive threshold
    can move them.
    """

def _solve_symbol_stream(raw, args, *, structure) -> AutoResult | None
    """Was _solve_letterless. Now also runs the homophonic family, at
    unit_size 2 when the recogniser saw clean alternation, and keeps its
    existing encodings + Polybius-unknown-square behaviour byte for byte."""

def _render_symbol_refusal(normalized, structure) -> str
    """The refusal, with the reason on it."""
```

`_paste_message` becomes:

1. `normalized = normalize(text)`
2. print `f"\n  Read {normalized.describe_input()}. Working..."` -- always, for
   every paste, including the letters-only case, where it reads
   `Read 891 symbols: 891 letters.`
3. `structure = paired.recognise(normalized.symbols)` when long enough
4. `if normalized.is_empty:` -- unchanged path, unchanged wording, now calling
   `_solve_symbol_stream`
5. `elif _non_letters_are_material(normalized):` -- try `_solve_symbol_stream`;
   on success print `_render_answer`; on failure print
   `_render_symbol_refusal` and enter the answer menu. **No fall-through to the
   letters-only pipeline and no automatic effort ladder on this path**, because
   escalating is how the defect spent three minutes arriving at a confident
   wrong answer.
6. else -- today's path, unchanged, including the ladder.

The `[Enter] try harder` branch at `cli.py:1781` gets the same treatment: on the
material path it re-runs the symbol path at the next effort, never the
letters-only pipeline.

The refusal text names what was recognised and where to go next, for example:

```
======================================================================
THIS IS A PAIRED-SYMBOL CIPHER, AND IT IS NOT SOLVED
======================================================================

  Read 1251 symbols: 891 letters and 360 digits.

  The symbols alternate strictly between two disjoint sets: 13 ranks
  (2-9, X, J, Q, K, A) and 4 suits (C, D, H, S). That is a 52-card
  playing-card deck, and every PAIR of symbols is one card -- 625
  cards here and one symbol left over, not 1251 letters.
  Alternation breaks once, at symbol 865; one symbol looks missing or
  extra there, which is also why the count is odd. Nothing was changed.
  The cards themselves alternate between two disjoint sets, so the
  unit is probably TWO cards and carries more than one letter.

  The toolkit could not recover the key. It is not offering a reading
  of the letters alone: that would discard 360 of 1251 symbols.

  Next:  cipher_tool homophonic <file>   more symbols than letters
         cipher_tool analyse <file>      the full measurement
         [l] show the letters-only reading anyway (NOT an answer)
```

## Access control

Not applicable, stated rather than scored: no server, no database, no auth, no
RLS, no security-definer function, no grant to any role. The toolkit is a local
CLI that reads a file and prints to a terminal. `tests/test_compliance.py`
already asserts it cannot reach a network, run a process, or load code at
runtime; the two new modules must pass it unchanged (standard library only, pure
ASCII, module docstring present).

The one real trust boundary is **untrusted input length**. Both new modules are
linear in the stream, the shuffle control is `O(trials * n)` with `trials`
fixed, and the homophonic search is bounded by iterations, restarts and the time
budget. No new path is quadratic.

## Migrations

None -- no database. The analogue is the `NormalizedText` contract, and the
additive-first rule applies to it verbatim:

| # | Does | Reversible? | Rollback |
|---|---|---|---|
| 1 | Add `symbols`, `symbol_positions`, `inventory` with defaults; nothing reads them | Yes | revert the commit |
| 2 | Add `paired.py`; nothing calls it | Yes | delete the module |
| 3 | Add `homophonic.py` + CLI subcommand; not yet in the stage list | Yes | revert |
| 4 | Add `Stage.reads`, the homophonic stage, `AutoResult.structure`, the cap | Yes | revert |
| 5 | Switch the paste screen's routing and wording onto them | Yes | revert |

Nothing is deleted, nothing is renamed, no field changes meaning. Step 5 is the
only one a user can see, and it is the last one.

## Failure modes

| What breaks | Who notices | How we detect it | How we undo it |
|---|---|---|---|
| A letters-only ciphertext behaves differently (ordering, confidence, wording) | The operator, mid-competition | `run_official.py official3 answers3` with NO third argument: 40 ciphertexts, 22 solved, 0 wrong. Plus the full unit suite | Revert step 5, then step 4 |
| Materiality threshold too low: a real message with a date or page numbers is refused | The operator, as a message that used to solve | A regression test with 900 letters and 8 digits must still take the letters path | Raise `MATERIAL_DIGIT_FRACTION`; one constant, no structural change |
| Materiality threshold too high: a digit-bearing symbol cipher still gets the letters pipeline | Nobody -- this is the silent failure that started this work | The reproduction script is a permanent test, and the cap in `auto_solve` is a second, independent guard | Lower the constant |
| The recogniser claims structure that is not there | The operator, as a confident wrong description | `shuffle_control` in the report, plus negative tests: English, ADFGVX, a separator stream, a shuffled card stream | Revert step 2; the report is descriptive, so nothing else depends on it |
| The homophonic stage burns the clock on messages it cannot help | The operator, as a slow paste screen | Stage timing appears in the auto report; the refusal gates run before any work | Move the stage to `minimum_effort="deep"`; one word |
| The homophonic search returns fluent nonsense at `strong` | The operator, and then the competition | `confidence_cap` below the measured units-per-symbol regime; a test asserts a short stream cannot reach `strong` | Raise `MIN_UNITS_PER_SYMBOL`; the refusal is the default |
| A new stage refuses `time_budget` and is silently dropped | Nobody, which is why it is on this list | `tests/test_auto.py::TestEveryStageToleratesATimeBudget` hands every stage a clock | Fix the signature; the test is the gate |
| `NormalizedText` built by hand somewhere gets a zero inventory and is treated as "no digits" | Nobody -- it fails onto today's behaviour | Explicit test constructing `NormalizedText` positionally and running the paste path | None needed; this is the designed null case |

## Rollback

Five commits, in the order above, each independently revertible with
`git revert`, in the reverse order. No data is written, no schema exists, no
state persists between runs, so a revert is complete the moment it lands and
takes as long as `git revert` plus a test run. The only user-visible surface is
the paste screen's wording and routing (step 5), so reverting step 5 alone
restores today's behaviour exactly while leaving the new modules dormant and
harmless.

Nothing here is irreversible.

## Test plan

Every test below must be **observed failing before the code that makes it pass**.
Baseline first (`python run_tests.py` green on the current tree, and
`scratchpad/failing_test.py` printing three FAILs -- observed 2026-08-18 15:48),
then each new test red for the right reason, then green.

### Positive

* `tests/test_normalize.py`
  * the operator message: `inventory.letters == 891`, `inventory.digits == 360`,
    `inventory.symbols == 1251`, `describe_input()` contains "1251 symbols",
    "891 letters", "360 digits".
  * `symbols` is the alphanumeric stream in order; `symbol_positions` indexes
    `original` exactly; the letters filtered out of `symbols` equal `letters`.
* `tests/test_paired.py`
  * a synthetic clean rank/suit stream of 600 symbols is detected, class sizes
    13 and 4, cells 300, `inventory_name == "playing-card deck"`,
    `clean is True`, `shuffle_control == 0.0`.
  * the real operator stream is detected with exactly one break, at index 865,
    `repairable_at == 865`, and the description says a symbol looks missing or
    extra there.
  * level 1 fires on the operator stream: the cells alternate between two
    disjoint sets (sizes 36 and 16).
* `tests/test_homophonic.py`
  * 52 symbols, two per letter, 400 units, fixed seed: >= 90 per cent exact
    letters (measured 100).
  * 36 symbols, 313 units, fixed seed: >= 90 per cent exact letters.
  * `full_score()` equals the incrementally maintained total after a climb.
  * `encrypt`/`decrypt` round-trip under a known key.
* `tests/test_cli.py`
  * end to end through the real screen:
    `python -m cipher_tool paste < operator_cipher.txt` (PYTHONPATH=src) prints
    "1251 symbols", "891 letters", "360 digits", names the deck, and does not
    print a monoalphabetic reading as the answer.
  * a letters-only Caesar still prints `BEST ANSWER` and the plaintext -- the
    existing `TestPasteSession` suite, unchanged and passing.
  * a numeric Polybius message is still solved (`TestPasteSolvesNonLetterMessages`).
* Archive: `run_official.py official3 answers3`, **no third argument**, still
  40 ciphertexts / 22 solved / 0 wrong. A harness stingier than the tool
  under-reports the tool; do not add a global budget.

### Negative

* `tests/test_paired.py`
  * 600 letters of English: `detected is False`, reason names the shared
    alphabet.
  * an ADFGVX ciphertext (6 symbols, no alternation): `detected is False`.
  * `"A1B1C1D1..."`: `detected is False` and the reason says separator, not
    paired alphabet.
  * the operator stream SHUFFLED with a fixed seed: `detected is False`. Same
    symbols, same counts, no structure -- this is the test that the recogniser
    is measuring order and not inventory.
* `tests/test_homophonic.py`
  * **the mutation that matters:** `constrain_slots=False` on the 52-symbol
    cipher collapses -- fewer than 9 distinct letters in the output and exact
    accuracy below 0.30 -- while the constrained run on the same seed exceeds
    0.90. Observe this failing by running the constrained assertion against the
    unconstrained search.
  * 26 or fewer distinct symbols: empty `CandidateSet`, with a note.
  * `units < 6 * symbols`: empty `CandidateSet`, with a note.
  * a short stream cannot produce a `strong` candidate (the
    `confidence_cap` path).
* `tests/test_cli.py`
  * the operator message never yields a candidate whose confidence is above
    `weak` at **fast**, at **normal** and at **deep**. A guard verified at one
    effort level says nothing about the others.
  * `[l]` prints the letters-only reading with the discard warning attached and
    the words "not an answer".
* `tests/test_auto.py`
  * a digit-material stream: every candidate from a `reads="letters"` stage
    carries `confidence_cap == "weak"` and a `discarded_symbols` diagnostic; a
    candidate from a `reads="symbols"` stage does not.

### Boundary

* Empty paste -- still "No letters were pasted" (`test_a_genuinely_empty_paste_still_says_so`).
* Digits only -- still routed as today, still explained
  (`TestNumericCiphertextIsNotCalledNothing`).
* Exactly at the materiality threshold: 200 symbols with 10 digits (5.0 per
  cent) is material; 200 symbols with 9 digits, and 1,000 symbols with 40
  digits (4 per cent), are not.
* 900 letters with 8 digits -- an ordinary message carrying a date -- takes the
  letters path and is solved exactly as today.
* A hand-built `NormalizedText` with a zeroed inventory: `_non_letters_are_material`
  is False and the paste path is today's path.
* A stream of exactly `MINIMUM_SYMBOLS - 1` symbols: recogniser refuses on
  length; at `MINIMUM_SYMBOLS` it may detect.
* An odd-length alternating stream: the lone tail symbol is reported, never
  paired with nothing.
* `time_budget=0` and `time_budget=-1` to `homophonic.solve`: empty set, no
  raise. `time_budget=2.0`: no raise (and the whole-plan clock test passes).
* One symbol repeated 1,000 times: not structure, no crash, no quadratic scan.
* `tests/test_compliance.py` unchanged and passing over the two new modules
  (pure ASCII, stdlib only, documented).

## Build order

1. **Baseline.** `python run_tests.py` (record the count) and
   `python scratchpad/failing_test.py` -- observe three FAILs. Also record
   `run_official.py official3 answers3` if the scoreboard is to be re-quoted.
2. **normalize.py.** Write the inventory tests, watch them fail, add
   `Inventory`, `symbols_only`, `inventory_of` and the three additive fields.
   Add the golden test that `original`/`letters`/`positions`/`groups` are
   unchanged for a corpus including digits, accents, BOM and punctuation.
   Claim 1 of the reproduction now passes.
3. **paired.py.** Tests first, including the shuffled-stream negative, then the
   module. No CLI wiring yet.
4. **homophonic.py.** Tests first: the two measured recoveries, the
   unconstrained-collapse mutation, the refusal gates, the `time_budget` cases.
   Then the module. Then the `homophonic` CLI subcommand and its `--slots`,
   `--unit`, `--restarts`, `--iterations` options.
5. **auto.py.** `Stage.reads`, the `reads` stamp on candidates, the homophonic
   stage, `AutoResult.structure` and its rendering, and the confidence cap.
   Re-run `TestEveryStageToleratesATimeBudget` and the whole suite.
6. **cli.py.** The inventory line, `_solve_symbol_stream`,
   `_render_symbol_refusal`, the routing at `_paste_message` and at the
   try-harder branch. Claims 2 and 3 of the reproduction now pass.
7. **Verify end to end through the real screen**, not the library:
   `PYTHONPATH=src python -m cipher_tool paste < operator_cipher.txt`, and a
   letters-only file the same way. Read the actual output.
8. **Regression.** Full suite; `run_official.py official3 answers3` with no
   third argument; confirm 22/40, 0 wrong.
9. **Docs.** README (the new family and the paste behaviour), ALGORITHMS.md (the
   recogniser and the annealing, with the measured numbers), CHANGELOG.md, and
   RULES_COMPLIANCE.md if it enumerates modules. Leave the tree uncommitted for
   review; do not push.

## Open questions

None that change this design. Three that change a constant:

1. Which slot multiset the measured 100 per cent runs used (uniform vs
   frequency). Fixes `DEFAULT_SLOT_MODEL` only; the design tries both.
2. `MATERIAL_DIGIT_FRACTION` = 0.05 and `MATERIAL_DIGIT_COUNT` = 10. The archive
   measures 0.0000 digits, so nothing there moves at any positive threshold.
3. Whether the refusal screen offers `[l]` at all. Removing it removes one
   screen state and no code path.

### Settled during implementation, 2026-08-18

1. **`DEFAULT_SLOT_MODEL = "uniform"`.** Measured both on the 36/313 shape:
   0.856 uniform and 0.856 frequency at six restarts, 0.901 and 0.901 at
   twelve. No difference, so the model that assumes least about the plaintext
   wins.
2. **Thresholds kept at 0.05 and 10.** Re-measured over the 40 files of
   `official3` with the shipped predicate: highest digit fraction 0.0000, at
   most 26 distinct alphanumeric symbols, and **0 files routed to the new
   path**.
3. **`[l]` exists**, as `_render_letters_only_reading`, printed only on the
   material path and always over the discard warning. The reading it shows is
   capped at `weak` by `auto_solve`, so the label under it agrees with the
   heading above it.

Two things moved that were not open questions:

* **`non_letters_are_material` lives in `auto.py`, not `cli.py`.** `auto_solve`
  needs it for the confidence cap and `cli` imports `auto`, not the reverse.
  `MATERIAL_DIGIT_FRACTION` and `MATERIAL_DIGIT_COUNT` moved with it.
* **The unit-size default keys off `report.detected`, not `report.clean`,** and
  a detected-but-broken alternation is a **refusal** rather than a fallback to
  unit size 1. Pairing past a break invents cells that were never written --
  on the real message, 96 distinct "cards" out of a 52-card deck at 6.5 units
  per symbol, which passes every count gate and would have been answered.
  Refusing there is what makes the operator's message come out as the PRD says
  it must.
