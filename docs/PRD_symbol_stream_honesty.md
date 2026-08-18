# PRD -- Symbol-stream honesty: stop discarding digits, read paired-symbol ciphers

**Status:** draft
**Date:** 2026-08-18 | **Repo:** CipherTool | **Related:** `docs/TDD_symbol_stream_honesty.md`, `docs/APP_FLOW_paste_symbol_stream.md`

## Problem

A ciphertext that mixes digits and letters has its digits silently discarded and
the wreckage is then solved as a monoalphabetic substitution and offered as an
answer. Reproduced end to end on a real pasted message: 1,251 alphanumeric
symbols, 891 letters and 360 digits. The paste screen printed

    Read 891 letters. Working...

escalated itself fast -> normal -> deep, and returned
`ASTETANTSILTANTITATIST...` (words found TIDES/DEALT/TIRES/TESTS/TASTE/START)
at confidence **promising**. Half the message was thrown away before the search
began and nothing on the screen said so. Observed again on the current tree at
2026-08-18 15:48 by `scratchpad/failing_test.py`: three assertions, three FAILs.

Root cause, located:

* `src/cipher_tool/normalize.py` -- `normalize()` and `letters_only()` keep A-Z
  and record nothing about what was dropped. `NormalizedText` has no way to say
  what it discarded.
* `src/cipher_tool/cli.py:1650` -- `if normalized.is_empty:` is the ONLY gate
  that routes to the symbol-capable solvers. One surviving letter is enough to
  send a symbol stream down the letters-only pipeline.
* `src/cipher_tool/cli.py:1668` -- prints a count of what SURVIVED as though it
  described what was pasted.

This is the third and worst variant of one recurring bug in this toolkit: a
numeric ciphertext once read as an empty paste; "no letters" and "nothing"
looked identical; now "letters plus digits" looks like "letters". The first two
refused. **This one answers.** A tool that refuses is annoying; a tool that
answers confidently from half a message teaches its user to trust it wrongly,
and a competition answer submitted from it is wrong in public.

Second, smaller problem behind the first: even routed correctly, the toolkit has
nowhere to send this message. It has no cipher family for a symbol stream with
more than 26 distinct symbols, and no way to describe paired-symbol structure.

## Who it is for

The operator: one person working National Cipher Challenge material offline,
alone, under time pressure, pasting a transcription into the paste screen and
reading what comes back. Nobody proof-reads the tool's output for them.

## Success looks like

- [ ] `scratchpad/failing_test.py` exits 0: all three assertions pass, having
      been observed failing first.
- [ ] The paste screen reports the true inventory of every paste:
      `Read 1251 symbols: 891 letters and 360 digits.`
- [ ] On that message the toolkit names the structure it found -- a paired
      symbol cipher over a 13-rank x 4-suit playing-card deck, 625 cards and
      one symbol left over, whose cards themselves pair into units -- and
      **refuses with that reason** rather than answering. (The measured 313
      units follow only after the single missing symbol is restored, which the
      toolkit reports and never does for itself.)
- [ ] No letters-only reading of a digit-bearing message is ever labelled above
      `weak`, at `fast`, at `normal` and at `deep`.
- [ ] A new homophonic-substitution family recovers a 52-symbol two-per-letter
      cipher from 400 units and a 36-symbol cipher from 313 units, at >= 90 per
      cent exact letters, from a fixed seed.
      *Met 2026-08-18: 0.983 and 0.901 (the second at twelve restarts).*
- [ ] A test observes the UNCONSTRAINED version of that search collapsing onto
      a five-letter degenerate text, because the slot constraint is the whole
      algorithm.
      *Met, but not where this expected it. On genuine homophonic ciphertext
      the unconstrained search matched or beat the constrained one at every
      shape tried. It collapses on a stream that is NOT homophonic -- 600
      units of random cards gave eight distinct letters at -1.248 per letter
      and `promising`, against 26 letters at -2.179 and `weak`. The test
      asserts the measured regime.*
- [ ] The official-archive scoreboard is unchanged: 40 ciphertexts, 22 solved,
      0 wrong.
- [ ] `tests/test_auto.py::TestEveryStageToleratesATimeBudget` still passes with
      the new stage in the plan.

## Requirements

**Must**

1. `NormalizedText` reports what it discarded: counts by class (letters,
   digits, other) and an ordered alphanumeric `symbols` view. Purely additive --
   `original`, `letters`, `positions` and `groups` stay byte-identical for every
   input.
2. A recogniser for paired-alphabet (fractionating) structure over a symbol
   stream: detect two disjoint symbol classes that strictly alternate, report
   class sizes, implied cell count and a plain-English description. Name a
   13x4 inventory of rank and suit letters as a playing-card deck explicitly.
3. The recogniser tolerates a single transcription slip by REPORTING where
   alternation breaks, and refuses to claim structure that is not there.
4. The paste screen always reports the true inventory; when non-letters are a
   material part of the stream it routes to the symbol-capable path instead of
   the letters-only pipeline; and when nothing solves it, it refuses with a
   reason that names what was recognised and what command to try next.
5. A new cipher family, homophonic substitution: a symbol stream where more
   than 26 distinct symbols map onto 26 letters. Module + CLI subcommand +
   auto stage + tests + docs.
6. The homophonic search uses a FIXED letter-slot multiset with swap-only
   moves. A test observes the unconstrained variant failing.
7. Every new auto stage accepts a time budget without raising, at every effort
   level.
8. No existing behaviour weakens: letters-only ciphertexts behave exactly as
   they do today, and the archive scoreboard does not regress.

**Should**

9. The structure report also appears in `analyse` and in the `auto` report, not
   only on the paste screen.
10. A second level of the recogniser: when the CELLS themselves alternate
    between two disjoint sets, say so, and say what it implies (the unit is two
    cells, so each unit carries more than one letter).
    *Built, and NARROWED during implementation: level 1 is only looked for
    when level 0 is clean. Past a slip the pairing is off by one and every
    cell after it is fabricated, and level 1 was measured "detecting" over
    exactly such a stream. So the operator's own message, which has a slip,
    gets no second-level claim -- the honest outcome, and not the one this
    expected.*
11. A control statistic in the report: how many random shuffles of the same
    symbol multiset alternate cleanly (expected 0, and it makes the claim
    falsifiable).

**Won't (this time)**

12. Solving the operator's card message. It is a 19 per cent transcription of a
    digraphic cipher; 313 units against a 52x52 digraph table cannot be
    recovered, and presenting a reading of it would be the original defect in a
    new coat. **Recognition plus honest refusal is the correct outcome.**
13. A digraph/word-code discriminator (the token-spectrum and mutual-information
    work that identified the unit). Measured outside the toolkit; not shipped.
14. Automatic repair of a broken transcription. The recogniser reports where
    alternation breaks; it never edits the message.
15. A statistics hypothesis that ranks the homophonic family. Ordering only; the
    stage runs regardless.

## Explicitly out of scope

* Any change to `normalize()`'s existing outputs, to the letters-only pipeline,
  or to any solver's search behaviour on letters-only input. This work must be
  invisible to every ciphertext that has no digits in it -- and every one of the
  40 archive ciphertexts has none (measured 2026-08-18: digit fraction 0.0000
  across all 40 files, maximum distinct alphanumeric symbols 26).
* New dependencies. Python standard library only, no network, pure ASCII source.
  These are competition rules, not preferences.
* Making the tool more confident anywhere. Every change here either lowers a
  confidence label or adds a refusal.
* Solving playing-card ciphers in general (Solitaire/Pontifex keystream, deck
  permutation ciphers). The recogniser names the inventory; it does not attack
  it.

## Safety and privacy

* **Personal data:** none. The toolkit is offline, has no accounts, no network,
  no database, and writes only `plaintext.txt` when the user presses `[f]`.
* **Who can see it:** whoever runs it locally. Nothing is transmitted.
* **Access revocation:** not applicable -- there is no auth surface. Stated
  plainly rather than scored as a pass.
* **Worst outcome if this is wrong:** the toolkit presents a reading of a
  message it has half-read, the operator submits it, and a public competition
  answer is wrong. That is exactly today's behaviour, which is why the change
  is worth making. The second-worst outcome is over-refusal: a solvable
  letters-plus-digits message is refused where it used to be solved. That is
  bounded by the materiality threshold and is recoverable by the user in one
  keystroke (the refusal screen offers the letters-only reading behind an
  explicit, labelled choice).
* **Resource safety:** the input is untrusted text of any length. Every new code
  path must be linear in the stream length, or explicitly bounded, so a large
  paste cannot hang the tool.

## Open questions

None that change the design. Three that change a constant, each with the
measurement that would settle it:

1. **Which slot multiset the measured 100 per cent runs used** -- uniform
   (S/26 per letter, remainder to the commonest) or frequency-proportional. The
   design tries uniform first and frequency second, so either answer works; the
   answer only fixes `DEFAULT_SLOT_MODEL`. Settle by re-running the measured
   52-symbol and 36-symbol recoveries under both.
2. **The materiality threshold** (proposed: digits >= 5 per cent of the
   alphanumeric stream AND >= 10 digits). The archive measures 0.0000, so any
   positive threshold is safe there; the question is only how tolerant to be of
   a date or a page number inside a real message.
3. **Whether the refusal screen should offer the letters-only reading at all**
   (proposed: yes, behind `[l]`, printed with a discard warning). Removing it
   removes a screen state and nothing else.

## Not doing / rejected alternatives

* **Silently stripping digits and warning afterwards.** Rejected: the warning is
  read after the answer, and people read answers.
* **Routing on `is_empty` alone, but with better wording.** Rejected: it is the
  gate itself that is wrong. "One letter present" must not mean "letters only".
* **Making the recogniser repair the stream and solve the repaired version.**
  Rejected: it would have to invent a symbol. It reports the break index and
  stops.
* **Capping confidence inside `candidates.py`.** Rejected: `Candidate` cannot
  know what the ciphertext was. The existing `confidence_cap` diagnostic is the
  right mechanism and the stage that read the letters view is the right place to
  set it.
* **Extending `polybius.solve_unknown_square` to more than 26 cells.** Rejected:
  above 26 cells it is a different cipher (many symbols onto one letter), a
  different search, and pretending otherwise would hide the new family inside a
  module whose docstring says the opposite.
* **`auto` refusing the same way the paste screen does.** Rejected, decided:
  `auto` is the expert entry point; it caps the letters-only candidates, prints
  the structure report, and still shows its full stage table. Only the paste
  screen -- the one-answer-out screen -- refuses.
