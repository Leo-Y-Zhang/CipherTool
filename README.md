# cipher_tool

[![tests](https://github.com/Leo-Y-Zhang/CipherTool/actions/workflows/ci.yml/badge.svg)](https://github.com/Leo-Y-Zhang/CipherTool/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)](pyproject.toml)
[![licence](https://img.shields.io/badge/licence-MIT-lightgrey)](LICENSE)

An offline classical cryptanalysis toolkit, written from scratch in pure
Python for a school team entering the National Cipher Challenge.

Every cipher and every attack in this repository was written by us. There are
no runtime dependencies, no third-party cryptanalysis code, and no network
access of any kind. See [RULES_COMPLIANCE.md](RULES_COMPLIANCE.md).

> **This is a locally written cryptanalysis toolkit. Competition eligibility
> depends on the current National Cipher Challenge rules. Verify the current
> rules before using it in a live round.** Nobody at the competition has seen
> or approved this software.
>
> Checked against the 2026 rules on 16 August 2026: all clear, no code change
> required. That check is dated because it goes stale -- see
> [RULES_COMPLIANCE.md](RULES_COMPLIANCE.md#status-of-the-rules-check).

---

# Getting it running on a new laptop

Three steps. There is nothing to install and no dependencies to fetch.

### Step 1 -- get Python (once per computer)

Check whether you already have it. Open a terminal and type:

```
python --version
```

If it prints `3.10` or higher, skip to step 2. If it says "not found" or
prints an older version, install it from
**<https://www.python.org/downloads/>**.

> **Windows users:** on the very first installer screen, tick
> **"Add python.exe to PATH"** before clicking Install. If you miss it, the
> launcher will not find Python and you will have to run the installer again.

### Step 2 -- download this repository

Either download the ZIP from the green **Code** button on GitHub and unzip
it, or, if you have git:

```
git clone https://github.com/Leo-Y-Zhang/CipherTool.git
```

### Step 3 -- double-click the launcher

| Your computer | Double-click this file |
|---|---|
| **Windows** | `cipher_tool.bat` |
| **Mac** | `cipher_tool.command` |
| **Linux** | `cipher_tool.sh` |

It asks you to paste the ciphertext. Paste it in -- **any layout, any
number of lines** -- then press **Enter on a blank line**. It solves it and
prints the plaintext:

```
  Paste your ciphertext below, then press Enter on a BLANK line.

WKHUH LVQRW KLQJV RIDWD OWRFK
DUDFW HUDVK DOIIL QLVKH GWDVN

  Read 50 letters. Working...

========================================================================
BEST ANSWER  (scores as clear English)
========================================================================

  THEREISNOTHINGSOFATALTOCHARACTERASHALFFINISHEDTASK

------------------------------------------------------------------------
  Cipher      : Caesar shift
  Key         : shift=3
  Confidence  : strong  (a heuristic, not a verdict)
  Words found : THERE, NOTHING, THINGS, FINISHED
  Agreed by   : Caesar shift, Affine, Vigenere

  READ IT BEFORE YOU SUBMIT IT. This is the best-scoring guess, not a
  proven answer.
```

Then choose:

```
[Enter] try harder      [a] all candidates      [w] why (full statistics)
[s] full command shell  [n] new message         [q] quit
```

**You do not have to ask it to try harder.** If the first pass does not read
as clear English the paste screen climbs `fast` to `normal` to `deep` on its
own, announcing each step, and stops only when a reading scores `strong` or
the search is exhausted. A long message can take a couple of minutes that
way. Press Enter to push it further by hand, or `s` for the command shell
when you want to drive it yourself.

**Paste another ciphertext straight at that prompt** and it is taken as a
new message rather than a menu key -- you do not have to press `n` first.
Anything else it does not recognise it says so, instead of quietly doing
something you did not ask for.

**If it cannot solve your message it says so**, rather than dressing up its
best guess as an answer:

```
BEST ANSWER  (WEAK -- this is probably NOT the plaintext)
  ...
  ... showing the first 240 letters only. Press [a] for the full text of
  every candidate.
  ...
  This did not score like English, and the search is now exhausted. Check
  the transcription, or use [s] and a crib from the story -- 'crib THE' --
  which is worth more than more searching.
```

A reading it does not believe is shown as a preview, not poured out in full:
a screenful of gibberish under the word ANSWER reads like an answer, and
buries the line that tells you what to do next.

**On a Mac**, the very first time, macOS will refuse to open a downloaded
script. Right-click `cipher_tool.command`, choose **Open**, then click
**Open** in the dialog. After that, double-clicking works normally. If Finder
says it is not executable, run this once in Terminal:

```
chmod +x cipher_tool.command
```

The launcher also takes commands directly, if you prefer a terminal:

```
cipher_tool.bat analyse message.txt          (Windows)
./cipher_tool.command analyse message.txt    (Mac)
```

### Optional -- install it as a proper command

Only if you want to type `cipher_tool` from anywhere:

```
python -m pip install -e .
cipher_tool analyse message.txt
```

### Check it works

```
python run_tests.py
```

You should see roughly 1,160 tests pass in about two minutes.

---

## Contents

- [Getting it running on a new laptop](#getting-it-running-on-a-new-laptop)
- [Five-minute tour](#five-minute-tour)
- [The competition workflow](#the-competition-workflow)
- [Commands](#commands)
- [What is implemented](#what-is-implemented)
- [How the cryptanalysis works](#how-the-cryptanalysis-works)
- [What we wrote ourselves](#what-we-wrote-ourselves)
- [Reading the output honestly](#reading-the-output-honestly)
- [Limitations](#limitations)
- [Testing](#testing)
- [Project layout](#project-layout)

---

## Five-minute tour

Put a ciphertext in a file, five-letter groups and all:

```
$ cat message.txt
WKHUH LVQRW KLQJV RIDWD OWRFK DUDFW HUDVK DOIIL QLVKH GWDVN
```

**Look at it first.** `analyse` measures the text and suggests what to try.
It never claims to have identified the cipher.

```
$ cipher_tool analyse message.txt
```

**Then let the pipeline try everything cheap:**

```
$ cipher_tool auto message.txt --fast
```

**Or attack one family directly:**

```
$ cipher_tool caesar message.txt --all
$ cipher_tool vigenere message.txt --evidence
$ cipher_tool substitution message.txt --restarts 40 --seed 1
```

**Work interactively when you are exploring:**

```
$ cipher_tool shell message.txt
cipher> analyse
cipher> vigenere --key-length 7
cipher> crib THE
cipher> top 10
cipher> auto deep
```

## The competition workflow

This is the order we actually use when a round opens.

**1. Save the ciphertext and write down the story.**

```bash
cipher_tool context message.txt --add people="Admiral Harrow"
cipher_tool context message.txt --add places="Portsmouth"
cipher_tool context message.txt --add phrases="MEET AT MIDNIGHT"
```

Names and places from the story are the best cribs and the best key guesses
you will get. The toolkit cannot find them; you can.

**2. Measure before you attack.**

```bash
cipher_tool analyse message.txt
```

Read the Index of Coincidence and the chi-squared figure. Those two numbers
usually tell you which half of the toolkit you need. Read the heuristics, but
treat them as suggestions.

**3. Spend thirty seconds on the cheap stuff.**

```bash
cipher_tool auto message.txt --fast
```

Part A of a round is frequently a Caesar, Atbash or affine cipher, and this
finishes in seconds.

**4. Attack the indicated family properly.**

```bash
cipher_tool substitution message.txt --restarts 60 --seed 1 --full
cipher_tool vigenere message.txt --max-key-length 24 --full
cipher_tool transposition message.txt --full
```

**5. Bring in what you know.**

```bash
cipher_tool crib message.txt "HARROW"
cipher_tool substitution message.txt --map Q=E --map X=T --restarts 60
```

A crib from the story is worth more than an hour of extra search.

**6. When you are stuck, go deep and go away.**

```bash
cipher_tool auto message.txt --deep --max-time 600 --top 20 --output run.txt
```

**7. Read the plaintext before you submit it.** The toolkit ranks guesses. A
human decides. Nothing here submits anything to anyone.

## Commands

Every command takes a file, or `--text "..."`, or `-` for standard input.

### Inspection

| Command | What it does |
|---|---|
| `show FILE` | print the original input and the normalised letters side by side |
| `analyse FILE` | full statistical report and heuristic family suggestions |
| `model` | describe the English scoring model and where it came from |

### Monoalphabetic

| Command | Notes |
|---|---|
| `caesar FILE [--shift N] [--all] [--encrypt]` | `--all` ranks every one of the 26 shifts |
| `atbash FILE [--encrypt]` | fixed key, nothing to search |
| `affine FILE [-a A -b B] [--encrypt]` | brute-forces all 312 valid keys |
| `substitution FILE [--restarts N] [--map C=P] [--key KEY] [--words LIST]` | hill climbing with restarts |
| `keyword FILE [--key WORD] [--words LIST] [--reverse]` | keyword alphabets |

### Polyalphabetic

| Command | Notes |
|---|---|
| `vigenere FILE [--key K] [--key-length N] [--max-key-length N] [--evidence]` | `--evidence` shows the key-length reasoning only |
| `beaufort FILE [--key K] [--variant]` | Beaufort and variant Beaufort |
| `autokey FILE [--primer P] [--mode plaintext\|ciphertext]` | both autokey kinds |

### Transposition

| Command | Notes |
|---|---|
| `railfence FILE [--rails N] [--offset N]` | exhaustive over rail counts |
| `columnar FILE [--key WORD] [--max-key-length N] [--complete]` | |
| `transposition FILE [--routes]` | all three families at once; `--routes` lists the routes |

### Digraphic and fractionating

| Command | Notes |
|---|---|
| `polybius FILE [--key WORD] [--square LETTERS]` | 5x5 and 6x6, configurable labels |
| `bifid FILE [--key WORD] [--period N]` | period variant supported |
| `playfair FILE [--key WORD] [--check]` | `--check` validates formatting only |
| `hill FILE [--matrix 3,3,2,5] [--key HILL] [--crib TEXT]` | 2x2 and 3x3 |

### Everything else

| Command | Notes |
|---|---|
| `encodings FILE` | hex, binary, decimal ASCII, Base64, Morse -- labelled NOT encryption |
| `crib FILE "THE"` | where a guessed plaintext could sit, under each family |
| `context FILE [--add FIELD=VALUE]` | the team's story notes |
| `auto FILE [--fast\|--normal\|--deep]` | the whole pipeline; `solve` is an alias |
| `shell [FILE]` | interactive session |

### Options shared by most commands

| Option | Meaning |
|---|---|
| `--top N` | how many candidates to show (default 5) |
| `--full` | print whole plaintexts rather than a preview |
| `--output FILE` | write the report to a file |
| `--seed N` | make a randomised search reproducible |
| `--max-time S` | stop searching after S seconds |
| `--quiet` | suppress the closing disclaimer |

**Supplying a key means "do this". Omitting it means "search".** Add
`--encrypt` to run a cipher forwards.

## What is implemented

**Monoalphabetic** -- Caesar, Atbash, affine, keyword alphabet, general
substitution.
**Polyalphabetic** -- Vigenere, Beaufort, variant Beaufort, plaintext
autokey, ciphertext autokey.
**Transposition** -- rail fence (with offset), columnar (simple, complete and
double), route and grid transpositions including boustrophedon, spirals from
each corner, and diagonal reads.
**Digraphic and fractionating** -- Polybius (5x5 with I/J merged or Q
dropped, 6x6 alphanumeric, arbitrary labels including ADFGX), Bifid with
period, Playfair, Hill 2x2 and 3x3.
**Encodings** (not ciphers) -- hexadecimal, binary, decimal ASCII, Base64,
Morse.

Each has encryption, decryption and an attack. Each attack returns ranked
candidates with evidence.

## How the cryptanalysis works

Full detail with the mathematics is in [ALGORITHMS.md](ALGORITHMS.md). The
short version:

### English scoring is the engine

Every automatic attack works by proposing decryptions and asking "does this
look like English?". We answer that two ways at once:

1. **An order-3 interpolated Markov model** over letters, built at runtime
   from about 24,000 words of English prose we wrote ourselves and shipped in
   `src/cipher_tool/data/`. Each letter is scored by how likely it is given
   the three before it, backing off to shorter contexts so that ordinary
   English quadgrams our modest corpus never happened to contain are not
   scored as impossible.
2. **Word coverage** -- what fraction of the candidate can be cut into real
   words, computed by dynamic programming over every possible segmentation.

Two signals, because either alone can be fooled. Letter statistics cannot
tell true English from a near-miss key; word coverage cannot tell English
from a text stuffed with short accidental words. A candidate is only labelled
`strong` when both agree.

### Diagnosis before attack

Two measurements do most of the work:

- **Index of Coincidence** -- the chance two random letters from the text
  match. A cipher that only *relabels* letters leaves it unchanged (English
  ~0.067); a cipher that uses *several alphabets* flattens it towards random
  (0.0385).
- **Chi-squared against English letter frequencies** -- a cipher that only
  *moves* letters leaves it small; one that *changes* them makes it large.

| IC | chi-squared | Suggests |
|---|---|---|
| ~0.067 | small | transposition |
| ~0.067 | large | monoalphabetic substitution |
| ~0.040 | large | polyalphabetic, or digraphic/fractionating |

### Key lengths

For repeating-key ciphers we run three independent tests and show all three:
Kasiski examination on repeated ciphertext runs, mean Index of Coincidence of
the columns at each period, and how well each column fits a single Caesar
shift by chi-squared. When they agree, `analyse` reports that as one finding
with two supports -- not as two findings, which would overstate the evidence.

### Searches

Substitution and Playfair use hill climbing from many random restarts, and
report how many restarts converged on the same answer. Columnar uses
exhaustive permutation search where feasible and a greedy column-pairing
chain beyond that. Caesar, Atbash, affine and rail fence are searched
exhaustively, because their key spaces are tiny.

## What we wrote ourselves

All of it. In particular, the parts that would have been easy to import:

- the extended Euclidean algorithm and modular inverses (`affine.py`,
  `hill.py`)
- matrix determinant, adjugate and inverse modulo 26 (`hill.py`)
- Index of Coincidence, chi-squared fitting, Kasiski (`statistics.py`)
- the n-gram language model and its smoothing (`scoring.py`)
- word-segmentation dynamic programming (`scoring.py`)
- hill climbing with restarts (`substitution.py`, `playfair.py`)
- pattern-signature word matching (`patterns.py`)
- the English prose corpus itself (`data/corpus_*.txt`)

The only standard-library routine doing real work is `base64`, used to decode
Base64 -- a text encoding, not a cipher. `RULES_COMPLIANCE.md` explains why we
judged that acceptable and how to remove it if your reading of the rules
differs.

## Reading the output honestly

Every solve prints candidates like this:

```
Candidate 1
  Method:      Vigenere
  Key/config:  key=THUNDER (length 7)
  Score:       -631.45 (-0.902 per letter)
  Confidence:  strong  [heuristic]
  Evidence:
    - kasiski_votes: 7 scored 225
    - mean_column_ic: 0.0684
    - word_coverage: 0.8900
    - words_seen: THROUGH, MESSAGE, HARBOUR
  Plaintext:   THEREISNOTHINGSOFATALTOCHARACTER...
```

**Things to know about that output:**

- `Confidence` is a heuristic derived from measurements, never a verdict. The
  strongest label is `strong`. There is deliberately no `solved`.
- `Score` is a log probability. Compare it only between candidates over the
  *same* text. The per-letter figure is the comparable one: about `-0.90` for
  real English, about `-2.75` for anything wrong.
- If the top two candidates are close, the report says so explicitly. That is
  the case where the number tells you nothing and you must read the text.
- `auto` lists every stage it ran **and every stage it skipped**, so the
  report never reads as "we tried everything" when it did not.

**A candidate is solved when a human reads it and it makes sense.** Not
before.

## Limitations

Stated plainly, because a tool that hides its weaknesses wastes your time.

- **Short texts defeat statistics.** Below roughly 100 letters, IC and
  chi-squared are noise. `analyse` says so and downgrades its own confidence,
  but it cannot fix it.
- **Playfair hill climbing is the weakest solver here, and needs a lot of
  ciphertext.** Measured, five samples per length, eight restarts each:

  | Ciphertext | Solved | Mean time |
  |---|---|---|
  | 300 letters | 1 / 5 | 19s |
  | 500 letters | 1 / 5 | 26s |
  | 800 letters | 5 / 5 | 8s |
  | 1200 letters | 3 / 5 | 27s |
  | 1800 letters | 3 / 5 | 29s |

  Below about 800 letters treat a Playfair result as a lead, not an answer.
  Even above it, expect to re-run with a different `--seed`. Raising
  `--restarts` helps more than adding ciphertext does. In none of the 25
  runs did a *wrong* answer come back labelled `strong`, which is the
  property that matters most: it fails visibly rather than silently.

  Note also that a Playfair plaintext is never character-identical to the
  original: doubled letters were split with a filler during encryption, and
  decryption cannot tell an inserted `X` from a written one. Expect to read
  `THE X END` and mentally delete the filler.
- **Bifid with an unknown keyed square is out of reach.** The solver searches
  periods against the standard square and against any keywords you supply
  with `--words`, but it does not search the 25! possible squares. Measured:
  with the keyword supplied it solves in a fifth of a second; without it, it
  fails and reports `unlikely` rather than guessing. If you suspect Bifid,
  feed it candidate keywords from the story via `context`.
- **A ciphertext-autokey primer cannot be recovered from the message.**
  Because the key is primer + ciphertext, everything after the primer's own
  length is forced by the ciphertext alone -- so every primer yields the same
  plaintext from that point on. The solver recovers the message but its
  primer is only the English model's preferred reading of the first few
  letters, and it says so in its diagnostics.
- **Hill 3x3 cannot be brute-forced.** The key space is 26^9. Only a supplied
  matrix or the known-plaintext attack will work, and the module says so
  rather than pretending to search.
- **Autokey attacks are weaker than Vigenere attacks** and often fail on
  short texts. The solver says so in its diagnostics.
- **Long Vigenere keys are hard.** A key as long as the message is a one-time
  pad and is not breakable this way. Keys beyond about 20 letters need much
  more ciphertext than a competition round usually provides.
- **Columnar with more than about 9 columns** falls back from exhaustive
  search to a greedy heuristic, which can miss.
- **Our language model is modest.** 102,000 letters of training text, against
  the hundreds of millions behind published frequency tables. It separates
  English from noise decisively, which is what search needs, but it is
  weaker at ranking two near-miss candidates against each other. This was a
  deliberate trade for provenance we can defend.
- **The heuristics are often wrong**, especially about digraphic ciphers,
  which look statistically similar to polyalphabetic ones.
- **Nulls, homophones, and unusual variants are not implemented.** If the
  setter has done something clever, this toolkit will not find it and you
  will have to think.

## Testing

```bash
python run_tests.py              # standard library only, no dependencies
python run_tests.py -v           # verbose
python run_tests.py test_caesar  # one module
python -m pytest -q              # if you prefer pytest
```

Every cipher is tested against hand-computed known plaintext/key/ciphertext
triples, round trips, awkward input (lowercase, punctuation, five-letter
grouping, empty input), invalid keys, and a solve that recovers a known
plaintext. There are also tests that observe failure modes -- things that
*should not* solve, asserting the toolkit says so honestly rather than
returning junk confidently.

`tests/test_compliance.py` is the competition audit: it fails the build on any
third-party import, any networking module, any URL in the source, any dynamic
code execution, any non-ASCII source, or a missing docstring. It also patches
out `socket` entirely and runs a full solve, proving dynamically that no code
path reaches the network.

## Project layout

```
CipherTool/
    cipher_tool.bat           double-click launcher, Windows
    cipher_tool.command       double-click launcher, Mac
    cipher_tool.sh            launcher, Linux
    README.md                 this file
    RULES_COMPLIANCE.md       the competition audit, and what to re-check
    ALGORITHMS.md             the mathematics
    CHANGELOG.md              our development record
    LICENSE                   MIT
    pyproject.toml            no runtime dependencies, deliberately
    run_tests.py              stdlib-only test runner
    .github/workflows/ci.yml  tests on Windows, Mac and Linux on every push
    src/cipher_tool/
        cli.py                the command line and interactive shell
        auto.py               the cross-solver pipeline
        normalize.py          input handling; original text never destroyed
        scoring.py            the English language model
        statistics.py         IC, chi-squared, Kasiski, the analyse report
        reference.py          hand-entered English reference tables
        patterns.py           word pattern signatures
        candidates.py         ranked candidates and confidence labelling
        context.py            the team's story notes
        cribs.py              crib placement under each family
        caesar.py atbash.py affine.py keyword_cipher.py substitution.py
        vigenere.py beaufort.py autokey.py
        rail_fence.py columnar.py transposition.py
        polybius.py bifid.py playfair.py hill.py
        encodings.py
        data/corpus_*.txt     our own English prose, the scoring corpus
    tests/                    one test module per source module
```

---

*Read [RULES_COMPLIANCE.md](RULES_COMPLIANCE.md) before using this in a live
competition round.*
