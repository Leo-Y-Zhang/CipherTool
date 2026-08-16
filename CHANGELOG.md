# Changelog

All notable changes to this toolkit. Keep adding to it as the team develops
the tool during the competition -- a record of what changed and why is worth
having when a round goes wrong at eleven at night.

The format is loosely [Keep a Changelog](#). Versions are our own; there is
nothing to publish to.

## [Unreleased]

Add entries here as you work. Suggested headings: `Added`, `Changed`,
`Fixed`, `Removed`.

## [1.0.0] - 2026-08-16

First complete version.

### Added

**Foundation**
- `normalize.py`: two synchronised views of the input (untouched original and
  letters-only), a position map so plaintext can be poured back into the
  original layout, accent folding, and grouping helpers. Whitespace is never
  treated as a plaintext word boundary anywhere in the toolkit.
- `scoring.py`: an order-3 interpolated Markov model of English built at
  runtime from our own prose corpus, plus word-coverage scoring by dynamic
  programming. Two independent signals, reported separately.
- `reference.py`: hand-entered English letter frequencies, digraph and
  trigraph tables, and a common-word list.
- `data/corpus_*.txt`: about 24,000 words of original English prose written
  for this project, in six registers, as the sole training text for the
  language model.
- `candidates.py`: ranked candidate management with de-duplication,
  agreement counting, score gaps and coarse confidence labelling.
- `patterns.py`: word pattern signatures (`HELLO -> 0-1-2-2-3`) and
  bijective mapping derivation.
- `statistics.py`: the `analyse` engine -- letter counts and frequencies,
  Index of Coincidence, chi-squared against English, repeated 2- to 5-grams
  with distances, Kasiski examination with weighted factor votes, IC by
  period, length factorisation, and heuristic cipher-family suggestions.

**Ciphers and attacks**
- Caesar, Atbash, affine, keyword substitution, general monoalphabetic
  substitution.
- Vigenere, Beaufort and variant Beaufort, plaintext and ciphertext autokey.
- Rail fence, columnar (including complete and double), route/grid
  transposition.
- Polybius squares, Bifid, Playfair, Hill (2x2 and 3x3).
- Encoding helpers: hexadecimal, binary, decimal ASCII, Base64, Morse.

**Tooling**
- `cli.py`: the `cipher_tool` command, including an interactive shell.
- `auto.py`: the cross-solver pipeline with `--fast`, `--normal`, `--deep`
  and `--max-time`.
- `cribs.py`: crib placement testing for substitution, Caesar, affine,
  Vigenere and transposition.
- `run_tests.py`: runs the whole suite with nothing but the standard library.

**Documentation**
- `README.md`, `ALGORITHMS.md`, `RULES_COMPLIANCE.md`, this changelog.

### Notes on decisions worth remembering

- **The confidence thresholds are measured, not guessed.** They come from
  training the scorer on five of the six corpus files and testing on the
  sixth. `tests/test_scoring.py::TestCalibrationSeparation` reproduces the
  measurement, so if the corpus changes and those tests fail, the thresholds
  in `candidates.py` need revisiting rather than the tests being relaxed.
- **Kasiski votes are weighted by repeat length.** A swarm of accidental
  trigram repeats would otherwise drown out one decisive seven-gram.
- **Agreeing evidence is reported once.** When Kasiski and the column-IC
  test pick the same key length, that is one finding with two supports, not
  two findings. Reporting it twice overstates the independent evidence.
- **IC is not reported for periods leaving fewer than twenty letters per
  column.** The measurement is meaningless on samples that small, and
  reporting it anyway is how people talk themselves into a wrong key length.
- **No runtime dependencies, deliberately.** See `RULES_COMPLIANCE.md`.
