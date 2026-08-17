# Changelog

All notable changes to this toolkit. Keep adding to it as the team develops
the tool during the competition -- a record of what changed and why is worth
having when a round goes wrong at eleven at night.

The format is loosely [Keep a Changelog](#). Versions are our own; there is
nothing to publish to.

## [Unreleased]

Add entries here as you work. Suggested headings: `Added`, `Changed`,
`Fixed`, `Removed`.

### Fixed

- **A block cipher walked around the "not encrypted" guard, and switched it
  off for everything else.** Found by running the tool on a block of website
  navigation text -- 451 letters, never encrypted. At `--fast` it correctly
  said `THIS TEXT DOES NOT APPEAR TO BE ENCRYPTED`; at `--deep` it announced
  `Hill 2x2, key=BAAB matrix=[[1,0],[0,1]], Confidence: strong`. That matrix
  is the identity. Hill 2x2 works on letter pairs, so on an odd-length input
  it hands back 450 letters, and the guard's exact
  `plaintext == source_letters` test therefore did not recognise it. The
  truncated identity then OUTSCORED the exact ones -- one letter fewer at the
  same score per letter is a better total -- took first place, and so turned
  the warning off for the Caesar shift 0, Vigenere `AA`, variant Beaufort `A`
  and affine `a=1 b=0` candidates ranked underneath it. Five do-nothing keys,
  all labelled `strong`. `is_identity` now compares the shared prefix and
  requires it to cover 90 per cent of the longer text, so padding or
  truncation no longer hides an identity. Searching harder made the tool more
  confident and less correct, which is the worst way round for this to fail.
- **The answer menu silently ate whatever was typed at it.** The loop ended
  with `# Anything else, including a bare Enter, means "search harder"`, and
  two separate failures came out of that one line. Pasting the real
  ciphertext at the `>` prompt -- the obvious thing to do, since the prompt
  is where you type -- discarded the message without a word and re-searched
  the PREVIOUS text; the tool thought for a while and printed a confident
  answer to a question nobody had asked. And any other unrecognised input
  advanced the effort level in silence, so once at `deep` every keystroke
  printed the same "already searched as hard as this tool goes" line whatever
  was typed, which made the menu look broken. A pasted message is now
  recognised and solved, with the rest of a multi-line paste read along with
  it, and anything else unrecognised is answered rather than obeyed. A bare
  Enter still means "search harder".
- **A short substitution solve could be labelled `strong`.** The confidence
  label reads the plaintext and nothing else, so it could not see that a
  twenty-six letter key has more freedom than a twenty letter ciphertext has
  evidence. Measured: `AHTLDSGAETSPNBLPFNPN` enciphers
  `ERANDSHEWASGOINGTOGO`, the climber read it as `YFORMANYCOASTERSITST`, and
  that fluent wrong sentence scored -0.850 per letter with 80 per cent word
  coverage -- clearing both `strong` thresholds. The same candidate already
  carried `short_text_warning` saying the answer meant little, so the
  headline contradicted its own evidence. Solvers can now record a
  `confidence_cap` in their diagnostics, and `substitution.solve` sets it to
  `promising` below `RELIABLE_CLIMB_LETTERS`. Nothing above that length, and
  no other solver, changes.

## [1.0.1] - 2026-08-16

Findings from independent end-to-end verification and a read-the-code review,
rather than from the unit tests -- which is the point worth remembering: every
item below passed its own module's tests.

### Fixed

- **Substitution hill climber: stale window-score cache.** After an accepted
  swap the cached per-window scores were not refreshed, so every later
  comparison measured against stale numbers. Measured effect: the climber
  reported a running score of +2,832,795 where the true score was -919, and
  because every swap then looked like an improvement the search never
  terminated. `tests/test_regressions.py` pins this three ways, including a
  check that no single swap can improve on the returned key.
- **`cipher_tool columnar --complete` was a silent no-op in search mode.**
  The flag was only forwarded on the supplied-key path.
- **CLI called three functions that do not exist**: a
  `PolybiusSquare.from_keyword` classmethod (the factories are `standard`,
  `without_q`, `six_by_six`, `adfgx`, `adfgvx`), `validate_ciphertext`
  without its required square, and `describe_key_lengths` with a
  `max_key_length` argument. Library tests all passed; only running the
  commands found them.
- **`beaufort --variant` was ignored when searching**, so asking for one
  variant still searched both.
- **`cribs.key_length_votes` counted every pair of offsets**, contradicting
  `statistics.repeat_distances` and this project's own documented principle
  that non-consecutive gaps double-count the same evidence. Now consecutive
  gaps only.
- **`CribReport.summary()` understated what it knew.** With a crib longer
  than the ciphertext it said "nothing is ruled out" while the letter-count
  test had already ruled transposition out.
- **Morse line breaks folded to a letter gap**, so Morse transcribed one word
  per line decoded with the words run together. A newline is now a word gap.
- **`encodings` command ignored its own English-score ranking**, so
  `72 69 76 76 79` was reported as hexadecimal (`rivvy`) above decimal ASCII
  (`HELLO`).
- **Bifid module docstring mis-described its own worked example**, naming the
  wrong plaintext letters as the source of a ciphertext letter's
  coordinates.
- **`scoring.normalised` docstring quoted pre-calibration figures** (-2.2 and
  -3.4) that the measurements had already superseded (-0.89 and -2.75).

### Changed

- `columnar.DEFAULT_MAX_EXHAUSTIVE` raised from 8 to 9 so it matches
  `DEFAULT_MAX_KEY_LENGTH`: every key length the solver sweeps by default is
  now enumerated in full. Measured on 181-letter texts with a 9-column key,
  exhaustive search found the true key 6 times out of 6 in 0.28s while the
  greedy fallback found it 5 times out of 6 -- and on a harder sample greedy
  missed it entirely rather than ranking it low. The greedy docstring, which
  called itself "usually close enough", was corrected to say so.
- `--max-exhaustive` exposed on the columnar command.
- `crib` gained `--key-length`, `--no-fixed-points` and `--limit`. Without a
  key length the Vigenere crib test could never build a partial key, which is
  the most useful thing it does.
- Reporting now distinguishes **corroboration from ambiguity**. Two methods
  producing the same plaintext is evidence, not a photo finish; `score_gap`
  measures against the best *competing* reading and agreement is reported as
  `CORROBORATION`. Previously a Caesar solve warned "the top two candidates
  are within 0.000 per letter" when candidate two was the same plaintext
  found via affine `a=1` -- which is the same cipher.
- README limitations now carry measured numbers for Playfair (solve rate by
  ciphertext length), Bifid with an unknown keyed square, and the fact that a
  ciphertext-autokey primer is mathematically unrecoverable.

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
