# Changelog

All notable changes to this toolkit. Keep adding to it as the team develops
the tool during the competition -- a record of what changed and why is worth
having when a round goes wrong at eleven at night.

The format is loosely [Keep a Changelog](#). Versions are our own; there is
nothing to publish to.

## [Unreleased]

Add entries here as you work. Suggested headings: `Added`, `Changed`,
`Fixed`, `Removed`.

### Added

- **`crossed.py` -- a digraph cipher whose two letters have their coordinates
  crossed.** `paired.py` could already recognise a message written as a
  52-card deck, name its two disjoint sub-decks and say the unit is two cells,
  and then stop, because it is a recogniser and says so. Nothing downstream
  could read that shape, so the only honest output was a refusal. This is the
  missing solver.

  The construction: index the first cell `6p + q` and the second `4u + v`;
  the first plaintext letter is `SQUARE_ONE[q][u]` and the second is
  `SQUARE_TWO[p][v]`. Each cell carries one coordinate of each letter.

  That crossing is also the way in. Read as a codebook the problem is 576
  cells against 1,661 units of message -- under two observations each,
  underdetermined, and a blind search over it produces confident garbage. Read
  as the construction it is **two 24-letter squares: 48 unknowns, not 576**,
  and 1,661 units is far more than enough.

  Recognition needs no key at all. Split each cell index into a high and a low
  coordinate and try the four ways of pairing one part of each cell: under the
  true pairing each half of the plaintext is a plain monoalphabetic
  substitution and keeps English's index of coincidence, and under any other
  the classes are mixtures and the distribution flattens. Measured over 192
  candidate readings of a real message: 0.0669 for the true split against
  0.0445 for the worst.

  Plain hill-climbing on the two squares is NOT enough, and the measurement
  says so: 12 restarts of 20,000 steps reached -1.78 per letter where the true
  key scores -0.9478. Simulated annealing from a frequency-ranked start
  reaches the true key on the first restart, about six seconds per restart on
  1,661 units.

  Two guards, both calibrated rather than guessed. The recognition bar sits at
  0.060 because the true split scores 0.0641 at 600 units and only 0.0580 at
  300, where the recovery gets 8 per cent of the letters right -- so the bar
  turns a length floor into a measurement. And below 900 units the reading is
  capped: at 600 units the attack recovered 97.5 per cent of the letters at
  0.72 word coverage, which clears both `strong` thresholds and is still not
  the plaintext.

- **`nihilist.py` -- a Polybius square plus a repeating additive key.** Three
  files in this repository already NAMED the Nihilist cipher as a competition
  staple that arrives as digits. None of them could read one, because its
  tokens are whitespace-separated, of variable width, and run past 99, so
  `"97 26 57 58 105"` was either invisible or destroyed by normalisation.

  The attack is a CONSTRAINT rather than a score, which is what makes it
  decisive rather than suggestive. Both coordinates are two digits with no
  carry, so a sum's tens and units each run 2..10 independently and a value
  ending in 1 is impossible whatever the key. Split by a candidate period:
  every value in a class shares one key coordinate `k`, so `value - k` must be
  a valid coordinate for EVERY value in the class. A wrong period spreads the
  class wider than any `k` can cover and is excluded outright.

  Measured on 1,494 tokens of real competition material, periods 1 to 14:
  every period except 7 and its multiple 14 left a class with zero feasible
  keys, and period 7 left exactly ONE feasible key in each of its seven
  classes. The key is not searched for, it is deduced. Subtracting it leaves a
  monoalphabetic substitution over 25 cells, which the substitution solver
  already handles -- and that solve recovers the square too, so the answer can
  be checked by hand, square and key word and all.

- **`seriated.py` -- a Polybius square whose two coordinates are written as
  two BLOCKS.** An ordinary Polybius interleaves them: row, column, row,
  column. This one writes every row coordinate first and every column
  coordinate after, so the two halves of the message are the two rows of the
  fractionation table read one after the other. Nothing paired symbols that
  way, and 2024 challenge 8B -- 3,025 digits over five symbols -- survived the
  Polybius square search, bifid at every period 2 to 30, an ADFGX-style
  columnar at every width 2 to 12, and the whole pipeline at deep effort,
  always at `weak`. The setter's own hint on that page is *"Sometimes you need
  to look at something from a different angle"*, which is about geometry, not
  keys.

  The search is tiny, and that is what makes the finding safe rather than
  lucky. A true split fractionation has two halves of EQUAL LENGTH by
  construction, so the split can only be at the middle give or take a stray
  symbol: thirteen candidates, not two thousand.

  MEASURED. On 2024 8B the split at 1,513 gives cell index of coincidence
  0.0663 and top-ten bigram share 0.1880. The reference is 2023 8B, an
  ordinary Polybius already solved and graded against its published decrypt:
  0.0692 and 0.1844 -- the same numbers. And the negative control, which is
  the part that matters: swept over ALL 2,600 split positions rather than just
  the middle, 2023 8B's best scores 0.0436 against a median of 0.0409, so the
  sweep finds nothing on a message that is not seriated, while 2024 8B's true
  split stands 0.0213 clear of every other split in its own message.

  One guard was added because a CONTROL produced it, not because anyone
  reasoned it out. Handed an ordinary interleaved Polybius whose plaintext
  repeated with a period dividing the half-length, the detector paired every
  symbol with an identical one and reported an index of coincidence of 0.204
  -- the highest number in the whole exercise, from five distinct cells, on
  the wrong cipher entirely. A high index of coincidence is also what a
  COLLAPSE looks like, so a floor on distinct cells now sits underneath it.

- **Playfair with the same-column rule reversed.** A pair sharing a column
  moves ALONG THE ROW instead of down the column. Setters vary this, and the
  variant is close to invisible to a square search: a same-column digraph is
  about one in seven, so two thirds of a message decrypts identically under
  either rule.

  That is not a curiosity, it is the most dangerous shape of answer this
  toolkit can produce. MEASURED on a real 1,502-letter competition message:
  the shipped square search converged on ONE square from 16 of 40 independent
  restarts, and that square decrypted **541 of 541 rectangle digraphs and 111
  of 111 same-row digraphs correctly, and 0 of 99 same-column digraphs.** The
  search had done its whole job; one rule out of three was wrong. The result
  was fluent English at 0.61 word coverage -- readable, plausible, wrong in 99
  places -- and no amount of extra searching would ever have fixed it.

  Under the variant rule the same square returns all 1,502 letters of the
  published plaintext, and the whole message now solves blind at `strong` with
  0.81 word coverage.

  Wired as a SECOND pass rather than always, so an ordinary Playfair pays
  nothing: the variant search runs only when no reading under the ordinary
  rule scored as clear English. And a variant reading of a square is offered
  only when it scores better than the ordinary reading of that same square, so
  an ordinary message never gains a second, worse answer.

### Fixed

- **A paste made entirely of marks was announced as "Read 0 symbols".**
  `normalize` keeps A-Z and 0-9 and counts everything else as `other`, and the
  inventory never mentioned `other` -- correctly, because in an ordinary paste
  those characters are transcription layout. But 2024 challenge 9B is written
  in three marks, `|` `/` and `\`, so for that message the marks ARE the
  message: 12,935 of them, reported as none.

  The rule is deliberately narrow: **marks are layout while a symbol stream
  exists, and are the message when none does.** An inventory that was never
  measured is all zeroes, has no marks either, and still reads "0 symbols", so
  every hand-built `NormalizedText` behaves exactly as before.

  **The sharpest version of it: the toolkit was already SOLVING some of these
  pastes while describing them as nothing.** Morse is dots and dashes, also a
  pure mark stream, and `encodings.solve` reads the original text rather than
  the letters view -- so a Morse paste decoded correctly underneath a banner
  that said "Read 0 symbols". A description can be wrong on an input the
  answer is right about.

  The refusal now names the notation instead of saying "none are letters
  (0 are digits)", which is true of 9B and tells the reader nothing: how many
  distinct marks, and which. It also stops handing out `cipher_tool polybius`
  for an alphabet it has just said nothing here can read -- that screen is
  reached only after every symbol solver has already declined, so naming them
  again is homework with a known ending.

  **No mark solver was added, on purpose.** 9B is ternary, nothing in the
  toolkit reads a ternary fractionation, and a stage that runs without being
  able to reach the answer reads as coverage. Five and six marks are the one
  case where transcribing to digits genuinely reaches a solver (a Polybius
  square, ADFGVX) and the screen says so; above ten distinct marks it is the
  opposite construction -- one mark per letter -- and calling that a
  fractionation would send the reader hunting for groups that are not there.

  Fifth variant of one bug. The four before it: a numeric ciphertext read as
  an empty paste; "no letters" and "nothing pasted" printed identically; a
  message of letters AND digits described by the count of what survived the
  filter; and a symbol stream refused because the guard keyed on digits rather
  than on the pairing.

- **The paste screen refused an odd-length digit stream, and then told the
  reader to run a command that would refuse it too.** The Polybius square
  search reads its input as coordinate PAIRS and returns nothing at all on an
  odd symbol count. `auto_solve`'s letterless branch already knew that and
  dropped the final unpaired symbol -- but `handle_paste` does not go through
  that branch, it goes through `cli._solve_symbol_stream`, so **the fix was
  real, tested, and unreachable from the only screen the operator uses.**
  A 3,025-digit challenge was answered with "this is not a letter cipher" and
  a suggestion to run `cipher_tool polybius`, which refuses it identically: a
  dead end presented as a next step.

  The last symbol is now dropped on that path too. The LAST one only --
  dropping a leading symbol misaligns every pair after it and turns a readable
  message into noise. Pinned by tests that drive the CLI helper rather than
  `auto_solve`, because that distinction is exactly what the defect turned on:
  the library call passed while the user journey failed.

- **A correct, unit-aware reading was printed under a warning saying it was
  the wrong unit.** The note "the reading below treats every symbol as one
  letter, if the pairing above is real that is the wrong unit" is about a
  reading that IGNORED the pairing, and it was printed unconditionally -- so a
  `crossed` answer that used the pairing arrived with a caveat contradicting
  it. A warning that contradicts the answer above it is worse than no warning,
  because it teaches people to ignore the warnings.

- **A reading made of one short word repeated is no longer labelled `strong`.**
  Both confidence signals read the plaintext and neither counted how many
  DIFFERENT letters it used, so `ANDANDAND...` scored -0.637 per letter with
  word coverage 1.000 -- **better than genuine English at -0.710** -- and was
  sold as `strong`. So were `THETHETHE...`, `IDIDID...` and `ISITISIT...`.
  This is not a curiosity: a search with more key freedom than its ciphertext
  collapses onto exactly this shape, so the scorer was rewarding the collapse
  instead of catching it.

  `candidates.looks_degenerate` now refuses to call such a reading better than
  `weak`, applied *after* the partial-prose promotion, because a degenerate
  text passes a window-by-window English test perfectly.

  Both of its tests are length-gated, from measurement rather than taste.
  `ATTACKATDAWN` is a perfectly good plaintext whose two commonest letters are
  58 per cent of it, so the share test waits until 40 letters -- where genuine
  English never passed 0.433 against a limit of 0.55, across 400 windows per
  length. A fixed floor of 15 distinct letters, the first threshold tried,
  would have rejected real 40-letter prose, which can use only 13, so the
  distinct-letter test waits until 200. Calibrated against the 48 published
  answers in the competition archive and 780 corpus samples: zero false
  positives, and all eight known collapse texts caught.

- **A homophonic run that simply did not find the key is no longer
  `promising`.** Units-per-symbol measures whether the attack had enough
  evidence; it says nothing about whether a given run succeeded. An unlucky
  seed fails on ciphertext that is comfortably long enough, and the failure was
  invisible, because a mis-keyed homophonic decrypt is English-shaped window by
  window by construction and trips the partial-prose promotion. Measured over
  24 runs at 400/626/820 units, eight seeds each: readings recovering 90 per
  cent or more of the letters scored word coverage 0.833 to 0.955, readings
  recovering under half scored 0.307 to 0.383, and nothing landed between.
  `RELIABLE_WORD_COVERAGE` sits in that empty band; with it, all four failing
  runs report `weak` and all twenty good ones `strong`.

- **A paired cipher written entirely in letters is no longer answered
  silently.** The honesty guard keyed on digits, but the notation is not the
  cipher: the same 52-card message transcribed with letter ranks and letter
  suits contains no digits at all, so it was treated as ordinary text. A
  cleanly recognised pairing now routes the way a digit-bearing one does, and a
  recognised structure is printed above the answer, not only on the screen that
  refuses. The refusal also stops telling a reader it declined to discard
  "0 of 1252 symbols".

- **`--max-time` is no longer ignored on the symbol-stream path** (a
  five-second budget ran for forty-one seconds), and the `[l]` screen no longer
  lists candidates that read the symbols under the heading "Letters-only
  candidates".

- **A message that mixes letters and digits is no longer answered from half of
  itself.** A real paste of 1,251 alphanumeric symbols -- 891 letters and 360
  digits -- printed `Read 891 letters. Working...`, dropped the digits in
  silence, escalated itself fast -> normal -> deep, and returned a
  monoalphabetic reading at `promising`. Half the message was thrown away
  before the search began and nothing on the screen said so.

  Three things were wrong and all three are fixed. `normalize()` had no way to
  say what it had discarded, so it now carries a `symbols` view and an
  `Inventory`, appended additively -- `original`, `letters`, `positions` and
  `groups` are byte-identical for every input. The paste screen routed on
  `normalized.is_empty`, so **one surviving letter was enough** to send a
  symbol stream down the letters-only pipeline; it now routes on whether the
  non-letters are a material part of the paste (>= 5 per cent of the symbol
  stream and >= 10 digits) and does not climb the effort ladder by itself on
  that path, because self-escalation is how three minutes were spent reaching
  a confident wrong answer. And the screen counted what SURVIVED the filter as
  though it described the paste; it now prints
  `Read 1251 symbols: 891 letters and 360 digits.`

  Behind the routing there is a second, independent guard in `auto_solve`, so
  the `auto` command and the library get it too: a stage is marked with the
  view it reads, and any candidate from a `reads="letters"` stage over a
  digit-bearing message is capped at `weak` and carries a `discarded_symbols`
  diagnostic naming what it did not see.

  MEASURED over the 40 official archive ciphertexts: digit fraction 0.0000 on
  every one of them, so no positive threshold can move that scoreboard.

### Added

- **A recogniser for paired-symbol ciphers** (`paired.py`). It solves nothing
  -- it decides whether a symbol stream is written in two disjoint alternating
  alphabets, reports the class sizes, the implied cell count and where
  alternation breaks, and names a 13-rank x 4-suit inventory as a playing-card
  deck. It tolerates a single transcription slip by REPORTING it, never by
  repairing it; refuses a one-symbol class as a separator rather than selling
  it as structure; and reports what fraction of random shuffles of the same
  symbols alternate too (expected 0.0, measured 0.0), which is what makes the
  claim falsifiable rather than an artefact of the inventory.

  `cipher_tool analyse` opens with the pasted inventory and, when there is
  one, the structure block. It is measured over the letters like everything
  else, so on a card cipher it opened with `Alphabetic characters : 891` and
  said nothing about the other 360 symbols -- and the refusal screen sends
  people there. `cipher_tool auto` prints the same block above its stage
  table.

- **Homophonic substitution** (`homophonic.py`, `cipher_tool homophonic`, a
  pipeline stage from `--fast`): more distinct symbols than there are letters,
  attacked by annealing over a FIXED letter-slot multiset with swap-only
  moves. MEASURED: 0.983 exact letters on a 52-symbol cipher from 400 units,
  0.901 on a 36-symbol one from 313.

  The slot constraint does not do what it looked like it did, and the
  measurement is worth more than the intuition. On genuine homophonic
  ciphertext the UNCONSTRAINED search matched or beat it at every shape tried,
  reaching 1.000. It earns its keep on a stream that is not homophonic: given
  600 units of uniformly random cards the free search collapses onto eight
  letters -- `SEEITSSESINILETSTATSSITSSITSSSTSEETSISASSITSLESS...` -- at
  -1.248 per letter and calls it `promising`, while the constrained search
  reports 26 letters at -2.179 and calls it `weak`. The nonsense scores nearly
  a whole log unit BETTER than the honest reading, so score alone cannot tell
  them apart. `constrain_slots=False` exists so a test watches that happen.

  Its honest limits are in ALGORITHMS.md, including the one that bites: a
  fixed multiset that is wrong cannot be climbed out of, and just above the
  27-symbol floor it produced 0.065, 0.000 and 0.850 exact letters at three
  seeds on the same cipher.

### Changed

- **The archive scoreboard is now graded against the published answers, over
  five seasons instead of two.** Forty ciphertexts from 2016, 2017, 2018,
  2023 and 2024. Of the twenty-two that publish a plaintext, **all twenty-two
  are solved at 100% of letters, and none came back `strong` without being
  solved**. The other eighteen read as coherent English but have no published
  answer and are recorded as unverified.

  The previous figure, eleven of seventeen, was measured through a harness
  that was wrong in three separate ways: it split the 2017 challenge 7B
  message into five files on four damaged characters and filed each fragment
  as an unsolved challenge; it paired answers to ciphertexts by position on
  the page; and it imposed a whole-pipeline time limit the paste screen never
  sets, which starved the Bifid square search of nine tenths of its budget
  and cost a challenge the real tool solves in ninety seconds. **Check your
  own measuring stick before filing a defect against the thing it measures.**

  The README's testing note also said the suite runs in about two minutes. It
  runs in fifteen minutes and forty-two seconds, and had been wrong for
  several releases.

### Added

- **An attack on two ciphers piled up: a polyalphabetic with a transposition
  laid over it** (`stacked.py`, `cipher_tool stacked`, a pipeline stage from
  `--normal`). The 2017 challenge 7B now comes back `strong` in under three
  seconds and matches the published answer exactly, all 3,583 letters. It
  used to be five separate files, all `weak`.

  **A perfect key that reads as nonsense means another layer, not a wrong
  key.** That is the lesson, and it was learned the expensive way: an earlier
  session recovered 7B's Vigenere key CORRECTLY, read the gibberish it
  produced, and concluded the alphabets must be mixed -- filing a second
  cipher as evidence for a harder version of the first, and recording a
  Quagmire solver as the biggest remaining gap. There is no Quagmire. The key
  it found, LATYCSE, is SCYTALE read backwards from a different starting
  point, and it was right.

  The attack cuts the stack at its joint, as the ADFGVX attack does. A
  columnar transposition reads out one whole column at a time, so each column
  arrives in the ciphertext as a contiguous run, and inside any one of those
  runs the key still advances with its own period. Split into `width`
  contiguous blocks, measure the index of coincidence of the cosets at
  spacing `period` INSIDE each block, and the right shape stands out --
  MEASURED on the real 7B ciphertext, 0.0661 against a worst case of 0.0404,
  with no knowledge of the key, the alphabet or the column order.

  Two rules, both measured rather than reasoned:

  * **The smallest shape wins, not the highest-scoring one.** Multiples of
    the true shape peak too, because splitting a real column in half leaves
    the key phase intact inside each half. Over 120 constructions the
    highest-scoring shape was NOT the true one in 73 of them.
  * **Plain English scores highly at every width and period**, so the sweep
    is meaningless on it. The whole-message index of coincidence is an early
    exit; what actually refuses English is that its smallest tied shape is a
    width of 1, meaning no transposition at all.

  MEASURED over 100 stacked messages built from five polyalphabetic keys,
  five transposition keys and four lengths: 61 read 99 per cent or more of
  the letters, 42 were exact to the last letter, and **no reading came back
  `strong` that was under 99 per cent correct**. The other 39 refuse
  honestly, as `weak` or `unlikely`.

  A search over the ragged grid's block layout was built and then **measured
  and removed**: it changed 61 to 62 and 42 to 40, which is noise, and cost
  complexity. The note in the module says so, so nobody spends an evening on
  it twice.


- **The permutation cipher: a fixed shuffle applied inside every block**
  (`permutation.py`, `cipher_tool permutation`, a pipeline stage from
  `--fast`, and a fourth family in `transposition.solve_all`). A whole
  family was missing, not a hard case of one that was present.

  Found by running the toolkit against the PUBLIC archive of past National
  Cipher Challenges rather than against messages it had encrypted itself.
  The 2018 challenge 6B ciphertext, 2,142 letters, came back `weak` from
  every transposition attack here: columnar searched widths to 63 including
  every complete-rectangle divisor, double columnar searched all 64 shapes to
  9x9, and route/grid tried every route. None of them can express "swap these
  five positions, over and over", because a columnar transposition moves a
  letter across the whole message and a permutation cipher never moves one
  out of its own block. It is a period-5 permutation under the key BAEDC and
  it now solves at `strong` in 0.9 seconds -- "INFILTRATING THE DELIBERATIONS
  OF OUR ENEMIES IS A PRINCIPAL GOAL...".

  The attack deliberately reuses the columnar machinery instead of inventing
  its own, because the two problems have the same shape once you look at
  them right: take the letters at block offset `a` of every block as a
  stripe, and scoring "stripe y follows stripe x" is the identical
  column-pair sum. Three helpers in `columnar.py` became public to say so.
  That is what makes an exhaustive sweep affordable -- 8! arrangements are
  scored against an 8-by-8 matrix, not against the message.

  Two details worth keeping. The ragged final block takes the key's relative
  order rather than being left alone: the 6B plaintext ends CONSTANTINOPLE
  and its last two letters are a block of two, so leaving short blocks
  untouched spells it CONSTANTINOPEL -- right for 2,140 letters and wrong for
  the two a reader looks at last. And the identity permutation is never
  offered, because it "decrypts" any text to itself and would top the
  ranking for every piece of plain English ever pasted in.

  MEASURED for wrong-and-strong, the property that matters most, across five
  DIFFERENT texts at each of five block sizes and six lengths from 60 to 500
  letters: 150 runs, the exact plaintext every time, and no case of a wrong
  answer labelled `strong`. So there is no confidence cap here, unlike
  Playfair -- a wrong block permutation is noise rather than near-English, so
  the score catches it. Decided by measurement, not by argument.


- **A search for an unknown Polybius square, and a paste screen that reads
  symbol streams** (`polybius.solve_unknown_square`, plus a stage that runs
  from `--fast`). Measured before: a keyed Polybius message with no keyword
  to hand scored `weak` and wrong, and handed the keyword scored `strong` and
  right -- and the paste screen could not even reach it, because it works on
  letters and a numeric ciphertext normalises to none.

  **No search over squares was needed**, which is the point worth keeping. A
  Polybius stream is a monoalphabetic substitution written two symbols at a
  time, so mapping each distinct cell to a letter turns it into an ordinary
  substitution cipher that this toolkit already breaks -- the same joint the
  ADFGVX attack cuts at. About a second, against the minutes a hill climb
  over 25! squares would have cost. The cell-to-letter mapping is reported,
  and it IS the square, read off by alignment.

  The paste screen now runs the solvers that read a symbol stream --
  `encodings` for hex, binary, decimal, Base64 and Morse, and this for a
  numeric cipher -- before falling back to explaining. Pasting a numeric
  Polybius message returns the plaintext at `strong`; pasting Morse decodes
  it. Pointing somebody at the right command was an improvement on telling
  them they had pasted nothing, but it was still homework.

- **A search for an unknown Bifid square** (`bifid.solve_unknown_square`,
  `cipher_tool bifid --search-square`, and a `--deep` pipeline stage). `solve`
  tries the squares it is handed and no others, so a keyed grid with no
  keyword available scored as noise -- honest, and still a hole, because a
  competition does not hand over the keyword. The grid is now hill-climbed
  cell by cell against the English score of its decryption, exactly as
  `playfair.py` already climbs its own. Measured: a keyed square recovered
  from 400 letters in about seven seconds and from 800 in seventeen.

  Periods are screened before any is climbed properly, and the screen had to
  be **much longer than the equivalent one for double columnar** -- measured
  on 500 letters at period 7, a 1,000-step screen ranked the true period
  SIXTH of nine, while 3,000 ranked it second and 6,000 first. The reason is
  worth keeping: a wrong transposition shape can never produce English at any
  search length, so a very short run separates it, but a Bifid period needs a
  real climb before its score moves at all.

  **Never exhaustive.** There are 25! squares, so a run that finds nothing is
  not evidence that there is nothing to find, and every candidate says so. If
  the story offers candidate keywords, `--words` is still faster and surer.

- **ADFGVX and ADFGX, with an attack** (`adfgvx.py`, `cipher_tool adfgvx`,
  and a stage that runs from `--fast`). Measured before it existed: a real
  ADFGVX message produced **no candidates at all** -- the pieces were both
  present, Polybius and columnar, and nothing joined them up.

  The attack undoes the transposition first, scoring a column order by the
  index of coincidence of the symbols paired up two at a time. When the order
  is right the pairs are the original square cells and the distribution is
  English-shaped; when it is wrong the pairs straddle cell boundaries and
  flatten. Measured on 600 letters: **0.0678 against 0.0398**, and it needs
  no knowledge of the square at all. The blind spot is handled rather than
  hoped away -- the index of coincidence cannot see the ORDER of the pairs,
  so several column orders tie exactly (twelve on that message, the true one
  among them, with a clear gap to the next value). The whole tied set is
  passed on, and the tie is broken by mapping cells to letters and running
  the existing substitution climber, because at that point it IS a
  monoalphabetic substitution. About six seconds end to end.

  It also **rebuilds the square**, by alignment rather than from the
  substitution key: cell *i* of the stream produced letter *i* of the
  plaintext, so reading the two together is the square. On the worked example
  the cells spell out the keyword it was built from. The transposition key
  alone would not let anybody re-read the message, and an answer nobody can
  check by hand is worth very little.

  Runs from `--fast` even though it is a two-stage cipher, because
  recognition is free: an ADFGVX message is written in five or six specific
  letters and has an even length, so everything else is refused before a
  single permutation is tried. Anything that is not ADFGVX gets a short
  explanation of why, not silence.

- **A solver for DOUBLE columnar transposition** (`columnar.solve_double`,
  `cipher_tool columnar --double`, and a `--deep` pipeline stage). Measured
  before it existed: `auto --deep` on a double columnar message returned a
  `promising` reading that was wrong, which is the worst pairing this toolkit
  can produce. It now solves the same message at `strong` in about a minute,
  and `ALGORITHMS.md` no longer claims there is no solver.

  The single-pass attack cannot be reused, for the reason `encrypt_double`
  already gave: after the second pass, letters that were neighbours in a
  plaintext row are no longer a fixed distance apart, so column-pair
  statistics have nothing to lock onto. Both permutations are therefore
  searched together by simulated annealing on the full plaintext score.
  Three design points were measured, not guessed: a lift-and-reinsert move as
  well as swaps, because a key off by one position cannot be repaired by
  swapping; twelve restarts rather than six, because diversification beat
  persistence (six restarts of 80,000 steps cost more than twelve of 30,000
  and were less reliable); and a cheap screening pass over every length pair
  before any is searched deeply, because depth-first spent a 40 second budget
  on four shapes of twenty-five when the answer was in the ninth -- screening
  found it in 21 seconds. Shapes are ordered by how likely a length is to be
  a real keyword, not by how cheap it is to try.

  **The search is randomised and never exhaustive**, and every candidate says
  so: two permutations of eight columns are 40,320 squared. A run that finds
  nothing is not evidence that there is nothing to find.

  The pipeline stage searches key lengths up to 8. Six was tried first and
  was worse than useless: a 7-letter keyword is ordinary in this
  competition, and on a 7x6 message the paste screen escalated all the way
  to `deep` and still reported `weak` -- honest, and no help at all to the
  person holding the ciphertext. Screening makes the wider ceiling cheap;
  all 49 shapes are covered in about thirty seconds on 400 letters.

### Changed

- **The paste flow now climbs the effort ladder by itself, and only stops on
  `strong`.** It used to run one `fast` pass, print whatever came back, and
  tell a user with a weak reading to "try 'normal' or 'deep'" -- options that
  exist on the command line and not on that screen, where the answer is
  Enter. On a real competition ciphertext that meant a screenful of gibberish
  under the headline `BEST ANSWER` and no obvious way forward. Measured on
  the progressive-shift cipher that prompted this: `fast` weak and wrong,
  `normal` **promising and still wrong**, `deep` strong and correct -- so
  escalating only while the reading is `weak` would have stopped at `normal`
  and returned a confident wrong answer, which is worse than the weak one it
  replaced. Only `strong` ends the search. Each step announces itself, so a
  two-minute solve is explained rather than silent. Text that was never
  encrypted is never escalated: there is no cipher to find, and that is the
  exact path on which a deeper search used to invent an identity key.

### Fixed

- **A message that is only partly prose was reported as a failure, and could
  not be solved at all.** Found by running the toolkit against the published
  National Cipher Challenge archive. The 2017 challenge 5A was decrypted
  perfectly and reported as `weak`: it carries a steganographic frieze, 1,500
  letters of black and white tiles enciphered along with the words, and
  averaged over the whole message the right answer scores -2.07 per letter
  with 36 per cent word coverage. Window by window it is -0.99 across the
  prose and -2.80 across the tiles.

  Worse than the label, the block corrupted the SEARCH: a key wrong
  everywhere scored -1.52 against the correct key's -2.07, because the
  correct one has to carry the tiles, so the climber was right to prefer the
  wrong key. 25, 80 and 200 restarts all plateaued on JUDIE/GUNE/OWT for
  JODIE/GONE/BUT, and neither a two-letter swap nor a three-letter cycle
  escaped it. Such a block is now found and set aside before any searching --
  distinct letters per hundred ran 18 to 23 across the prose and exactly 2
  across the frieze, so detection needs no fine judgement -- and every
  candidate reports which letters were removed. Challenge 5A now solves in
  5.0 seconds against 110 seconds of failure.

  Supporting this, `EnglishScorer.english_fraction` measures how much of a
  text reads as English window by window, and a candidate that scores weak
  overall but is more than a third English by that measure is raised to
  `promising` -- no further, since part of it genuinely is not English.

- **Reversed plaintext was never considered.** Also found in the archive: the
  2017 challenge 8A decrypted correctly and read backwards, scoring -1.999
  per letter forwards against -1.176 reversed, so the pipeline discarded the
  right answer. Every candidate's reverse is now scored and offered when it
  is clearly better; that challenge went from unsolved to the full plaintext
  at `strong` in 7.6 seconds.

- **Playfair called a wrong answer `strong` on short text, and the README
  claimed it never did.** The README said "in none of the 25 runs did a wrong
  answer come back labelled `strong`, which is the property that matters
  most". Checking that took one sweep, and it was false: at 300 letters the
  climb returned `...UNDERSZCOMYMAND...` against a true
  `...UNDERMYCOMXMAND...` -- fluent, 81 per cent word coverage, rare letters
  wrong -- and labelled it `strong`.

  Measured across five DIFFERENT texts per length (compared against the TRUE
  decryption, since Playfair inserts fillers and is never character-identical
  to the original): 200 letters 1/5 exact, 300 3/5, 400 4/5 with one
  wrong-and-`strong`, 500 and 600 4/5, 800 and 1000 5/5. So the floor is 800
  -- four correct in five is what `promising` means. Below it the label is
  now capped, using the same mechanism `substitution.solve` already carries,
  and the search no longer abandons half its restarts on a `strong` probe the
  length cannot support. Re-swept afterwards: zero wrong-and-`strong` at every
  length, and 800+ still earns `strong` at five out of five.

- **A numeric ciphertext was reported as an empty paste.** The paste screen
  works on letters, so a Polybius ciphertext -- 400 digits, no letters --
  normalised to nothing and the user was told "No letters were pasted, so
  there is nothing to work on. Run it again and paste the ciphertext when
  prompted." They had pasted the ciphertext. It is the one reply that
  guarantees somebody tries the identical thing again, and it matters for
  the competition specifically: Polybius, Nihilist and ADFGVX-written-with-
  digits all arrive as numbers, and this toolkit has solvers for that shape.
  It now says what it actually received, and names `encodings` and
  `polybius` as the commands that can work on it.
- **A weak reading was printed in full under the word ANSWER.** Forty-nine
  lines of gibberish, with the one useful line -- what to do next -- pushed
  off the screen underneath it. Weak and unlikely readings now show their
  first 240 letters and say so; `[a]` still prints every candidate in full.
  The closing advice also stops recommending a search that has already run.
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
