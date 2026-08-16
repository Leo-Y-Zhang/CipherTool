# Algorithms

Concise explanations of every algorithm in the toolkit, with the mathematics
written out. The intended reader is a teammate who wants to understand *why*
a technique works, not just how to run it.

Each section names the file it describes, so you can read the code alongside.

**Contents**

- [Part 1: Foundations](#part-1-foundations)
  - [Normalisation](#normalisation-normalizepy)
  - [English scoring](#english-scoring-scoringpy)
  - [Word coverage](#word-coverage-scoringpy)
  - [Statistics: IC, chi-squared, Kasiski](#statistics-statisticspy)
  - [Pattern signatures](#pattern-signatures-patternspy)
  - [Candidate management](#candidate-management-candidatespy)
- [Part 2: Substitution ciphers](#part-2-substitution-ciphers)
- [Part 3: Polyalphabetic ciphers](#part-3-polyalphabetic-ciphers)
- [Part 4: Transposition ciphers](#part-4-transposition-ciphers)
- [Part 5: Digraphic and fractionating ciphers](#part-5-digraphic-and-fractionating-ciphers)
- [Part 6: Encodings](#part-6-encodings)
- [Part 7: Cribs](#part-7-cribs)

Notation: plaintext letters are `P`, ciphertext letters `C`, key letters `K`,
all as numbers `0..25` with `A = 0`. All arithmetic is modulo 26 unless
stated.

---

# Part 1: Foundations

## Normalisation (`normalize.py`)

Competition ciphertext arrives as something like

```
HEALI OPASD EHANS XQKTR
```

Those five-letter groups are a transcription convenience. They say nothing
about where plaintext words begin and end. A tool that splits on whitespace
and looks for "words" will confidently produce nonsense.

So `normalize()` builds two views of the input and keeps both:

- `original` -- the input, untouched, byte for byte.
- `letters` -- uppercase `A-Z` only, which is what every attack reads.

and a `positions` tuple where `positions[i]` is the index in `original` of
`letters[i]`. That map is what makes `relayout()` possible: given a recovered
plaintext, it pours the letters back into the original punctuation and
spacing so a human can read the result in its original shape.

Accented characters are folded rather than dropped. `unicodedata.normalize`
decomposes a character into a base letter plus combining marks, and we remove
the marks -- so a paste out of a PDF yields the letters it appears to contain
instead of silently losing them.

The only thing the toolkit ever does with the whitespace groups is *report*
them: if every group is the same length, `analyse` says so and explicitly
notes that this is formatting, not word boundaries.

## English scoring (`scoring.py`)

Every automatic solver works the same way: propose a decryption, ask "does
this look like English?", keep what scores best. The quality of that judgement
sets the quality of everything else.

### The model

We score a candidate plaintext as the log probability of its letter sequence
under an **interpolated Markov chain of order three** -- each letter predicted
from the three before it:

```
score(text) = sum over i of  log10 P(x_i | x_{i-3}, x_{i-2}, x_{i-1})
```

### Why interpolation is necessary

The obvious approach is to count quadgrams in a corpus and use their
frequencies directly. That fails here. There are `26^4 = 456,976` possible
quadgrams, and our corpus contains only about 102,000 letters, in which we
observe roughly 20,600 distinct quadgrams. A direct frequency model would
assign probability zero -- score minus infinity -- to the other 95%, including
plenty of perfectly ordinary English. Every candidate would look impossible.

So we back off to shorter contexts, which are far better observed, using
**add-K interpolation**:

```
P1(d)       = (C1(d)    + 1)                / (N       + 26)
P2(d|c)     = (C2(cd)   + K * P1(d))        / (C1(c)   + K)
P3(d|bc)    = (C3(bcd)  + K * P2(d|c))      / (C2(bc)  + K)
P4(d|abc)   = (C4(abcd) + K * P3(d|bc))     / (C3(abc) + K)
```

where `Cn` is the count of an n-gram in the corpus and `N` the total letters.

Read the last line as: *trust the quadgram count in proportion to how often
we have seen its trigram context; otherwise fall back towards what the
trigram model believes.* `K` is the number of imaginary observations given to
the fallback, so a context seen exactly `K` times splits its belief evenly
between the two orders. We use `K = 6`; the result is not sensitive to it.

Each order is defined in terms of the one below, so the recursion bottoms out
at a Laplace-smoothed unigram model in which no letter is impossible.

### Implementation

The four count tables are flat Python lists indexed by base-26 arithmetic:
the quadgram `abcd` lives at `((a*26 + b)*26 + c)*26 + d`. Counting is a
single pass over the corpus.

The order-3 log-probability table has 456,976 entries and is built lazily on
first use, because `analyse` never needs it.

Scoring slides a window with modular arithmetic instead of re-slicing the
string:

```python
index = ((v0*26 + v1)*26 + v2)*26
for position in range(3, count):
    index += values[position]
    total += log4[index]
    index = (index % 17576) * 26     # 17576 = 26^3: drop the oldest letter
```

`index % 17576` discards the leading letter and `* 26` makes room for the
next -- so each letter costs one modulo, one multiply, one add and one list
lookup. This runs at roughly 19 million letter-scorings per second, which is
what makes hill climbing practical in pure Python.

That optimisation is also exactly where an off-by-one would hide, and a
silently wrong scorer would poison every solver in the toolkit. So
`tests/test_scoring.py` re-derives the same score with a naive
re-slicing loop over 150 random strings and asserts they agree to nine
decimal places.

### Where the corpus comes from

`data/corpus_*.txt` is about 24,000 words of English prose **written for this
project** in six registers: narrative, dialogue, correspondence, expository
writing, popular history and everyday journalism. Nothing was downloaded or
copied. See `RULES_COMPLIANCE.md` section 6.

### Calibration

Training on five of the six corpus files and testing on the sixth -- text the
model has never seen -- gives, in mean log10 probability per letter:

| Text | n-gram/letter | Word coverage |
|---|---|---|
| Real English (held out) | -0.89 (worst -1.03) | 0.85 |
| Key with 2 letters wrong | -1.21 (worst -1.66) | 0.65 |
| Wrong Caesar shift | -2.69 | 0.05 |
| Scrambled monoalphabetic | -2.73 | 0.07 |
| Uniformly random letters | -2.77 | 0.06 |

Two things matter here. First, the gap between English and everything wrong
is enormous -- about 1.8 log units per letter, so over 300 letters a correct
decryption beats a wrong one by 540 orders of magnitude. Second, a
nearly-correct key lands in between, which is exactly the band the
`promising` confidence label describes.

The confidence thresholds in `candidates.py` sit in those gaps, and
`tests/test_scoring.py::TestCalibrationSeparation` reproduces the measurement.
If the corpus changes and those tests fail, the thresholds need revisiting --
not the tests relaxing.

## Word coverage (`scoring.py`)

Letter statistics alone cannot tell true English from a near-miss key that
happens to produce English-ish letter runs. So we measure a second, largely
independent thing: what fraction of the candidate can be cut into real words.

This is a dynamic program. Let `best[i]` be the greatest number of letters of
`text[:i]` coverable by non-overlapping known words. Then

```
best[0] = 0
best[i] = max( best[i-1],                                  # cover nothing here
               max over word lengths L of
                   best[i-L] + weight(L)  if text[i-L:i] is a known word )
```

and coverage is `best[n] / n`. Skipping a letter is always allowed, which is
what lets a partly-correct decryption earn partial credit rather than falling
off a cliff.

Two-letter words are weighted at half value. Random letters accidentally
contain a great many instances of `AT`, `IN` and `SO`, and without the
discount a text of noise scores far too well.

The lexicon is the hand-typed common-word list in `reference.py` plus every
distinct word in our own corpus: about 3,900 words. Single letters other than
`A` and `I` are excluded, since otherwise every possible text would be fully
covered.

## Statistics (`statistics.py`)

### Index of Coincidence

The probability that two letters drawn at random from the text, without
replacement, are the same letter:

```
IC = sum over letters of  n_i * (n_i - 1)  /  ( N * (N - 1) )
```

For a flat distribution over 26 letters, `IC = 1/26 = 0.0385`. English prose
is lumpy, so `IC` is about `0.0667`.

The crucial property: **a monoalphabetic cipher only relabels letters, so it
leaves IC exactly unchanged.** A polyalphabetic cipher spreads each plaintext
letter across several cipher letters and pulls IC towards 0.0385. So IC is
the first question to ask of any ciphertext, and it splits the world in two.

### Chi-squared against English

How far the letter distribution is from English:

```
chi2 = sum over letters of  (observed_i - expected_i)^2 / expected_i
```

reported per letter so that texts of different lengths compare.

The crucial property here: **a transposition cipher only reorders letters, so
its letter frequencies are exactly those of the plaintext** and chi-squared
stays small. A substitution cipher has an English-like IC but a scrambled
distribution, so chi-squared is large.

Together the two measurements separate the three main families:

| IC | chi-squared | Suggests |
|---|---|---|
| ~0.067 | small | Transposition (letters moved, not changed) |
| ~0.067 | large | Monoalphabetic substitution (letters changed, not moved) |
| ~0.040 | large | Polyalphabetic, or a digraphic/fractionating cipher |

This table is the whole basis of the heuristic report -- and it is a
heuristic, not an identification.

### Kasiski examination

If a repeated plaintext string happens to line up with the same part of a
repeating key, it enciphers to the same ciphertext string. So the distance
between two identical ciphertext runs is likely to be a multiple of the key
length.

Collect the distances, factorise them, and the factor appearing most often is
the prime suspect. Two refinements matter:

**Only consecutive gaps are counted.** For a run appearing at positions
`0, 4, 10` we count gaps `4` and `6`, not also `10`. Every pairwise distance
is a sum of consecutive gaps, so counting them all double-counts the same
evidence and inflates the tally.

**Votes are weighted by repeat length.** A repeat of length `n` gets weight
`n - 2`. Long repeats are far less likely to be coincidence, and without the
weighting a swarm of accidental trigram repeats drowns out one decisive
seven-gram.

Finally, multiples of a stronger candidate are hidden from the shortlist: if
3 is the real key length then 6, 9 and 12 also divide every gap that 3 does,
and presenting them as separate findings is misleading.

### Index of Coincidence by period

An independent check on the same question. Split the ciphertext into `p`
columns, where column `i` holds every letter enciphered by key position `i`.
If `p` really is the key length, each column is a plain Caesar shift of
English -- and Caesar shifts preserve IC, so each column reads about 0.067.
Wrong periods leave columns that are still a mixture, so they stay near
0.0385.

Periods that would leave columns of fewer than twenty letters are **not
reported at all**. IC is far too noisy on samples that small, and reporting
it anyway is how people talk themselves into a wrong key length.

When Kasiski and the column-IC test agree, `analyse` reports that as **one**
finding with two supports, not two findings. Two agreeing measurements are
stronger evidence than either alone, but they are not two independent
discoveries.

## Pattern signatures (`patterns.py`)

A pattern signature records the shape of repeated letters in a word and
throws away the letters:

```
HELLO  -> 0-1-2-2-3
PEOPLE -> 0-1-2-0-3-1
ATTACK -> 0-1-1-0-2-3
```

A monoalphabetic substitution replaces each letter consistently, so it
**cannot change a word's signature**. Whatever `HELLO` becomes, it is still
five letters with the third and fourth identical and every other distinct. So
if a ciphertext token is a whole plaintext word, the plaintext must be an
English word with the same signature -- which usually cuts thousands of
candidates to a handful.

A hyphen separator is used rather than bare digits because a word with more
than ten distinct letters would otherwise be ambiguous: `0110` could be
`0,1,1,0` or `0,11,0`.

Each surviving candidate then supplies letter equations, and
`mapping_from_pair` rejects a pair unless the implied mapping is a
**bijection** -- one cipher letter cannot mean two plain letters, *and* two
cipher letters cannot mean the same plain letter. Enforcing the second
direction is what prunes most spurious signature matches.

**Caveat, enforced in code.** This only helps when word divisions survive in
the ciphertext, which in this competition they usually do not. So every
entry point takes the candidate words *explicitly*; nothing in the module
ever splits ciphertext on whitespace by itself.

## Candidate management (`candidates.py`)

Every solver returns a `CandidateSet`, never a single answer. Each candidate
carries its method, key, score, plaintext and a diagnostics dictionary of
whatever evidence the solver can offer.

Duplicates -- the same plaintext from the same method -- are merged, and an
`agreements` counter records how many times the search arrived there
independently. For a hill climber with random restarts that count is real
evidence: twenty restarts converging on one key is worth far more than one
restart finding it once.

`score_gap()` reports the per-letter margin between the best and second-best
candidate. A large gap means the search found one clearly better answer; a
tiny gap means several keys explain the ciphertext about equally well, which
is exactly when a human must look at the text rather than the number.

Confidence labels are deliberately coarse and pessimistic. `strong` requires
**both** a good n-gram score and good word coverage, because either alone can
be fooled. When coverage was not measured, the label is capped below
`strong`. The strongest label available is `strong`; there is no `solved`.

---

# Part 2: Substitution ciphers

*(Caesar, Atbash, affine, keyword and general monoalphabetic substitution --
see the module docstrings in `caesar.py`, `atbash.py`, `affine.py`,
`keyword_cipher.py` and `substitution.py`.)*

# Part 3: Polyalphabetic ciphers

*(Vigenere, Beaufort, autokey -- see `vigenere.py`, `beaufort.py`,
`autokey.py`.)*

# Part 4: Transposition ciphers

*(Rail fence, columnar, route/grid -- see `rail_fence.py`, `columnar.py`,
`transposition.py`.)*

# Part 5: Digraphic and fractionating ciphers

*(Polybius, Bifid, Playfair, Hill -- see `polybius.py`, `bifid.py`,
`playfair.py`, `hill.py`.)*

# Part 6: Encodings

*(Hexadecimal, binary, decimal ASCII, Base64, Morse -- see `encodings.py`.)*

# Part 7: Cribs

*(Crib placement under each cipher family -- see `cribs.py`.)*
