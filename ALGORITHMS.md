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

### Saying what was thrown away

The letters-only view is a filter, and for a long time it could not say what
it had removed. That is how a message of 1,251 alphanumeric symbols -- 891
letters and 360 digits -- reached the paste screen as `Read 891 letters`, had
its digits dropped before any search began, and came back as a monoalphabetic
substitution labelled `promising`. Nothing on that screen was false. Nothing
on it was the truth either.

So `normalize()` builds, in the *same* single pass, two more views:

- `symbols` -- uppercase `A-Z` **and** `0-9`, in order, with its own
  `symbol_positions` map into `original`;
- `inventory` -- an `Inventory` counting letters, digits, other characters and
  whitespace, whose `digit_fraction` is measured over the symbol stream rather
  than over the whole input (spaces are layout, and dividing by them would
  make the same message look less numeric merely for being printed in groups
  of five).

They are appended to `NormalizedText` **with defaults**, and an all-zero
inventory means NOT MEASURED rather than "measured and found empty". Anything
that builds a `NormalizedText` by hand gets zeroes, so every predicate over an
inventory has to read zeroes as "behave exactly as the toolkit did before this
existed". `auto.non_letters_are_material` does, and that is the case that
reaches a user.

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

### Degeneracy: the failure both signals share

Requiring two signals does not help when the same text fools both. Neither
counts how many **different** letters a reading uses, so a text built from one
short word repeated satisfies them completely. Measured on 626 letters:

| reading | n-gram/letter | word coverage | distinct letters |
|---|---|---|---|
| `ANDANDAND...` | **-0.637** | 1.000 | 3 |
| `THETHETHE...` | -0.760 | 1.000 | 3 |
| `IDIDID...` | -0.756 | 0.748 | 2 |
| real English | -0.710 | 0.960 | 24 |

`ANDANDAND...` outscores genuine English, and every one of those was labelled
`strong`. Word coverage cannot object -- a repeated real word is fully covered
by construction -- and neither can the windowed English test, which is why
`looks_degenerate` is applied *after* the partial-prose promotion rather than
before it.

This is the shape a search collapses onto when it has more key freedom than
ciphertext, so the scorer was rewarding the collapse instead of catching it.

Both tests are length-gated, and the gate is the interesting part. Short text
is repetitive by nature: `ATTACKATDAWN` is a perfectly good plaintext whose
two commonest letters are 58 per cent of it. Swept over 400 windows per
length, genuine English last exceeds the 0.55 share limit at 12 letters and is
clear of it from 15; from 40 upward it never passed 0.433. The share test
therefore starts at 40. The first distinct-letter threshold tried was a flat
"fewer than 15", which would have **rejected real 40-letter prose** -- it can
use only 13 -- so that test waits until 200 letters, where genuine English
holds 19 or more.

Calibrated against the 48 published answers in the competition archive and 780
corpus samples from 40 to 3,000 letters: zero false positives, and all eight
known collapse texts caught. Rejecting a correct answer would be worse than
anything this guard prevents, so the thresholds sit well clear of real text on
one side and well clear of every observed collapse on the other.

---

# Part 2: Substitution ciphers

## Caesar shift (`caesar.py`)

Every letter moves the same number of places along the alphabet:

```
E(P) = (P + s) mod 26
D(C) = (C - s) mod 26
```

There are 26 keys and one of them does nothing, so 25 useful ones. Decryption
is the same operation with the shift negated, which is exactly what `decrypt`
does -- it calls the shared `_apply_shift` with `-s mod 26`.

### The attack

Brute force is complete here, so the only question is how to judge the 26
readings. We judge each one twice, with two measures that fail in different
ways.

**The n-gram score** (`scoring.EnglishScorer`) is the primary ranking. It
knows about order: `THE` and `QXZ` are both three letters, but only one is
English.

**Chi-squared** against English letter frequencies is the second, order-blind
opinion:

```
chi2 = sum over letters j of  (observed_j - expected_j)^2 / expected_j
```

reported per letter so texts of different lengths compare. A shift is a
*rigid rotation* of the frequency histogram, so exactly one rotation lines the
tall E, T, A spikes up with the spikes English expects. Every other rotation
drops a common letter where a rare one was expected, and dividing by that
small expected count makes the penalty enormous. That is why frequency fitting
works so well on a shift and so badly on a general substitution, where no
single rotation can fix anything.

All 26 shifts are scored from **one** pass of letter counting. Under
decryption shift `s` the count of plaintext letter `j` is the count of
ciphertext letter `(j + s) mod 26`, so the 26 candidate histograms are 26
rotations of one 26-entry vector. That is what makes `best_shift_by_chi_squared`
cheap enough for the Vigenere and substitution solvers to call it per column.

Every candidate carries `chi_squared`, `rank_by_chi2`, `rank_by_ngram` and
`measures_agree`. Agreement between two independent measures is the evidence.

### Limitations

`solve` returns **all 26** candidates, not the best few: `top` is accepted for
a uniform solver interface but does no filtering, and `time_budget` never
triggers, because hiding the margin between the winner and the field would
throw away the most useful thing a Caesar attack produces. Trim with
`CandidateSet.top(n)`; read the margin with `score_gap()`. Chi-squared alone is
unreliable on a few dozen letters, which is when the two measures disagree and
a human should look. Empty input gives an empty `CandidateSet`, and
`chi_squared_by_shift` returns `inf` for every shift rather than zeros, which
would look like a perfect fit.

## Atbash (`atbash.py`)

Atbash writes the alphabet backwards under itself, so A becomes Z, B becomes
Y, and M and N swap. The whole cipher is one line:

```
E(P) = 25 - P
```

It has **no key at all**. Decryption is the same operation, because
`25 - (25 - P) = P`; Atbash is an *involution*, and `decrypt` is implemented
as a call to `encrypt` so the property is asserted in tests rather than
assumed.

It is the affine cipher with `a = 25, b = 25`, since `25P + 25 = -P - 1 = 25 - P`
(mod 26, because `25 = -1`). It is **not** a Caesar shift: a shift slides the
alphabet along, Atbash reflects it.

### "Attacking" a cipher with no key

There is nothing to search, so `solve` returns exactly one candidate. The only
useful work it can do is decipher the one possible way and then report
honestly whether the result looks like English, so that an operator who tried
Atbash on something that is not Atbash is told so rather than handed confident
nonsense. Three pieces of evidence come back:

- the n-gram score and word coverage of the reading, via `annotate`;
- `chi_squared` of that reading against English letter frequencies;
- `ciphertext_ic`, the index of coincidence of the **ciphertext**.

The last one is a prior check and the sharpest of the three. Atbash only
relabels letters, so it cannot change IC. If the ciphertext IC is more than
`0.015` away from the English value of about `0.0667`, a `warning` is added to
the diagnostics: no monoalphabetic cipher can turn this text into English, and
the operator should be looking at a polyalphabetic or fractionating cipher.

### Limitations

With no key there is no search, so there is no rank, no margin and no
convergence evidence -- judge the plaintext and the scores, never the
position in a list. A `weak` or `unlikely` label here simply means the text is
not Atbash. The IC warning rules out the whole monoalphabetic family, not
Atbash specifically, so its absence is not positive evidence.

## Affine (`affine.py`)

The affine cipher multiplies and then adds:

```
E(P) = (a * P + b) mod 26
```

### Why `a` must be coprime with 26

Encryption is only useful if it can be undone, so `E` must be a bijection on
the 26 letters. Multiplication by `a` is a bijection modulo 26 exactly when
`gcd(a, 26) = 1`. If `d = gcd(a, 26) > 1` then

```
a * (P + 26/d) = a*P + 26 * (a/d) = a*P   (mod 26)
```

so `P` and `P + 26/d` encipher to the same letter and the cipher collapses `d`
plaintext letters onto one ciphertext letter. Concretely `a = 2` sends `P` and
`P + 13` to the same place, and `a = 13` sends `P` and `P + 2`. Since
`26 = 2 * 13`, the multipliers that fail are the even ones and 13, leaving
**12** usable values:

```
1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25
```

`VALID_MULTIPLIERS` is *computed* from `extended_gcd` at import rather than
typed in, so the list cannot drift from the arithmetic that justifies it. The
key space is `12 * 26 = 312`. Two members are ciphers of their own: `a = 1`
gives `E(P) = P + b`, which is **Caesar with shift b**, and `a = 25, b = 25`
gives **Atbash**. `describe_key` names both, and the solver reports them as
`equivalent_to` in the diagnostics -- which matters when you are deciding what
the next round of a puzzle will be.

### Decryption and the extended Euclidean algorithm

To invert `C = a*P + b` we need `a_inv` with `a * a_inv = 1 (mod 26)`, since
then `a_inv * (C - b) = P`:

```
D(C) = a_inv * (C - b) mod 26
```

We compute `a_inv` ourselves rather than calling `pow(a, -1, 26)`, because the
algorithm that finds it is also the proof that it exists. Ordinary Euclid
replaces `(r_prev, r)` with `(r, r_prev mod r)` until the remainder is 0. The
extended version carries a coefficient `s` alongside each remainder with
`a*s + m*t = r`, starting from the trivial identities `a*1 + m*0 = a` and
`a*0 + m*1 = m`, and applying to the coefficients exactly the subtraction it
applies to the remainders: if `r_next = r_prev - q*r` then
`s_next = s_prev - q*s`. Working `a = 5`, `m = 26`:

```
  r_prev  r    q      s_prev  s
   5      26   0       1      0
   26     5    5       0      1
   5      1    5       1     -5
   1      0    -      -5     26
```

The loop ends when the remainder hits 0, so `gcd(5, 26) = 1` with `s = -5`:
check `5*(-5) + 26*1 = 1`. Since `26*t` vanishes modulo 26, `5 * (-5)` leaves
remainder 1, and `a_inv = -5 mod 26 = 21` (indeed `5 * 21 = 105 = 4*26 + 1`).
If the gcd is not 1 there is no inverse, and the same computation says so,
which is why `modular_inverse` raises a truthful error instead of guessing.

Worked example with `a = 5, b = 8`. `H = 7`, so `5*7 + 8 = 43 = 17 = R`;
`HELLO` enciphers to `RCLLA`. Back: `R = 17`, `21 * (17 - 8) = 189 = 7 = H`.

### The attack and its limitations

Brute force over all 312 keys, ranked by the n-gram model with chi-squared
alongside. No frequency shortcut is needed at this size and one would only add
a failure mode. All 312 candidates are returned, so `top` does no filtering
here either.

An exhaustive search cannot miss the key, so if the best candidate still reads
as nonsense the text is not an affine cipher -- a finding, not a failure.
Decrypting is instant but word-coverage annotation is not, so on a very long
ciphertext 312 annotations take a few seconds. `time_budget` stops the search
cleanly, records `time_budget_hit`, and reports how much of the key space
`keys_tested` covered; at least one key is always tried, so a budget of 0
returns one honest candidate rather than nothing.

## Keyword substitution (`keyword_cipher.py`)

(The module is `keyword_cipher`, not `keyword`, because `keyword` is a
standard library module and shadowing it would break anything in the process
that inspects Python reserved words.)

A keyword substitution is an ordinary monoalphabetic substitution whose cipher
alphabet is grown from a memorable word. Write the keyword with repeated
letters dropped, then follow it with every unused letter in order:

```
keyword    SECRET
reduced    SECRT
remainder  ABDFGHIJKLMNOPQUVWXYZ
cipher     SECRTABDFGHIJKLMNOPQUVWXYZ
```

Enciphering is then a table lookup:

```
plain   A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
cipher  S E C R T A B D F G H I J K L M N O P Q U V W X Y Z
```

so `ATTACKATDAWN` becomes `SQQSCHSQRSWK`. Decryption inverts the table:
`invert_alphabet` builds the row that undoes it, and `decrypt_with_alphabet`
enciphers with that.

Two variants are supported. **Offset** (`start_letter`) writes the keyword
under a plain letter other than A, rotating the finished row and wrapping the
remainder round the end. `keyword_alphabet("SECRET", "D")` gives
`XYZSECRTABDFGHIJKLMNOPQUVW` -- the row shifted three places, with `UVW`
wrapped to the front as `XYZ`. This multiplies the keyspace by 26 and removes
the construction's most obvious weakness, which is that with `start_letter='A'`
the tail of the alphabet is usually a long stretch of letters enciphering to
themselves. **Reversed remainder** (`reverse_remainder`) fills the unused
letters backwards from Z instead: `SECRTZYXWVUQPONMLKJIHGFDBA`.

### Attack 1: dictionary search

`solve` builds the alphabet for every candidate keyword, decrypts and scores.
Choosing the alphabet from a word throws the 26! keyspace away: the default
word list is every lexicon word of 3 to 12 letters, **3835** of them, times the
start letters swept times two remainder directions. Measured on our lexicon
that is 7277 trials, of which 393 are duplicate alphabets that get skipped, and
the whole search takes about half a second. Only a heap of the best
`max(top*3, 10)` trials is annotated, because word-coverage segmentation is far
too expensive to pay 7000 times.

Note the default `start_letters="A"`: only the textbook construction is swept,
so a keyword written under a different start letter is **missed** unless you
pass `start_letters=ALPHABET`, at 26 times the cost. `try_reversed` is on by
default. The search is complete only with respect to the word list.

### Attack 2: working backwards from a recovered alphabet

Suppose the hill climber in `substitution.py` has already recovered an
alphabet. Was it built from a keyword, and which? The structure to look for is
the **tail**: everything after the keyword is in strict alphabetical order.
`candidate_keywords` undoes each of the 26 rotations, in each of the 2
remainder directions, and finds the longest strictly ordered suffix; whatever
precedes it is the keyword.

Worked example on `SECRTABDFGHIJKLMNOPQUVWXYZ` at offset 0. Walk backwards
from Z while each letter still continues the run:

```
S E C R T | A B D F G H I J K L M N O P Q U V W X Y Z
keyword   |<---------- ordered tail, 21 letters ------>
          |
          the run stops here: T does not come before A
```

Head `SECRT` is the keyword stem, tail length 21. How much evidence is that?
For a permutation chosen at random the chance its last `T` letters happen to
ascend is exactly `1/T!`, because all `T!` orderings are equally likely. A tail
of 6 is 1 in 720, of 10 is 1 in 3.6 million, and of 21 is 1 in 5.1e19 -- which
is what the tool reports. We test 52 alignments (26 rotations, 2 directions),
so multiply by 52 for the chance that *some* alignment fires by accident.
`minimum_tail` defaults to 6 for exactly this reason: below that the method is
noise. A genuinely random alphabet returns an empty list, which is the honest
answer.

Finally the dictionary reads the stem back: `SECRT` is matched to `SECRET`,
because the keyword lost its repeated letters when the alphabet was built and
`SECRT` is all that can ever be recovered.

### Limitations

**The tail can swallow the keyword's last letter.** If the keyword's final
letter falls alphabetically before the remainder's first, it joins the ordered
run. Measured on `ZEBRA`, whose alphabet is `ZEBRACDFGHIJKLMNOPQSTUVWXY`: the
top suggestion is `keyword=EBR start=B` with a tail of 23, and `ZEBRA` itself
comes back at rank 6 of 46, flagged as a rotated duplicate. The `extend`
option (default 3) offers one-letter-longer readings for this reason, and every
one of them regenerates the identical alphabet, so only the dictionary can
choose between them.

**One alphabet has a family of descriptions.** Cutting the same cyclic row one
place further along turns `SECRT|ABDF...YZ` into `Z|SECRT|ABDF...Y`, a
perfectly valid key that rebuilds the identical alphabet. 42 of the 46 ZEBRA
readings are such rotations. They are marked `rotated_duplicate` and ranked
below the un-rotated ones, because the longest-tailed reading is the one a
human is likely to have started from -- but as the ZEBRA case shows, that
heuristic can demote the reading the setter actually wrote.

**A climbed alphabet is wrong where the evidence is.** A hill climber recovers
common letters reliably and rare ones barely at all, so a recovered alphabet is
usually right except for a couple of rare letters -- enough to break the
ordered tail and shift the keyword by a letter or two. Read the suggestion as a
strong hint and confirm it by putting the keyword back through `decrypt`.

Recovering the keyword is worth doing even when the plaintext already reads:
in this competition the keyword is usually a word from the story, and it often
gives away the setting of the next part.

## General monoalphabetic substitution (`substitution.py`)

A monoalphabetic substitution replaces every occurrence of one plaintext
letter with one fixed ciphertext letter and never changes that choice, so the
key is a bijection of the alphabet onto itself. There are

```
26! = 403,291,461,126,605,635,584,000,000
```

such keys, which is why the cipher cannot be broken by trying them all and why
it is nevertheless broken in seconds. Caesar, Atbash, affine and keyword are
all special cases whose keys happen to be describable in a few characters;
this module knows nothing about that structure and solves them anyway, less
efficiently than their own solvers.

A `SubstitutionKey` maps **ciphertext letters to plaintext letters** -- the
direction a solver works in. Construction rejects both a cipher letter standing
for two plain letters and two cipher letters standing for the same plain
letter; the second is the one people forget, and rejecting it is what makes
crib fitting worth doing. Keys may be partial, and `apply` renders undecided
letters as `.` -- a placeholder that must not be a letter, so a guess cannot be
mistaken for a claim. `solve` reports `key=<26 letters>` as a decryption
alphabet that pastes straight back into `from_alphabet`.

### The objective

Single-letter frequencies are not enough: many wrong keys reproduce English
letter frequencies almost exactly, because they only have to permute the rare
letters among themselves. Four-letter statistics are much harder to fake, so
the climber maximises

```
sum over every 4-letter window of  log10 P(d | abc)
```

from `EnglishScorer.table()`. It deliberately ignores the first three letters,
which have no full context: they are a handful of terms out of hundreds and
leaving them out keeps the incremental arithmetic exact. This is *not* the
number a candidate reports -- `window_score` exposes it so a human can compare
two keys on the footing the search actually used.

### The climb

- **State**: a complete 26-letter key, held as `plain_of`, a list of 26 plain
  letter indices, one per cipher letter.
- **Move**: swap the plain letters assigned to two cipher letters. There are
  `26 * 25 / 2 = 325` such swaps.
- **Acceptance**: strictly greedy. Keep the swap if the score rises, otherwise
  undo it and move on. There is no annealing and no sideways move.
- **Local optimum**: a complete pass over all 325 pairs that improves nothing.
  No single swap helps. Record the key and stop.
- **Restart**: begin again from a fresh random bijection. `restarts` defaults
  to 25. The first climb starts from `start_key` if given, otherwise from
  `frequency_guess` (commonest ciphertext letter is E, next T, and so on down
  ETAOIN SHRDLU) completed deterministically; the rest are uniformly random
  bijections drawn from a private `random.Random(seed)`, so a seeded run
  reproduces and the global `random` module is never touched.

Restarts are the important part. A single climb often stalls with two or three
letters transposed, because escaping needs two simultaneous swaps and the
search only makes one at a time. Different starting points stall in different
places, so the evidence that the search converged is **not** the winner's
score -- a hill climber always returns something -- but **how many independent
restarts arrived at the same answer**. `CandidateSet` deduplicates on
(method, plaintext) and counts that as `agreements`.

Measured on 25 restarts of ordinary English prose:

| Letters | Recovered? | agreements | distinct local optima |
|---|---|---|---|
| 200 | yes | 4 | 24 |
| 300 | yes | 4 | 24 |
| 400 | yes | 9 | 18 |

Even when it works, the correct key is a minority result, and agreement grows
with the text. A solve where the best key was found once out of 25 deserves
far less trust than one where it was found nine times.

### Incremental rescoring

Rescoring the whole text for each of the 325 candidate swaps would cost
`325 * (n - 3)` table lookups per pass, and a solve makes hundreds of passes.
Nearly all of it recomputes windows that cannot have changed.

Swapping the plain letters of cipher letters X and Y changes the decryption
**only at positions holding an X or a Y**. A window starting at `p` covers
`p, p+1, p+2, p+3`, so a position `i` can only affect windows starting in
`[i-3, i]`. The climber precomputes, for each cipher letter, the set of window
starts it touches, and for each pair the union of the two sets. Scoring a swap
then costs `|union|` lookups instead of `n - 3`.

The "before" figure costs nothing: a running per-window score array is kept in
step with the current key, so the old score is a sum of cached floats. Only
"after" needs lookups. On acceptance the cache is refreshed over exactly those
windows and the running total is updated by `after - before`.

Measured on a 400-letter ciphertext: 397 windows, mean pair union 109
(median 102), so about **3.6 times fewer lookups** per swap test -- and much
better than that for the rare letters most of the 325 pairs involve. The ratio
holds at 800 letters, so the saving is a constant factor rather than a
vanishing one. Maintaining a total by addition of deltas is the classic place
for a silent bug, so `full_score` recomputes the objective from scratch and the
tests assert the two agree; they match to about 1e-13.

### Limitations

- **Short texts.** Below `RELIABLE_CLIMB_LETTERS = 150` the quadgram signal is
  too thin and the climb will confidently return a wrong key. `solve` records
  `short_text_warning` rather than hiding it, but it still returns an answer.
- **It optimises English-likeness, not correctness.** A key that scores well on
  gibberish is still gibberish. Read the plaintext.
- **Letters that never occur are never learned.** If a cipher letter does not
  appear in the text, nothing in the text says what it stands for. A swap
  between two such letters cannot change the decryption at all, so the climber
  skips it outright; and any plain letter such a cipher letter ends up holding
  came from the start key, not from evidence. `frequency_guess` leaves absent
  letters undecided and `_complete_deterministically` fills them in English
  frequency order, so the reported alphabet is a guess in exactly those
  positions -- which is what wrecks keyword recovery from a climbed alphabet.
- **It can simply fail.** Measured on a 362-letter text of repetitive prose,
  all 25 restarts reached distinct local optima, `agreements` was 1, and the
  best candidate was wrong and labelled `weak`. The diagnostics said so; that
  is the system working.
- **A wrong crib poisons everything.** Letters held by `fixed` are never
  swapped, so a wrong crib makes every restart wrong in the same way. The
  diagnostics report `held_fixed` for this reason.
- **No manufactured agreement.** When the text is shorter than one window, or
  `fixed` leaves fewer than two letters free, `searchable` is `False` and one
  attempt is run instead of 25 identical ones.
- **Nothing splits ciphertext on whitespace.** Competition ciphertext is
  printed in five-letter groups whose spacing carries no information about the
  plaintext. `analyse_words` takes its words explicitly and raises rather than
  accepting a single string to split.

# Part 3: Polyalphabetic ciphers

## Vigenere (`vigenere.py`)

### The cipher

A Vigenere cipher is a Caesar shift whose amount changes from letter to letter
according to a short repeating keyword. For a key of length `n`:

```
C_i = (P_i + K_{i mod n}) mod 26
P_i = (C_i - K_{i mod n}) mod 26
```

Encryption and decryption are the same loop with the sign flipped, which is
literally how `encrypt` and `decrypt` are written. Both work on letters only
and return uppercase letters only; the key advances on letters, never on
spaces or punctuation, and `clean_key` normalises the key exactly as
ciphertext is normalised, so `"le mon"` and `"LEMON!"` are the same key. A key
of all `A` adds zero every time and is the identity -- `_make_candidate` sets
an `identity_key` diagnostic when the search returns one, because that is the
solver telling you the text was never a Vigenere.

Why frequency analysis fails: a plaintext `E` becomes one of `n` different
ciphertext letters depending on where it falls, so the single-letter
distribution smears out and the Index of Coincidence drops from the English
0.0667 towards 1/26 = 0.0385. The whole game is therefore finding `n`.

### Stage 1: three independent estimates of the key length

The code implements three, and reports all three, because they fail in
different ways.

**1. Kasiski examination** (`kasiski_analysis`, built on
`statistics.kasiski_factor_votes`). If a repeated plaintext run happens to
line up with the same phase of the key, it enciphers identically, so the gap
between the two ciphertext repeats is a multiple of `n`. Every divisor from 2
to `max_key_length` of every consecutive gap gets a vote, weighted by run
length (`size - minimum_run + 1`, defaults `minimum_run = 3`,
`maximum_run = 12`). This is *structural* evidence -- it never looks at
English letter frequencies -- but it needs the text to be long enough to
contain lucky repeats.

**2. Index of Coincidence by period** (`ic_analysis`). Split into `p` columns
and average their ICs. A Caesar shift only relabels the alphabet, so it cannot
change IC; if `p = n` every column is a single Caesar shift of English and
reads about 0.0667, and if `p` is wrong the columns are mixtures and read near
0.0385. This is *statistical* evidence. Periods that would leave fewer than
`MINIMUM_COLUMN_FOR_IC = 20` letters per column are **not reported at all** --
IC on twelve letters is noise. So this method simply goes silent on long keys
over short texts.

**3. Column chi-squared fit** (`column_fit_analysis`). For each period, take
each column and try all 26 key letters, scoring each by chi-squared per letter
against English frequencies (`column_shift_fits`), and keep the best. The
implementation rotates a 26-entry count vector rather than building 26
decrypted strings, so trying all 26 shifts costs 26 operations regardless of
column length.

The raw winning chi-squared is *not* what the ranking uses. Chi-squared per
letter is inflated on small samples -- roughly `25/n` even for a perfect model
-- so raw values quietly favour short periods, whose columns are longer. Each
column is therefore also scored as a **ratio**: best chi-squared divided by
the average over all 26 shifts. For a genuine Caesar column one shift is
dramatically better than the field and the ratio is near 0; for a mixed column
nothing stands out and the ratio approaches 1. `worst_ratio` -- the worst
column's ratio -- is reported separately, because a key length has to explain
the *whole* text: five columns fitting and one not is evidence against the
period, and an average hides that where a maximum does not. Periods leaving
fewer than `MINIMUM_COLUMN_FOR_FIT = 8` letters per column are skipped.

### Combining and ranking the three

`estimate_key_lengths` scales each signal to roughly 0..1 and blends them:

```
kasiski_score = votes / top_votes
ic_score      = clip( (mean_ic - 0.0385) / (0.0667 - 0.0385), 0, 1 )
fit_score     = clip( 1 - mean_ratio,  0, 1 )
worst_score   = clip( 1 - worst_ratio, 0, 1 )

combined = 0.30*kasiski + 0.30*ic + 0.25*fit + 0.15*worst
```

A signal that could not be measured contributes **zero**, not a guessed value,
so a length resting on less evidence scores lower -- the conservative
direction to be wrong in. Whatever was actually measured is listed in
`evidence_available`. The blend is explicitly a presentation heuristic for
ordering a shortlist, not a probability.

**Why multiples must be demoted.** If the key really has length 4, then
splitting into 8 columns also works: each of those 8 columns is *half* of a
column that was already a single Caesar shift, so it is still a single Caesar
shift. Every multiple of the true length therefore looks good, and a table
listing 4, 8, 12 and 16 as four separate findings is reporting one fact four
times. So a length is demoted when some proper divisor of it (checked upwards
from 1, first match wins) scores within `DEMOTION_TOLERANCE = 0.05` of it. The
slack exists because the longer period's columns are shorter and noisier, so
it can edge ahead by luck. Demoted lengths are **kept** in the list and
flagged, not dropped, because occasionally the shorter period is the accident:
a keyword like `ABCABC` genuinely has period 3 and the toolkit should say so.
They simply sort below everything that is not a multiple.

`solve` then attacks the top `lengths_to_try = 4` entries of that ordering, so
the answer never rests on the estimator having got its first choice right.
`cipher_tool vigenere <file> --evidence` prints the table and stops.

### Stage 2: the column attack

Once `n` is known the cipher collapses. `normalize.columns(text, n)` splits
letter `i` into bucket `i mod n`, so column `j` contains exactly the letters
enciphered by key letter `K_j` -- one fixed shift, i.e. a Caesar cipher. `n`
columns, `n` independent Caesars, each solvable by trying all 26 shifts.

Worked example, `ATTACKATDAWN` under `LEMON`:

```
plaintext   A  T  T  A  C  K  A  T  D  A  W  N
key         L  E  M  O  N  L  E  M  O  N  L  E
ciphertext  L  X  F  O  P  V  E  F  R  N  H  R
position    0  1  2  3  4  5  6  7  8  9  10 11
```

Split the ciphertext `LXFOPVEFRNHR` into 5 columns:

```
column 0 (key L): positions 0, 5, 10  ->  L V H
column 1 (key E): positions 1, 6, 11  ->  X E R
column 2 (key M): positions 2, 7      ->  F F
column 3 (key O): positions 3, 8      ->  O R
column 4 (key N): positions 4, 9      ->  P N
```

Every letter in column 0 was shifted by `L`, and by nothing else. On a real
ciphertext each column would hold dozens of letters and
`best_shift_for_column` picks the shift whose decrypted column has the
smallest chi-squared distance from English; `solve_key_for_length` does this
for all `n` columns and spells the key out of the winners.

### The refinement pass

`refine_key` walks the key one position at a time, tries all 26 letters there,
and keeps whatever maximises the n-gram score of the **whole decrypted text**,
repeating until a full pass changes nothing or `max_rounds = 4` passes have
run.

This catches what chi-squared cannot. Chi-squared judges a key letter using
only its own column's 26 letter counts -- perhaps twenty letters of evidence.
At that sample size two shifts easily produce similar-looking frequency
profiles and the wrong one wins on noise. The n-gram model asks a completely
different question: does the decryption read like English *in sequence*? The
letters flanking any position belong to *other* columns, so the model is using
cross-column information that the per-column test cannot see. One wrong key
letter out of six leaves every sixth letter wrong, which wrecks the quadgram
score even while that column's own frequency profile looks passable. The cost
is `26n` decryptions per pass rather than the `26^n` of a full search.

### Exhaustive search for short keys

`brute_force_length` scores every one of the `26^length` keys: 26 at length 1,
676 at 2, 17,576 at 3. `solve` runs it for lengths 1 to `brute_force_up_to`
(default 3) *as well as* the statistical attack, because exhaustive search
cannot be fooled by a short ciphertext where chi-squared has nothing to work
with. Ranking is done on the first `BRUTE_FORCE_SAMPLE = 200` letters with a
stripped-down quadgram scorer that skips the first three letters; the handful
of survivors are rescored on the full text before anything is reported.
Lengths above `MAXIMUM_BRUTE_FORCE_LENGTH = 4` are **refused with an error**,
not attempted: `26^5 = 11,881,376` keys is not a search, it is a hang.

### Honest limitations

- The key-length ceiling is `MAXIMUM_KEY_LENGTH = 20`, and
  `estimate_key_lengths` further caps at `len(letters) // 2`. A longer key is
  never considered and the solver cannot tell you that is what happened.
- Unmeasured evidence counts as zero. On a short text, IC may be measurable at
  period 3 but not at period 6, so a true period of 6 can be scored down and
  demoted behind 3 purely because one of its three signals was unavailable.
- Kasiski needs lucky repeats; short coincidental trigram repeats add noise
  that the length weighting only partly suppresses.
- `refine_key` is a coordinate-wise hill climb. It can stall at a local
  maximum, and the returned `Refinement` reports exactly what changed rather
  than claiming correctness.
- The combined score orders a shortlist. It is not a probability, and the
  solver deliberately tries several lengths regardless of it.

## Beaufort and variant Beaufort (`beaufort.py`)

### Three rules, side by side

All three periodic ciphers repeat a short key under the plaintext and add or
subtract modulo 26. They differ only in the direction of the arithmetic:

```
Vigenere          C_i = (P_i + K_i) mod 26      P_i = (C_i - K_i) mod 26
Beaufort          C_i = (K_i - P_i) mod 26      P_i = (K_i - C_i) mod 26
variant Beaufort  C_i = (P_i - K_i) mod 26      P_i = (C_i + K_i) mod 26
```

**Beaufort is self-reciprocal.** Substitute its rule into itself:

```
encrypt(encrypt(P)) = K - (K - P) = P   (mod 26)
```

so encrypting a Beaufort ciphertext a second time with the same key returns
the plaintext. `beaufort_encrypt` and `beaufort_decrypt` are the same call to
`_combine(text, key, "key_minus_text")`; the two names exist only so calling
code reads honestly. That is why the cipher was convenient on a Royal Navy
slide rule -- one routine, both jobs.

**Variant Beaufort is not**, and its encryption rule `C = P - K` is *exactly
Vigenere decryption*. Its own decryption `P = C + K` is exactly Vigenere
encryption. The module provides `vigenere_encrypt` / `vigenere_decrypt`
alongside for direct comparison.

Worked example, `ATTACK` under `LEMON`:

```
Vigenere          ATTACK -> LXFOPV
Beaufort          ATTACK -> LLTOLB     and LLTOLB -> ATTACK with the same key
variant Beaufort  ATTACK -> PPHMPZ     which is Vigenere-decrypt(ATTACK, LEMON)
```

### The attack, and why the column rule matters

Key length is found the same way as for Vigenere: any bijection of the
alphabet -- a shift, a reflection, or both -- relabels letters without merging
them, so IC is untouched, and Kasiski works because a repeated plaintext run
at the same key offset still gives an identical ciphertext run whichever
direction the arithmetic runs.

But the rule *inside* a column is not the same:

```
Vigenere          P = C - K     a shift of the column by -K
variant Beaufort  P = C + K     a shift of the column by +K
Beaufort          P = K - C     a REFLECTION of the column, then +K
```

The Beaufort map `C -> K - C` reverses the alphabet as well as moving it. A
solver that assumes "each column is a Caesar shift, so try all 26 shifts and
take the best chi-squared" finds **nothing** on Beaufort ciphertext, because
no shift of the column produces English until the reflection is undone. So
`column_key_letter` derives the column rule separately per variant rather than
reusing one shift solver, and `tests/test_beaufort.py` contains the experiment
that observes the wrong solver failing.

`column_key_letter` returns the winning key value, its chi-squared, *and the
margin over the runner-up*. A column whose best and second-best key letters
score almost the same has not been solved, it has been guessed, and
`derive_key` averages those margins into `column_margin_mean` so the report
says so.

`solve` then: (1) derives one chi-squared key per `(variant, length)` pair;
(2) scores each decryption with the order-3 model; (3) hill-climbs the best
`refine_top = 5` of them letter by letter under the n-gram score, sweeping
until a whole sweep changes nothing; (4) reports every length tried for both
variants in one ranked list, with the IC at that period and the Kasiski votes
attached as independent evidence.

### Honest limitations

- **The solver does not use the key-length evidence to choose what to try.**
  It brute-forces every length from `min_key_length = 1` to
  `max_key_length = 20` for both variants; IC and Kasiski are computed once
  and attached to the diagnostics for a human to check, nothing more. That is
  40 chi-squared key derivations on any input, and it means a wrong length
  that happens to score well is ranked purely on its plaintext score.
- Only the top 5 attempts are hill-climbed, so a correct key length whose
  chi-squared pass scored badly may never be polished into a readable answer.
- The hill climb is steepest-ascent over one letter at a time and can stall at
  a local maximum.
- `diagnostics["meets_english_threshold"]` is true only when the per-letter
  n-gram score beats `ENGLISH_THRESHOLD = -1.80`. When it is false, nothing
  has been solved whatever the ranking says.
- CLI wrinkle: `--variant` only selects the variant for `--key` encryption and
  decryption. When searching, `cipher_tool beaufort` always attacks **both**
  variants and `--variant` is ignored.

## Autokey: plaintext and ciphertext forms (`autokey.py`)

### The idea, and the two forms

A repeating key is what kills Vigenere: it makes the ciphertext `m`
interleaved Caesar shifts, and Kasiski plus IC find `m` without guessing.
Vigenere's own 1586 fix was to stop repeating -- start the key with a short
**primer** and continue it with the message. The two classical forms differ in
*which* message continues it, and they behave completely differently under
attack.

```
plaintext autokey    K = primer + P
                     K_i = primer_i   for i < m ;   K_i = P_{i-m}   for i >= m
                     C_i = (P_i + K_i) mod 26

ciphertext autokey   K = primer + C
                     K_i = primer_i   for i < m ;   K_i = C_{i-m}   for i >= m
                     C_i = (P_i + K_i) mod 26
```

Worked example, `ATTACK` under primer `KEY`:

```
plaintext autokey    key stream K E Y A T T   ->  KXRAVD
                     (positions 3,4,5 take their key from P_0,P_1,P_2 = A,T,T)

ciphertext autokey   key stream K E Y K X R   ->  KXRKZB
                     (positions 3,4,5 take their key from C_0,C_1,C_2 = K,X,R)
```

### Decryption

Ciphertext autokey decrypts in one pass with no bookkeeping: every key letter
after the primer is a ciphertext letter and the receiver has the whole
ciphertext in front of them from the start.

The plaintext form cannot do that. The key letter for position `i` is the
plaintext letter from position `i - m`, which the receiver does not have until
they have decrypted that far. So `plaintext_autokey_decrypt` **builds its own
key as it goes**, reading its own output back in as key material:

```
positions 0 .. m-1   use the primer, the only key we were given
positions m onwards  use plain[position - m], a letter this very loop
                     produced a moment ago
```

Checking the example above: `K - K = A`, `X - E = T`, `R - Y = T`, then
`A - plain[0] = A - A = A`, `V - plain[1] = V - T = C`, `D - plain[2] = D - T = K`.

The consequence is error propagation: one wrong letter poisons every `m`-th
letter after it. That is a nuisance for a legitimate receiver and a gift to
the cryptanalyst, because a nearly-right primer still produces obvious
nonsense rather than a near miss.

### Attacking the ciphertext form -- fatally weak

Substitute `K_i = C_{i-m}` into decryption:

```
P_i = (C_i - K_i) = (C_i - C_{i-m}) mod 26     for i >= m
```

The primer has vanished. Every plaintext letter after the first `m` is a
difference of two ciphertext letters and needs no key at all
(`_ciphertext_tail`), so the only real unknown is `m`, found by trying each
value and scoring the result.

The opening `m` letters are then chosen by hill climb, and the primer read off
as `primer_i = (C_i - P_i) mod 26`. The search runs **right to left**, and the
direction matters: the letter immediately before the forced tail sits inside
quadgrams that are otherwise entirely known, so the model has real evidence
about it, whereas the first letter has almost none until its neighbours settle.
Measured on this toolkit's corpus, a left-to-right sweep settled on the opening
`ETWOSTEVERY` where the true `ALMOSTEVERY` scored better.

Be clear about the asymmetry, which the diagnostics state explicitly: from
position `m` onwards the plaintext is *forced* and independent of the primer;
the opening `m` letters are only the language model's preferred reading. Any
opening whatsoever is consistent with some primer, so those letters are not
recoverable by cryptanalysis at all. Measured on this toolkit's corpus, a
420-letter message with a six-letter primer decrypted perfectly from position
six while the model preferred the opening `ENINEL` to the true `ALMOST`. A
candidate can carry a completely correct message and a wrong primer.

### Attacking the plaintext form -- genuinely harder

Substituting `K_i = P_{i-m}` gives a recurrence, not a formula:

```
P_i = (C_i - P_{i-m}) mod 26
```

Guess the `m` primer letters and the whole message unrolls, so the cipher is
exactly as strong as the primer is long. Two structural facts drive the
attack:

**The message splits into `m` independent chains.** Positions `j, j+m, j+2m,
...` depend only on primer letter `j`, so each primer letter can be tested
alone (`_chain_values`). A chain is ordinary English sampled every `m` letters,
which leaves single-letter frequencies untouched, so chi-squared on each chain
gives a usable first guess at every primer letter independently
(`initial_primer`), together with the margin over the runner-up.

**A wrong primer letter alternates sign down its chain.** With a wrong guess
`g' = g + e`:

```
P'_j      = P_j      - e
P'_{j+m}  = P_{j+m}  + e
P'_{j+2m} = P_{j+2m} - e
```

The error flips sign at every step, so a wrong guess does not shift a chain
uniformly the way a wrong Caesar key would -- it produces a text no single
shift can repair. This is precisely why the ordinary per-column Vigenere
attack finds nothing on autokey ciphertext.

`solve` then tries, per primer length: the per-chain chi-squared guess; an
exhaustive pass over all `26^m` primers whenever `26^m <= exhaustive_limit`
(default 20,000, so **only lengths 1 to 3**), screened on a decrypted prefix
of `probe_length = 60` letters with the best 12 rescored in full; and finally
a hill climb over the primer under the whole-text n-gram score, applied to the
best `climb_top = 4` lengths, with `restarts = 2` seeded random restarts to
escape a local maximum. `diagnostics["search"]` records every stage that ran,
not only the one that won.

### Honest limitations -- these attacks fail often

- **There is no key length to find.** Kasiski and IC, which do most of the
  work against Vigenere, say nothing useful here, so the search leans entirely
  on the English language model.
- The search is exhaustive only for primers of length 1 to 3. Beyond that it
  is a hill climb, which can stop at a wrong primer that scores well.
- A primer longer than `max_primer` is never tried and the solver **cannot
  tell you that is what happened** -- it only records `max_primer_tried`. Note
  the two defaults differ: `autokey.solve` defaults to 8, but the CLI's
  `--max-primer` defaults to **6**.
- Under roughly 200 letters the n-gram score of a wrong primer is often as
  good as the right one, and the solver will confidently rank nonsense first.
  A `short_text_warning` diagnostic is attached below 200 letters.
- `seed` defaults to `None`, so the random restarts are not reproducible
  unless you pass `--seed`.
- As with Beaufort, treat a candidate as unsolved unless
  `meets_english_threshold` (per-letter n-gram score above `-1.80`) is true
  *and* the plaintext actually reads.

# Part 4: Transposition ciphers

## The family tell: letters moved, not changed

Every cipher in this part is a **permutation of the plaintext letters**. No
letter is ever replaced, so the ciphertext contains exactly the same multiset
of letters as the plaintext. Two consequences follow, and together they are
the whole reason this family is recognisable:

- The Index of Coincidence is **unchanged**, so it stays at the English value
  of about `0.067`.
- The chi-squared distance from English, `statistics.chi_squared_english`, is
  **unchanged too** -- not merely similar, but bit for bit identical, because
  it is computed from the letter counts alone.

Measured on a 182-letter sample enciphered with a columnar key:

```
plaintext   IC = 0.0634   chi2/letter = 0.2021
ciphertext  IC = 0.0634   chi2/letter = 0.2021
```

Contrast that with substitution (Part 2). A monoalphabetic substitution also
leaves IC alone -- it relabels letters consistently -- but it *scrambles the
distribution*, so chi-squared goes large. That is the split Part 1's table
makes:

```
IC ~0.067  chi2 small   ->  transposition   (moved, not changed)
IC ~0.067  chi2 large   ->  monoalphabetic  (changed, not moved)
IC ~0.040  chi2 large   ->  polyalphabetic / digraphic
```

`statistics.cipher_family_hypotheses` turns that into the concrete test
`chi < 0.12 and ic > 0.058`, labelled `likely` at 100 letters or more and
`possible` below. So: **unreadable text whose letter statistics are flawless
English is a transposition until proved otherwise.**

One presentation detail applies to all the transposition modules. Every
solver here sets
a candidate's `display` to `group_text(plaintext)` -- plain five-letter groups
-- and deliberately *not* `NormalizedText.relayout()`. Relayout pours
plaintext letter `i` back into the position of ciphertext letter `i`, which is
meaningful for a substitution and meaningless here, because a transposition
moved the letters. Inventing a word layout would be inventing evidence.

## Rail fence (`rail_fence.py`)

### What it does

Write the plaintext downwards across `rails` imaginary rails, bouncing off the
top and the bottom, then read the rails off in order.

```
rails = 3, plaintext ATTACKATDAWN

    A . . . C . . . D . . .
    . T . A . K . T . A . N
    . . T . . . A . . . W .

    ciphertext = ACD + TAKTAN + TAW = ACDTAKTANTAW
```

### The cycle length

Going down all the rails and back up visits the top and bottom rails once each
and every middle rail twice, so the walk repeats after

```
rails + (rails - 2) = 2 * rails - 2
```

steps. That is `cycle_length(rails)`. The rail used at step `t` therefore
depends only on `t` modulo the cycle:

```
p = (t + offset) mod (2 * rails - 2)

rail(t) = p                    if p < rails      (travelling down)
          (2 * rails - 2) - p  otherwise         (travelling back up)
```

The second line is a reflection about the bottom rail. `rail_sequence` is that
one loop, and encryption, decryption and the attack are all bookkeeping on top
of it. `rail_sequence(6, 3)` is `[0, 1, 2, 1, 0, 1]`.

### The offset variant

`offset` starts the walk part way through the zigzag, which is a genuine extra
key: with 3 rails and offset 1 the first letter lands on the middle rail
already heading down, and the whole pattern of rail lengths changes. Because
the walk is periodic, `_normalise_offset` reduces the offset modulo the cycle
length -- that is an identity of the cipher, not a guess. Negative offsets and
non-integers (including `bool`) are rejected outright.

### Why decryption is the fiddly half

Encryption is one pass: drop each letter into its rail's bucket, concatenate
the buckets. Decryption cannot start until you know **how many letters landed
on each rail**, because the ciphertext is the rails laid end to end and the
rails are not equal lengths. `rail_counts` replays the walk and tallies it,
rather than using a closed form -- the counts depend on the length, the rail
count *and* the offset, and a partial final cycle skews the tail.

Worked example, `CIPHER` with 3 rails, offset 0:

```
walk       C  I  P  H  E  R
rail       0  1  2  1  0  1

rail 0 : C E        (2 letters)
rail 1 : I H R      (3 letters)
rail 2 : P          (1 letter)

ciphertext = CE + IHR + P = CEIHRP
```

To decrypt `CEIHRP` you must first compute `rail_counts(6, 3, 0) = [2, 3, 1]`,
*then* slice `CE | IHR | P`, *then* walk `0,1,2,1,0,1` taking the next unused
letter from each rail: `C, I, P, H, E, R`. Slice it into three equal pieces
instead and you get a confidently wrong answer rather than an obviously wrong
one. With offset 1 the same word gives counts `[1, 3, 2]` and ciphertext
`HCPEIR` -- same letters, different cut points.

### The attack

`solve` is exhaustive. It tries every rail count from 2 to
`min(length - 1, max_rails)` -- `max_rails` defaults to `DEFAULT_MAX_RAILS =
20` -- and every offset in `range(cycle_length(rails))`. That is
`sum(2r - 2) for r in 2..20 = 380` decryptions, measured at 0.35 s for a
182-letter text. Each is scored with the English model and the top 5 returned,
with `rails`, `offset`, `cycle_length` and the `rail_lengths` in the
diagnostics.

Options: `max_rails`, `time_budget` (stops cleanly, sets `time_budget_hit`),
and `seed`, which is accepted and **ignored** -- there is no randomness -- so a
caller can pass one option set to every solver.

**Limitations.** Texts under 3 letters return an empty set. Rail counts above
20 are never tried; above that most rails hold one or two letters, which is
barely an encryption, but if a setter used 25 rails we would miss it. On short
texts several offsets can score within noise of each other, so the returned
list really is a ranking, not a verdict.

## Columnar transposition (`columnar.py`)

### What it does

Write the plaintext into a grid row by row under a keyword, then read the
columns out in the alphabetical order of the keyword's letters.

```
keyword  Z E B R A S
column   0 1 2 3 4 5

         W E A R E D
         I S C O V E
         R E D F L E
         E A T O N C
         E

read order  A(4) B(2) E(1) R(3) S(5) Z(0)
ciphertext  EVLN + ACDT + ESEA + ROFO + DEEC + WIREE
```

### key_order: keyword to permutation

`key_order(keyword)[j]` is the grid column read out `j`-th.
`key_order("ZEBRAS") == (4, 2, 1, 3, 5, 0)`. Everything outside A-Z is stripped
by `clean_key` first, so `"Zebra's!"` and `"ZEBRAS"` are the same key, and a
key reducing to fewer than two letters is an error.

Repeated keyword letters are a real ambiguity, and the module states its
convention instead of leaving it implicit: ties break **left to right**, so
`key_order("BANANA") == (1, 3, 5, 0, 2, 4)` -- the A at position 1 is read
before the A at position 3, and both before the B. This is enforced by sorting
on `(letter, position)` rather than relying on Python's sort being stable.
`keyword_from_order` inverts the process to a canonical keyword for reporting,
so a recovered permutation `(4, 2, 1, 3, 0)` prints as `ECBDA`.

### The ragged last row

This is where almost every implementation is wrong. `column_lengths(length,
count)` computes

```
rows      = ceil(length / count)
remainder = length mod count

column c is long  (rows letters)      when c < remainder
column c is short (rows - 1 letters)  otherwise
```

with every column equal when `remainder == 0`. The leftover letters go in the
**leftmost** columns because the last row is filled left to right like every
other row. `column_lengths(25, 6) == [5, 4, 4, 4, 4, 4]`.

Crucially this is indexed by **grid** position, not by read order. Decryption
must therefore go in this order and no other:

1. `column_lengths` gives the length of each grid column.
2. The key says grid column `order[j]` was written out `j`-th, so the `j`-th
   block of ciphertext has the length of grid column `order[j]`. **A short
   block can appear anywhere in the ciphertext, not only at the end.**
3. Only now can the ciphertext be sliced and the grid read back row by row,
   skipping columns with no letter on the last row.

Worked example on a length that is not a multiple of the key length --
`ATTACKATDAWNNOW`, 15 letters, keyword `KEYS`:

```
KEYS -> order (1, 0, 3, 2)          column_lengths(15, 4) = [4, 4, 4, 3]

  col   0 1 2 3
        A T T A
        C K A T
        D A W N
        N O W .          <- column 3 is short

read col1, col0, col3, col2:
        TKAO  ACDN  ATN  TAWW
ciphertext = TKAOACDNATNTAWW
```

The short 3-letter block is the **third** chunk of ciphertext, because grid
column 3 happens to be read third. Cut the ciphertext into equal blocks and
hope, and you get `TKAO | ACDN | ATNT | AWW` -- every letter after the eighth
is wrong. That bug passes any test whose length happens to divide exactly,
which is exactly why it survives in so much code.

### Complete columnar and double columnar

`encrypt(..., complete=True)` pads with `filler` (default `"X"`) to a whole
number of rows first, so every column is the same length and the ragged
problem disappears -- for the sender, and also for the codebreaker, who loses
a clue. `ATTACKATDAWNNOW` under `KEYS` with `complete=True` gives
`TKAOACDNATNXTAWW`, 16 letters. On the way back, `complete=True` **asserts**
the rectangle is full: a length that does not divide the column count raises
rather than silently falling back to a ragged grid. `strip_filler` is off by
default, because a plaintext is perfectly entitled to end in X and `rstrip`
cannot tell the difference.

`encrypt_double(text, first_key, second_key)` enciphers, then enciphers the
result under a second key. Two passes destroy the neat column structure that
makes one pass breakable: letters that were neighbours in a plaintext row are
no longer a fixed distance apart, so the column-pair statistics below have
nothing to lock onto. With `complete=True` the text is padded **once**, to a
multiple of `lcm(len(first), len(second))`, so that both rectangles come out
full. Padding at each pass separately would be wrong -- the second pass would
be padding the *ciphertext* of the first, and stripping that on the way back
shifts every letter of the first-pass grid.

```
ATTACKATDAWNNOW  keys KEYS then CAB   ->  KANNWACATWTODTA
same, complete=True (15 padded to 24) ->  KXCXTXAXAXDXNXWXTOANAXTW
```

**`solve` attacks a single pass only, and must not be pointed at two.** What
it returns for a double transposition is its best single-pass reading of
something that never was one -- measured, that came back labelled
`promising`, which is the worst possible combination of wrong and confident.
Two passes have their own solver, `solve_double`, described below.

### Attacking two passes

`decrypt_double` needs both keys; `solve_double` searches for them. It cannot
reuse anything above, and the reason is in `encrypt_double`'s own docstring:
after the second pass, letters that were neighbours in a plaintext row are no
longer a fixed distance apart, so the column-pair statistics that make the
single-pass attack cheap have nothing to lock onto. The only signal left is
the score of the finished plaintext, so both permutations are searched
together by simulated annealing, paying a full rescore per step.

Three things about it were measured rather than assumed, and each one changed
the design:

- **Annealing, not hill climbing, and swaps are not enough.** A key that is
  right except for one column sitting one place too early needs every later
  column to shift, which no single swap achieves and a chain of swaps reaches
  only through worse-scoring intermediates. Adding a lift-and-reinsert move
  makes that whole family of near misses one step away.
- **Diversification beats persistence.** On a 7x6 pair over 400 letters, six
  restarts of 30,000 steps returned a near miss; twelve restarts of the same
  length found the key, while six restarts of 80,000 steps cost two and a
  half times as much and did not. Hence twelve restarts by default.
- **Screen every shape before searching any of them deeply.** Searching each
  length pair to full depth in turn spent a 40 second budget on four of
  twenty-five shapes, and the answer was in the ninth. A short screening run
  per shape, then the full search on the best three, found the same key in 21
  seconds. A wrong shape cannot produce English however long it is climbed,
  which is what makes a short screen enough to rank one.

Shapes are also tried in order of how likely a length is to be a real
keyword -- four to nine first, two last -- rather than cheapest first.
Cheapest-first optimises the time taken to fail.

**This search is never exhaustive and every candidate says so.** Two
permutations of eight columns are 40,320 squared; nothing here can be
enumerated, and a run that finds nothing is not evidence that there is
nothing to find.

### The attack

Only the order of the letters is unknown, so the search is over permutations
of the columns. The scoring insight is that **neighbouring grid columns sit
next to each other on every row**. Cut the ciphertext into blocks, and for
each ordered pair `(x, y)` ask how English the down-the-page pairs
`x[0]y[0], x[1]y[1], ...` look:

```
side[x][y] = sum over rows i of  log10 P(y[i] | x[i])
wrap[x][y] = sum over rows i of  log10 P(y[i+1] | x[i])

score(arrangement) = wrap[last][first] + sum of side along the chain
```

`side` is "does block y read as the column immediately right of x". `wrap` is
the join from the end of one row to the start of the next -- the only term
tying the two ends of the arrangement together. Sums stop at the shorter
block, which is correct because a short column is short at the *bottom* only.

The bigram table is derived from the scorer's own public interface, not a
second table, so the filter and the ranking agree about English:
`score_values([a, b]) - score_values([a])` is exactly `log10 P(b | a)`. It
costs 702 scorer calls and is `lru_cache`d.

The module records its own measurement of whether `wrap` earns its place: over
175 short samples (40 to 200 letters, 2 to 7 columns) the exhaustive search
put the true arrangement first **163 times with the wrap term and 145 without**.

Scoring an arrangement is thus `count` additions rather than several hundred
letter-scorings, which is what makes exhaustive search affordable. Two
subtleties are handled rather than ignored:

**Which blocks are long?** With a ragged last row the block lengths depend on
which *grid* column each block came from -- the very thing being searched for.
So the search enumerates the `C(count, remainder)` patterns of long and short
blocks; inside a pattern the boundaries are fixed and the adjacency matrices
can be built once. Summed over patterns this is exactly `count!` arrangements,
none counted twice.

**The feasible limit is 9 columns.** `DEFAULT_MAX_EXHAUSTIVE = 9`, set equal
to `DEFAULT_MAX_KEY_LENGTH` so that every key length the solver tries by
default is enumerated in full (9! = 362,880 arrangements, 0.28 s). Beyond that
`solve` falls back to `_greedy_chains` -- start from each
possible block, repeatedly append whichever unused block scores best beside
the current tail -- then hill-climbs with `_improve`, taking the single best
pairwise swap until no swap helps, plus `DEFAULT_RESTARTS = 8` randomised
restarts per pattern. Swaps are only allowed within a class, long with long
and short with short, since a long block cannot sit where the ragged row
leaves a short column.

Either way the adjacency score is **only a filter**. The best
`DEFAULT_REFINE = 15` arrangements per key length are decrypted properly and
re-scored with the full English model, so the ranking never depends on the
shortcut. `DEFAULT_MAX_KEY_LENGTH = 9` is the library default sweep, and
diagnostics report `search: exhaustive` or `greedy (not exhaustive)` per grid.
The command line exposes both ceilings as `--max-key-length` and
`--max-exhaustive`.

**Limitations, measured.**

- Greedy is much weaker than exhaustive, and this is why the default no
  longer uses it for any length the solver sweeps. On a 181-letter text with
  a 9-column key, twelve seeds put the true key first only **5 times out of
  12**, and on the other 7 runs it was not in the returned 15 at all.
  Exhaustive search finds it every time for 0.28 s. Greedy now applies only
  to key lengths above 9, where enumeration is genuinely infeasible
  (10! = 3.6 million); treat a greedy first place with suspicion, and check
  `search` in the diagnostics to see which path produced your answer.
- `GREEDY_PATTERN_LIMIT = 60` caps the long/short patterns tried in greedy
  mode, and it takes the **lexicographically first 60** of
  `combinations(range(count), remainder)`, which biases towards patterns whose
  long blocks come early in the ciphertext. The `tried/total` count is always
  in the diagnostics -- read it.
- `complete=True` restricts the sweep to column counts dividing the length,
  which is a strong constraint; the length's divisors from 2 to 30 are always
  reported (`plausible_column_counts`) so you can apply it by eye.
- Column counts leaving fewer than two rows are skipped, and texts under 4
  letters return an empty set.
- The solver never strips filler, so a recovered complete columnar still
  carries its trailing Xs.

## The permutation cipher (`permutation.py`)

### What it does

Cut the plaintext into blocks of a fixed size and rearrange the letters
inside each block the same way every time. With the key `BAEDC`, whose
letters sort to `ABCDE` and so give the read order `(1, 0, 4, 3, 2)`:

    plaintext   I N F I L | T R A T I | N G T H E
    read order  1 0 4 3 2 | 1 0 4 3 2 | 1 0 4 3 2
    ciphertext  N I L I F | R T I T A | G N E H T

Only positions change, never identities, so the ciphertext carries exactly
the letter frequencies of the plaintext: an English index of coincidence and
a tiny chi-squared over text that reads as nonsense.

### Why it is not a columnar transposition

It looks like one and it is not, and the difference is the whole reason the
module exists. A columnar transposition writes the message into a grid and
reads out one WHOLE COLUMN at a time, so a letter at plaintext position 3 can
end up thousands of places away. A permutation cipher never moves a letter
out of its own block, so displacement is bounded by the block size. No
columnar key describes that, at any width.

This was not reasoned out in advance. It came from running the toolkit
against the public archive of past National Cipher Challenges, which is the
one test that cannot be graded on a curve of the tool's own encryptions. The
2018 challenge 6B message, 2,142 letters, was reported `weak` by every
transposition attack in the toolkit -- columnar to width 63 including every
complete-rectangle divisor of 2,142, double columnar over all 64 shapes to
9x9, rail fence, and every route and grid. The message is a period-5
permutation under the key `BAEDC`.

### The attack

The same shape as the columnar attack, and it reuses that code rather than
restating it. Take the letters at block offset `a` of every block as a
"stripe". Scoring "stripe y sat immediately after stripe x" is then the same
sum the columnar solver already computes for neighbouring columns:

    side[x][y] = sum over blocks b of log P( stripe_y[b] | stripe_x[b] )

with a second matrix for the join from the end of one block to the start of
the next. If y really did follow x, every one of those pairs is a genuine
English bigram and the sum is far better than any wrong pairing.

That is what makes exhaustive search affordable. Ranking a permutation costs
`n` table lookups against an `n` by `n` matrix, not a decryption and a
rescoring of the whole message, so every one of the 40,320 arrangements of
eight positions is tried in well under a second. Block sizes above eight fall
back to restarts and hill climbing, and the candidates say which they got.

The hill climb uses two neighbourhoods, not one. Swaps alone search a
permutation badly: an arrangement that is right except that one position sits
a single place too early needs everything after it to shift along, and no
swap of two positions does that. Lifting one position out and reinserting it
elsewhere does it in a single move.

### Two details that are easy to get wrong

**The ragged last block is permuted too.** A short final block takes the
key's entries that still point at a real letter, in their original order --
the only reading that stays a permutation and the only one that agrees with
the full-block case. It matters: the 2018 6B plaintext ends CONSTANTINOPLE,
and its last two letters are a block of two. Leaving short blocks alone
spells it CONSTANTINOPEL, which is right for 2,140 letters and wrong for the
two a reader looks at last.

**The identity permutation is never offered.** It would "decrypt" any text to
itself, so it would top the ranking for every piece of plain English ever
pasted in -- a confident answer to a question nobody asked.

### Confidence, measured rather than argued

Swept over five DIFFERENT texts at each of five block sizes and six lengths
from 60 to 500 letters: 150 runs, the exact plaintext in every one, and no
case of a wrong answer labelled `strong`. So no confidence cap is applied
here. That is the opposite of the Playfair result, and the reason is that a
near-miss means different things in the two ciphers: Playfair degrades into
fluent near-English, while a wrong block permutation is noise the score
catches. A cap this solver does not need would only weaken correct answers.

### Honest limitations

- Block sizes above eight are searched, not enumerated, and say so.
- A permutation cipher whose block size exceeds the ceiling (12 by default)
  is not attempted at all.
- Two blocks of text is the minimum; below that the adjacency sums have
  nothing to work with and the solver returns nothing rather than guessing.

## Route and grid transposition (`transposition.py`)

### What it does

Write the plaintext into a rectangle along one path through the cells, read it
out along another. The key is the pair of paths plus the shape.

```
ATTACKATDAWN in a 3x4 grid, filled row by row:

    A T T A
    C K A T
    D A W N

rows                   ATTACKATDAWN     (the fill itself)
columns                ACDTKATAWATN
boustrophedon_rows     ATTATAKCDAWN     (odd rows reversed)
diagonals              ATCTKDAAATWN     (anti-diagonals, each downwards)
diagonals_alternating  ACTTKDAAATWN     (same, alternately down and up)
spiral_cw_top_left     ATTATNWADCKA
```

A route is stored as the function producing its cell ordering, so the cipher
is the composition of two permutations:

```
ciphertext[j] = plaintext[ F^-1(R[j]) ]        F = fill route, R = read route
```

Since the inverse of `R . F^-1` is `F . R^-1`, **decrypting "filled along F,
read along R" is the same operation as encrypting the swapped pair.** That is
what `both_directions` (default `True`) exploits: only four plain fills are
tried, and swapping the pairs covers a sender who wrote along a spiral and
read along the rows.

### The routes actually implemented

`ROUTES` holds 15: `rows`, `columns`, `reverse`, `boustrophedon_rows`,
`boustrophedon_columns`, `diagonals`, `diagonals_alternating`, and eight
spirals -- `spiral_cw_*` and `spiral_acw_*` from each of `top_left`,
`top_right`, `bottom_right`, `bottom_left`. `FILL_ROUTES` is only four:
`rows`, `columns`, `boustrophedon_rows`, `boustrophedon_columns`. With
`both_directions` that gives **100 ordered pairs** per shape (56 without).
`cipher_tool transposition --routes` prints the table.

Each route is written as a plain enumeration rather than clever arithmetic,
because a route that visits a cell twice or misses one corrupts a decryption
silently; the spiral additionally carries an unreachable guard that raises if
it ever becomes trapped.

### grid_shapes and the ragged caveat

`grid_shapes(length, min_side, max_side, allow_ragged=...)` proposes the
rectangles. With `allow_ragged=False` these are exactly the factorisations of
the length, taken from `statistics.divisors` -- so a **prime length returns
nothing at all**, which is an honest and useful answer: the sender must have
used a ragged grid or a different cipher. With `allow_ragged=True` one shape
per column count is added, `rows = ceil(length / cols)`, since any taller grid
would leave a whole empty row. `DEFAULT_MIN_SIDE = 2`, `DEFAULT_MAX_SIDE = 40`,
widened to `isqrt(length) + 1` for texts too long for any 40-sided grid.

The caveat that matters: **where the blank cells sit on a ragged grid is a
convention, not a fact.** Filled row by row the blanks are obviously the end
of the last row; filled along a spiral, a sender might leave them at the end
of the spiral, or at the bottom right, or pad the message instead. Those give
different ciphertexts and nothing in the ciphertext says which was meant. So
only `rows` is marked `ragged_safe`, and every other fill is **skipped on a
ragged shape rather than guessed at**. The read route may still be anything --
once the blank cells are known, reading past them is unambiguous.

The cost is real and is reported. On a 182-letter text, `solve` tried 848
combinations and recorded `skipped_ambiguous_ragged_fills: 2752`: exact shapes
get all 100 route pairs, ragged shapes get 14. In practice `both_directions`
only buys you anything on an exact rectangle.

### The attack

There is no clever statistic here and no need for one -- the key space is
small enough to enumerate. `solve` walks every (shape, fill, read) triple,
scores each decryption with the English model, and keeps the best
`DEFAULT_REFINE = 40` as candidates. Exact rectangles are searched **first**,
deliberately: if a time budget cuts the search short, the work that got done
should be the likely work. Measured at 0.08 s for 848 combinations on 182
letters. Options: `rows`/`cols` to pin a shape, `routes`, `fills`,
`both_directions`, `allow_ragged`, `min_side`, `max_side`, `refine`,
`time_budget`, and `seed` (accepted and ignored).

**Limitations.** Exhaustive over what it was given is not exhaustive over all
route ciphers -- a route outside the fixed list of 15 is simply not found, and
neither is a keyed route (columns read in a keyword order), which is a
columnar and belongs to the module above. Every candidate's diagnostics
therefore name the exact grids, routes and fills tested, plus the ciphertext's
own `chi_squared_english` so that the family evidence travels with the answer.
Two different (fill, read) pairs can describe the same permutation, so the
same plaintext may surface under several keys; `CandidateSet` merges those and
counts the agreements rather than hiding them.

## The family dispatcher: solve_all (`transposition.py`)

`solve_all` is what `cipher_tool transposition` calls. It runs
`rail_fence.solve`, `columnar.solve`, `permutation.solve` and the route
`solve` above, and merges everything into one ranked `CandidateSet`.
Candidates keep the method that produced them, so the merged ranking says
which family each answer came from, and **nothing is dropped merely because
another family scored better** -- the attacks disagree often enough on short
texts that hiding the runners-up would be hiding the uncertainty. Each family
is asked for `max(top, 3)` candidates.

A `time_budget` is shared out by fixed weights:

```
_FAMILY_WEIGHTS = {"rail_fence": 0.12, "columnar": 0.45,
                   "permutation": 0.18, "routes": 0.25}
```

Every family gets at least 0.05 s even if the budget is tiny, because "we ran
out of time" is more useful when it arrives with whatever was managed. A
family that overruns its share eats into the next one's; a family that finds
the clock already spent is skipped and named in
`families_skipped_no_time`. `families_run` always lists what actually ran.

Option routing is worth knowing: `time_budget`, `seed`, `max_rails` and
`max_key_length` are handled by the dispatcher, and **everything else is
forwarded to the route solver only**. So `refine` reaches the grid search but
not the columnar one, and a columnar-only option such as `key_length` raises
`unknown option(s) for transposition.solve`. If you want to steer the columnar
attack -- and `max_exhaustive` is the one worth steering -- call
`columnar.solve` directly.

# Part 5: Digraphic and fractionating ciphers

## Polybius square (`polybius.py`)

**The cipher.** A Polybius square writes the alphabet into a labelled grid, and
each letter becomes the pair "row label, column label".

```
     1  2  3  4  5
  1  A  B  C  D  E
  2  F  G  H  I  K
  3  L  M  N  O  P
  4  Q  R  S  T  U
  5  V  W  X  Y  Z

  A -> 11    S -> 43    ATTACK -> 11 44 44 11 13 25
```

Twenty-six letters do not fit twenty-five cells, so something must give. The
module supports both conventions and a way of avoiding the choice:
`PolybiusSquare.standard()` merges I and J (the default), `without_q()` drops Q
instead (optionally with `merge_q_into="K"` for the "write KW for QU" habit),
and `six_by_six()` uses A-Z plus 0-9 in thirty-six cells so nothing is lost at
all. Any of them can be keyed -- the keyword is written in first with repeats
removed, then the unused symbols in order, so `keyed_alphabet("MONARCHY", ...)`
gives `MONARCHYBDEFGIKLPQSTUVWXZ`. Labels are free text: `"12345"` by default,
`ADFGX_LABELS` for the 1918 field cipher (`adfgx()`, `adfgvx()`), or anything
else with distinct symbols on each axis. Internally coordinates are always
zero-based `(row, column)`; the labels are only a presentation layer.

**Decryption** reads the stream two symbols at a time and looks up the cell. An
odd number of symbols raises, because one symbol is missing or spurious and
decoding anyway would shift every letter after it.

**Merging is lossy and the tool never guesses.** Cell (2,4) above always decodes
to I. A decode reading `MAIL` might have been `MAJL`; only a human reading the
sentence can decide. `prepare()` lets you see the loss before it happens --
`standard().prepare("JAM") == "IAM"`.

**The attack.** A bare Polybius encoding is a fixed substitution written in
pairs, so `solve()` does not search: it decodes under every square worth trying
-- standard, drop-Q, the 6x6, plus any keyed square built from `keywords=` --
and under the row/column transpose of each (`try_transpose=True` by default),
then ranks by English score. The label set is read off the ciphertext, because a
stream uses exactly as many distinct symbols as the square has rows; known sets
that merely *cover* what was seen are tried too. Note that `solve()` reads the
*original* text rather than the letters-only view, since a numeric square
encodes to digits and normalisation throws digits away.

**Limitations.** If the grid holds a genuinely scrambled alphabet with no
keyword behind it, trying standard squares will never find it -- that case is a
25-symbol monoalphabetic substitution over the *pairs*, and the right tool is
the general substitution solver applied to the paired-up text.
`label_permutations=True` widens the search only over the order of the labels
(120 orders for a 5x5, applied to both axes at once), which is a much smaller
question than the order of the alphabet inside the grid.

## Bifid (`bifid.py`)

**The cipher.** Bifid is what a Polybius square is *for*. Delastelle's idea
(1901) is to take each letter apart into its two coordinates, then recombine the
pieces out of step, so that every ciphertext letter is built from parts of two
different plaintext letters.

Write the coordinates of each letter as a column underneath it; read the whole
row line, then the whole column line, as one stream; cut the stream into pairs
and read each pair back as a coordinate.

```
Square: the standard 5x5 above (I and J share a cell).

  plaintext    A   T   T   A   C   K
  row          1   4   4   1   1   2
  column       1   4   4   1   3   5

  stream       1 4 4 1 1 2 | 1 4 4 1 3 5
               \--rows---/   \-columns-/

  re-split     (1,4) (4,1) (1,2) (1,4) (4,1) (3,5)
  ciphertext     D     Q     B     D     Q     P

  ATTACK -> DQBDQP
```

Look at the third ciphertext letter, B = (1,2). Its row coordinate is the *row
of C* and its column coordinate is the *row of K* -- two plaintext letters five
places apart, and neither of them contributed a column. That is fractionation:
each plaintext letter's information is smeared across two ciphertext letters and
each ciphertext letter mixes two plaintext letters. There is no longer any
letter that E maps to, so single-letter frequency analysis has nothing to bite
on and the letter distribution goes flat.

**Decryption** reverses the read. Write the coordinates of the ciphertext in
pairs, run them out as one stream of 2n numbers, cut it in half: the first half
is the plaintext rows and the second half its columns.

**The period.** Fractionating a whole message spreads a letter's halves
arbitrarily far apart, which is strong but means one transmission error destroys
everything after it. The practical variant -- the one competitions use -- cuts
the message into blocks of a fixed *period* and fractionates each block
independently. A short final block needs no special handling, because the
arithmetic only ever refers to its own block's length; there is no padding and
the ciphertext is always the same length as the plaintext. With `period=5`,
`ATTACK` splits as `ATTAC | K` and gives `DQATCK`. Note that `period=1` is the
identity, so if the solver ranks period 1 first it is telling you the text was
never fractionated.

**The attack.** `solve()` decrypts under every period from 1 to `max_period`
(default 15) plus whole-message fractionation (`include_whole_message=True`),
using the two standard 5x5 squares and any keyed square built from `keywords=`,
and ranks by English score. A square that cannot even hold the ciphertext
letters -- a drop-Q square against a text containing Q -- is ruled out and
recorded in `squares_ruled_out` rather than silently skipped.

**Limitations.** `solve()` does not *search* for an unknown square. If the grid
is a scrambled alphabet with no keyword behind it, every period will score as
noise, and the honest reading of that output is "the square is not one of
these", not "the text is not Bifid". Also, `decrypt()` runs the ciphertext
through `prepare()` first, so a J in a ciphertext handed to an I/J square is
folded to I before decryption.

### Searching for the square

`solve_unknown_square()` closes that gap by hill-climbing the twenty-five cells
against the English score of the decryption, exactly as `playfair.py` climbs
its own square, and it works for the same reason: a square that is nearly right
decrypts into nearly-English, so the score slopes towards the answer. Annealing
rather than a greedy climb, because swapping two cells of a nearly-right square
usually scores worse before it scores better. Measured: a keyed square
recovered from 400 letters in about seven seconds, from 800 in seventeen.

**The period is screened first, and the screen has to be far longer than the
equivalent one in the double-columnar search.** That contrast is the useful
thing to remember. There, a wrong column shape cannot produce English at any
search length, so a very short run separates right from wrong. Here even the
CORRECT period looks like noise until its square has been climbed properly, so
a short screen ranks almost at random. MEASURED on 500 letters at period 7,
screening periods 1 to 9: a 1,000-step screen ranked the true period **sixth of
nine**, 3,000 ranked it second, and 6,000 ranked it first. The shortlist keeps
three, so 3,000 is enough -- but 1,000 would have been worse than not screening
at all, because it would confidently discard the answer.

**Never exhaustive.** There are 25! squares, roughly 1.55 x 10^25, so a run
that finds nothing is not evidence that there is nothing to find, and every
candidate says as much. A keyword from the story, fed in through `--words`,
beats this search every time.

## Playfair (`playfair.py`)

**The cipher.** Playfair enciphers *pairs* of letters with a keyed 5x5 square
(keyword first, repeats removed, then the rest of the alphabet; one letter is
left out, `omit="J"` by default with J folded onto I).

```
Keyword MONARCHY:      M O N A R
                       C H Y B D
                       E F G I K
                       L P Q S T
                       U V W X Z

same ROW     -- each letter moves right, wrapping:   MO -> ON
same COLUMN  -- each letter moves down, wrapping:    ME -> CL
otherwise    -- opposite corners of the rectangle:   HI -> BF
                (H is row 1 col 1, I is row 2 col 3;
                 take (1,3) = B and (2,1) = F)
```

Doubled letters must be broken up first: `LL` would sit in the same row *and*
the same column, both rules would fire, and the map would stop being
invertible. So `prepare_digraphs` pushes the filler `X` between them (`BALLOON`
-> `BA LX LO ON`) and pads a lone final letter the same way. The awkward case is
a doubled letter that *is* the filler: `XX` must not become `XX` again, so the
alternative filler `Q` is used, giving `XQ XQ`. Getting that wrong is the
classic implementation bug -- it either loops forever or emits a pair the cipher
cannot encipher. Both fillers must be in the square and must differ.

**Decryption** applies the same three rules with the row and column steps
reversed (left, up); the rectangle rule is its own inverse. It returns the
*prepared* plaintext: `decrypt(encrypt(t), k) == prepare_text(t)`, fillers and
all, because neither the fillers nor the I/J merge are recoverable.

**What the ciphertext gives away**, all reported by `validate_ciphertext`: it
always has an even number of letters; the omitted letter never appears; and no
digraph at an even offset is ever a doubled letter, so an `LL` there proves the
text is mis-transcribed, offset by one, or not Playfair.

**The attack.** 25! is about 1.5 x 10^25 squares, so `solve()` climbs instead of
searching. Simulated annealing accepts a worsening move of size `delta` with
probability `exp(delta / T)`, cooling `T` linearly to zero, so the walk starts
nearly random and ends as strict hill climbing. Moves are 90 per cent
single-letter swaps, 6 per cent whole-row or whole-column swaps, and the rest
mirrors and a full reversal -- the coarse moves exist because a square that is
right apart from two exchanged rows is five letter-swaps from the truth with
every square in between scoring far worse. The annealer's fitness is a *digraph*
model counted from the project corpus, not the order-3 model, for two measured
reasons: it is about six times faster (only distinct ciphertext digraphs need
scoring, at most 600 of them however long the message), and it solved 4 of 6
seeded runs where the order-3 model solved 0, because order-3 scores also swing
on surrounding context and add noise to exactly the small differences the
annealer must judge. Each restart's result is then polished by steepest ascent
under the order-3 model over all 300 letter swaps and 23 structural moves, so
the reported square is a local optimum of the model that ranks it. Defaults:
`restarts=4`, `iterations=200000`, `temperature` = 0.030 per digraph (the CLI
passes `--restarts 6`). Pass `seed=` for a deterministic run.

**Limitations, stated plainly.** This attack needs a lot of ciphertext. Below
about 200 letters it usually fails, and it fails *confidently* -- a top
candidate is always returned, always scored, and still wrong; the `outlook`
diagnostic on every candidate says so in words. It is marginal to about 300 and
reliable somewhere around 300-500. Even on success the recovered square is
usually not the sender's: cyclically rotating all rows or all columns leaves the
cipher completely unchanged, so every square has 25 equivalent forms, and
`canonical_square` exists so two squares can be compared honestly.

## Hill (`hill.py`)

**The cipher.** Hill (1929) is block substitution by linear algebra. Fix a block
size `n` and an `n x n` key matrix `K` over the integers mod 26, cut the
plaintext into blocks written as column vectors, and

```
c = K p   (mod 26)          p = K^-1 c   (mod 26)

K = key_from_string("HILL") = [[ 7,  8],
                               [11, 11]]

block "AT" = (0, 19):   7*0 +  8*19 = 152 = 22 (mod 26) -> W
                       11*0 + 11*19 = 209 =  1 (mod 26) -> B

ATTACK -> WBDBQC
```

Every ciphertext letter of a block depends on every plaintext letter of that
block, which is what flattens the single-letter statistics.

**The inverse.** "Inverse" here means inverse in the ring of integers mod 26,
not over the reals. The adjugate identity `adj(M) M = det(M) I` holds over any
commutative ring, so

```
inv(M) = det(M)^-1 * adj(M)   (mod 26)

det = 7*11 - 8*11 = -11 = 15 (mod 26)      15 * 7 = 105 = 1, so det^-1 = 7
adj = [[11, -8], [-11, 7]] = [[11, 18], [15, 7]]
inv = 7 * adj = [[25, 22], [1, 23]]  (mod 26)
```

The construction needs `det(M)^-1` to exist, so the inverse exists **if and only
if** `gcd(det M, 26) = 1`. Since `26 = 2 x 13`, that means exactly: the
determinant must be odd and not a multiple of 13. Twelve residues qualify (1, 3,
5, 7, 9, 11, 15, 17, 19, 21, 23, 25). The "only if" half matters too: if
`gcd(det M, 26) = g > 1` then any inverse `N` would need
`det(N) det(M) = det(I) = 1` with the left side a multiple of `g`, so no such
`N` exists. `matrix_inverse` raises `NotInvertibleError` naming the determinant
and the shared factor, and `encrypt` refuses such a key outright rather than
producing a ciphertext nobody -- including the sender -- could decipher. Note
that encryption pads a short final block with `X` and records nothing: a
trailing X is unrecoverable in principle, not just here.

**The real attack: known plaintext.** Hill is linear, so `n` matched blocks
determine the key outright. Stack the matched plaintext and ciphertext blocks as
the *columns* of two `n x n` matrices; matrix multiplication acts on columns
independently, so `K P = C`, and if `P` is invertible mod 26

```
K = C P^-1   (mod 26)

Crib "ATTACK" against "WBDBQC", n = 2, blocks AT|TA and WB|DB:

  P = [[ 0, 19],   det = -361 = 3 (mod 26),  3^-1 = 9
       [19,  0]]   adj = [[0, 7], [7, 0]],   P^-1 = 9 * adj = [[0, 11], [11, 0]]

  C = [[22,  3],
       [ 1,  1]]

  K = C P^-1 = [[22*0 + 3*11,  22*11 + 3*0],  = [[ 7,  8],  = HILL
                [ 1*0 + 1*11,   1*11 + 1*0]]     [11, 11]]
```

`n^2` known letters are enough, and a competition crib of a dozen usually
supplies them. `known_plaintext_attack` tries every window of `n` consecutive
blocks first, then combinations drawn from the first ten blocks, because roughly
a third of selections give a singular `P` (any two equal blocks, or a plaintext
like `ABAB`). Every recovered key is then **verified** by re-encrypting the
whole matched plaintext -- that check is what catches a crib placed at the wrong
offset, which otherwise produces a perfectly well-formed matrix that decrypts
nothing. Cribs must start on a block boundary: `crib_at` must be a multiple of
the block size, because Hill blocks are counted from letter 0.

**Brute force, and where it stops.** For `n = 2` there are `26^4 = 456,976`
matrices, of which 157,248 (34.4 per cent) are invertible -- small enough to test
every one. `solve()` does so in two stages: a digraph model scores every key
over the first `sample_blocks` (200) blocks, then the order-3 model rescores the
best `shortlist` (40) over the whole text, which is what separates real English
from merely plausible digraphs. Keys are visited with the most English-looking
rows first, since real keys are usually written as a keyword; that only matters
under a `time_budget`, and a cut-short search records its coverage and says the
untested remainder is biased away from keyword-shaped keys.

**Limitations.** For `n = 3` the space is `26^9 = 5,429,503,678,976` and for
`n = 4` about `4.4 x 10^22`. Neither is searchable, so `solve()` refuses: with
`size >= 3` and no key or crib it returns *no candidates* and explains in
`.notes` that a crib or a supplied key is the only way in. That is a deliberate
refusal, not an oversight -- statistical attacks on 3x3 Hill exist in the
literature but need far more ciphertext than a competition round supplies, and
this toolkit has not implemented one. `solve()` also defaults to `size=2` when
you do not say otherwise, and ignores trailing letters that cannot fill a whole
block (recording how many), since a genuine Hill ciphertext always divides
exactly.

## ADFGVX and ADFGX

`adfgvx.py`. The German army cipher of 1918, and the only one here that is two
ciphers stacked. Each plaintext letter is looked up in a keyed Polybius square
and replaced by its **row and column labels**, so the message doubles in length
and is written in six letters -- A, D, F, G, V, X, chosen because they are far
apart in Morse and hard to mishear. That symbol stream is then columnar-
transposed under a second keyword.

Stacking is what made it strong, and the reason is worth stating precisely,
because it is also the reason the attack works. Fractionation alone is a
monoalphabetic substitution and falls to letter frequencies. Transposition
alone leaves letters untouched and falls to column-pair statistics. Together,
each destroys what the other's attack needs: the transposition tears every
cell in half and scatters the halves, and the fractionation leaves no letter
statistics to transpose.

### The attack: cut the problem at the joint

Undo the transposition **first**, and score a candidate column order by the
index of coincidence of the symbols *paired up two at a time*. When the order
is right the pairs ARE the original cells, so their distribution is the
distribution of English letters -- lumpy, IC near 0.067. When it is wrong the
pairs straddle cell boundaries and flatten towards the 0.028 of thirty-six
equiprobable cells. MEASURED on a 600-letter message: **0.0678 for the true
key against 0.0398 for a wrong one.** A wide, cheap signal -- and it needs no
knowledge of the square whatsoever, which is the whole point.

**It has one blind spot, and it is handled rather than hoped away.** The index
of coincidence is a property of the *multiset* of pairs, so it cannot see
their order. Several column orders produce the same pairs in a different
sequence and score identically. On that message twelve orders tied at the
maximum, the true one among them, with a clear gap of 0.0124 to the next
distinct value. So the sweep does not pick a winner; it hands the whole tied
set to the next stage.

Breaking the tie costs almost nothing, because what remains **is** a
monoalphabetic substitution: map each distinct cell to a letter and the
message becomes an ordinary substitution cipher, which `substitution.py`
already solves well. The order yielding the most English-looking plaintext
wins. Whole attack, 600 letters, six-column key: about six seconds.

No new cryptanalysis is invented here. The work was in finding the joint.

### Recovering the square

The transposition key alone does not let anybody re-read the message, so the
square is reported too, and it is reconstructed by alignment rather than from
the substitution key: cell *i* of the stream produced letter *i* of the
plaintext, so reading the two together IS the square. On the worked example
the cells come back as `AA=M AD=O AF=N AG=A AV=R AX=C DA=H DD=Y` -- which
spells MONARCHY, the keyword the square was built from. A competition answer
nobody can check by hand is worth very little.

### Recognition

Unusually reliable, and done before anything expensive: an ADFGVX ciphertext
is written in five or six specific letters and nothing else, and its length is
always even because every plaintext letter became exactly two symbols.
Anything else is refused outright rather than guessed at, which is why this
stage runs from `--fast` despite being a two-stage cipher -- it costs nothing
when it does not apply.

# Part 6: Encodings

Everything in this part lives in one file, `encodings.py`, and none of it is
encryption.

Hexadecimal, binary, decimal ASCII, Base64 and Morse are **reversible
representations of text**. There is no key, there is nothing secret, and
anybody who recognises the notation can read the message. `48 45 4C 4C 4F`
has not been encrypted; it has been written down in a different alphabet.
Decoding one is not cryptanalysis, and the code refuses to let you pretend
otherwise: the module has no `encrypt`/`decrypt` pair, only `encode_*` and
`decode_*`, every candidate it produces is named
`Encoding: <format> (NOT encryption)`, and every report carries the
`NOT_ENCRYPTION` warning string.

The reason the toolkit needs them at all is that competition ciphertext is
often wrapped in one of these notations -- as the first layer of a multi-stage
puzzle, or simply as the way the text was transcribed. Peeling the wrapper off
is the step *before* the cryptanalysis starts.

## Paired-symbol recognition (`paired.py`)

This module solves nothing. It looks at a symbol stream and decides whether it
is written in **two disjoint alphabets that strictly alternate** -- a rank and
a suit, a row and a column -- so that two symbols stand for one cell. Reading
such a stream a letter at a time gives nonsense; reading only the letters in
it gives worse than nonsense, a shorter mutilated message that still scores
well enough for a substitution solver to answer confidently.

The whole cost is linear. `_clean_prefix` walks the stream once, adding each
token to its parity class and stopping the moment a token turns up in the
other one, and the same routine run over the reversed stream gives the longest
clean suffix. Three rules keep the claim honest:

1. **A class of size one is a separator, not an alphabet.** `A1B1C1D1`
   alternates perfectly and carries no structure. Detecting it would be a
   confident wrong description, which is the failure this module exists to
   prevent -- nobody proof-reads a description the way they proof-read a
   plaintext.
2. **A single transcription slip is reported, never repaired.** If the clean
   prefix and the clean suffix together cover all but one token, one symbol is
   missing or extra at that index. The tail's parity is then flipped relative
   to the head, so the two classes have to be matched the other way round --
   getting that backwards still detects, still finds the right break index,
   and reports one 17-symbol alphabet instead of 13 ranks against 4 suits.
   Past the break the pairing is off by one and manufactures cells that do not
   exist, so the distinct-cell count is taken over the clean prefix alone: on
   a real 52-card message with one symbol missing, counting the whole stream
   gave 95 distinct cards out of a possible 52. For the same reason the second
   level -- do the CELLS themselves alternate, making the unit two cells and
   so more than one letter? -- is only looked for when level 0 is clean.
   MEASURED: on a two-level stream with one symbol deleted it still detected,
   over a cell stream that was half fabricated.
3. **The claim is falsifiable.** When it detects, the module shuffles the same
   symbol multiset 200 times and reports what fraction of the shuffles
   alternate too. Expected 0.0, measured 0.0. That number is what separates
   "these symbols alternate" from "these symbols are of two kinds".

Only one inventory is ever named, and only on an exact match: four suits, and
at least ten distinct ranks all drawn from the rank letters. Anything looser
would let the report announce a deck of cards over a stream that happens to
use the letter K.

## Homophonic substitution (`homophonic.py`)

Several symbols per letter, so the flat frequency profile every monoalphabetic
attack looks for is never there. Fifty-two symbols onto twenty-six letters --
a playing card per letter-half -- is the classic shape.

The state is an assignment of symbols to letters. The **slot multiset** -- how
many symbols stand for each letter -- is fixed before the search starts, and
the only move is a **swap** of the letters on two symbols, which preserves
that multiset by construction. Scoring is the order-3 quadgram window score
the substitution solver optimises, maintained incrementally by the same trick
`substitution._HillClimber` uses, re-derived over symbols: a swap changes the
decryption only where those two symbols stand, so only the windows those
positions touch are rescored. The running total is delta-maintained, which is
the classic place for a silent bug, so `full_score()` recomputes it and every
real run reports the largest disagreement as `score_audit_gap`.

Annealing rather than hill-climbing, geometric cooling from 1.5 to 0.06 over
the iteration count, six restarts -- half seeded from the symbol frequencies,
half from a shuffle.

### What the constraint is actually for

MEASURED 2026-08-18. **With** the constraint: 0.983 exact letters on a
52-symbol cipher from 400 units, 0.901 on a 36-symbol one from 313 (at twelve
restarts; six gives 0.856).

**Without** it -- letting any symbol take any letter -- nothing bad happens on
genuine homophonic ciphertext. At 52/400, 52/600, 36/313, 40/320 and 52/340
the free search matched or beat the constrained one, reaching 1.000. The
constraint earns its keep on a stream that is *not* homophonic, which is the
case that matters, because that is when a search manufactures a reading of
something it cannot read. On 600 units of uniformly random cards:

| | letters used | per letter | label |
|---|---|---|---|
| constrained | 26 | -2.179 | `weak` |
| unconstrained | 8 | -1.248 | `promising` |

with the free search producing
`SEEITSSESINILETSTATSSITSSITSSSTSEETSISASSITSLESS...`. Under a quadgram model
a five-letter language scores better per letter than English when nothing
forbids it, and the collapsed reading beats the honest one by nearly a whole
log unit. **Score alone cannot tell those two apart.** A key that must spend a
symbol on Z cannot become a key that only writes AEILNRST, and that is the
entire defence. `constrain_slots=False` exists so a test can watch the
collapse; it changes the move set, not a constant.

### Honest limits

* The search is annealing, not exhaustive, and says so in its diagnostics.
* Below 6.0 units per symbol it refuses; below 8.0 it caps its own label at
  `promising`. The two measured recoveries sat at 9.1 and 9.5 units per
  *observed* symbol.
* **A fixed multiset that is wrong cannot be climbed out of**, because no swap
  can change it. MEASURED on a 32-symbol cipher of which only 28 symbols occur
  in 260 units: 0.065, 0.000 and 0.850 exact letters at three seeds, and the
  0.000 run was labelled `promising`. Just above the 27-symbol floor the
  assumed multiset is most likely to be wrong and the family is least
  distinguishable from a plain substitution. The units-per-symbol gates do not
  see this and nothing here pretends they do.
* It does not repair a broken transcription. When `paired.recognise` reports
  that alternation breaks, every unit past the break is a fiction, and this
  module refuses and names the index rather than reading one.

## The Playfair same-column variant (`playfair.py`)

### What changes

A pair sharing a COLUMN moves along the row, exactly as a pair sharing a row
does, instead of down the column. Nothing else changes.

### Why it is dangerous rather than merely different

A same-column digraph is about one in seven, so **two thirds of a message
decrypts identically under either rule**. That is more than enough for the
square search to converge on the RIGHT square and then read a fluent, partly
wrong plaintext off it.

MEASURED on a real 1,502-letter competition message. The square the search
found -- agreed on by 16 of 40 independent restarts -- decrypted:

| case | digraphs | correct |
|---|---|---|
| rectangle | 541 | 541 |
| same row | 111 | 111 |
| same column | 99 | **0** |

The search had done its whole job. One rule out of three was wrong, and the
answer came back as fluent English at 0.61 word coverage, wrong in 99 places.
**No amount of extra searching fixes that**, because the search was already at
its optimum. Under the variant rule the same square returns all 1,502 letters
of the published plaintext.

### How it is wired, and what that costs

A SECOND pass, not an always-on doubling. It runs only when no reading under
the ordinary rule scored as clear English, so an ordinary Playfair pays
nothing -- the screen is free, because a legible answer means there is nothing
here to look for. And a variant reading of a particular square is added only
when it scores better than the ordinary reading of that same square, so an
ordinary message never collects a second, worse answer beside a correct one.

## Crossed-coordinate digraph cipher (`crossed.py`)

### The construction

A unit is two cells and carries one plaintext DIGRAPH. Index the first cell
`6p + q` and the second `4u + v`. Then

    first plaintext letter  = SQUARE_ONE[q][u]
    second plaintext letter = SQUARE_TWO[p][v]

Each cell carries one coordinate of each letter, so the two letters'
coordinates are interleaved ACROSS the two cells. `paired.py` could already
recognise a message written this way -- it names a 52-card deck, its two
disjoint sub-decks of 36 and 16, and a unit of two cells, unaided -- and then
stops, because it is a recogniser. This module reads it.

### Why it is breakable at all

Read as a codebook the problem is 576 cells against about 1,661 units of
message: under two observations per cell, underdetermined, and a blind search
over it produces confident garbage. Read as the construction, `576 = 24 x 24`
and the unknowns are two 24-letter squares -- **48, not 576**. The
construction is the way in, not more text.

### Stage one: recognition, which needs no key

Split each cell index into a high and a low coordinate and try the four ways
of pairing one part of the first cell with one part of the second. Under the
TRUE pairing each half of the plaintext is a plain monoalphabetic substitution
and keeps English's index of coincidence; under any other, the classes are
mixtures and the distribution flattens.

Measured over 192 candidate readings of a real message: **0.0669 for the true
split against 0.0445 for the worst.** No key is involved, so this stage cannot
be fooled by search size.

### Stage two: climbing the two squares

Plain hill-climbing is NOT enough, and the measurement says so: 12 restarts of
20,000 steps reached -1.78 per letter where the true key scores -0.9478,
because a swap inside one square is judged through the other square's noise.
Simulated annealing from a frequency-ranked start reaches the true key on the
first restart.

### What it will not do

It does not search cell orderings freely: only ascending, descending and the
two playing-card conventions are tried. A message ordered by some private
convention is refused rather than guessed at. The bound is stated in the
module rather than hidden.

### Two calibrated guards

* The recognition bar is 0.060 because the true split scores 0.0641 at 600
  units and 0.0580 at 300 -- where the recovery gets 8 per cent of the letters
  right. The bar turns a length floor into a measurement.
* Below 900 units the reading is capped. At 600 units the attack recovered
  97.5 per cent of the letters at 0.72 word coverage, which clears both
  `strong` thresholds and is still not the plaintext.

## Nihilist (`nihilist.py`)

### The construction

Plaintext letter and key letter are both looked up in the same 5x5 Polybius
square and written as a two-digit coordinate, row then column, each digit
1-5. The ciphertext number is their sum. There is NO CARRY between the digits,
so the tens and the units of a sum each run 2..10 independently.

### The attack is a constraint, not a score

Every value in one residue class shares a key coordinate `k`, so `value - k`
must be a valid coordinate for EVERY value in the class. Sweep `k` over the 25
coordinates and keep the survivors. A wrong period mixes key letters, the
class spreads wider than any single `k` can cover, and no `k` survives -- the
period is EXCLUDED, not merely outscored. That is a stronger kind of evidence
than a statistic, because a statistic can be fooled by search size and a value
ending in 1 cannot be a Nihilist sum whatever the key.

Measured on 1,494 tokens of real material, periods 1 to 14: every period
except 7 and its multiple 14 left a class with zero feasible keys, and period
7 left EXACTLY ONE feasible key in each of its seven classes.

Subtracting the key leaves a monoalphabetic substitution over 25 cells, which
`substitution.py` already solves -- and that solve recovers the square too, so
the key word and the square can both be checked by hand.

### Why it needs the raw text

The tokens are whitespace-separated, of variable width, and run past 99.
Normalising keeps the digits and throws the separators away, which turns
`"97 26 57"` into `"972657"` -- six single digits instead of three numbers.
So `parse` reads the raw text, and `auto_solve` structurally cannot reach this
family, because by the time it sees the message the boundaries are gone. The
paste screen can, and does.

## Polybius with split coordinates (`seriated.py`)

### The construction

An ordinary Polybius interleaves the coordinates: row, column, row, column.
This one writes **every row coordinate first and every column coordinate
after**, so the two halves of the message are the two rows of the
fractionation table read one after the other.

### Finding it

Score the pairing rather than the letters: the index of coincidence of the
cells, or the share of the message held by its ten commonest cell bigrams. A
substitution renames cells and cannot move either; a WRONG pairing produces
cells that are not letters and both collapse towards uniform.

The search is TINY, and that is what makes the finding safe. A true split
fractionation has two halves of equal length by construction, so the split can
only be at the middle give or take a stray symbol -- thirteen candidates, not
two thousand. There is nowhere to go fishing.

### The measurements, including the negative control

On the message this was built for, the split at 1,513 gives cell index of
coincidence **0.0663** and top-ten bigram share **0.1880**. The reference is an
ordinary Polybius already solved and graded against its published decrypt:
**0.0692** and **0.1844** -- the same numbers.

The control is the part that matters. Swept over ALL 2,600 split positions
rather than just the middle, the ordinary Polybius message's best split scores
0.0436 against a median of 0.0409: the sweep finds NOTHING on a message that
is not seriated. The seriated message's true split stands 0.0213 clear of the
best of every other split in its own text. One position, no plateau, no second
candidate.

### The guard a control produced

Handed an ordinary INTERLEAVED Polybius whose plaintext repeated with a period
dividing the half-length, the detector paired every symbol with an identical
one and reported an index of coincidence of **0.204** -- the highest number
anywhere in this exercise, from five distinct cells, on the wrong cipher
entirely. A high index of coincidence is also what a COLLAPSE looks like, and
nothing else in the detector could tell the two apart. A floor on distinct
cells now sits underneath it.

## Stacked ciphers: a polyalphabetic under a transposition (`stacked.py`)

### The problem

Every other solver in this toolkit attacks ONE cipher. Stack two and each of
them fails, and fails in the most misleading way available: **the correct
intermediate answer is not English, so the scorer that is supposed to
recognise success rejects it.**

The 2017 National Cipher Challenge, challenge 7B, is a Vigenere under the key
SCYTALE followed by a six-column transposition. Attacked as a Vigenere it
reaches `weak`. Attacked as a transposition it reaches `weak`. An earlier
attempt recovered its key CORRECTLY -- LATYCSE, which is SCYTALE read
backwards from a different starting point -- read the gibberish that came
out, and concluded that the alphabets must be mixed. A Quagmire solver was
written down as the biggest remaining gap in the toolkit. There is no
Quagmire in the archive at all.

**A perfect key that reads as nonsense means another layer, not a wrong key.**

### Cutting it at the joint

The same move as ADFGVX: find a statistic that survives the OUTER layer, use
it to strip the inner one, and hand what is left to a solver that works.

A columnar transposition reads out one whole column at a time, so each column
arrives in the ciphertext as a CONTIGUOUS run. If a periodic polyalphabetic
was applied before the transposition, then inside any one of those runs the
key still advances with its own period. So:

    split the ciphertext into `width` contiguous blocks, and measure the mean
    index of coincidence of the cosets at spacing `period` WITHIN each block.

Right shape: every coset is one monoalphabetic image of English, and the mean
sits near 0.066. Wrong shape: cosets straddle alphabets and it falls towards
0.038. MEASURED on the real 3,583-letter 7B ciphertext, 0.0661 at width 6 and
period 7 against a worst case of 0.0404 -- and it needs no knowledge of the
key, the alphabet or the column order.

### Two rules, both measured rather than reasoned

**The smallest shape wins, not the highest-scoring one.** Multiples of the
true shape peak too, because splitting a real column in half leaves the key
phase intact inside each half. Over 120 constructions the highest-scoring
shape was NOT the true one in 73 of them. The first test written for this
rule used a case where the truth happened to score highest, and it passed
with the rule deleted.

**Plain English scores highly at every width and period**, because English
cosets have an English index of coincidence however you cut them. Measured on
3,500 letters of prose the WORST setting still scored 0.0649, so the detector
cannot tell prose from a peeled stack. The whole-message index of coincidence
is checked first -- but that is an EARLY EXIT, not the safety net. What
actually refuses English is that its smallest tied shape is a width of 1,
which means no transposition at all and nothing here to attack. Measured over
sixteen English-shaped texts, not one got past with a width of 2 or more.

### Taking the layer off

No search. Each block is lined up against the first by cross-correlating its
coset letter counts; the aligned cosets are pooled, which is what makes it
robust -- on the real message that turns forty-two cosets of eighty letters
into seven of five hundred; the pooled cosets are lined up with each other
the same way; and one chi-squared against English fixes the single remaining
absolute shift. On 7B that chi-squared is 67 for the right shift against
13,410 for the runner-up.

What falls out is the transposed plaintext, letter for letter, and
`columnar.solve` finishes the job with the width already pinned.

### Reading the key it gives back

The key is reported as it runs along a COLUMN, which is not the order the
setter wrote it in. Going down a column steps the plaintext index by `width`,
so the key returns sampled with stride `width mod period`, from an unknown
starting point. On 7B that stride is 6 mod 7, which is -1: the key comes back
as SELATYC, and SCYTALE backwards is ELATYCS, of which SELATYC is a rotation.
The letters are right; the reading order is a decimation.

### The ragged grid, and a search that did not pay for itself

When the message is not a whole number of rows, some columns hold one more
letter -- and those are the first few columns of the GRID, which arrive in
the ciphertext in KEY order. So which contiguous blocks are long depends on
the key, which is exactly what is not known yet. MEASURED on a 4,000-letter
message under ZEBRAS the blocks run 666, 667, 667, 667, 666, 667 while the
obvious guess gives 667, 667, 667, 667, 666, 666.

Searching every assignment was built and then measured against guessing, over
the same 100 stacked messages: 61 readings of 99 per cent or better against
62, 42 exact against 40, no confident-and-wrong answer either way. No
difference that could be told from noise, so the guess ships and the search
is gone. It leaves a handful of letters at block boundaries -- three in 2,000
on the measured case -- which is why an answer here is worth re-reading at
the joins.

### How well it works, and how honestly

MEASURED over 100 stacked messages from five polyalphabetic keys, five
transposition keys and four lengths from 1,200 to 4,000 letters:

| | |
|---|---|
| shape detected exactly | 59 |
| read 99 per cent or more of the letters | 61 |
| exact to the last letter | 42 |
| labelled `strong` | 61 |
| **labelled `strong` while under 99 per cent** | **0** |

The 39 it does not get come back `weak` or `unlikely`, which is the honest
answer. On the real 2017 7B message it is exact: all 3,583 letters, `strong`,
under three seconds.

### Honest limitations

- About 900 letters is the floor. A 600-letter version of the same
  construction is not detectable at all, and the solver says `unlikely`
  rather than guessing.
- Only a COLUMNAR transposition over the polyalphabetic. A block permutation
  laid over one would destroy the contiguity the detector depends on.
- A polyalphabetic key with repeated structure can make a divisor of its
  period look tied -- PALIMPSEST has P in positions 0 and 5 -- and those
  cases come back `weak`.

## Hexadecimal (`encodings.py`)

Each byte is written as two digits base 16, using `0-9` and `A-F`.

```
H = 72 decimal = 0x48        0x48 = 4*16 + 8 = 72
"HELLO" -> 48 45 4C 4C 4F
```

`decode_hex` first strips layout, then converts. Layout is stripped in two
passes: `_SEPARATORS` (`[\s,:;|_-]+`) removes whitespace, commas, colons,
semicolons, pipes, underscores and hyphens, then `_HEX_PREFIX` removes `0x`
and `\x` case-insensitively. Removing the prefixes cannot destroy data because
`x` is not a hexadecimal digit, so those two characters can only ever be a
prefix. The remaining digits go to the standard library's `bytes.fromhex`.

```python
decode_hex("48 45 4C")      == "HEL"
decode_hex("48454c")        == "HEL"
decode_hex("0x48:0x45:0x4C") == "HEL"
```

It raises `ValueError` naming the offending characters if anything is outside
the alphabet, or if the digit count is odd (two digits per byte, so an odd
count cannot be a whole number of characters).

`encode_hex(text, separator="", upper=True)` goes the other way; note the
default is **no separator and uppercase**, so `"HELLO"` becomes `48454C4C4F`.
Only ASCII text can be encoded -- `_text_to_bytes` raises a `ValueError`
naming the character and its position for anything else.

Limitation: decoded bytes that are not printable ASCII come back as `?`.
`_bytes_to_text` substitutes rather than raising, because a decode that is 95
per cent readable is exactly the evidence a solver wants to see. The
consequence is that the decoded string is for **reading, not round-tripping**
-- you cannot re-encode it to recover the original bytes.

## Binary (`encodings.py`)

Each character is written as a group of bits, most significant first.

```
H = 72 = 64 + 8 = 01001000   (8 bits)
                   1001000   (7 bits: ASCII only ever needed 7)
"HI" -> 01001000 01001001
```

The only interesting decision is the group width, and `_binary_bytes` takes
evidence in a fixed order:

1. an explicit `bits=` argument (7 or 8 only; anything else raises);
2. the transcription's own grouping -- if there are at least two groups, all
   the same width, and that width is 7 or 8, the writer has told us directly;
3. failing that, divisibility of the total digit count. If the length fits
   both 7 and 8 it decodes both ways and prefers 8 **unless** the 7-bit
   reading is strictly more printable.

`decode_binary(text, bits=None)` infers by default; `encode_binary` defaults
to `bits=8, separator=" "`, and refuses `bits=7` if any byte exceeds 127.
A length that is a multiple of neither 7 nor 8 raises rather than guessing.

Limitation: only 7 and 8 are supported. A puzzle using 5-bit Baudot or 6-bit
codes is out of scope, and rule 3 makes the width inference a genuine guess on
lengths like 56 that satisfy both readings.

## Decimal ASCII (`encodings.py`)

Each character is written as its ASCII code in ordinary decimal, separated by
whitespace, commas, semicolons or pipes (`_NUMBER_SPLIT` is `[\s,;|]+`).

```
"HELLO" -> 72 69 76 76 79
```

`decode_decimal` accepts `"72 69 76 76 79"` and `"72,69,76,76,79"` alike. Any
token that is not a run of digits raises, and so does any value above 255 --
the module does not decode wider code points.

Limitation: **A1Z26 letter numbering is not implemented.** A list where every
value lies in 1..26 is far more likely to be A=1, B=2 than decimal ASCII, and
the code deliberately does not guess: read as ASCII those values are control
characters, so the detector's printable test throws the whole reading out and
no claim is made either way.

## Base64 (`encodings.py`)

Three bytes (24 bits) are cut into four 6-bit groups, each group written as one
of 64 characters `A-Z a-z 0-9 + /`. If the input length is not a multiple of
three the output is padded to a multiple of four with `=`.

```
"Man" = 4D 61 6E = 010011 010110 000101 101110
                      19     22      5     46
                       T      W      F      u     -> "TWFu"
```

This is the one part of the module that leans on a real library:
`decode_base64` and `encode_base64` call the standard library's `base64`
(with `binascii` for its error type). `base64` ships with Python, is not a
third-party package and is not a deciphering tool -- Base64 is a transport
notation defined in RFC 4648. Every actual cryptanalytic algorithm in the
toolkit is written from scratch.

Before handing over, `_base64_bytes` does its own checks so the error messages
are useful: whitespace is removed (Base64 is routinely wrapped at 76 columns),
characters outside the alphabet are named, more than two `=` is rejected, and
the length must be a multiple of four. The library call then uses
`validate=True` so malformed input fails loudly rather than being silently
skipped over.

Limitation: **URL-safe Base64 (`-` and `_` instead of `+` and `/`) is not
decoded.** Those two characters are treated as layout separators elsewhere in
the module, and resolving the ambiguity was judged not worth the false
positives it would create.

## Morse code (`encodings.py`)

Each character is a string of dots and dashes; letters are separated by a
single space, words by `/` or by two or more spaces.

```
H .... | E . | L .-.. | L .-.. | O ---
.... . .-.. .-.. ---  /  .-- --- .-. .-.. -..     ->  "HELLO WORLD"
```

The table is **ours**: `_MORSE_PAIRS` is the ITU alphabet typed out in the
file -- 26 letters, 10 digits and the 18 punctuation marks ITU actually
defines -- and built into `MORSE_TABLE` plus a reversed lookup. Because a
duplicated entry would silently make one character undecodable, the module
checks at import that the mapping is one-to-one and raises immediately if it
is not.

Two normalisations happen first. `_` is folded to `-`, since transcriptions
use either for a dash and the underscore character itself is encoded `..--.-`;
and tabs, carriage returns and newlines become spaces. A literal `/` can only
be a word separator, because the `/` character is itself encoded as `-..-.`.

`encode_morse` defaults to `letter_separator=" "` and `word_separator=" / "`,
uppercases its input and raises for any character with no symbol.
`decode_morse` raises naming the offending token.

Limitation, and it bites quietly: a newline is folded to a **single** space,
which is a letter separator. Morse transcribed one word per line therefore
decodes as a single run-together word unless the line breaks happen to carry
trailing spaces.

## Identification: what `identify` demands before it speaks (`encodings.py`)

This is where a tool like this can do real damage, because these notations
overlap ordinary text completely:

* every letter of `DEFACEDBEEF` is a hexadecimal digit;
* every letter of an uppercase English ciphertext is a Base64 character;
* `01001000` is valid binary, valid hexadecimal, a valid decimal number and a
  valid Base64 string.

So "this text *could* be parsed as X" is worth nothing. Guessing wrongly is
worse than not guessing, because it invites a teammate to throw away real
ciphertext and spend the evening analysing rubbish. Every detector must
therefore clear three gates before it says a word.

**1. Alphabet.** Not one character outside the format's alphabet, once layout
has been removed. Morse is the strictest: after normalisation the text may
contain nothing but `.`, `-`, `/` and spaces, which is what keeps English
prose and its ellipses out.

**2. Shape.** The length must be consistent with the format: an even number of
hexadecimal digits, a whole number of 7-bit or 8-bit binary groups, a multiple
of four Base64 characters. Arbitrary text has an arbitrary length, so roughly
half the false positives die here. Minimum sizes apply too --
`MIN_HEX_DIGITS = 4`, `MIN_BINARY_GROUPS = 2`, `MIN_DECIMAL_VALUES = 2`,
`MIN_BASE64_CHARS = 8` with `MIN_BASE64_BYTES = 6`, `MIN_MORSE_TOKENS = 3`.

**3. Product.** The decode must produce mostly printable ASCII --
`MIN_PRINTABLE_RATIO = 0.9`, counting bytes 32..126 plus tab, newline and
carriage return. This is the test that does the real work, and the argument
for it is a probability one:

```
P(a random byte is printable)      ~ 95/256 ~ 0.37
P(12 random bytes are all printable) ~ (95/256)^12 ~ 1 in 100,000
```

and the odds collapse further with every extra byte. Text that merely *looks*
like hexadecimal decodes to high-bit noise and is rejected. So `is_hex`
returns **False** for `"DEADBEEF"`: it parses perfectly, but it decodes to
`DE AD BE EF`, four bytes above 127 that no message would contain.

**4. Base64 only: a character a ciphertext could not contain.** The Base64
alphabet contains the entire uppercase Latin alphabet, so gates 1 and 2 are no
defence against a 16-letter chunk of ciphertext. `_examine_base64` therefore
also demands at least one character from `_BASE64_EVIDENCE` -- a lowercase
letter, a digit, `+`, `/` or `=`. Real Base64 of six or more bytes of text
essentially always contains one; an uppercase-only ciphertext never does.
Base64 is also held to the strictest product test in the module: **every**
decoded byte must be printable, not merely 90 per cent. `ATTACKATDAWNXYZQ` is
16 characters, a clean multiple of four, and is correctly not reported.

**Confidence is two-valued on purpose**: `possible` and `likely`, with no
`certain`. A short input can satisfy every gate by accident, and the module
says so instead of pretending. `_confidence_for` upgrades to `likely` only
when the decode is at least 4 bytes, **entirely** printable, and at least 90
per cent drawn from the characters English is actually written with (letters,
digits, spaces, `.,;:'!?-()/"`). So even a textbook `... --- ...` is reported
as `Morse code: possible` -- `SOS` is three bytes.

**Multiple answers are honest answers.** `identify` runs all five tests and
returns every one that passes, sorted by confidence and then by the fixed
`FORMAT_NAMES` order. It returns an empty list -- the common and correct
answer for real ciphertext -- when nothing clears the bar, and
`describe_guesses` then prints "No encoding was identified. Treat the text as
ciphertext."

Worked example, and a trap worth knowing:

```
input:   72 69 76 76 79
decimal: 72 69 76 76 79           -> "HELLO"    likely
hex:     72 69 76 76 79 (spaces stripped, 10 digits, even)
         0x72 0x69 0x76 0x76 0x79 -> "rivvy"    likely
```

Both readings are genuinely valid and both decode to printable text, so both
are reported. They tie on confidence, so the tie is broken by `FORMAT_NAMES`
order -- and **hexadecimal is listed first**, above the reading that is
obviously right. `identify` ranks by evidence of *format*, not by quality of
English.

`solve()` is what sorts that out: it scores each decode with the English
scorer from Part 1 and ranks by that, so `HELLO` (-6.3) comes above `rivvy`
(-11.9), while still returning both. Note that `solve` reads
`NormalizedText.original`, not `letters` -- encodings live in the digits,
spacing and punctuation that normalisation throws away, and `01001000` has no
letters at all. Its `top` defaults to 5 and `include_possible` to `True`; its
`time_budget` and `seed` parameters exist only so every solver in the toolkit
shares one signature, and have no effect, because five fixed tests are already
exhaustive and deterministic. Candidates carry `display=None`, because the
decoded text is a different length from the input and cannot be poured back
into the input's layout.

One thing to know about the command line: `cipher_tool encodings` calls
`identify` and `describe_guesses`, **not** `solve`. It prints the evidence
lines and the not-encryption note; it does not rank the decodes by English.

# Part 7: Cribs

## What a crib test does and does not claim (`cribs.py`)

A crib is a guessed piece of plaintext -- the phrase you think ends the
letter, the name the story keeps repeating, the place the last message
mentioned. Cribs are worth far more than their length suggests, because a
classical cipher is a *deterministic* function of the key: fix a few plaintext
letters and you usually fix a large part of the key.

Every function in this module answers exactly one question:

```
If this crib really is in the plaintext, WHERE could it be,
and what would the key have to look like there?
```

It never answers "the crib is here". A surviving offset means *not yet ruled
out*, which is a far weaker statement than *right*, and `CribReport.render()`
prints that warning at the end of every report in as many words.

The genuinely strong results here are the **negatives**. Under a
monoalphabetic substitution a five-letter crib typically leaves dozens of
surviving offsets in a few hundred letters -- useless on its own. But if it
survives at *no* offset, then either the crib is wrong or the cipher is not
monoalphabetic, and that is a real conclusion. The report is careful to keep
three states apart:

- field is `None` -- the family was never tested;
- field is `[]` -- the family was tested and nothing survived (ruled out);
- field is non-empty -- possibilities, with evidence attached.

`CribReport.possible_methods()` returns the families still standing, in the
fixed order of `METHODS`: substitution, caesar, affine, vigenere,
transposition. Absence from that tuple is the useful half.

Positions everywhere are indices into the **letters-only** text, never into
the original string. Five-letter grouping destroys original offsets, so letter
positions are the only ones that survive normalisation. A crib is cleaned the
same way ciphertext is, so `"meet me"` and `"MEETME"` are the same crib; a
crib with no letters in it raises `ValueError` rather than silently matching
everywhere.

## Substitution placements: the bijection test (`cribs.py`)

`substitution_placements(ciphertext, crib, *, known=None, no_fixed_points=False)`
slides the crib along every offset and keeps an offset only if the crib and
the ciphertext window under it define a consistent **bijection**. Two
conditions, both enforced by `patterns.mapping_from_pair`:

1. no cipher letter may stand for two different plain letters;
2. no two cipher letters may stand for the same plain letter.

Condition 1 is just "the shapes match" -- it is what a naive pattern check
already does. **Condition 2 is what does most of the pruning**, and it is the
reason this is worth writing properly. A substitution alphabet is a
permutation, so it is injective as well as functional.

Worked example, crib `MEET` against ciphertext `XPQQTY`:

```
offset 0   window X P Q Q     X->M, P->E, then Q->E.
                  M E E T     E is already claimed by P, so two cipher
                              letters would mean the same plain letter.
                              REJECTED by condition 2.

offset 1   window P Q Q T     P->M, Q->E, Q->E, T->T.
                  M E E T     Consistent both ways. SURVIVES.

offset 2   window Q Q T Y     Q->M, then Q->E.
                  M E E T     One cipher letter, two plain letters.
                              REJECTED by condition 1.
```

The survivor fixes three letters of the alphabet, not four -- `Placement.fixes`
counts distinct cipher letters, which is the honest measure of a crib's
strength. Repetition is what makes a crib bite: `MEETING` has signature
`0-1-1-2-3-4-5`, and over uniformly random letters only about one window in
fifty can match it. A crib with no repeats constrains only condition 2 and
will survive at a large fraction of offsets; that is a property of the crib,
not a fault in the code. Use the longest and most repetitive crib you have.

`known=` takes a partial `SubstitutionKey` (or a plain cipher->plain mapping)
you already believe, and drops any offset contradicting it -- again in both
directions, via `SubstitutionKey.with_pair`. A contradiction is returned as a
dropped offset, not raised as an error, because testing a wrong guess is the
normal case.

### The `no_fixed_points` flag, honestly

`no_fixed_points=True` discards any offset that would need a ciphertext letter
to stand for itself. **It defaults to False and should almost always stay
there.**

The flag exists for one famous special case. An Enigma machine can never
encipher a letter as itself, which is exactly what let Bletchley Park slide a
crib along a message and reject every position where a letter agreed with the
ciphertext. Ordinary pencil-and-paper substitutions have no such property: a
keyword alphabet routinely leaves several letters where they started, and the
Cipher Challenge uses keyword alphabets constantly.

So switching it on without independent reason will throw away the truth.
`tests/test_cribs.py::test_no_fixed_points_can_discard_the_truth` demonstrates
precisely that on the `XPQQTY` / `MEET` example above: the only surviving
offset needs `T -> T`, and turning the flag on returns an empty list.

Instead of filtering by default, the toolkit *reports* fixed points.
`Placement.fixed_points` lists them, and the rendered report marks such an
offset with a leading `*` plus a footnote saying it is allowed by an ordinary
substitution and impossible only for some machine ciphers. When the flag is
on, the report also prints a warning that it may have discarded the answer.

## Caesar and affine placements (`cribs.py`)

For these two families there is nothing to reason backwards about, because the
key space is tiny. `caesar_placements` enciphers the crib under all 26 shifts
and searches the ciphertext for the result; `affine_placements` does the same
over the 12 x 26 = 312 usable keys (`a` must be coprime with 26, so
`a` is one of 1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25).

```
ciphertext  DWWDFNDWGDZQ
crib        ATTACK

shift 0  -> ATTACK   not present
shift 1  -> BUUBDL   not present
shift 2  -> CVVCEM   not present
shift 3  -> DWWDFN   found at offset 0   <-- consistent
...
```

Result: `[ShiftPlacement(shift=3, positions=(0,))]`. The shift reported is the
*enciphering* shift, matching the rest of the toolkit, so
`caesar.decrypt(ciphertext, 3)` is the reading that contains the crib.

Search is by `str.find` in a loop rather than a regular expression, so
overlapping matches are found: a crib like `THATTHAT` can genuinely overlap
itself.

What makes this worth doing is **completeness**. Every key is tried, so an
empty result is a real negative: under no shift does this ciphertext contain
this crib, and the Caesar family is eliminated unless the crib itself is
wrong. The same holds for the 312 affine keys.

Solving algebraically would also work -- two letter pairs give
`a * (P1 - P2) = C1 - C2` mod 26 and then `b` follows -- but it needs care
when `P1 - P2` shares a factor with 26, and enciphering a short crib 312 times
costs nothing. The obviously-correct method wins.

Two honest limitations. First, a crib made of one repeated letter (`AAA`)
gives one equation in two unknowns, so a great many affine keys will "fit";
short cribs are weakest of all here. Second, the 312 keys include the identity
`a=1 b=0`, so a plaintext-in-the-clear message will report a hit that means
nothing -- `AffinePlacement.describe()` labels it, along with the other
special cases (`a=1` is a pure Caesar, `a=25 b=25` is Atbash).

## Vigenere: subtracting the crib hands you the key (`cribs.py`)

This is the most productive function in the module, and the reason is worth
stating carefully. A Vigenere ciphertext is the plaintext plus a repeating
key, letter by letter:

```
C[i] = (P[i] + K[i mod L]) mod 26
```

Rearranged:

```
K[i mod L] = (C[i] - P[i]) mod 26
```

So if the crib really is the plaintext at offset `i`, subtracting it from the
ciphertext there does not merely *test* a hypothesis -- **it hands over a
piece of the key**, already aligned at key position `i mod L`. That is a
different kind of result from every other function here.

Worked example. Take `ATTACKATDAWN` under the key `LEMON`:

```
P  A T T A C K A T D A W N
K  L E M O N L E M O N L E
C  L X F O P V E F R N H R
```

Now hand the solver only `LXFOPVEFRNHR` and the crib `ATTACK`, and let it
subtract at offset 0:

```
window   L(11) X(23) F(5)  O(14) P(15) V(21)
crib     A(0)  T(19) T(19) A(0)  C(2)  K(10)
C - P    11    4     -14   14    13    11
mod 26   11    4     12    14    13    11
         L     E     M     O     N     L
```

The fragment is `LEMONL`. The key was never guessed; it fell out of one
subtraction. A crib as long as the key gives the whole key at once, and a
shorter crib still fixes several letters, with the rest finished by frequency
fitting on the remaining columns.

With `key_length=L` supplied, each fragment is folded onto the key positions
it would occupy -- slot `(position + index) % L` -- and any slot that receives
two different letters kills the offset. Offset 0 above lands on slots
0,1,2,3,4,0 and slot 0 receives `L` twice, so it is consistent and yields
`partial_key = "LEMON"`. Offset 1 gives fragment `XMVPTU` on slots 1,2,3,4,0,1,
where slot 1 gets `X` then `U`: inconsistent, `partial_key = None`. That test
alone throws away most offsets.

Without a key length, **nothing is rejected** -- there is nothing yet to
reject it with -- and `consistent_fragments()` returns the input unchanged,
which is the honest answer rather than an empty one. Every offset is returned
either way, so you can see how hard the test actually pruned; filter on
`.consistent` yourself.

`word_key_fragments()` flags fragments that read as an English word, testing
both the raw fragment and (when the key length was assumed and every slot
filled) the completed key. Treat a hit as a place to look first, nothing more:
it is a prior about the *person who chose the key*, not a fact about the
cipher. A real key fails it whenever the crib straddles a wrap-round, since
the fragment is then a rotation (`MONLE` for `LEMON`). Readings shorter than
`MINIMUM_CRIB_LENGTH` (3) are skipped.

## Key-length evidence from repeated fragments (`cribs.py`)

Even with no key length assumed, the fragments carry information. Two offsets
produce the *same* fragment exactly when the two ciphertext windows are
identical, because the crib cancels out of `C - P`. So `repeated_fragments()`
is a repeated-substring search at the crib's length, and a fragment repeating
at spacing `d` is Kasiski's argument narrowed to that length: an identical
ciphertext stretch happens when identical plaintext meets identical key, and
the key repeats only after a whole number of periods, so **L divides d**.

`key_length_votes()` turns that into one vote per divisor of each spacing.
Worked example, `ATTACKATDAWN` written twice under the key `KEY`:

```
ciphertext  KXRKGIKXBKALKXRKGIKXBKAL
crib        ATTACK

repeated fragments and their offsets:
  KEYKEY at 0, 12      RRNIIN at 2, 14     KNPKVR at 3, 15
  XYRGGA at 1, 13      GPRXZA at 4, 16     IREBIQ at 5, 17
  KEIKYB at 6, 18

every spacing is 12, so every divisor of 12 in range scores:
  2 (7)  3 (7)  4 (7)  6 (7)  12 (7)
```

Read that honestly. The votes correctly narrow the field to divisors of 12,
and `KEYKEY` sitting there in plain sight is a strong hint -- re-running with
`key_length=3` leaves exactly offsets 0 and 12 consistent, both giving
`KEY`. But the vote counts alone cannot choose 3 over 12, because a longer key
also divides the spacing. Votes are hints.

Limitations to know:

- 1 is excluded (it divides everything) and so is anything above
  `MAXIMUM_VOTED_KEY_LENGTH = 20`.
- Only gaps between **consecutive** occurrences are counted, matching
  `statistics.repeat_distances`. Every gap between a non-adjacent pair is a
  sum of consecutive gaps, so counting all pairs would count the same
  evidence twice and inflate a fragment that happens to appear three or four
  times. Even so, treat the absolute counts as suggestive rather than
  measured.
- Short repeats happen by chance. For serious key-length work use
  `vigenere.estimate_key_lengths`, which measures the whole text rather than
  one crib's windows.

## Transposition: a strong negative, a weak positive (`cribs.py`)

A transposition rearranges the letters and changes none of them, so the
**multiset of letters is an invariant of the entire family**. Two consequences
follow, and they are wildly unequal in value.

**The strong one, a negative.** If the ciphertext does not hold enough copies
of some crib letter, then no rearrangement of it can contain the crib, and
every transposition -- rail fence, columnar, block permutation, route, grid,
any of them -- is
eliminated at a stroke. It costs one pass of letter counting.

```
ciphertext  ATTACKATDAWN
crib        ZEBRA

Z needs 1 but only 0 present
E needs 1 but only 0 present
B needs 1 but only 0 present
R needs 1 but only 0 present

RULED OUT. No transposition of this ciphertext contains ZEBRA.
```

Almost nothing else in classical cryptanalysis eliminates a whole family that
cleanly, which is why the check is worth running even when you expect it to
pass.

**The weak one, a positive.** If every letter is present in sufficient
quantity, that is *not* evidence the crib is there. Any English text of a few
hundred letters holds enough Es and Ts for almost any crib, so `possible=True`
means very little. `transposition_crib_help` therefore attempts no automated
placement at all -- under a transposition the crib's letters are no longer
adjacent, so there is no offset to slide and no key fragment to read off.

What it offers instead is the position of every crib letter, sorted rarest
first, because that is the list a human actually works from: a crib letter
occurring twice gives two places to check, one occurring ninety times gives
ninety. `rarest_letters(count=3)` names where to start. The rendered report
prints the honest label itself -- "this is a weak result -- it is a letter
count, not a placement".

It also reports `column_counts` from `columnar.plausible_column_counts`: the
divisors of the length between 2 and 30. That is context only, and the report
says so -- it applies solely if the sender padded to a complete rectangle,
because an incomplete columnar leaves every column count possible.

## Suggested cribs and team context (`cribs.py`, `context.py`)

`suggest_cribs()` returns entries from `reference.SUGGESTED_CRIBS`, a
hand-typed list of about thirty-five words and phrases that turn up constantly
in Cipher Challenge stories -- `MESSAGE`, `SINCERELY`, `MIDNIGHT`, `ATTACK`,
`DAWN` and so on. The list is a prior about the competition's writing style
and nothing else. **Not one of these words is known to be in your
ciphertext.** `describe_suggestions()` prints exactly that caveat above the
list.

Filtering is deliberately crude: a crib must be at least
`MINIMUM_CRIB_LENGTH` (3) letters, must fit in the text, and must not exceed
`max(minimum_length, length // 2)` -- a crib longer than half the message is a
guess about most of the plaintext rather than a foothold in it. Results are
sorted longest first, because a longer crib constrains harder.

The better source is `context.py`, which is somewhere for the team to type
what they read in the story. It has seven fields -- `people`, `places`,
`dates`, `phrases`, `fragments`, `keywords`, `notes` -- stored as plain JSON
beside the ciphertext (`message.txt` gets `message.txt.context.json`) so it
can be edited by hand, diffed and committed. **There is no search and no
network access of any kind**; everything in the file was typed by a person.

Two derived lists feed the solvers:

- `crib_candidates()` draws on `fragments`, `phrases`, `people` and `places`
  -- the entries most likely to appear verbatim -- at 3 letters or more.
- `keyword_candidates()` draws on `keywords`, `people`, `places`, `dates` and
  `phrases` at 2 letters or more, since setters like naming a key after
  someone in the story.

Both split multi-word entries as well as keeping them whole, so
`"Admiral Harrow"` yields `ADMIRALHARROW`, `ADMIRAL` and `HARROW` -- the
plaintext might carry the surname without the rank. Note that `notes` feeds
neither list; it is a scratchpad for humans.

These are **candidates to test, never assumptions**. Every one is placed and
scored like any other guess, and a context entry that leads nowhere simply
scores badly. `ContextNotes.render()` ends with that sentence, and
`merge_cribs()` combines context cribs with any given on the command line,
capped at 40.

## What the command line actually exposes (`cli.py`)

Worth knowing before you reach for an option that is not there.
`cipher_tool crib <file> "THE" "MEETING"` runs `test_crib` for each word and
prints each report. The subcommand accepts only:

- `--methods NAME...` -- limit to some of the five families; an unrecognised
  name is an error rather than a silent skip, because silently testing nothing
  looks exactly like a crib that fits nowhere;
- `--no-context` -- ignore the saved story notes, which are merged in by
  default when reading from a file;
- `--key-length N` -- tell the Vigenere test the key length you believe. This
  is the flag worth reaching for: without it you get every offset's key
  fragment and the vote evidence, but with it the test can check that a
  fragment is consistent with a repeating key of that length and hand you a
  `partial_key`;
- `--no-fixed-points` -- assume no letter stands for itself. Most
  substitutions DO have fixed points, so this is off by default and should
  only be used with independent evidence;
- `--limit N` -- how many placements to list per method.

`test_crib` also takes `known` (a partial substitution key already believed)
and `scorer`, which have no flag; pass those from Python if you need them.
