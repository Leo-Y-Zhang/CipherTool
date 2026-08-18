# Competition rules compliance

**This document is not a claim of approval.** Nobody at the National Cipher
Challenge has seen this toolkit, reviewed it, or endorsed it. What follows is
a record of the design decisions we took to keep it inside the rules as we
understand them, written so that a teacher, a judge or a teammate can check
the claims for themselves.

## Status of the rules check

| | |
|---|---|
| **Checked against** | the 2026 rules |
| **Checked on** | 16 August 2026 |
| **Checked by** | a member of the team, reading the published rules |
| **Outcome** | **All clear.** Nothing in the 2026 rules conflicts with a design decision recorded below. No code change was required. |

The eight questions in
[What to re-check](#what-to-re-check-when-the-2026-rules-are-published)
were worked through against the published text. In particular,
self-written software remains permitted, the standard library raised no
issue, and the use of `base64` for decoding a text encoding was judged
acceptable.

**This status is dated on purpose.** It applies to the 2026 rules as
published at that date and to the code as it stood then. If the rules are
amended mid-season, or if the toolkit gains a dependency or a new
capability, the check is stale and the table above is worth less than
nothing -- because it will read as current when it is not. Re-run it and
re-date it.

---

## The rule we designed around

The National Cipher Challenge has historically allowed competitors to use
software they write themselves, and prohibited the use of deciphering tools
found elsewhere on the web.

We took the strict reading: *everything that does cryptanalysis in this
toolkit must be ours, and nothing may consult an outside service.* Where we
had a choice between a convenient dependency and writing the code ourselves,
we wrote it ourselves.

---

## 1. Every algorithm is ours

Every cipher and every attack in `src/cipher_tool/` was written for this
project. We did not copy, port, or adapt a third-party solver, and we did not
work from someone else's implementation.

This includes the parts it would have been easiest to borrow:

| Thing we could have taken from a library | Where we wrote it instead |
|---|---|
| Extended Euclidean algorithm / modular inverse | `affine.py`, `hill.py` |
| Matrix determinant, adjugate and inverse mod 26 | `hill.py` |
| Index of Coincidence | `statistics.py` |
| Kasiski examination | `statistics.py` |
| Chi-squared frequency fitting | `statistics.py` |
| n-gram language model and its smoothing | `scoring.py` |
| Word-segmentation dynamic programming | `scoring.py` |
| Hill-climbing search with restarts | `substitution.py`, `playfair.py` |
| Simulated annealing over a constrained assignment | `homophonic.py` |
| Permutation search for transposition | `columnar.py`, `transposition.py` |
| Pattern-signature word matching | `patterns.py` |

The only place we use a standard-library routine that does any real work is
`base64` in `encodings.py`, and Base64 is a text encoding, not a cipher. See
[section 5](#5-dependencies).

**How to check:** read the code. Every module carries a docstring explaining
the mathematics of the cipher and of the attack, and the non-obvious steps are
commented inline. If a teammate cannot explain a module after reading it, that
is a bug in the module and we should fix it.

## 2. No external deciphering service is used

The toolkit does not contain, call, wrap, or fall back to:

- dCode, CyberChef, quipqiup, Boxentriq, Cryptii, or any other online solver
- any cipher-solving API of any kind
- any LLM or AI service (OpenAI, Anthropic, Google, or any other)
- any hosted dictionary, corpus, or frequency-table service

**How to check:** `tests/test_compliance.py` fails the build if any module in
the package imports `socket`, `ssl`, `urllib`, `http`, `requests`, `httpx`,
`aiohttp`, `ftplib`, `smtplib`, `telnetlib`, `webbrowser`, `subprocess`,
`ctypes`, or several others, or if a URL appears anywhere in the source.

## 3. No competition ciphertext leaves this machine

The toolkit has no network code at all, so there is nothing to send anything
anywhere. There is no telemetry, no analytics, no crash reporting, no update
check, and no "phone home" of any kind.

**How to check, two ways:**

- *Statically*: the import audit in section 2.
- *Dynamically*: `test_a_full_solve_never_opens_a_socket` replaces
  `socket.socket`, `socket.create_connection`, `socket.getaddrinfo` and
  `socket.gethostbyname` with functions that raise, then runs the entire
  `auto` pipeline on a ciphertext. If any code path tried to reach the
  network, that test fails loudly. A grep can be fooled; this cannot.

The only file access is reading the ciphertext file you name and reading our
own prose corpus out of `src/cipher_tool/data/`. The toolkit writes nothing
unless you pass `--output`.

## 4. No automatic submission

The toolkit produces candidate plaintexts locally and prints them. It does
not, and cannot, submit anything to the competition website.

There is no submit command, no browser automation, no form posting, and no
clipboard automation. A human team member must read the candidate, decide
whether it is right, and enter it themselves.

**How to check:** section 3 rules out any network path. In addition,
`test_nothing_submits_answers_anywhere` searches the source for submission
verbs and fails if it finds any.

## 5. Dependencies

**Runtime dependencies: none.** `pyproject.toml` declares
`dependencies = []`, and the compliance test asserts that it stays that way
and that no `requirements.txt`, `Pipfile`, `poetry.lock` or `environment.yml`
appears in the repository.

The toolkit uses only the Python standard library. Every import in the
package is checked against `sys.stdlib_module_names` by
`test_every_import_is_standard_library_or_our_own`.

### Why each standard-library module is acceptable

| Module | What we use it for | Cryptanalysis in it? |
|---|---|---|
| `math` | `log10`, `isqrt`, `gcd` | No |
| `collections` | `Counter`, `defaultdict` | No |
| `itertools` | permutations for transposition search | No |
| `functools` | `lru_cache` on the shared scorer | No |
| `dataclasses`, `typing` | plain data structures and type hints | No |
| `random` | seeded restarts for hill climbing | No |
| `re`, `string`, `unicodedata` | input normalisation | No |
| `pathlib` | reading our own corpus files | No |
| `time` | honouring `--max-time` budgets | No |
| `argparse`, `sys`, `textwrap` | the command line interface | No |
| `ast` | the compliance audit reads our own source | No |
| `base64`, `binascii` | decoding Base64 and hex in `encodings.py` | No -- see below |
| `unittest` | the test suite | No |

**On `base64`:** Base64 and hexadecimal are *encodings*, not ciphers. They
hide nothing and require no key. Decoding them is not cryptanalysis any more
than reading a file in UTF-8 is. `encodings.py` states this in its docstring
and labels every result `(NOT encryption)`. We judged that using the standard
library's Base64 decoder is no different from using its string handling. If
your reading of the current rules disagrees, the function is four lines and
can be replaced with our own alphabet lookup -- see
[section 9](#what-to-re-check-when-the-2026-rules-are-published).

### The optional test-runner dependency

`pyproject.toml` has an optional `dev` extra containing `pytest`. It is
**not required**: the tests are plain `unittest` cases and
`python run_tests.py` runs the whole suite with nothing but the standard
library. `pytest` runs tests; it contains no cryptanalysis. We kept it
optional so that the toolkit and its tests together depend on nothing outside
the standard library.

## 6. The English scoring resources are ours

This is the part of a cryptanalysis toolkit most likely to smuggle in
outside material, so we were deliberate about it.

Automatic solving needs a way to judge whether a candidate plaintext looks
like English. The usual approach is to download a quadgram frequency file
built from a large corpus. **We did not do that.** Instead:

- **`src/cipher_tool/data/corpus_*.txt`** contains roughly 24,000 words
  (about 102,000 letters) of ordinary English prose **written for this
  project**: narrative, dialogue, correspondence, expository writing,
  popular history and everyday journalism. It is not scraped, not copied from
  any book, article or website, and not downloaded. It is in the repository
  in plain text and you can read all of it.
- **`scoring.py`** builds the language model *at runtime* by counting letters,
  pairs, triples and quadruples in that prose. No pre-computed frequency file
  ships with the toolkit; the model is derived from our own text every time
  the program starts.
- **`reference.py`** contains a table of English letter frequencies, a list of
  common digraphs and trigraphs, and a list of about 900 common English words,
  all typed in by us. The letter frequencies are a widely published statistic
  about the English language -- a fact, like the boiling point of water, not
  software. They are used only as a starting hint for chi-squared fitting and
  for the frequency display; the actual scoring model comes from the corpus.
- **The lexicon** used for word-coverage scoring and pattern matching is the
  hand-typed word list plus every distinct word appearing in our own corpus,
  about 3,900 words in total. Nothing was downloaded.

**Trade-off, stated honestly:** a 100,000-letter corpus is small compared with
the corpora behind published frequency tables, so our model is weaker than a
professionally trained one. We accepted that in exchange for provenance we can
defend. We compensated with an interpolated backoff model, which degrades
gracefully on sparse data, and by adding word coverage as a second,
independent signal. Measured on held-out text the model separates real English
(-0.89 per letter) from wrong decryptions (-2.70 per letter) cleanly, which is
all the search actually needs. See `ALGORITHMS.md`.

## 7. Nothing here circumvents any rule

There is no functionality whose purpose is to get around a competition rule.
Specifically there is no code that:

- disguises the origin of a solution
- fetches solving code at runtime (`test_no_dynamic_code_execution` fails the
  build on any `eval`, `exec`, `compile` or `__import__` call)
- automates interaction with the competition site
- shares ciphertext or solutions between teams

## 8. What the toolkit will not do for you

Stated plainly, because overclaiming is its own kind of rule problem:

- It **ranks guesses**; it does not decide. Every solver returns several
  candidates with scores and evidence, and the strongest confidence label
  available is `strong` -- never `solved`.
- Its cipher-family suggestions are labelled **HEURISTIC** and are often
  wrong.
- A candidate is only worth submitting when a human has read it and it makes
  linguistic and contextual sense.

## What to re-check when the 2026 rules are published

Worked through on 16 August 2026 against the 2026 rules: all clear, no code
change required (see [Status of the rules check](#status-of-the-rules-check)).

Kept in full below because the list is the thing to re-run, not the answer.
Do it again if the rules are amended, if the toolkit gains a dependency, or
at the start of the next season. Tick each one against the published text,
not against this document.

1. **Is self-written software still permitted?** Everything else depends on
   this. If the answer is no, do not use the toolkit at all.
2. **Are there restrictions on *which* techniques may be automated?** Some
   competitions restrict automation on early, teaching-focused rounds. If so,
   restrict yourselves to `analyse` and manual commands for those rounds.
3. **Is the standard library acceptable, or must even that be justified?**
   Section 5 lists every module and what it does, ready to quote.
4. **Is `base64` acceptable?** If not, replace `decode_base64` in
   `encodings.py` with our own 64-character alphabet lookup. It is a small
   change and removes the last standard-library routine that does anything
   more than arithmetic and string handling.
5. **Are numerical libraries permitted?** We assumed not and wrote pure
   Python throughout, including the Hill cipher matrix arithmetic. If they
   *are* permitted, we still do not need one -- no change required. Do **not**
   add one just because it is allowed; pure Python keeps the code
   unquestionably ours.
6. **Are locally built language models acceptable?** Section 6 describes
   exactly what ours is and where every byte came from.
7. **Is there a rule about team members' individual contributions?** If so,
   record who wrote which module; `git log` is the evidence.
8. **Is there anything about disclosing tool use when submitting?** If a
   declaration is required, declare it. Do not omit it because the software
   is self-written.

If any answer conflicts with a design decision above, change the code, then
change this document. Do not leave the two disagreeing.

---

## Running the audit

```
cd CipherTool
python run_tests.py test_compliance -v
```

Every claim in sections 1 to 7 that can be checked by a machine is checked
there. The claims that cannot be machine-checked -- that we wrote the
algorithms ourselves, and that the corpus is original -- rest on our word and
on the git history.

---

*This is a locally written cryptanalysis toolkit. Competition eligibility
depends on the current National Cipher Challenge rules. Verify the current
rules before using it in a live round.*
