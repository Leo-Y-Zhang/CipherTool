# CipherTool (cipher_tool)

An offline, dependency-free classical cryptanalysis toolkit — Caesar through
Hill, ADFGVX, Playfair, Vigenère and more — written from scratch in pure
Python for a school team's National Cipher Challenge entries. Every cipher
and attack is original code with no third-party cryptanalysis library, no
runtime dependencies, and no network access (see `RULES_COMPLIANCE.md`); the
project puts heavy emphasis on honestly-labelled confidence (no "solved"
label, only "strong" at most) and documents its own failure modes candidly,
including two prior incorrect timing claims it caught and fixed in place.

## Directory layout

- `src/cipher_tool/` — the package, including `data/*.txt` (the ~24,000-word
  English corpus used to build the runtime Markov model).
- `tests/` — one `test_<cipher-or-topic>.py` per cipher/feature (40 files),
  e.g. `test_caesar.py`, `test_vigenere.py`, `test_hill.py`,
  `test_playfair.py`, `test_compliance.py` (the competition-rules audit).
- `run_tests.py` — the test runner (stdlib `unittest`, no pytest required).
- `conftest.py` — root-level, used when running under pytest.
- `ALGORITHMS.md`, `RULES_COMPLIANCE.md`, `SECURITY.md` — design/compliance
  docs; `docs/`, `examples/`.
- `cipher_tool.sh` / `.command` / `.bat` — launcher scripts for macOS/Linux
  and Windows.

## Install

Nothing is required — stdlib only. Optional, for a console script / pytest:
```
python -m pip install -e .
```
or add the `dev` extra (`pip install -e '.[dev]'`) for `pytest>=7`, which is
test-runner sugar only, not a functional dependency.

## Lint / format

No linter, formatter, or type-checker (no ruff/mypy config) is present in
this repo; `pyproject.toml` declares no `[tool.ruff]` section and CI runs no
lint step. The closest thing to a static check is the compliance suite below.

## Test

Full suite is 1,556 tests (~3,000 subtests) and is genuinely slow — the
README's own budget is ~30 minutes ("nearly all of it is the randomised
climbs, which are slow on purpose"); this container observed 588 of the
then 1,505 pass with zero failures before a 600s timeout cut it off. Do not
run the full suite in an interactive/time-boxed session:
```
python run_tests.py            # full suite, budget ~27-33 min
```
Fastest useful subset — the competition-compliance audit (13 tests, ~2.4s,
the same one CI runs):
```
python run_tests.py test_compliance -v
```
Other fast, non-randomized single-module runs work the same way, e.g.
`python run_tests.py test_caesar -v`. Avoid `test_readable`/`test_auto`-style
randomized-solve modules without a generous timeout — they contain the slow
hill-climbing cases. `pytest -q` also works (same tests, unittest-based) if
pytest is installed.

## Verification gate (source of truth)

There is no separate `verify.py`; CI's `test` job is the gate and runs, in
order: `python run_tests.py` (full suite), then
`python run_tests.py test_compliance -v` (the competition rules audit — no
third-party imports, no networking, no URLs, ASCII-only source, every
module/public function docstringed, docs present), then a CLI smoke test that
Caesar-decodes a fixed ciphertext end to end via `cipher_tool auto`. A
separate `packaging` job installs the wheel and asserts `pip list` contains
nothing beyond `cipher-tool, pip, setuptools, wheel, packaging` — i.e. that
the zero-dependency claim holds after a real install, not just in
`pyproject.toml`.

## Environment caveats (from audit)

- The full test run cannot fit in a 10-minute box; budget separately or run
  only the fast subsets above in an interactive session.
- CI's `test` job runs on a 3-OS matrix (ubuntu/windows/macos) x 2 Python
  versions (3.10/3.13) with a 90-minute timeout — nothing here is
  Windows-only or platform-gated, the matrix exists to prove portability of a
  pure-stdlib tool, not because of OS-specific code paths.
- Some benchmark claims in the README (e.g. the Playfair hill-climbing
  timing table) require re-running randomized multi-restart searches and are
  not independently reproducible within a short time box; treat them as
  documented but not gate-checked here.

## CI / conventions

- `ci.yml` (`test` job): no `pip install` step at all, deliberately — proves
  the toolkit truly needs nothing beyond the stdlib to run its tests.
- `packaging` job: `pip install .`, then a `pip list --format=json` diff
  against the allowed-package set (see above).
- No coverage floor is enforced; no ruff/mypy config exists.
