"""The cross-solver pipeline behind ``cipher_tool auto``.

What it does
------------
Runs the cheap diagnostics first, then the cheap attacks, then the expensive
searches, collecting every candidate into one ranked list. It reports what it
tried, what it found, and -- just as importantly -- what it did not get round
to.

Two design decisions worth knowing about
----------------------------------------
**Statistics choose the ORDER, not the shortlist.** It is tempting to read
the Index of Coincidence, conclude "this is polyalphabetic", and skip the
substitution solver entirely. We do not do that. The heuristics in
``statistics.py`` are frequently wrong, especially on short texts, and a
pipeline that acts on a wrong guess produces confident nonsense. Instead the
statistics move the most plausible families to the front of the queue, so
that when a time budget runs out the work that got done is the work most
likely to matter. Everything appropriate to the effort level is still tried
if there is time.

**Nothing is hidden.** Every stage appears in the report with its outcome:
solved-looking, nothing found, skipped for effort level, or cut short by the
clock. A stage that never ran is stated, not omitted -- otherwise the report
reads as "we tried everything" when it did not.

Effort levels
-------------
``fast``   -- seconds. Cheap ciphers plus a shallow substitution and Vigenere
              pass. Use this first, always.
``normal`` -- tens of seconds. Adds Beaufort, autokey, the full transposition
              family, Bifid, Polybius and keyword recovery.
``deep``   -- minutes. Adds Playfair hill climbing and the Hill 2x2 brute
              force, and multiplies the search effort of everything else.

``max_time`` overrides all of it: stages run in order until the clock runs
out, and the rest are reported as skipped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from . import (
    adfgvx,
    affine,
    atbash,
    autokey,
    beaufort,
    bifid,
    caesar,
    columnar,
    encodings,
    hill,
    keyword_cipher,
    playfair,
    polybius,
    rail_fence,
    substitution,
    transposition,
    vigenere,
)
from .candidates import Candidate, CandidateSet, render_candidates
from .normalize import NormalizedText, normalize
from .scoring import EnglishScorer, annotate, default_scorer
from .statistics import TextStatistics, analyse, summarise

EFFORT_LEVELS = ("fast", "normal", "deep")


@dataclass
class StageReport:
    """What one solver in the pipeline did."""

    name: str
    family: str
    ran: bool
    seconds: float = 0.0
    candidates: int = 0
    best_score: float | None = None
    best_confidence: str | None = None
    note: str = ""

    def describe(self) -> str:
        """One line for the pipeline summary table."""
        if not self.ran:
            return f"  {self.name:<26} skipped   {self.note}"
        outcome = (
            f"{self.candidates} candidate(s), best {self.best_confidence}"
            if self.candidates
            else "nothing found"
        )
        note = f"  {self.note}" if self.note else ""
        return f"  {self.name:<26} {self.seconds:6.2f}s  {outcome}{note}"


@dataclass
class AutoResult:
    """Everything the pipeline produced."""

    normalized: NormalizedText
    stats: TextStatistics
    candidates: CandidateSet
    stages: list[StageReport] = field(default_factory=list)
    effort: str = "normal"
    seconds: float = 0.0
    time_budget: float | None = None
    budget_exhausted: bool = False

    def render(self, *, top: int = 10, full_text: bool = False,
               show_stats: bool = False) -> str:
        """Format the full report for the terminal."""
        from .statistics import render_report

        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("AUTOMATIC ANALYSIS")
        lines.append("=" * 72)
        lines.append(f"Effort level : {self.effort}")
        lines.append(f"Ciphertext   : {summarise(self.stats)}")
        lines.append(f"Elapsed      : {self.seconds:.2f}s")
        if self.time_budget is not None:
            lines.append(f"Time budget  : {self.time_budget:.0f}s"
                         + ("  (EXHAUSTED)" if self.budget_exhausted else ""))
        lines.append("")

        if show_stats:
            lines.append(render_report(self.stats))
            lines.append("")

        lines.append("What the statistics suggest (HEURISTIC, often wrong)")
        lines.append("-" * 52)
        for number, hypothesis in enumerate(self.stats.hypotheses[:4], start=1):
            lines.append(f"  [{number}] {hypothesis.family} "
                         f"({hypothesis.confidence})")
        lines.append("")

        lines.append("Stages")
        lines.append("-" * 52)
        for stage in self.stages:
            lines.append(stage.describe())
        lines.append("")

        skipped = [stage for stage in self.stages if not stage.ran]
        if skipped:
            lines.append(
                f"NOT TRIED: {', '.join(stage.name for stage in skipped)}."
            )
            lines.append(
                "  Raise the effort level or the time budget to include them."
            )
            lines.append("")

        lines.append(render_candidates(
            self.candidates.ranked(), top=top, full_text=full_text,
            title="Ranked candidates",
        ))

        if self.candidates.looks_unencrypted():
            lines.append("")
            lines.append("=" * 72)
            lines.append("THIS TEXT DOES NOT APPEAR TO BE ENCRYPTED")
            lines.append("=" * 72)
            lines.append(
                "The best-scoring result handed the input straight back. "
                "Every cipher family here has a key that does nothing -- "
                "Caesar shift 0, Vigenere key AAA, affine a=1 b=0 -- and on "
                "readable English each solver finds its own. That is not a "
                "solve, and the keys below should not be reported as one."
            )
            lines.append(
                "Either this is already the plaintext, or only part of the "
                "message was pasted in."
            )

        agreeing = self.candidates.corroborations()
        if len(agreeing) > 1:
            lines.append("")
            lines.append(
                "CORROBORATION: " + ", ".join(agreeing)
                + " independently produced the same plaintext. Agreement "
                "between unrelated attacks is strong evidence."
            )

        gap = self.candidates.score_gap()
        if gap is not None:
            lines.append("")
            if gap < 0.10:
                lines.append(
                    f"WARNING: the best candidate leads the best COMPETING "
                    f"reading by only {gap:.3f} per letter. The search has "
                    "NOT singled one out; read them both."
                )
            else:
                lines.append(
                    f"The best candidate leads the best competing reading by "
                    f"{gap:.3f} per letter."
                )

        best = self.candidates.best()
        if best is not None and best.confidence() in {"weak", "unlikely"}:
            lines.append("")
            lines.append(
                "Nothing scored well. That usually means the cipher is not in "
                "the set tried, the key is longer than the search allowed, or "
                "there is not enough ciphertext. Try --deep, supply a crib, or "
                "add story context."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

#: Each stage: name, family it belongs to, minimum effort level, relative cost
#: weight (for dividing a time budget), and the call itself.
@dataclass(frozen=True)
class Stage:
    """One solver invocation in the pipeline."""

    name: str
    family: str
    minimum_effort: str
    weight: float
    run: Callable[..., CandidateSet]
    options: dict[str, Any] = field(default_factory=dict)

    def allowed(self, effort: str) -> bool:
        """True if this stage runs at the given effort level."""
        return EFFORT_LEVELS.index(effort) >= EFFORT_LEVELS.index(
            self.minimum_effort
        )


def build_stages(effort: str, top: int, seed: int | None) -> list[Stage]:
    """The stage list for an effort level, in default (cheapest-first) order.

    Search sizes scale with effort. The numbers are deliberately explicit
    rather than hidden inside the solvers, so that what ``--fast`` actually
    means is readable here.
    """
    scale = {"fast": 0, "normal": 1, "deep": 2}[effort]

    restarts = (6, 25, 80)[scale]
    max_key_length = (12, 20, 26)[scale]
    brute_force = (2, 3, 3)[scale]
    max_primer = (3, 5, 8)[scale]
    max_period = (8, 15, 20)[scale]
    playfair_restarts = (0, 0, 8)[scale]
    # Every key length up to 9 is searched even at --fast. Columnar keys of
    # seven to nine letters are ordinary in this competition, and since
    # columnar.DEFAULT_MAX_EXHAUSTIVE enumerates up to 9 the cost is small.
    # MEASURED on 300-letter messages with keys of 6, 7, 8 and 9 columns: a
    # ceiling of 6 solved one of the four in 0.34s, a ceiling of 9 solved all
    # four in 1.74s. Roughly a third of a second per message to stop missing
    # three quarters of real columnar keys is not a trade worth making.
    # Beyond 9 the search falls back to the greedy chain, which is markedly
    # weaker, so only --deep goes there and its candidates say so.
    columnar_max = (9, 9, 12)[scale]
    # The ADFGVX sweep enumerates every column order, so its cost is the
    # factorial of this: 6! is 720 orders, 8! is 40,320. Six covers the
    # commonest keyword lengths cheaply; deep goes to eight.
    adfgvx_max = (6, 7, 8)[scale]

    stages: list[Stage] = [
        Stage("encodings", "encoding", "fast", 0.2, encodings.solve,
              {"top": top}),
        Stage("Caesar", "monoalphabetic", "fast", 0.2, caesar.solve,
              {"top": top}),
        Stage("Atbash", "monoalphabetic", "fast", 0.1, atbash.solve,
              {"top": 1}),
        Stage("affine", "monoalphabetic", "fast", 0.3, affine.solve,
              {"top": top}),
        Stage("rail fence", "transposition", "fast", 0.3, rail_fence.solve,
              {"top": top}),
        Stage("substitution", "monoalphabetic", "fast", 8.0,
              substitution.solve,
              {"top": top, "restarts": restarts, "seed": seed}),
        Stage("Vigenere", "polyalphabetic", "fast", 4.0, vigenere.solve,
              {"top": top, "max_key_length": max_key_length,
               "brute_force_up_to": brute_force}),
        Stage("columnar", "transposition", "fast", 3.0, columnar.solve,
              {"top": top, "max_key_length": columnar_max, "seed": seed}),
        # Runs from --fast despite being a two-stage cipher, because it costs
        # nothing when it does not apply: an ADFGVX message is written in
        # five or six specific letters and has an even length, so anything
        # else is rejected before a single permutation is tried. Waiting for
        # --deep would mean the paste screen escalating twice, and taking
        # minutes, to reach a cipher it can recognise instantly.
        Stage("ADFGVX", "fractionating", "fast", 6.0, adfgvx.solve,
              {"top": top, "max_key_length": adfgvx_max, "seed": seed}),
        # Cheap for the same reason ADFGVX is: it refuses anything that is
        # not an even-length symbol stream before doing any work, and when
        # it does apply it is one substitution climb rather than a search
        # over squares. About a second.
        Stage("Polybius (unknown square)", "fractionating", "fast", 1.5,
              polybius.solve_unknown_square, {"top": top, "seed": seed}),

        Stage("keyword substitution", "monoalphabetic", "normal", 2.0,
              keyword_cipher.solve, {"top": top, "seed": seed}),
        Stage("Beaufort", "polyalphabetic", "normal", 4.0, beaufort.solve,
              {"top": top, "max_key_length": max_key_length}),
        Stage("autokey", "polyalphabetic", "normal", 5.0, autokey.solve,
              {"top": top, "max_primer": max_primer, "seed": seed}),
        Stage("route/grid transposition", "transposition", "normal", 3.0,
              transposition.solve_all,
              {"top": top, "max_key_length": columnar_max, "seed": seed}),
        Stage("Polybius", "fractionating", "normal", 1.0, polybius.solve,
              {"top": top}),
        Stage("Bifid", "fractionating", "normal", 3.0, bifid.solve,
              {"top": top, "max_period": max_period}),

        # Double columnar carries its own time_budget rather than trusting
        # the caller to set max_time, because without one it is unbounded in
        # a way no other stage is: the shapes grow as the SQUARE of the key
        # ceiling and each one is a randomised climb, so an ordinary --deep
        # run would sit there for hours. When the caller does set max_time,
        # the loop below overwrites this with that stage's fair share, which
        # is the right way round -- an explicit budget should beat a
        # defensive default.
        #
        # The settings are deliberately weaker than the module's own
        # defaults. This is a chance for a double transposition to fall out
        # of a general search, not a full attack on one; somebody who
        # suspects it should pin the key lengths and let the solver run
        # properly. Candidates report shapes_screened and
        # shapes_fully_searched against shapes_available, so a partial sweep
        # is visible rather than quietly implied.
        # The ceiling is 8, not 6, and that was measured rather than chosen.
        # A 7-letter keyword is ordinary in this competition, and with the
        # ceiling at 6 the paste screen escalated all the way to deep on a
        # 7x6 message and still reported `weak` -- honest, and useless. The
        # screen-then-refine search covers all 49 shapes of a ceiling of 8 in
        # about thirty seconds on 400 letters, because screening is cheap and
        # only the shortlist is searched properly. The budget is set well
        # above that so an awkward length has room, not because it is
        # expected to be needed.
        Stage("double columnar", "transposition", "deep", 25.0,
              columnar.solve_double,
              {"top": top, "max_key_length": 8, "seed": seed,
               "restarts": 6, "iterations": 15_000, "time_budget": 60.0}),
        # Bifid against a square nobody supplied. Held to --deep because,
        # unlike ADFGVX, there is no cheap way to recognise it: a Bifid
        # ciphertext looks like any other letter salad, so the cost is paid
        # on every message that gets this far rather than only the ones this
        # can help. Budgeted for the same reason.
        # Runs at the module's OWN default effort, not a trimmed-down
        # version of it. MEASURED on a 400-letter keyed message at period 7:
        # three restarts of 6,000 steps failed inside a 45 second budget and
        # reported `weak`, while four of 8,000 solved it at `strong` in 42.7
        # seconds. A stage that runs, spends half a minute and cannot reach
        # the answer is worse than no stage at all, because its presence
        # reads as coverage -- the same mistake the double columnar stage
        # made with a key-length ceiling of six.
        Stage("Bifid (unknown square)", "fractionating", "deep", 15.0,
              bifid.solve_unknown_square,
              {"top": top, "seed": seed, "max_period": 12,
               "restarts": bifid.DEFAULT_CLIMB_RESTARTS,
               "iterations": bifid.DEFAULT_CLIMB_ITERATIONS,
               "time_budget": 90.0}),
        Stage("Playfair", "digraphic", "deep", 10.0, playfair.solve,
              {"top": top, "restarts": playfair_restarts, "seed": seed}),
        Stage("Hill 2x2", "digraphic", "deep", 10.0, hill.solve,
              {"top": top, "size": 2}),
    ]
    return [stage for stage in stages if stage.allowed(effort)]


def order_stages(stages: Sequence[Stage], stats: TextStatistics) -> list[Stage]:
    """Move the families the statistics favour to the front.

    This changes only the ORDER. Nothing is dropped, because the heuristics
    are not reliable enough to justify dropping anything. The point is that
    if the clock runs out, the work that got done is the work most likely to
    have been worth doing.
    """
    preferred: list[str] = []
    for hypothesis in stats.hypotheses:
        family = hypothesis.family.lower()
        if "transposition" in family:
            preferred.append("transposition")
        elif "monoalphabetic" in family:
            preferred.append("monoalphabetic")
        elif "polyalphabetic" in family or "repeating key" in family:
            preferred.append("polyalphabetic")
        elif "digraphic" in family or "playfair" in family:
            preferred.append("digraphic")
        elif "fractionating" in family or "polybius" in family:
            preferred.append("fractionating")
        elif "encoding" in family or "numeric" in family:
            preferred.append("encoding")

    def rank(stage: Stage) -> tuple[int, float]:
        try:
            priority = preferred.index(stage.family)
        except ValueError:
            priority = len(preferred) + 1
        # Within a priority band, cheap stages still go first.
        return (priority, stage.weight)

    return sorted(stages, key=rank)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def auto_solve(
    source: str | NormalizedText,
    *,
    scorer: EnglishScorer | None = None,
    effort: str = "normal",
    top: int = 10,
    seed: int | None = None,
    max_time: float | None = None,
    stages: Sequence[Stage] | None = None,
) -> AutoResult:
    """Run the whole pipeline and return every candidate it found.

    Parameters
    ----------
    effort:
        ``"fast"``, ``"normal"`` or ``"deep"``. See the module docstring.
    max_time:
        Overall wall-clock budget in seconds. Stages run in order until it is
        spent; the remainder are reported as skipped. ``None`` means no limit
        beyond what the effort level implies.
    seed:
        Makes every randomised search reproducible.
    """
    if effort not in EFFORT_LEVELS:
        raise ValueError(
            f"Unknown effort level {effort!r}. "
            f"Choose one of: {', '.join(EFFORT_LEVELS)}."
        )

    engine = scorer or default_scorer()
    normalized = normalize(source) if isinstance(source, str) else source
    stats = analyse(normalized)

    result = AutoResult(
        normalized=normalized,
        stats=stats,
        # Telling the candidate set what the ciphertext was lets it recognise
        # a "decryption" that handed the input straight back.
        candidates=CandidateSet(source_letters=normalized.letters),
        effort=effort,
        time_budget=max_time,
    )

    if normalized.is_empty:
        result.stages.append(StageReport(
            "input check", "none", ran=False,
            note="the input contains no letters, so nothing was attempted",
        ))
        return result

    plan = list(stages) if stages is not None else build_stages(effort, top, seed)
    plan = order_stages(plan, stats)

    started = time.monotonic()
    deadline = started + max_time if max_time is not None else None
    total_weight = sum(stage.weight for stage in plan) or 1.0

    for index, stage in enumerate(plan):
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            result.budget_exhausted = True
            result.stages.append(StageReport(
                stage.name, stage.family, ran=False,
                note="time budget exhausted before this stage",
            ))
            continue

        options = dict(stage.options)
        if deadline is not None:
            # Share what is left in proportion to the remaining stages'
            # weights, so one expensive search cannot eat the whole budget.
            remaining_weight = sum(s.weight for s in plan[index:]) or 1.0
            share = (deadline - now) * (stage.weight / remaining_weight)
            options["time_budget"] = max(0.05, share)

        stage_started = time.monotonic()
        try:
            found = stage.run(normalized, scorer=engine, **options)
        except TypeError as error:
            # A solver that does not accept time_budget is not a failure --
            # retry without it rather than losing the stage.
            if "time_budget" in options and "time_budget" in str(error):
                options.pop("time_budget")
                found = stage.run(normalized, scorer=engine, **options)
            else:
                result.stages.append(StageReport(
                    stage.name, stage.family, ran=False,
                    note=f"error: {error}",
                ))
                continue
        except Exception as error:  # a broken solver must not kill the run
            result.stages.append(StageReport(
                stage.name, stage.family, ran=False,
                note=f"error: {type(error).__name__}: {error}",
            ))
            continue

        elapsed = time.monotonic() - stage_started
        produced = list(found.ranked()) if found else []
        result.candidates.extend(produced)

        note = ""
        for candidate in produced[:1]:
            if candidate.diagnostics.get("time_budget_hit"):
                note = "search cut short by the clock"
        extra_notes = getattr(found, "notes", ())
        if extra_notes:
            note = (note + "  " if note else "") + "; ".join(extra_notes)

        best = produced[0] if produced else None
        result.stages.append(StageReport(
            stage.name, stage.family, ran=True, seconds=elapsed,
            candidates=len(produced),
            best_score=best.score if best else None,
            best_confidence=best.confidence() if best else None,
            note=note,
        ))

    _add_reversed_readings(result.candidates, engine)

    result.seconds = time.monotonic() - started
    if deadline is not None and time.monotonic() >= deadline:
        result.budget_exhausted = True
    return result


#: How much better, per letter, a backwards reading must score before it is
#: offered. Generous, because the difference is not subtle when it is real:
#: measured on the case that prompted this, -1.999 forwards against -1.176
#: backwards, a gap of 0.82.
REVERSAL_MARGIN = 0.25

#: Candidates checked for a backwards reading. The check is one extra score
#: each, which is nothing beside the searches that produced them, but there
#: is no point running it over a long tail of readings nobody will look at.
REVERSAL_DEPTH = 10


def _add_reversed_readings(
    candidates: CandidateSet, engine: EnglishScorer
) -> None:
    """Offer the backwards reading of a candidate when it is the better one.

    Writing the message backwards is a real competition trick, and it was
    found here by running the tool against the published National Cipher
    Challenge archive rather than against anything invented for a test. The
    2018 challenge 8A ciphertext decrypted correctly and then read

        MQYRAMNIAGAREHTEGOTKROWOTOTINUTROPPOEHTEVAHEWEPOHYUO

    at -1.999 per letter with 31 per cent word coverage, so the pipeline
    called it `weak` and moved on. Backwards it is -1.176 with 73 per cent:
    OUY HOPE WE HAVE THE OPPORTUNIT[Y] TO WORK TOGETHER AGAIN MARY. The
    decryption was correct and the answer was already on the screen; nothing
    thought to read it the other way round.

    The reversed reading is ADDED rather than substituted, so both are in
    the ranking and the score decides. A margin keeps it quiet: English read
    backwards still scores far worse than English, so a real reversal shows
    up as a large gap rather than a close call.
    """
    for candidate in candidates.top(REVERSAL_DEPTH):
        if candidate.diagnostics.get("plaintext_reversed"):
            continue
        backwards = candidate.plaintext[::-1]
        if len(backwards) < 20:
            continue
        gain = engine.normalised(backwards) - engine.normalised(
            candidate.plaintext
        )
        if gain < REVERSAL_MARGIN:
            continue
        # Re-derive the evidence for the text we are actually offering. The
        # first version of this copied the forward candidate's diagnostics
        # and kept its word_coverage, so a perfect reading of the 2018 8A
        # message was judged on the gibberish it came from -- 0.214 coverage
        # against its real 0.9 -- and came back `weak`. Confidence reads
        # these fields, so stale ones are not cosmetic.
        diagnostics = dict(candidate.diagnostics)
        annotate(diagnostics, backwards, engine)
        diagnostics["plaintext_reversed"] = True
        diagnostics["reversal_gain_per_letter"] = round(gain, 3)
        diagnostics["note"] = (
            "the decryption read backwards; the message was written in "
            "reverse before it was enciphered"
        )
        candidates.add(Candidate(
            method=f"{candidate.method} (plaintext reversed)",
            key=candidate.key,
            score=engine.score(backwards),
            plaintext=backwards,
            diagnostics=diagnostics,
        ))


def quick_triage(source: str | NormalizedText) -> str:
    """A few seconds of cheap checks, for deciding what to do next.

    Runs the statistics and the trivially cheap ciphers only. Useful when a
    new ciphertext arrives and you want to know whether it is a five-minute
    problem or a two-hour one.
    """
    result = auto_solve(source, effort="fast", top=3, max_time=10.0)
    lines = [summarise(result.stats), ""]
    for hypothesis in result.stats.hypotheses[:3]:
        lines.append(f"  {hypothesis.confidence:<9} {hypothesis.family}")
    best = result.candidates.best()
    lines.append("")
    if best is not None and best.confidence() in {"strong", "promising"}:
        lines.append(f"A cheap attack already looks {best.confidence()}: "
                     f"{best.method} {best.key}")
        lines.append(f"  {best.preview()}")
    else:
        lines.append("No cheap attack worked. Run the full pipeline.")
    return "\n".join(lines)
