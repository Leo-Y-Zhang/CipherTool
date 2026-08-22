"""Undo a block permutation hiding UNDER another cipher.

The gap this closes
-------------------
The toolkit could already solve a substitution, and could already solve a
transposition. It could not solve one applied to the other. 2025 challenge 6A
was exactly that -- a monoalphabetic substitution whose letters had then been
shuffled in blocks of four -- and it survived the substitution solver at every
effort level, correctly reported as ``weak`` for five minutes of search.

Why it can be attacked without the key
--------------------------------------
The trick is to score the ORDER of the letters with a statistic that does not
care what the letters are called. Take the share of the text held by its ten
commonest bigrams:

* a substitution RELABELS letters, so it permutes the names of the bigrams and
  leaves every count untouched -- the statistic does not move;
* a transposition MOVES letters, so it destroys adjacency -- the statistic
  collapses.

So the statistic isolates the transposition, and the transposition can be
stripped off **before** anything is known about the cipher underneath. That is
what makes this cheap enough to run on every paste: the search is over block
permutations only, never over keys.

Measured on 2025 6A: 13.09% as received, 18.53% once the period-4 permutation
``(0, 3, 2, 1)`` is undone, against 18.27% for 5B -- a confirmed monoalphabetic
solve of comparable length. The un-permuted text then solves ``strong`` on the
first try.

The honesty rule this module exists to keep
-------------------------------------------
**An exhaustive search over ``w!`` orders will ALWAYS find one that scores above
average.** Reporting the best of 40,320 arrangements as a discovery is how a
search manufactures a finding. So every width is scored against its OWN null
control: the identical exhaustive search run over a SHUFFLE of the same
letters. A width is only reported when it beats its control by
``MINIMUM_GAP`` -- beating the raw text is explicitly not the bar.

This was not theoretical. On the first pass over 6A an unkeyed columnar search
returned width 3 as its best hit, "beating" the raw text -- and the shuffle
control scored higher still. Without the control that would have been filed as
the answer.
"""

from __future__ import annotations

import collections
import itertools
import random
from collections.abc import Sequence
from dataclasses import dataclass

#: Widths searched at all.
MAX_WIDTH = 8

#: Up to this width every order is tried (6! = 720, instant). Above it the
#: search hill-climbs instead: 8! is 40,320 orders, and paying that twice per
#: width -- once for the text, once for its null control -- took minutes on a
#: challenge-length message, which is not a cost a paste screen can carry.
EXHAUSTIVE_WIDTH = 6

#: Restarts for the hill climb above EXHAUSTIVE_WIDTH. Fixed-seed, so a
#: reported gap is reproducible rather than a lucky start.
CLIMB_RESTARTS = 12

#: How many of the commonest bigrams the concentration statistic counts. Ten is
#: enough to track the head of the English bigram distribution without letting
#: the long tail wash the signal out.
TOP_BIGRAMS = 10

#: A width must beat its own shuffled control by this much before it is
#: reported. 6A's real gap was 5.4 percentage points; the largest gap seen on
#: pure noise across widths 2-8 was under 1.
MINIMUM_GAP = 0.02

#: Below this there are too few bigrams for the statistic to mean anything.
MINIMUM_LETTERS = 200

#: Fixed so a rerun on the same text gives the same control, and so a reported
#: gap is reproducible rather than a lucky shuffle.
CONTROL_SEED = 0


@dataclass(frozen=True)
class Scramble:
    """A block permutation that beat its own null control.

    Attributes
    ----------
    width, permutation:
        The block length, and the order that undoes it.
    score, control:
        Bigram concentration of the un-permuted text, and the best score the
        same search reached on a shuffle of the same letters.
    text:
        The un-permuted letters, ready for the ordinary pipeline.
    """

    width: int
    permutation: tuple[int, ...]
    score: float
    control: float
    text: str

    @property
    def gap(self) -> float:
        return self.score - self.control


def concentration(letters: str, top: int = TOP_BIGRAMS) -> float:
    """Share of the text held by its ``top`` commonest bigrams.

    Substitution-invariant by construction: renaming letters renames bigrams
    without changing any count.
    """
    if len(letters) < 3:
        return 0.0
    counts = collections.Counter(
        letters[index:index + 2] for index in range(len(letters) - 1)
    )
    return sum(n for _, n in counts.most_common(top)) / (len(letters) - 1)


def apply_inverse(letters: str, permutation: Sequence[int]) -> str:
    """Undo ``permutation`` applied to every whole block of ``len(permutation)``.

    A trailing partial block is left in place rather than padded, because
    padding would invent letters the message does not contain.
    """
    width = len(permutation)
    if width < 2:
        return letters
    out: list[str] = []
    limit = len(letters) - width + 1
    for start in range(0, limit, width):
        block = letters[start:start + width]
        rebuilt = [""] * width
        for source, target in enumerate(permutation):
            rebuilt[target] = block[source]
        out.append("".join(rebuilt))
    out.append(letters[len(letters) - (len(letters) % width or width):]
               if len(letters) % width else "")
    return "".join(out)


def _best(letters: str, width: int) -> tuple[float, tuple[int, ...]]:
    """Best NON-IDENTITY rearrangement, and the permutation that reached it.

    ⚠ The identity is excluded deliberately, and leaving it in was a real
    defect caught by the first control run: at width 2 the identity "wins" on
    any English-derived text, because it changes nothing and so keeps the
    natural bigram structure, while the shuffled control has none. The
    detector then fired on 5B and 7A -- neither of which is scrambled --
    reporting the permutation ``(0, 1)``, i.e. announcing that it had undone
    nothing. A permutation that does nothing is not a finding.

    Exhaustive up to ``EXHAUSTIVE_WIDTH`` and hill-climbed above it. The first
    version searched all ``w!`` orders at every width and took MINUTES on a
    challenge-length text -- unusable on a screen whose promise is paste in,
    answer out, and doubly so because the null control doubles the work. Above
    the cut the climb tries every pairwise swap and keeps improvements, from
    several fixed-seed starts, which reaches the same optimum on this objective
    for a fraction of the evaluations.
    """
    identity = tuple(range(width))
    if width <= EXHAUSTIVE_WIDTH:
        best_score, best_perm = 0.0, identity
        for permutation in itertools.permutations(range(width)):
            if permutation == identity:
                continue
            score = concentration(apply_inverse(letters, permutation))
            if score > best_score:
                best_score, best_perm = score, permutation
        return best_score, best_perm

    generator = random.Random(CONTROL_SEED)
    best_score, best_perm = 0.0, identity
    for _ in range(CLIMB_RESTARTS):
        current = list(range(width))
        generator.shuffle(current)
        score = concentration(apply_inverse(letters, tuple(current)))
        improved = True
        while improved:
            improved = False
            for i in range(width):
                for j in range(i + 1, width):
                    current[i], current[j] = current[j], current[i]
                    trial = concentration(apply_inverse(letters, tuple(current)))
                    if trial > score:
                        score, improved = trial, True
                    else:
                        current[i], current[j] = current[j], current[i]
        if score > best_score and tuple(current) != identity:
            best_score, best_perm = score, tuple(current)
    return best_score, best_perm


def detect(letters: str, *, max_width: int = MAX_WIDTH,
           minimum_gap: float = MINIMUM_GAP) -> Scramble | None:
    """Find a block permutation that restores bigram structure, or None.

    Returns the strongest width whose score beats its OWN shuffled control by
    ``minimum_gap``. Returns None when nothing does -- which is the common case
    and must stay cheap to say.
    """
    letters = "".join(c for c in letters.upper() if c.isalpha())
    if len(letters) < MINIMUM_LETTERS:
        return None

    shuffled = list(letters)
    random.Random(CONTROL_SEED).shuffle(shuffled)
    shuffled = "".join(shuffled)

    # TWO bars, and both are needed. The shuffle control alone was not enough:
    # it only asks "is this better than noise", which any English-derived text
    # passes. The text's OWN concentration asks the question that actually
    # matters -- "is rearranging it better than LEAVING IT ALONE" -- and an
    # unscrambled cipher fails that one flat.
    leave_alone = concentration(letters)

    found: list[Scramble] = []
    for width in range(2, max_width + 1):
        score, permutation = _best(letters, width)
        control, _ = _best(shuffled, width)
        if (score - control >= minimum_gap
                and score - leave_alone >= minimum_gap):
            found.append(Scramble(
                width=width, permutation=permutation, score=score,
                control=control, text=apply_inverse(letters, permutation),
            ))
    if not found:
        return None
    # Prefer the SHORTEST width among the strong hits. A period-4 permutation
    # also shows up at width 8 as itself repeated, and reporting 8 would name a
    # longer key than the message actually uses.
    found.sort(key=lambda s: (s.width, -s.gap))
    return found[0]


METHOD = "Substitution under a block permutation"


def solve(
    source,
    *,
    scorer=None,
    top: int = 5,
    **options,
):
    """Strip a block permutation, then solve the cipher underneath.

    Returns an EMPTY set -- never a best guess -- when no scramble beats its
    own null control. That is the common case and it must stay cheap and
    silent, because a stage that always proposes something turns every message
    into a scrambled one.

    The permutation is reported in the key alongside the substitution's own
    key, so the answer says how it was reached and can be checked by hand.
    """
    from . import substitution  # local: avoids a cycle
    from .candidates import CandidateSet
    from .normalize import normalize

    text = normalize(source) if isinstance(source, str) else source
    letters = text.letters

    found = detect(letters, **{
        k: v for k, v in options.items()
        if k in {"max_width", "minimum_gap"}
    })
    if found is None:
        return CandidateSet(source_letters=letters)

    inner = substitution.solve(
        found.text, scorer=scorer, top=top,
        **{k: v for k, v in options.items()
           if k in {"seed", "time_budget", "restarts"}},
    )

    rebuilt = CandidateSet(source_letters=letters)
    for candidate in inner.ranked()[:top]:
        candidate.method = f"{METHOD} (period {found.width})"
        candidate.key = (
            f"blocks of {found.width} reordered {found.permutation}; "
            f"{candidate.key}"
        )
        candidate.diagnostics = dict(candidate.diagnostics)
        candidate.diagnostics.update({
            "scramble_width": found.width,
            "scramble_permutation": list(found.permutation),
            "order_score": found.score,
            "order_control": found.control,
            "order_gap": found.gap,
        })
        rebuilt.add(candidate)
    return rebuilt
