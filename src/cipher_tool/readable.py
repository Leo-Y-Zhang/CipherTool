"""Turn a recovered plaintext into something a person can actually read.

A decryption comes back as one unbroken run of capitals, because the cipher
destroyed the spaces and the punctuation and English's own casing along with
them::

    TIMBE RANDB RUSHW OODSO MEFOU RHUND REDYA RDSDO WNSTR EAMWH ERETH EBANK

Those five-letter groups are the *ciphertext's* layout, not the message's, and
reproducing them makes a perfect decryption look like a failed one. This module
puts back the two things that can be put back honestly -- word spacing and
sentence case -- and refuses to invent the rest.

WHAT IS RESTORED, AND WHY ONLY THIS
===================================

**Spacing** comes from :meth:`~cipher_tool.scoring.EnglishScorer.segment`,
which is a guess and says so. It is gated: below
:data:`SEGMENTATION_THRESHOLD` of letters falling inside known words the split
invents breaks that are not there, and a wrong split misleads more than an
unbroken run does, because the reader blames the decryption rather than the
lexicon.

**Case** is restored in exactly two places, and both are guaranteed by English
rather than guessed:

* the first letter of the message;
* the standalone pronoun ``I`` -- the only single letter that is always
  capitalised. Measured on held-out corpus prose: **146 of 146 correct**.

Everything else is lowercased. That is a deliberate refusal, and two more
ambitious schemes were built and measured before settling on it. Both were far
worse than doing nothing:

1. **A learned list of proper nouns**, mined from the corpora -- a word is a
   name if it appears capitalised mid-sentence and rarely otherwise. Held out
   one corpus file at a time: precision 0.92 but **recall 0.14**. It can only
   capitalise names it has already met, and a real message's names are new
   ones. Eighty-six per cent of names would still come out lowercase, so the
   handful that did get capitals would look arbitrary rather than correct.

2. **Treating any word the lexicon cannot explain as a name.** This is the
   signal the toolkit already has, and it is the more tempting idea. Held out
   the same way: recall 0.67 but **precision 0.13 -- 1,197 wrong capitals for
   172 right ones**. The failures are ordinary words the lexicon happens to
   lack: ``Mare``, ``Enclosed``, ``Berth``, ``Grateful``, ``Regret``. A
   capitalised ``Mare`` in the middle of a sentence reads as a broken
   decryption; a lowercase ``mira salt`` reads as the tool declining to guess.
   Refusing is the cheaper error.

**Sentence boundaries are not restored at all.** Without punctuation there is
nothing to find them with, and the obvious model -- naive Bayes over which word
ends a sentence and which word starts the next -- was measured across all six
corpus files at **precision 0.06 to 0.21 and recall 0.02 to 0.09**. It would
scatter capitals into the middles of sentences while missing nineteen
boundaries in twenty. So the reading is one continuous lowercase run with a
single capital at the front, which is what the evidence supports.

THE INVARIANT
=============

A reading adds spaces and changes case. It does nothing else. Strip the spaces
out and upper-case what is left and you get the plaintext back, letter for
letter -- :class:`~tests.test_readable.TestLettersSurvive` pins that on
ordinary English, on text the lexicon cannot explain, and on the mixture. The
letters are what the decryption produced; the spacing and the capitals are this
module's opinion, and anything shown to a person must say which is which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .normalize import letters_only

__all__ = [
    "SEGMENTATION_THRESHOLD",
    "Reading",
    "read_plaintext",
    "sentence_case",
    "wrap_words",
]


#: How much of a plaintext must fall inside recognised words before the spaced
#: version is worth showing. Below this the split invents word breaks that are
#: not there, which makes a correct decryption look wrong.
SEGMENTATION_THRESHOLD = 0.85


def sentence_case(pieces: Sequence[str]) -> str:
    """Join *pieces* into a lower-case sentence with the two safe capitals.

    The capitals are the opening letter and the pronoun ``I``; see the module
    docstring for the measurements that ruled out every other rule tried.
    """
    words: list[str] = []
    for index, piece in enumerate(pieces):
        if not piece:
            continue
        if piece.upper() == "I":
            # The one single letter English always capitalises, wherever it
            # sits. Checked before the opening-letter rule so a message that
            # begins "I AM WRITING" is not capitalised twice over.
            words.append("I")
            continue
        word = piece.lower()
        if index == 0:
            word = word[:1].upper() + word[1:]
        words.append(word)
    return " ".join(words)


def wrap_words(text: str, width: int) -> list[str]:
    """Wrap on spaces only, so a copied line never splits a word.

    A piece longer than *width* is emitted on a line of its own and left
    intact: an unexplained run is the part of the reading a person most needs
    to see whole, and breaking it would hide that it was one run.
    """
    if not text:
        return []
    if width <= 0:
        return [text]
    lines: list[str] = []
    current = ""
    for word in text.split():
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


@dataclass(frozen=True)
class Reading:
    """A plaintext with the spaces and the case put back, plus how far to trust it.

    Attributes:
        text:
            The reading itself: words separated by single spaces, lower case
            but for the opening letter and any standalone ``I``.
        pieces:
            The word split the reading was built from, upper case, exactly as
            the segmenter returned it.
        coverage:
            Fraction of LETTERS falling inside words the lexicon holds.
        trustworthy:
            Whether the split is worth showing at all. Compare *coverage*
            against :data:`SEGMENTATION_THRESHOLD`.
        submission_safe:
            Whether EVERY piece is a word the lexicon holds -- a far stricter
            bar than *trustworthy*, and the one to use before putting a
            reading anywhere it might be pasted as an answer. Coverage counts
            letters and so cannot see a word cut in half: every fragment of
            ``CH A RLES`` is inside something, which is how a real message
            scored 0.889 while 43 of its 670 tokens were not words at all.
    """

    text: str
    pieces: tuple[str, ...]
    coverage: float
    trustworthy: bool
    submission_safe: bool

    def wrapped(self, width: int) -> list[str]:
        """The reading broken into lines of at most *width*, never mid-word."""
        return wrap_words(self.text, width)

    def __bool__(self) -> bool:
        return bool(self.text)


def read_plaintext(text: str, *, scorer: Any | None = None) -> Reading:
    """Space and case *text* for a human reader.

    *text* may be anything -- raw plaintext, five-letter groups, an already
    spaced sentence. It is reduced to letters first, so callers never have to
    normalise on our behalf and a reading of a reading is the same reading.

    Pass *scorer* when one is already built; otherwise the shared default is
    used. Nothing here is expensive except building a scorer from scratch.
    """
    if scorer is None:  # imported lazily: scoring builds a language model
        from .scoring import default_scorer

        scorer = default_scorer()

    letters = letters_only(text)
    if not letters:
        return Reading("", (), 0.0, False, False)

    pieces = tuple(scorer.segment(letters))
    if not pieces:
        return Reading("", (), 0.0, False, False)

    lexicon = scorer.lexicon
    total = sum(len(piece) for piece in pieces)
    known = sum(len(piece) for piece in pieces if piece in lexicon)
    coverage = known / total if total else 0.0

    return Reading(
        text=sentence_case(pieces),
        pieces=pieces,
        coverage=coverage,
        trustworthy=coverage >= SEGMENTATION_THRESHOLD,
        submission_safe=all(piece in lexicon for piece in pieces),
    )
