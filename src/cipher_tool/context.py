"""Story and context notes supplied by the team.

The National Cipher Challenge wraps its ciphertexts in a story. That story is
evidence: it names people, places, dates and organisations, and those names
are exactly the words most likely to appear in the plaintext or to have been
used as a key. A human reading the story knows things no statistical attack
can discover.

This module gives the team somewhere to write that down, in a form the
solvers can use as **optional** cribs and keyword guesses.

What it does NOT do
-------------------
* It does not go and find context. There is no search, no lookup, no network
  access of any kind. Everything here was typed in by a person.
* It does not assume any of it is correct. Context supplies *candidates* --
  cribs to test and keywords to try. Every one is tested and scored like any
  other guess, and a context entry that leads nowhere simply scores badly.

Storage
-------
Notes live in a JSON file beside the ciphertext: ``message.txt`` gets
``message.context.json``. Plain JSON so it can be read, edited by hand,
diffed and committed alongside the ciphertext.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from .normalize import letters_only

#: The fields a team can fill in. Order matters: it is the display order.
FIELDS: tuple[tuple[str, str], ...] = (
    ("people", "Names of people in the story"),
    ("places", "Places, ships, buildings, organisations"),
    ("dates", "Dates, years, times mentioned"),
    ("phrases", "Phrases you suspect appear in the plaintext"),
    ("fragments", "Known or strongly suspected plaintext fragments"),
    ("keywords", "Words you suspect were used as a key"),
    ("notes", "Anything else worth remembering"),
)

#: Words too short to be useful as cribs. A two-letter crib fits almost
#: anywhere and rules out almost nothing.
MINIMUM_CRIB_LENGTH = 3


@dataclass
class ContextNotes:
    """Human-entered background for one ciphertext.

    Every field is a list of free-text strings. They are stored exactly as
    typed so the team can re-read them; the letters-only forms used for
    cribbing are derived on demand.
    """

    people: list[str] = field(default_factory=list)
    places: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    fragments: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # -- persistence -------------------------------------------------------

    @staticmethod
    def path_for(ciphertext_path: str | Path) -> Path:
        """Where the notes for a given ciphertext file live."""
        path = Path(ciphertext_path)
        return path.with_suffix(path.suffix + ".context.json") if path.suffix \
            else path.with_name(path.name + ".context.json")

    @classmethod
    def load(cls, ciphertext_path: str | Path) -> "ContextNotes":
        """Read the notes for *ciphertext_path*, or return empty notes.

        A missing file is normal, not an error: it just means nobody has
        written anything down yet.
        """
        target = cls.path_for(ciphertext_path)
        if not target.exists():
            return cls()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{target} is not valid JSON ({error}). Fix or delete it."
            ) from error
        if not isinstance(raw, dict):
            raise ValueError(f"{target} should contain a JSON object.")
        known = {name for name, _ in FIELDS}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"{target} has unrecognised fields: {sorted(unknown)}. "
                f"Allowed fields are: {sorted(known)}."
            )
        return cls(**{
            name: list(raw.get(name, [])) for name in known
        })

    def save(self, ciphertext_path: str | Path) -> Path:
        """Write the notes beside *ciphertext_path* and return the path."""
        target = self.path_for(ciphertext_path)
        target.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return target

    # -- editing -----------------------------------------------------------

    def add(self, field_name: str, value: str) -> None:
        """Append *value* to one field, ignoring exact duplicates."""
        known = {name for name, _ in FIELDS}
        if field_name not in known:
            raise ValueError(
                f"Unknown context field {field_name!r}. "
                f"Choose one of: {', '.join(sorted(known))}."
            )
        entry = value.strip()
        if not entry:
            raise ValueError("Refusing to add an empty context entry.")
        current: list[str] = getattr(self, field_name)
        if entry not in current:
            current.append(entry)

    def is_empty(self) -> bool:
        """True when nothing has been written down yet."""
        return all(not getattr(self, name) for name, _ in FIELDS)

    # -- derived material for the solvers ---------------------------------

    def crib_candidates(self) -> list[str]:
        """Letters-only cribs worth testing, longest first.

        Drawn from fragments, phrases, people and places -- the entries most
        likely to appear verbatim in the plaintext. Multi-word entries also
        contribute their individual words, since a plaintext might contain
        the surname without the first name.

        These are candidates to TEST. Nothing here is assumed to be present.
        """
        candidates: set[str] = set()
        for source in (self.fragments, self.phrases, self.people, self.places):
            for entry in source:
                whole = letters_only(entry)
                if len(whole) >= MINIMUM_CRIB_LENGTH:
                    candidates.add(whole)
                for word in entry.split():
                    piece = letters_only(word)
                    if len(piece) >= MINIMUM_CRIB_LENGTH:
                        candidates.add(piece)
        return sorted(candidates, key=lambda word: (-len(word), word))

    def keyword_candidates(self) -> list[str]:
        """Letters-only words worth trying as a cipher key, longest first.

        Everything a person wrote down is a plausible key: setters like
        naming a key after someone or somewhere in the story. Dates are
        included because years are sometimes spelled out.
        """
        candidates: set[str] = set()
        sources = (self.keywords, self.people, self.places, self.dates,
                   self.phrases)
        for source in sources:
            for entry in source:
                whole = letters_only(entry)
                if len(whole) >= 2:
                    candidates.add(whole)
                for word in entry.split():
                    piece = letters_only(word)
                    if len(piece) >= 2:
                        candidates.add(piece)
        return sorted(candidates, key=lambda word: (-len(word), word))

    # -- display -----------------------------------------------------------

    def render(self, ciphertext_path: str | Path | None = None) -> str:
        """Format the notes for the terminal."""
        lines: list[str] = []
        if ciphertext_path is not None:
            lines.append(f"Context notes for {ciphertext_path}")
            lines.append(f"Stored in {self.path_for(ciphertext_path)}")
            lines.append("")

        if self.is_empty():
            lines.append("No context has been recorded yet.")
            lines.append("")
            lines.append("Add some with, for example:")
            lines.append('  cipher_tool context message.txt '
                         '--add people="Admiral Harrow"')
            lines.append('  cipher_tool context message.txt '
                         '--add fragments="MEET ME AT THE HARBOUR"')
            lines.append("")
            lines.append("Fields you can use:")
            for name, description in FIELDS:
                lines.append(f"  {name:<10} {description}")
            return "\n".join(lines)

        for name, description in FIELDS:
            entries: list[str] = getattr(self, name)
            if not entries:
                continue
            lines.append(f"{name}  ({description})")
            for entry in entries:
                lines.append(f"  - {entry}")
            lines.append("")

        cribs = self.crib_candidates()
        if cribs:
            lines.append("Cribs this suggests testing (candidates only):")
            lines.append("  " + ", ".join(cribs[:20]))
            if len(cribs) > 20:
                lines.append(f"  ... and {len(cribs) - 20} more")
            lines.append("")
            lines.append("  Test one with:")
            lines.append(f'    cipher_tool crib <file> "{cribs[0]}"')
            lines.append("")

        keys = self.keyword_candidates()
        if keys:
            lines.append("Keys this suggests trying (candidates only):")
            lines.append("  " + ", ".join(keys[:20]))
            if len(keys) > 20:
                lines.append(f"  ... and {len(keys) - 20} more")
            lines.append("")
            lines.append("  Try them with:")
            lines.append(
                f'    cipher_tool vigenere <file> --key {keys[0]}'
            )
            lines.append(
                f'    cipher_tool keyword <file> --words "{",".join(keys[:5])}"'
            )
            lines.append("")

        lines.append(
            "NOTE: none of this is assumed to be correct. These are guesses "
            "for the solvers to test and score like any other."
        )
        return "\n".join(lines)


def parse_assignment(text: str) -> tuple[str, str]:
    """Split a ``field=value`` command-line assignment.

    >>> parse_assignment('people=Admiral Harrow')
    ('people', 'Admiral Harrow')
    """
    if "=" not in text:
        known = ", ".join(name for name, _ in FIELDS)
        raise ValueError(
            f"Expected field=value, got {text!r}. Fields are: {known}."
        )
    name, _, value = text.partition("=")
    return name.strip(), value.strip()


def merge_cribs(
    notes: ContextNotes, extra: Iterable[str] = (), limit: int = 40
) -> list[str]:
    """Combine context cribs with any supplied on the command line."""
    seen: list[str] = []
    for word in list(extra) + notes.crib_candidates():
        cleaned = letters_only(word)
        if len(cleaned) >= MINIMUM_CRIB_LENGTH and cleaned not in seen:
            seen.append(cleaned)
    return seen[:limit]
