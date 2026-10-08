"""The documentation makes claims. These check the cheap ones.

What a test here cannot check is whether a sentence is true, only that what the sentence
names exists: a tool, a command, a key, a file. That is still most of how documentation
goes wrong, since it is code that moves and the prose that stays.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DESIGN = DOCS / "design"

#: The design notes that exist. The ones the shell and the syntax-tree classifier need are
#: written with them, so they are not in this list yet: 0008, 0009 and 0011 are left for
#: when Bash, Git and that classifier are built. The numbering has gaps until then.
DESIGN_NOTES = (
    "0001-scope.md",
    "0002-pure-loop.md",
    "0003-capability-negotiation.md",
    "0004-text-tool-protocol.md",
    "0005-role-model-routing.md",
    "0006-ui-protocol.md",
    "0007-edit-by-string-not-lineno.md",
    "0010-two-phase-executor.md",
    "0012-redaction-before-transcript.md",
    "0013-thin-context.md",
    "0014-openai-compat-first-class.md",
    "0015-bypass-does-not-disable-the-sandbox.md",
    "0016-attribution-policy.md",
)

REQUIRED_SECTIONS = ("## Why", "## Costs", "## Rejected alternatives")


def _sections(text: str) -> dict[str, str]:
    """The body under each ``## `` heading of a note, by heading."""
    found: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line.strip()
            found[current] = []
        elif current is not None:
            found[current].append(line)
    return {heading: "\n".join(body).strip() for heading, body in found.items()}


def _notes() -> list[Path]:
    """Every design note there is. Never an empty list: a check that loops over none passes."""
    notes = sorted(DESIGN.glob("*.md"))
    assert len(notes) >= len(DESIGN_NOTES), (
        f"found {len(notes)} design notes where {len(DESIGN_NOTES)} are due, so a check that "
        "read them all would have read too few"
    )
    return notes


def test_the_design_notes_that_are_due_all_exist():
    present = sorted(note.name for note in DESIGN.glob("*.md"))
    missing = [name for name in DESIGN_NOTES if name not in present]
    assert missing == [], f"design notes that should exist but do not: {missing}; found {present}"


def test_every_design_note_has_the_three_required_sections_and_no_others():
    for note in _notes():
        headings = [
            line.strip() for line in note.read_text().splitlines() if line.startswith("## ")
        ]
        assert headings == list(REQUIRED_SECTIONS), (
            f"{note.name} has the sections {headings}, and must have exactly {REQUIRED_SECTIONS}"
        )


def test_no_section_of_a_design_note_is_a_heading_with_nothing_under_it():
    for note in _notes():
        for heading, body in _sections(note.read_text()).items():
            assert len(body.split()) >= 25, (
                f"{note.name}: {heading} has {len(body.split())} words, too few to say anything"
            )


def test_every_rejected_alternative_is_a_list_item_and_there_are_at_least_two():
    for note in _notes():
        body = _sections(note.read_text())["## Rejected alternatives"]
        items = [line for line in body.splitlines() if re.match(r"^(- |\d+\. )", line)]
        assert len(items) >= 2, (
            f"{note.name} rejects {len(items)} alternatives; a decision with fewer was not one"
        )


def test_a_design_note_is_named_for_its_number_and_titled_with_it():
    for note in _notes():
        number = note.name[:4]
        assert number.isdigit(), f"{note.name} does not begin with its number"
        first = note.read_text().splitlines()[0]
        assert first.startswith(f"# {number}: "), f"{note.name} is titled {first!r}"


def test_the_scope_note_lists_what_v01_does_not_do():
    text = (DESIGN / "0001-scope.md").read_text().lower()
    for absent in ("mcp", "sub-agent", "checkpoint", "hook", "windows"):
        assert absent in text, f"0001-scope.md does not say that v0.1 lacks {absent}"
