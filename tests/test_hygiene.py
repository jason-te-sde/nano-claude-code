import base64
import hashlib
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

from nanoclaude.permissions.redact import ENTROPY_FLOOR, shannon_entropy
from tests.keys import pem

ROOT = Path(__file__).resolve().parents[1]


def run_hygiene(extra_env=None):
    # "bash" is resolved from PATH deliberately, matching how the CI workflow
    # and preflight.sh both invoke it; the argument list is fixed, not
    # attacker-influenced input.
    return subprocess.run(  # noqa: S603
        ["bash", str(ROOT / "scripts" / "check-hygiene.sh")],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=extra_env,
    )


def test_repository_passes_hygiene():
    result = run_hygiene()
    assert result.returncode == 0, result.stdout + result.stderr


def test_legitimate_claude_mentions_are_not_flagged(tmp_path):
    """The check targets attribution shapes, not the word Claude."""
    sample = ROOT / "NOTICE"
    text = sample.read_text()
    assert "Anthropic" in text  # trademark statement must survive
    assert run_hygiene().returncode == 0


def test_attribution_patterns_file_has_the_five_expected_patterns():
    """The script's own count guard is only a floor (>= 5), not this exact set.

    A sixth, unrelated line would still pass that floor, and five *wrong*
    lines would pass a bare count too. This is the one place the exact
    content is pinned, and the place to update when the pattern set changes
    on purpose.

    Every pattern below is split across two source lines (or, for the robot
    emoji, written as a \\U escape instead of the literal glyph). grep
    matches per physical line, so no single line here reproduces a full
    attribution shape the way one contiguous literal would — otherwise this
    test would make check-hygiene.sh flag this file for containing exactly
    what it exists to verify. The trailing comment after each first half
    also keeps `ruff format` from rejoining the pair onto one line. Do not
    "clean up" these splits; each pair evaluates to one unbroken string.
    """
    patterns_file = ROOT / "scripts" / "attribution-patterns.txt"
    lines = patterns_file.read_text().splitlines()
    expected = [
        (
            "Co-Authored-By:"  # split: see the function's docstring
            ".*(Claude|Anthropic|noreply@anthropic)"
        ),
        (
            "Generated with "  # split: see the function's docstring
            ".*(Claude|Claude Code)"
        ),
        (
            "(written|created|authored)"  # split: see the function's docstring
            " (by|with) (Claude|an? AI|ChatGPT|Copilot)"
        ),
        "\U0001f916",  # U+1F916 ROBOT FACE, escaped rather than the literal glyph
        (
            "AI-"  # split: see the function's docstring
            "generated"
        ),
    ]
    assert sorted(lines) == sorted(expected)


# ---- no credential-shaped literal in a tracked file
#
# This repository is public, and a scanner that reads it takes a row of base64 or a random-looking
# token for what it would be anywhere else, whatever the file says it is for. A sample of either
# is built when a test runs (``tests/keys.py``, a slice of a hash, base64 of ``bytes(range(256))``)
# or is plainly fake text.

#: A run of the characters base64 is written in, sixteen or more, with the ``=`` padding a row may
#: end in. It does not continue a name: a run after ``_`` or ``-`` is the tail of an identifier or
#: of an id, not a token.
_BASE64_RUN = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9+/]{16,}={0,2}")


def credential_shaped(text: str) -> list[str]:
    """The runs in ``text`` that look like a token or a row of a key's body.

    A run does when it has letters of both cases and a digit and is random enough for the
    scrubber to take it for a secret (``ENTROPY_FLOOR``), which a key's body in base64 and a
    token have, and which a hash (no capitals), a documented example key (no lower case), a run
    of one repeated character and an identifier do not. A token that happens to lack one of the
    three is not found: this is a guard against the usual shapes, not a proof.
    """
    return [
        run
        for run in (found.group(0) for found in _BASE64_RUN.finditer(text))
        if any(c.islower() for c in run)
        and any(c.isupper() for c in run)
        and any(c.isdigit() for c in run)
        and shannon_entropy(run) >= ENTROPY_FLOOR
    ]


def tracked_text_files() -> Iterator[tuple[str, str]]:
    """Every tracked file that is text, as (name, content), but the recorded provider responses,
    whose ids are not credentials."""
    listing = subprocess.run(
        ["git", "ls-files", "-z"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for name in filter(None, listing.split("\0")):
        if name.startswith("tests/cassettes/"):
            continue
        try:
            yield name, (ROOT / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # gone from the working tree, or not text


def test_the_detector_finds_a_row_of_a_key_and_a_token_and_leaves_plain_text_alone():
    row = pem(lines=1)[1][0]
    digest = base64.b64encode(hashlib.sha256(b"a sample").digest()).decode()
    token = digest.replace("+", "x").replace("/", "y")[:22]
    assert credential_shaped(f"  {row}") == [row]
    assert credential_shaped(f'password = "{token}"') == [token]
    assert credential_shaped(f"{row[:32]}=") == [f"{row[:32]}="]
    for plain in (
        hashlib.sha256(b"a sample").hexdigest(),  # a hash has no capitals
        "AKIAIOSFODNN7EXAMPLE",  # the documented example key has no lower case
        "A" * 40,
        "a1" * 16,
        "Ab1" * 12,  # three classes, and nothing random in them
        "chatcmpl-OpenAITextFixture01",  # the tail of an id
        "call_OpenAIReadFixture01",
        "src/nanoclaude/permissions/redact.py",
        "ToolSpec/ModelRequest/ModelReply/Transcript",
    ):
        assert credential_shaped(plain) == [], plain


def test_the_scan_reads_the_sources_and_the_tests_it_is_meant_to_guard():
    """A scan that read nothing would pass: it has to be seen to read the places a literal of
    the kind has been, the library and the tests."""
    names = {name for name, _ in tracked_text_files()}
    assert "src/nanoclaude/permissions/redact.py" in names
    assert "tests/tools/test_keyblocks.py" in names


def test_no_tracked_file_holds_a_credential_shaped_literal():
    found = [
        f"{name}:{number}: {run[:6]}... ({len(run)} characters)"
        for name, text in tracked_text_files()
        for number, line in enumerate(text.splitlines(), 1)
        for run in credential_shaped(line)
    ]
    assert not found, (
        "a credential-shaped literal is in a tracked file; build it when the test runs "
        "(tests/keys.py, a slice of a hash) or write plainly fake text instead:\n"
        + "\n".join(found)
    )
