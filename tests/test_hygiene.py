import subprocess
from pathlib import Path

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
