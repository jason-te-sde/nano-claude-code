#!/usr/bin/env python3
"""Produce every number the README quotes, and write them to metrics.json.

Nothing goes in the README that this script did not print: a count of tests or a percentage
of coverage is easy to state from memory and wrong within a week. ``tests/test_docs.py``
reads the README against the cheap figures here, so one that has drifted fails the suite.

The figures, and how each is made:

``tests``              how many tests ``pytest --collect-only`` finds.
``coverage_percent``   what ``pytest --cov=nanoclaude`` reports, to a tenth. It needs the whole
                       suite to run, so only this script computes it; nothing in the suite can.
``lines.src``          physical lines in the ``.py`` files under ``src/``.
``lines.tests``        the same under ``tests/``.
``tools``              how many tools the default registry holds.
``adapters``           how many adapters the configuration accepts.
``danger_corpus``      how many of the dangerous commands in ``tests/corpus/dangerous.jsonl``
                       the regex classifier clears. The corpus does not exist yet; until it
                       does this says "not measured yet" and gives no figure.

Usage: python scripts/measure.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "corpus" / "dangerous.jsonl"
PACKAGE = "nanoclaude"

_COLLECTED = re.compile(r"^(\d+) tests? collected", re.MULTILINE)


class MeasureError(RuntimeError):
    """A figure could not be produced. The message says which, and why."""


def _clean_environment() -> dict[str, str]:
    """This process's environment without what an enclosing coverage run hands its children.

    A measurement started from inside a ``pytest --cov`` run (this script's own tests do that)
    would otherwise be measured by the outer run as well, into the outer run's data.
    """
    return {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("COV_CORE_", "COVERAGE_"))
    }


def _pytest(
    root: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argument list, this interpreter
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=env if env is not None else _clean_environment(),
    )


def parse_collected(output: str) -> int:
    """The test count from the last line of ``pytest --collect-only -q``."""
    found = _COLLECTED.findall(output)
    if not found:
        raise MeasureError(
            "pytest did not say how many tests it collected; its output ended:\n"
            + "\n".join(output.splitlines()[-10:])
        )
    return int(found[-1])


def count_tests(root: Path = ROOT) -> int:
    """How many tests pytest collects in ``root``."""
    result = _pytest(root, "--collect-only")
    if result.returncode != 0:
        raise MeasureError(
            f"pytest could not collect the tests in {root} (exit {result.returncode}):\n"
            + "\n".join((result.stdout + result.stderr).splitlines()[-15:])
        )
    return parse_collected(result.stdout)


def run_coverage(root: Path = ROOT, package: str = PACKAGE) -> tuple[float, int]:
    """Run the suite under coverage. Returns the percentage and pytest's exit status.

    A suite that fails still has a coverage figure, and the status is returned so that a
    figure from a failing run is said to be one. The files coverage writes go to a temporary
    directory, so that measuring leaves the project as it found it.
    """
    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "coverage.json"
        env = {**_clean_environment(), "COVERAGE_FILE": str(Path(scratch) / ".coverage")}
        result = _pytest(root, f"--cov={package}", f"--cov-report=json:{report}", env=env)
        if result.returncode not in (0, 1):
            # 1 is "some tests failed"; anything else is pytest not running the suite at all.
            raise MeasureError(
                f"pytest could not run the suite for coverage (exit {result.returncode}):\n"
                + "\n".join((result.stdout + result.stderr).splitlines()[-15:])
            )
        if not report.is_file():
            raise MeasureError(
                f"pytest wrote no coverage report for {package!r} -- is it the package the "
                "tests import?"
            )
        data = json.loads(report.read_text(encoding="utf-8"))
    return round(data["totals"]["percent_covered"], 1), result.returncode


def coverage_percent(root: Path = ROOT, package: str = PACKAGE) -> float:
    """The percentage of ``package`` the suite in ``root`` covers, to a tenth."""
    return run_coverage(root, package)[0]


def source_lines(root: Path = ROOT) -> dict[str, int]:
    """Physical lines in the Python files under ``src/`` and under ``tests/``."""

    def count(where: str) -> int:
        files = [path for path in (root / where).rglob("*.py") if "__pycache__" not in path.parts]
        return sum(len(path.read_text(encoding="utf-8").splitlines()) for path in files)

    return {"src": count("src"), "tests": count("tests")}


def tool_count() -> int:
    """How many tools the default registry offers a model: read from it, not written down."""
    from nanoclaude.tools.registry import default_registry
    from nanoclaude.tools.todo import TodoState

    return len(default_registry(TodoState()).specs())


def adapter_count() -> int:
    """How many adapters a model entry in the configuration may name."""
    from nanoclaude.config.schema import KNOWN_ADAPTERS

    return len(KNOWN_ADAPTERS)


def danger_corpus(corpus: Path = CORPUS) -> dict[str, Any]:
    """How many of the corpus's dangerous commands the regex classifier clears.

    One JSON object per line, each with a ``command`` and a ``label``; a row labelled
    ``dangerous`` is one the classifier should not clear. Without the file nothing is
    measured and the answer says so: a figure for a corpus that is not there would be made up.
    """
    if not corpus.is_file():
        return {"measured": False, "status": f"not measured yet: {corpus.name} does not exist"}

    from nanoclaude.permissions.danger import DangerLevel
    from nanoclaude.permissions.danger.regex import RegexClassifier

    classifier = RegexClassifier()
    dangerous = missed = 0
    for number, line in enumerate(corpus.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            command, label = row["command"], row["label"]
        except (ValueError, KeyError, TypeError) as exc:
            raise MeasureError(
                f"{corpus}, line {number}: each row needs a command and a label ({exc!r})"
            ) from exc
        if label != "dangerous":
            continue
        dangerous += 1
        if classifier.classify(command).level is DangerLevel.SAFE:
            missed += 1
    if dangerous == 0:
        raise MeasureError(f"{corpus} has no dangerous rows, so there is nothing to measure")
    return {
        "measured": True,
        "dangerous": dangerous,
        "missed_by_regex": missed,
        "status": "measured",
    }


def cheap_metrics() -> dict[str, Any]:
    """Every figure that can be had without running the suite: what the docs test checks."""
    return {
        "tests": count_tests(),
        "lines": source_lines(),
        "tools": tool_count(),
        "adapters": adapter_count(),
        "danger_corpus": danger_corpus(),
    }


def measure() -> dict[str, Any]:
    """All the figures, coverage included."""
    metrics = cheap_metrics()
    percent, status = run_coverage()
    metrics["coverage_percent"] = percent
    # A coverage figure from a run with failures is a figure all the same, and says so here.
    metrics["suite_passed"] = status == 0
    return metrics


def render(metrics: dict[str, Any]) -> str:
    return json.dumps(metrics, indent=2, sort_keys=True) + "\n"


def main() -> int:
    try:
        text = render(measure())
    except MeasureError as exc:
        print(f"measure: {exc}", file=sys.stderr)
        return 1
    (ROOT / "metrics.json").write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
