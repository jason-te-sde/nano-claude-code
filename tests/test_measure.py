"""scripts/measure.py is the only source of the numbers the README quotes.

It is tested like any other code, because a figure that is wrong in one place is wrong in
the README, where nobody checks it.
"""

import json
import textwrap
from pathlib import Path

import pytest

from nanoclaude.tools.registry import default_registry
from nanoclaude.tools.todo import TodoState
from tests.script_modules import load_script

measure = load_script("measure")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return path


def test_the_tool_count_is_what_the_registry_holds_and_not_a_number_written_down():
    assert measure.tool_count() == len(default_registry(TodoState()).specs())


def test_the_adapter_count_is_the_number_of_adapters_the_config_accepts():
    from nanoclaude.config.schema import KNOWN_ADAPTERS

    assert measure.adapter_count() == len(KNOWN_ADAPTERS)


@pytest.mark.parametrize(
    ("output", "count"),
    [
        ("tests/a.py::test_x\n\n2826 tests collected in 0.84s\n", 2826),
        ("1 test collected in 0.01s\n", 1),
        ("some warning mentioning a test\n17 tests collected in 2.50s\n", 17),
    ],
)
def test_the_collected_count_is_read_from_the_summary_line(output, count):
    assert measure.parse_collected(output) == count


def test_a_run_that_collected_nothing_is_an_error_and_not_a_count_of_zero():
    with pytest.raises(measure.MeasureError, match="collected"):
        measure.parse_collected("ERROR: file or directory not found: tests\n")


def test_the_test_count_is_what_pytest_collects_in_the_project(tmp_path):
    _write(
        tmp_path / "tests" / "test_a.py",
        """
        import pytest

        def test_one(): pass

        @pytest.mark.parametrize("n", [1, 2, 3])
        def test_many(n): pass
        """,
    )
    assert measure.count_tests(tmp_path) == 4


def test_a_project_whose_tests_cannot_be_collected_is_an_error_and_not_a_smaller_count(tmp_path):
    _write(tmp_path / "tests" / "test_fine.py", "def test_one():\n    pass\n")
    _write(tmp_path / "tests" / "test_broken.py", "def test_two(:\n")
    with pytest.raises(measure.MeasureError, match="could not collect"):
        measure.count_tests(tmp_path)


def test_the_lines_are_counted_in_python_files_under_src_and_under_tests(tmp_path):
    _write(tmp_path / "src" / "pkg" / "a.py", "x = 1\ny = 2\nz = 3\n")
    _write(tmp_path / "src" / "pkg" / "sub" / "b.py", "x = 1\n")
    _write(tmp_path / "src" / "pkg" / "data.toml", "a = 1\nb = 2\n")
    _write(tmp_path / "src" / "pkg" / "__pycache__" / "c.py", "ignored = True\n")
    _write(tmp_path / "tests" / "test_a.py", "a\nb\n")
    _write(tmp_path / "scripts" / "tool.py", "not counted\n")
    assert measure.source_lines(tmp_path) == {"src": 4, "tests": 2}


def test_the_coverage_is_the_percentage_pytest_cov_reports_for_the_package(tmp_path):
    _write(
        tmp_path / "mini" / "__init__.py",
        """
        FIRST = 1
        SECOND = 2
        THIRD = 3

        def pick(flag):
            if flag:
                return 1
            return 2
        """,
    )
    _write(
        tmp_path / "tests" / "test_mini.py",
        """
        from mini import pick

        def test_pick():
            assert pick(True) == 1
        """,
    )
    # Seven statements and the test runs six: the last return is not reached. 6 of 7 is
    # 85.714..., and the figure is given to a tenth.
    assert measure.coverage_percent(tmp_path, package="mini") == 85.7


def test_measuring_coverage_leaves_no_coverage_files_in_the_project(tmp_path):
    _write(tmp_path / "mini" / "__init__.py", "VALUE = 1\n")
    _write(tmp_path / "tests" / "test_mini.py", "import mini\n\ndef test_it():\n    assert mini\n")
    measure.coverage_percent(tmp_path, package="mini")
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".coverage")) == []
    assert not (tmp_path / "coverage.json").exists()


def test_a_measurement_does_not_inherit_the_coverage_run_around_it(monkeypatch):
    monkeypatch.setenv("COV_CORE_SOURCE", "somewhere")
    monkeypatch.setenv("COVERAGE_FILE", "elsewhere")
    monkeypatch.setenv("PATH_KEPT", "yes")
    environment = measure._clean_environment()
    assert "COV_CORE_SOURCE" not in environment
    assert "COVERAGE_FILE" not in environment
    assert environment["PATH_KEPT"] == "yes"


def test_a_run_that_reports_no_coverage_is_an_error(tmp_path):
    _write(tmp_path / "tests" / "test_mini.py", "def test_it():\n    pass\n")
    with pytest.raises(measure.MeasureError, match="coverage"):
        measure.coverage_percent(tmp_path, package="no_such_package")


def test_the_danger_corpus_is_not_measured_while_the_file_does_not_exist(tmp_path):
    result = measure.danger_corpus(tmp_path / "dangerous.jsonl")
    assert result["measured"] is False
    assert "not measured yet" in result["status"]
    assert "dangerous" not in result, "a figure was reported for a corpus that is not there"


def test_the_corpus_counts_the_dangerous_commands_the_regex_classifier_clears(tmp_path):
    corpus = tmp_path / "dangerous.jsonl"
    rows = [
        {"command": "rm -rf /", "label": "dangerous"},
        {"command": "rm$IFS-rf$IFS/", "label": "dangerous"},
        {"command": "r''m -rf /", "label": "dangerous"},
        {"command": "ls -la", "label": "safe"},
    ]
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n")
    result = measure.danger_corpus(corpus)
    # The first row is blocked by the pattern for rm -rf; the next two are the ones the
    # patterns cannot see through (KNOWN_BLIND_SPOTS), and the last is not dangerous.
    assert result == {
        "measured": True,
        "dangerous": 3,
        "missed_by_regex": 2,
        "status": "measured",
    }


def test_a_corpus_with_no_dangerous_row_measures_nothing_and_says_so(tmp_path):
    corpus = tmp_path / "dangerous.jsonl"
    corpus.write_text(json.dumps({"command": "ls", "label": "safe"}) + "\n")
    with pytest.raises(measure.MeasureError, match="no dangerous"):
        measure.danger_corpus(corpus)


def test_a_corpus_row_without_a_command_or_label_names_its_line(tmp_path):
    corpus = tmp_path / "dangerous.jsonl"
    corpus.write_text(json.dumps({"command": "ls", "label": "safe"}) + '\n{"command": "x"}\n')
    with pytest.raises(measure.MeasureError, match="line 2"):
        measure.danger_corpus(corpus)


def test_the_whole_is_the_cheap_figures_and_the_coverage_run(monkeypatch):
    monkeypatch.setattr(measure, "count_tests", lambda *_args, **_kwargs: 7)
    monkeypatch.setattr(measure, "run_coverage", lambda *_args, **_kwargs: (91.5, 0))
    everything = measure.measure()
    assert set(everything) == set(measure.cheap_metrics()) | {"coverage_percent", "suite_passed"}
    assert everything["coverage_percent"] == 91.5
    assert everything["suite_passed"] is True
    assert everything["tests"] == 7
    assert everything["tools"] == measure.tool_count()


def test_a_coverage_figure_from_a_run_with_failures_says_the_suite_did_not_pass(monkeypatch):
    monkeypatch.setattr(measure, "count_tests", lambda *_args, **_kwargs: 7)
    monkeypatch.setattr(measure, "run_coverage", lambda *_args, **_kwargs: (91.5, 1))
    assert measure.measure()["suite_passed"] is False


def test_a_run_that_could_not_start_the_suite_is_an_error_and_not_a_figure(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = --no-such-option\n")
    _write(tmp_path / "tests" / "test_mini.py", "def test_it():\n    pass\n")
    with pytest.raises(measure.MeasureError, match="exit 4"):
        measure.run_coverage(tmp_path, package="mini")


def test_the_metrics_are_printed_as_sorted_json_that_reads_back():
    metrics = {"tests": 5, "adapters": 3, "lines": {"tests": 2, "src": 1}}
    text = measure.render(metrics)
    assert json.loads(text) == metrics
    assert text.index('"adapters"') < text.index('"lines"') < text.index('"tests"')
    assert text.endswith("\n")
