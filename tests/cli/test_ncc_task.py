"""``ncc`` and the ``Task`` tool: what a run does with a call that starts a sub-agent.

The models are scripted and the configuration, the router, the policy and the store are real,
so these are the shipped wiring: ``Task`` is in the registry ``ncc`` builds, it asks by
default, a headless run declines it and says so, the rule that allows it is accepted in the
configuration, and the sub-agent runs on the model the ``explore`` role is routed to.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from nanoclaude.cli.main import EXIT_CODES
from nanoclaude.providers.base import Usage
from nanoclaude.providers.pricing import PriceBook
from nanoclaude.testing.scripted import calls, says
from tests.cli.helpers import run_ncc

EXIT_CODE_COMPLETED = EXIT_CODES["completed"]

TASK = calls("Task", {"agent": "explore", "prompt": "where is it defined?"}, call_id="task_1")


def test_a_headless_run_declines_a_task_call_and_says_which_rule_lets_it_run(
    ncc_home, project, serve, capsys
):
    clients = serve(m=[TASK, says("I could not search.")], other=[])
    code, out, err = run_ncc(
        capsys, "--root", str(project), "--role", "explore=other", "-p", "find it"
    )

    assert (code, out) == (EXIT_CODES["completed"], "I could not search.\n")
    assert err == (
        'ncc: declined (default.ask) Task explore — add "Task" to allow under [permissions] in '
        "your config to let -p run it\n"
    )
    assert clients["other"].requests == [], "the sub-agent was started without being allowed"


def test_a_task_call_that_a_rule_allows_runs_on_the_explore_roles_model_and_is_not_declined(
    ncc_home, project, serve, capsys
):
    config = ncc_home / ".nanoclaude" / "config.toml"
    config.write_text(config.read_text() + '\n[permissions]\nallow = ["Read", "Task"]\n')
    clients = serve(m=[TASK, says("It is in a.py.")], other=[says("a.py:3 defines it.")])
    code, out, err = run_ncc(
        capsys, "--root", str(project), "--role", "explore=other", "-p", "find it"
    )

    assert (code, out, err) == (EXIT_CODES["completed"], "It is in a.py.\n", "")
    assert len(clients["other"].requests) == 1
    assert len(clients["m"].requests) == 2, "the main model was asked twice, not three times"
    results = clients["m"].requests[1].transcript.messages[-1].blocks
    assert [(b.tool_use_id, b.content) for b in results] == [("task_1", "a.py:3 defines it.")]


def test_the_cost_of_a_sub_agent_is_in_the_json_output_and_the_audit_marks_its_calls(
    ncc_home, project, serve, capsys
):
    (project / "a.py").write_text("x = 1\n")
    config = ncc_home / ".nanoclaude" / "config.toml"
    config.write_text(config.read_text() + '\n[permissions]\nallow = ["Read", "Task"]\n')
    task = calls(
        "Task",
        {"agent": "explore", "prompt": "where is it defined?"},
        call_id="task_1",
        input_tokens=100,
        output_tokens=10,
    )
    serve(
        m=[task, says("done", input_tokens=200, output_tokens=20)],
        other=[
            calls("Read", {"path": "a.py"}, call_id="sub_1", input_tokens=700, output_tokens=40),
            says("found", input_tokens=900, output_tokens=60),
        ],
    )
    code, out, _ = run_ncc(
        capsys,
        "--root",
        str(project),
        "--role",
        "explore=other",
        "--output-format",
        "json",
        "-p",
        "find it",
    )
    assert code == EXIT_CODE_COMPLETED, out
    result = json.loads(out)
    assert result["result"] == "done"
    # The main model's 300 tokens on the main model, and the sub-agent's 1,600 and 100 on the
    # model of the explore role, which is priced differently: both are in what the run cost.
    assert result["usage"]["input_tokens"] == 300 + 1600
    assert result["usage"]["output_tokens"] == 30 + 100
    prices = PriceBook.load()
    main_cost = prices.cost(Usage(300, 30), "anthropic", "claude-sonnet-5")
    explore_cost = prices.cost(Usage(1600, 100), "anthropic", "claude-haiku-4-5")
    assert main_cost is not None and explore_cost is not None and explore_cost > 0
    assert result["cost_usd"] == pytest.approx(main_cost + explore_cost)
    assert result["cost_usd"] > main_cost, "the explore role's cost is not in the total"

    database = sqlite3.connect(ncc_home / ".nanoclaude" / "sessions.db")
    try:
        rows = database.execute(
            "SELECT tool_use_id, tool, parent_call_id, outcome FROM tool_calls ORDER BY ts, rowid"
        ).fetchall()
    finally:
        database.close()
    assert rows == [("task_1", "Task", None, "ok"), ("sub_1", "Read", "task_1", "ok")], rows
