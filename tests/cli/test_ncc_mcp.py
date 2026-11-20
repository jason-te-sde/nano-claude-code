"""``ncc`` with MCP servers: they are started with the session, their tools are registered, and
every one is ended when the session is, however it ends.

The model is scripted and the server is the repository's own fake one, started as a real child
process, so these are the shipped wiring from the configuration file to the model's next request.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from nanoclaude.agent.router import Router
from nanoclaude.agent.session import Session
from nanoclaude.cli import main as ncc_main
from nanoclaude.cli.main import EXIT_CODES
from nanoclaude.cli.repl import run_repl
from nanoclaude.mcp.servers import start_servers
from nanoclaude.prompts import MCP_RESULTS_ARE_DATA
from nanoclaude.providers.base import ModelReply, ModelRequest
from nanoclaude.testing.scripted import calls, says
from nanoclaude.testing.session import ScriptedClient
from tests.cli.helpers import INTERRUPTED, ScriptedPrompter, run_ncc, tool_results
from tests.mcp.helpers import (
    FAKE_SERVER,
    all_gone,
    log_of,
    pids_of,
    received_methods,
)

ECHO = calls("mcp__fake__echo", {"text": "hi"}, call_id="mcp_1")


def configure(home: Path, *tables: str) -> None:
    """Add ``tables`` to the person's configuration."""
    config = home / ".nanoclaude" / "config.toml"
    config.write_text(config.read_text() + "\n" + "\n".join(tables) + "\n")


def mcp_server(name: str, mode: str = "modern", *settings: str, enabled: bool = True) -> str:
    """The ``[mcp.<name>]`` table of a server that is the fake one in ``mode``."""
    args = ["-I", "-S", str(FAKE_SERVER), mode, *settings]
    return (
        f"[mcp.{name}]\n"
        f"command = {json.dumps(sys.executable)}\n"
        f"args = {json.dumps(args)}\n"
        f"enabled = {json.dumps(enabled)}\n"
    )


ALLOW_ECHO = '[permissions]\nallow = ["Read", "mcp__fake__echo"]\n'


# --------------------------------------------------------------------------
# The model calls an MCP tool
# --------------------------------------------------------------------------


def test_an_mcp_tool_an_allow_rule_lets_run_gives_the_next_request_its_wrapped_result(
    ncc_home, project, serve, capsys
):
    configure(ncc_home, mcp_server("fake"), ALLOW_ECHO)
    clients = serve(m=[ECHO, says("It said hi.")])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "say hi")
    assert (code, out, err) == (EXIT_CODES["completed"], "It said hi.\n", "")

    first, second = clients["m"].requests
    assert "mcp__fake__echo" in [spec.name for spec in first.tools]
    assert first.system.count(MCP_RESULTS_ARE_DATA) == 1
    (result,) = tool_results(clients["m"])
    assert (result.tool_use_id, result.is_error) == ("mcp_1", False)
    assert result.content == (
        "<mcp-result server=\"fake\" tool=\"echo\">\necho:{'text': 'hi'}\n</mcp-result>"
    )
    assert second.transcript.messages[-1].blocks[0] == result


def test_a_call_no_rule_allows_is_declined_and_the_server_is_never_asked(
    ncc_home, project, serve, capsys
):
    configure(ncc_home, mcp_server("fake"))
    clients = serve(m=[ECHO, says("I could not.")])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "say hi")
    assert (code, out) == (EXIT_CODES["completed"], "I could not.\n")
    assert err == (
        "ncc: declined (default.ask) mcp__fake__echo server fake — "
        'add "mcp__fake__echo" to allow under [permissions] in your config to let -p run it\n'
    )
    assert "tools/call" not in received_methods(log_of(ncc_home, "fake")), (
        "the server was called for a call that was declined"
    )
    (result,) = tool_results(clients["m"])
    assert result.is_error


def test_the_output_cap_in_the_configuration_is_what_an_mcp_result_is_held_to(
    ncc_home, project, serve, capsys
):
    configure(ncc_home, mcp_server("fake"), ALLOW_ECHO, "[limits]\noutput_cap_bytes = 300\n")
    clients = serve(m=[calls("mcp__fake__echo", {"text": "x" * 2000}, call_id="big"), says("ok")])
    code, _, _ = run_ncc(capsys, "--root", str(project), "-p", "say a lot")
    assert code == EXIT_CODES["completed"]
    (result,) = tool_results(clients["m"])
    assert "[output cut:" in result.content and len(result.content.encode()) < 400
    assert result.content.endswith("</mcp-result>"), "the cut took the closing tag with it"


def test_a_run_with_no_server_is_told_nothing_about_mcp_tools(ncc_home, project, serve, capsys):
    clients = serve(m=[says("hi")])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert (code, out, err) == (EXIT_CODES["completed"], "hi\n", "")
    (request,) = clients["m"].requests
    assert MCP_RESULTS_ARE_DATA not in request.system
    assert not [spec.name for spec in request.tools if spec.name.startswith("mcp__")]


# --------------------------------------------------------------------------
# A server that does not start
# --------------------------------------------------------------------------


def test_a_server_that_fails_is_one_line_on_stderr_and_the_run_goes_on(
    ncc_home, project, serve, capsys
):
    configure(
        ncc_home,
        mcp_server("bad", "crash-on-start"),
        mcp_server("fake"),
        ALLOW_ECHO,
    )
    clients = serve(m=[ECHO, says("done")])
    code, out, err = run_ncc(
        capsys, "--root", str(project), "--output-format", "json", "-p", "say hi"
    )
    assert code == EXIT_CODES["completed"], (out, err)
    assert json.loads(out)["result"] == "done", "stdout is not the one JSON object"
    (line,) = err.splitlines()
    assert line.startswith("warning: MCP server 'bad' failed to start (")
    assert "its tools are not registered" in line
    names = [spec.name for spec in clients["m"].requests[0].tools]
    assert "mcp__fake__echo" in names and not [n for n in names if n.startswith("mcp__bad__")]


def test_a_disabled_server_is_not_started_and_is_not_a_warning(ncc_home, project, serve, capsys):
    configure(ncc_home, mcp_server("fake", enabled=False))
    clients = serve(m=[says("hi")])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert (code, out, err) == (EXIT_CODES["completed"], "hi\n", "")
    assert pids_of(ncc_home, "fake") == [], "a server that is disabled was started"
    assert not [s for s in clients["m"].requests[0].tools if s.name.startswith("mcp__")]


def test_no_server_is_started_for_a_session_that_cannot_start_for_want_of_a_key(
    ncc_home, project, capsys, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    configure(ncc_home, mcp_server("fake"))
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert (code, out) == (EXIT_CODES["config"], "")
    assert "ANTHROPIC_API_KEY" in err
    assert pids_of(ncc_home, "fake") == [], "a server was started for a session that could not run"


def test_the_servers_started_for_a_session_that_cannot_be_taken_up_are_ended(
    ncc_home, project, serve, capsys
):
    configure(ncc_home, mcp_server("fake"))
    serve(m=[])
    code, out, err = run_ncc(capsys, "--root", str(project), "--resume", "nosuchid", "-p", "hello")
    assert code == EXIT_CODES["usage"] and out == "", err
    pids = pids_of(ncc_home, "fake")
    assert len(pids) == 1, "the server was not started, so nothing was checked for being ended"
    assert asyncio.run(all_gone(pids)), "the server was left running by a session that never was"


# --------------------------------------------------------------------------
# Every server is ended when the session is, however it ends
# --------------------------------------------------------------------------


def ended(home: Path) -> bool:
    return asyncio.run(all_gone(pids_of(home, "fake")))


def test_the_server_is_ended_when_a_headless_run_finishes(ncc_home, project, serve, capsys):
    configure(ncc_home, mcp_server("fake"), ALLOW_ECHO)
    serve(m=[ECHO, says("done")])
    assert run_ncc(capsys, "--root", str(project), "-p", "say hi")[0] == 0
    assert ended(ncc_home), "the server outlived a run that finished"


def test_the_server_is_ended_when_the_run_fails_with_an_exception(ncc_home, project, serve, capsys):
    from nanoclaude.providers.base import ModelError

    configure(ncc_home, mcp_server("fake"))
    serve(m=[ModelError("the provider is down", retryable=False)])
    code, _, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert code == EXIT_CODES["provider"], err
    assert ended(ncc_home), "the server outlived a run that failed"


class Interrupting(ScriptedClient):
    """A model that is waiting for its answer when the person presses Ctrl+C."""

    async def complete(self, _request, *, on_text=None):  # noqa: ARG002
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.sleep(10)
        raise AssertionError("the interrupt did not end the run")


@pytest.fixture
def python_handles_ctrl_c():
    """asyncio.run takes Ctrl+C over only where Python's own handler is the one installed."""
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, previous)


def test_the_server_is_ended_when_the_run_is_stopped_with_ctrl_c(
    ncc_home, project, capsys, monkeypatch, python_handles_ctrl_c
):
    configure(ncc_home, mcp_server("fake"))
    clients = {"m": Interrupting([])}
    monkeypatch.setattr(ncc_main, "Router", lambda config, cache: Router(config, cache, clients))
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert (code, out, err) == (EXIT_CODES["interrupted"], "", INTERRUPTED)
    assert ended(ncc_home), "the server outlived a run that was stopped with Ctrl+C"


def test_the_servers_are_ended_when_ctrl_c_stops_the_run_while_they_are_still_starting(
    ncc_home, project, serve, capsys, monkeypatch, python_handles_ctrl_c
):
    # One is up and one never answers; the person presses Ctrl+C before the session exists.
    configure(
        ncc_home,
        mcp_server("fake"),
        mcp_server("mute", "never-answers-stubborn") + "timeout = 600\n",
    )
    serve(m=[])

    async def interrupted(config: Any, **options: Any) -> Any:
        asyncio.get_running_loop().call_later(1.0, os.kill, os.getpid(), signal.SIGINT)
        graces = {"stdin_grace": 0.3, "term_grace": 0.3}
        return await start_servers(config, **options, transport_options=graces)

    monkeypatch.setattr("nanoclaude.cli.main.start_servers", interrupted)
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert (code, out, err) == (EXIT_CODES["interrupted"], "", INTERRUPTED)
    for name in ("fake", "mute"):
        pids = pids_of(ncc_home, name)
        assert len(pids) == 1, f"{name} was not started when Ctrl+C came"
        assert asyncio.run(all_gone(pids)), f"{name} was left running"


SIGNALS = [
    pytest.param(signal.SIGTERM, 143, "SIGTERM", id="SIGTERM"),
    pytest.param(signal.SIGHUP, 129, "SIGHUP", id="SIGHUP"),
]


@pytest.fixture
def a_handler_of_the_tests_own():
    """Catch SIGTERM and SIGHUP in this process, as a program that did nothing about them is
    not, so that a test of what ncc does with them cannot end the test run if it does nothing."""
    previous = {sig: signal.signal(sig, lambda *_: None) for sig in (signal.SIGTERM, signal.SIGHUP)}
    yield
    for sig, handler in previous.items():
        signal.signal(sig, handler)


@pytest.fixture
def quick_closes(monkeypatch):
    """Servers that are ended in the order of the design wait two seconds, and then two more,
    for one that ignores its stdin and SIGTERM: here they wait a third of a second."""

    async def started(config: Any, **options: Any) -> Any:
        graces = {"stdin_grace": 0.3, "term_grace": 0.3}
        return await start_servers(config, **options, transport_options=graces)

    monkeypatch.setattr("nanoclaude.cli.main.start_servers", started)


class Signalling(ScriptedClient):
    """A model that is waiting for its answer when a signal arrives."""

    def __init__(self, signum: int) -> None:
        super().__init__([])
        self.signum = signum

    async def complete(self, _request, *, on_text=None):  # noqa: ARG002
        os.kill(os.getpid(), self.signum)
        await asyncio.sleep(10)
        raise AssertionError("the signal did not end the run")


@pytest.mark.parametrize(("signum", "exit_code", "name"), SIGNALS)
def test_a_signal_ends_a_headless_run_and_every_server_with_it_even_one_that_ignores_term(
    ncc_home,
    project,
    capsys,
    monkeypatch,
    python_handles_ctrl_c,
    a_handler_of_the_tests_own,
    quick_closes,
    signum,
    exit_code,
    name,
):
    configure(ncc_home, mcp_server("fake", "ignore-term"))
    clients = {"m": Signalling(signum)}
    monkeypatch.setattr(ncc_main, "Router", lambda config, cache: Router(config, cache, clients))
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert (code, out) == (exit_code, "")
    assert err == f"error: ended by {name} \u2014 nothing more was done; run again to retry\n"
    assert ended(ncc_home), f"the server outlived a run that {name} ended"
    assert clients["m"].closed, "the session was not closed"


@pytest.mark.parametrize(("signum", "exit_code", "name"), SIGNALS)
def test_a_signal_still_gives_json_its_one_object(
    ncc_home,
    project,
    capsys,
    monkeypatch,
    python_handles_ctrl_c,
    a_handler_of_the_tests_own,
    quick_closes,
    signum,
    exit_code,
    name,
):
    configure(ncc_home, mcp_server("fake"))
    clients = {"m": Signalling(signum)}
    monkeypatch.setattr(ncc_main, "Router", lambda config, cache: Router(config, cache, clients))
    code, out, err = run_ncc(
        capsys, "--root", str(project), "--output-format", "json", "-p", "hello"
    )
    payload = json.loads(out)
    assert code == exit_code == payload["exit_code"] and payload["is_error"] is True
    assert err == f"error: {payload['error']}\n" and name in payload["error"]


@pytest.mark.parametrize(("signum", "exit_code", "name"), SIGNALS)
def test_a_signal_ends_the_repl_and_every_server_with_it(
    ncc_home,
    project,
    serve,
    capsys,
    monkeypatch,
    python_handles_ctrl_c,
    a_handler_of_the_tests_own,
    quick_closes,
    signum,
    exit_code,
    name,
):
    configure(ncc_home, mcp_server("fake", "ignore-term"))
    serve(m=[])

    async def waits(session: Session) -> None:
        os.kill(os.getpid(), signum)
        await asyncio.sleep(10)
        raise AssertionError("the signal did not end the REPL")

    repl = Repl()
    repl.body = waits
    monkeypatch.setattr(ncc_main, "run_repl", repl)
    code, out, err = run_ncc(capsys, "--root", str(project))
    assert (code, out) == (exit_code, "")
    assert name in err, err
    assert ended(ncc_home), f"the server outlived a REPL that {name} ended"


def test_a_signal_while_the_servers_are_still_starting_ends_them_too(
    ncc_home,
    project,
    serve,
    capsys,
    monkeypatch,
    python_handles_ctrl_c,
    a_handler_of_the_tests_own,
):
    configure(
        ncc_home,
        mcp_server("fake"),
        mcp_server("mute", "never-answers-stubborn") + "timeout = 600\n",
    )
    serve(m=[])

    async def terminated(config: Any, **options: Any) -> Any:
        asyncio.get_running_loop().call_later(1.0, os.kill, os.getpid(), signal.SIGTERM)
        graces = {"stdin_grace": 0.3, "term_grace": 0.3}
        return await start_servers(config, **options, transport_options=graces)

    monkeypatch.setattr("nanoclaude.cli.main.start_servers", terminated)
    code, out, _ = run_ncc(capsys, "--root", str(project), "-p", "hello")
    assert (code, out) == (143, "")
    for name in ("fake", "mute"):
        pids = pids_of(ncc_home, name)
        assert len(pids) == 1, f"{name} was not started when the signal came"
        assert asyncio.run(all_gone(pids)), f"{name} was left running"


def test_a_hangup_that_the_process_was_told_to_ignore_is_still_ignored(
    ncc_home, project, capsys, monkeypatch, python_handles_ctrl_c
):
    # nohup does this, so that a terminal that closes does not end what was started in it.
    seen: list[object] = []

    class Looking(ScriptedClient):
        async def complete(
            self, request: ModelRequest, *, on_text: Callable[[str], None] | None = None
        ) -> ModelReply:
            seen.append(signal.getsignal(signal.SIGHUP))
            return await super().complete(request, on_text=on_text)

    clients = {"m": Looking([says("hi")])}
    monkeypatch.setattr(ncc_main, "Router", lambda config, cache: Router(config, cache, clients))
    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        code, out, _ = run_ncc(capsys, "--root", str(project), "-p", "hello")
    finally:
        signal.signal(signal.SIGHUP, previous)
    assert (code, out) == (0, "hi\n")
    assert seen == [signal.SIG_IGN], f"ncc took over a hangup that was being ignored: {seen}"


class Repl:
    """A stand-in for ``run_repl`` that does ``body`` with the session and returns ``result``."""

    def __init__(self) -> None:
        self.session: Session | None = None
        self.body: Any = None
        self.result = 0

    async def __call__(self, session: Session, _console: Console, *_a: object, **_k: object) -> int:
        self.session = session
        if self.body is not None:
            await self.body(session)
        return self.result


def test_the_server_is_ended_when_the_repl_ends(ncc_home, project, serve, capsys, monkeypatch):
    configure(ncc_home, mcp_server("fake"))
    serve(m=[])
    repl = Repl()
    monkeypatch.setattr(ncc_main, "run_repl", repl)
    assert run_ncc(capsys, "--root", str(project))[0] == 0
    assert repl.session is not None and repl.session.mcp is not None
    assert ended(ncc_home), "the server outlived the REPL"


def test_the_server_is_ended_when_the_repl_fails(ncc_home, project, serve, capsys, monkeypatch):
    configure(ncc_home, mcp_server("fake"))
    serve(m=[])
    repl = Repl()

    async def breaks(_session: Session) -> None:
        raise RuntimeError("the REPL broke")

    repl.body = breaks
    monkeypatch.setattr(ncc_main, "run_repl", repl)
    code, _, err = run_ncc(capsys, "--root", str(project))
    assert code == EXIT_CODES["stopped"] and "RuntimeError" in err, err
    assert ended(ncc_home), "the server outlived a REPL that failed"


def test_the_server_is_ended_when_the_person_leaves_with_two_ctrl_c(
    ncc_home, project, serve, capsys, monkeypatch
):
    configure(ncc_home, mcp_server("fake"))
    serve(m=[])
    keyboard = ScriptedPrompter(KeyboardInterrupt(), KeyboardInterrupt())

    async def with_keyboard(
        session: Session, console: Console, history_path: str | None = None
    ) -> int:
        return await run_repl(session, console, history_path, prompt=keyboard)

    monkeypatch.setattr(ncc_main, "run_repl", with_keyboard)
    code, out, _ = run_ncc(capsys, "--root", str(project))
    assert code == EXIT_CODES["completed"] and "bye" in out
    assert ended(ncc_home), "the server outlived a REPL that was left with Ctrl+C"


def test_a_server_that_fails_is_a_warning_before_the_repl_and_the_repl_goes_on(
    ncc_home, project, serve, capsys, monkeypatch
):
    configure(ncc_home, mcp_server("bad", "crash-on-start"))
    serve(m=[])
    repl = Repl()
    monkeypatch.setattr(ncc_main, "run_repl", repl)
    code, out, err = run_ncc(capsys, "--root", str(project))
    assert code == EXIT_CODES["completed"] and out == ""
    assert err.startswith("warning: MCP server 'bad' failed to start (") and err.count("\n") == 1
    assert repl.session is not None and repl.session.mcp is not None
    assert [r.state for r in repl.session.mcp.reports()] == ["failed"]
