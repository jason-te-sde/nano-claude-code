"""Tests for how ``ncc`` turns its flags and its configuration into a session.

Everything is checked through what the session then does: a scripted model asks for a tool
and the result it is shown says whether the sandbox, the mode or the redactor the flag made
were the ones in force. The session itself is looked at only where nothing it does would
show the difference (the roles a flag routed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoclaude.agent.session import Session
from nanoclaude.cli.main import EXIT_CODES, build_parser
from nanoclaude.config.schema import ROLES
from nanoclaude.conversation.store import Store
from nanoclaude.permissions.policy import PermissionMode
from nanoclaude.providers.base import ModelReply
from nanoclaude.testing.scripted import calls, says
from tests.cli.helpers import run_ncc, tool_results

# --------------------------------------------------------------------------
# Where the sandbox is: --root and --add-dir
# --------------------------------------------------------------------------


def test_a_root_that_starts_with_a_tilde_is_the_home_it_names_and_not_a_directory_called_tilde(
    ncc_home, serve, capsys, tmp_path, monkeypatch
):
    # Path("~/proj").resolve() is <cwd>/~/proj: absolute, so the sandbox takes it without a
    # word, and every file the model asks for is outside it.
    (ncc_home / "proj").mkdir()
    (ncc_home / "proj" / "a.txt").write_text("hello from the project\n")
    monkeypatch.chdir(tmp_path)
    clients = serve(m=[calls("Read", {"path": "a.txt"}, call_id="r1"), says("done")])
    code, out, err = run_ncc(capsys, "--root", "~/proj", "-p", "read it")
    assert (code, out, err) == (EXIT_CODES["completed"], "done\n", "")
    (result,) = tool_results(clients["m"])
    assert not result.is_error and "hello from the project" in result.content


def test_an_add_dir_that_starts_with_a_tilde_is_the_home_it_names(
    ncc_home, project, serve, capsys, tmp_path, monkeypatch
):
    lib = ncc_home / "lib"
    lib.mkdir()
    (lib / "x.txt").write_text("library code\n")
    monkeypatch.chdir(tmp_path)
    clients = serve(m=[calls("Read", {"path": str(lib / "x.txt")}, call_id="r1"), says("done")])
    code, _, _ = run_ncc(capsys, "--root", str(project), "--add-dir", "~/lib", "-p", "read it")
    assert code == EXIT_CODES["completed"]
    (result,) = tool_results(clients["m"])
    assert not result.is_error and "library code" in result.content


def test_without_add_dir_a_directory_beside_the_project_is_outside_the_sandbox(
    ncc_home, project, serve, capsys
):
    lib = ncc_home / "lib"
    lib.mkdir()
    (lib / "x.txt").write_text("library code\n")
    clients = serve(m=[calls("Read", {"path": str(lib / "x.txt")}, call_id="r1"), says("done")])
    run_ncc(capsys, "--root", str(project), "-p", "read it")
    (result,) = tool_results(clients["m"])
    assert result.is_error and "sandbox.outside-root" in result.content


def test_a_relative_add_dir_is_relative_to_where_ncc_was_run(
    ncc_home, project, serve, capsys, monkeypatch
):
    lib = project.parent / "lib"
    lib.mkdir()
    (lib / "x.txt").write_text("library code\n")
    monkeypatch.chdir(project)
    clients = serve(m=[calls("Read", {"path": str(lib / "x.txt")}, call_id="r1"), says("done")])
    run_ncc(capsys, "--add-dir", "../lib", "-p", "read it")
    (result,) = tool_results(clients["m"])
    assert not result.is_error and "library code" in result.content


def test_the_default_root_is_the_directory_ncc_was_run_in(
    ncc_home, project, serve, capsys, monkeypatch
):
    (project / "a.txt").write_text("here\n")
    monkeypatch.chdir(project)
    clients = serve(m=[calls("Read", {"path": "a.txt"}, call_id="r1"), says("done")])
    run_ncc(capsys, "-p", "read it")
    (result,) = tool_results(clients["m"])
    assert not result.is_error and "here" in result.content


@pytest.mark.parametrize("flag", ["--root", "--add-dir"])
def test_a_directory_that_does_not_exist_is_a_usage_error_naming_the_flag(
    ncc_home, project, capsys, flag, tmp_path
):
    missing = tmp_path / "nowhere [x]"
    if flag == "--root":
        argv = ["--root", str(missing), "-p", "hi"]
    else:
        argv = ["--root", str(project), "--add-dir", str(missing), "-p", "hi"]
    code, out, err = run_ncc(capsys, *argv)
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err == f"error: {flag} {missing} is not a directory \u2014 pass one that exists\n"


def test_a_tilde_with_no_home_to_expand_to_is_a_usage_error(ncc_home, project, capsys):
    code, out, err = run_ncc(
        capsys, "--root", str(project), "--add-dir", "~no-such-user-here/lib", "-p", "hi"
    )
    assert code == EXIT_CODES["usage"]
    assert out == "" and err.startswith("error: cannot expand '~no-such-user-here/lib'")


# --------------------------------------------------------------------------
# Which model plays which role: --model and --role
# --------------------------------------------------------------------------


@pytest.mark.parametrize("flags", [["--model", "other"], ["--role", "main=other"]])
def test_the_main_role_can_be_pointed_at_another_model(ncc_home, project, serve, capsys, flags):
    clients = serve(m=[says("from m")], other=[says("from other")])
    code, out, _ = run_ncc(capsys, "--root", str(project), *flags, "-p", "hi")
    assert (code, out) == (EXIT_CODES["completed"], "from other\n")
    assert clients["m"].requests == []


def test_a_role_can_be_pointed_at_another_model_and_the_others_are_left_alone(
    ncc_home, project, serve, capsys, sessions
):
    serve(m=[says("ok")], other=[])
    run_ncc(
        capsys, "--root", str(project), "--role", "plan=other", "--role", "verify=other", "-p", "hi"
    )
    roles = sessions[0].router.config.roles
    assert (roles.plan, roles.verify) == ("other", "other")
    assert [roles.alias_for(role) for role in ROLES if role not in ("plan", "verify")] == ["m"] * 4


@pytest.mark.parametrize("pair", ["plan", "plan=", "=other", "nonsense=other", "x[/]=other"])
def test_a_role_flag_that_is_not_a_role_and_a_model_is_a_usage_error(
    ncc_home, project, capsys, pair
):
    code, out, err = run_ncc(capsys, "--root", str(project), "--role", pair, "-p", "hi")
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err.startswith(f"error: --role expects ROLE=ALIAS, not {pair!r} — ")
    assert "one of: main, explore, plan, verify, compact, title" in err


def test_a_role_pointed_at_a_model_nobody_defined_is_a_configuration_error(
    ncc_home, project, capsys
):
    code, out, err = run_ncc(capsys, "--root", str(project), "--role", "plan=ghost", "-p", "hi")
    assert code == EXIT_CODES["config"]
    assert out == ""
    assert "role 'plan' points at model 'ghost', which is not defined" in err


# --------------------------------------------------------------------------
# --max-turns
# --------------------------------------------------------------------------


def two_turns() -> list[ModelReply]:
    return [calls("Read", {"path": "a.txt"}, call_id="r1"), says("read it")]


def test_max_turns_stops_a_prompt_that_wants_more_and_says_how_to_go_on(
    ncc_home, project, serve, capsys
):
    (project / "a.txt").write_text("x\n")
    serve(m=two_turns())
    code, out, err = run_ncc(capsys, "--root", str(project), "--max-turns", "1", "-p", "go")
    assert code == EXIT_CODES["stopped"]
    assert out == ""
    assert err == (
        "error: the turn limit of 1 was reached before the task was finished "
        "— raise it with --max-turns, or give the model a smaller task\n"
    )


def test_the_same_prompt_finishes_within_a_larger_limit(ncc_home, project, serve, capsys):
    (project / "a.txt").write_text("x\n")
    serve(m=two_turns())
    code, out, _ = run_ncc(capsys, "--root", str(project), "--max-turns", "2", "-p", "go")
    assert (code, out) == (EXIT_CODES["completed"], "read it\n")


def test_a_turn_limit_is_an_error_in_the_json_with_the_loops_own_line_as_the_result(
    ncc_home, project, serve, capsys
):
    (project / "a.txt").write_text("x\n")
    serve(m=two_turns())
    code, out, _ = run_ncc(
        capsys, "--root", str(project), "--max-turns", "1", "-p", "go", "--output-format", "json"
    )
    payload = json.loads(out)
    assert code == EXIT_CODES["stopped"]
    assert payload["stop_reason"] == "turn_limit" and payload["is_error"] is True
    assert payload["result"] == "Stopped after 1 turn without finishing the task."
    assert payload["turns"] == 1


def test_the_limit_in_the_configuration_applies_when_no_flag_gives_one(
    ncc_home, project, serve, capsys
):
    config = ncc_home / ".nanoclaude" / "config.toml"
    config.write_text(config.read_text() + "\n[limits]\nmax_turns = 1\n")
    (project / "a.txt").write_text("x\n")
    serve(m=two_turns())
    code, _, err = run_ncc(capsys, "--root", str(project), "-p", "go")
    assert code == EXIT_CODES["stopped"] and "turn limit of 1" in err


@pytest.mark.parametrize("value", ["0", "-3", "many", "1.5"])
def test_max_turns_must_be_a_whole_number_of_at_least_one(capsys, value):
    code, out, err = run_ncc(capsys, "--max-turns", value, "-p", "hi")
    assert code == EXIT_CODES["usage"]
    assert out == "" and "--max-turns" in err and "at least 1" in err


# --------------------------------------------------------------------------
# Permission modes
# --------------------------------------------------------------------------

WRITE = calls("Write", {"path": "out.txt", "content": "written\n"}, call_id="w1")


def written(project: Path) -> bool:
    return (project / "out.txt").exists()


def test_headless_declines_what_the_policy_wanted_confirmed(ncc_home, project, serve, capsys):
    clients = serve(m=[WRITE, says("done")])
    code, out, _ = run_ncc(capsys, "--root", str(project), "-p", "write it")
    assert (code, out) == (EXIT_CODES["completed"], "done\n")
    assert not written(project)
    (result,) = tool_results(clients["m"])
    assert result.is_error


def test_accept_edits_lets_headless_write_files(ncc_home, project, serve, capsys):
    serve(m=[WRITE, says("done")])
    run_ncc(capsys, "--root", str(project), "--mode", "accept-edits", "-p", "write it")
    assert written(project)


def test_plan_mode_refuses_writes_by_rule(ncc_home, project, serve, capsys):
    clients = serve(m=[WRITE, says("done")])
    run_ncc(capsys, "--root", str(project), "--mode", "plan", "-p", "write it")
    assert not written(project)
    (result,) = tool_results(clients["m"])
    assert "mode.plan-read-only" in result.content


@pytest.mark.parametrize(
    "flags",
    [
        ["--dangerously-skip-permissions"],
        ["--mode", "bypass", "--dangerously-skip-permissions"],
        ["--mode=bypass", "--dangerously-skip-permissions"],
    ],
    ids=["the flag alone", "mode and flag", "mode spelled with ="],
)
def test_the_flag_with_the_long_name_turns_permission_prompts_off(
    ncc_home, project, serve, capsys, sessions, flags
):
    serve(m=[WRITE, says("done")])
    code, out, _ = run_ncc(capsys, "--root", str(project), *flags, "-p", "write it")
    assert (code, out) == (EXIT_CODES["completed"], "done\n")
    assert written(project)
    assert sessions[0].policy.mode is PermissionMode.BYPASS


@pytest.mark.parametrize(
    "flags",
    [["--mode", "bypass"], ["--mode=bypass"], ["--mode", "bypass", "--mode", "bypass"]],
    ids=["mode bypass", "mode=bypass", "twice"],
)
def test_mode_bypass_alone_is_refused_and_nothing_runs(
    ncc_home, project, serve, capsys, stores, flags
):
    clients = serve(m=[WRITE, says("done")])
    code, out, err = run_ncc(capsys, "--root", str(project), *flags, "-p", "write it")
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err == (
        "error: --mode bypass needs --dangerously-skip-permissions — pass that flag "
        "as well, or choose another mode\n"
    )
    assert not written(project)
    assert clients["m"].requests == []
    assert stores == []  # refused before anything was opened


@pytest.mark.parametrize(
    "flag",
    [
        "--dangerously-skip-permission",
        "--dangerously-skip",
        "--dangerously",
        "--dang",
        "--skip-permissions",
        "--bypass",
        "-dangerously-skip-permissions",
    ],
)
def test_no_other_spelling_of_the_flag_turns_permission_prompts_off(
    ncc_home, project, serve, capsys, flag
):
    clients = serve(m=[WRITE, says("done")])
    code, out, _ = run_ncc(capsys, "--root", str(project), flag, "-p", "write it")
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert not written(project)
    assert clients["m"].requests == []


@pytest.mark.parametrize("mode", ["default", "plan", "accept-edits"])
def test_a_mode_that_asks_for_protection_cannot_be_combined_with_turning_it_off(
    ncc_home, project, serve, capsys, mode
):
    clients = serve(m=[WRITE, says("done")])
    code, out, err = run_ncc(
        capsys,
        "--root",
        str(project),
        "--mode",
        mode,
        "--dangerously-skip-permissions",
        "-p",
        "write it",
    )
    assert code == EXIT_CODES["usage"]
    assert out == ""
    assert err == (
        f"error: --mode {mode} contradicts --dangerously-skip-permissions — pass only one of them\n"
    )
    assert not written(project) and clients["m"].requests == []


def test_a_configuration_file_cannot_turn_permission_prompts_off(ncc_home, project, serve, capsys):
    config = ncc_home / ".nanoclaude" / "config.toml"
    config.write_text(config.read_text() + '\n[permissions]\nmode = "bypass"\n')
    clients = serve(m=[WRITE, says("done")])
    code, out, err = run_ncc(capsys, "--root", str(project), "-p", "write it")
    assert code == EXIT_CODES["config"]
    assert out == "" and "mode" in err
    assert not written(project) and clients["m"].requests == []


def test_a_project_configuration_cannot_either(ncc_home, project, serve, capsys):
    (project / ".nanoclaude").mkdir()
    (project / ".nanoclaude" / "config.toml").write_text(
        '[permissions]\nmode = "bypass"\nallow = ["Write"]\n'
    )
    clients = serve(m=[WRITE, says("done")])
    code, _, _ = run_ncc(capsys, "--root", str(project), "-p", "write it")
    assert code == EXIT_CODES["config"]
    assert not written(project) and clients["m"].requests == []


def test_the_default_mode_is_not_bypass_and_no_flag_leaves_it_so(
    ncc_home, project, serve, capsys, sessions
):
    serve(m=[says("ok")])
    run_ncc(capsys, "--root", str(project), "-p", "hi")
    assert sessions[0].policy.mode is PermissionMode.DEFAULT


def test_turning_prompts_off_says_so_on_stderr_and_still_keeps_the_sandbox(
    ncc_home, project, serve, capsys
):
    escape = project.parent / "escaped.txt"
    clients = serve(
        m=[calls("Write", {"path": str(escape), "content": "x"}, call_id="w1"), says("done")]
    )
    code, out, err = run_ncc(
        capsys, "--root", str(project), "--dangerously-skip-permissions", "-p", "go"
    )
    assert (code, out) == (EXIT_CODES["completed"], "done\n")
    assert err.startswith("warning: permission prompts are off ")
    assert "sandbox" in err
    assert not escape.exists()  # spec 6.2.1: bypass skips the asking, not the sandbox
    (result,) = tool_results(clients["m"])
    assert "sandbox.outside-root" in result.content


# --------------------------------------------------------------------------
# --allow-secrets
# --------------------------------------------------------------------------

SECRET_NOTE = "the key is AKIAIOSFODNN7EXAMPLE and that is all\n"  # noqa: S105


def test_replies_are_redacted_unless_secrets_are_allowed(ncc_home, project, serve, capsys):
    (project / "notes.txt").write_text(SECRET_NOTE)
    clients = serve(m=[calls("Read", {"path": "notes.txt"}, call_id="r1"), says("done")])
    run_ncc(capsys, "--root", str(project), "-p", "read it")
    (result,) = tool_results(clients["m"])
    assert "[redacted:aws-key]" in result.content and "AKIAIOSFODNN7EXAMPLE" not in result.content


def test_allow_secrets_stops_the_redaction_and_the_refusal_of_credentials_files(
    ncc_home, project, serve, capsys
):
    (project / "notes.txt").write_text(SECRET_NOTE)
    (project / ".env").write_text("PLAIN=1\n")
    clients = serve(
        m=[
            calls("Read", {"path": "notes.txt"}, call_id="r1"),
            calls("Read", {"path": ".env"}, call_id="r2"),
            says("done"),
        ]
    )
    code, _, err = run_ncc(capsys, "--root", str(project), "--allow-secrets", "-p", "read them")
    assert code == EXIT_CODES["completed"]
    (notes,) = tool_results(clients["m"], 1)
    (env,) = tool_results(clients["m"], 2)
    assert "AKIAIOSFODNN7EXAMPLE" in notes.content
    assert not env.is_error and "PLAIN=1" in env.content
    assert err.startswith("warning: --allow-secrets is on ")


def test_without_allow_secrets_a_credentials_file_is_refused(ncc_home, project, serve, capsys):
    (project / ".env").write_text("PLAIN=1\n")
    clients = serve(m=[calls("Read", {"path": ".env"}, call_id="r1"), says("done")])
    run_ncc(capsys, "--root", str(project), "-p", "read it")
    (env,) = tool_results(clients["m"])
    assert env.is_error and "secret.path" in env.content


def test_no_warning_is_printed_when_no_switch_was_thrown(ncc_home, project, serve, capsys):
    serve(m=[says("ok")])
    _, _, err = run_ncc(capsys, "--root", str(project), "-p", "hi")
    assert err == ""


# --------------------------------------------------------------------------
# The session is wired to the rest of the configuration
# --------------------------------------------------------------------------


def test_the_session_keeps_its_files_under_the_home_ncc_was_given(
    ncc_home, project, serve, capsys, sessions
):
    serve(m=[says("ok")])
    run_ncc(capsys, "--root", str(project), "-p", "hi")
    session = sessions[0]
    assert session.root == str(project)
    assert session.home == str(ncc_home)
    assert (ncc_home / ".nanoclaude" / "sessions.db").exists()
    assert session.policy.sandbox.roots == (str(project),)


def test_instructions_in_the_home_that_ncc_was_given_reach_the_model(
    ncc_home, project, serve, capsys, tmp_path, monkeypatch
):
    # A home that ncc was told to use is not necessarily the one the shell has.
    monkeypatch.setenv("HOME", str(tmp_path / "somebody-elses-home"))
    (ncc_home / ".nanoclaude" / "NANO.md").write_text("Always answer in rhyme.\n")
    clients = serve(m=[says("ok")])
    run_ncc(capsys, "--root", str(project), "-p", "hi")
    assert "Always answer in rhyme." in clients["m"].requests[0].system


def audited_arguments(home, session_id):
    store = Store(home / ".nanoclaude" / "sessions.db")
    store.open()
    (row,) = store.db.execute(
        "SELECT args_json FROM tool_calls WHERE session_id = ?", (session_id,)
    ).fetchall()
    return row["args_json"]


@pytest.mark.parametrize(
    ("flags", "kept"), [([], False), (["--allow-secrets"], True)], ids=["default", "allowed"]
)
def test_the_audit_records_a_secret_in_a_call_only_when_secrets_are_allowed(
    ncc_home, project, serve, capsys, flags, kept
):
    write = calls("Write", {"path": "n.txt", "content": SECRET_NOTE}, call_id="w1")
    serve(m=[write, says("done")])
    _, out, _ = run_ncc(
        capsys,
        "--root",
        str(project),
        "--mode",
        "accept-edits",
        *flags,
        "-p",
        "write it",
        "--output-format",
        "json",
    )
    recorded = audited_arguments(ncc_home, json.loads(out)["session_id"])
    assert ("AKIAIOSFODNN7EXAMPLE" in recorded) is kept
    assert ("[redacted:aws-key]" in recorded) is not kept


def test_the_parser_still_takes_every_flag_the_spec_lists():
    args = build_parser().parse_args(
        [
            "-p",
            "x",
            "--output-format",
            "json",
            "--root",
            "r",
            "--model",
            "m",
            "--role",
            "plan=m",
            "--mode",
            "plan",
            "--add-dir",
            "a",
            "--add-dir",
            "b",
            "--max-turns",
            "5",
            "-c",
            "--allow-secrets",
            "--no-color",
        ]
    )
    assert (args.output_format, args.add_dir, args.max_turns) == ("json", ["a", "b"], 5)
    assert args.continue_last is True and args.mode == "plan"


def test_sessions_are_not_shared_between_runs_unless_asked(
    ncc_home, project, serve, capsys, sessions
):
    serve(m=[says("a"), says("b")])
    run_ncc(capsys, "--root", str(project), "-p", "one")
    first = sessions[0]
    run_ncc(capsys, "--root", str(project), "-p", "two")
    assert isinstance(first, Session) and sessions[1].session_id != first.session_id
