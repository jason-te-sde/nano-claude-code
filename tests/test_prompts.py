"""Tests for nanoclaude.prompts.

prompts.py is not under src/nanoclaude/context/, so its tests sit here at the
top level of tests/, next to test_boundaries.py and test_hygiene.py, rather
than under tests/context/ -- mirroring how every other top-level source module
in this package has its tests directly under tests/.
"""

from nanoclaude.prompts import COMMAND_TOOLS, SYSTEM_PROMPT, build_system_prompt


def test_system_prompt_sets_expectations_about_tools_and_the_sandbox():
    assert "tools" in SYSTEM_PROMPT
    assert "working directory" in SYSTEM_PROMPT


def test_the_working_directory_is_always_named(tmp_path):
    prompt = build_system_prompt(str(tmp_path))
    assert f"The working directory is {tmp_path}." in prompt
    assert SYSTEM_PROMPT in prompt


def test_instructions_are_included_when_given(tmp_path):
    prompt = build_system_prompt(str(tmp_path), instructions="Never touch migrations/.")
    assert "Never touch migrations/." in prompt


def test_instructions_are_omitted_when_empty(tmp_path):
    prompt = build_system_prompt(str(tmp_path))
    assert "Project instructions" not in prompt


def test_a_tool_protocol_is_appended_when_given(tmp_path):
    prompt = build_system_prompt(str(tmp_path), tool_protocol="Use <tool> tags.")
    assert "Use <tool> tags." in prompt


def test_a_tool_protocol_is_omitted_when_not_given(tmp_path):
    prompt = build_system_prompt(str(tmp_path))
    assert "<tool>" not in prompt


#: What the prompt says that only makes sense with a tool that runs commands.
RUN_THE_TESTS = "After changing code, run the project's tests"
SHELL_COMMANDS = "over cat, grep and find"
REFUSED_COMMANDS = "Commands that could damage"


def test_a_prompt_for_tools_that_cannot_run_commands_does_not_tell_the_model_to_run_tests(tmp_path):
    for tools in ((), ("Read", "Edit", "Write", "Glob", "Grep", "TodoWrite")):
        prompt = build_system_prompt(str(tmp_path), tools=tools)
        assert RUN_THE_TESTS not in prompt, tools
        # Nor does it leave the model to find out: it is told the checks it cannot run.
        assert "cannot run commands" in prompt, tools


def test_a_prompt_for_tools_that_cannot_run_commands_names_no_shell_that_is_not_there(tmp_path):
    prompt = build_system_prompt(str(tmp_path), tools=("Read", "Grep", "Glob"))
    assert SHELL_COMMANDS not in prompt
    assert REFUSED_COMMANDS not in prompt


def test_a_prompt_for_a_tool_that_runs_commands_tells_the_model_to_run_the_tests(tmp_path):
    prompt = build_system_prompt(str(tmp_path), tools=("Read", "Bash"))
    assert RUN_THE_TESTS in prompt
    assert SHELL_COMMANDS in prompt and REFUSED_COMMANDS in prompt
    assert "cannot run commands" not in prompt


def test_the_tools_that_run_commands_are_the_ones_the_registry_could_name():
    # The one place that says which tool counts. Bash is the only one the spec has (5.1).
    assert frozenset({"Bash"}) == COMMAND_TOOLS


def test_the_constant_is_the_prompt_for_a_registry_without_a_shell(tmp_path):
    assert SYSTEM_PROMPT in build_system_prompt(str(tmp_path), tools=("Read",))
    assert RUN_THE_TESTS not in SYSTEM_PROMPT
