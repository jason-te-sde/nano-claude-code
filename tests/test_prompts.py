"""Tests for nanoclaude.prompts.

prompts.py is not under src/nanoclaude/context/, so its tests sit here at the
top level of tests/, next to test_boundaries.py and test_hygiene.py, rather
than under tests/context/ -- mirroring how every other top-level source module
in this package has its tests directly under tests/.
"""

from nanoclaude.prompts import SYSTEM_PROMPT, build_system_prompt


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
