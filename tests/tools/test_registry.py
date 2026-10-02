"""The brief (task-11-brief.md Step 9) ships registry.py with no accompanying
tests. These are written fresh against its committed behavior rather than
transcribed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from nanoclaude.permissions.policy import PermissionRequest
from nanoclaude.providers.base import ToolSpec
from nanoclaude.tools.base import ToolContext, ToolOutcome, ok
from nanoclaude.tools.read import ReadTool
from nanoclaude.tools.registry import ToolRegistry, UnknownToolError, default_registry
from nanoclaude.tools.todo import TodoState, TodoWriteTool


class _FakeTool:
    """A second, differently-named tool: the registry has only Read otherwise,
    which cannot by itself distinguish "returns its one tool" from "returns
    tools in sorted order".
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def read_only(self) -> bool:
        return True

    def spec(self) -> ToolSpec:
        return ToolSpec(self._name, f"{self._name} description", {"type": "object"})

    def permission_request(
        self, _ctx: ToolContext, _arguments: Mapping[str, Any]
    ) -> PermissionRequest:
        return PermissionRequest(self._name, self._name, (), is_write=False)

    async def run(
        self, _ctx: ToolContext, call_id: str, _arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        return ok(call_id, "fake")


def test_duplicate_tool_names_are_rejected_at_construction():
    with pytest.raises(ValueError, match="duplicate tool name 'Read'"):
        ToolRegistry([ReadTool(), ReadTool()])


def test_contains_reports_registered_tool_names():
    registry = ToolRegistry([ReadTool()])
    assert "Read" in registry
    assert "Write" not in registry


def test_iterating_the_registry_yields_its_tools():
    registry = ToolRegistry([ReadTool()])
    assert [tool.name for tool in registry] == ["Read"]


def test_len_counts_the_registered_tools():
    assert len(ToolRegistry([ReadTool()])) == 1
    assert len(ToolRegistry([])) == 0


def test_get_returns_the_named_tool():
    registry = ToolRegistry([ReadTool()])
    assert isinstance(registry.get("Read"), ReadTool)


def test_get_on_an_unknown_name_lists_what_is_registered_instead():
    registry = ToolRegistry([ReadTool()])
    with pytest.raises(UnknownToolError, match="no tool named 'Write'; registered: Read"):
        registry.get("Write")


def test_specs_are_sorted_by_name_regardless_of_registration_order():
    registry = ToolRegistry([_FakeTool("Zeta"), _FakeTool("Alpha"), ReadTool()])
    assert [spec.name for spec in registry.specs()] == ["Alpha", "Read", "Zeta"]


def test_default_registry_includes_read_write_edit_glob_grep_and_todowrite():
    registry = default_registry()
    assert "Read" in registry
    assert "Write" in registry
    assert "Edit" in registry
    assert "Glob" in registry
    assert "Grep" in registry
    assert "TodoWrite" in registry
    assert len(registry) == 6


async def test_default_registry_wires_a_given_todo_state_into_todowrite(ctx):
    """Tasks 24-26 construct a TodoState and call default_registry(todo) so
    that later reads of the state see what TodoWrite wrote (task-15-brief.md's
    split note). A registry that silently built its own TodoState instead
    would leave the caller's instance permanently empty.
    """
    state = TodoState()
    registry = default_registry(state)
    await registry.get("TodoWrite").run(
        ctx, "t1", {"todos": [{"content": "a", "status": "pending"}]}
    )
    assert [item.content for item in state.items] == ["a"]


def test_default_registry_without_a_todo_state_builds_its_own():
    registry = default_registry()
    assert isinstance(registry.get("TodoWrite"), TodoWriteTool)


def test_every_path_taking_tool_resolves_its_path_into_resolved_paths(ctx, tmp_repo):
    """permissions/policy.py's evaluate() and redact.py's secret-path check both
    key exclusively off PermissionRequest.resolved_paths, never .subject (see
    policy.py's module docstring and evaluate()'s rows 2-3). A tool whose
    permission_request builds a subject but forgets to also resolve it into
    resolved_paths would sail past the sandbox and the secret-path check for
    every call, silently.

    Only Read is registered as of this task; Write and Edit (Task 12) and Glob
    and Grep (Task 14) each register later and, per their briefs, take a `path`
    argument too. Walking the live registry rather than naming Read keeps this
    covering all of them without being revisited -- the final assert makes it
    fail loudly, rather than vacuously pass, should some future registry drop
    every path-taking tool.

    The call made on each tool is a *valid* one, not a minimal one: every other
    `required` string property (Write's `content`, Glob's and Grep's `pattern`)
    gets a placeholder value too, built from the tool's own schema rather than
    hand-listed per tool. permission_request is expected to raise on a genuinely
    missing required argument (see test_glob.py's and test_grep.py's own
    "raises on a missing pattern" tests, mirroring test_edit.py's), so this test
    must not be the thing supplying that missing argument.
    """
    (tmp_repo / "a.py").write_text("x = 1\n")
    expected = ctx.resolve("a.py")

    checked = 0
    for tool in default_registry():
        schema = tool.spec().schema
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping) or "path" not in properties:
            continue
        checked += 1
        arguments: dict[str, Any] = {"path": "a.py"}
        for name in schema.get("required", []):
            if name == "path":
                continue
            prop = properties.get(name, {})
            if isinstance(prop, Mapping) and prop.get("type") == "string":
                arguments[name] = "*"
        request = tool.permission_request(ctx, arguments)
        assert expected in request.resolved_paths, (
            f"{tool.name}.permission_request() did not resolve its path "
            "argument into resolved_paths"
        )

    assert checked > 0, "no tool in the registry declares a `path` argument"
