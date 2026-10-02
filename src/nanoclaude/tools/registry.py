# src/nanoclaude/tools/registry.py
"""The set of tools a session exposes and the lookup from a call to one."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

from nanoclaude.providers.base import ToolSpec
from nanoclaude.tools.base import Tool

if TYPE_CHECKING:
    # Only for the default_registry() parameter annotation below -- every
    # constructor call still goes through that function's own deferred import,
    # same as every other tool. (A bare `from __future__ import annotations`
    # does not make this unnecessary: mypy still resolves the name against
    # what is imported in this module's scope, not the callee's.)
    from nanoclaude.tools.todo import TodoState


class UnknownToolError(KeyError):
    """The model called something that is not registered.

    Its own type because it means the tool list sent to the provider and this
    registry have drifted apart -- a bug here, not a mistake by the model.
    """


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            self._tools[tool.name] = tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._tools))
            raise UnknownToolError(f"no tool named {name!r}; registered: {known}") from exc

    def specs(self) -> tuple[ToolSpec, ...]:
        """Stable order. Providers with automatic prefix caching need it."""
        return tuple(self._tools[name].spec() for name in sorted(self._tools))


def default_registry(todo_state: TodoState | None = None) -> ToolRegistry:
    from nanoclaude.tools.edit import EditTool
    from nanoclaude.tools.glob import GlobTool
    from nanoclaude.tools.grep import GrepTool
    from nanoclaude.tools.read import ReadTool
    from nanoclaude.tools.todo import TodoState, TodoWriteTool
    from nanoclaude.tools.write import WriteTool

    return ToolRegistry(
        [
            ReadTool(),
            WriteTool(),
            EditTool(),
            GlobTool(),
            GrepTool(),
            # `is not None`, not `or`: the session passes its own, initially empty
            # TodoState and must get that same object back. `or` would silently
            # swap in a fresh one the day TodoState gains a __len__.
            TodoWriteTool(todo_state if todo_state is not None else TodoState()),
        ]
    )
