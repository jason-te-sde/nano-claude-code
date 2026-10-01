# src/nanoclaude/tools/registry.py
"""The set of tools a session exposes and the lookup from a call to one."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from nanoclaude.providers.base import ToolSpec
from nanoclaude.tools.base import Tool


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


def default_registry() -> ToolRegistry:
    from nanoclaude.tools.edit import EditTool
    from nanoclaude.tools.read import ReadTool
    from nanoclaude.tools.write import WriteTool

    return ToolRegistry([ReadTool(), WriteTool(), EditTool()])
