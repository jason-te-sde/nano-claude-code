# src/nanoclaude/tools/todo.py
"""TodoWrite: the model's own task list, rendered for the user.

It executes nothing. Its value is that keeping the list forces the model to
decompose before it starts and to come back after each step, and that the user
can see what it thinks it is doing without reading the tool calls.

One task in progress at a time is enforced rather than requested: a model that
marks four things in progress has stopped using the list as a plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from nanoclaude.permissions.policy import PermissionRequest
from nanoclaude.providers.base import ToolSpec
from nanoclaude.tools.base import ToolContext, ToolOutcome, failed, ok

Status = Literal["pending", "in_progress", "completed"]
STATUSES: tuple[str, ...] = ("pending", "in_progress", "completed")
MARKS = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}


# Hashable, deliberately: both fields are plain strings (Status is a Literal
# of str values, str at runtime) -- no list, dict or other unhashable field --
# so frozen=True's generated __hash__ never raises. Unlike ToolContext
# (tools/base.py) or ToolSpec (providers/base.py), which hold a mapping and so
# declare __hash__ = None on purpose, TodoItem has nothing that would make
# hashing unsafe. Pinned by test_todo_item_is_hashable_because_it_holds_only_strings.
@dataclass(frozen=True, slots=True)
class TodoItem:
    content: str
    status: Status


@dataclass(slots=True)
class TodoState:
    """Session-scoped. Not persisted in v0.1.

    Deliberately *not* hashable: this holds the live, in-place list for the
    session (``run()`` below reassigns ``.items`` wholesale on every call), and
    a mutable object whose contents change under it must never be used as a
    set member or dict key. dataclass's default for eq=True, frozen=False
    already sets ``__hash__ = None`` -- the same mechanism Executor
    (agent/executor.py) relies on -- so no explicit override is needed here;
    pinned by test_todo_state_is_not_hashable so a future edit (e.g. flipping
    on frozen=True) cannot make it hashable silently.
    """

    items: list[TodoItem] = field(default_factory=list)


DESCRIPTION = """Create and update the task list for this session.

Use it when a task needs three or more steps, or when the user gives you several things
to do. Mark a task `in_progress` before you start it, not after, and keep only one task
`in_progress` at a time. Mark it `completed` as soon as it is done, rather than batching
completions at the end.

Skip it for a single straightforward task: a one-item list is noise."""

SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "todos": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "status": {"type": "string", "enum": list(STATUSES)},
                },
                "required": ["content", "status"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["todos"],
    "additionalProperties": False,
}


class TodoWriteTool:
    name = "TodoWrite"
    # changes session state, but touches no files. False (not True) on purpose:
    # TodoState.items is reassigned wholesale, not merged, so two TodoWrite calls
    # racing would be a lost update, not just a harmless interleaving. The
    # executor (agent/executor.py's _consecutive_read_only_runs) only runs
    # read_only calls concurrently via asyncio.gather; False keeps every
    # TodoWrite call in a batch serialized, the same as Write and Edit.
    read_only = False

    def __init__(self, state: TodoState) -> None:
        self._state = state

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, DESCRIPTION, SCHEMA)

    def permission_request(
        self, _ctx: ToolContext, _arguments: Mapping[str, Any]
    ) -> PermissionRequest:
        return PermissionRequest(self.name, "todos", (), is_write=False)

    async def run(
        self, _ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        raw = arguments.get("todos")
        if not isinstance(raw, list) or not raw:
            return failed(call_id, "todos must be a list with at least one entry")
        items: list[TodoItem] = []
        for index, entry in enumerate(raw, start=1):
            if not isinstance(entry, dict):
                return failed(call_id, f"todo {index} must be an object")
            content, status = entry.get("content"), entry.get("status")
            if not isinstance(content, str) or not content.strip():
                return failed(call_id, f"todo {index}: content must be a non-empty string")
            if status not in STATUSES:
                return failed(call_id, f"todo {index}: status must be one of {', '.join(STATUSES)}")
            items.append(TodoItem(content.strip(), status))

        in_progress = [i for i in items if i.status == "in_progress"]
        if len(in_progress) > 1:
            return failed(
                call_id,
                f"{len(in_progress)} tasks are in_progress. Work on one task at a time.",
            )

        self._state.items = items
        rendered = "\n".join(f"{MARKS[i.status]} {i.content}" for i in items)
        done = sum(1 for i in items if i.status == "completed")
        return ok(call_id, f"{done}/{len(items)} complete\n{rendered}")
