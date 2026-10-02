# tests/tools/test_todo.py
from collections.abc import Hashable

import pytest

from nanoclaude.tools.todo import TodoItem, TodoState, TodoWriteTool

EXPECTED_DESCRIPTION = """Create and update the task list for this session.

Use it when a task needs three or more steps, or when the user gives you several things
to do. Mark a task `in_progress` before you start it, not after, and keep only one task
`in_progress` at a time. Mark it `completed` as soon as it is done, rather than batching
completions at the end.

Skip it for a single straightforward task: a one-item list is noise."""


async def test_a_list_is_stored_and_rendered(ctx):
    state = TodoState()
    outcome = await TodoWriteTool(state).run(
        ctx,
        "t1",
        {
            "todos": [
                {"content": "find callers", "status": "completed"},
                {"content": "update them", "status": "in_progress"},
                {"content": "run tests", "status": "pending"},
            ]
        },
    )
    assert not outcome.is_error
    assert len(state.items) == 3
    assert "find callers" in outcome.content


async def test_two_in_progress_items_are_rejected(ctx):
    outcome = await TodoWriteTool(TodoState()).run(
        ctx,
        "t1",
        {
            "todos": [
                {"content": "a", "status": "in_progress"},
                {"content": "b", "status": "in_progress"},
            ]
        },
    )
    assert outcome.is_error and "one task" in outcome.content


async def test_an_unknown_status_is_rejected(ctx):
    outcome = await TodoWriteTool(TodoState()).run(
        ctx, "t1", {"todos": [{"content": "a", "status": "almost"}]}
    )
    assert outcome.is_error and "status must be" in outcome.content


async def test_writing_a_new_list_replaces_the_old_one(ctx):
    state = TodoState()
    tool = TodoWriteTool(state)
    await tool.run(ctx, "t1", {"todos": [{"content": "a", "status": "pending"}]})
    await tool.run(ctx, "t2", {"todos": [{"content": "b", "status": "pending"}]})
    assert [i.content for i in state.items] == ["b"]


# -- Completeness beyond the brief's four given tests: every other refusal
# branch in run(), plus spec()/permission_request()/read_only, none of which
# the four tests above touch at all.


async def test_an_empty_todo_list_is_rejected(ctx):
    outcome = await TodoWriteTool(TodoState()).run(ctx, "t1", {"todos": []})
    assert outcome.is_error and "at least one entry" in outcome.content


async def test_a_non_list_todos_value_is_rejected(ctx):
    outcome = await TodoWriteTool(TodoState()).run(ctx, "t1", {"todos": "not-a-list"})
    assert outcome.is_error and "at least one entry" in outcome.content


async def test_a_non_object_todo_entry_is_rejected(ctx):
    outcome = await TodoWriteTool(TodoState()).run(ctx, "t1", {"todos": ["not-an-object"]})
    assert outcome.is_error and "must be an object" in outcome.content


async def test_blank_content_is_rejected(ctx):
    outcome = await TodoWriteTool(TodoState()).run(
        ctx, "t1", {"todos": [{"content": "   ", "status": "pending"}]}
    )
    assert outcome.is_error and "non-empty string" in outcome.content


async def test_non_string_content_is_rejected(ctx):
    outcome = await TodoWriteTool(TodoState()).run(
        ctx, "t1", {"todos": [{"content": 123, "status": "pending"}]}
    )
    assert outcome.is_error and "non-empty string" in outcome.content


async def test_content_is_stripped_of_surrounding_whitespace(ctx):
    state = TodoState()
    await TodoWriteTool(state).run(
        ctx, "t1", {"todos": [{"content": "  a  ", "status": "pending"}]}
    )
    assert state.items[0].content == "a"


async def test_a_fully_completed_list_reports_all_done(ctx):
    outcome = await TodoWriteTool(TodoState()).run(
        ctx, "t1", {"todos": [{"content": "a", "status": "completed"}]}
    )
    assert not outcome.is_error and "1/1 complete" in outcome.content


def test_todo_write_is_not_read_only():
    """TodoState is shared, mutable session state: run() reassigns `.items`
    wholesale (not a merge), so two TodoWrite calls racing would be a lost
    update. read_only=True would let the executor (agent/executor.py's
    _consecutive_read_only_runs) run several TodoWrite calls in one batch
    concurrently via asyncio.gather; False keeps them serialized, same as
    Write and Edit.
    """
    assert TodoWriteTool(TodoState()).read_only is False


def test_permission_request_is_a_fixed_non_write_descriptor(ctx):
    """Unlike Glob/Grep/Edit, permission_request here never inspects its
    arguments at all -- there is no path to resolve and nothing in `todos`
    changes what gets asked -- so even a nonsense payload must still return
    the same fixed, non-write descriptor rather than raising.
    """
    request = TodoWriteTool(TodoState()).permission_request(ctx, {"anything": "goes"})
    assert request.resolved_paths == ()
    assert request.is_write is False


def test_the_description_is_byte_identical_to_the_spec():
    """It is a prompt. Changing a word needs a design note, so pin the shape."""
    assert TodoWriteTool(TodoState()).spec().description == EXPECTED_DESCRIPTION


def test_todo_item_is_hashable_because_it_holds_only_strings():
    """TodoItem is frozen with two plain str fields (Status is a Literal of
    str values, str at runtime) -- no list, dict or other unhashable field --
    so dataclass's generated __hash__ never raises. Same reasoning as
    EditSpec (test_edit_algorithm.py's
    test_editspec_is_hashable_because_it_holds_only_strings_and_a_bool).
    """
    assert hash(TodoItem("a", "pending")) == hash(TodoItem("a", "pending"))
    assert len({TodoItem("a", "pending"), TodoItem("a", "completed")}) == 2


def test_todo_state_is_not_hashable():
    """Mutable session state. dataclass's default for eq=True, frozen=False
    already sets __hash__ = None -- the same mechanism Executor
    (agent/executor.py) relies on, with nothing to override explicitly -- so
    this only pins that default against a future edit (e.g. flipping on
    frozen=True) silently making it hashable while still mutable.
    """
    assert not isinstance(TodoState(), Hashable)
    with pytest.raises(TypeError, match="unhashable type: 'TodoState'"):
        hash(TodoState())
