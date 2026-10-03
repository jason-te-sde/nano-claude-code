"""Tests for nanoclaude.agent.executor and nanoclaude.agent.ui.

Transcribed from task-16-brief.md Step 1, with two kinds of change beyond it:

1. The three precedence notes at the top of the brief (every audited call gets
   an outcome; the refusal text format; the purity-test boundary) are pinned
   here where they touch this module -- the outcome ones get their own tests
   below, reading the ``tool_calls`` row back.
2. ``test_cancelling_the_batch_keeps_completed_results`` replaces the brief's
   ``"Bash"`` call (``{"command": "sleep 5"}``) with a monkeypatched
   ``WriteTool.run``. Task 13 (the Bash tool) is parked -- see
   ``.superpowers/sdd/2026-09-30-nano-claude-code-v0.1/progress.md`` line 1736
   ("Task 13 is parked") -- and this task's own BASE (52c26ee) only consumes
   Tasks 4, 6, 10 and 11, so ``default_registry()`` has no "Bash" tool to call.
   An unknown-tool call is refused synchronously in the decide phase and never
   suspends at all, so the brief's literal form would finish in milliseconds
   and ``task.cancel()`` would hit an already-completed task rather than an
   in-flight one. Monkeypatching gets a real ``await`` suspension point
   without building the out-of-scope Bash tool, and lets the test assert
   something the brief's version did not: that the first write's effect is
   still on disk after the second one is cancelled mid-flight.

Beyond the brief's 8 given tests, this file adds coverage the task's own
completeness bar asks for: a dedicated proof that concurrent reads are capped
at MAX_CONCURRENT_READS (not just "fast enough"), that writes to *different*
paths still run one at a time, both directions of an ALWAYS grant's scope
(does not retroact within a batch, does carry into the next one), that ONCE
does not carry over, hashability of the new frozen ``_Plan``, and direct
coverage of the UI protocol's three concrete implementations.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Hashable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Never

import pytest

from nanoclaude.agent.executor import MAX_CONCURRENT_READS, Executor, _Plan
from nanoclaude.agent.loop import RunTools, observe, start, step
from nanoclaude.agent.ui import UI, Approval, AutoApprove, AutoDecline, SilentUI
from nanoclaude.conversation.store import Store
from nanoclaude.conversation.transcript import ToolUseBlock
from nanoclaude.permissions.audit import AuditLog
from nanoclaude.permissions.policy import Decision, PermissionRequest, PermissionResult
from nanoclaude.testing.scripted import calls as model_calls
from nanoclaude.testing.scripted import calls_many
from nanoclaude.tools.base import ToolArgumentError, ToolContext, ToolOutcome, ok
from nanoclaude.tools.registry import default_registry


def executor(policy, ui=None, store=None):
    return Executor(
        registry=default_registry(),
        policy=policy,
        ui=ui or AutoApprove(),
        audit=AuditLog(store) if store is not None else None,
        session_id="s1",
    )


def _store(tmp_repo: Path) -> Store:
    store = Store(tmp_repo / "audit.db")
    store.open()
    store.create_session("s1", cwd=str(tmp_repo), roles={})
    return store


async def test_the_audit_records_the_turn_the_state_is_on_unless_told_another(policy, tmp_repo):
    (tmp_repo / "a.py").write_text("a\n")
    (tmp_repo / "b.py").write_text("b\n")
    store = _store(tmp_repo)
    batch = executor(policy, store=store)
    on_turn_two = replace(start("hi"), turn=2)
    await batch.run_batch((ToolUseBlock("t1", "Read", {"path": "a.py"}),), on_turn_two)
    await batch.run_batch((ToolUseBlock("t2", "Read", {"path": "b.py"}),), on_turn_two, turn=9)
    rows = store.db.execute("SELECT tool_use_id, turn FROM tool_calls ORDER BY tool_use_id")
    assert [(row["tool_use_id"], row["turn"]) for row in rows] == [("t1", 2), ("t2", 9)]


async def test_results_come_back_in_the_order_the_model_asked(policy, tmp_repo):
    (tmp_repo / "a.py").write_text("a\n")
    (tmp_repo / "b.py").write_text("b\n")
    calls = (
        ToolUseBlock("t1", "Read", {"path": "a.py"}),
        ToolUseBlock("t2", "Read", {"path": "b.py"}),
    )
    outcomes = await executor(policy).run_batch(calls, start("hi"))
    assert [o.tool_use_id for o in outcomes] == ["t1", "t2"]


async def test_arguments_that_do_not_match_the_schema_become_an_error_result(policy):
    """Review Focus #3. The model does this regularly; it must be correctable."""
    calls = (
        ToolUseBlock("t1", "Read", {}),  # missing required path
        ToolUseBlock("t2", "Read", {"path": 42}),  # wrong type
        ToolUseBlock("t3", "Read", {"path": "a.py", "offset": "first"}),
        ToolUseBlock("t4", "Nope", {"x": 1}),  # unknown tool
    )
    outcomes = await executor(policy).run_batch(calls, start("hi"))
    assert all(o.is_error for o in outcomes)
    # Each index pins the specific branch that produces it, not just
    # "is_error" -- t1 and t2 both go through _decide's permission_request
    # path (never reach the tool at all), t3 goes through _run's own
    # ToolArgumentError catch (the call *was* permitted; the bad argument is
    # only discovered once the tool itself inspects it), and t4 never resolves
    # to a tool at all. A regression collapsing any two of these branches
    # together would still leave the shared `all(o.is_error ...)` check green.
    assert "path must be a string" in outcomes[0].content
    assert "path must be a string" in outcomes[1].content
    assert "offset must be an integer" in outcomes[2].content
    assert "no tool named 'Nope'" in outcomes[3].content
    # Note 2's "Refused (<rule>): <reason>" is the format _decide's own DENY
    # branch (policy refusals) already used verbatim in the brief; applying it
    # to _refused()'s two early-exit paths too (t1/t2's tool.bad-arguments and
    # t4's tool.unknown, neither of which the brief's sketch wrapped) is this
    # task's own addition, made for internal consistency -- the model sees one
    # shape for every refusal, not one shape for policy refusals and a bare
    # message for the other two. Pinned explicitly here since it is the one
    # case neither a "contains path must be a string" nor an "is_error" check
    # would otherwise notice if it regressed.
    assert outcomes[0].content.startswith("Refused (tool.bad-arguments): ")
    assert outcomes[3].content.startswith("Refused (tool.unknown): ")


async def test_a_path_that_raises_a_bare_valueerror_while_resolving_is_a_bad_arguments_refusal(
    policy,
):
    """Distinct from test_arguments_that_do_not_match_the_schema...'s t1/t2:
    those never get past require_str, so they raise ToolArgumentError, caught
    by _decide's first except clause. A NUL byte is a syntactically valid
    string -- require_str accepts it -- but Path.resolve() rejects it with a
    bare ValueError, which is not a ToolArgumentError and so only the second,
    broader except clause catches it.

    The exact message is interpreter-dependent: Python 3.11 raises
    ValueError("embedded null byte"); 3.13 raises
    ValueError("lstat: embedded null character in path") (confirmed by running
    both). "embedded null" is the fragment both share.
    """
    calls = (ToolUseBlock("t1", "Read", {"path": "a\x00b"}),)
    outcomes = await executor(policy).run_batch(calls, start("hi"))
    assert outcomes[0].is_error
    assert "ValueError" in outcomes[0].content
    assert "embedded null" in outcomes[0].content


async def test_nothing_runs_before_every_call_has_been_decided(policy, tmp_repo):
    """A user who declines the first of two writes must not find the second applied."""
    target_a, target_b = tmp_repo / "a.txt", tmp_repo / "b.txt"

    class DeclineFirst(SilentUI):
        def __init__(self) -> None:
            self.seen: list[str] = []

        async def confirm(
            self, call: ToolUseBlock, _request: PermissionRequest, _result: PermissionResult
        ) -> Approval:
            self.seen.append(call.id)
            return Approval.NO if call.id == "t1" else Approval.ONCE

    ui = DeclineFirst()
    calls = (
        ToolUseBlock("t1", "Write", {"path": "a.txt", "content": "A"}),
        ToolUseBlock("t2", "Write", {"path": "b.txt", "content": "B"}),
    )
    outcomes = await executor(policy, ui).run_batch(calls, start("hi"))
    assert ui.seen == ["t1", "t2"]  # both asked, in order
    assert outcomes[0].is_error and "declined" in outcomes[0].content
    assert not target_a.exists()
    assert target_b.read_text() == "B"


async def test_a_denied_call_is_never_executed(policy, tmp_repo):
    calls = (ToolUseBlock("t1", "Read", {"path": "/etc/passwd"}),)
    outcomes = await executor(policy).run_batch(calls, start("hi"))
    assert outcomes[0].is_error
    assert "sandbox.outside-root" in outcomes[0].content


async def test_always_upgrades_the_rest_of_the_session(policy, tmp_repo):
    class AlwaysAllow(SilentUI):
        async def confirm(
            self, _call: ToolUseBlock, _request: PermissionRequest, _result: PermissionResult
        ) -> Approval:
            return Approval.ALWAYS

    ex = executor(policy, AlwaysAllow())
    calls = (ToolUseBlock("t1", "Write", {"path": "a.txt", "content": "A"}),)
    await ex.run_batch(calls, start("hi"))
    assert "Write" in ex.grants.tools


async def test_read_only_calls_run_concurrently(policy, tmp_repo):
    for name in "abcd":
        (tmp_repo / f"{name}.py").write_text(name)
    calls = tuple(
        ToolUseBlock(f"t{i}", "Read", {"path": f"{name}.py"}) for i, name in enumerate("abcd")
    )
    ex = executor(policy)
    started = asyncio.get_running_loop().time()
    await ex.run_batch(calls, start("hi"))
    assert asyncio.get_running_loop().time() - started < 2.0


async def test_writes_to_the_same_path_in_one_batch_are_serialised(policy, tmp_repo):
    calls = (
        ToolUseBlock("t1", "Write", {"path": "a.txt", "content": "first"}),
        ToolUseBlock("t2", "Write", {"path": "a.txt", "content": "second"}),
    )
    outcomes = await executor(policy).run_batch(calls, start("hi"))
    # The second write needs the file to have been read, so it fails -- which is
    # itself the proof that they ran in order rather than racing.
    assert not outcomes[0].is_error
    assert outcomes[1].is_error
    assert (tmp_repo / "a.txt").read_text() == "first"


async def test_a_tool_raising_an_unexpected_oserror_does_not_kill_the_batch(
    policy, tmp_repo, monkeypatch
):
    from nanoclaude.tools import read as read_module

    def boom(path: str) -> Never:
        raise OSError("device not configured")

    monkeypatch.setattr(read_module, "read_text", boom)
    (tmp_repo / "a.py").write_text("x")
    calls = (ToolUseBlock("t1", "Read", {"path": "a.py"}),)
    outcomes = await executor(policy).run_batch(calls, start("hi"))
    assert outcomes[0].is_error and "device not configured" in outcomes[0].content


async def test_cancelling_the_batch_keeps_completed_results(policy, tmp_repo, monkeypatch):
    """See the module docstring: ``"Bash"``/``"sleep 5"`` is replaced by a
    monkeypatched ``WriteTool.run`` that only sleeps for call ``t2``, so t1's
    write is genuinely real and t2's is genuinely slow and cancellable. This
    also lets the test assert the thing its name claims -- that completed
    results are kept -- rather than only that cancellation raises.
    """
    from nanoclaude.tools.write import WriteTool

    real_run = WriteTool.run

    async def slow_for_t2(
        self: WriteTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        if call_id == "t2":
            await asyncio.sleep(5)
            return ok(call_id, "done")
        return await real_run(self, ctx, call_id, arguments)

    monkeypatch.setattr(WriteTool, "run", slow_for_t2)
    ex = executor(policy)
    calls = (
        ToolUseBlock("t1", "Write", {"path": "a.txt", "content": "A"}),
        ToolUseBlock("t2", "Write", {"path": "b.txt", "content": "B"}),
    )
    task = asyncio.create_task(ex.run_batch(calls, start("hi")))
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (tmp_repo / "a.txt").read_text() == "A"
    assert not (tmp_repo / "b.txt").exists()


# Tests below close gaps beyond the brief's 8 given tests: the "every audited
# call gets an outcome" requirement (note 1) needs its own tests reading the
# tool_calls row back, since none of the tests above wire up a real AuditLog;
# hashability of the new frozen _Plan, per this task's own hashability rule;
# a real proof that MAX_CONCURRENT_READS caps concurrency rather than merely
# completing quickly; serial writes to *different* paths; both directions of
# an ALWAYS grant's scope; ONCE not persisting; and the UI protocol's three
# concrete implementations.


async def test_a_declined_call_is_recorded_with_outcome_declined_and_zero_duration(
    policy, tmp_repo
):
    store = _store(tmp_repo)
    ex = executor(policy, AutoDecline(), store)
    calls = (ToolUseBlock("t1", "Write", {"path": "a.txt", "content": "A"}),)
    outcomes = await ex.run_batch(calls, start("hi"))
    assert outcomes[0].is_error
    row = store.db.execute(
        "SELECT outcome, duration_ms FROM tool_calls WHERE tool_use_id = ?", ("t1",)
    ).fetchone()
    assert (row["outcome"], row["duration_ms"]) == ("declined", 0)


async def test_a_refused_call_is_recorded_with_outcome_refused_and_zero_duration(policy, tmp_repo):
    store = _store(tmp_repo)
    ex = executor(policy, AutoApprove(), store)
    calls = (ToolUseBlock("t1", "Read", {"path": "/etc/passwd"}),)
    outcomes = await ex.run_batch(calls, start("hi"))
    assert outcomes[0].is_error
    row = store.db.execute(
        "SELECT outcome, duration_ms FROM tool_calls WHERE tool_use_id = ?", ("t1",)
    ).fetchone()
    assert (row["outcome"], row["duration_ms"]) == ("refused", 0)


async def test_an_unknown_tool_refusal_is_also_recorded_with_an_outcome(policy, tmp_repo):
    """Distinct from the DENY-by-policy case above: this refusal never reaches
    evaluate() at all (registry.get() raises first), so it is a different code
    path to the same "every audited call gets an outcome" requirement.
    """
    store = _store(tmp_repo)
    ex = executor(policy, AutoApprove(), store)
    outcomes = await ex.run_batch((ToolUseBlock("t1", "Nope", {}),), start("hi"))
    assert outcomes[0].is_error
    row = store.db.execute(
        "SELECT outcome, duration_ms FROM tool_calls WHERE tool_use_id = ?", ("t1",)
    ).fetchone()
    assert (row["outcome"], row["duration_ms"]) == ("refused", 0)


async def test_an_ok_call_is_still_recorded_with_a_real_duration(policy, tmp_repo):
    """The companion case: a NULL outcome must mean only "never finished", so
    a call that *did* finish must not also show outcome IS NULL.
    """
    store = _store(tmp_repo)
    (tmp_repo / "a.py").write_text("x")
    ex = executor(policy, AutoApprove(), store)
    outcomes = await ex.run_batch((ToolUseBlock("t1", "Read", {"path": "a.py"}),), start("hi"))
    assert not outcomes[0].is_error
    row = store.db.execute(
        "SELECT outcome, duration_ms FROM tool_calls WHERE tool_use_id = ?", ("t1",)
    ).fetchone()
    assert row["outcome"] == "ok"
    assert row["duration_ms"] is not None


def test_plan_is_not_hashable():
    # Mirrors LoopState/Done/RunTools/ToolUseBlock/Message/Transcript/
    # ModelRequest/ModelReply/ToolSpec/ToolContext: frozen=True with the
    # default eq=True would generate a real __hash__ that lies about being
    # usable (isinstance(x, Hashable) is True) and only raises when actually
    # called, naming "ToolUseBlock" (the field that is genuinely unhashable)
    # rather than this class. Constructed directly, the same way
    # test_loop.py builds an otherwise-impossible LoopState by hand.
    plan = _Plan(
        ToolUseBlock("t1", "Read", {"path": "a.py"}),
        None,
        PermissionRequest("Read", "a.py"),
        PermissionResult(Decision.ALLOW, "tool.read-only", "Read only reads"),
        None,
    )
    assert not isinstance(plan, Hashable)  # type: ignore[unreachable]
    with pytest.raises(TypeError, match="unhashable type: '_Plan'"):
        hash(plan)


async def test_concurrent_reads_are_capped_at_max_concurrent_reads(policy, tmp_repo, monkeypatch):
    """ "Completes quickly" (the brief's own concurrency test, above) would
    pass just as well if every read ran serially -- each is fast regardless.
    This instead measures the actual peak number of reads in flight at once
    against MAX_CONCURRENT_READS directly, with more calls than the cap.
    """
    from nanoclaude.tools.read import ReadTool

    (tmp_repo / "a.py").write_text("x")
    current = 0
    peak = 0

    async def tracked_run(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.05)
        current -= 1
        return ok(call_id, "ok")

    monkeypatch.setattr(ReadTool, "run", tracked_run)
    calls = tuple(
        ToolUseBlock(f"t{i}", "Read", {"path": "a.py"}) for i in range(MAX_CONCURRENT_READS + 4)
    )
    await executor(policy).run_batch(calls, start("hi"))
    assert peak == MAX_CONCURRENT_READS


async def test_writes_to_different_paths_still_run_one_at_a_time(policy, tmp_repo, monkeypatch):
    """The same-path test above proves ordering by way of the read-before-write
    invariant; it cannot tell "serialised" apart from "merely not racing on
    this one path". This instead instruments two *unrelated* paths directly:
    every write's run() call must fully finish (both its start and its end
    recorded) before the next one's run() is even invoked.
    """
    from nanoclaude.tools.write import WriteTool

    real_run = WriteTool.run
    order: list[str] = []

    async def tracked_run(
        self: WriteTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        order.append(f"{call_id}:start")
        result = await real_run(self, ctx, call_id, arguments)
        order.append(f"{call_id}:end")
        return result

    monkeypatch.setattr(WriteTool, "run", tracked_run)
    calls = (
        ToolUseBlock("t1", "Write", {"path": "a.txt", "content": "A"}),
        ToolUseBlock("t2", "Write", {"path": "b.txt", "content": "B"}),
    )
    await executor(policy).run_batch(calls, start("hi"))
    assert order == ["t1:start", "t1:end", "t2:start", "t2:end"]


async def test_an_always_grant_does_not_apply_within_the_same_batch(policy, tmp_repo):
    """Both directions of the ALWAYS scope, part 1: a grant issued by an
    earlier call in a batch must not retroactively approve a later call in
    that *same* batch for the same tool, because every call's decision is
    taken from one snapshot of self.grants, before any of them is confirmed.
    """
    seen: list[str] = []

    class AlwaysAllow(SilentUI):
        async def confirm(
            self, call: ToolUseBlock, _request: PermissionRequest, _result: PermissionResult
        ) -> Approval:
            seen.append(call.id)
            return Approval.ALWAYS

    calls = (
        ToolUseBlock("t1", "Write", {"path": "a.txt", "content": "A"}),
        ToolUseBlock("t2", "Write", {"path": "b.txt", "content": "B"}),
    )
    await executor(policy, AlwaysAllow()).run_batch(calls, start("hi"))
    # If the grant from t1 applied retroactively, t2's decision would already
    # be ALLOW by the time it is considered and confirm() would never be
    # called for it -- seen would be ["t1"] only.
    assert seen == ["t1", "t2"]


async def test_a_session_grant_from_an_earlier_batch_is_honoured_by_a_later_one(policy, tmp_repo):
    """Both directions of the ALWAYS scope, part 2: a grant that already
    exists at the start of a batch (as if set by an ALWAYS in an earlier one,
    per test_always_upgrades_the_rest_of_the_session above) *does* apply --
    AutoDecline would refuse if it were ever asked, so success here proves the
    call was never asked at all.
    """
    ex = executor(policy, AutoDecline())
    ex.grants = ex.grants.with_tool("Write")
    outcomes = await ex.run_batch(
        (ToolUseBlock("t1", "Write", {"path": "a.txt", "content": "A"}),), start("hi")
    )
    assert not outcomes[0].is_error
    assert (tmp_repo / "a.txt").read_text() == "A"


async def test_once_does_not_persist_to_a_later_batch(policy, tmp_repo):
    ex = executor(policy, AutoApprove())  # AutoApprove answers ONCE, never ALWAYS
    await ex.run_batch(
        (ToolUseBlock("t1", "Write", {"path": "a.txt", "content": "A"}),), start("hi")
    )
    assert "Write" not in ex.grants.tools
    ex.ui = AutoDecline()
    outcomes = await ex.run_batch(
        (ToolUseBlock("t2", "Write", {"path": "b.txt", "content": "B"}),), start("hi")
    )
    assert outcomes[0].is_error
    assert not (tmp_repo / "b.txt").exists()


async def test_silent_ui_declines_every_confirmation():
    ui = SilentUI()
    assert isinstance(ui, UI)
    call = ToolUseBlock("t1", "Write", {"path": "a.txt", "content": "x"})
    request = PermissionRequest("Write", "a.txt", is_write=True)
    result = PermissionResult(Decision.ASK, "default.ask", "needs confirmation")
    assert await ui.confirm(call, request, result) is Approval.NO


async def test_auto_approve_answers_once():
    ui = AutoApprove()
    assert isinstance(ui, UI)
    call = ToolUseBlock("t1", "Write", {"path": "a.txt", "content": "x"})
    request = PermissionRequest("Write", "a.txt", is_write=True)
    result = PermissionResult(Decision.ASK, "default.ask", "needs confirmation")
    assert await ui.confirm(call, request, result) is Approval.ONCE


async def test_auto_decline_behaves_like_silent_ui():
    ui = AutoDecline()
    assert isinstance(ui, UI)
    call = ToolUseBlock("t1", "Write", {"path": "a.txt", "content": "x"})
    request = PermissionRequest("Write", "a.txt", is_write=True)
    result = PermissionResult(Decision.ASK, "default.ask", "needs confirmation")
    assert await ui.confirm(call, request, result) is Approval.NO


# Tests below pin the coordinator's review-round fixes: a write is a barrier
# that preserves request order across tool classes (not just within one), and
# on_decision now fires for every refusal, not only the ones that reach
# evaluate().


async def test_a_write_immediately_followed_by_a_read_sees_the_write(policy, tmp_repo):
    """The bug this section fixes: the old scheduling ran every read-only
    call before every serial call regardless of request order, so a Read
    placed after a Write in the same batch still saw whatever was on disk
    *before* the batch ran.
    """
    calls = (
        ToolUseBlock("t1", "Write", {"path": "x.txt", "content": "first\n"}),
        ToolUseBlock("t2", "Read", {"path": "x.txt"}),
    )
    outcomes = await executor(policy).run_batch(calls, start("hi"))
    assert not outcomes[0].is_error
    assert "first" in outcomes[1].content


async def test_scheduling_groups_consecutive_reads_and_runs_a_write_alone_between_them(
    policy, tmp_repo, monkeypatch
):
    """[Read, Read, Write, Read, Read]: the first two overlap, the write runs
    fully alone (no read overlaps it), and the last two overlap. Also covers
    "a mixed batch keeps call order": outcomes come back in request order
    regardless of which calls ran concurrently.
    """
    from nanoclaude.tools.read import ReadTool
    from nanoclaude.tools.write import WriteTool

    order: list[str] = []

    async def tracked_read(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        order.append(f"{call_id}:start")
        await asyncio.sleep(0.05)
        order.append(f"{call_id}:end")
        return ok(call_id, "read")

    async def tracked_write(
        self: WriteTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        order.append(f"{call_id}:start")
        await asyncio.sleep(0.05)
        order.append(f"{call_id}:end")
        return ok(call_id, "written")

    monkeypatch.setattr(ReadTool, "run", tracked_read)
    monkeypatch.setattr(WriteTool, "run", tracked_write)
    calls = (
        ToolUseBlock("t1", "Read", {"path": "a.py"}),
        ToolUseBlock("t2", "Read", {"path": "b.py"}),
        ToolUseBlock("t3", "Write", {"path": "c.txt", "content": "C"}),
        ToolUseBlock("t4", "Read", {"path": "d.py"}),
        ToolUseBlock("t5", "Read", {"path": "e.py"}),
    )
    outcomes = await executor(policy).run_batch(calls, start("hi"))
    assert [o.tool_use_id for o in outcomes] == ["t1", "t2", "t3", "t4", "t5"]
    assert order == [
        "t1:start",
        "t2:start",
        "t1:end",
        "t2:end",
        "t3:start",
        "t3:end",
        "t4:start",
        "t5:start",
        "t4:end",
        "t5:end",
    ]


async def test_the_cap_holds_within_a_read_group_inside_a_mixed_batch(
    policy, tmp_repo, monkeypatch
):
    """test_concurrent_reads_are_capped_at_max_concurrent_reads (above) covers
    an all-read batch, which is a single group under the new scheduling --
    indistinguishable from the old one. This instead puts the oversized read
    group ahead of a write in the same batch, so the group being measured is
    not the whole batch.
    """
    from nanoclaude.tools.read import ReadTool
    from nanoclaude.tools.write import WriteTool

    current = 0
    peak = 0

    async def tracked_read(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.02)
        current -= 1
        return ok(call_id, "read")

    async def stub_write(
        self: WriteTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        return ok(call_id, "written")

    monkeypatch.setattr(ReadTool, "run", tracked_read)
    monkeypatch.setattr(WriteTool, "run", stub_write)
    reads = tuple(
        ToolUseBlock(f"r{i}", "Read", {"path": "a.py"}) for i in range(MAX_CONCURRENT_READS + 4)
    )
    calls = (*reads, ToolUseBlock("w1", "Write", {"path": "x.txt", "content": "X"}))
    await executor(policy).run_batch(calls, start("hi"))
    assert peak == MAX_CONCURRENT_READS


async def test_a_refused_call_between_two_reads_does_not_split_the_concurrent_group(
    policy, tmp_repo, monkeypatch
):
    """A refused call never executes, so grouping is computed over the
    sequence of calls that actually run -- not the raw request sequence -- and
    this one, sitting between two reads, must not break them into two
    sequential groups.
    """
    from nanoclaude.tools.read import ReadTool

    order: list[str] = []

    async def tracked_read(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        order.append(f"{call_id}:start")
        await asyncio.sleep(0.05)
        order.append(f"{call_id}:end")
        return ok(call_id, "read")

    monkeypatch.setattr(ReadTool, "run", tracked_read)
    calls = (
        ToolUseBlock("t1", "Read", {"path": "a.py"}),
        ToolUseBlock("t2", "Read", {"path": "/etc/passwd"}),  # denied; never reaches run()
        ToolUseBlock("t3", "Read", {"path": "b.py"}),
    )
    outcomes = await executor(policy).run_batch(calls, start("hi"))
    assert outcomes[1].is_error
    assert order == ["t1:start", "t3:start", "t1:end", "t3:end"]


async def test_after_a_write_then_read_in_one_batch_a_later_edit_succeeds(policy, tmp_repo):
    """End-to-end proof, through the real agent.loop machinery rather than
    Executor alone, that the scheduling fix keeps read_state honest. Before
    the fix, the Read in [Write(x), Read(x)] ran before the Write; once
    observe() folded both outcomes in request order, the Read's stamp of the
    OLD content overwrote the Write's stamp of the NEW one, and this Edit
    would have been wrongly refused as "changed since it was read" even
    though nothing touched the file outside the agent.
    """
    ex = executor(policy)
    turn1 = step(
        start("hi"),
        calls_many(
            ("Write", {"path": "x.txt", "content": "first\n"}, "t1"),
            ("Read", {"path": "x.txt"}, "t2"),
        ),
    )
    assert isinstance(turn1, RunTools)
    outcomes = await ex.run_batch(turn1.calls, turn1.state)
    assert "first" in outcomes[1].content
    state = observe(turn1.state, outcomes)

    turn2 = step(
        state,
        model_calls(
            "Edit",
            {"path": "x.txt", "edits": [{"old_string": "first", "new_string": "second"}]},
            call_id="t3",
        ),
    )
    assert isinstance(turn2, RunTools)
    edit_outcomes = await ex.run_batch(turn2.calls, turn2.state)
    assert not edit_outcomes[0].is_error
    assert (tmp_repo / "x.txt").read_text() == "second\n"


async def test_on_decision_is_called_for_refusals_that_never_reach_evaluate(policy, tmp_repo):
    """_refused() covers two early-exit paths that never reach evaluate() at
    all: an unknown tool name, and arguments bad enough that
    permission_request() itself raises. Both must still notify the UI -- a
    front end that renders decisions from on_decision alone must not silently
    miss them just because the audit log already hears about them.
    """

    class RecordingUI(SilentUI):
        def __init__(self) -> None:
            self.decisions: list[tuple[str, str]] = []

        def on_decision(
            self, call: ToolUseBlock, _request: PermissionRequest, result: PermissionResult
        ) -> None:
            self.decisions.append((call.id, result.rule))

    (tmp_repo / "a.py").write_text("x")
    ui = RecordingUI()
    calls = (
        ToolUseBlock("t1", "Nope", {}),  # tool.unknown, never reaches evaluate()
        ToolUseBlock("t2", "Read", {}),  # tool.bad-arguments, never reaches evaluate()
        ToolUseBlock("t3", "Read", {"path": "a.py"}),  # ordinary decision, for contrast
    )
    await executor(policy, ui).run_batch(calls, start("hi"))
    assert ui.decisions == [
        ("t1", "tool.unknown"),
        ("t2", "tool.bad-arguments"),
        ("t3", "rule.allow"),
    ]


# --- A bug in one tool must not end the session ---------------------------
#
# run_batch used to let any exception other than ToolArgumentError and OSError
# out of _run. One raised by a tool aborted the whole batch, threw away the
# outcomes of calls that had already finished, and reached the person as a
# traceback. The three tests after the first pin the edges of the clause that
# fixes it: it must come after the two existing clauses, and it must not catch
# what is not an Exception.


class _Abort(BaseException):
    """Stands in for KeyboardInterrupt and SystemExit.

    Raising either of those inside a task makes asyncio re-raise it into the
    event loop, which would end the whole pytest run rather than this test. A
    BaseException of our own takes the same path through ``except Exception``.
    """


def _audit_row(store: Store, call_id: str) -> Any:
    return store.db.execute(
        "SELECT outcome, duration_ms, error, decision, rule FROM tool_calls WHERE tool_use_id = ?",
        (call_id,),
    ).fetchone()


async def test_a_tool_raising_an_unexpected_exception_costs_one_call_not_the_batch(
    policy, tmp_repo, monkeypatch
):
    from nanoclaude.tools.read import ReadTool

    for name, text in (("a.py", "alpha"), ("b.py", "bravo"), ("c.py", "charlie")):
        (tmp_repo / name).write_text(f"{text}\n")
    real_run = ReadTool.run

    async def explode_for_t2(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        if call_id == "t2":
            raise RuntimeError("list index out of range")
        return await real_run(self, ctx, call_id, arguments)

    class Seen(SilentUI):
        def __init__(self) -> None:
            self.outcomes: list[ToolOutcome] = []

        def on_outcome(self, _call: ToolUseBlock, outcome: ToolOutcome) -> None:
            self.outcomes.append(outcome)

    monkeypatch.setattr(ReadTool, "run", explode_for_t2)
    ui, store = Seen(), _store(tmp_repo)
    calls = (
        ToolUseBlock("t1", "Read", {"path": "a.py"}),
        ToolUseBlock("t2", "Read", {"path": "b.py"}),
        ToolUseBlock("t3", "Read", {"path": "c.py"}),
    )
    outcomes = await executor(policy, ui, store).run_batch(calls, start("hi"))

    # Every call is answered, in the order asked, and the siblings really ran.
    assert [o.tool_use_id for o in outcomes] == ["t1", "t2", "t3"]
    assert [o.is_error for o in outcomes] == [False, True, False]
    assert "alpha" in outcomes[0].content and "charlie" in outcomes[2].content

    # What the model is told: where it happened, what it was, that it is a bug.
    bug = outcomes[1].content
    assert "internal error" in bug.lower() and "Read" in bug, bug
    assert "RuntimeError: list index out of range" in bug, bug
    assert "bug" in bug, bug
    # The person watching sees the failed call like any other outcome.
    assert [o.tool_use_id for o in ui.outcomes] == ["t1", "t2", "t3"]

    # The audit row says what happened, and the others are unaffected.
    row = _audit_row(store, "t2")
    assert row["outcome"] == "error", dict(row)
    assert row["error"] == "RuntimeError: list index out of range", dict(row)
    assert _audit_row(store, "t1")["outcome"] == "ok"
    assert _audit_row(store, "t3")["outcome"] == "ok"


async def test_what_an_unexpected_exception_says_is_scrubbed_like_any_tool_output(
    policy, tmp_repo, monkeypatch
):
    """An exception message can quote a value the tool was handling. The result
    goes into the transcript and the audit row goes into the database, and
    neither may carry a credential. Built at run time so no literal in this file
    is shaped like one."""
    from nanoclaude.tools.read import ReadTool

    secret = "sk-" + "Qz7" * 12
    store = _store(tmp_repo)
    (tmp_repo / "a.py").write_text("x")

    async def leak(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        raise ValueError(f"cannot use token {secret}")

    monkeypatch.setattr(ReadTool, "run", leak)
    outcomes = await executor(policy, AutoApprove(), store).run_batch(
        (ToolUseBlock("t1", "Read", {"path": "a.py"}),), start("hi")
    )
    assert secret not in outcomes[0].content and "[redacted:" in outcomes[0].content
    row = _audit_row(store, "t1")
    assert secret not in row["error"] and "[redacted:" in row["error"], dict(row)


async def test_a_tool_argument_error_is_still_a_bad_arguments_result_not_an_internal_error(
    policy, tmp_repo, monkeypatch
):
    """Raised from run() rather than from permission_request(), which is where
    an argument the tool only inspects once it executes surfaces. It has its
    own clause; the new one must sit after it and not swallow it."""
    from nanoclaude.tools.base import ToolArgumentError
    from nanoclaude.tools.read import ReadTool

    async def reject(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        raise ToolArgumentError("offset must be an integer, got str")

    monkeypatch.setattr(ReadTool, "run", reject)
    (tmp_repo / "a.py").write_text("x")
    store = _store(tmp_repo)
    outcomes = await executor(policy, AutoApprove(), store).run_batch(
        (ToolUseBlock("t1", "Read", {"path": "a.py"}),), start("hi")
    )
    assert outcomes[0].content == "bad arguments: offset must be an integer, got str"
    assert _audit_row(store, "t1")["error"] == "offset must be an integer, got str"


async def test_an_oserror_is_still_reported_as_itself_not_an_internal_error(
    policy, tmp_repo, monkeypatch
):
    from nanoclaude.tools.read import ReadTool

    async def fail(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        raise OSError("device not configured")

    monkeypatch.setattr(ReadTool, "run", fail)
    (tmp_repo / "a.py").write_text("x")
    store = _store(tmp_repo)
    outcomes = await executor(policy, AutoApprove(), store).run_batch(
        (ToolUseBlock("t1", "Read", {"path": "a.py"}),), start("hi")
    )
    assert outcomes[0].content == "OSError: device not configured"
    assert _audit_row(store, "t1")["error"] == "device not configured"


async def test_cancelling_a_running_tool_still_cancels_the_batch(policy, tmp_repo, monkeypatch):
    """CancelledError is a BaseException, so ``except Exception`` lets it through.
    Written with a write rather than a read on purpose: a read runs inside
    asyncio.gather, which cancels the batch by itself whatever _run does with
    the error, so a clause that wrongly swallowed it would go unnoticed there.
    A write is awaited directly, so swallowing the cancellation shows."""
    from nanoclaude.tools.write import WriteTool

    started = asyncio.Event()

    async def hang(
        self: WriteTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        started.set()
        await asyncio.Event().wait()  # never set; only cancellation ends this
        raise AssertionError("the tool was allowed to finish")

    monkeypatch.setattr(WriteTool, "run", hang)
    store = _store(tmp_repo)
    ex = executor(policy, AutoApprove(), store)
    call = ToolUseBlock("t1", "Write", {"path": "a.txt", "content": "A"})
    task = asyncio.create_task(ex.run_batch((call,), start("hi")))
    # Bounded: a tool that never starts must fail this test, not hang the run.
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Decided, never finished: the row's outcome stays NULL, which is the only
    # thing a NULL there is allowed to mean.
    assert _audit_row(store, "t1")["outcome"] is None


async def test_an_exception_that_is_not_an_exception_still_propagates(
    policy, tmp_repo, monkeypatch
):
    from nanoclaude.tools.write import WriteTool

    async def abort(
        self: WriteTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        raise _Abort("stop everything")

    monkeypatch.setattr(WriteTool, "run", abort)
    call = ToolUseBlock("t1", "Write", {"path": "a.txt", "content": "A"})
    with pytest.raises(_Abort, match="stop everything"):
        await executor(policy).run_batch((call,), start("hi"))


# --- The same invariant for the question asked before a call runs -------------
#
# Before a tool runs the executor asks it what the call would touch
# (permission_request). That is tool code too, and an exception from it other
# than the ones that mean "these arguments are bad" used to leave run_batch the
# same way an exception from run() did: the batch aborted, the finished calls'
# outcomes were thrown away, and the person got a traceback.


async def test_a_tool_whose_permission_request_raises_costs_one_call_not_the_batch(
    policy, tmp_repo, monkeypatch
):
    from nanoclaude.tools.read import ReadTool

    for name, text in (("a.py", "alpha"), ("b.py", "bravo"), ("c.py", "charlie")):
        (tmp_repo / name).write_text(f"{text}\n")
    real_request, real_run = ReadTool.permission_request, ReadTool.run
    ran: list[str] = []

    def explode_for_b(
        self: ReadTool, ctx: ToolContext, arguments: Mapping[str, Any]
    ) -> PermissionRequest:
        if arguments.get("path") == "b.py":
            raise RuntimeError("path table is corrupt")
        return real_request(self, ctx, arguments)

    async def tracking_run(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        ran.append(call_id)
        return await real_run(self, ctx, call_id, arguments)

    class Seen(SilentUI):
        def __init__(self) -> None:
            self.decisions: list[tuple[str, str]] = []
            self.outcomes: list[str] = []

        def on_decision(
            self, call: ToolUseBlock, _request: PermissionRequest, result: PermissionResult
        ) -> None:
            self.decisions.append((call.id, result.rule))

        def on_outcome(self, call: ToolUseBlock, _outcome: ToolOutcome) -> None:
            self.outcomes.append(call.id)

    monkeypatch.setattr(ReadTool, "permission_request", explode_for_b)
    monkeypatch.setattr(ReadTool, "run", tracking_run)
    ui, store = Seen(), _store(tmp_repo)
    calls = (
        ToolUseBlock("t1", "Read", {"path": "a.py"}),
        ToolUseBlock("t2", "Read", {"path": "b.py"}),
        ToolUseBlock("t3", "Read", {"path": "c.py"}),
    )
    outcomes = await executor(policy, ui, store).run_batch(calls, start("hi"))

    # Every call is answered, in the order asked; the siblings really ran.
    assert [o.tool_use_id for o in outcomes] == ["t1", "t2", "t3"]
    assert [o.is_error for o in outcomes] == [False, True, False]
    assert "alpha" in outcomes[0].content and "charlie" in outcomes[2].content
    # The call whose question could not be answered never ran.
    assert sorted(ran) == ["t1", "t3"]

    # What the model is told: where it happened, what it was, that it is a bug.
    bug = outcomes[1].content
    assert "internal error" in bug.lower() and "Read" in bug, bug
    assert "RuntimeError: path table is corrupt" in bug, bug
    assert "bug" in bug, bug

    # The audit row: not executed, so no duration, and the outcome is an error,
    # not a refusal. The decision and rule are the nearest true ones.
    row = _audit_row(store, "t2")
    assert row["outcome"] == "error", dict(row)
    assert row["error"] == "RuntimeError: path table is corrupt", dict(row)
    assert (row["decision"], row["rule"]) == ("deny", "tool.internal-error"), dict(row)
    assert row["duration_ms"] == 0, dict(row)
    assert _audit_row(store, "t1")["outcome"] == "ok"
    assert _audit_row(store, "t3")["outcome"] == "ok"

    # The person's front end hears about it however it renders: as a decision, and
    # once as the outcome of a call, like any other that ended in an error.
    assert ui.decisions == [
        ("t1", "rule.allow"),
        ("t2", "tool.internal-error"),
        ("t3", "rule.allow"),
    ]
    assert ui.outcomes.count("t2") == 1 and sorted(ui.outcomes) == ["t1", "t2", "t3"]


@pytest.mark.parametrize(
    "malformed",
    [
        PermissionRequest("Read", None),  # type: ignore[arg-type]
        PermissionRequest("Read", "b.py", resolved_paths=None),  # type: ignore[arg-type]
    ],
    ids=["subject-is-none", "resolved-paths-is-none"],
)
async def test_a_request_the_policy_cannot_evaluate_costs_one_call_not_the_batch(
    policy, tmp_repo, monkeypatch, malformed
):
    """The tool answered its permission request, wrongly, and evaluating the answer is
    what raises. That is the same failure as a tool that cannot answer: the call ends
    as an error under tool.internal-error and does not run, and the rest of the batch
    goes on. Left to escape, it would abort run_batch and leave the calls decided
    before it, and the ones after it, pending."""
    from nanoclaude.tools.read import ReadTool

    for name, text in (("a.py", "alpha"), ("b.py", "bravo"), ("c.py", "charlie")):
        (tmp_repo / name).write_text(f"{text}\n")
    real_request, real_run = ReadTool.permission_request, ReadTool.run
    ran: list[str] = []

    def malformed_for_b(
        self: ReadTool, ctx: ToolContext, arguments: Mapping[str, Any]
    ) -> PermissionRequest:
        if arguments.get("path") == "b.py":
            answer: PermissionRequest = malformed
            return answer
        return real_request(self, ctx, arguments)

    async def tracking_run(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        ran.append(call_id)
        return await real_run(self, ctx, call_id, arguments)

    monkeypatch.setattr(ReadTool, "permission_request", malformed_for_b)
    monkeypatch.setattr(ReadTool, "run", tracking_run)
    ui, store = _Recorder(), _store(tmp_repo)
    calls = (
        ToolUseBlock("t1", "Read", {"path": "a.py"}),
        ToolUseBlock("t2", "Read", {"path": "b.py"}),
        ToolUseBlock("t3", "Read", {"path": "c.py"}),
    )
    outcomes = await executor(policy, ui, store).run_batch(calls, start("hi"))

    assert [o.tool_use_id for o in outcomes] == ["t1", "t2", "t3"]
    assert [o.is_error for o in outcomes] == [False, True, False]
    assert "alpha" in outcomes[0].content and "charlie" in outcomes[2].content
    assert sorted(ran) == ["t1", "t3"]  # the call whose request could not be judged never ran
    # Which exception the policy raises depends on the field and on the Python; the form
    # the model and the audit row get is the same for all of them.
    assert re.match(r"Internal error in Read: (AttributeError|TypeError): ", outcomes[1].content)
    assert "bug" in outcomes[1].content
    row = _audit_row(store, "t2")
    assert (row["decision"], row["rule"], row["outcome"]) == (
        "deny",
        "tool.internal-error",
        "error",
    )
    assert re.match(r"(AttributeError|TypeError): ", row["error"]), dict(row)
    assert [result.rule for _, result in ui.decisions] == [
        "rule.allow",
        "tool.internal-error",
        "rule.allow",
    ]


async def test_an_exception_that_is_not_an_exception_still_propagates_from_the_policy(
    policy, tmp_repo, monkeypatch
):
    def abort(request: PermissionRequest, *_rest: object) -> PermissionResult:
        raise _Abort("stop everything")

    monkeypatch.setattr("nanoclaude.agent.executor.evaluate", abort)
    (tmp_repo / "a.py").write_text("x")
    with pytest.raises(_Abort, match="stop everything"):
        await executor(policy).run_batch(
            (ToolUseBlock("t1", "Read", {"path": "a.py"}),), start("hi")
        )


async def test_what_a_failing_permission_request_says_is_scrubbed_like_any_tool_output(
    policy, tmp_repo, monkeypatch
):
    from nanoclaude.tools.read import ReadTool

    secret = "sk-" + "Qz7" * 12  # built at run time: no literal here is shaped like a key
    store = _store(tmp_repo)
    (tmp_repo / "a.py").write_text("x")

    def leak(self: ReadTool, ctx: ToolContext, arguments: Mapping[str, Any]) -> PermissionRequest:
        raise RuntimeError(f"cannot classify {secret}")

    monkeypatch.setattr(ReadTool, "permission_request", leak)
    outcomes = await executor(policy, AutoApprove(), store).run_batch(
        (ToolUseBlock("t1", "Read", {"path": "a.py"}),), start("hi")
    )
    assert secret not in outcomes[0].content and "[redacted:" in outcomes[0].content
    row = _audit_row(store, "t1")
    assert secret not in row["error"] and "[redacted:" in row["error"], dict(row)


#: What a message can hold that a terminal would act on: a colour sequence and a
#: carriage return that redraws the line.
ESCAPES = "\x1b[31mred\x1b[0m and\rmore"


def _free_of_escapes(text: str) -> bool:
    return "\x1b" not in text and "\r" not in text


async def test_an_internal_error_from_run_carries_no_terminal_control_sequences(
    policy, tmp_repo, monkeypatch
):
    from nanoclaude.tools.read import ReadTool

    async def broken(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        raise RuntimeError(ESCAPES)

    class Seen(SilentUI):
        def __init__(self) -> None:
            self.outcomes: list[ToolOutcome] = []

        def on_outcome(self, _call: ToolUseBlock, outcome: ToolOutcome) -> None:
            self.outcomes.append(outcome)

    monkeypatch.setattr(ReadTool, "run", broken)
    (tmp_repo / "a.py").write_text("x")
    ui, store = Seen(), _store(tmp_repo)
    outcomes = await executor(policy, ui, store).run_batch(
        (ToolUseBlock("t1", "Read", {"path": "a.py"}),), start("hi")
    )
    assert "RuntimeError: red andmore" in outcomes[0].content
    row = _audit_row(store, "t1")
    assert row["error"] == "RuntimeError: red andmore", dict(row)
    assert all(_free_of_escapes(o.content) for o in (*outcomes, *ui.outcomes))


async def test_a_credential_split_by_an_escape_sequence_is_still_scrubbed(
    policy, tmp_repo, monkeypatch
):
    """The scrubber matches a credential as it is written. An escape sequence in the
    middle of one hides it from the scrubber and is gone once the text is stripped, so
    stripping has to come first. Built at run time so no literal here is shaped like a
    key."""
    from nanoclaude.tools.read import ReadTool

    secret = "sk-" + "Qz7" * 12
    split = secret[:10] + "\x1b[0m" + secret[10:]

    async def leak(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        raise ValueError(f"cannot use token {split}")

    monkeypatch.setattr(ReadTool, "run", leak)
    (tmp_repo / "a.py").write_text("x")
    store = _store(tmp_repo)
    outcomes = await executor(policy, AutoApprove(), store).run_batch(
        (ToolUseBlock("t1", "Read", {"path": "a.py"}),), start("hi")
    )
    row = _audit_row(store, "t1")
    for text in (outcomes[0].content, row["error"]):
        assert secret not in text and "[redacted:" in text, text


async def test_an_internal_error_from_a_permission_request_carries_no_terminal_control_sequences(
    policy, tmp_repo, monkeypatch
):
    from nanoclaude.tools.read import ReadTool

    def broken(self: ReadTool, ctx: ToolContext, arguments: Mapping[str, Any]) -> PermissionRequest:
        raise RuntimeError(ESCAPES)

    class Seen(SilentUI):
        def __init__(self) -> None:
            self.reasons: list[str] = []

        def on_decision(
            self, _call: ToolUseBlock, _request: PermissionRequest, result: PermissionResult
        ) -> None:
            self.reasons.append(result.reason)

    monkeypatch.setattr(ReadTool, "permission_request", broken)
    (tmp_repo / "a.py").write_text("x")
    ui, store = Seen(), _store(tmp_repo)
    outcomes = await executor(policy, ui, store).run_batch(
        (ToolUseBlock("t1", "Read", {"path": "a.py"}),), start("hi")
    )
    assert "RuntimeError: red andmore" in outcomes[0].content
    row = _audit_row(store, "t1")
    assert row["error"] == "RuntimeError: red andmore", dict(row)
    assert ui.reasons == ["RuntimeError: red andmore"]
    assert _free_of_escapes(outcomes[0].content)


@pytest.mark.parametrize("where", ["permission_request", "run"])
def test_a_real_keyboard_interrupt_in_a_tool_ends_the_batch_and_is_not_made_a_result(
    policy, tmp_repo, monkeypatch, where
):
    """The stand-in the tests above use is a BaseException that is not an Exception.
    KeyboardInterrupt is both of those things and one more: asyncio re-raises it out of
    the task that raised it and through run_until_complete. Inside the event loop pytest
    runs its async tests in, that would end the whole test run, so this one is a plain
    test with a loop of its own."""
    from nanoclaude.tools.read import ReadTool

    def interrupted_request(
        self: ReadTool, ctx: ToolContext, arguments: Mapping[str, Any]
    ) -> PermissionRequest:
        raise KeyboardInterrupt

    async def interrupted_run(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        ReadTool,
        where,
        interrupted_request if where == "permission_request" else interrupted_run,
    )
    (tmp_repo / "a.py").write_text("x")
    batch = executor(policy).run_batch((ToolUseBlock("t1", "Read", {"path": "a.py"}),), start("hi"))
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(KeyboardInterrupt):
            loop.run_until_complete(batch)
    finally:
        # Whatever the interrupted batch left running is cancelled and finished before
        # the loop goes, or the loop's closing would warn about it.
        unfinished = asyncio.all_tasks(loop)
        for task in unfinished:
            task.cancel()
        if unfinished:
            loop.run_until_complete(asyncio.gather(*unfinished, return_exceptions=True))
        loop.close()


@pytest.mark.parametrize(
    "error",
    [
        ToolArgumentError("offset must be an integer, got str"),
        ValueError("embedded null byte"),
        OSError("device not configured"),
    ],
    ids=["tool-argument-error", "value-error", "os-error"],
)
async def test_what_a_permission_request_may_legitimately_raise_is_still_a_refusal(
    policy, tmp_repo, monkeypatch, error
):
    """These three have their own clause and mean "these arguments are bad", which
    the model can correct. The new clause sits after them and must not take them.
    ValueError and OSError would pass a looser check (their text is in the new
    message too), so the wording and the audit row are what is pinned."""
    from nanoclaude.tools.read import ReadTool

    def reject(self: ReadTool, ctx: ToolContext, arguments: Mapping[str, Any]) -> PermissionRequest:
        raise error

    monkeypatch.setattr(ReadTool, "permission_request", reject)
    (tmp_repo / "a.py").write_text("x")
    store = _store(tmp_repo)
    outcomes = await executor(policy, AutoApprove(), store).run_batch(
        (ToolUseBlock("t1", "Read", {"path": "a.py"}),), start("hi")
    )
    assert outcomes[0].content.startswith("Refused (tool.bad-arguments): bad arguments: ")
    row = _audit_row(store, "t1")
    assert (row["outcome"], row["rule"]) == ("refused", "tool.bad-arguments"), dict(row)


async def test_an_exception_that_is_not_an_exception_still_propagates_from_a_permission_request(
    policy, tmp_repo, monkeypatch
):
    from nanoclaude.tools.read import ReadTool

    def abort(self: ReadTool, ctx: ToolContext, arguments: Mapping[str, Any]) -> PermissionRequest:
        raise _Abort("stop everything")

    monkeypatch.setattr(ReadTool, "permission_request", abort)
    (tmp_repo / "a.py").write_text("x")
    with pytest.raises(_Abort, match="stop everything"):
        await executor(policy).run_batch(
            (ToolUseBlock("t1", "Read", {"path": "a.py"}),), start("hi")
        )


# --- Text the model chose never reaches anything raw ----------------------------
#
# Every string the executor builds from what the model sent (a path quoted in a
# refusal, a tool name it made up, the message of an error a tool raised over its
# arguments) goes through sanitize() before it reaches the model, the front end or
# the audit log. The front end prints what it is given, and a terminal acts on
# what it prints: one test per path, each reading everything that path produced.

#: A window-title change, a clear-screen and a carriage return, in the order a
#: hostile string would carry them.
HOSTILE = "\x1b]0;TITLE\x07\x1b[2J\r"


class _Recorder(SilentUI):
    """Remembers everything the executor tells the front end."""

    def __init__(self) -> None:
        self.decisions: list[tuple[PermissionRequest, PermissionResult]] = []
        self.outcomes: list[ToolOutcome] = []

    def on_decision(
        self, _call: ToolUseBlock, request: PermissionRequest, result: PermissionResult
    ) -> None:
        self.decisions.append((request, result))

    def on_outcome(self, _call: ToolUseBlock, outcome: ToolOutcome) -> None:
        self.outcomes.append(outcome)


def _every_text_the_executor_wrote(
    outcomes: tuple[ToolOutcome, ...], ui: _Recorder, store: Store
) -> list[str]:
    """What the model, the front end and the audit table were each handed.

    Not the subject of a request a tool built (the path or the command it asked
    about): the executor hands that on as it is, because the rules judge the
    original, and a front end that shows it has to render it safely.
    """
    texts = [outcome.content for outcome in (*outcomes, *ui.outcomes)]
    for request, result in ui.decisions:
        texts += [request.tool, result.rule, result.reason]
    columns = "tool, args_json, decision, rule, outcome, error"
    for row in store.db.execute(f"SELECT {columns} FROM tool_calls"):  # noqa: S608
        texts += [value for value in row if isinstance(value, str)]
    return texts


def _what_a_terminal_would_act_on(texts: list[str]) -> list[str]:
    return [text for text in texts if any(char in text for char in ("\x1b", "\x07", "\r"))]


async def test_a_policy_refusal_quoting_the_models_path_carries_no_terminal_sequences(
    policy, tmp_repo
):
    ui, store = _Recorder(), _store(tmp_repo)
    elsewhere = tmp_repo.parent / "elsewhere"  # beside the sandbox root, not in it
    outcomes = await executor(policy, ui, store).run_batch(
        (ToolUseBlock("t1", "Read", {"path": f"{elsewhere}/{HOSTILE}passwd"}),), start("hi")
    )
    # The words are still there; what a terminal would act on is not.
    quoted = f"{elsewhere}/passwd is outside the working directory"
    assert outcomes[0].content.startswith(f"Refused (sandbox.outside-root): {quoted}")
    [(_, result)] = ui.decisions
    assert result.reason.startswith(quoted)
    assert _what_a_terminal_would_act_on(_every_text_the_executor_wrote(outcomes, ui, store)) == []


async def test_an_unknown_tool_name_carrying_terminal_sequences_is_shown_and_stored_without_them(
    policy, tmp_repo
):
    ui, store = _Recorder(), _store(tmp_repo)
    outcomes = await executor(policy, ui, store).run_batch(
        (ToolUseBlock("t1", f"Nope{HOSTILE}", {}),), start("hi")
    )
    assert outcomes[0].content.startswith("Refused (tool.unknown): ")
    [(request, _)] = ui.decisions
    assert request.tool == "Nope"
    stored = store.db.execute("SELECT tool FROM tool_calls WHERE tool_use_id = 't1'").fetchone()
    assert stored["tool"] == "Nope"
    assert _what_a_terminal_would_act_on(_every_text_the_executor_wrote(outcomes, ui, store)) == []


@pytest.mark.parametrize(
    ("where", "error", "content", "recorded"),
    [
        (
            "permission_request",
            ToolArgumentError(f"offset {HOSTILE}must be a number"),
            "Refused (tool.bad-arguments): bad arguments: offset must be a number",
            None,
        ),
        (
            "permission_request",
            OSError(f"cannot stat {HOSTILE}file"),
            "Refused (tool.bad-arguments): bad arguments: OSError: cannot stat file",
            None,
        ),
        (
            "run",
            ToolArgumentError(f"offset {HOSTILE}must be a number"),
            "bad arguments: offset must be a number",
            "offset must be a number",
        ),
        (
            "run",
            OSError(f"cannot stat {HOSTILE}file"),
            "OSError: cannot stat file",
            "cannot stat file",
        ),
    ],
    ids=[
        "tool-argument-error-in-the-permission-request",
        "os-error-in-the-permission-request",
        "tool-argument-error-in-run",
        "os-error-in-run",
    ],
)
async def test_the_message_of_an_error_over_a_calls_arguments_carries_no_terminal_sequences(
    policy, tmp_repo, monkeypatch, where, error, content, recorded
):
    from nanoclaude.tools.read import ReadTool

    def raising_request(
        self: ReadTool, ctx: ToolContext, arguments: Mapping[str, Any]
    ) -> PermissionRequest:
        raise error

    async def raising_run(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        raise error

    monkeypatch.setattr(
        ReadTool, where, raising_request if where == "permission_request" else raising_run
    )
    (tmp_repo / "a.py").write_text("x")
    ui, store = _Recorder(), _store(tmp_repo)
    outcomes = await executor(policy, ui, store).run_batch(
        (ToolUseBlock("t1", "Read", {"path": "a.py"}),), start("hi")
    )
    assert outcomes[0].content == content
    assert _audit_row(store, "t1")["error"] == recorded
    assert _what_a_terminal_would_act_on(_every_text_the_executor_wrote(outcomes, ui, store)) == []
