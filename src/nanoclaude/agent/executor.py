# src/nanoclaude/agent/executor.py
"""Running the tool calls from one model turn.

Two phases, and the split is the point.

**Decide everything first.** Resolve paths, classify commands, evaluate policy,
ask the user -- all before any tool runs. Confirmation prompts then arrive in
the order the model asked, with nothing already half-done behind them, and a
user who declines the first of five writes does not discover that the third one
already landed.

**Then execute in request order, concurrent only within a read-only run.** A
write is a barrier: everything requested before it finishes before it starts,
and nothing requested after it starts until it is done -- a Read placed right
after a Write in the same batch must see what the Write just produced, not
what was there before it. Maximal runs of *consecutive* read-only calls
between one barrier and the next still go concurrently, capped by a
semaphore; a refused or declined call never executes and so cannot split a
run of reads around it. Concurrency buys latency within one run; no amount of
it is worth an undefined order between a write and whatever depends on it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from nanoclaude.agent.loop import LoopState
from nanoclaude.agent.ui import UI, Approval, AutoDecline
from nanoclaude.conversation.transcript import ToolUseBlock
from nanoclaude.permissions.audit import AuditLog
from nanoclaude.permissions.policy import (
    Decision,
    Grants,
    PermissionRequest,
    PermissionResult,
    Policy,
    evaluate,
)
from nanoclaude.permissions.redact import Redactor
from nanoclaude.tools.base import Tool, ToolArgumentError, ToolContext, ToolOutcome, failed
from nanoclaude.tools.registry import ToolRegistry, UnknownToolError

MAX_CONCURRENT_READS = 8


def _refusal_text(rule: str, reason: str) -> str:
    """``Refused (<rule>): <reason>`` -- for the model, not a person.

    Sentence-cased on purpose: spec 17.9's user-facing form is a different,
    lowercase string (``refused (<rule>): ...``). Whatever renders a refusal
    to a *person* (Task 25) must format it from the ``PermissionResult`` the
    audit log and ``UI.on_decision`` already carry, not by echoing or
    re-casing this one -- this string's only reader is the model, in the next
    turn's tool_result.
    """
    return f"Refused ({rule}): {reason}"


def _consecutive_read_only_runs(plans: Sequence[_Plan]) -> list[tuple[bool, list[_Plan]]]:
    """Split ``plans`` into maximal runs of consecutive read-only / non-
    read-only tools, preserving their relative order.

    ``plans`` must already be filtered down to ones that actually execute --
    refused and declined calls have their outcome already recorded and never
    reach here, which is what keeps them from splitting a run of reads around
    them: a run is computed over this filtered sequence, so a call that was
    never going to run is simply not in it to begin with.
    """
    runs: list[tuple[bool, list[_Plan]]] = []
    for plan in plans:
        tool = plan.tool
        assert tool is not None  # noqa: S101 - only plans that execute reach here
        read_only = tool.read_only
        if runs and runs[-1][0] is read_only:
            runs[-1][1].append(plan)
        else:
            runs.append((read_only, [plan]))
    return runs


@dataclass(frozen=True, slots=True)
class _Plan:
    call: ToolUseBlock
    tool: Tool | None
    request: PermissionRequest
    result: PermissionResult
    refusal: ToolOutcome | None

    # Declared unhashable rather than left to the default: call is a
    # ToolUseBlock, already unhashable because it holds decoded-JSON
    # arguments (see transcript.py). Left to frozen=True's default eq=True,
    # dataclass would generate a real __hash__ -- isinstance(x, Hashable)
    # would say True -- that only raises TypeError when actually called, and
    # names "ToolUseBlock" rather than this class. Nothing needs a _Plan in a
    # set or as a dict key; run_batch keys its results dict on plan.call.id,
    # which is what identifies a call anyway.
    __hash__ = None  # type: ignore[assignment]


@dataclass
class Executor:
    registry: ToolRegistry
    policy: Policy
    ui: UI = field(default_factory=AutoDecline)
    audit: AuditLog | None = None
    session_id: str = "session"
    grants: Grants = field(default_factory=Grants)
    redactor: Redactor = field(default_factory=Redactor)

    def context(self, state: LoopState) -> ToolContext:
        return ToolContext(
            sandbox=self.policy.sandbox,
            policy=self.policy,
            redactor=self.redactor,
            read_state=state.read_state,
            root=self.policy.sandbox.roots[0],
        )

    async def run_batch(
        self, calls: Sequence[ToolUseBlock], state: LoopState
    ) -> tuple[ToolOutcome, ...]:
        ctx = self.context(state)
        # Every call's _Plan is built from this one snapshot of self.grants,
        # taken before any of them is confirmed. An ALWAYS answer to call 1
        # updates self.grants in the loop below, but call 2's _Plan was
        # already decided from grants as they stood before that -- so call 2
        # still asks (or still refuses) even if it is for the exact tool
        # call 1 just granted. The grant takes effect starting with the next
        # run_batch, not retroactively within this one; that is what keeps
        # the first phase's promise that no call's outcome in a batch depends
        # on how a *later* one in the same batch turns out.
        plans = [self._decide(ctx, call, state.turn) for call in calls]
        results: dict[str, ToolOutcome] = {}

        for plan in plans:
            if plan.refusal is not None:
                results[plan.call.id] = plan.refusal
                self._audit_outcome(
                    plan.call.id,
                    outcome="refused",
                    duration_ms=0,
                    bytes_out=len(plan.refusal.content.encode("utf-8")),
                    error=None,
                )
                continue
            if plan.result.decision is not Decision.ASK:
                continue
            approval = await self.ui.confirm(plan.call, plan.request, plan.result)
            if approval is Approval.NO:
                declined = ToolOutcome(
                    plan.call.id,
                    f"The user declined to run {plan.call.name}. Do not try to achieve "
                    "the same thing another way; say what you were attempting and stop.",
                    is_error=True,
                )
                results[plan.call.id] = declined
                self._audit_outcome(
                    plan.call.id,
                    outcome="declined",
                    duration_ms=0,
                    bytes_out=len(declined.content.encode("utf-8")),
                    error=None,
                )
            elif approval is Approval.ALWAYS:
                self.grants = self.grants.with_tool(plan.call.name)

        # Request order is preserved across tool classes, not just within one:
        # a write must finish before whatever the model asked for next can
        # start, because that next call may depend on what the write just
        # did (a Read straight after a Write must see the new content, not
        # whatever was on disk before the batch started). Only *consecutive*
        # read-only calls -- a run with no write, refused, or declined call
        # breaking it up -- share a semaphore-capped asyncio.gather. Every
        # non-read-only call still runs by itself, one at a time, in the
        # order requested, exactly as it always has.
        runnable = [p for p in plans if p.call.id not in results and p.tool is not None]
        limit = asyncio.Semaphore(MAX_CONCURRENT_READS)

        async def guarded(plan: _Plan) -> ToolOutcome:
            async with limit:
                return await self._run(ctx, plan)

        for read_only, group in _consecutive_read_only_runs(runnable):
            if read_only:
                for plan, outcome in zip(
                    group, await asyncio.gather(*(guarded(p) for p in group)), strict=True
                ):
                    results[plan.call.id] = outcome
            else:
                for plan in group:
                    results[plan.call.id] = await self._run(ctx, plan)

        return tuple(results[call.id] for call in calls)

    def _decide(self, ctx: ToolContext, call: ToolUseBlock, turn: int) -> _Plan:
        try:
            tool = self.registry.get(call.name)
        except UnknownToolError as exc:
            return self._refused(call, str(exc), turn, "tool.unknown")
        try:
            request = tool.permission_request(ctx, call.arguments)
        except ToolArgumentError as exc:
            return self._refused(call, f"bad arguments: {exc}", turn, "tool.bad-arguments")
        except (ValueError, OSError) as exc:
            return self._refused(
                call, f"bad arguments: {type(exc).__name__}: {exc}", turn, "tool.bad-arguments"
            )

        result = evaluate(request, self.policy, self.grants)
        self.ui.on_decision(call, request, result)
        self._audit_decision(call, turn, result)
        if result.decision is Decision.DENY:
            return _Plan(
                call,
                None,
                request,
                result,
                ToolOutcome(call.id, _refusal_text(result.rule, result.reason), is_error=True),
            )
        return _Plan(call, tool, request, result, None)

    def _refused(self, call: ToolUseBlock, message: str, turn: int, rule: str) -> _Plan:
        # Reaches here before evaluate() ever runs (an unknown tool name, or
        # arguments bad enough that permission_request() itself raises), so
        # there is no real PermissionRequest to report -- the same placeholder
        # used below is what on_decision and the audit log both see.
        request = PermissionRequest(call.name, "")
        result = PermissionResult(Decision.DENY, rule, message)
        self.ui.on_decision(call, request, result)
        self._audit_decision(call, turn, result)
        return _Plan(
            call,
            None,
            request,
            result,
            ToolOutcome(call.id, _refusal_text(rule, message), is_error=True),
        )

    def _audit_decision(self, call: ToolUseBlock, turn: int, result: PermissionResult) -> None:
        if self.audit is None:
            return
        self.audit.record_decision(
            self.session_id,
            turn=turn,
            tool_use_id=call.id,
            tool=call.name,
            arguments=call.arguments,
            decision=result.decision,
            rule=result.rule,
        )

    def _audit_outcome(
        self, call_id: str, *, outcome: str, duration_ms: int, bytes_out: int, error: str | None
    ) -> None:
        # The single place every path that closes out a call's audit row
        # goes through -- refused and declined (both added here per this
        # task's "every audited call gets an outcome" requirement, both
        # always duration_ms=0) and the real ok/error outcome from _run below.
        # A NULL outcome in the tool_calls table then means exactly one thing:
        # the call was decided and the process crashed before any of these
        # three paths got a chance to record what happened to it.
        # Centralising the self.audit is None guard here also means that
        # guard is written, and can be wrong, in exactly one place.
        if self.audit is None:
            return
        self.audit.record_outcome(
            self.session_id,
            call_id,
            outcome=outcome,
            duration_ms=duration_ms,
            bytes_out=bytes_out,
            error=error,
        )

    async def _run(self, ctx: ToolContext, plan: _Plan) -> ToolOutcome:
        assert plan.tool is not None  # noqa: S101 - refused plans never reach here
        started = time.monotonic()
        error: str | None = None
        try:
            outcome = await plan.tool.run(ctx, plan.call.id, plan.call.arguments)
        except ToolArgumentError as exc:
            outcome = ToolOutcome(plan.call.id, f"bad arguments: {exc}", is_error=True)
            error = str(exc)
        except OSError as exc:
            outcome = ToolOutcome(plan.call.id, f"{type(exc).__name__}: {exc}", is_error=True)
            error = str(exc)
        except Exception as exc:
            # A bug in one tool is that call's failure, not the batch's. Left to
            # escape, it leaves run_batch, discards the outcomes of the calls
            # that already finished, and reaches the person as a traceback. It
            # sits after the two clauses above so the failures a tool is allowed
            # to have keep their own wording, and it names Exception rather than
            # BaseException so cancellation and KeyboardInterrupt still stop the
            # batch instead of being reported to the model as a bug. The text is
            # scrubbed like any tool output: an exception can quote the value
            # the tool was handling, and this goes to the transcript and the
            # audit table.
            error, _ = self.redactor.scrub(f"{type(exc).__name__}: {exc}")
            outcome = failed(
                plan.call.id,
                f"Internal error in {plan.call.name}: {error}\n"
                "This is a bug in the tool, not in your arguments. Do not repeat the call; "
                "try another approach, or tell the user what failed.",
            )
        self.ui.on_outcome(plan.call, outcome)
        self._audit_outcome(
            plan.call.id,
            outcome="error" if outcome.is_error else "ok",
            duration_ms=int((time.monotonic() - started) * 1000),
            bytes_out=len(outcome.content.encode("utf-8")),
            error=error,
        )
        return outcome
