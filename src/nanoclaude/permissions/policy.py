"""One pure function decides whether a tool call may happen.

The order in :func:`evaluate` is the security model, so it is written once, as a
sequence of early returns, in the order given in spec 6.2. The first five checks
are the hard-denial band: no mode, no rule and no session grant can get past
them, which is why ``BYPASS`` sits *after* them rather than at the top.

Nothing here touches the filesystem. Paths arriving in a
:class:`PermissionRequest` are already resolved -- following a symlink after the
permission check is how sandboxes get escaped.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

import pathspec

from nanoclaude.permissions.danger import DangerLevel, DangerVerdict
from nanoclaude.permissions.rules import RuleSet
from nanoclaude.permissions.sandbox import Sandbox

#: Tools that cannot change anything, and so never need confirming on their own.
READ_ONLY_TOOLS = frozenset({"Read", "Grep", "Glob"})

#: Tools ACCEPT_EDITS mode auto-approves.
EDIT_TOOLS = frozenset({"Edit", "Write"})


class PermissionMode(StrEnum):
    DEFAULT = "default"
    PLAN = "plan"
    ACCEPT_EDITS = "accept-edits"
    BYPASS = "bypass"


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    tool: str
    subject: str
    resolved_paths: tuple[str, ...] = ()
    is_write: bool = False
    danger: DangerVerdict | None = None


@dataclass(frozen=True, slots=True)
class PermissionResult:
    decision: Decision
    rule: str
    reason: str

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW


@dataclass(frozen=True, slots=True)
class Grants:
    tools: frozenset[str] = field(default_factory=frozenset)

    def with_tool(self, tool: str) -> Grants:
        return replace(self, tools=self.tools | {tool})


@dataclass(frozen=True, slots=True)
class Policy:
    sandbox: Sandbox
    rules: RuleSet
    mode: PermissionMode = PermissionMode.DEFAULT
    secret_paths: tuple[str, ...] = ()
    allow_secrets: bool = False

    def relative(self, path: str) -> str:
        for root in self.sandbox.roots:
            if path.startswith(root):
                return path[len(root) :].lstrip("/")
        return path.lstrip("/")


#: evaluate()'s default when a caller has not granted anything yet this
#: session. A module-level singleton rather than a call in the signature
#: itself -- ruff's B008 flags any function call in an argument default,
#: since Python evaluates it once at def time and every caller who skips the
#: argument then shares that object. Grants is frozen, so there is no
#: mutation hazard either way; this only satisfies the linter.
_NO_GRANTS = Grants()


def evaluate(
    request: PermissionRequest, policy: Policy, grants: Grants = _NO_GRANTS
) -> PermissionResult:
    subjects = (request.subject, policy.relative(request.subject))

    # 1. rule.deny
    denied = policy.rules.first_match("deny", request.tool, *subjects)
    if denied is not None:
        return PermissionResult(Decision.DENY, "rule.deny", f"denied by your rule {denied.source}")

    # 2. secret.path
    if not policy.allow_secrets:
        for path in request.resolved_paths:
            if _matches_any(policy.secret_paths, (path, policy.relative(path))):
                return PermissionResult(
                    Decision.DENY,
                    "secret.path",
                    f"{path} looks like a credentials file and is never read. "
                    "Pass --allow-secrets, or add an allow rule for this exact path.",
                )

    # 3. sandbox.outside-root / sandbox.symlink-escape
    for path in request.resolved_paths:
        violation = (
            policy.sandbox.check_write(path)
            if request.is_write
            else policy.sandbox.check_read(path)
        )
        if violation is not None:
            roots = ", ".join(policy.sandbox.roots)
            return PermissionResult(
                Decision.DENY,
                violation,
                f"{path} is outside the working directory ({roots}). Use --add-dir to widen it.",
            )

    # 4. bash.dangerous / bash.unparseable
    if request.danger is not None:
        if request.danger.level is DangerLevel.BLOCKED:
            return PermissionResult(
                Decision.DENY,
                "bash.dangerous",
                f"refused by the {request.danger.classifier} classifier: {request.danger.reason}",
            )
        if request.danger.level is DangerLevel.UNPARSEABLE:
            return PermissionResult(
                Decision.DENY,
                "bash.unparseable",
                "this command could not be parsed, so it cannot be judged safe. "
                "Rewrite it more simply.",
            )

    # 5. mode.plan-read-only
    if policy.mode is PermissionMode.PLAN and request.is_write:
        return PermissionResult(
            Decision.DENY,
            "mode.plan-read-only",
            "plan mode is read-only. Present your plan and exit plan mode first.",
        )

    # 6. mode.bypass
    if policy.mode is PermissionMode.BYPASS:
        return PermissionResult(Decision.ALLOW, "mode.bypass", "permission prompts are disabled")

    # 7. mode.accept-edits
    if policy.mode is PermissionMode.ACCEPT_EDITS and request.tool in EDIT_TOOLS:
        return PermissionResult(
            Decision.ALLOW, "mode.accept-edits", "file edits are auto-approved in this mode"
        )

    # 8. rule.allow
    allowed = policy.rules.first_match("allow", request.tool, *subjects)
    if allowed is not None:
        return PermissionResult(Decision.ALLOW, "rule.allow", f"allowed by {allowed.source}")

    # 9. grant.session
    if request.tool in grants.tools:
        return PermissionResult(
            Decision.ALLOW, "grant.session", f"{request.tool} was approved for this session"
        )

    # 10. tool.read-only
    if request.tool in READ_ONLY_TOOLS:
        return PermissionResult(Decision.ALLOW, "tool.read-only", f"{request.tool} only reads")

    # 11. rule.ask
    asked = policy.rules.first_match("ask", request.tool, *subjects)
    if asked is not None:
        return PermissionResult(Decision.ASK, "rule.ask", f"{asked.source} requires confirmation")

    # 12. default.ask
    return PermissionResult(
        Decision.ASK, "default.ask", f"{request.tool} changes state and has no matching rule"
    )


def _matches_any(patterns: tuple[str, ...], candidates: tuple[str, ...]) -> bool:
    if not patterns:
        return False
    spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    return any(spec.match_file(c.lstrip("/")) or spec.match_file(c) for c in candidates)
