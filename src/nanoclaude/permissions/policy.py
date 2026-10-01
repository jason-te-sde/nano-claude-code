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
from pathlib import PurePosixPath

from nanoclaude.permissions.danger import DangerLevel, DangerVerdict
from nanoclaude.permissions.rules import RuleSet, glob_matches_any
from nanoclaude.permissions.sandbox import Sandbox, is_within

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

    def __post_init__(self) -> None:
        # Normalized rather than merely typed: tools: frozenset[str] is
        # hashable only if a frozenset is actually what ends up stored there,
        # and nothing before this stopped a caller from passing a plain,
        # unhashable set instead -- frozen=True's generated __hash__ exists
        # regardless of what the field holds, so isinstance(x, Hashable)
        # would still report True, and only calling hash(x) would raise,
        # naming "set" rather than this class. Mirrors Sandbox.__post_init__
        # (sandbox.py), which normalizes its own roots the same way.
        object.__setattr__(self, "tools", frozenset(self.tools))

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
        # is_within, not str.startswith: a sibling directory that merely
        # shares a root's string prefix -- "/psrc" against root "/p" -- is
        # nowhere near inside it (sandbox.py's is_within docstring names this
        # exact hazard for the same reason). str.startswith would chop "/p"
        # off "/psrc/evil.py" and hand "src/evil.py" to the rule matcher as a
        # plausible in-root relative path, which a rule like
        # ``Read(src/**)`` would then wrongly allow.
        #
        # Non-absolute subjects ("npm test", "*.py" -- Bash commands and Glob
        # patterns are not paths) are returned unchanged: is_within requires
        # both sides absolute and raises otherwise, and there is no root to
        # be relative to regardless.
        candidate = PurePosixPath(path)
        if not candidate.is_absolute():
            return path
        for root in self.sandbox.roots:
            if is_within(root, path):
                return str(candidate.relative_to(root))
        # Outside every root: returned unchanged rather than lstrip("/")-ed.
        # A path that is not inside any root should not produce a fabricated
        # relative candidate at all -- lstrip("/") on "/etc/passwd" yields
        # "etc/passwd", which is exactly the shape a rule like
        # ``Read(etc/**)`` would wrongly match, for a path nowhere near any
        # sandboxed root.
        return path


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
            if glob_matches_any(policy.secret_paths, (path, policy.relative(path))):
                return PermissionResult(
                    Decision.DENY,
                    "secret.path",
                    f"{path} looks like a credentials file and is never read. "
                    # Row 2 runs six rows before row 8 (rule.allow): no allow
                    # rule, however exact, can ever reach this request. Only
                    # --allow-secrets (this same row's own guard) or removing
                    # the path from secret_paths changes the outcome.
                    "Pass --allow-secrets to override.",
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
            # Branched on violation, not one message for both ids: outside-root
            # means the path itself is nowhere near any root, and --add-dir is
            # the remedy. symlink-escape means the opposite -- the path *is*
            # inside a root, only its resolved parent is not -- so "is outside
            # the working directory" would be false and --add-dir would not
            # help; the thing to do is inspect the symlink itself.
            if violation == "sandbox.symlink-escape":
                message = (
                    f"{path} is inside the working directory ({roots}), but its "
                    "containing directory resolves outside it once symlinks are "
                    "followed. This looks like a symlink planted to escape the "
                    "sandbox on write; inspect it by hand before writing here."
                )
            else:
                message = (
                    f"{path} is outside the working directory ({roots}). Use --add-dir to widen it."
                )
            return PermissionResult(Decision.DENY, violation, message)

    # 4. bash.dangerous / bash.unparseable
    if request.danger is not None:
        if request.danger.level is DangerLevel.BLOCKED:
            return PermissionResult(
                Decision.DENY,
                "bash.dangerous",
                f"refused by the {request.danger.classifier} classifier: "
                f"{request.danger.reason}. Rewrite the command to avoid the "
                "flagged pattern, or run it yourself outside the agent.",
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
