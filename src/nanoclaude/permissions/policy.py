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
        # Read through an object-typed local rather than checking self.tools
        # directly: tools is declared frozenset[str], so mypy --strict can
        # prove a direct `isinstance(self.tools, str)` dead (frozenset and
        # str have "distinct disjoint bases") and refuses it as unreachable.
        # That proof holds only for a *typed* caller -- the entire point of
        # this guard is the untyped or Any-typed one the proof cannot see,
        # e.g. a config loader -- so the check must survive at runtime; this
        # local only widens what mypy believes, not what is actually there.
        tools_as_given: object = self.tools
        if isinstance(tools_as_given, str):
            # frozenset() does not reject a string -- it iterates one, same
            # as any other sequence, which turns a single tool name into a
            # grant for each of its characters instead of a grant for the
            # tool. That grant is strictly narrower than the caller intended
            # rather than wider (it matches nothing a real tool name is ever
            # equal to, including the very name that was passed), so this is
            # a correctness and debuggability defect rather than a safety
            # one -- but a silent wrong answer here is worth turning into a
            # loud one before it reaches the normalization below.
            raise TypeError(
                f"Grants.tools got the string {tools_as_given!r}, not a "
                "collection of tool names -- frozenset() would scatter it "
                "into one grant per character instead of one grant for the "
                'whole name. Pass a collection instead, such as {"Bash"} or '
                '["Bash"].'
            )
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

    def __post_init__(self) -> None:
        # Normalized rather than merely typed: secret_paths: tuple[str, ...]
        # is hashable only if a tuple is actually what ends up stored there,
        # and nothing before this stopped a caller from passing a plain,
        # unhashable list instead -- frozen=True's generated __hash__ exists
        # regardless of what the field holds, so isinstance(x, Hashable)
        # would still report True, and only calling hash(x) would raise,
        # naming "list" rather than this class. In practice the raise does
        # not wait for some explicit hash(policy) nobody calls: evaluate()'s
        # secret.path check hands this field straight to glob_matches_any,
        # whose @functools.cache-d helper hashes its patterns argument on
        # every call, so an unnormalized list reaches that cache and raises
        # from inside the decision path itself. Mirrors Grants.__post_init__
        # (above) and Sandbox.__post_init__ (sandbox.py), which normalize
        # their own fields the same way.
        object.__setattr__(self, "secret_paths", tuple(self.secret_paths))

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


def _unpromotable(danger: DangerVerdict | None) -> bool:
    """True when a SAFE verdict must not be promoted to ALLOW by rows 8-9.

    A verdict's ``authoritative`` defaults to True; the one thing allowed to
    set it False is a fallback classifier owning up to being unable to
    actually clear a command (spec 6.4's degradation guarantee -- "without
    [the AST classifier] ... any command it cannot clear resolves to `ask`,
    never `allow`"). SAFE-but-not-authoritative therefore means "nothing
    refused this," not "this is safe," so rule.allow and grant.session must
    not treat it as a clearance. ``danger is None`` (every file tool) and a
    BLOCKED/UNPARSEABLE verdict both return False here: the former never had
    an opinion to be non-authoritative about, and the latter is already
    denied earlier, at row 4, regardless of this flag.
    """
    return danger is not None and danger.level is DangerLevel.SAFE and not danger.authoritative


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
    if allowed is not None and not _unpromotable(request.danger):
        return PermissionResult(Decision.ALLOW, "rule.allow", f"allowed by {allowed.source}")

    # 9. grant.session
    if request.tool in grants.tools and not _unpromotable(request.danger):
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
