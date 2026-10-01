from typing import Any

from nanoclaude.permissions.danger import DangerLevel, DangerVerdict
from nanoclaude.permissions.policy import (
    Decision,
    Grants,
    PermissionMode,
    PermissionRequest,
    PermissionResult,
    Policy,
    evaluate,
)
from nanoclaude.permissions.rules import RuleSet
from nanoclaude.permissions.sandbox import Sandbox

ROOT = "/p"


def policy(**kw: Any) -> Policy:
    # base is explicitly dict[str, Any], not left for mypy to infer: the four
    # literal values below (a Sandbox, a RuleSet, a PermissionMode, a tuple)
    # share no common base but object, so an inferred dict[str, object] makes
    # every keyword below an arg-type error at the **base call -- object is
    # not assignable to Sandbox, RuleSet, and so on. Any is the correct type
    # for a kwargs-merging helper like this one; the annotation changes
    # nothing at runtime.
    base: dict[str, Any] = {
        "sandbox": Sandbox((ROOT,)),
        "rules": RuleSet.build(allow=["Read", "Glob", "Grep"], ask=["Bash", "Write", "Edit"]),
        "mode": PermissionMode.DEFAULT,
        "secret_paths": ("**/.env*", "**/id_rsa"),
    }
    base.update(kw)
    return Policy(**base)


def req(tool="Read", subject="/p/a.py", paths=("/p/a.py",), is_write=False, danger=None):
    return PermissionRequest(tool, subject, paths, is_write, danger)


def test_a_deny_rule_beats_everything():
    p = policy(rules=RuleSet.build(allow=["Read"], deny=["Read(**/secret.txt)"]))
    result = evaluate(req(subject="/p/secret.txt", paths=("/p/secret.txt",)), p, Grants())
    assert (result.decision, result.rule) == (Decision.DENY, "rule.deny")


def test_secret_paths_are_denied_before_the_sandbox_is_even_consulted():
    result = evaluate(req(subject="/p/.env", paths=("/p/.env",)), policy(), Grants())
    assert (result.decision, result.rule) == (Decision.DENY, "secret.path")


def test_outside_the_sandbox_is_denied():
    result = evaluate(req(subject="/etc/passwd", paths=("/etc/passwd",)), policy(), Grants())
    assert result.rule == "sandbox.outside-root"


def test_a_dangerous_command_is_denied():
    verdict = DangerVerdict(DangerLevel.BLOCKED, ("rm.recursive-force",), "regex")
    result = evaluate(
        req(tool="Bash", subject="rm -rf /", paths=(), danger=verdict), policy(), Grants()
    )
    assert (result.decision, result.rule) == (Decision.DENY, "bash.dangerous")


def test_an_unparseable_command_is_denied():
    verdict = DangerVerdict(DangerLevel.UNPARSEABLE, (), "ast")
    result = evaluate(req(tool="Bash", subject="$(", paths=(), danger=verdict), policy(), Grants())
    assert result.rule == "bash.unparseable"


def test_plan_mode_denies_writes_but_allows_reads():
    p = policy(mode=PermissionMode.PLAN)
    assert evaluate(req(tool="Write", is_write=True), p, Grants()).rule == "mode.plan-read-only"
    assert evaluate(req(tool="Read"), p, Grants()).decision is Decision.ALLOW


def test_bypass_skips_confirmation_but_not_the_hard_denials():
    """Spec 6.2.1. --dangerously-skip-permissions removes prompts, not the sandbox."""
    p = policy(mode=PermissionMode.BYPASS)
    assert evaluate(req(tool="Bash", subject="ls", paths=()), p, Grants()).rule == "mode.bypass"
    assert (
        evaluate(req(subject="/etc/passwd", paths=("/etc/passwd",)), p, Grants()).rule
        == "sandbox.outside-root"
    )
    assert evaluate(req(subject="/p/.env", paths=("/p/.env",)), p, Grants()).rule == "secret.path"
    verdict = DangerVerdict(DangerLevel.BLOCKED, ("rm.recursive-force",), "regex")
    assert (
        evaluate(req(tool="Bash", subject="rm -rf /", paths=(), danger=verdict), p, Grants()).rule
        == "bash.dangerous"
    )


def test_accept_edits_allows_file_edits_and_still_asks_for_bash():
    p = policy(mode=PermissionMode.ACCEPT_EDITS)
    assert evaluate(req(tool="Edit", is_write=True), p, Grants()).rule == "mode.accept-edits"
    assert evaluate(req(tool="Bash", subject="ls", paths=()), p, Grants()).decision is Decision.ASK


def test_a_session_grant_upgrades_ask_to_allow():
    # frozenset({"Bash"}), not the {"Bash"} set literal: Grants.tools is typed
    # frozenset[str], and a plain set is a different, unhashable type that
    # mypy --strict correctly rejects as an argument here.
    result = evaluate(
        req(tool="Bash", subject="ls", paths=()), policy(), Grants(frozenset({"Bash"}))
    )
    assert (result.decision, result.rule) == (Decision.ALLOW, "grant.session")


def test_a_read_only_tool_with_no_matching_rule_is_still_allowed():
    p = policy(rules=RuleSet.build(ask=["Bash"]))
    assert evaluate(req(tool="Glob", subject="*.py", paths=()), p, Grants()).rule == (
        "tool.read-only"
    )


def test_anything_unmatched_falls_through_to_ask():
    p = policy(rules=RuleSet.build())
    result = evaluate(req(tool="Write", is_write=True), p, Grants())
    assert (result.decision, result.rule) == (Decision.ASK, "default.ask")


def test_the_reason_always_names_something_actionable():
    result = evaluate(req(subject="/etc/passwd", paths=("/etc/passwd",)), policy(), Grants())
    assert "/etc/passwd" in result.reason and ROOT in result.reason


# The tests below pin names from the brief's Interfaces -> Produces list, or
# branches left uncovered by the twelve tests above (confirmed by
# --cov-report=term-missing), rather than any of the brief's own steps.


def test_with_tool_grants_the_named_tool_without_mutating_the_original():
    # Grants is explicitly named with .with_tool(name) in the Produces list,
    # and none of the twelve tests above call it -- evaluate() only ever
    # receives grants built directly from a Grants(...) literal.
    before = Grants()
    after = before.with_tool("Bash")
    assert "Bash" in after.tools
    assert before.tools == frozenset()  # frozen: the original is untouched


def test_permission_result_allowed_is_true_only_for_allow():
    allowed = evaluate(req(tool="Read"), policy(), Grants())
    denied = evaluate(req(subject="/etc/passwd", paths=("/etc/passwd",)), policy(), Grants())
    asked = evaluate(req(tool="Bash", subject="ls", paths=()), policy(), Grants())
    assert allowed.allowed is True
    assert denied.allowed is False
    assert asked.allowed is False


def test_permission_request_is_hashable():
    # Fields are tool: str, subject: str, resolved_paths: tuple[str, ...],
    # is_write: bool, danger: DangerVerdict | None -- DangerVerdict is itself
    # hashable (tests/permissions/test_danger.py), so this composes cleanly.
    # Pinned so a later field of a mapping or other unhashable type gets
    # caught the same way ToolUseBlock's was (conversation/transcript.py).
    one = PermissionRequest("Read", "/p/a.py", ("/p/a.py",), False, None)
    other = PermissionRequest("Read", "/p/a.py", ("/p/a.py",), False, None)
    assert hash(one) == hash(other)


def test_permission_result_is_hashable():
    # Fields are decision: Decision (a str enum), rule: str, reason: str --
    # all already hashable. Pinned for the same reason as the test above.
    one = PermissionResult(Decision.ALLOW, "rule.allow", "ok")
    other = PermissionResult(Decision.ALLOW, "rule.allow", "ok")
    assert hash(one) == hash(other)


def test_grants_is_hashable():
    # tools: frozenset[str] -- hashable as long as it is actually built as a
    # frozenset (see test_a_session_grant_upgrades_ask_to_allow, which pins
    # the one call site here that has to pass a real frozenset rather than a
    # plain set for exactly this reason).
    one = Grants(frozenset({"Bash"}))
    other = Grants(frozenset({"Bash"}))
    assert hash(one) == hash(other)


def test_policy_is_hashable():
    # Fields are sandbox: Sandbox (hashable, tests/permissions/test_sandbox.py),
    # rules: RuleSet (hashable, tests/permissions/test_rules.py), mode: a str
    # enum, secret_paths: tuple[str, ...], allow_secrets: bool -- all
    # hashable, so this composes cleanly all the way down.
    one = Policy(Sandbox((ROOT,)), RuleSet.build(allow=["Read"]))
    other = Policy(Sandbox((ROOT,)), RuleSet.build(allow=["Read"]))
    assert hash(one) == hash(other)


def test_a_policy_with_no_secret_paths_configured_never_denies_on_that_basis():
    # The other direction of test_secret_paths_are_denied_before_the_sandbox_is
    # _even_consulted: that test pins the rule firing when secret_paths is
    # non-empty, but never exercises secret_paths=(), the dataclass's own
    # default. Without this, a regression that always ran the secret check --
    # even with nothing configured to check against -- could not be told
    # apart from the guard clause that skips it, since both currently read as
    # "not denied by secret.path" on every test above.
    p = policy(secret_paths=())
    result = evaluate(req(subject="/p/.env", paths=("/p/.env",)), p, Grants())
    assert (result.decision, result.rule) == (Decision.ALLOW, "rule.allow")


# The three tests below were added after actually deleting rows from evaluate()
# one at a time and finding that the full suite, including every test above,
# stayed green -- the task's own standard for an unpinned branch. Each
# docstring records what deleting the row did instead, with nothing left to
# catch it.


def test_allow_secrets_opts_out_of_the_secret_path_check():
    """Row 2 is also gated by Policy.allow_secrets, default False.

    Replacing ``if not policy.allow_secrets:`` with ``if True:`` left every
    test above green, because none of them ever set allow_secrets=True.
    """
    p = policy(allow_secrets=True)
    result = evaluate(req(subject="/p/.env", paths=("/p/.env",)), p, Grants())
    assert (result.decision, result.rule) == (Decision.ALLOW, "rule.allow")


def test_an_explicit_allow_rule_produces_rule_allow():
    """Row 8. Deleting it left every test above green.

    Every other ALLOW test above uses a read-only tool (Read/Glob/Grep),
    which row 10 (tool.read-only) also allows -- deleting row 8 just made
    those requests fall through to row 10 instead, and no test checked
    .rule precisely enough to notice. Bash is never read-only, so an
    explicit allow rule for it can only be satisfied by row 8.
    """
    p = policy(rules=RuleSet.build(allow=["Bash"]))
    result = evaluate(req(tool="Bash", subject="ls", paths=()), p, Grants())
    assert (result.decision, result.rule) == (Decision.ALLOW, "rule.allow")


def test_an_explicit_ask_rule_produces_rule_ask():
    """Row 11. Deleting it left every test above green.

    test_accept_edits_allows_file_edits_and_still_asks_for_bash sends this
    same Bash request through this same row, but asserts only .decision --
    row 12 (default.ask) produces the same Decision.ASK, so deleting row 11
    made that request fall through to row 12 and nothing noticed.
    """
    result = evaluate(req(tool="Bash", subject="ls", paths=()), policy(), Grants())
    assert (result.decision, result.rule) == (Decision.ASK, "rule.ask")
