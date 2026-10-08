from typing import Any

import pytest

from nanoclaude.permissions.danger import DangerLevel, DangerVerdict
from nanoclaude.permissions.danger.regex import RegexClassifier
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
    # Row 2 runs six rows before row 8 (rule.allow): no allow rule can ever
    # override it, so the message must not suggest one can.
    assert "allow rule" not in result.reason


def test_outside_the_sandbox_is_denied():
    result = evaluate(req(subject="/etc/passwd", paths=("/etc/passwd",)), policy(), Grants())
    assert result.rule == "sandbox.outside-root"


def test_a_sibling_directory_sharing_a_roots_string_prefix_does_not_reach_rule_allow():
    """Policy.relative used str.startswith, so a sibling like /psrc (sharing
    the string prefix of root /p without being inside it, the same hazard
    sandbox.is_within's own docstring names for /tmp/project-evil against
    /tmp/project) got "/p" chopped off and the rest -- "src/evil.py" --
    handed to the rule matcher as a plausible in-root relative path, which
    an allow rule meant for the real root's src/** would then wrongly match.

    Write, not Read: Read is read-only and would fall through to
    tool.read-only regardless of whether rule.allow fires, which would still
    prove row 8 is closed but leave the overall decision ambiguous. Write has
    no such fallback -- the only way this reaches anything but default.ask is
    if row 8 (wrongly) fires.
    """
    p = Policy(Sandbox(("/p",)), RuleSet.build(allow=["Write(src/**)"]))
    request = PermissionRequest("Write", "/psrc/evil.py", (), True, None)
    result = evaluate(request, p, Grants())
    assert result.rule != "rule.allow"
    assert (result.decision, result.rule) == (Decision.ASK, "default.ask")


def test_a_symlink_escape_violation_reaches_evaluate_with_its_own_rule_id(tmp_path):
    """sandbox.symlink-escape never flowed through evaluate() in any test
    before this one -- nothing pinned that row 3 passes it through unchanged
    rather than rewriting it to sandbox.outside-root. Setup mirrors
    test_check_write_catches_a_symlinked_parent_that_was_never_resolved in
    test_sandbox.py: a path built by plain string-joining, never passed
    through Sandbox.resolve(), whose parent directory is a symlink escaping
    the root -- check_read is fooled (textually inside root); only
    check_write's own parent re-resolution catches it.
    """
    outside = tmp_path.parent / "outside_dir_for_policy_symlink_test"
    outside.mkdir(exist_ok=True)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    p = Policy(Sandbox((str(root),)), RuleSet.build())
    target = str(root / "escape" / "new.txt")
    result = evaluate(PermissionRequest("Write", target, (target,), True, None), p, Grants())

    assert result.rule == "sandbox.symlink-escape"
    # The message must not be outside-root's -- the path *is* textually
    # inside the root; only its resolved parent is not -- so "is outside the
    # working directory" would be false here, and --add-dir is not the fix.
    assert "is outside the working directory" not in result.reason
    assert "--add-dir" not in result.reason


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
    # Row 1 beats row 6 too. Before this assertion, deny= appeared in exactly
    # one test (test_a_deny_rule_beats_everything), which runs in DEFAULT mode
    # with no grants -- rows 2-7 never fire for that request either way, so
    # row 1 could have been moved anywhere between rows 2 and 7, including
    # below mode.bypass, and that test alone would stay green.
    p_bypass_with_deny = policy(
        mode=PermissionMode.BYPASS, rules=RuleSet.build(deny=["Read(**/secret.txt)"])
    )
    denied_under_bypass = evaluate(
        req(subject="/p/secret.txt", paths=("/p/secret.txt",)), p_bypass_with_deny, Grants()
    )
    assert denied_under_bypass.rule == "rule.deny"


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

    # Row 1 beats row 9 too: a deny rule fires even when the same tool has a
    # session grant. The grant is for Read specifically -- the tool being
    # denied -- so that if row 1 were ever moved below row 9, this would
    # flip to ALLOW via grant.session instead of catching the misordering.
    p_with_deny = policy(rules=RuleSet.build(deny=["Read(**/secret.txt)"]))
    denied = evaluate(
        req(subject="/p/secret.txt", paths=("/p/secret.txt",)),
        p_with_deny,
        Grants(frozenset({"Read"})),
    )
    assert denied.rule == "rule.deny"


def test_a_read_only_tool_with_no_matching_rule_is_still_allowed():
    p = policy(rules=RuleSet.build(ask=["Bash"]))
    assert evaluate(req(tool="Glob", subject="*.py", paths=()), p, Grants()).rule == (
        "tool.read-only"
    )


def test_anything_unmatched_falls_through_to_ask():
    p = policy(rules=RuleSet.build())
    result = evaluate(req(tool="Write", is_write=True), p, Grants())
    assert (result.decision, result.rule) == (Decision.ASK, "default.ask")


_REASON_CASES = [
    pytest.param(
        lambda: evaluate(
            req(subject="/p/secret.txt", paths=("/p/secret.txt",)),
            policy(rules=RuleSet.build(allow=["Read"], deny=["Read(**/secret.txt)"])),
            Grants(),
        ),
        "rule.deny",
        "Read(**/secret.txt)",
        id="rule.deny",
    ),
    pytest.param(
        lambda: evaluate(req(subject="/p/.env", paths=("/p/.env",)), policy(), Grants()),
        "secret.path",
        "--allow-secrets",
        id="secret.path",
    ),
    pytest.param(
        lambda: evaluate(req(subject="/etc/passwd", paths=("/etc/passwd",)), policy(), Grants()),
        "sandbox.outside-root",
        "--add-dir",
        id="sandbox.outside-root",
    ),
    pytest.param(
        lambda: evaluate(
            req(
                tool="Bash",
                subject="rm -rf /",
                paths=(),
                danger=DangerVerdict(DangerLevel.BLOCKED, ("rm.recursive-force",), "regex"),
            ),
            policy(),
            Grants(),
        ),
        "bash.dangerous",
        "rm.recursive-force",
        id="bash.dangerous",
    ),
    pytest.param(
        lambda: evaluate(
            req(
                tool="Bash",
                subject="$(",
                paths=(),
                danger=DangerVerdict(DangerLevel.UNPARSEABLE, (), "ast"),
            ),
            policy(),
            Grants(),
        ),
        "bash.unparseable",
        "Rewrite",
        id="bash.unparseable",
    ),
    pytest.param(
        lambda: evaluate(
            req(tool="Write", is_write=True), policy(mode=PermissionMode.PLAN), Grants()
        ),
        "mode.plan-read-only",
        "exit plan mode",
        id="mode.plan-read-only",
    ),
    pytest.param(
        lambda: evaluate(
            req(tool="Bash", subject="ls", paths=()), policy(mode=PermissionMode.BYPASS), Grants()
        ),
        "mode.bypass",
        "disabled",
        id="mode.bypass",
    ),
    pytest.param(
        lambda: evaluate(
            req(tool="Edit", is_write=True), policy(mode=PermissionMode.ACCEPT_EDITS), Grants()
        ),
        "mode.accept-edits",
        "auto-approved",
        id="mode.accept-edits",
    ),
    pytest.param(
        lambda: evaluate(
            req(tool="Bash", subject="ls", paths=()),
            policy(rules=RuleSet.build(allow=["Bash"])),
            Grants(),
        ),
        "rule.allow",
        "allowed by Bash",
        id="rule.allow",
    ),
    pytest.param(
        lambda: evaluate(
            req(tool="Bash", subject="ls", paths=()), policy(), Grants(frozenset({"Bash"}))
        ),
        "grant.session",
        "approved for this session",
        id="grant.session",
    ),
    pytest.param(
        lambda: evaluate(
            req(tool="Glob", subject="*.py", paths=()),
            policy(rules=RuleSet.build(ask=["Bash"])),
            Grants(),
        ),
        "tool.read-only",
        "only reads",
        id="tool.read-only",
    ),
    pytest.param(
        lambda: evaluate(req(tool="Bash", subject="ls", paths=()), policy(), Grants()),
        "rule.ask",
        "requires confirmation",
        id="rule.ask",
    ),
    pytest.param(
        lambda: evaluate(req(tool="Write", is_write=True), policy(rules=RuleSet.build()), Grants()),
        "default.ask",
        "no matching rule",
        id="default.ask",
    ),
]


@pytest.mark.parametrize("make_result, expected_rule, fragment", _REASON_CASES)
def test_the_reason_always_names_something_actionable(make_result, expected_rule, fragment):
    """Every one of the twelve rows, not just sandbox.outside-root -- the
    original, single-case form of this test exercised one row in twelve
    despite its own name's "always".
    """
    result = make_result()
    assert result.rule == expected_rule
    assert fragment in result.reason


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
    # tools: frozenset[str] -- hashable because __post_init__ normalizes
    # whatever it is given into a real frozenset (see the next test for the
    # trap that existed before that normalization).
    one = Grants(frozenset({"Bash"}))
    other = Grants(frozenset({"Bash"}))
    assert hash(one) == hash(other)


def test_grants_normalizes_a_plain_set_into_a_hashable_frozenset():
    """The conditional-hashability trap this task's global constraints name,
    word for word: Grants({"Bash"}) used to store the plain set unchanged.
    isinstance(x, Hashable) still reported True -- frozen=True's generated
    __hash__ exists regardless of what tools actually holds -- and only
    calling hash(x) raised, naming "set" rather than Grants.

    Grants({"Bash"}) is itself a type mypy --strict correctly refuses for a
    *typed* caller (tools is declared frozenset[str], and a plain set is a
    different, unrelated type) -- hence the ignore below. The point of this
    test is the untyped or dynamically-built caller __post_init__ now
    protects regardless: Tasks 7-8's session-grant consumers, a config
    loader, anything that hands Grants a set it built elsewhere.
    """
    grants = Grants({"Bash"})  # type: ignore[arg-type]
    assert isinstance(grants.tools, frozenset)
    assert hash(grants) == hash(Grants(frozenset({"Bash"})))


def test_policy_normalizes_a_list_of_secret_paths_so_evaluate_does_not_raise():
    """The conditional-hashability trap this task's global constraints name,
    for Policy.secret_paths rather than Grants.tools (see
    test_grants_normalizes_a_plain_set_into_a_hashable_frozenset above):
    RuleSet.build's own allow/ask/deny parameters are typed list[str] | None,
    so a list is a natural shape for a caller to reach for here too, and
    before this normalization isinstance(p, Hashable) still reported True
    while hash(p) raised, naming "list" rather than Policy.

    The call that actually matters is evaluate() itself, not an explicit
    hash(p) nobody makes in real use: the secret.path check hands
    secret_paths straight to glob_matches_any, whose @functools.cache-d
    helper hashes its patterns argument on every call, so an unnormalized
    list reached that cache and raised TypeError from inside the decision
    path, for a request that used to return a normal DENY result.
    """
    p = Policy(Sandbox((ROOT,)), RuleSet.build(), secret_paths=["**/.env*"])  # type: ignore[arg-type]
    assert isinstance(p.secret_paths, tuple)
    assert hash(p) == hash(Policy(Sandbox((ROOT,)), RuleSet.build(), secret_paths=("**/.env*",)))
    result = evaluate(PermissionRequest("Read", "/p/.env", ("/p/.env",), False, None), p, Grants())
    assert (result.decision, result.rule) == (Decision.DENY, "secret.path")


def test_grants_rejects_a_bare_string_instead_of_scattering_it_into_characters():
    """frozenset() does not reject a string -- it iterates one like any
    other sequence, so Grants("Bash") used to silently produce
    frozenset({"B", "a", "s", "h"}): a grant that matches no real tool name,
    not even "Bash" itself. mypy --strict already rejects this at a typed
    call site (tools: frozenset[str] is not satisfied by str, hence the
    ignore below), so this guards the untyped or dynamically-built caller
    instead -- the same population Grants.__post_init__'s existing
    normalization protects against an unhashable plain set.
    """
    with pytest.raises(TypeError, match="one grant per character"):
        Grants("Bash")  # type: ignore[arg-type]


def test_relative_of_a_path_equal_to_a_root_returns_a_single_dot():
    """PurePosixPath.relative_to gives "." for a path equal to the root
    itself; the pre-round-1 code (plain string slicing) gave "" for the same
    input. Pinning the current behavior as a decision on record rather than
    an accident: a rule written as "**" or "*" matches both "." and "" --
    and, regardless, already matched this same request via the raw,
    un-relativized candidate that also sits in evaluate()'s `subjects` tuple
    -- so the only rule shape this could ever change the outcome for is one
    whose subject is the literal "." itself (e.g. Read(.)), and nothing in
    this suite, or the brief, writes one.
    """
    p = Policy(Sandbox((ROOT,)), RuleSet.build())
    assert p.relative(ROOT) == "."


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


def test_evaluate_defaults_to_no_grants_when_the_argument_is_omitted():
    """_NO_GRANTS (evaluate()'s module-level default for its grants
    parameter) was constructed at import time but never actually exercised
    as a default anywhere above -- every call passed Grants() explicitly, so
    line coverage reported it covered only because the assignment statement
    itself runs at import. Bash has no grant here, so omitting the argument
    must still fall through to row 11 (rule.ask), not row 9 (grant.session),
    proving the default really does behave like an empty Grants rather than,
    say, None or a crash.
    """
    result = evaluate(req(tool="Bash", subject="ls", paths=()), policy())
    assert (result.decision, result.rule) == (Decision.ASK, "rule.ask")


# Step 7b: the degradation guarantee (spec 6.4, 17) enforced in code, not just
# in prose. A SAFE verdict whose classifier was not authoritative -- every SAFE
# the regex classifier gives, since it can refuse a command but never clear
# one -- must not be promoted to ALLOW by row 8 or row 9. All four directions are pinned, not
# just the negative case: a guard proven only by its denial is the defect the
# completeness clause (spec, test-completeness) names.
SAFE_AST = DangerVerdict(DangerLevel.SAFE, (), "ast")
SAFE_REGEX_ONLY = DangerVerdict(DangerLevel.SAFE, (), "regex", authoritative=False)


def test_an_allow_rule_promotes_a_command_the_ast_classifier_cleared():
    p = policy(rules=RuleSet.build(allow=["Bash(npm test:*)"]))
    result = evaluate(req(tool="Bash", subject="npm test", paths=(), danger=SAFE_AST), p)
    assert (result.decision, result.rule) == (Decision.ALLOW, "rule.allow")


def test_an_allow_rule_does_not_promote_a_command_only_the_regex_classifier_cleared():
    # The constraint this pins: regex-only means "not cleared", not "safe".
    p = policy(rules=RuleSet.build(allow=["Bash(npm test:*)"], ask=["Bash"]))
    result = evaluate(req(tool="Bash", subject="npm test", paths=(), danger=SAFE_REGEX_ONLY), p)
    assert result.decision is Decision.ASK
    assert result.rule != "rule.allow"


def test_a_session_grant_promotes_a_command_the_ast_classifier_cleared():
    p = policy(rules=RuleSet.build(ask=["Bash"]))
    result = evaluate(
        req(tool="Bash", subject="ls", paths=(), danger=SAFE_AST), p, Grants(frozenset({"Bash"}))
    )
    assert (result.decision, result.rule) == (Decision.ALLOW, "grant.session")


def test_a_session_grant_does_not_promote_a_command_only_the_regex_classifier_cleared():
    p = policy(rules=RuleSet.build(ask=["Bash"]))
    result = evaluate(
        req(tool="Bash", subject="ls", paths=(), danger=SAFE_REGEX_ONLY),
        p,
        Grants(frozenset({"Bash"})),
    )
    assert result.decision is Decision.ASK
    assert result.rule != "grant.session"


def test_a_blocked_verdict_is_refused_whether_or_not_it_is_authoritative():
    # Row 4 precedes both gated rows, so the flag must change nothing here.
    for authoritative in (True, False):
        verdict = DangerVerdict(
            DangerLevel.BLOCKED, ("rm.recursive-force",), "regex", authoritative=authoritative
        )
        p = policy(rules=RuleSet.build(allow=["Bash"]))
        result = evaluate(req(tool="Bash", subject="rm -rf /", paths=(), danger=verdict), p)
        assert result.rule == "bash.dangerous"


def test_todowrite_is_allowed_without_asking_like_a_read_only_tool():
    # Spec 5.1: TodoWrite is not read-only, but its default tier is "allow" --
    # it changes only the session's own task list.
    result = evaluate(
        req(tool="TodoWrite", subject="", paths=()), policy(rules=RuleSet.build()), Grants()
    )
    assert (result.decision, result.rule) == (Decision.ALLOW, "tool.read-only")


def test_a_file_changing_tool_is_not_in_the_allowed_without_asking_set():
    result = evaluate(
        req(tool="Write", subject="/p/a.py", paths=("/p/a.py",)),
        policy(rules=RuleSet.build()),
        Grants(),
    )
    assert result.decision is Decision.ASK


@pytest.mark.parametrize("tool", ["Read", "Grep", "Glob", "TodoWrite"])
def test_an_ask_rule_cannot_make_a_tool_that_is_allowed_without_asking_ask(tool):
    """Row 10 comes before row 11, so the way to keep a read from happening is deny.

    The shipped configuration lists TodoWrite among the tools it asks about, and it is
    allowed all the same; this is the reason, and the configuration reference says so.
    """
    p = policy(rules=RuleSet.build(ask=[tool]))
    result = evaluate(req(tool=tool, subject="x", paths=()), p, Grants())
    assert (result.decision, result.rule) == (Decision.ALLOW, "tool.read-only")


# The regex classifier's own verdicts, not hand-built ones: what it says about a
# command it found nothing in is "nothing refused this", never a clearance.
def test_the_regex_classifier_never_clears_a_command_it_found_nothing_in():
    verdict = RegexClassifier().classify("npm test")
    assert verdict.level is DangerLevel.SAFE
    assert verdict.authoritative is False


def test_an_allow_rule_does_not_promote_the_regex_classifiers_own_safe_verdict():
    p = policy(rules=RuleSet.build(allow=["Bash(npm test:*)"], ask=["Bash"]))
    danger = RegexClassifier().classify("npm test")
    result = evaluate(req(tool="Bash", subject="npm test", paths=(), danger=danger), p)
    assert result.decision is Decision.ASK
    assert result.rule != "rule.allow"


def test_a_session_grant_does_not_promote_the_regex_classifiers_own_safe_verdict():
    p = policy(rules=RuleSet.build(ask=["Bash"]))
    danger = RegexClassifier().classify("ls")
    result = evaluate(
        req(tool="Bash", subject="ls", paths=(), danger=danger), p, Grants(frozenset({"Bash"}))
    )
    assert result.decision is Decision.ASK
    assert result.rule != "grant.session"
