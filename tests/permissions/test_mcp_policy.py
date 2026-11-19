"""How the policy treats a tool that an MCP server brought, and the one rule id made for it.

An MCP tool is somebody else's code behind a name the model chooses, so it is not read-only,
not an edit and not allowed without asking: it asks, as anything that changes state does.
``mcp.changed`` is for the one case that is worse than that: a tool whose definition is not
what the person last saw. It asks in every mode, and neither a rule nor a grant lets it go
unasked.
"""

import pytest

from nanoclaude.permissions.policy import (
    Decision,
    Grants,
    PermissionMode,
    PermissionRequest,
    Policy,
    evaluate,
)
from nanoclaude.permissions.rules import RuleSet
from nanoclaude.permissions.sandbox import Sandbox

TOOL = "mcp__github__create_issue"
OTHER = "mcp__github__delete_repo"


def policy(mode: PermissionMode = PermissionMode.DEFAULT, **rules: list[str]) -> Policy:
    return Policy(Sandbox(("/p",)), RuleSet.build(**rules), mode)


def call(tool=TOOL, *, changed=False):
    """What an MCP tool's permission request is: no path, a subject that names the server,
    and a write, since nothing says that the tool does not change anything."""
    return PermissionRequest(tool, "server github", (), True, None, definition_changed=changed)


def decide(request, p, grants=None):
    result = evaluate(request, p, grants or Grants())
    return result.decision, result.rule


def test_an_mcp_tool_asks_by_default_and_the_reason_says_what_it_is():
    result = evaluate(call(), policy(), Grants())
    assert (result.decision, result.rule) == (Decision.ASK, "default.ask")
    assert "MCP server" in result.reason, result.reason


def test_accept_edits_does_not_allow_an_mcp_tool():
    assert decide(call(), policy(PermissionMode.ACCEPT_EDITS)) == (Decision.ASK, "default.ask")


def test_bypass_allows_an_mcp_tool_as_it_allows_any_tool():
    assert decide(call(), policy(PermissionMode.BYPASS)) == (Decision.ALLOW, "mode.bypass")


def test_plan_mode_refuses_an_mcp_tool_because_nothing_says_it_only_reads():
    assert decide(call(), policy(PermissionMode.PLAN)) == (Decision.DENY, "mode.plan-read-only")


def test_an_allow_rule_for_the_exact_name_allows_that_tool_and_no_other_mcp_tool():
    p = policy(allow=[TOOL])
    assert decide(call(), p) == (Decision.ALLOW, "rule.allow")
    assert decide(call(OTHER), p) == (Decision.ASK, "default.ask")


def test_an_ask_rule_and_a_deny_rule_for_the_exact_name_apply():
    assert decide(call(), policy(ask=[TOOL])) == (Decision.ASK, "rule.ask")
    assert decide(call(), policy(deny=[TOOL])) == (Decision.DENY, "rule.deny")


def test_a_session_grant_for_an_mcp_tool_allows_it_for_the_rest_of_the_session():
    granted = Grants(frozenset({TOOL}))
    assert decide(call(), policy(), granted) == (Decision.ALLOW, "grant.session")
    assert decide(call(OTHER), policy(), granted) == (Decision.ASK, "default.ask")


# ---- mcp.changed


ALL_MODES = list(PermissionMode)
UNCHANGED_ALLOWS = {
    # What the same call comes to when its definition has not changed, in each mode.
    PermissionMode.DEFAULT: (Decision.ALLOW, "rule.allow"),
    PermissionMode.ACCEPT_EDITS: (Decision.ALLOW, "rule.allow"),
    PermissionMode.BYPASS: (Decision.ALLOW, "mode.bypass"),
    PermissionMode.PLAN: (Decision.DENY, "mode.plan-read-only"),
}


@pytest.mark.parametrize("mode", [m for m in ALL_MODES if m is not PermissionMode.PLAN])
def test_a_changed_tool_asks_whatever_the_mode_the_rules_and_the_grants_say(mode):
    p = policy(mode, allow=[TOOL])
    granted = Grants(frozenset({TOOL}))
    # The guard: without the row, the same call is allowed, by the rule, the grant or the mode.
    assert decide(call(), p, granted) == UNCHANGED_ALLOWS[mode]
    result = evaluate(call(changed=True), p, granted)
    assert (result.decision, result.rule) == (Decision.ASK, "mcp.changed"), (mode, result)
    assert "definition" in result.reason and "changed" in result.reason, result.reason


@pytest.mark.parametrize("mode", ALL_MODES)
def test_a_changed_tool_is_never_allowed_in_any_mode_with_any_rule_or_grant(mode):
    for rules in ({}, {"allow": [TOOL]}, {"allow": [TOOL], "ask": [TOOL]}):
        for granted in (Grants(), Grants(frozenset({TOOL}))):
            result = evaluate(call(changed=True), policy(mode, **rules), granted)
            assert result.decision is not Decision.ALLOW, (mode, rules, granted, result)


def test_a_deny_rule_still_wins_over_a_changed_definition():
    result = evaluate(call(changed=True), policy(deny=[TOOL], allow=[TOOL]), Grants())
    assert (result.decision, result.rule) == (Decision.DENY, "rule.deny")


def test_plan_mode_still_refuses_a_changed_tool_which_is_the_stricter_answer():
    result = evaluate(call(changed=True), policy(PermissionMode.PLAN), Grants())
    assert (result.decision, result.rule) == (Decision.DENY, "mode.plan-read-only")


def test_an_unchanged_tool_next_to_a_changed_one_is_not_asked_about_again():
    p = policy(allow=[TOOL, OTHER])
    assert decide(call(OTHER, changed=True), p) == (Decision.ASK, "mcp.changed")
    assert decide(call(TOOL), p) == (Decision.ALLOW, "rule.allow")


def test_the_request_does_not_say_changed_unless_it_is_told_to():
    assert PermissionRequest("Read", "/p/a", ("/p/a",)).definition_changed is False
