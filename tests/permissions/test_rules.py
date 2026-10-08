import pytest

from nanoclaude.permissions.rules import KNOWN_TOOLS, Rule, RuleSet, glob_matches_any
from nanoclaude.tools.registry import default_registry


def test_a_bare_tool_name_matches_every_call_to_it():
    rule = Rule.parse("Read")
    assert rule.matches("Read", "/p/a.py", "a.py")
    assert not rule.matches("Write", "/p/a.py", "a.py")


def test_a_glob_matches_the_path_relative_to_the_root():
    rule = Rule.parse("Read(**/.env*)")
    assert rule.matches("Read", "/p/svc/.env.local", "svc/.env.local")
    assert not rule.matches("Read", "/p/src/main.py", "src/main.py")


def test_a_glob_also_matches_the_absolute_path():
    """A rule the user wrote as an absolute path must still work."""
    rule = Rule.parse("Read(/etc/**)")
    assert rule.matches("Read", "/etc/passwd", "../../etc/passwd")


def test_a_bash_prefix_rule_matches_on_word_boundaries():
    rule = Rule.parse("Bash(npm test:*)")
    assert rule.matches("Bash", "npm test", "npm test")
    assert rule.matches("Bash", "npm test -- --watch", "npm test -- --watch")
    assert not rule.matches("Bash", "npm testify", "npm testify")  # boundary, not prefix
    assert not rule.matches("Bash", "rm -rf /", "rm -rf /")


def test_an_exact_bash_rule_matches_only_that_command():
    rule = Rule.parse("Bash(git status)")
    assert rule.matches("Bash", "git status", "git status")
    assert not rule.matches("Bash", "git status --short", "git status --short")


def test_malformed_rules_are_rejected_at_parse_time():
    with pytest.raises(ValueError, match="unbalanced"):
        Rule.parse("Bash(npm test")


def test_a_lone_closing_paren_with_no_opening_one_is_also_rejected():
    """Distinct branch from the one above: no "(" at all, but a stray ")"."""
    with pytest.raises(ValueError, match="unbalanced"):
        Rule.parse("Bash)")


def test_an_empty_subject_is_rejected_at_parse_time():
    """Read() parses without error and silently matches nothing, ever.

    That is the dangerous failure direction for a deny rule: a user who
    writes one believes it refuses something, and it never fires.
    """
    with pytest.raises(ValueError, match="empty subject"):
        Rule.parse("Read()")


def test_an_empty_prefix_subject_is_also_rejected():
    """Same hazard, reached through the :* prefix form instead: Read(:*)."""
    with pytest.raises(ValueError, match="empty subject"):
        Rule.parse("Read(:*)")


def test_ruleset_returns_the_first_matching_rule():
    rules = RuleSet.build(allow=["Read", "Glob"], ask=["Bash"], deny=["Read(**/.env*)"])
    assert rules.first_match("deny", "Read", "/p/.env", ".env") is not None
    assert rules.first_match("allow", "Read", "/p/a.py", "a.py") is not None
    assert rules.first_match("allow", "Bash", "ls", "ls") is None


def test_rule_is_hashable():
    # Every field (tool: str, subject: str | None, prefix: bool, source: str) is
    # already hashable on its own, so frozen=True's generated __hash__ is not a
    # trap here, unlike ToolUseBlock (conversation/transcript.py), which holds a
    # Mapping. Pinned so a later field of a mapping or other unhashable type
    # gets caught the same way those were.
    assert hash(Rule.parse("Read(**/.env*)")) == hash(Rule.parse("Read(**/.env*)"))


def test_ruleset_is_hashable():
    # Holds three tuples of Rule, and Rule is hashable (see above), so the
    # generated __hash__ composes cleanly. Pinned for the same reason as
    # test_rule_is_hashable.
    one = RuleSet.build(allow=["Read"], ask=["Bash"], deny=["Read(**/.env*)"])
    other = RuleSet.build(allow=["Read"], ask=["Bash"], deny=["Read(**/.env*)"])
    assert hash(one) == hash(other)


def test_a_newline_in_a_directory_name_does_not_stop_a_leading_double_star_matching():
    assert glob_matches_any(("**/.env",), ("odd\ndir/.env",))
    assert glob_matches_any(("**/.env",), ("a/odd\ndir/.env",))
    assert Rule.parse("Read(**/vault.txt)").matches("Read", "/p/a\nb/vault.txt", "a\nb/vault.txt")


def test_a_newline_does_not_make_a_glob_match_what_it_should_not():
    assert not glob_matches_any(("**/.env",), ("odd\ndir/.envrc.txt",))
    assert not glob_matches_any(("src/**/*.py",), ("src/odd\ndir/a.txt",))
    # Two different names stay different: the newline is replaced, not dropped.
    assert not glob_matches_any(("src/ab.txt",), ("src/a\nb.txt",))


# ---- a rule that names no tool is refused, not left to match nothing for ever

REAL_TOOLS = sorted(tool.name for tool in default_registry())


@pytest.mark.parametrize("name", REAL_TOOLS)
def test_a_rule_naming_a_tool_that_exists_is_accepted(name):
    assert Rule.parse(name).tool == name
    assert Rule.parse(f"{name}(**/x)").tool == name


@pytest.mark.parametrize("name", ["Bash", "Git"])
def test_the_tools_that_are_being_built_are_accepted_as_reserved_names(name):
    assert name not in REAL_TOOLS, "built now: it no longer needs to be reserved"
    assert Rule.parse(name).tool == name
    assert Rule.parse(f"{name}(status)").tool == name
    assert Rule.parse("Bash(npm test:*)").prefix


def test_the_known_tools_are_the_registered_ones_and_the_reserved_ones():
    """A tool added to the registry and not to the list a rule may name would be a tool
    that no rule could ever govern."""
    assert set(KNOWN_TOOLS) == set(REAL_TOOLS) | {"Bash", "Git"}


@pytest.mark.parametrize(
    "rule",
    [
        "read(secrets/**)",  # tool names are case-sensitive
        "READ(secrets/**)",
        "Raed(secrets/**)",
        "read",
        "Reads",
        "Wrte(src/**)",
        "grep(x)",
        "mcp__server__tool",
        "(secrets/**)",
        "bash(npm test:*)",  # the prefix form
    ],
)
def test_a_rule_naming_no_tool_is_refused_and_the_message_lists_the_tools(rule):
    with pytest.raises(ValueError, match="unknown tool") as raised:
        Rule.parse(rule)
    message = str(raised.value)
    assert all(name in message for name in KNOWN_TOOLS), message
    assert "\n" not in message
    assert "case-sensitive" in message


def test_the_message_names_the_tool_as_it_was_written():
    with pytest.raises(ValueError, match=r"unknown tool 'Raed'"):
        Rule.parse("Raed(secrets/**)")


def test_a_ruleset_refuses_a_rule_naming_no_tool_whichever_list_it_is_in():
    for kind in ("allow", "ask", "deny"):
        with pytest.raises(ValueError, match="unknown tool 'read'"):
            RuleSet.build(**{kind: ["Write", "read(secrets/**)"]})


def test_spaces_around_a_tool_name_are_not_part_of_it():
    assert Rule.parse("  Read ( **/x )").tool == "Read"
