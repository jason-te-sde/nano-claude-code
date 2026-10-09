import pytest

from nanoclaude.permissions.rules import (
    KNOWN_TOOLS,
    Rule,
    RuleKind,
    RuleSet,
    fold_spelling,
    glob_matches_any,
)
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


# ---- refusals compare what the filesystem compares: case and Unicode form

CAFE_NFC = "caf\u00e9"
CAFE_NFD = "cafe\u0301"


def test_fold_spelling_makes_case_and_unicode_form_equal():
    assert fold_spelling("SECRETS/X") == fold_spelling("secrets/x")
    assert fold_spelling(CAFE_NFC) == fold_spelling(CAFE_NFD)
    assert fold_spelling(CAFE_NFC.upper()) == fold_spelling(CAFE_NFD)
    # casefold, not lower: the German sharp s is "ss" to a filesystem that folds.
    assert fold_spelling("STRA\u00dfE") == fold_spelling("strasse")


def test_a_character_class_still_holds_one_character_after_the_name_is_case_folded():
    """Case folding can split a letter into a base and a mark (the small j with a caron
    becomes two), so the folded form is composed again: a class that held one character
    would otherwise hold two, and match neither spelling."""
    assert glob_matches_any(("x[\u01f0]y",), ("x\u01f0y",), fold=True)
    assert glob_matches_any(("x[\u01f0]y",), ("xj\u030cy",), fold=True)


def test_fold_spelling_leaves_different_names_different():
    assert fold_spelling("secrets") != fold_spelling("secret")
    assert fold_spelling("a/b") != fold_spelling("ab")
    assert fold_spelling(CAFE_NFC) != fold_spelling("cafe")


def test_glob_matches_any_folds_only_when_asked():
    assert not glob_matches_any(("secrets/**",), ("SECRETS/x",))
    assert glob_matches_any(("secrets/**",), ("SECRETS/x",), fold=True)
    assert not glob_matches_any((CAFE_NFC,), (CAFE_NFD,))
    assert glob_matches_any((CAFE_NFC,), (CAFE_NFD,), fold=True)


def test_a_character_class_written_in_the_composed_form_matches_the_decomposed_name():
    """Folding composes the pattern as well as the candidate: a class holding the one
    composed character would otherwise hold two, and match neither."""
    assert glob_matches_any((f"caf[{CAFE_NFC[-1]}]/**",), (f"{CAFE_NFD}/x",), fold=True)


def test_a_folded_match_is_still_a_match_of_the_pattern_and_not_of_anything_near_it():
    assert not glob_matches_any(("secrets/**",), ("secret/x",), fold=True)
    assert not glob_matches_any(("secrets/**",), ("SECRETSX/x",), fold=True)
    assert not glob_matches_any(("**/.env",), ("/p/.ENVRC",), fold=True)


def test_a_deny_rule_matches_after_folding_in_all_three_of_its_forms():
    glob, exact, prefix = (
        Rule.parse("Read(secrets/**)"),
        Rule.parse("Bash(Make Clean)"),
        Rule.parse("Bash(rm:*)"),
    )
    assert glob.matches("Read", "/p/SECRETS/x", "SECRETS/x", fold=True)
    assert exact.matches("Bash", "make clean", "make clean", fold=True)
    assert prefix.matches("Bash", "RM -rf build", "RM -rf build", fold=True)
    assert not prefix.matches("Bash", "RMDIR x", "RMDIR x", fold=True)  # still a word boundary


def test_a_deny_rule_with_glob_characters_in_its_exact_form_matches_the_folded_command():
    """A rule's subject is compared as it is written before it is tried as a glob, and the
    command ``ls [A]`` is that subject, folded: ``[a]`` there is text, not a class."""
    assert Rule.parse("Bash(LS [a])").matches("Bash", "ls [A]", "ls [A]", fold=True)


def test_a_prefix_deny_rule_in_capitals_matches_the_lowercase_command():
    assert Rule.parse("Bash(RM:*)").matches("Bash", "rm -rf build", "rm -rf build", fold=True)
    assert not Rule.parse("Bash(RM:*)").matches("Bash", "rm -rf build", "rm -rf build")


def test_a_rule_compares_exactly_unless_told_to_fold():
    glob, exact, prefix = (
        Rule.parse("Read(secrets/**)"),
        Rule.parse("Bash(Make Clean)"),
        Rule.parse("Bash(rm:*)"),
    )
    assert not glob.matches("Read", "/p/SECRETS/x", "SECRETS/x")
    assert not exact.matches("Bash", "make clean", "make clean")
    assert not prefix.matches("Bash", "RM -rf build", "RM -rf build")


def test_a_tool_name_is_not_folded_even_for_a_refusal():
    assert not Rule.parse("Read(secrets/**)").matches(
        "read", "/p/secrets/x", "secrets/x", fold=True
    )


def test_only_the_deny_list_of_a_ruleset_folds():
    rules = RuleSet.build(allow=["Read(docs/**)"], ask=["Read(docs/**)"], deny=["Read(docs/**)"])
    cases: list[tuple[RuleKind, bool]] = [("deny", True), ("allow", False), ("ask", False)]
    for kind, expected in cases:
        found = rules.first_match(kind, "Read", "/p/DOCS/x", "DOCS/x") is not None
        assert found is expected, kind
