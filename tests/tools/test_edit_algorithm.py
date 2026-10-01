import pytest
from hypothesis import given
from hypothesis import strategies as st

from nanoclaude.tools.edit import EditError, EditSpec, apply_edits, detect_eol, unified_diff


def test_crlf_line_endings_survive_an_edit():
    """Review Focus #1. A naive read-modify-write normalises to LF and the
    resulting diff touches every line of the file."""
    content = "one\r\ntwo\r\nthree\r\n"
    result = apply_edits(content, [EditSpec("two", "TWO")])
    assert result == "one\r\nTWO\r\nthree\r\n"
    assert "\n" not in result.replace("\r\n", "")


def test_a_multi_line_old_string_matches_across_crlf():
    content = "def f():\r\n    return 1\r\n"
    result = apply_edits(content, [EditSpec("def f():\n    return 1", "def f():\n    return 2")])
    assert result == "def f():\r\n    return 2\r\n"


def test_detect_eol_picks_the_dominant_ending():
    assert detect_eol("a\r\nb\r\n") == "\r\n"
    assert detect_eol("a\nb\n") == "\n"
    assert detect_eol("a\r\nb\nc\r\n") == "\r\n"
    assert detect_eol("no newline at all") == "\n"


def test_a_non_unique_old_string_is_an_error_naming_the_count():
    with pytest.raises(EditError, match="appears 3 times"):
        apply_edits("x\nx\nx\n", [EditSpec("x", "y")])


def test_replace_all_replaces_every_occurrence():
    assert apply_edits("x\nx\nx\n", [EditSpec("x", "y", replace_all=True)]) == "y\ny\ny\n"


def test_a_missing_old_string_is_an_error():
    with pytest.raises(EditError, match="was not found"):
        apply_edits("hello\n", [EditSpec("goodbye", "hi")])


def test_edits_apply_in_order_each_to_the_previous_result():
    result = apply_edits("a\n", [EditSpec("a", "b"), EditSpec("b", "c")])
    assert result == "c\n"


def test_a_failing_later_edit_discards_the_whole_batch():
    """All or nothing. A half-applied rename leaves code that compiles as neither."""
    with pytest.raises(EditError):
        apply_edits("a\nb\n", [EditSpec("a", "A"), EditSpec("zzz", "Z")])
    # apply_edits is pure, so the caller still holds the original; the tool
    # test asserts the file on disk is untouched.


def test_curly_quotes_in_the_file_match_straight_quotes_from_the_model():
    content = "msg = “hello”\n"  # U+201C/U+201D curly double quotes
    result = apply_edits(content, [EditSpec('msg = "hello"', 'msg = "goodbye"')])
    assert result == 'msg = "goodbye"\n'


def test_straight_quotes_in_the_file_match_curly_ones_from_the_model():
    content = 'msg = "hello"\n'
    result = apply_edits(content, [EditSpec("msg = “hello”", "msg = 'hi'")])
    assert result == "msg = 'hi'\n"


@given(
    prefix=st.text(alphabet="abc\n", max_size=20),
    needle=st.text(alphabet="xyz", min_size=1, max_size=5),
    suffix=st.text(alphabet="abc\n", max_size=20),
)
def test_a_successful_single_edit_removes_exactly_one_occurrence(prefix, needle, suffix):
    content = prefix + needle + suffix
    if content.count(needle) != 1:
        return
    result = apply_edits(content, [EditSpec(needle, "REPLACED")])
    assert result.count("REPLACED") == 1
    assert len(result) == len(content) - len(needle) + len("REPLACED")


# The tests below are not in the brief. They close gaps the brief's own tests
# leave open: two refusals apply_edits can raise that no given test triggers,
# unified_diff (a "Produces" name in the brief with no direct test otherwise),
# and the hashability the task notes ask to be pinned explicitly.


def test_an_empty_edit_list_is_an_error():
    """apply_edits([]) is reachable independently of the tool layer, which
    never calls it with an empty list (_parse_edits refuses first) -- so only
    a direct test of the pure function reaches this line."""
    with pytest.raises(EditError, match="no edits were given"):
        apply_edits("hello\n", [])


def test_an_empty_old_string_is_an_error():
    with pytest.raises(EditError, match="old_string must not be empty"):
        apply_edits("hello\n", [EditSpec("", "x")])


def test_unified_diff_marks_removed_and_added_lines_with_the_path():
    diff = unified_diff("a\nb\nc\n", "a\nB\nc\n", "x.py")
    assert "--- a/x.py" in diff
    assert "+++ b/x.py" in diff
    assert "-b" in diff
    assert "+B" in diff


def test_editspec_is_hashable_because_it_holds_only_strings_and_a_bool():
    """EditSpec is frozen with only two str fields and a bool -- no list, dict
    or other unhashable field -- so dataclass's generated __hash__ never
    raises. Unlike ToolContext (tools/base.py) or ToolSpec (providers/base.py),
    which hold a mapping or other unhashable state and so declare __hash__ =
    None on purpose, EditSpec has nothing that would make hashing unsafe."""
    assert hash(EditSpec("a", "b")) == hash(EditSpec("a", "b"))
    assert len({EditSpec("a", "b"), EditSpec("a", "b", replace_all=True)}) == 2


def test_crlf_inside_an_edit_is_folded_to_an_lf_file():
    # The edit's own strings carry CRLF while the file is LF: the edit must
    # still match, and must not introduce CRLF into the file.
    content = "a\nb\nc\n"
    result = apply_edits(content, [EditSpec("a\r\nb", "x\r\ny")])
    assert result == "x\ny\nc\n"
