"""What becomes of a server's tools on the way into the registry.

A server's tool list is somebody else's text and somebody else's names, and the model and the
terminal are about to read both. These tests hold the adapter to what it promises: the name a
tool gets, what its description and schema are cut down to, what is dropped and why, and what a
fingerprint pin does when a definition is not what it was.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nanoclaude.config.schema import McpServerConfig
from nanoclaude.mcp.pins import MAX_PINS_PER_SERVER, PINS_FILENAME, PinStore, fingerprint
from nanoclaude.mcp.tool import (
    MAX_DESCRIPTION_BYTES,
    MAX_REGISTRY_NAME_CHARS,
    MAX_SCHEMA_BYTES,
    MAX_SCHEMA_DEPTH,
    TRUNCATED,
    AdaptedTools,
    McpTool,
    adapt_tools,
    clean_description,
    clean_schema,
    combine_tools,
    registry_name,
    schema_problem,
)
from nanoclaude.permissions.rules import MCP_TOOL_NAME
from nanoclaude.providers.base import ToolSpec
from nanoclaude.tools.base import Tool
from tests.mcp.stubs import OBJECT, StubClient, tool


def server(name: str = "github", timeout: int = 30) -> McpServerConfig:
    return McpServerConfig(name=name, command="x", timeout=timeout)


def nested(depth: int) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object"}
    for _ in range(depth - 1):
        schema = {"type": "object", "n": schema}
    return schema


async def adapt(
    tmp_path: Path,
    *tools: Any,
    name: str = "github",
    pins: PinStore | None = None,
    cap: int = 100_000,
) -> AdaptedTools:
    store = pins if pins is not None else PinStore.load(tmp_path / PINS_FILENAME)
    return await adapt_tools(StubClient(tools), server(name), store, output_cap=cap)


def names(adapted: AdaptedTools) -> list[str]:
    return [t.name for t in adapted.tools]


def test_the_limits_are_the_ones_the_design_gives():
    # The tests below are written in these names; this is the one place that says the numbers.
    assert MAX_DESCRIPTION_BYTES == 2048
    assert MAX_SCHEMA_BYTES == 64 * 1024
    assert MAX_SCHEMA_DEPTH == 32
    assert MAX_REGISTRY_NAME_CHARS == 64


# ---- names


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("create_issue", "create_issue"),
        ("Create_Issue2", "Create_Issue2"),
        ("create-issue", "create_issue"),
        ("a.b c/d", "a_b_c_d"),
        ("café", "caf_"),  # a character, not a byte: one underscore
        ("\u03b1\u03b2", "__"),
        ("x\ny\tz", "x_y_z"),
        ("a\x1b[2Jb", "a__2Jb"),
    ],
)
def test_a_registry_name_has_every_character_outside_ascii_letters_digits_and_underscore_replaced(
    original, expected
):
    assert registry_name("github", original) == f"mcp__github__{expected}"


async def test_a_tool_is_registered_as_mcp_server_tool_and_is_a_tool(tmp_path):
    adapted = await adapt(tmp_path, tool("create-issue"), tool("list"))
    assert names(adapted) == ["mcp__github__create_issue", "mcp__github__list"]
    assert adapted.warnings == ()
    for t in adapted.tools:
        assert isinstance(t, Tool)
        assert MCP_TOOL_NAME.fullmatch(t.name)


async def test_the_call_to_the_server_uses_the_name_the_server_gave_the_tool(tmp_path, ctx):
    client = StubClient([tool("create-issue")])
    adapted = await adapt_tools(
        client, server(), PinStore.load(tmp_path / PINS_FILENAME), output_cap=1000
    )
    await adapted.tools[0].run(ctx, "c1", {"title": "t"})
    assert [(name, arguments) for name, arguments, _ in client.calls] == [
        ("create-issue", {"title": "t"})
    ]


async def test_two_tools_of_one_server_with_one_registry_name_are_both_dropped_and_named(tmp_path):
    adapted = await adapt(tmp_path, tool("a-b"), tool("a.b"), tool("fine"))
    assert names(adapted) == ["mcp__github__fine"]
    (warning,) = adapted.warnings
    assert "'a-b'" in warning and "'a.b'" in warning, warning
    assert "mcp__github__a_b" in warning
    assert "github" in warning


async def test_a_tool_listed_twice_under_one_name_is_dropped_both_times(tmp_path):
    adapted = await adapt(tmp_path, tool("dup"), tool("dup"), tool("fine"))
    assert names(adapted) == ["mcp__github__fine"]
    assert len(adapted.warnings) == 1 and "dup" in adapted.warnings[0]


async def test_three_that_collide_are_all_dropped_in_one_warning(tmp_path):
    adapted = await adapt(tmp_path, tool("a-b"), tool("a.b"), tool("a b"))
    assert names(adapted) == []
    assert len(adapted.warnings) == 1
    assert all(f"'{n}'" in adapted.warnings[0] for n in ("a-b", "a.b", "a b"))


async def test_a_collision_drops_a_tool_that_would_have_been_dropped_anyway_too(tmp_path):
    # The names are what collide; whether a schema is good is a different question.
    adapted = await adapt(tmp_path, tool("a-b"), tool("a.b", schema={"type": "string"}))
    assert names(adapted) == []


@pytest.mark.parametrize(("length", "kept"), [(63, True), (64, True), (65, False)])
async def test_a_registry_name_over_64_characters_is_dropped_with_a_warning(tmp_path, length, kept):
    prefix = "mcp__github__"
    original = "t" * (length - len(prefix))
    adapted = await adapt(tmp_path, tool(original), tool("short"))
    assert (f"{prefix}{original}" in names(adapted)) is kept
    assert "mcp__github__short" in names(adapted)
    assert len(adapted.warnings) == (0 if kept else 1)
    if not kept:
        assert f"{length} characters" in adapted.warnings[0]
        assert str(MAX_REGISTRY_NAME_CHARS) in adapted.warnings[0]


async def test_a_long_server_name_makes_a_long_registry_name(tmp_path):
    name = "s" * 24
    adapted = await adapt(tmp_path, tool("t" * 34), tool("t" * 33), name=name)
    assert names(adapted) == [f"mcp__{name}__{'t' * 33}"]
    assert len(adapted.warnings) == 1


async def test_a_tool_with_no_name_to_make_a_registry_name_of_is_dropped(tmp_path):
    adapted = await adapt(tmp_path, tool(""), tool("fine"))
    assert names(adapted) == ["mcp__github__fine"]
    assert len(adapted.warnings) == 1 and "no name" in adapted.warnings[0]


async def test_a_warning_shows_a_servers_text_with_its_control_sequences_removed_and_cut(tmp_path):
    evil = "x\x1b[2J\x1b]0;owned\x07" + "n" * 500
    adapted = await adapt(tmp_path, tool(evil, schema=[1]))
    (warning,) = adapted.warnings
    assert "\x1b" not in warning and "\x07" not in warning
    assert len(warning) < 400, len(warning)
    assert "\n" not in warning


# ---- descriptions


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("plain", "plain"),
        ("keeps\ttabs\nand newlines", "keeps\ttabs\nand newlines"),
        ("\x1b[31mred\x1b[0m text", "red text"),
        ("\x1b[2J\x1b[Hcleared", "cleared"),
        ("\x1b]0;window title\x07done", "done"),
        ("bell\x07 back\x08 nul\x00 del\x7f cr\r", "bell back nul del cr"),
        ("c1\x9b2J\x9d0;T\x9c", "c12J0;T"),
        ("lone \ud800 surrogate", "lone � surrogate"),
    ],
)
def test_a_description_has_controls_and_ansi_removed_and_keeps_tab_and_newline(raw, cleaned):
    assert clean_description(raw) == cleaned


def test_a_description_of_exactly_the_limit_is_left_whole():
    text = "x" * MAX_DESCRIPTION_BYTES
    assert clean_description(text) == text


def test_a_description_one_byte_over_is_cut_and_marked():
    text = "x" * (MAX_DESCRIPTION_BYTES + 1)
    assert clean_description(text) == "x" * MAX_DESCRIPTION_BYTES + TRUNCATED


def test_the_marker_is_the_ellipsis_and_the_word_truncated():
    assert TRUNCATED == "…(truncated)"


@pytest.mark.parametrize("length", [1500, 1501, 1502, 1503])
def test_a_description_is_cut_on_a_character_boundary_whatever_the_alignment(length):
    # Three bytes a character, and 2,048 is not a multiple of three, so the limit falls inside
    # a character for most alignments: the character is dropped whole and none is torn.
    text = "a" * (length % 3) + "€" * length
    cleaned = clean_description(text)
    assert cleaned.endswith(TRUNCATED)
    kept = cleaned.removesuffix(TRUNCATED)
    assert "�" not in kept
    assert len(kept.encode("utf-8")) <= MAX_DESCRIPTION_BYTES
    assert len(kept.encode("utf-8")) > MAX_DESCRIPTION_BYTES - 3
    assert text.startswith(kept)


def test_controls_are_removed_before_the_cut_so_they_do_not_use_up_the_limit():
    text = "\x00" * 5000 + "tail"
    assert clean_description(text) == "tail"


async def test_the_description_the_model_is_given_is_the_cleaned_one(tmp_path):
    evil = "Reads notes.\x1b[2J\x1b]0;owned\x07\x00 tab\there\nline"
    (adapted,) = (await adapt(tmp_path, tool("evil", evil))).tools
    assert adapted.spec().description == "Reads notes. tab\there\nline"


async def test_a_missing_description_is_an_empty_one(tmp_path):
    (adapted,) = (await adapt(tmp_path, tool("t", None))).tools
    assert adapted.spec().description == ""


@pytest.mark.parametrize("description", [5, ["x"], {"a": 1}, True])
async def test_a_description_that_is_not_text_drops_the_tool_with_a_warning(tmp_path, description):
    adapted = await adapt(tmp_path, tool("t", description), tool("fine"))
    assert names(adapted) == ["mcp__github__fine"]
    assert "description" in adapted.warnings[0]


# ---- input schemas


def test_a_schema_that_is_an_object_of_type_object_has_no_problem():
    assert schema_problem({"type": "object", "properties": {"a": {"type": "string"}}}) is None
    assert schema_problem(OBJECT) is None


@pytest.mark.parametrize(
    "schema",
    [
        None,
        [],
        [1],
        "object",
        5,
        False,
        {},
        {"properties": {}},
        {"type": "string"},
        {"type": ["object", "null"]},
        {"type": None},
        {"type": "Object"},
    ],
)
def test_a_schema_that_is_not_an_object_of_type_object_is_a_problem(schema):
    assert schema_problem(schema) is not None


@pytest.mark.parametrize(("depth", "problem"), [(1, False), (31, False), (32, False), (33, True)])
def test_a_schema_deeper_than_32_levels_is_a_problem(depth, problem):
    found = schema_problem(nested(depth))
    assert (found is not None) is problem, (depth, found)
    if found is not None:
        assert str(MAX_SCHEMA_DEPTH) in found


def test_depth_counts_lists_too():
    schema: Any = {"type": "object", "anyOf": []}
    node = schema["anyOf"]
    for _ in range(MAX_SCHEMA_DEPTH):
        inner: list[Any] = []
        node.append(inner)
        node = inner
    assert schema_problem(schema) is not None


def test_a_schema_nested_far_deeper_than_python_would_recurse_is_a_problem_not_a_crash():
    deep = nested(5000)
    assert schema_problem(deep) is not None


def test_the_size_limit_is_the_serialized_length_in_bytes():
    base = len(json.dumps({"type": "object", "description": ""}, separators=(",", ":")))
    exactly = {"type": "object", "description": "x" * (MAX_SCHEMA_BYTES - base)}
    assert len(json.dumps(exactly, separators=(",", ":")).encode()) == MAX_SCHEMA_BYTES
    assert schema_problem(exactly) is None
    over = {"type": "object", "description": "x" * (MAX_SCHEMA_BYTES - base + 1)}
    problem = schema_problem(over)
    assert problem is not None and str(MAX_SCHEMA_BYTES) in problem


def test_the_size_limit_counts_bytes_and_not_characters():
    text = "€" * (MAX_SCHEMA_BYTES // 3)  # 3 bytes each: the characters alone are 65,535
    assert schema_problem({"type": "object", "description": text}) is not None


@pytest.mark.parametrize(
    "ref",
    [
        "https://example.com/s.json",
        "http://x/y",
        "file:///etc/passwd",
        "other.json",
        "",
        " #/x",
        "../up.json",
        "//host/x",
    ],
)
def test_a_ref_that_does_not_start_with_a_hash_is_a_problem_wherever_it_is(ref):
    nested_in_a_list = {"type": "object", "anyOf": [{"allOf": [{"$ref": ref}]}]}
    in_a_property = {"type": "object", "properties": {"a": {"$ref": ref}}}
    top_level = {"type": "object", "$ref": ref}
    for schema in (nested_in_a_list, in_a_property, top_level):
        problem = schema_problem(schema)
        assert problem is not None and "$ref" in problem, (ref, schema)


@pytest.mark.parametrize("ref", ["#", "#/$defs/x", "#/properties/a", "#anchor"])
def test_a_ref_that_starts_with_a_hash_is_not_a_problem(ref):
    assert schema_problem({"type": "object", "properties": {"a": {"$ref": ref}}}) is None


@pytest.mark.parametrize("ref", [5, None, ["#/x"], {"x": 1}])
def test_a_ref_that_is_not_text_is_a_problem_too(ref):
    assert schema_problem({"type": "object", "properties": {"a": {"$ref": ref}}}) is not None


def test_text_that_merely_mentions_a_ref_is_not_a_ref():
    schema = {"type": "object", "description": 'see "$ref": "http://x" for more'}
    assert schema_problem(schema) is None


async def test_a_tool_that_fails_the_schema_check_is_dropped_with_a_warning_naming_the_reason(
    tmp_path,
):
    adapted = await adapt(
        tmp_path,
        tool("good"),
        tool("not_object", schema={"type": "string"}),
        tool("no_schema", schema=None),
        tool("deep", schema=nested(33)),
        tool("remote", schema={"type": "object", "properties": {"a": {"$ref": "http://x/y"}}}),
        tool("big", schema={"type": "object", "description": "x" * 70_000}),
    )
    assert names(adapted) == ["mcp__github__good"]
    assert len(adapted.warnings) == 5
    joined = "\n".join(adapted.warnings)
    for tool_name in ("not_object", "no_schema", "deep", "remote", "big"):
        assert f"'{tool_name}'" in joined, joined
    assert "$ref" in joined and "more than 32 levels" in joined and "70" in joined


DIRTY = "a\x1b[2Jb\x00c\x07d\x9be"
CLEAN = "abcde"


def test_every_string_value_in_the_cleaned_schema_is_cleaned_not_only_the_descriptions():
    # The model reads all of them: an enum's values, a default, the examples, a pattern, a
    # comment, an extension's text.
    schema: dict[str, Any] = {
        "type": "object",
        "description": DIRTY,
        "title": DIRTY,
        "$comment": DIRTY,
        "x-hint": DIRTY,
        "properties": {
            "a": {
                "type": "string",
                "enum": [DIRTY, "plain", 3],
                "default": DIRTY,
                "const": DIRTY,
                "examples": [DIRTY, [DIRTY]],
                "pattern": DIRTY,
                "format": DIRTY,
                "x-nested": {"x-deep": [{"x-text": DIRTY}]},
            }
        },
        "required": ["a"],
    }
    cleaned = clean_schema(schema)
    assert cleaned["description"] == cleaned["title"] == cleaned["$comment"] == CLEAN
    assert cleaned["x-hint"] == CLEAN
    prop = cleaned["properties"]["a"]
    assert prop["enum"] == [CLEAN, "plain", 3]
    assert prop["default"] == prop["const"] == prop["pattern"] == prop["format"] == CLEAN
    assert prop["examples"] == [CLEAN, [CLEAN]]
    assert prop["x-nested"] == {"x-deep": [{"x-text": CLEAN}]}
    assert cleaned["required"] == ["a"]
    assert schema["properties"]["a"]["default"] == DIRTY, "the input is not changed"


def test_tabs_and_newlines_in_a_schemas_text_are_kept():
    schema: dict[str, Any] = {"type": "object", "description": "one\ntwo\tthree"}
    assert clean_schema(schema) == schema


def test_the_cleaned_schema_has_no_lone_surrogate_anywhere_so_a_request_can_be_written():
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"k": {"enum": ["v\udfff"], "default": "d\ud800", "x-a": ["\ud800"]}},
    }
    json.dumps(clean_schema(schema), ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize(
    "key",
    ["a\x00b", "a\x1b[2Jb", "a\x07", "a\x9bb", "a\x7fb", "a\rb", "a\ud800b", "\x1b]0;t\x07x"],
    ids=repr,
)
def test_a_key_with_a_control_character_in_it_is_a_schema_problem_wherever_it_is(key):
    # Renaming a property would make the arguments the model writes not the ones the server
    # reads, so such a schema is refused, not cleaned.
    top: dict[str, Any] = {"type": "object", key: 1}
    in_a_property = {"type": "object", "properties": {key: {"type": "string"}}}
    deep = {"type": "object", "anyOf": [{"properties": {"a": {"properties": {key: {}}}}}]}
    in_required = {"type": "object", "properties": {"a": {"type": "string", key: {"x": 1}}}}
    for schema in (top, in_a_property, deep, in_required):
        problem = schema_problem(schema)
        assert problem is not None and "key" in problem, (key, schema)
        assert "\x1b" not in problem and "\x00" not in problem and "\x07" not in problem


def test_a_key_with_a_tab_or_a_newline_in_it_is_not_a_problem_because_those_are_kept():
    assert schema_problem({"type": "object", "properties": {"a\tb": {}, "c\nd": {}}}) is None


@pytest.mark.parametrize("name", ["$dynamicRef", "$recursiveRef"])
@pytest.mark.parametrize(
    "ref", ["https://example.com/s.json", "file:///etc/passwd", "other", "", 5]
)
def test_the_other_references_are_refused_like_ref_when_they_leave_the_schema(name, ref):
    schema = {"type": "object", "properties": {"a": {name: ref}}}
    problem = schema_problem(schema)
    assert problem is not None and name in problem, (name, ref)


@pytest.mark.parametrize("name", ["$dynamicRef", "$recursiveRef"])
def test_the_other_references_are_allowed_when_they_stay_inside_the_schema(name):
    assert schema_problem({"type": "object", "properties": {"a": {name: "#meta"}}}) is None
    assert schema_problem({"type": "object", "properties": {"a": {name: "#"}}}) is None


async def test_a_tool_with_a_control_character_in_a_property_name_is_dropped_with_a_warning(
    tmp_path,
):
    adapted = await adapt(
        tmp_path,
        tool("good"),
        tool("bad", schema={"type": "object", "properties": {"a\x1b[2Jb": {"type": "string"}}}),
        tool("dyn", schema={"type": "object", "properties": {"a": {"$dynamicRef": "http://x/y"}}}),
    )
    assert names(adapted) == ["mcp__github__good"]
    assert len(adapted.warnings) == 2
    assert "'bad'" in adapted.warnings[0] and "key" in adapted.warnings[0]
    assert "'dyn'" in adapted.warnings[1] and "$dynamicRef" in adapted.warnings[1]
    assert all("\x1b" not in w for w in adapted.warnings)


def test_numbers_booleans_and_nulls_in_a_schema_are_left_as_they_are():
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 0, "default": None}},
        "additionalProperties": False,
    }
    assert clean_schema(schema) == schema


def test_a_property_called_description_is_not_taken_for_a_description():
    schema = {"type": "object", "properties": {"description": {"type": "string"}}}
    assert clean_schema(schema) == schema


async def test_the_schema_the_model_is_given_is_the_cleaned_one(tmp_path):
    schema = {"type": "object", "properties": {"a": {"description": "x\x1b[2Jy"}}}
    (adapted,) = (await adapt(tmp_path, tool("t", schema=schema))).tools
    assert adapted.spec().schema["properties"]["a"]["description"] == "xy"


# ---- the tool itself


async def test_a_tool_is_not_read_only_and_its_spec_is_its_name_description_and_schema(tmp_path):
    schema = {"type": "object", "properties": {"text": {"type": "string"}}}
    (adapted,) = (await adapt(tmp_path, tool("echo", "Says it back.", schema))).tools
    assert adapted.read_only is False
    assert adapted.spec() == ToolSpec("mcp__github__echo", "Says it back.", schema)


async def test_the_permission_request_names_the_server_has_no_path_and_is_a_write(tmp_path, ctx):
    (adapted,) = (await adapt(tmp_path, tool("echo"))).tools
    request = adapted.permission_request(ctx, {"text": "hi"})
    assert request.tool == "mcp__github__echo"
    assert request.subject == "server github"
    assert request.resolved_paths == ()
    assert request.is_write is True
    assert request.definition_changed is False


async def test_building_the_tools_reads_no_file_and_the_tool_does_not_touch_the_client(tmp_path):
    client = StubClient([tool("echo")])
    store = PinStore(tmp_path / ".nanoclaude" / PINS_FILENAME)
    built = McpTool(
        client,
        server=server(),
        remote_name="echo",
        description="d",
        schema=OBJECT,
        pins=store,
        output_cap=1000,
    )
    assert built.name == "mcp__github__echo"
    assert client.listed == 0 and client.calls == []
    assert not (tmp_path / ".nanoclaude").exists()


async def test_the_only_calls_adapting_makes_are_listing_the_tools(tmp_path):
    client = StubClient([tool("echo")])
    await adapt_tools(client, server(), PinStore.load(tmp_path / PINS_FILENAME), output_cap=1000)
    assert client.listed == 1 and client.calls == []


# ---- fingerprint pins


def pinned(tmp_path: Path) -> dict[str, Any]:
    found: dict[str, Any] = json.loads((tmp_path / PINS_FILENAME).read_text(encoding="utf-8"))
    return found


async def test_a_tool_seen_for_the_first_time_is_recorded_and_is_not_marked_changed(tmp_path):
    adapted = await adapt(tmp_path, tool("echo", "Says it back.", OBJECT))
    assert pinned(tmp_path) == {"github": {"echo": fingerprint("Says it back.", OBJECT)}}
    assert adapted.warnings == ()
    assert adapted.tools[0].changed is False


async def test_the_pin_is_made_of_the_description_and_schema_the_model_is_given(tmp_path):
    raw = "Says\x1b[2J it"
    await adapt(tmp_path, tool("echo", raw, OBJECT))
    assert pinned(tmp_path)["github"]["echo"] == fingerprint("Says it", OBJECT)


async def test_the_pin_is_keyed_by_the_registry_name_that_rules_and_grants_use(tmp_path):
    # Under the server, the tool as the registry names it: mcp__github__create_issue.
    await adapt(tmp_path, tool("create-issue"))
    assert list(pinned(tmp_path)["github"]) == ["create_issue"]


async def test_a_definition_that_changes_under_another_spelling_of_the_name_is_changed(tmp_path):
    # An allow rule or a grant for mcp__github__create_issue covers both spellings, so the
    # pin has to as well: a definition changed between them cannot be a new tool.
    await adapt(tmp_path, tool("create-issue", "creates an issue"))
    again = await adapt(tmp_path, tool("create_issue", "creates an issue and mails the team"))
    assert [t.name for t in again.tools] == ["mcp__github__create_issue"]
    assert again.tools[0].changed is True


async def test_the_same_definition_under_another_spelling_of_the_name_is_not_changed(tmp_path):
    await adapt(tmp_path, tool("create-issue", "creates an issue"))
    again = await adapt(tmp_path, tool("create.issue", "creates an issue"))
    assert again.tools[0].changed is False


async def test_a_renewed_pin_is_kept_under_the_registry_name_too(tmp_path):
    await adapt(tmp_path, tool("create-issue", "one"))
    (changed,) = (await adapt(tmp_path, tool("create.issue", "two"))).tools
    changed.on_grant()
    assert list(pinned(tmp_path)["github"]) == ["create_issue"]
    assert pinned(tmp_path)["github"]["create_issue"] == fingerprint("two", OBJECT)


async def test_a_tool_whose_definition_is_what_it_was_is_not_changed_in_the_next_session(tmp_path):
    await adapt(tmp_path, tool("echo", "Says it back.", OBJECT))
    before = (tmp_path / PINS_FILENAME).read_bytes()
    again = await adapt(tmp_path, tool("echo", "Says it back.", OBJECT))
    assert again.tools[0].changed is False
    assert (tmp_path / PINS_FILENAME).read_bytes() == before


async def test_a_tool_whose_description_differs_from_its_pin_is_changed_and_the_pin_stays(tmp_path):
    await adapt(tmp_path, tool("echo", "Says it back.", OBJECT))
    before = (tmp_path / PINS_FILENAME).read_bytes()
    again = await adapt(tmp_path, tool("echo", "Says it back. Also mails it.", OBJECT))
    assert again.tools[0].changed is True
    assert (tmp_path / PINS_FILENAME).read_bytes() == before, "listing a tool does not renew a pin"


async def test_a_tool_whose_schema_differs_from_its_pin_is_changed(tmp_path):
    await adapt(tmp_path, tool("echo", "d", OBJECT))
    wider = {"type": "object", "properties": {"to": {"type": "string"}}}
    again = await adapt(tmp_path, tool("echo", "d", wider))
    assert again.tools[0].changed is True


async def test_a_change_that_cleaning_removes_is_not_a_change(tmp_path):
    await adapt(tmp_path, tool("echo", "Says it back."))
    again = await adapt(tmp_path, tool("echo", "Says\x1b[2J it\x00 back."))
    assert again.tools[0].changed is False


async def test_a_changed_tool_stays_changed_and_so_does_the_request_for_the_whole_session(
    tmp_path, ctx
):
    await adapt(tmp_path, tool("echo", "one"))
    (changed,) = (await adapt(tmp_path, tool("echo", "two"))).tools
    for _ in range(3):
        assert changed.permission_request(ctx, {}).definition_changed is True


async def test_only_the_tool_that_changed_is_marked_and_the_others_are_not(tmp_path):
    await adapt(tmp_path, tool("a", "one"), tool("b", "one"))
    again = await adapt(tmp_path, tool("a", "two"), tool("b", "one"), tool("c", "new"))
    assert {t.name: t.changed for t in again.tools} == {
        "mcp__github__a": True,
        "mcp__github__b": False,
        "mcp__github__c": False,
    }


async def test_a_pin_is_per_server_so_another_servers_tool_of_the_same_name_is_new(tmp_path):
    store = PinStore.load(tmp_path / PINS_FILENAME)
    await adapt(tmp_path, tool("echo", "one"), name="github", pins=store)
    other = await adapt(tmp_path, tool("echo", "two"), name="fs", pins=store)
    assert other.tools[0].changed is False
    assert set(pinned(tmp_path)) == {"github", "fs"}


async def test_a_tool_that_went_away_and_came_back_different_is_changed(tmp_path):
    await adapt(tmp_path, tool("echo", "one"), tool("other", "x"))
    await adapt(tmp_path, tool("other", "x"))
    back = await adapt(tmp_path, tool("echo", "two"))
    assert back.tools[0].changed is True


async def test_a_dropped_tool_is_not_pinned(tmp_path):
    await adapt(tmp_path, tool("good"), tool("bad", schema={"type": "string"}))
    assert list(pinned(tmp_path)["github"]) == ["good"]


async def test_answering_always_for_a_changed_tool_renews_its_pin_for_the_next_session(
    tmp_path, ctx
):
    await adapt(tmp_path, tool("echo", "one"))
    (changed,) = (await adapt(tmp_path, tool("echo", "two"))).tools
    changed.on_grant()
    assert pinned(tmp_path)["github"]["echo"] == fingerprint("two", OBJECT)
    # This session still treats it as changed: the person was asked about it once, not told that
    # it stopped being a risk.
    assert changed.permission_request(ctx, {}).definition_changed is True
    (next_session,) = (await adapt(tmp_path, tool("echo", "two"))).tools
    assert next_session.changed is False


async def test_a_grant_for_a_tool_that_has_not_changed_writes_nothing(tmp_path):
    (echo,) = (await adapt(tmp_path, tool("echo", "one"))).tools
    (tmp_path / PINS_FILENAME).unlink()
    echo.on_grant()
    assert not (tmp_path / PINS_FILENAME).exists()


async def test_a_pin_that_cannot_be_renewed_is_not_an_error_for_the_call_that_was_approved(
    tmp_path, monkeypatch
):
    await adapt(tmp_path, tool("echo", "one"))
    (changed,) = (await adapt(tmp_path, tool("echo", "two"))).tools

    def refuse(self: Path, target: Path) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "replace", refuse)
    changed.on_grant()  # does not raise
    assert pinned(tmp_path)["github"]["echo"] == fingerprint("one", OBJECT)


async def test_pins_that_cannot_be_saved_are_a_warning_and_the_tools_are_still_returned(
    tmp_path, monkeypatch
):
    def refuse(self: Path, target: Path) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "replace", refuse)
    adapted = await adapt(tmp_path, tool("echo"))
    assert names(adapted) == ["mcp__github__echo"]
    (warning,) = adapted.warnings
    assert "read-only file system" in warning and "github" in warning
    assert str(tmp_path / PINS_FILENAME) in warning


# ---- more than one server


async def test_tools_of_two_servers_are_combined_with_their_warnings(tmp_path):
    store = PinStore.load(tmp_path / PINS_FILENAME)
    first = await adapt(tmp_path, tool("a"), tool("x y", schema=[1]), name="one", pins=store)
    second = await adapt(tmp_path, tool("b"), name="two", pins=store)
    combined = combine_tools([first, second])
    assert names(combined) == ["mcp__one__a", "mcp__two__b"]
    assert combined.warnings == first.warnings + second.warnings


async def test_two_servers_whose_tools_come_to_one_registry_name_lose_both_tools(tmp_path):
    # Server "a" with the tool "b__t", and server "a__b" with the tool "t": both are
    # mcp__a__b__t, which the registry would refuse to hold.
    store = PinStore.load(tmp_path / PINS_FILENAME)
    first = await adapt(tmp_path, tool("b__t"), tool("keep1"), name="a", pins=store)
    second = await adapt(tmp_path, tool("t"), tool("keep2"), name="a__b", pins=store)
    combined = combine_tools([first, second])
    assert names(combined) == ["mcp__a__keep1", "mcp__a__b__keep2"]
    (warning,) = combined.warnings
    assert "mcp__a__b__t" in warning and "'a'" in warning and "'a__b'" in warning


def test_combining_nothing_is_nothing():
    assert combine_tools([]) == AdaptedTools((), ())


async def test_tools_past_the_pin_limit_are_registered_unpinned_and_the_warning_says_how_many(
    tmp_path,
):
    store = PinStore(
        tmp_path / PINS_FILENAME,
        {"github": {f"old_{n}": "a" * 64 for n in range(MAX_PINS_PER_SERVER)}},
    )
    adapted = await adapt(tmp_path, tool("new_a"), tool("new_b"), tool("new_c"), pins=store)
    assert names(adapted) == ["mcp__github__new_a", "mcp__github__new_b", "mcp__github__new_c"]
    assert not any(t.changed for t in adapted.tools)
    (warning,) = adapted.warnings
    assert "3 of its tools" in warning and str(MAX_PINS_PER_SERVER) in warning
    assert "github" in warning and str(tmp_path / PINS_FILENAME) in warning
    assert store.get("github", "new_a") is None
    assert not (tmp_path / PINS_FILENAME).exists(), "nothing was new enough to write"


async def test_a_tool_that_is_pinned_already_is_still_checked_when_the_server_is_at_the_limit(
    tmp_path,
):
    pins = {f"old_{n}": "a" * 64 for n in range(MAX_PINS_PER_SERVER - 1)}
    pins["known"] = fingerprint("one", OBJECT)
    store = PinStore(tmp_path / PINS_FILENAME, {"github": pins})
    adapted = await adapt(tmp_path, tool("known", "two"), pins=store)
    assert adapted.tools[0].changed is True
    assert adapted.warnings == ()
