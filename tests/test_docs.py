"""The documentation makes claims. These check the cheap ones.

What a test here cannot check is whether a sentence is true, only that what the sentence
names exists: a tool, a command, a key, a file. That is still most of how documentation
goes wrong, since it is code that moves and the prose that stays.
"""

import base64
import hashlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import tomllib
from dataclasses import fields, replace
from pathlib import Path

import pytest

from nanoclaude.agent.loop import StopReason
from nanoclaude.cli.commands import COMMANDS
from nanoclaude.cli.headless import DECLINED_PREFIX, declined_line
from nanoclaude.cli.init import PRESETS
from nanoclaude.cli.main import EXIT_CODES, build_parser, main
from nanoclaude.cli.repl import TEXT_TOOL_BANNER
from nanoclaude.config.load import CONFIG_DIRNAME, CONFIG_FILENAME, ConfigError, load_config
from nanoclaude.config.schema import ROLES, LimitsConfig, ModelConfig, PermissionsConfig, UiConfig
from nanoclaude.context.git import ALLOWED_SUBCOMMAND
from nanoclaude.context.instructions import (
    MAX_INSTRUCTION_BYTES,
    MAX_INSTRUCTION_TOTAL_BYTES,
    load_instructions,
)
from nanoclaude.context.projectmap import project_map
from nanoclaude.conversation.store import Store
from nanoclaude.permissions import redact
from nanoclaude.permissions.audit import AuditLog
from nanoclaude.permissions.danger.regex import RegexClassifier
from nanoclaude.permissions.policy import (
    PROTECTED_DIRECTORIES,
    Decision,
    PermissionMode,
    PermissionRequest,
    PermissionResult,
    Policy,
    evaluate,
)
from nanoclaude.permissions.redact import SECRET_PATH_PATTERNS, Redactor
from nanoclaude.permissions.rules import KNOWN_TOOLS, RuleSet, glob_matches_any
from nanoclaude.permissions.sandbox import Sandbox
from nanoclaude.private import create_private_file, make_private_directories
from nanoclaude.providers.capabilities import CACHE_FILENAME, CONSERVATIVE_DEFAULT, capabilities_for
from nanoclaude.providers.pricing import PriceBook
from nanoclaude.tools.edit import EditTool
from nanoclaude.tools.fs import stamp_of
from nanoclaude.tools.glob import GlobTool
from nanoclaude.tools.grep import GrepTool
from nanoclaude.tools.read import ReadTool
from nanoclaude.tools.registry import default_registry
from nanoclaude.tools.todo import TodoState
from tests.conftest import mode_of
from tests.keys import pem
from tests.script_modules import load_script

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "nanoclaude"
DOCS = ROOT / "docs"
DESIGN = DOCS / "design"
README = ROOT / "README.md"
SECURITY = ROOT / "SECURITY.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
CHANGELOG = ROOT / "CHANGELOG.md"
CONFIGURATION = DOCS / "configuration.md"
ARCHITECTURE = DOCS / "architecture.md"
TESTING = DOCS / "testing.md"

#: The documents about the program as it is, as opposed to the design notes, which argue for
#: decisions and may name things that are not built. Checks that a document names only what
#: exists run over these.
DOCUMENTS = (README, CONFIGURATION, ARCHITECTURE, TESTING, SECURITY, CONTRIBUTING, CHANGELOG)

#: The design notes that exist. The ones the shell and the syntax-tree classifier need are
#: written with them, so they are not in this list yet: 0008, 0009 and 0011 are left for
#: when Bash, Git and that classifier are built. The numbering has gaps until then.
DESIGN_NOTES = (
    "0001-scope.md",
    "0002-pure-loop.md",
    "0003-capability-negotiation.md",
    "0004-text-tool-protocol.md",
    "0005-role-model-routing.md",
    "0006-ui-protocol.md",
    "0007-edit-by-string-not-lineno.md",
    "0010-two-phase-executor.md",
    "0012-redaction-before-transcript.md",
    "0013-thin-context.md",
    "0014-openai-compat-first-class.md",
    "0015-bypass-does-not-disable-the-sandbox.md",
    "0016-attribution-policy.md",
)

REQUIRED_SECTIONS = ("## Why", "## Costs", "## Rejected alternatives")


def _sections(text: str) -> dict[str, str]:
    """The body under each ``## `` heading of a note, by heading."""
    found: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line.strip()
            found[current] = []
        elif current is not None:
            found[current].append(line)
    return {heading: "\n".join(body).strip() for heading, body in found.items()}


def _notes() -> list[Path]:
    """Every design note there is. Never an empty list: a check that loops over none passes."""
    notes = sorted(DESIGN.glob("*.md"))
    assert len(notes) >= len(DESIGN_NOTES), (
        f"found {len(notes)} design notes where {len(DESIGN_NOTES)} are due, so a check that "
        "read them all would have read too few"
    )
    return notes


def test_the_design_notes_that_are_due_all_exist():
    present = sorted(note.name for note in DESIGN.glob("*.md"))
    missing = [name for name in DESIGN_NOTES if name not in present]
    assert missing == [], f"design notes that should exist but do not: {missing}; found {present}"


def test_every_design_note_has_the_three_required_sections_and_no_others():
    for note in _notes():
        headings = [
            line.strip() for line in note.read_text().splitlines() if line.startswith("## ")
        ]
        assert headings == list(REQUIRED_SECTIONS), (
            f"{note.name} has the sections {headings}, and must have exactly {REQUIRED_SECTIONS}"
        )


def test_no_section_of_a_design_note_is_a_heading_with_nothing_under_it():
    for note in _notes():
        for heading, body in _sections(note.read_text()).items():
            assert len(body.split()) >= 25, (
                f"{note.name}: {heading} has {len(body.split())} words, too few to say anything"
            )


def test_every_rejected_alternative_is_a_list_item_and_there_are_at_least_two():
    for note in _notes():
        body = _sections(note.read_text())["## Rejected alternatives"]
        items = [line for line in body.splitlines() if re.match(r"^(- |\d+\. )", line)]
        assert len(items) >= 2, (
            f"{note.name} rejects {len(items)} alternatives; a decision with fewer was not one"
        )


def test_a_design_note_is_named_for_its_number_and_titled_with_it():
    for note in _notes():
        number = note.name[:4]
        assert number.isdigit(), f"{note.name} does not begin with its number"
        first = note.read_text().splitlines()[0]
        assert first.startswith(f"# {number}: "), f"{note.name} is titled {first!r}"


def test_the_scope_note_lists_what_v01_does_not_do():
    text = (DESIGN / "0001-scope.md").read_text().lower()
    for absent in ("mcp", "sub-agent", "checkpoint", "hook", "windows"):
        assert absent in text, f"0001-scope.md does not say that v0.1 lacks {absent}"


# --------------------------------------------------------------------------
# What a document names must exist, and what exists must be named
# --------------------------------------------------------------------------

#: A slash command in backticks: ``/help``, or ``/export [file]``. A path such as
#: ``/api/show`` has more after its first word and is not one.
_SLASH_COMMAND = re.compile(r"`/([a-z]+)(?=[`\s\[<])")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The body under a ``## `` heading, up to the next one."""
    found = re.search(
        rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL
    )
    assert found is not None, f"no section called {heading!r}"
    return found.group(1)


def _table_rows(text: str) -> list[list[str]]:
    """The cells of every row of every markdown table in ``text``, without the header rule."""
    rows = []
    for line in text.splitlines():
        if line.startswith("|") and not re.fullmatch(r"\|[ :|-]+\|", line):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def test_every_slash_command_a_document_names_exists():
    for document in DOCUMENTS:
        for name in _SLASH_COMMAND.findall(_text(document)):
            assert name in COMMANDS, f"{document.name} names /{name}, which is not a command"


def test_every_slash_command_is_in_the_configuration_reference():
    text = _text(CONFIGURATION)
    for name in COMMANDS:
        assert f"`/{name}" in text, f"/{name} is a command and the reference does not describe it"


def test_every_configuration_key_is_in_the_reference():
    text = _text(CONFIGURATION)

    def first_column(heading: str) -> set[str]:
        rows = _table_rows(_section(text, heading))
        return {row[0].strip("`") for row in rows if row[0].startswith("`")}

    for heading, names in (
        ("Models", [f.name for f in fields(ModelConfig)]),
        ("Limits", [f.name for f in fields(LimitsConfig)]),
    ):
        missing = [name for name in names if name not in first_column(heading)]
        assert missing == [], f"the {heading} table does not have a row for {missing}"
    for heading, names in (
        ("Roles", list(ROLES)),
        ("Permissions", [f.name for f in fields(PermissionsConfig)]),
        ("The `[ui]` section", [f.name for f in fields(UiConfig)]),
    ):
        body = _section(text, heading)
        missing = [name for name in names if f"`{name}`" not in body]
        assert missing == [], f"the section on {heading} does not name {missing}"


def test_the_limits_in_the_reference_have_the_defaults_the_code_has():
    rows = _table_rows(_section(_text(CONFIGURATION), "Limits"))
    documented = {row[0].strip("`"): row[1] for row in rows if row[0].startswith("`")}
    defaults = LimitsConfig()
    for field in fields(LimitsConfig):
        assert field.name in documented, f"the limits table has no row for {field.name}"
        said = float(documented[field.name].replace(",", ""))
        assert said == float(getattr(defaults, field.name)), (
            f"the reference gives {field.name} as {documented[field.name]}; "
            f"the code's default is {getattr(defaults, field.name)}"
        )


def test_the_permission_defaults_in_the_reference_are_the_code_s():
    text = _text(CONFIGURATION)
    defaults = PermissionsConfig()
    for key in ("allow", "ask"):
        listed = "[" + ", ".join(f'"{rule}"' for rule in getattr(defaults, key)) + "]"
        assert f"{key} = {listed}" in text, (
            f"the reference does not give the default {key} list {listed}"
        )


def test_the_exit_codes_in_the_reference_are_the_ones_the_program_returns():
    rows = _table_rows(
        _section(_text(CONFIGURATION), "The command line").split("**Exit codes.**")[1]
    )
    documented = {int(row[0].strip("`")) for row in rows if re.fullmatch(r"`\d+`", row[0])}
    assert documented == set(EXIT_CODES.values()), (
        f"the reference documents the exit codes {sorted(documented)}; "
        f"the program returns {sorted(set(EXIT_CODES.values()))}"
    )


def test_the_environment_variables_are_the_ones_the_program_reads():
    rows = _table_rows(_section(_text(CONFIGURATION), "Environment variables"))
    source = "\n".join(path.read_text(encoding="utf-8") for path in SRC.rglob("*.py"))
    documented = set(re.findall(r"`([A-Z][A-Z0-9_]{3,})`", " ".join(row[0] for row in rows)))
    assert documented, "no environment variable was found in the reference's table"
    for name in documented:
        assert name in source, f"the reference documents {name} and nothing reads it"
    for name in set(re.findall(r"NANOCLAUDE_[A-Z_]+", source)):
        assert name in documented, f"the program reads {name} and the reference does not say so"


def test_the_files_the_reference_says_ncc_keeps_are_the_ones_it_keeps():
    text = _text(CONFIGURATION)
    main = _text(SRC / "cli" / "main.py")
    for name in (CONFIG_FILENAME, CACHE_FILENAME):
        assert f"~/{CONFIG_DIRNAME}/{name}" in text, f"the reference does not say where {name} is"
    for name in ("sessions.db", "history"):
        assert f'"{name}"' in main, f"the program no longer keeps {name} where it was"
        assert f"~/{CONFIG_DIRNAME}/{name}" in text, f"the reference does not say where {name} is"


def test_the_presets_in_the_reference_are_the_ones_init_offers():
    rows = _table_rows(_section(_text(CONFIGURATION), "ncc init"))
    documented = {(row[2].strip("`"), row[3].strip("`")) for row in rows if row[0].isdigit()}
    offered = {(preset.default_model, preset.key_env or "none") for preset in PRESETS.values()}
    assert documented == offered, f"the reference lists {documented}; init offers {offered}"


#: The ids of hosted models that the documents may use: the ones the capability table has a
#: row for, which were checked against each provider's own list. A legacy id is not among them.
CURRENT_ANTHROPIC = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5", "claude-fable-5-1"}


def test_every_hosted_model_id_in_the_documents_is_a_current_one_the_program_knows():
    for document in DOCUMENTS:
        text = _text(document)
        for model in set(re.findall(r"\bclaude-(?:sonnet|opus|haiku|fable)-\d+(?:-\d+)*\b", text)):
            assert model in CURRENT_ANTHROPIC, f"{document.name} uses {model}, which is not current"
            assert capabilities_for("anthropic", model) is not CONSERVATIVE_DEFAULT
        for model in set(re.findall(r"\bdeepseek/[a-z0-9.-]+", text)):
            assert model == PRESETS["openrouter"].default_model, (
                f"{document.name} uses {model}; the model checked against OpenRouter's list is "
                f"{PRESETS['openrouter'].default_model}"
            )


# --------------------------------------------------------------------------
# The examples are run
# --------------------------------------------------------------------------

#: A home configuration that defines every alias the examples refer to, so that an excerpt, or
#: a project file that only chooses among models, has something to be laid over.
EXAMPLE_HOME = """\
[models.sonnet]
adapter = "anthropic"
model   = "claude-sonnet-5-5"

[models.opus]
adapter = "anthropic"
model   = "claude-opus-5-5"

[models.haiku]
adapter = "anthropic"
model   = "claude-haiku-5-5"

[models.local]
adapter = "ollama"
model   = "qwen3-coder"

[models.cheap]
adapter     = "openai_compat"
base_url    = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
model       = "deepseek/deepseek-v4-pro"
"""

EXAMPLE_KINDS = {
    "# ~/.nanoclaude/config.toml": "home",
    "# ~/.nanoclaude/config.toml (excerpt)": "excerpt",
    "# .nanoclaude/config.toml (in the project)": "project",
}


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Every fenced block of a document, as (the language after the fence, its lines)."""
    blocks: list[tuple[str, str]] = []
    language: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("```"):
            if language is None:
                language, lines = line[3:].strip(), []
            else:
                blocks.append((language, "\n".join(lines)))
                language = None
        elif language is not None:
            lines.append(line)
    return blocks


def _toml_examples() -> list[tuple[str, str, str]]:
    """(document, kind, text) for each TOML block in the documents."""
    found = []
    for document in DOCUMENTS:
        for language, body in _fenced_blocks(_text(document)):
            if language == "toml":
                found.append((document.name, body.splitlines()[0], body))
    return found


def test_there_are_toml_examples_to_check_and_each_kind_is_among_them():
    kinds = {EXAMPLE_KINDS.get(first) for _, first, _ in _toml_examples()}
    assert kinds == {"home", "excerpt", "project"}, f"found the kinds {kinds}"


@pytest.mark.parametrize(("document", "first", "body"), _toml_examples())
def test_every_toml_example_says_whose_file_it_is_and_loads(document, first, body, tmp_path):
    assert first in EXAMPLE_KINDS, (
        f"a TOML example in {document} must begin with a comment saying whose file it is, "
        f"one of {sorted(EXAMPLE_KINDS)}; it begins {first!r}"
    )
    kind = EXAMPLE_KINDS[first]
    tomllib.loads(body)  # a syntax error is named as one, before the loader gets to it
    home, project = tmp_path / "home", tmp_path / "project"
    (home / CONFIG_DIRNAME).mkdir(parents=True)
    (project / CONFIG_DIRNAME).mkdir(parents=True)
    home_file = home / CONFIG_DIRNAME / CONFIG_FILENAME
    if kind == "home":
        home_file.write_text(body)
        load_config(home=str(home), project=None, env={})
    elif kind == "excerpt":
        home_file.write_text(EXAMPLE_HOME + "\n" + body)
        load_config(home=str(home), project=None, env={})
    else:
        home_file.write_text(EXAMPLE_HOME)
        (project / CONFIG_DIRNAME / CONFIG_FILENAME).write_text(body)
        load_config(home=str(home), project=str(project), env={})


def _ncc_commands() -> list[str]:
    return [
        command
        for document in DOCUMENTS
        for command in re.findall(r"^\s*(ncc [^\n]+)$", _text(document), re.MULTILINE)
    ]


def test_every_command_in_the_documents_runs(tmp_path):
    """Documented commands are executed, not written from memory."""
    executed = 0
    for command in _ncc_commands():
        if "-p " in command or "init" in command:
            continue  # these need a key, or a terminal
        arguments = shlex.split(command)[1:]
        result = subprocess.run(  # noqa: S603 - this interpreter, arguments from our own documents
            [sys.executable, "-m", "nanoclaude.cli.main", *arguments],
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "NANOCLAUDE_HOME": str(tmp_path)},
            timeout=60,
        )
        executed += 1
        assert result.returncode in (0, 2), (
            f"{command!r} exited {result.returncode}: {result.stderr}"
        )
        if arguments in (["--version"], ["--help"]):
            assert result.returncode == 0, f"{command!r} exited {result.returncode}"
    assert executed >= 1, "no documented command was run, so none was checked"


# --------------------------------------------------------------------------
# Links
# --------------------------------------------------------------------------

_LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)")


def _anchors(text: str) -> set[str]:
    """The anchors GitHub makes for the headings of a document."""
    found = set()
    for heading in re.findall(r"^#{1,6} +(.+?) *$", text, re.MULTILINE):
        slug = re.sub(r"[^a-z0-9 _-]", "", heading.replace("`", "").lower())
        found.add(slug.replace(" ", "-"))
    return found


def test_the_anchor_of_a_heading_is_made_as_github_makes_it():
    assert _anchors("## What v0.1 does not do\n## `ncc init` and more\n") == {
        "what-v01-does-not-do",
        "ncc-init-and-more",
    }


def _every_document() -> list[Path]:
    top = [ROOT / name for name in ("README.md", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md")]
    return [path for path in (*top, *sorted(DOCS.rglob("*.md"))) if path.exists()]


def test_every_relative_link_in_the_documents_resolves():
    checked = 0
    for document in _every_document():
        for target in _LINK.findall(_text(document)):
            if re.match(r"[a-z]+:", target):
                continue  # a web address, a mail address
            path, _, anchor = target.partition("#")
            resolved = (document.parent / path).resolve() if path else document
            assert resolved.exists(), (
                f"{document.relative_to(ROOT)} links to {target}, which is not there"
            )
            if anchor and resolved.suffix == ".md":
                assert anchor in _anchors(_text(resolved)), (
                    f"{document.relative_to(ROOT)} links to {target}, which has no such heading"
                )
            checked += 1
    assert checked >= 5, f"only {checked} links were found, so the documents were barely checked"


# --------------------------------------------------------------------------
# The architecture and testing documents
# --------------------------------------------------------------------------


def test_the_rule_ids_in_the_architecture_document_are_the_ones_the_code_decides_with():
    ids = set()
    for name in ("permissions/policy.py", "permissions/sandbox.py", "agent/executor.py"):
        ids |= set(
            re.findall(
                r'"((?:rule|secret|sandbox|bash|mode|grant|tool|default)\.[a-z-]+)"',
                _text(SRC / name),
            )
        )
    rows = _table_rows(_text(ARCHITECTURE))
    documented = {row[0].strip("`") for row in rows if re.fullmatch(r"`[a-z]+\.[a-z-]+`", row[0])}
    assert len(ids) >= 17, f"found only {sorted(ids)} in the source, so the pattern is wrong"
    assert documented == ids, (
        f"in the document and not the code: {sorted(documented - ids)}; "
        f"in the code and not the document: {sorted(ids - documented)}"
    )


def test_the_architecture_document_states_the_three_extensions_of_the_plan():
    text = _text(ARCHITECTURE)
    assert "**Three extensions of the plan.**" in text
    assert "tool.internal-error" in text
    assert "sandbox.protected-path" in text
    assert "messages_archive(session_id, seq, role, blocks_json, created_at, archived_at)" in text


def test_the_testing_document_says_what_the_tests_do_not_cover():
    section = _section(_text(TESTING), "What these tests do not cover")
    leads = [
        match.group(1).lower() for match in re.finditer(r"^- \*\*(.+?)\*\*", section, re.MULTILINE)
    ]
    assert len(leads) >= 6, f"only {len(leads)} things are listed as not covered"
    for topic in (
        "provider",
        "windows",
        "concurrent",
        "hostile",
        "models",
        "danger classifier",
    ):
        assert any(topic in lead for lead in leads), (
            f"no item of the section is about {topic!r}; the items are {leads}"
        )
    assert "cassette" in section.lower(), "the section does not say what the cassettes are"


# --------------------------------------------------------------------------
# What the documents say is absent stays absent until they are corrected
# --------------------------------------------------------------------------
#
# A document that says "nothing reads this yet" is true on the day it is written and false
# the day somebody builds the reader. These fail on that day, and the failure message says
# which document to correct.


def _source() -> dict[str, str]:
    return {
        str(path.relative_to(SRC)): path.read_text(encoding="utf-8") for path in SRC.rglob("*.py")
    }


def test_nothing_reads_the_shell_limits_yet():
    readers = {
        name
        for name, text in _source().items()
        if re.search(r"\b(?:bash_timeout_s|output_cap_bytes)\b", text)
    }
    assert readers == {"config/schema.py", "config/load.py", "cli/init.py"}, (
        f"{sorted(readers)} read bash_timeout_s or output_cap_bytes; "
        "docs/configuration.md says that nothing does yet"
    )


def test_only_the_main_and_compact_roles_are_sent_requests():
    used = set()
    for text in _source().values():
        used |= set(re.findall(r'router\.(?:client_for|capabilities_for)\(\s*"(\w+)"', text))
        used |= set(re.findall(r'self\._complete\(\s*"(\w+)"', text))
    assert used == {"main", "compact"}, (
        f"requests are sent to the roles {sorted(used)}; docs/configuration.md and "
        "docs/design/0005-role-model-routing.md say that only main and compact are"
    )


def test_nothing_acts_on_the_capability_fields_the_notes_say_are_not_consulted():
    for name, text in _source().items():
        if name == "providers/capabilities.py":
            continue
        found = re.findall(r"\.(?:parallel_tools|vision|reasoning)\b|capabilities\.cache\b", text)
        assert found == [], (
            f"{name} reads {found}; docs/design/0003-capability-negotiation.md says that "
            "parallel_tools, cache, reasoning and vision are not consulted"
        )


def test_nothing_reads_the_ui_settings():
    readers = {
        name
        for name, text in _source().items()
        if re.search(r"UiConfig|diff_style|config\.ui\b|\.theme\b", text)
    }
    assert readers == {"config/schema.py", "config/load.py"}, (
        f"{sorted(readers)} read the [ui] settings; docs/configuration.md says that nothing does"
    )


def test_ncc_has_no_audit_command_and_the_scope_note_says_so(capsys):
    assert main(["audit"]) == EXIT_CODES["usage"]
    capsys.readouterr()
    assert "`ncc audit`" in _text(DESIGN / "0001-scope.md")


def test_the_regex_classifier_marks_its_clear_verdicts_as_unreliable():
    """A safe from a list of patterns means "nothing refused this", and says so."""
    verdict = RegexClassifier().classify("ls -la")
    assert verdict.authoritative is False, (
        "the regex classifier's safe verdicts are marked authoritative; "
        "docs/architecture.md says they are not"
    )
    assert "marked not authoritative" in _text(ARCHITECTURE)


async def test_glob_lists_the_names_of_files_the_project_map_leaves_out(tmp_repo, ctx):
    """A known gap: the map asks the policy what it would refuse a Read of, and Glob does not."""
    for name in (
        ".env",
        "id_rsa",
        ".ssh/config",
        ".gnupg/pubring.kbx",
        "docs/private.md",
        "app.py",
    ):
        file = tmp_repo / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("x\n")
    policy = replace(
        ctx.policy, rules=RuleSet.build(allow=["Read", "Glob"], deny=["Read(docs/private.md)"])
    )
    refused = (".env", "id_rsa", ".ssh/config", ".gnupg/pubring.kbx", "docs/private.md")

    shown = {line.strip() for line in project_map(str(tmp_repo), policy=policy).splitlines()}
    assert {"app.py", ".ssh/", ".gnupg/", "docs/"} <= shown, (
        f"the map shows {sorted(shown)}: a directory whose contents are refused is listed by its "
        "name, which is what SECURITY.md and docs/design/0013-thin-context.md say"
    )
    assert shown.isdisjoint({Path(name).name for name in refused}), (
        f"the map shows {sorted(shown)}, and names a file that the policy would refuse a Read of"
    )

    listing = await GlobTool().run(replace(ctx, policy=policy), "t1", {"pattern": "**/*"})
    listed = set(listing.content.splitlines()[1:])
    missing = [name for name in refused if name not in listed]
    assert missing == [], (
        f"Glob no longer lists {missing}; correct SECURITY.md, docs/design/0013-thin-context.md, "
        "docs/testing.md and CHANGELOG.md, which say that it asks the policy nothing"
    )
    for document in (SECURITY, DESIGN / "0013-thin-context.md", TESTING, CHANGELOG):
        assert "`Glob`" in _text(document), f"{document.name} does not say what Glob lists"


def test_the_text_tool_banner_the_documents_quote_is_the_one_the_repl_prints():
    assert TEXT_TOOL_BANNER in _text(DESIGN / "0004-text-tool-protocol.md")
    assert TEXT_TOOL_BANNER in _text(CONFIGURATION)


def test_every_test_a_document_names_exists():
    """A document that says a test pins something has to name a test that is there."""
    defined = set()
    for path in (ROOT / "tests").rglob("*.py"):
        defined |= set(re.findall(r"^\s*(?:async )?def (test_\w+)", _text(path), re.MULTILINE))
    named = 0
    for document in _every_document():
        for name in set(re.findall(r"`(test_\w+)`", _text(document))):
            named += 1
            assert name in defined, (
                f"{document.relative_to(ROOT)} names {name}, which is not a test"
            )
    assert named >= 40, f"only {named} tests are named in the documents, so few were checked"


# --------------------------------------------------------------------------
# The README
# --------------------------------------------------------------------------

README_SECTIONS = [
    "Install",
    "Why this and not Claude Code",
    "Roles",
    "Safety",
    "Configuration",
    "What v0.1 does not do",
    "Numbers",
]


def test_the_readme_has_the_sections_it_is_meant_to_have_and_no_others():
    headings = re.findall(r"^## (.+?)\s*$", _text(README), re.MULTILINE)
    assert headings == README_SECTIONS


def test_the_readme_opens_with_the_one_line_that_says_what_this_is():
    assert (
        "A terminal coding agent that works with Anthropic, OpenAI-compatible and local models."
        in _text(README)
    )


def test_every_tool_in_the_readme_exists_and_every_tool_is_in_the_readme():
    registry = {tool.name for tool in default_registry(TodoState())}
    assert registry, "the registry is empty"
    readme = _text(README)
    for name in registry:
        assert name in readme, f"{name} is not mentioned in the README"
    rows = _table_rows(_section(readme, "Safety"))
    listed = {row[0].strip("`") for row in rows if re.fullmatch(r"`[A-Z][A-Za-z]+`", row[0])}
    assert listed == registry, (
        f"the README's tool table has {sorted(listed)}; the registry has {sorted(registry)}"
    )


def test_the_readme_says_which_tools_are_not_built_and_names_no_other_as_built():
    readme = _text(README)
    assert "Not built yet: `Bash` and `Git`" in readme
    registry = {tool.name for tool in default_registry(TodoState())}
    assert "Bash" not in registry
    assert "Git" not in registry


def test_every_slash_command_in_the_readme_exists():
    readme = _text(README)
    for name in _SLASH_COMMAND.findall(readme):
        assert name in COMMANDS, f"/{name} is documented but not implemented"


def test_the_readme_quotes_no_unmeasured_benchmark():
    """Numbers appear here only when scripts/measure.py produced them."""
    readme = _text(README).lower()
    for claim in ("faster than", "% success", "outperforms", "success rate", "benchmark"):
        assert claim not in readme, f"the README says {claim!r}"


def test_the_readme_says_init_writes_the_configuration_and_then_checks_the_key():
    readme = _text(README)
    assert "before saving" not in readme, "init does not check the key before it saves"
    assert "writes the file first and checks the key afterwards" in readme


def test_the_readme_says_why_a_project_cannot_grant_itself_permissions():
    sentence = (
        "a cloned repository must not be able to grant itself permissions or redirect your API key"
    )
    assert sentence.lower() in _text(README).lower()
    assert sentence.lower() in _text(CONFIGURATION).lower()


def test_the_readme_repeats_the_list_in_the_scope_note_word_for_word():
    note = _text(DESIGN / "0001-scope.md").split("## Why")[0]
    in_note = [line for line in note.splitlines() if line.startswith("- ")]
    section = _section(_text(README), "What v0.1 does not do")
    in_readme = [line for line in section.splitlines() if line.startswith("- ")]
    assert len(in_note) >= 20, "the scope note lists too few items to be the whole list"
    assert in_readme == in_note


def test_the_security_document_lists_an_untrusted_project_configuration_and_what_it_may_set():
    text = _text(SECURITY)
    assert "A project's configuration file" in text
    for setting in (
        "allow",
        "base_url",
        "api_key_env",
        "deny",
        "ask",
        "max_turns",
        "bash_timeout_s",
    ):
        assert f"`{setting}`" in text, f"SECURITY.md does not mention {setting}"


# --------------------------------------------------------------------------
# The policy, the audit and git
# --------------------------------------------------------------------------

NUMBER_WORDS = {
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21,
    "twenty-two": 22,
}

#: The rule ids decided before the policy runs. The others are what ``evaluate`` decides with.
_DECIDED_BEFORE_THE_POLICY = {"tool.unknown", "tool.bad-arguments", "tool.internal-error"}

#: An item of the reference's numbered list begins with the id or ids of its check.
_LEADING_IDS = re.compile(r"`[a-z]+\.[a-z-]+`(?: and `[a-z]+\.[a-z-]+`)*")


def _rule_ids_in_the_table() -> list[str]:
    rows = _table_rows(_text(ARCHITECTURE))
    return [row[0].strip("`") for row in rows if re.fullmatch(r"`[a-z]+\.[a-z-]+`", row[0])]


def _rule_ids_in_the_order_of_the_checks() -> list[str]:
    """The ids of the numbered list in the reference, in the order it gives them."""
    section = _section(_text(CONFIGURATION), "Permissions")
    ids: list[str] = []
    for item in re.findall(r"^\d+\. (.*)$", section, re.MULTILINE):
        leading = _LEADING_IDS.match(item)
        if leading is not None:
            ids += re.findall(r"`([a-z]+\.[a-z-]+)`", leading.group(0))
    return ids


def _policy_rule_ids_in_the_order_the_source_names_them() -> list[str]:
    source = _text(SRC / "permissions" / "policy.py")
    found = re.findall(r'"((?:rule|secret|sandbox|bash|mode|grant|tool|default)\.[a-z-]+)"', source)
    return list(dict.fromkeys(found))


def test_the_architecture_document_counts_its_rule_ids():
    ids = _rule_ids_in_the_table()
    found = re.search(r"These ([a-z-]+) exist", _text(ARCHITECTURE))
    assert found is not None, "the architecture document does not say how many rule ids exist"
    assert NUMBER_WORDS.get(found.group(1)) == len(ids), (
        f"the architecture document says that {found.group(1)} rule ids exist; "
        f"its table has {len(ids)}"
    )


def test_the_architecture_table_lists_the_rule_ids_in_the_order_the_policy_decides_them():
    # The sandbox's first row is decided in sandbox.py, so policy.py does not name it.
    in_the_table = [
        i
        for i in _rule_ids_in_the_table()
        if i not in _DECIDED_BEFORE_THE_POLICY and i != "sandbox.outside-root"
    ]
    in_the_source = _policy_rule_ids_in_the_order_the_source_names_them()
    assert len(in_the_source) >= 14, f"found only {in_the_source}, so the pattern is wrong"
    assert in_the_table == in_the_source, (
        f"the table gives the order {in_the_table}; "
        f"policy.py names them in the order {in_the_source}"
    )


def test_the_reference_lists_the_checks_in_the_order_of_the_architecture_table():
    in_the_reference = _rule_ids_in_the_order_of_the_checks()
    in_the_table = [i for i in _rule_ids_in_the_table() if i not in _DECIDED_BEFORE_THE_POLICY]
    assert len(in_the_reference) >= 15, f"found only {in_the_reference} in the reference's list"
    assert in_the_reference == in_the_table, (
        f"the reference gives the order {in_the_reference}; the table gives {in_the_table}"
    )


def test_the_documents_count_the_hard_denials_as_the_checks_before_bypass():
    section = _section(_text(CONFIGURATION), "Permissions")
    numbered = re.findall(r"^(\d+)\. `([a-z]+\.[a-z-]+)`", section, re.MULTILINE)
    position = next(int(number) for number, rule in numbered if rule == "mode.bypass")
    words = {6: "six", 7: "seventh"}
    assert position == 7, (
        f"mode.bypass is check {position} of the reference's list, not the seventh"
    )
    assert f"The first {words[position - 1]} are the hard denials" in _text(CONFIGURATION)
    note = _text(DESIGN / "0015-bypass-does-not-disable-the-sandbox.md")
    assert f"Bypass is the {words[position]}, and the {words[position - 1]} before it" in note


def _decide(policy: Policy, tool: str, relative: str, *, write: bool) -> PermissionResult:
    path = f"{policy.sandbox.roots[0]}/{relative}"
    return evaluate(PermissionRequest(tool, path, (path,), is_write=write), policy)


@pytest.mark.parametrize("mode", list(PermissionMode))
def test_a_write_into_git_or_nanoclaude_is_refused_in_every_mode(tmp_path, mode):
    policy = Policy(
        sandbox=Sandbox((str(tmp_path),)),
        rules=RuleSet.build(allow=["Read", "Write"]),
        mode=mode,
        secret_paths=SECRET_PATH_PATTERNS,
    )
    for relative in (
        ".git/config",
        "sub/.GIT/hooks/pre-commit",
        ".git",
        ".nanoclaude/config.toml",
        "a/.nanoclaude/sessions.db",
    ):
        refused = _decide(policy, "Write", relative, write=True)
        assert (refused.decision, refused.rule) == (Decision.DENY, "sandbox.protected-path"), (
            mode,
            relative,
            refused,
        )
    for relative in (".github/workflows/ci.yml", ".gitignore", "src/gitlab.py"):
        assert _decide(policy, "Write", relative, write=True).rule != "sandbox.protected-path"
    assert _decide(policy, "Read", ".git/config", write=False).rule != "sandbox.protected-path"


def test_the_documents_name_the_protected_directories_and_the_rule_that_refuses_them():
    assert PROTECTED_DIRECTORIES == (".git", ".nanoclaude"), (
        f"the protected directories are {PROTECTED_DIRECTORIES}; "
        "correct the documents that name them"
    )
    for document in (ARCHITECTURE, CONFIGURATION):
        assert "`sandbox.protected-path`" in _text(document), document.name
    for document in (ARCHITECTURE, CONFIGURATION, SECURITY, README):
        for name in PROTECTED_DIRECTORIES:
            assert f"`{name}`" in _text(document), f"{document.name} does not name {name}"


def test_the_audit_record_is_as_the_documents_describe_it(tmp_path):
    written = set(
        re.findall(r'"(ok|error|refused|declined|cancelled)"', _text(SRC / "agent" / "executor.py"))
    )
    paragraph = next(
        part
        for part in _text(ARCHITECTURE).split("\n\n")
        if part.startswith("**The audit record**")
    )
    sessions = _section(_text(CONFIGURATION), "Sessions")
    for where, text in (("the architecture document", paragraph), ("the reference", sessions)):
        documented = set(re.findall(r"`(ok|error|refused|declined|cancelled)`", text))
        assert documented == written, (
            f"{where} lists the outcomes {sorted(documented)}; "
            f"the executor writes {sorted(written)}"
        )

    triggers = _text(SRC / "conversation" / "store.py").count("CREATE TRIGGER")
    assert triggers == 3, f"the store has {triggers} triggers; the architecture document says three"
    assert "three triggers" in paragraph

    store = Store(tmp_path / "sessions.db")
    store.open()
    store.create_session("s1", cwd="/p", roles={})
    log = AuditLog(store)
    log.record_decision(
        "s1",
        turn=0,
        tool_use_id="t1",
        tool="Write",
        arguments={"path": "a"},
        decision=Decision.ASK,
        rule="default.ask",
    )
    refused = (
        "DELETE FROM tool_calls",
        (
            "INSERT OR REPLACE INTO tool_calls (session_id, turn, tool_use_id, ts, tool, "
            "args_json, decision, rule) VALUES ('s1', 0, 't1', 0, 'Write', '{}', 'allow', 'x')"
        ),
        "UPDATE tool_calls SET decision = 'allow'",
    )
    for statement in refused:
        with pytest.raises(sqlite3.DatabaseError):
            store.db.execute(statement)
    log.record_outcome("s1", "t1", outcome="declined", duration_ms=0, bytes_out=0, error=None)
    log.record_outcome("s1", "t1", outcome="ok", duration_ms=1, bytes_out=1, error=None)
    assert store.db.execute("SELECT outcome FROM tool_calls").fetchone()["outcome"] == "declined"
    with pytest.raises(sqlite3.DatabaseError):
        store.db.execute("UPDATE tool_calls SET outcome = 'ok'")


def test_git_is_run_only_as_rev_parse_and_the_documents_say_so():
    assert ALLOWED_SUBCOMMAND == "rev-parse"
    for document in (ARCHITECTURE, SECURITY, README):
        assert "git rev-parse" in _text(document), f"{document.name} does not say that"


def test_a_default_root_that_is_too_wide_is_refused_before_anything_is_opened(
    tmp_path, monkeypatch, capsys
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("NANOCLAUDE_HOME", str(home))
    monkeypatch.setenv("HOME", str(home))
    for wide in ("/", str(home), str(tmp_path)):
        monkeypatch.chdir(wide)
        assert main(["-p", "hello"]) == EXIT_CODES["usage"], wide
        error = capsys.readouterr().err
        assert error.startswith("error: ") and "--root" in error, (wide, error)
    assert not (home / ".nanoclaude").exists(), "something was opened before the root was refused"
    text = _text(CONFIGURATION)
    assert "`/`, a home directory" in text and "before it opens anything" in text


# --------------------------------------------------------------------------
# The reference
# --------------------------------------------------------------------------


def _credentials_list_in_the_reference() -> list[str]:
    section = _section(_text(CONFIGURATION), "Permissions")
    blocks = [body for language, body in _fenced_blocks(section) if language == ""]
    assert len(blocks) == 1, (
        f"found {len(blocks)} unlabelled code blocks in the Permissions section"
    )
    return blocks[0].splitlines()


def test_the_credentials_list_in_the_reference_is_the_one_the_policy_refuses():
    documented = _credentials_list_in_the_reference()
    assert all(pattern.startswith("**/") for pattern in SECRET_PATH_PATTERNS), (
        "a credentials pattern that is not in any directory; the reference says that they are"
    )
    refused = [pattern.removeprefix("**/") for pattern in SECRET_PATH_PATTERNS]
    assert sorted(documented) == sorted(refused), (
        f"in the reference and not the policy: {sorted(set(documented) - set(refused))}; "
        f"in the policy and not the reference: {sorted(set(refused) - set(documented))}"
    )
    for public in ("certs/server.crt", "keys/id_ed25519.pub"):
        assert not glob_matches_any(SECRET_PATH_PATTERNS, (public,), fold=True), public


def test_a_deny_rule_folds_case_and_an_allow_rule_does_not(tmp_path):
    policy = Policy(
        sandbox=Sandbox((str(tmp_path),)),
        rules=RuleSet.build(allow=["Write(Docs/**)"], deny=["Read(secrets/**)"]),
        mode=PermissionMode.DEFAULT,
        secret_paths=SECRET_PATH_PATTERNS,
    )
    assert _decide(policy, "Read", "Secrets/key.txt", write=False).rule == "rule.deny"
    assert _decide(policy, "Read", ".ENV", write=False).rule == "secret.path"
    assert _decide(policy, "Write", "Docs/a.md", write=True).rule == "rule.allow"
    assert _decide(policy, "Write", "docs/a.md", write=True).rule == "default.ask"
    for document in (CONFIGURATION, ARCHITECTURE):
        assert "case folding" in _text(document), document.name
    assert "`Read(secrets/**)`" in _text(CONFIGURATION)
    assert "`Secrets/key.txt`" in _text(CONFIGURATION)


def test_a_rule_naming_a_tool_that_does_not_exist_is_refused_and_two_names_are_reserved(tmp_path):
    registry = {tool.name for tool in default_registry(TodoState())}
    assert set(KNOWN_TOOLS) - registry == {"Bash", "Git"}, (
        "the names a rule may use that no tool has are not Bash and Git; correct "
        "docs/configuration.md and SECURITY.md, which say that those two are reserved"
    )
    home = tmp_path / "home"
    (home / CONFIG_DIRNAME).mkdir(parents=True)
    config = home / CONFIG_DIRNAME / CONFIG_FILENAME
    config.write_text(EXAMPLE_HOME + '\n[permissions]\ndeny = ["read(.env)"]\n')
    with pytest.raises(ConfigError, match="unknown tool 'read'"):
        load_config(home=str(home), project=None, env={})
    config.write_text(EXAMPLE_HOME + '\n[permissions]\nallow = ["Bash(npm test:*)", "Git"]\n')
    load_config(home=str(home), project=None, env={})
    for document in (CONFIGURATION, SECURITY):
        assert "reserved" in _text(document), document.name


def _flag_choices(flag: str) -> tuple[str, ...]:
    for action in build_parser()._actions:
        if flag in action.option_strings:
            return tuple(str(choice) for choice in action.choices or ())
    raise AssertionError(f"ncc has no flag {flag}")


def test_the_flag_choices_in_the_reference_are_the_parsers():
    text = _section(_text(CONFIGURATION), "The command line")
    for flag in ("--mode", "--output-format"):
        found = re.search(rf"`{flag} ([^`]+)`", text)
        assert found is not None, f"the reference does not give the choices of {flag}"
        assert tuple(found.group(1).split("\\|")) == _flag_choices(flag), flag
    assert "bypass" not in _flag_choices("--mode")


def _json_paragraph() -> str:
    return next(part for part in _text(CONFIGURATION).split("\n\n") if part.startswith("**JSON.**"))


def test_the_json_object_has_the_keys_and_stop_reasons_the_reference_names(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("NANOCLAUDE_HOME", str(tmp_path))
    assert main(["--output-format", "json"]) == EXIT_CODES["usage"]  # a failure is one object too
    payload = json.loads(capsys.readouterr().out)
    paragraph = _json_paragraph()
    for key in (*payload, *payload["usage"]):
        assert f"`{key}`" in paragraph, f"the reference does not name {key}"
    assert payload["stop_reason"] == "error" and payload["exit_code"] == EXIT_CODES["usage"]
    sentence = re.search(r"`stop_reason` is one of (.*?)\.", paragraph)
    assert sentence is not None, "the reference does not list the stop reasons"
    main_source = _text(SRC / "cli" / "main.py")
    assert '"cut_off"' in main_source and '"error"' in main_source
    emitted = {reason.value for reason in StopReason} | {"cut_off", "error"}
    assert set(re.findall(r"`([a-z_]+)`", sentence.group(1))) == emitted


def test_the_declined_line_the_reference_shows_is_the_one_ncc_prints():
    request = PermissionRequest("Write", "/p/src/x.py", ("/p/src/x.py",), is_write=True)
    line = declined_line(
        "Write", request, rule="default.ask", root="/p", mode=PermissionMode.DEFAULT
    )
    assert f"`{DECLINED_PREFIX}{line}`" in _text(CONFIGURATION)


def test_the_prices_the_reference_gives_are_the_price_books():
    book = PriceBook.load()
    haiku = book.price_for("anthropic", "claude-haiku-5-5")
    assert haiku is not None and haiku.threshold_tokens is not None
    assert haiku.above_threshold is not None
    tiered = [model for (_, model), price in book.prices.items() if price.threshold_tokens]
    assert tiered == ["claude-haiku-5-5"], (
        f"{tiered} are priced by prompt length; the reference says one is"
    )
    text = _section(_text(CONFIGURATION), "Models")
    for needle in (
        f"{haiku.threshold_tokens:,} tokens",
        f"${haiku.input_per_mtok:.2f}",
        f"${haiku.output_per_mtok:.2f}",
        f"${haiku.above_threshold.input_per_mtok:.2f}",
        f"${haiku.above_threshold.output_per_mtok:.2f}",
        f"dated {book.last_updated.isoformat()}",
    ):
        assert needle in text, f"the reference's Prices paragraph does not say {needle!r}"


def test_the_byte_caps_of_the_instruction_files_are_the_ones_the_documents_give():
    for document in (CONFIGURATION, ARCHITECTURE, SECURITY, DESIGN / "0013-thin-context.md"):
        text = _text(document)
        for cap in (MAX_INSTRUCTION_BYTES, MAX_INSTRUCTION_TOTAL_BYTES):
            assert f"{cap:,} bytes" in text, f"{document.name} does not give the cap {cap:,}"


def test_the_instruction_files_are_layered_and_the_total_is_shared_as_the_reference_says(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    home, repository = tmp_path / "home", tmp_path / "repository"
    middle = repository / "services"
    project = middle / "app"
    (home / ".nanoclaude").mkdir(parents=True)
    project.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)  # noqa: S607
    for directory, name, size in (
        (home / ".nanoclaude", "global", 30_000),
        (repository, "top", 10_000),
        (middle, "middle", 10_000),
        (project, "root", 30_000),
    ):
        (directory / "NANO.md").write_text((f"{name}-file " + "x" * size)[:size])
    policy = Policy(
        sandbox=Sandbox((str(project),)),
        rules=RuleSet.build(allow=["Read"]),
        mode=PermissionMode.DEFAULT,
        secret_paths=SECRET_PATH_PATTERNS,
    )
    loaded = load_instructions(
        str(project), cwd=str(project), home=str(home), policy=policy, redactor=Redactor()
    )
    # The global file and the root's are always loaded, 60,000 of the 64,000 bytes. What is left
    # goes to the file nearest the root first, which is cut to it; the file at the top of the
    # repository finds nothing left, and is replaced by a line.
    assert "[truncated at 4000 bytes]" in loaded
    assert "not loaded: instructions are capped at 64000 bytes in total]" in loaded
    assert "top-file" not in loaded
    order = [loaded.index(f"{name}-file ") for name in ("global", "middle", "root")]
    assert order == sorted(order), (
        "the files are not in the prompt from the global one to the root's"
    )
    assert loaded.index("repository/NANO.md not loaded") < loaded.index("middle-file ")
    for document in (CONFIGURATION, SECURITY):
        assert "nearest" in _text(document), document.name


def test_the_modes_the_reference_gives_for_what_ncc_keeps_are_the_ones_it_makes(tmp_path):
    make_private_directories(tmp_path / "a" / "b")
    create_private_file(tmp_path / "a" / "b" / "history")
    assert mode_of(tmp_path / "a" / "b") == 0o700
    assert mode_of(tmp_path / "a" / "b" / "history") == 0o600
    text = _text(CONFIGURATION)
    assert "mode 0700" in text and "0600" in text


def test_a_rules_glob_is_also_matched_against_the_absolute_path_without_its_leading_slash(
    tmp_path,
):
    """A known issue (docs/configuration.md and SECURITY.md say so, and how to avoid it)."""
    root = Path(os.path.realpath(tmp_path))
    top = root.parts[1]  # "tmp" on Linux, "private" on macOS: the first directory under /

    def policy_with(*, allow: tuple[str, ...] = (), deny: tuple[str, ...] = ()) -> Policy:
        return Policy(
            sandbox=Sandbox((str(root),)),
            rules=RuleSet.build(allow=list(allow), deny=list(deny)),
            mode=PermissionMode.DEFAULT,
            secret_paths=SECRET_PATH_PATTERNS,
        )

    denying = policy_with(deny=(f"Read({top}/**)",))
    assert _decide(denying, "Read", "src/a.py", write=False).rule == "rule.deny", (
        "a rule for a directory at the top of the file system no longer matches the files under "
        "it; correct docs/configuration.md and SECURITY.md, which call that a known issue"
    )
    allowing = policy_with(allow=(f"Write({top}/**)",))
    assert _decide(allowing, "Write", "src/a.py", write=True).rule == "rule.allow"
    # What the documents advise instead: the absolute path of the project directory.
    narrow = policy_with(allow=(f"Write({root}/out/**)",))
    assert _decide(narrow, "Write", "out/a.py", write=True).rule == "rule.allow"
    assert _decide(narrow, "Write", "src/a.py", write=True).rule == "default.ask"
    for document in (CONFIGURATION, SECURITY):
        text = _text(document)
        assert "known issue" in text and "`Write(/home/you/app/tmp/**)`" in text, document.name


def test_the_limits_the_reference_calls_reserved_are_read_by_nothing():
    rows = _table_rows(_section(_text(CONFIGURATION), "Limits"))
    reserved = {
        row[0].strip("`"): row[2]
        for row in rows
        if row[0] in ("`bash_timeout_s`", "`output_cap_bytes`")
    }
    assert set(reserved) == {"bash_timeout_s", "output_cap_bytes"}
    for key, meaning in reserved.items():
        assert "Reserved" in meaning, f"the limits table does not call {key} reserved"


# --------------------------------------------------------------------------
# The known limits in SECURITY.md
# --------------------------------------------------------------------------

KNOWN_LIMITS = (
    "**`Grep` and private keys.**",
    "**Editing a key's first line.**",
    "**Short and plain secrets.**",
    "**Names.**",
    "**Your instruction file.**",
    "**Files you made.**",
    "**A rule's glob and the leading slash.**",
    "**Line numbers.**",
)


def test_security_states_each_of_its_known_limits():
    text = _text(SECURITY)
    missing = [limit for limit in KNOWN_LIMITS if limit not in text]
    assert missing == [], f"SECURITY.md does not state {missing}"


async def test_grep_modes_that_show_no_lines_say_whether_a_pattern_matches_inside_a_key(
    ctx, tmp_repo
):
    """A known limit: nothing shows the block's text, and a match inside it is still a match."""
    block, rows = pem(lines=3)
    (tmp_repo / "deploy.sh").write_text("echo start\n" + "\n".join(block) + "\necho end\n")
    probe = rows[0][:8]
    shown: dict[str, str] = {}
    for mode in ("content", "files", "count"):
        outcome = await GrepTool().run(ctx, "t", {"pattern": probe, "output_mode": mode})
        shown[mode] = outcome.content
        assert "1 match(es)" in outcome.content and "deploy.sh" in outcome.content, (
            f"{mode} mode no longer reports a match inside a key block; correct SECURITY.md, "
            f"which says that it does. Grep said: {outcome.content}"
        )
        assert not any(row in outcome.content for row in rows), (
            f"{mode} mode shows a row of a private key"
        )
    assert "[redacted:private-key]" in shown["content"]
    none = await GrepTool().run(ctx, "t", {"pattern": "ZZZZZZZZ", "output_mode": "count"})
    assert "No matches" in none.content


async def test_an_approved_edit_of_a_begin_line_leaves_a_block_that_is_no_longer_masked(
    ctx, tmp_repo
):
    """A known limit: a block is recognised from its BEGIN line, and the edit took that away."""
    block, rows = pem(lines=3)
    target = tmp_repo / "deploy.sh"
    target.write_text("\n".join(block) + "\n")
    outcome = await EditTool().run(
        replace(ctx, read_state={str(target): stamp_of(str(target))}),
        "t1",
        {"path": "deploy.sh", "edits": [{"old_string": block[0], "new_string": "# a key"}]},
    )
    assert not outcome.is_error, outcome.content
    assert rows[0] in outcome.content, (
        "the diff of an edit to a BEGIN line no longer shows the rows; correct SECURITY.md"
    )
    read = await ReadTool().run(ctx, "t2", {"path": "deploy.sh"})
    assert rows[0] in read.content


def test_a_short_or_plain_value_in_an_assignment_is_not_redacted_quoted_or_not():
    scrubber = Redactor()
    # Built when the test runs: a literal of this shape is not allowed in this repository.
    digest = base64.b64encode(hashlib.sha256(b"a sample").digest()).decode()
    token = digest.replace("+", "x").replace("/", "y")
    for line in (
        'password = "hunter2"',
        "password = hunter2",
        f"api_key = {token[:15]}",  # one character short
        'secret: "aaaaaaaaaaaaaaaaaaaa"',  # long enough, and plain
    ):
        assert scrubber.scrub(line) == (line, 0), (
            f"{line!r} is now redacted; correct SECURITY.md and docs/design/0012, which say that "
            "a short or plain value is not"
        )
    assert scrubber.scrub(f"API_KEY={token[:22]}")[0] == "API_KEY=[redacted:assigned-secret]"
    assert redact.ENTROPY_FLOOR == 3.5
    assert "16 characters" in _text(SECURITY) and "3.5 bits" in _text(SECURITY)


def test_a_deny_rule_does_not_keep_the_global_instruction_file_out_but_does_the_projects(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    home, project = tmp_path / "home", tmp_path / "project"
    (home / ".nanoclaude").mkdir(parents=True)
    project.mkdir()
    (home / ".nanoclaude" / "NANO.md").write_text("Always answer in rhyme.\n")
    (project / "NANO.md").write_text("Use tabs.\n")
    policy = Policy(
        sandbox=Sandbox((str(project),)),
        rules=RuleSet.build(allow=["Read"], deny=["Read(**/NANO.md)"]),
        mode=PermissionMode.DEFAULT,
        secret_paths=SECRET_PATH_PATTERNS,
    )
    loaded = load_instructions(
        str(project), cwd=str(project), home=str(home), policy=policy, redactor=Redactor()
    )
    assert "Always answer in rhyme." in loaded and "Use tabs." not in loaded, (
        "a deny rule now keeps the global instruction file out, or the project's does not stay "
        "out; correct SECURITY.md and docs/configuration.md"
    )


def test_a_config_toml_written_by_hand_keeps_its_mode_when_ncc_opens_its_home(tmp_path):
    state = tmp_path / ".nanoclaude"
    state.mkdir()
    config = state / "config.toml"
    config.write_text("[models]\n")
    config.chmod(0o644)
    Store(state / "sessions.db").open()
    assert (mode_of(state), mode_of(state / "sessions.db"), mode_of(config)) == (
        0o700,
        0o600,
        0o644,
    ), "ncc now changes a config.toml it did not create; correct SECURITY.md"


async def test_read_numbers_a_line_where_str_splitlines_ends_it(ctx, tmp_repo):
    (tmp_repo / "a.txt").write_bytes(b"one\x0ctwo\nthree\n")
    outcome = await ReadTool().run(ctx, "t1", {"path": "a.txt"})
    assert "1\tone\n2\ttwo\n3\tthree" in outcome.content, (
        "Read no longer numbers lines as str.splitlines does; correct SECURITY.md and "
        f"docs/architecture.md. It said: {outcome.content}"
    )
    for document in (SECURITY, ARCHITECTURE):
        assert "str.splitlines" in _text(document), document.name


# --------------------------------------------------------------------------
# The design notes and the README
# --------------------------------------------------------------------------


def test_the_kinds_the_redaction_note_names_are_the_ones_the_scrubber_writes():
    source = _text(SRC / "permissions" / "redact.py")
    written = {label for label, _ in redact._SHAPES}
    written |= set(re.findall(r"\[redacted:([a-z-]+)\]", source))
    note = _text(DESIGN / "0012-redaction-before-transcript.md")
    sentence = re.search(r"The kinds are named in the marker: (.*?)\.\s", note, re.DOTALL)
    assert sentence is not None, "0012 does not list the kinds of credential it recognises"
    documented = set(re.findall(r"`([a-z]+(?:-[a-z]+)*)`", sentence.group(1)))
    assert documented == written, (
        f"in the note and not the scrubber: {sorted(documented - written)}; "
        f"in the scrubber and not the note: {sorted(written - documented)}"
    )


def test_the_attribution_note_and_the_guide_state_the_rule_and_not_how_commits_are_made():
    for document in (DESIGN / "0016-attribution-policy.md", CONTRIBUTING):
        text = " ".join(_text(document).lower().split())
        for phrase in ("tools that write commit messages", "act of deleting"):
            assert phrase not in text, f"{document.name} says {phrase!r}"


def test_the_readme_says_what_containment_and_redaction_add():
    text = _text(README)
    for phrase in (
        "`.git` or a `.nanoclaude` directory",
        "private-key block",
        "git rev-parse",
        "Instruction files",
        "names the rule that asked",
        "`--root`",
    ):
        assert phrase in text, f"the README does not say {phrase!r}"


def test_the_changelog_lists_what_the_two_fixes_added():
    text = _text(CHANGELOG)
    for phrase in (
        "`sandbox.protected-path`",
        "`exit_code`",
        "`--mode`",
        "`git rev-parse`",
        "Claude Haiku 5.5",
        "`shell` and `openai` extras",
    ):
        assert phrase in text, f"the changelog does not mention {phrase!r}"


# --------------------------------------------------------------------------
# The numbers
# --------------------------------------------------------------------------


def _numbers_table() -> dict[str, str]:
    rows = _table_rows(_section(_text(README), "Numbers"))
    return {row[1].strip("`"): row[2] for row in rows if re.fullmatch(r"`[a-z_.]+`", row[1])}


def _plain_number(text: str) -> float:
    return float(text.replace(",", "").rstrip("%"))


def test_the_readmes_numbers_are_the_ones_measure_prints():
    measure = load_script("measure")
    cheap = measure.cheap_metrics()
    live = {
        "tests": cheap["tests"],
        "lines.src": cheap["lines"]["src"],
        "lines.tests": cheap["lines"]["tests"],
        "tools": cheap["tools"],
        "adapters": cheap["adapters"],
    }
    documented = _numbers_table()
    assert set(documented) == {*live, "coverage_percent", "danger_corpus"}, (
        f"the README's table has the keys {sorted(documented)}"
    )
    for key, value in live.items():
        assert _plain_number(documented[key]) == value, (
            f"the README says {key} is {documented[key]}; scripts/measure.py finds {value}. "
            "Run python scripts/measure.py and update the Numbers table."
        )
    corpus = cheap["danger_corpus"]
    expected = (
        f"{corpus['missed_by_regex']} of {corpus['dangerous']}"
        if corpus["measured"]
        else "not measured yet"
    )
    assert documented["danger_corpus"] == expected
    # Coverage needs the whole suite to run, so it cannot be recomputed from inside it.
    assert 0 < _plain_number(documented["coverage_percent"]) <= 100


def test_every_figure_has_the_command_that_produced_it():
    rows = _table_rows(_section(_text(README), "Numbers"))
    figures = [row for row in rows if re.fullmatch(r"`[a-z_.]+`", row[1])]
    assert len(figures) == 7
    for row in figures:
        assert "python scripts/measure.py" in row[3], f"{row[1]} has no command"


def test_no_figure_is_quoted_in_the_readme_outside_the_numbers_table():
    readme = _text(README)
    prose = readme.replace(_section(readme, "Numbers"), "")
    quoted = re.findall(
        r"\d[\d,]*(?:\.\d+)?%?\s+(?:tests|lines|tools|adapters|coverage)\b|coverage of \d", prose
    )
    assert quoted == [], f"the README quotes {quoted} outside the table that measure.py feeds"
