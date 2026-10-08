"""The documentation makes claims. These check the cheap ones.

What a test here cannot check is whether a sentence is true, only that what the sentence
names exists: a tool, a command, a key, a file. That is still most of how documentation
goes wrong, since it is code that moves and the prose that stays.
"""

import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import fields
from pathlib import Path

import pytest

from nanoclaude.cli.commands import COMMANDS
from nanoclaude.cli.init import PRESETS
from nanoclaude.cli.main import EXIT_CODES, main
from nanoclaude.config.load import CONFIG_DIRNAME, CONFIG_FILENAME, load_config
from nanoclaude.config.schema import ROLES, LimitsConfig, ModelConfig, PermissionsConfig, UiConfig
from nanoclaude.permissions.danger.regex import RegexClassifier
from nanoclaude.providers.capabilities import CACHE_FILENAME, CONSERVATIVE_DEFAULT, capabilities_for
from nanoclaude.tools.registry import default_registry
from nanoclaude.tools.todo import TodoState
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
CURRENT_ANTHROPIC = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-4-5", "claude-fable-5-1"}


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
model   = "claude-haiku-4-5"

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


def test_the_architecture_document_states_the_two_extensions_of_the_plan():
    text = _text(ARCHITECTURE)
    assert "tool.internal-error" in text
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


def test_the_regex_classifier_does_not_yet_mark_its_clear_verdicts_as_unreliable():
    """The architecture document says so, as something the shell tool's task has to do."""
    verdict = RegexClassifier().classify("ls -la")
    assert verdict.authoritative is True, (
        "the regex classifier now marks its safe verdicts as not authoritative; "
        "correct the paragraph on the danger classifier in docs/architecture.md"
    )
    assert "does not mark its verdicts so yet" in _text(ARCHITECTURE)


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
