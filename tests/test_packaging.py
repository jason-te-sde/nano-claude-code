"""Tests for what pyproject.toml promises an install will bring, and for what says to ask for it.

An extra is a promise: install with it and something works that did not. One that installs
packages nothing imports promises nothing, and costs the person a download. And an install
line that names an extra pyproject does not define is quietly accepted by pip, which only
warns, so CI would go on green while testing something other than what it says.
"""

import ast
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "nanoclaude"

#: Where somebody is told how to install the project, with extras. The CI workflow and the
#: contributing guide are the two that install from a checkout.
INSTALL_INSTRUCTIONS = (".github/workflows/ci.yml", "CONTRIBUTING.md", "README.md")

#: The extra that holds the tools used to work on the project and not to run it.
DEV_EXTRA = "dev"

_REQUIREMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_BRACKETED = re.compile(r"\[([A-Za-z0-9_.,\- ]+)\]")


def declared_extras() -> dict[str, list[str]]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras: dict[str, list[str]] = data["project"].get("optional-dependencies", {})
    return extras


def imported_names() -> set[str]:
    """The top-level name of every module the package's own source imports."""
    names: set[str] = set()
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module.split(".")[0])
    return names


def import_name(requirement: str) -> str:
    """``tree-sitter>=0.22`` -> ``tree_sitter``: the distribution's name as it is imported."""
    match = _REQUIREMENT_NAME.match(requirement)
    assert match is not None, f"not a requirement: {requirement!r}"
    return match.group(0).lower().replace("-", "_").replace(".", "_")


def test_there_are_extras_to_check_and_a_dev_one_that_is_exempt():
    # A test that loops over nothing passes having checked nothing.
    assert DEV_EXTRA in declared_extras()
    assert imported_names() >= {"httpx", "rich"}


def test_every_extra_but_dev_installs_only_what_the_source_imports():
    imported = imported_names()
    unused = sorted(
        f"{extra}: {requirement}"
        for extra, requirements in declared_extras().items()
        if extra != DEV_EXTRA
        for requirement in requirements
        if import_name(requirement) not in imported
    )
    assert unused == [], (
        "these extras install packages that nothing under src/nanoclaude imports; "
        f"remove them until the code that uses them lands: {unused}"
    )


@pytest.mark.parametrize("instructions", INSTALL_INSTRUCTIONS)
def test_an_install_instruction_names_only_extras_that_exist(instructions):
    extras = declared_extras()
    named: set[str] = set()
    for line in (ROOT / instructions).read_text(encoding="utf-8").splitlines():
        if "install" not in line:
            continue
        for group in _BRACKETED.findall(line):
            named.update(name.strip() for name in group.split(","))
    assert sorted(named - extras.keys()) == [], (
        f"{instructions} asks pip for extras that pyproject.toml does not define; "
        f"it defines {sorted(extras)}"
    )


@pytest.mark.parametrize("instructions", [".github/workflows/ci.yml", "CONTRIBUTING.md"])
def test_the_checkout_installs_say_which_extras_they_want(instructions):
    # The parametrized test above passes for a file that names no extras at all, which is
    # also what it would see if these lines were changed to something it cannot read.
    text = (ROOT / instructions).read_text(encoding="utf-8")
    assert re.search(r'pip install -e "\.\[[^\]]*\b' + DEV_EXTRA + r'\b[^\]]*\]"', text), (
        f"{instructions} no longer has an editable install that asks for the {DEV_EXTRA} extra"
    )
