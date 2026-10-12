import hashlib
import os
from pathlib import Path

import pytest

from nanoclaude.context.instructions import (
    INSTRUCTION_FILENAMES,
    MAX_INSTRUCTION_BYTES,
    MAX_INSTRUCTION_TOTAL_BYTES,
    _read_one,
)
from nanoclaude.permissions.redact import Redactor
from tests.context.helpers import load, policy_for
from tests.fifo import call_without_blocking, make_fifo

#: A credential shape, built when the module is imported: this repository is public.
TOKEN = "ghp_" + hashlib.sha256(b"instructions").hexdigest()[:36]


def test_instruction_filenames_puts_nano_before_claude():
    """The order is the priority: _read_one returns on the first match it finds,
    so if this tuple were ever reordered, CLAUDE.md would silently start
    winning over NANO.md whenever both exist.
    """
    assert INSTRUCTION_FILENAMES == ("NANO.md", "CLAUDE.md")


def test_nano_md_at_the_root_is_loaded(tmp_repo):
    (tmp_repo / "NANO.md").write_text("Always run make lint.\n")
    assert "make lint" in load(tmp_repo)


def test_claude_md_is_read_when_nano_md_is_absent(tmp_repo):
    (tmp_repo / "CLAUDE.md").write_text("Use tabs.\n")
    assert "Use tabs" in load(tmp_repo)


def test_nano_md_wins_when_both_exist(tmp_repo):
    (tmp_repo / "NANO.md").write_text("nano wins\n")
    (tmp_repo / "CLAUDE.md").write_text("claude loses\n")
    loaded = load(tmp_repo)
    assert "nano wins" in loaded and "claude loses" not in loaded


def test_a_nested_directory_layers_on_top_of_the_root(tmp_repo):
    (tmp_repo / "NANO.md").write_text("root rule\n")
    nested = tmp_repo / "service"
    nested.mkdir()
    (nested / "NANO.md").write_text("service rule\n")
    loaded = load(tmp_repo, cwd=nested)
    assert loaded.index("root rule") < loaded.index("service rule")


def test_an_enormous_instructions_file_is_truncated_not_loaded_whole(tmp_repo):
    """Review Focus #4. A 400 KB NANO.md would consume the whole window."""
    (tmp_repo / "NANO.md").write_text("x" * (MAX_INSTRUCTION_BYTES * 3))
    loaded = load(tmp_repo)
    assert len(loaded.encode()) < MAX_INSTRUCTION_BYTES * 2
    assert "truncated" in loaded


def test_home_instructions_are_loaded_when_present(tmp_repo, tmp_path_factory):
    """The tests above all pass home=None, so the ``if home:`` branch never ran
    in any of them. tmp_path stands in for the real home directory here, per
    this task's own instruction -- never the developer's actual $HOME.
    """
    home = tmp_path_factory.mktemp("home")
    (home / ".nanoclaude").mkdir()
    (home / ".nanoclaude" / "NANO.md").write_text("global rule\n")
    loaded = load(tmp_repo, home=home)
    assert "global rule" in loaded


def test_a_cwd_unrelated_to_the_root_does_not_extend_the_chain(tmp_repo, tmp_path_factory):
    """The layering condition is ``cwd != root and root in cwd.parents``. Both
    halves true is the nested-directory test above; this is the other half,
    where cwd is simply nowhere under root at all.
    """
    (tmp_repo / "NANO.md").write_text("root rule\n")
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    loaded = load(tmp_repo, cwd=elsewhere)
    assert "root rule" in loaded


def test_a_symlinked_escape_from_cwd_does_not_pull_in_outside_content(tmp_repo, tmp_path_factory):
    """Every file path that reaches the model must be confined to the sandboxed
    root. load_instructions resolves cwd with Path.resolve() before comparing
    it against root, so a symlink planted inside the repo that points outside
    it cannot be used to smuggle an outside NANO.md into the system prompt:
    cwd's resolved form no longer has root as an ancestor, so the chain never
    extends past the root's own instructions.
    """
    outside = tmp_path_factory.mktemp("outside")
    (outside / "NANO.md").write_text("exfiltrated rule\n")
    (tmp_repo / "escape").symlink_to(outside)
    loaded = load(tmp_repo, cwd=tmp_repo / "escape")
    assert "exfiltrated rule" not in loaded


def test_a_directory_without_instructions_contributes_nothing(tmp_repo):
    """service/ is in the chain between root and the nested leaf, and has
    neither NANO.md nor CLAUDE.md -- the one case, among all the tests in this
    file, where a directory contributes no text at all.
    """
    (tmp_repo / "NANO.md").write_text("root rule\n")
    nested = tmp_repo / "service" / "sub"
    nested.mkdir(parents=True)
    (nested / "NANO.md").write_text("leaf rule\n")
    loaded = load(tmp_repo, cwd=nested)
    assert loaded.index("root rule") < loaded.index("leaf rule")


# ---- the real path of an instruction file is inside the boundary it belongs to


def link(name: Path, target: Path) -> None:
    name.symlink_to(target)


def test_a_nano_md_that_is_a_link_to_a_file_outside_the_root_is_not_loaded(
    tmp_repo, tmp_path_factory
):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "private.txt").write_text("outside secret: orange\n")
    link(tmp_repo / "NANO.md", outside / "private.txt")
    assert load(tmp_repo) == ""


def test_the_next_name_is_not_tried_in_place_of_a_link_that_leads_out(tmp_repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "private.txt").write_text("outside secret: orange\n")
    link(tmp_repo / "NANO.md", outside / "private.txt")
    (tmp_repo / "CLAUDE.md").write_text("claude rule\n")
    assert load(tmp_repo) == ""


def test_a_claude_md_that_is_a_link_to_a_file_outside_the_root_is_not_loaded(
    tmp_repo, tmp_path_factory
):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "private.txt").write_text("outside secret: orange\n")
    link(tmp_repo / "CLAUDE.md", outside / "private.txt")
    assert load(tmp_repo) == ""


def test_a_nested_directorys_link_to_a_file_outside_the_root_is_not_loaded(
    tmp_repo, tmp_path_factory
):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "private.txt").write_text("outside secret: orange\n")
    (tmp_repo / "NANO.md").write_text("root rule\n")
    nested = tmp_repo / "service"
    nested.mkdir()
    link(nested / "NANO.md", outside / "private.txt")
    loaded = load(tmp_repo, cwd=nested)
    assert "root rule" in loaded and "orange" not in loaded


def test_a_link_to_a_file_elsewhere_in_the_root_is_loaded(tmp_repo):
    """``CLAUDE.md`` pointing at the project's other agent file is a common arrangement."""
    (tmp_repo / "docs").mkdir()
    (tmp_repo / "docs" / "rules.md").write_text("shared rule\n")
    link(tmp_repo / "NANO.md", tmp_repo / "docs" / "rules.md")
    assert "shared rule" in load(tmp_repo)


def test_a_link_to_a_file_through_a_directory_link_that_leads_out_is_not_loaded(
    tmp_repo, tmp_path_factory
):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "NANO.md").write_text("outside secret: orange\n")
    link(tmp_repo / "elsewhere", outside)
    link(tmp_repo / "NANO.md", tmp_repo / "elsewhere" / "NANO.md")
    assert load(tmp_repo) == ""


def test_the_global_file_may_not_be_a_link_out_of_the_global_directory(tmp_repo, tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    (home / ".nanoclaude").mkdir()
    (home / "notes.txt").write_text("a file of the person's, elsewhere in home: orange\n")
    link(home / ".nanoclaude" / "NANO.md", home / "notes.txt")
    assert "orange" not in load(tmp_repo, home=home)


def test_the_global_file_may_be_a_link_within_the_global_directory(tmp_repo, tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    (home / ".nanoclaude").mkdir()
    (home / ".nanoclaude" / "rules.md").write_text("global rule: banana\n")
    link(home / ".nanoclaude" / "NANO.md", home / ".nanoclaude" / "rules.md")
    assert "banana" in load(tmp_repo, home=home)


def test_the_boundary_is_a_parameter_of_the_per_file_reader(tmp_repo):
    """A directory is judged against the boundary it is given, not against itself: a
    chain that reaches above the root passes the repository's top level here."""
    (tmp_repo / "NANO.md").write_text("top rule\n")
    nested = tmp_repo / "service"
    nested.mkdir()
    link(nested / "NANO.md", tmp_repo / "NANO.md")
    policy, redactor = policy_for(tmp_repo), Redactor()
    inside_top = _read_one(nested, within=tmp_repo, policy=policy, redactor=redactor)
    assert inside_top is not None and "top rule" in inside_top.text
    inside_itself = _read_one(nested, within=nested, policy=policy, redactor=redactor)
    assert inside_itself is None


def test_the_boundary_is_compared_as_a_real_path(tmp_repo, tmp_path_factory):
    """A boundary reached through a link (a repository opened by a symlinked path) is the
    directory it leads to, as the file's real path is."""
    (tmp_repo / "NANO.md").write_text("top rule\n")
    alias = tmp_path_factory.mktemp("aliases") / "project"
    alias.symlink_to(tmp_repo, target_is_directory=True)
    loaded = _read_one(tmp_repo, within=alias, policy=policy_for(tmp_repo), redactor=Redactor())
    assert loaded is not None and "top rule" in loaded.text


def test_a_link_that_leads_to_something_that_is_not_a_file_is_not_loaded(tmp_repo):
    (tmp_repo / "docs").mkdir()
    link(tmp_repo / "NANO.md", tmp_repo / "docs")
    (tmp_repo / "CLAUDE.md").write_text("claude rule\n")
    assert load(tmp_repo) == ""


def test_a_link_to_a_pipe_is_not_opened(tmp_repo):
    """Opening a FIFO for reading waits for a writer, and the session with it."""
    pipe = make_fifo(tmp_repo / "pipe")
    link(tmp_repo / "NANO.md", pipe)
    assert call_without_blocking(pipe, load, tmp_repo) == ""


def test_a_dangling_link_is_not_loaded_and_does_not_let_the_next_name_in(tmp_repo):
    link(tmp_repo / "NANO.md", tmp_repo / "missing.md")
    (tmp_repo / "CLAUDE.md").write_text("claude rule\n")
    assert load(tmp_repo) == ""


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads files whatever their mode")
def test_a_file_that_cannot_be_read_is_skipped_not_an_error(tmp_repo):
    (tmp_repo / "NANO.md").write_text("root rule\n")
    (tmp_repo / "NANO.md").chmod(0)
    try:
        assert load(tmp_repo) == ""
    finally:
        (tmp_repo / "NANO.md").chmod(0o644)


def test_a_directory_called_nano_md_is_not_the_file_and_claude_md_is_read(tmp_repo):
    (tmp_repo / "NANO.md").mkdir()
    (tmp_repo / "CLAUDE.md").write_text("claude rule\n")
    assert "claude rule" in load(tmp_repo)


# ---- an instruction file inside the sandbox is read as Read would read it


def test_a_link_to_a_credentials_file_inside_the_root_is_not_loaded(tmp_repo):
    (tmp_repo / ".env").write_text("PGPASSWORD=orange\n")
    link(tmp_repo / "NANO.md", tmp_repo / ".env")
    assert load(tmp_repo) == ""


def test_a_read_deny_rule_keeps_an_instruction_file_out(tmp_repo):
    (tmp_repo / "NANO.md").write_text("root rule\n")
    nested = tmp_repo / "service"
    nested.mkdir()
    (nested / "NANO.md").write_text("service rule: orange\n")
    loaded = load(tmp_repo, cwd=nested, deny=["Read(service/**)"])
    assert "root rule" in loaded and "orange" not in loaded


def test_a_read_deny_rule_for_the_file_itself_keeps_it_out_and_the_next_name_is_not_tried(
    tmp_repo,
):
    (tmp_repo / "NANO.md").write_text("nano rule: orange\n")
    (tmp_repo / "CLAUDE.md").write_text("claude rule\n")
    assert load(tmp_repo, deny=["Read(NANO.md)"]) == ""


def test_a_deny_rule_for_the_name_of_a_link_keeps_it_out_though_its_target_is_not_denied(tmp_repo):
    (tmp_repo / "docs").mkdir()
    (tmp_repo / "docs" / "rules.md").write_text("shared rule: orange\n")
    link(tmp_repo / "NANO.md", tmp_repo / "docs" / "rules.md")
    assert load(tmp_repo, deny=["Read(NANO.md)"]) == ""


def test_a_deny_rule_is_judged_by_where_a_link_leads_as_well_as_by_its_name(tmp_repo):
    (tmp_repo / "private").mkdir()
    (tmp_repo / "private" / "rules.md").write_text("private rule: orange\n")
    link(tmp_repo / "NANO.md", tmp_repo / "private" / "rules.md")
    assert load(tmp_repo, deny=["Read(private/**)"]) == ""


def test_the_global_file_is_outside_the_sandbox_and_is_not_judged_by_it(tmp_repo, tmp_path_factory):
    """The sandbox answers for what is in the project; a deny rule written for the
    project's files does not reach the person's own file in their home."""
    home = tmp_path_factory.mktemp("home")
    (home / ".nanoclaude").mkdir()
    (home / ".nanoclaude" / "NANO.md").write_text("global rule: banana\n")
    assert "banana" in load(tmp_repo, home=home, deny=["Read(**/NANO.md)"])


# ---- what reaches the prompt has been through the session's redactor


def test_a_credential_in_an_instruction_file_does_not_reach_the_prompt(tmp_repo):
    (tmp_repo / "NANO.md").write_text(f"Deploy with {TOKEN} from CI.\n")
    loaded = load(tmp_repo)
    assert TOKEN not in loaded
    assert "[redacted:github-token]" in loaded and "Deploy with" in loaded


def test_a_credential_in_the_global_file_or_a_nested_one_does_not_reach_the_prompt(
    tmp_repo, tmp_path_factory
):
    home = tmp_path_factory.mktemp("home")
    (home / ".nanoclaude").mkdir()
    (home / ".nanoclaude" / "NANO.md").write_text(f"global {TOKEN}\n")
    nested = tmp_repo / "service"
    nested.mkdir()
    (nested / "NANO.md").write_text(f"nested {TOKEN}\n")
    assert TOKEN not in load(tmp_repo, cwd=nested, home=home)


def test_the_redactor_that_is_used_is_the_one_that_was_given(tmp_repo):
    (tmp_repo / "NANO.md").write_text(f"Deploy with {TOKEN}.\n")
    assert TOKEN in load(tmp_repo, redactor=Redactor(enabled=False))


def test_a_credential_that_straddles_the_per_file_cut_leaves_no_fragment(tmp_repo):
    """Cut first and scrubbed after, the front half of the token would be left as text
    nothing recognises."""
    (tmp_repo / "NANO.md").write_text("-" * (MAX_INSTRUCTION_BYTES - 10) + TOKEN + "\n" * 3)
    loaded = load(tmp_repo)
    assert TOKEN[:6] not in loaded


# ---- the chain as a whole is capped

FILLERS = {"global": "^", "root": "|", "a": "%", "b": "~", "c": "`"}


def chain(
    tmp_repo: Path, tmp_path_factory: pytest.TempPathFactory, sizes: dict[str, int]
) -> tuple[Path, Path, Path]:
    """A home with a global file, a root with one, and ``a/b/c`` below it each with one, of
    the given sizes; a size of 0 leaves that file out. Each is filled with its own
    character, so what of it reached the output can be counted."""
    home = tmp_path_factory.mktemp("home")
    (home / ".nanoclaude").mkdir()
    deepest = tmp_repo / "a" / "b" / "c"
    deepest.mkdir(parents=True)
    places = {
        "global": home / ".nanoclaude",
        "root": tmp_repo,
        "a": tmp_repo / "a",
        "b": tmp_repo / "a" / "b",
        "c": deepest,
    }
    for name, size in sizes.items():
        if size:
            (places[name] / "NANO.md").write_text(FILLERS[name] * size)
    return home, deepest, tmp_repo


def counts(loaded: str) -> dict[str, int]:
    return {name: loaded.count(char) for name, char in FILLERS.items()}


def test_the_chain_limit_is_twice_the_per_file_limit():
    assert MAX_INSTRUCTION_BYTES == 32_000
    assert MAX_INSTRUCTION_TOTAL_BYTES == 64_000


def test_a_chain_that_fits_is_loaded_whole(tmp_repo, tmp_path_factory):
    home, deepest, _ = chain(
        tmp_repo, tmp_path_factory, {"global": 1_000, "root": 2_000, "a": 3_000, "c": 4_000}
    )
    loaded = load(tmp_repo, cwd=deepest, home=home)
    assert counts(loaded) == {"global": 1_000, "root": 2_000, "a": 3_000, "b": 0, "c": 4_000}
    assert "not loaded" not in loaded and "truncated" not in loaded


def test_files_below_the_root_get_what_is_left_nearest_the_root_first(tmp_repo, tmp_path_factory):
    home, deepest, _ = chain(
        tmp_repo,
        tmp_path_factory,
        {"global": 10_000, "root": 20_000, "a": 30_000, "b": 30_000, "c": 100},
    )
    loaded = load(tmp_repo, cwd=deepest, home=home)
    # 64,000 less 10,000 and 20,000 is 34,000: all of a's 30,000, then what is left of it.
    assert counts(loaded) == {"global": 10_000, "root": 20_000, "a": 30_000, "b": 4_000, "c": 0}
    assert "[truncated at 4000 bytes]" in loaded
    where = os.path.realpath(deepest / "NANO.md")
    note = f"[{where} not loaded: instructions are capped at 64000 bytes in total]"
    assert note in loaded


def test_the_output_order_is_global_then_root_then_down_to_the_cwd(tmp_repo, tmp_path_factory):
    home, deepest, _ = chain(
        tmp_repo,
        tmp_path_factory,
        {"global": 10_000, "root": 20_000, "a": 30_000, "b": 30_000, "c": 100},
    )
    loaded = load(tmp_repo, cwd=deepest, home=home)
    positions = [loaded.index(FILLERS[name]) for name in ("global", "root", "a", "b")]
    assert positions == sorted(positions)
    assert loaded.index(FILLERS["b"]) < loaded.index("not loaded")


def test_the_global_and_root_files_are_always_included_each_cut_at_the_per_file_limit(
    tmp_repo, tmp_path_factory
):
    home, deepest, _ = chain(
        tmp_repo,
        tmp_path_factory,
        {"global": 50_000, "root": 50_000, "a": 5_000, "b": 5_000},
    )
    loaded = load(tmp_repo, cwd=deepest, home=home)
    assert counts(loaded) == {"global": 32_000, "root": 32_000, "a": 0, "b": 0, "c": 0}
    assert loaded.count("[truncated at 32000 bytes]") == 2
    assert loaded.count("not loaded: instructions are capped at 64000 bytes in total]") == 2


def test_a_file_below_the_root_is_still_cut_at_the_per_file_limit_when_more_is_left(
    tmp_repo, tmp_path_factory
):
    home, deepest, _ = chain(tmp_repo, tmp_path_factory, {"root": 1_000, "a": 50_000})
    loaded = load(tmp_repo, cwd=deepest, home=home)
    assert counts(loaded)["a"] == 32_000
    assert "[truncated at 32000 bytes]" in loaded


def test_a_file_that_gets_nothing_is_one_line_and_an_empty_one_is_not_mentioned(
    tmp_repo, tmp_path_factory
):
    home, deepest, _ = chain(
        tmp_repo, tmp_path_factory, {"global": 32_000, "root": 32_000, "a": 10, "b": 10}
    )
    (deepest / "NANO.md").write_text("")
    loaded = load(tmp_repo, cwd=deepest, home=home)
    notes = [line for line in loaded.splitlines() if "not loaded" in line]
    assert len(notes) == 2, notes
    assert all(line.startswith("[") and line.endswith("]") for line in notes)
    assert str(deepest / "NANO.md") not in "\n".join(notes)


def test_the_note_stays_one_line_whatever_the_directory_is_called(tmp_repo, tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    (home / ".nanoclaude").mkdir()
    (home / ".nanoclaude" / "NANO.md").write_text(FILLERS["global"] * 32_000)
    (tmp_repo / "NANO.md").write_text(FILLERS["root"] * 32_000)
    odd = tmp_repo / "odd\nname"
    odd.mkdir()
    (odd / "NANO.md").write_text("rule\n")
    loaded = load(tmp_repo, cwd=odd, home=home)
    notes = [line for line in loaded.splitlines() if "not loaded" in line]
    assert len(notes) == 1 and notes[0].endswith("capped at 64000 bytes in total]")
    assert notes[0].startswith("[") and "odd" in notes[0] and "name" in notes[0]


def test_a_skipped_file_uses_none_of_the_budget(tmp_repo, tmp_path_factory):
    home, deepest, _ = chain(
        tmp_repo, tmp_path_factory, {"global": 32_000, "root": 31_000, "a": 0, "b": 0, "c": 0}
    )
    outside = tmp_path_factory.mktemp("outside")
    (outside / "big.txt").write_text(FILLERS["a"] * 30_000)
    link(tmp_repo / "a" / "NANO.md", outside / "big.txt")
    (tmp_repo / "a" / "b" / "NANO.md").write_text(FILLERS["b"] * 900)
    loaded = load(tmp_repo, cwd=deepest, home=home)
    assert counts(loaded)["a"] == 0 and counts(loaded)["b"] == 900
    assert "not loaded" not in loaded


@pytest.mark.parametrize("name", ["NANO.md", "CLAUDE.md"])
def test_either_file_name_takes_part_in_the_cap(tmp_repo, tmp_path_factory, name):
    home, deepest, _ = chain(tmp_repo, tmp_path_factory, {"global": 32_000, "root": 32_000})
    (deepest / name).write_text("rule\n")
    loaded = load(tmp_repo, cwd=deepest, home=home)
    assert "rule" not in loaded.replace("not loaded", "") and "not loaded" in loaded


def test_the_global_and_root_files_are_loaded_even_when_they_alone_exceed_the_total(
    tmp_repo, tmp_path_factory, monkeypatch
):
    """With the constants as they are, two files cut at the per-file limit can never exceed
    the total, so 'always' cannot be told from 'what is left'; a smaller total can."""
    monkeypatch.setattr("nanoclaude.context.instructions.MAX_INSTRUCTION_TOTAL_BYTES", 40_000)
    home, deepest, _ = chain(
        tmp_repo, tmp_path_factory, {"global": 32_000, "root": 32_000, "a": 5_000}
    )
    loaded = load(tmp_repo, cwd=deepest, home=home)
    assert counts(loaded) == {"global": 32_000, "root": 32_000, "a": 0, "b": 0, "c": 0}
    assert "capped at 40000 bytes in total]" in loaded


def test_what_is_left_for_the_rest_is_counted_from_what_the_always_files_actually_took(
    tmp_repo, tmp_path_factory
):
    """A global file a little over the per-file limit is cut to it, and it is the cut size
    that comes off the total."""
    home, deepest, _ = chain(tmp_repo, tmp_path_factory, {"global": 34_000, "root": 0, "a": 32_000})
    loaded = load(tmp_repo, cwd=deepest, home=home)
    assert counts(loaded) == {"global": 32_000, "root": 0, "a": 32_000, "b": 0, "c": 0}
    assert "[truncated at 32000 bytes]" in loaded


# ---- the cut is by bytes, and a file longer than what was read is marked as cut


def test_a_file_of_wide_characters_is_cut_at_the_limit_in_bytes(tmp_repo):
    (tmp_repo / "NANO.md").write_text("\u00e9" * 30_000)
    loaded = load(tmp_repo)
    assert loaded.count("\u00e9") == MAX_INSTRUCTION_BYTES // 2
    assert "[truncated at 32000 bytes]" in loaded


def test_a_file_that_is_cut_where_it_is_read_says_so_even_when_what_is_left_is_short(tmp_repo):
    """Scrubbing shrinks what was read to less than the limit, but the file went on."""
    word = "ghp_" + hashlib.sha256(b"shrink").hexdigest()[:36] + " "
    (tmp_repo / "NANO.md").write_text(word * 1_200)  # 49,200 bytes, about 28,000 once scrubbed
    loaded = load(tmp_repo)
    assert "[truncated at 32000 bytes]" in loaded
    assert TOKEN[:6] not in loaded
