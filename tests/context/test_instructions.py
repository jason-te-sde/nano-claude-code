from nanoclaude.context.instructions import (
    INSTRUCTION_FILENAMES,
    MAX_INSTRUCTION_BYTES,
    load_instructions,
)


def test_instruction_filenames_puts_nano_before_claude():
    """The order is the priority: _read_one returns on the first match it finds,
    so if this tuple were ever reordered, CLAUDE.md would silently start
    winning over NANO.md whenever both exist.
    """
    assert INSTRUCTION_FILENAMES == ("NANO.md", "CLAUDE.md")


def test_nano_md_at_the_root_is_loaded(tmp_repo):
    (tmp_repo / "NANO.md").write_text("Always run make lint.\n")
    assert "make lint" in load_instructions(str(tmp_repo), cwd=str(tmp_repo), home=None)


def test_claude_md_is_read_when_nano_md_is_absent(tmp_repo):
    (tmp_repo / "CLAUDE.md").write_text("Use tabs.\n")
    assert "Use tabs" in load_instructions(str(tmp_repo), cwd=str(tmp_repo), home=None)


def test_nano_md_wins_when_both_exist(tmp_repo):
    (tmp_repo / "NANO.md").write_text("nano wins\n")
    (tmp_repo / "CLAUDE.md").write_text("claude loses\n")
    loaded = load_instructions(str(tmp_repo), cwd=str(tmp_repo), home=None)
    assert "nano wins" in loaded and "claude loses" not in loaded


def test_a_nested_directory_layers_on_top_of_the_root(tmp_repo):
    (tmp_repo / "NANO.md").write_text("root rule\n")
    nested = tmp_repo / "service"
    nested.mkdir()
    (nested / "NANO.md").write_text("service rule\n")
    loaded = load_instructions(str(tmp_repo), cwd=str(nested), home=None)
    assert loaded.index("root rule") < loaded.index("service rule")


def test_an_enormous_instructions_file_is_truncated_not_loaded_whole(tmp_repo):
    """Review Focus #4. A 400 KB NANO.md would consume the whole window."""
    (tmp_repo / "NANO.md").write_text("x" * (MAX_INSTRUCTION_BYTES * 3))
    loaded = load_instructions(str(tmp_repo), cwd=str(tmp_repo), home=None)
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
    loaded = load_instructions(str(tmp_repo), cwd=str(tmp_repo), home=str(home))
    assert "global rule" in loaded


def test_a_cwd_unrelated_to_the_root_does_not_extend_the_chain(tmp_repo, tmp_path_factory):
    """The layering condition is ``cwd != root and root in cwd.parents``. Both
    halves true is the nested-directory test above; this is the other half,
    where cwd is simply nowhere under root at all.
    """
    (tmp_repo / "NANO.md").write_text("root rule\n")
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    loaded = load_instructions(str(tmp_repo), cwd=str(elsewhere), home=None)
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
    loaded = load_instructions(str(tmp_repo), cwd=str(tmp_repo / "escape"), home=None)
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
    loaded = load_instructions(str(tmp_repo), cwd=str(nested), home=None)
    assert loaded.index("root rule") < loaded.index("leaf rule")
