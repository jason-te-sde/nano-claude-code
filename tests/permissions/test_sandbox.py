import pytest

from nanoclaude.permissions.sandbox import Sandbox, is_within


def test_is_within_is_not_a_prefix_match():
    assert is_within("/work/project", "/work/project/src/a.py")
    assert is_within("/work/project", "/work/project")
    assert not is_within("/work/project", "/work/project-evil/a.py")  # the trap
    assert not is_within("/work/project", "/work")


def test_relative_paths_resolve_against_the_root_not_the_process_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir("/")
    sandbox = Sandbox((str(tmp_path),))
    assert sandbox.resolve("src/a.py", base=str(tmp_path)) == str(tmp_path / "src" / "a.py")


def test_a_path_outside_the_root_is_refused(tmp_path):
    sandbox = Sandbox((str(tmp_path),))
    assert sandbox.check_read(sandbox.resolve("/etc/passwd", base=str(tmp_path))) == (
        "sandbox.outside-root"
    )


def test_a_symlink_inside_the_root_pointing_out_is_refused(tmp_path):
    """Review Focus #2. Containment on the literal path passes; on realpath it fails."""
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "link.txt").symlink_to(outside)

    sandbox = Sandbox((str(root),))
    resolved = sandbox.resolve("link.txt", base=str(root))
    assert resolved == str(outside.resolve())
    assert sandbox.check_read(resolved) == "sandbox.outside-root"


def test_writing_through_a_symlinked_parent_directory_is_refused(tmp_path):
    """The agent can create the symlink itself, so the parent must be resolved too.

    resolve() realpaths the *whole* raw path, and realpath resolves every
    component that exists even when the trailing component does not -- so by the
    time check_write sees it, the symlinked "escape" directory is already gone,
    replaced by where it really points. check_read's plain outside-root check
    therefore already catches this particular case; see
    test_check_write_catches_a_symlinked_parent_that_was_never_resolved below for
    the input shape that actually needs check_write's own parent re-resolution.
    """
    outside = tmp_path.parent / "outside_dir"
    outside.mkdir()
    root = tmp_path / "repo"
    root.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    sandbox = Sandbox((str(root),))
    target = str(root / "escape" / "new.txt")
    resolved = sandbox.resolve(target, base=str(root))
    assert sandbox.check_write(resolved) == "sandbox.outside-root"


def test_check_write_catches_a_symlinked_parent_that_was_never_resolved(tmp_path):
    """Pins the sandbox.symlink-escape branch itself.

    check_write's contract is to be the final word, independent of whether its
    caller was careful. Feed it a path built by plain string-joining -- never
    passed through resolve() -- whose parent directory is a symlink escaping the
    root. A plain containment check on that string is fooled (it is textually
    beneath the root); only re-resolving the parent, which check_write does and
    check_read does not, catches it.
    """
    outside = tmp_path.parent / "outside_dir2"
    outside.mkdir()
    root = tmp_path / "repo2"
    root.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    sandbox = Sandbox((str(root),))
    unresolved = str(root / "escape" / "new.txt")
    assert sandbox.check_read(unresolved) is None  # fooled: textually inside root
    assert sandbox.check_write(unresolved) == "sandbox.symlink-escape"


def test_check_write_parent_check_considers_every_root(tmp_path):
    # check_write's own parent-resolution re-checks against *every* configured
    # root, same as check_read -- not just the one the symlink happens to sit
    # under. A single-root fixture cannot tell any() from all() here (they agree
    # when there is only one root), so this needs two.
    root_a = tmp_path / "root_a"
    root_a.mkdir()
    root_b = tmp_path / "root_b"
    root_b.mkdir()
    (root_a / "link_to_b").symlink_to(root_b, target_is_directory=True)

    sandbox = Sandbox((str(root_a), str(root_b)))
    unresolved = str(root_a / "link_to_b" / "new.txt")
    assert sandbox.check_read(unresolved) is None  # textually under root_a
    assert sandbox.check_write(unresolved) is None  # parent resolves into root_b, still configured


def test_a_new_file_in_the_root_is_allowed_for_write(tmp_path):
    sandbox = Sandbox((str(tmp_path),))
    assert sandbox.check_write(sandbox.resolve("brand/new.txt", base=str(tmp_path))) is None


def test_additional_roots_participate_in_every_check(tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    sandbox = Sandbox((str(tmp_path / "main"), str(extra)))
    (tmp_path / "main").mkdir()
    assert sandbox.check_read(str(extra / "a.txt")) is None


def test_absolute_paths_are_required():
    with pytest.raises(ValueError, match="absolute"):
        is_within("relative", "/work/x")


def test_absolute_paths_are_required_for_the_candidate_too():
    # The guard is `not root.is_absolute() or not candidate.is_absolute()` --
    # an independent check for each side. The case above only ever makes the
    # root side relative; this pins the other half, which would stay green if
    # deleted since the first test never exercises a relative candidate.
    with pytest.raises(ValueError, match="absolute"):
        is_within("/work/project", "relative")


def test_contains_is_true_inside_and_false_outside(tmp_path):
    sandbox = Sandbox((str(tmp_path),))
    inside = sandbox.resolve("a.txt", base=str(tmp_path))
    assert sandbox.contains(inside) is True
    assert sandbox.contains("/etc/passwd") is False


def test_resolve_rejects_an_empty_path(tmp_path):
    sandbox = Sandbox((str(tmp_path),))
    with pytest.raises(ValueError, match="empty"):
        sandbox.resolve("", base=str(tmp_path))


def test_resolve_expands_a_home_relative_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = Sandbox((str(tmp_path),))
    assert sandbox.resolve("~/a.txt", base="/somewhere/else") == str(tmp_path / "a.txt")


def test_a_root_given_via_a_symlink_still_recognizes_files_inside_it(tmp_path):
    """A root is a path like any other: judged by location, not spelling.

    Every candidate compared against a root has already been through resolve().
    Without resolving roots the same way, a project opened through a symlinked
    working directory -- or, on macOS, simply living under /tmp or /var, which
    are themselves symlinks to /private/tmp and /private/var -- would have every
    single file inside it rejected as outside-root, because the always-resolved
    candidate could never textually match the never-resolved root.
    """
    real_root = tmp_path / "real_repo"
    real_root.mkdir()
    link_root = tmp_path / "repo_link"
    link_root.symlink_to(real_root, target_is_directory=True)
    sibling = tmp_path / "sibling.txt"
    sibling.write_text("not part of the project")

    sandbox = Sandbox((str(link_root),))

    inside = sandbox.resolve("a.txt", base=str(link_root))
    assert sandbox.check_read(inside) is None

    outside = sandbox.resolve(str(sibling), base=str(link_root))
    assert sandbox.check_read(outside) == "sandbox.outside-root"


def test_sandbox_rejects_a_relative_root(monkeypatch, tmp_path):
    # Without this guard, Path.resolve() in __post_init__ would silently join
    # a relative root against the process cwd -- the same configuration
    # string then denoting a different sandbox depending on where the
    # process happened to be standing when it was constructed. Checked
    # before any resolving happens, and from two different cwds, so a guard
    # that merely resolved-then-compared-to-original would not quietly pass.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="roots must be absolute"):
        Sandbox(("relative/root",))


def test_sandbox_is_hashable():
    # Unlike LoopState/Done/RunTools (tests/agent/test_loop.py) or Message/
    # Transcript (tests/conversation/test_transcript.py), Sandbox holds only a
    # tuple[str, ...] -- every element a plain, already-hashable string -- so
    # frozen=True's generated __hash__ is not a trap here, unlike those classes'
    # Mapping- and ToolUseBlock-holding fields. Pinned so a later field of a
    # mapping or other unhashable type gets caught the same way those were.
    sandbox = Sandbox(("/a", "/b"))
    assert hash(sandbox) == hash(Sandbox(("/a", "/b")))
