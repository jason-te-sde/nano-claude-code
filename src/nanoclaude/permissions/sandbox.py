"""Where the agent is allowed to touch the filesystem.

Two rules, and the order matters. Resolution happens first and always: a path is
judged by what it points at, not by how it was spelled. Containment is then a
comparison between resolved absolute paths.

Writes additionally resolve the *parent*, on top of the whole-path resolution
``resolve()`` already does. The two usually agree: realpath walks through any
symlinked directory on the way to a leaf whether or not that leaf itself exists,
so a properly end-to-end-resolved path already reveals a symlinked-parent
escape. This second check earns its keep when ``check_write`` is instead handed
a path that skipped that resolution -- built, say, by joining an
already-resolved directory with a bare filename. A plain containment check on
that string is fooled, because it is textually beneath the root; only
re-resolving the parent catches the escaping symlink.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


def is_within(root: str, path: str) -> bool:
    """True if ``path`` is ``root`` or sits underneath it.

    ``PurePosixPath`` rather than ``str.startswith`` because ``/tmp/project-evil``
    starts with ``/tmp/project`` and is nowhere near inside it.

    Known limitation: this comparison is case-sensitive, but macOS's default
    filesystem is not -- ``realpath`` does not canonicalise a path's case, so
    ``/Users/x/Project`` and ``/Users/x/project`` can be the same file on disk
    while comparing unequal here. That can make this function reject a
    differently-cased spelling of a location that is genuinely inside the
    root. It cannot do the opposite: case-insensitivity only ever collapses
    distinct spellings onto the *same* file, so a stricter, case-sensitive
    comparison can never admit a path that actually denotes somewhere else.
    Do not "fix" this by case-folding before comparing -- that would trade a
    false reject for a possible false admit.
    """
    root_path, candidate = PurePosixPath(root), PurePosixPath(path)
    if not root_path.is_absolute() or not candidate.is_absolute():
        raise ValueError(f"is_within needs absolute paths, got {root!r} and {path!r}")
    return candidate == root_path or root_path in candidate.parents


@dataclass(frozen=True, slots=True)
class Sandbox:
    roots: tuple[str, ...]

    def __post_init__(self) -> None:
        # Reject a relative root before resolving anything. resolve() below
        # takes an explicit `base` precisely so a candidate path never depends
        # on the process's ambient cwd; silently letting Path.resolve() join a
        # relative root against that same cwd would undermine the identical
        # discipline this class is otherwise careful about -- the same
        # configuration string would then denote a different sandbox
        # depending on where the process happened to be standing at
        # construction time. is_within's own absolute-path guard cannot catch
        # this after the fact, because by the time anything calls is_within
        # the root below has already been made absolute.
        for root in self.roots:
            if not Path(root).is_absolute():
                raise ValueError(f"Sandbox roots must be absolute, got {root!r}")
        # A root is a path like any other: judged by the location it denotes, not
        # by how it was spelled. Every candidate this class compares against a
        # root has already been through realpath (see resolve() below), so a root
        # reached through a symlink -- a project opened via a symlinked working
        # directory, or on macOS, anything under /tmp or /var, which are
        # themselves symlinks to /private/tmp and /private/var -- would never
        # textually match those always-resolved candidates, and every file in an
        # otherwise-legitimate sandbox would be rejected as outside-root.
        # Resolving roots once here, up front, keeps both sides of every
        # comparison in the same, real, symlink-free form.
        object.__setattr__(self, "roots", tuple(str(Path(root).resolve()) for root in self.roots))

    def resolve(self, raw: str, *, base: str) -> str:
        """Absolute, symlink-free. Works for files that do not exist yet."""
        if not raw:
            raise ValueError("path must not be empty")
        expanded = Path(raw).expanduser()
        if not expanded.is_absolute():
            expanded = Path(base) / expanded
        return str(expanded.resolve())

    def contains(self, resolved: str) -> bool:
        return any(is_within(root, resolved) for root in self.roots)

    def check_read(self, resolved: str) -> str | None:
        return None if self.contains(resolved) else "sandbox.outside-root"

    def check_write(self, resolved: str) -> str | None:
        outside = self.check_read(resolved)
        if outside is not None:
            return outside
        parent = str(Path(resolved).parent.resolve())
        if not any(is_within(root, parent) for root in self.roots):
            return "sandbox.symlink-escape"
        return None
