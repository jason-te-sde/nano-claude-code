"""Running git in a repository that is not ours to trust.

A repository's own files can name programs for git to run: ``core.fsmonitor`` in
``.git/config`` is run by ``git status``, a ``filter.<name>.clean`` command is run to
compare a file whose stat data changed with what the index holds, and a
``post-index-change`` hook fires whenever git writes the index, which ``status`` does on
its own to refresh a stale one. Assembling the context runs git in the project on every
request, on a repository somebody else may have prepared, so none of those may happen.

Every git command ncc runs goes through :func:`run_git`, and a test fails if a second place
spells out ``"git"`` as a command. What it switches off:

* ``-c core.fsmonitor=false`` -- nothing is asked to watch the tree.
* ``GIT_OPTIONAL_LOCKS=0`` -- ``status`` takes no lock it could do without, and so does
  not rewrite the index, and so fires no hook.
* ``GIT_ATTR_SOURCE`` set to the empty tree -- no ``.gitattributes`` applies, so no
  attribute chooses a filter driver. A git older than 2.40 does not know the variable
  and ignores it, which leaves a configured clean filter free to run there. The id is the
  SHA-1 one: a SHA-256 repository refuses it, and git then exits non-zero, which the
  caller reads as no answer.

The cost of the last one is that an attribute which would have normalised line endings no
longer does, so a file whose only difference is that can be counted as changed.
"""

from __future__ import annotations

import os
import subprocess

#: The id of the tree with nothing in it, which git knows without it being stored.
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def run_git(root: str, *args: str, timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    """``git <args>`` run in ``root``, with output captured and nothing the repository names run.

    Raises :class:`OSError` when git cannot be run at all and
    :class:`subprocess.SubprocessError` on a timeout; a git that ran and failed is a
    result with a non-zero ``returncode``.
    """
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_ATTR_SOURCE": EMPTY_TREE}
    return subprocess.run(  # noqa: S603
        ["git", "-c", "core.fsmonitor=false", *args],  # noqa: S607
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
