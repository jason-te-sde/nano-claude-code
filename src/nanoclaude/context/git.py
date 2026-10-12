"""Running git in a repository that is not ours to trust.

A repository's own files can name programs for git to run: ``core.fsmonitor`` in
``.git/config`` is run by ``git status``, a ``filter.<name>.clean`` command is run to
compare a file whose stat data changed with what the index holds (chosen by a
``.gitattributes`` file, by ``.git/info/attributes`` or by ``core.attributesFile``), and a
``post-index-change`` hook fires whenever git writes the index, which ``status`` does on
its own to refresh a stale one. In a partial clone ``status`` can also fetch objects over
the network. Assembling the context runs git in the project on every request, on a
repository somebody else may have prepared, so none of those may happen.

The rule that keeps them from happening is not a list of switches but a smaller surface:
the only git commands ncc runs are ``rev-parse`` forms, which read refs and the repository's
own layout and never the working tree, and :func:`run_git` raises on any other subcommand, so
that the rule is held by the code and a caller that wants ``status`` has to change this
function in a diff somebody reads. Every git command goes through it, and a test fails if a
second place spells out ``"git"`` as a command.

What it also switches off, because a rule that holds in one place should not be the only
thing between the repository and a program:

* ``-c core.fsmonitor=false`` -- nothing is asked to watch the tree.
* ``GIT_OPTIONAL_LOCKS=0`` -- a command takes no lock it could do without, and so does not
  rewrite the index, and so fires no hook.
* ``GIT_ATTR_SOURCE`` set to the empty tree -- no ``.gitattributes`` applies. A git older
  than 2.40 does not know the variable and ignores it, and it does not reach
  ``.git/info/attributes`` or ``core.attributesFile`` at all, which is why it is not what
  the rule relies on.
"""

from __future__ import annotations

import os
import subprocess

#: The id of the tree with nothing in it, which git knows without it being stored.
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


#: The one subcommand ncc runs. See the module docstring for why it is the only one.
ALLOWED_SUBCOMMAND = "rev-parse"


def run_git(root: str, *args: str, timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    """``git <args>`` run in ``root``, with output captured and nothing the repository names run.

    ``args`` must begin with ``rev-parse``; anything else, including an option placed in
    front of it, raises :class:`ValueError` before a process is started. Raises
    :class:`OSError` when git cannot be run at all and :class:`subprocess.SubprocessError`
    on a timeout; a git that ran and failed is a result with a non-zero ``returncode``.
    """
    if not args or args[0] != ALLOWED_SUBCOMMAND:
        raise ValueError(
            f"ncc runs only 'git {ALLOWED_SUBCOMMAND}' forms, which never read the working tree; "
            f"refusing 'git {' '.join(args)}'"
        )
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
