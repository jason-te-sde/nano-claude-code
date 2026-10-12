"""Project instructions: NANO.md, layered from the repository root downwards.

CLAUDE.md is read when NANO.md is absent, because most repositories that
already have project instructions for an agent have them under that name and
asking people to duplicate the file is a bad first impression.

What reaches the system prompt from here is the text of files, and a prompt is read by
the model and sent to its provider on every request, so each file is held to what the
other ways a file's contents reach the model are held to:

* **Contained.** A file's real path has to be inside the boundary it belongs to: the
  sandbox root for the root's file and the directories below it, ``<home>/.nanoclaude``
  for the global one. ``NANO.md`` that is a link to a file anywhere else on the machine
  (``~/.ssh/id_rsa``, say) is a way to put that file in the prompt, so it is skipped.
* **Refused like a read.** A file inside the sandbox is asked of the policy's read verdict
  (``read_refusal``): a ``Read(...)`` deny rule, or a credentials path under another
  name (``NANO.md`` a link to ``.env``), means it is not loaded.
* **Scrubbed.** The text goes through the session's redactor before it is placed in the
  prompt, whole, before it is cut, so that a credential which straddles a cut is not left
  as a fragment nothing recognises.
* **Capped as a chain.** Each file is cut at :data:`MAX_INSTRUCTION_BYTES`, and the chain
  as a whole at :data:`MAX_INSTRUCTION_TOTAL_BYTES`.

The name that is there decides. If ``NANO.md`` is there (a link counts, whatever it leads
to) and is skipped, ``CLAUDE.md`` is not read in its place: a file the person did not mean
to load is not replaced by one they did not choose to load either.

The size caps matter more than they look. Instructions are in every request, so a 400 KB
file is not a large prompt, it is a session that cannot hold a conversation. The caps count
the bytes of a file's text; the comment line before it, the truncation marker and the
one-line note for a file that did not fit are not counted.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from nanoclaude.permissions.policy import Policy, read_refusal
from nanoclaude.permissions.redact import Redactor
from nanoclaude.permissions.sandbox import is_within
from nanoclaude.tools.base import sanitize

INSTRUCTION_FILENAMES = ("NANO.md", "CLAUDE.md")
MAX_INSTRUCTION_BYTES = 32_000
MAX_INSTRUCTION_TOTAL_BYTES = 64_000
#: Read past the per-file cap by this much, so that a credential which straddles it is
#: whole when the text is scrubbed, and cut afterwards.
_READ_SLACK_BYTES = 4_096


@dataclass(frozen=True, slots=True)
class _Instruction:
    """One instruction file that passed every check, as far as it was read."""

    path: Path
    #: Scrubbed and stripped, and not cut to any cap.
    text: str
    #: True when the file goes on past what was read.
    longer: bool


def _is_there(candidate: Path) -> bool:
    """Whether ``candidate`` is the file the directory has under this name.

    A link is there whatever it leads to, a dangling one too: where it leads is the
    question the containment check is for, and a link that leads out is not made to
    look absent so that the next name can take its place.
    """
    return candidate.is_symlink() or candidate.is_file()


def _read_one(
    directory: Path, *, within: Path, policy: Policy, redactor: Redactor
) -> _Instruction | None:
    """The instruction file of ``directory``, or None if it has none or it may not be loaded.

    ``within`` is the boundary the file's real path has to be inside. It is a parameter and
    not the directory itself because a chain reaches directories whose boundary is further
    up (the root's, the repository's top level, the global directory's own).
    """
    for name in INSTRUCTION_FILENAMES:
        candidate = directory / name
        if not _is_there(candidate):
            continue
        real = os.path.realpath(candidate)
        if not is_within(os.path.realpath(within), real):
            return None
        # What the policy says of a file in the project. A file outside the sandbox that
        # passed its own boundary is the global one, which the sandbox has no say over.
        if policy.sandbox.contains(real) and read_refusal(policy, str(candidate), real):
            return None
        try:
            if not stat.S_ISREG(os.lstat(real).st_mode):
                return None
            with Path(real).open("rb") as handle:
                data = handle.read(MAX_INSTRUCTION_BYTES + _READ_SLACK_BYTES + 1)
        except OSError:
            return None
        longer = len(data) > MAX_INSTRUCTION_BYTES + _READ_SLACK_BYTES
        text = data[: MAX_INSTRUCTION_BYTES + _READ_SLACK_BYTES].decode("utf-8", errors="replace")
        scrubbed, _ = redactor.scrub(text)
        return _Instruction(candidate, scrubbed.strip(), longer)
    return None


def _fit(item: _Instruction, limit: int) -> tuple[str, int]:
    """The text of ``item`` cut to ``limit`` bytes with the marker when it did not fit whole,
    and how many of its bytes that is."""
    encoded = item.text.encode("utf-8")
    if len(encoded) <= limit and not item.longer:
        return item.text, len(encoded)
    kept = encoded[:limit].decode("utf-8", errors="ignore")
    return f"{kept}\n\n[truncated at {limit} bytes]".strip(), len(kept.encode("utf-8"))


def _one_line(text: str) -> str:
    return " ".join(sanitize(text).split())


def load_instructions(
    root: str, *, cwd: str, home: str | None, policy: Policy, redactor: Redactor
) -> str:
    """Global, then root, then each directory down to ``cwd``. Later wins.

    ``policy`` and ``redactor`` are the session's, and there is no default for either: a
    loader that can be called without them is a way past every refusal and every scrub the
    other ways a file reaches the model have.

    The global file and the root's are always loaded, each cut at
    :data:`MAX_INSTRUCTION_BYTES`. Every other file gets what is left of
    :data:`MAX_INSTRUCTION_TOTAL_BYTES`, the one nearest the root first: whole if it fits,
    cut with the marker if it does not, and replaced by one line saying so if nothing is left.
    """
    root_path, cwd_path = Path(root).resolve(), Path(cwd).resolve()
    chain = [root_path]
    if cwd_path != root_path and root_path in cwd_path.parents:
        relative = cwd_path.relative_to(root_path)
        current = root_path
        for part in relative.parts:
            current = current / part
            chain.append(current)

    # (directory, boundary, always loaded), in the order they appear in the prompt.
    sources: list[tuple[Path, Path, bool]] = []
    if home:
        global_directory = Path(home) / ".nanoclaude"
        sources.append((global_directory, global_directory, True))
    sources.append((root_path, root_path, True))
    sources.extend((directory, root_path, False) for directory in chain[1:])

    items = [
        (_read_one(directory, within=within, policy=policy, redactor=redactor), always)
        for directory, within, always in sources
    ]
    left = MAX_INSTRUCTION_TOTAL_BYTES
    for item, always in items:
        if item is not None and always:
            left -= min(len(item.text.encode("utf-8")), MAX_INSTRUCTION_BYTES)

    parts: list[str] = []
    for item, always in items:
        if item is None:
            continue
        allowance = MAX_INSTRUCTION_BYTES if always else min(MAX_INSTRUCTION_BYTES, max(left, 0))
        if not always and allowance == 0 and item.text:
            parts.append(
                f"[{_one_line(str(item.path))} not loaded: instructions are capped at "
                f"{MAX_INSTRUCTION_TOTAL_BYTES} bytes in total]"
            )
            continue
        text, used = _fit(item, allowance)
        if not always:
            left -= used
        parts.append(f"<!-- {item.path} -->\n{text}")
    return "\n\n".join(parts)
