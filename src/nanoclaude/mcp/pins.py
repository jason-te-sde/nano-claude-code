# src/nanoclaude/mcp/pins.py
"""What a tool looked like when the person first saw it, and the file that remembers it.

A server's tools are described to the model by text the server wrote, and that text is read
again at every start. A description that changes between two sessions changes what the model
is told a tool does, after the person has watched it do it. So the first time a tool is seen
its *fingerprint*, a SHA-256 of its description and its input schema as the model would be
given them, is written down (a *pin*); at every later start a tool whose fingerprint is not
its pin is marked changed, and a changed tool asks before every call (rule ``mcp.changed``).

The pins are ``{server: {tool: fingerprint}}`` in one file under ncc's home, ``mcp-pins.json``,
which is private from the moment it exists and is replaced as a whole, so that a write that is
interrupted leaves the old file. Tools are keyed by the name the server uses, not by the
name in the registry. A pin is never removed: a tool that goes away and comes back different
is a changed tool.

Two sessions can be open at once, each learning a different server. :meth:`PinStore.save`
therefore reads the file again and puts only its own new pins into what it finds, so that one
does not erase what the other wrote; two writes at the same instant can still lose one, and
what is lost is a pin, which makes a tool "first seen" again and not "changed".
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nanoclaude.private import write_private

#: The pins' file, inside the directory ncc keeps its state in.
PINS_FILENAME = "mcp-pins.json"

_STATE_DIRNAME = ".nanoclaude"


class PinStoreError(Exception):
    """The pins' file cannot be read as pins. Worded for a person: what, an em-dash, what to do."""


def pins_path(home: Path) -> Path:
    """Where the pins are kept: in ``.nanoclaude`` of ``home``, the directory ncc keeps it in."""
    return Path(home) / _STATE_DIRNAME / PINS_FILENAME


def fingerprint(description: str, schema: Mapping[str, Any]) -> str:
    """The SHA-256, in hex, of a tool's ``description`` and input ``schema``.

    Both are what the model is given (the description cleaned and cut, the schema cleaned),
    serialized in one canonical form: the keys sorted, no whitespace, every character
    outside ASCII escaped, so that two spellings of one definition have one fingerprint and
    a lone surrogate does not stop it being computed. The two are a JSON object's members, so
    that no text of one can be mistaken for the start of the other.
    """
    canonical = json.dumps(
        {"description": description, "inputSchema": schema},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _read(path: Path) -> dict[str, dict[str, str]]:
    """The pins in ``path``; none when it is not there; :class:`PinStoreError` when it is wrong."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except UnicodeDecodeError as exc:
        raise PinStoreError(
            f"the pins file {path} is not valid UTF-8 text — delete it to forget every pin"
        ) from exc
    except OSError as exc:
        raise PinStoreError(
            f"could not read the pins file {path} ({exc.strerror or exc}) — check that it is a "
            "readable file"
        ) from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise PinStoreError(
            f"the pins file {path} is not valid JSON ({exc}) — fix it, or delete it to forget "
            "every pin"
        ) from exc
    if not _is_pins(data):
        raise PinStoreError(
            f"the pins file {path} is not an object of servers, each an object of tool "
            "fingerprints that are strings — fix it, or delete it to forget every pin"
        )
    return {server: dict(tools) for server, tools in data.items()}


def _is_pins(data: object) -> bool:
    return isinstance(data, dict) and all(
        isinstance(server, str)
        and isinstance(tools, dict)
        and all(isinstance(tool, str) and isinstance(value, str) for tool, value in tools.items())
        for server, tools in data.items()
    )


class PinStore:
    """The pins of every server, in memory, and the file they are saved to.

    :meth:`load` reads the file; :meth:`record` notes a pin in memory only; :meth:`save`
    writes what was recorded. Nothing is read or written at construction.
    """

    def __init__(self, path: Path, pins: Mapping[str, Mapping[str, str]] | None = None) -> None:
        self._path = Path(path)
        self._pins: dict[str, dict[str, str]] = {
            server: dict(tools) for server, tools in (pins or {}).items()
        }
        #: What was recorded since the last save: the only entries a save puts in the file.
        self._unsaved: dict[tuple[str, str], str] = {}

    @classmethod
    def load(cls, path: Path) -> PinStore:
        """The store of the file at ``path``, empty when there is none."""
        return cls(path, _read(Path(path)))

    @property
    def path(self) -> Path:
        return self._path

    @property
    def unsaved(self) -> bool:
        """Whether something was recorded that :meth:`save` has not yet written."""
        return bool(self._unsaved)

    def get(self, server: str, tool: str) -> str | None:
        """The fingerprint pinned for ``tool`` of ``server``, or None if it was never seen."""
        return self._pins.get(server, {}).get(tool)

    def record(self, server: str, tool: str, pinned: str) -> None:
        """Pin ``tool`` of ``server`` to the fingerprint ``pinned``, replacing what was pinned."""
        self._pins.setdefault(server, {})[tool] = pinned
        self._unsaved[(server, tool)] = pinned

    def save(self) -> None:
        """Write what was recorded into the file, keeping every pin that is there.

        Does nothing when nothing was recorded. The file is read again first, so that a pin
        another process saved is kept, and if it can no longer be read it is not replaced
        (:class:`PinStoreError`): it is somebody's, and the person is told. A write that
        fails raises ``OSError`` and leaves the old file whole.
        """
        if not self._unsaved:
            return
        on_disk = _read(self._path)
        for (server, tool), pinned in self._unsaved.items():
            on_disk.setdefault(server, {})[tool] = pinned
        text = json.dumps(on_disk, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        write_private(self._path, text, replace=True)
        self._pins = on_disk
        self._unsaved = {}
