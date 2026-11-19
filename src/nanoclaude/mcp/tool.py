# src/nanoclaude/mcp/tool.py
"""A tool an MCP server brought, as a tool of the registry.

A server's tool list is text and names that somebody else chose, which the model and the
terminal are about to read, and which the person approved once when they first saw them. So
before a tool is registered it goes through everything here, and a tool that fails any of it
is left out with a warning, never registered quietly changed:

* **Its name** is ``mcp__<server>__<tool>``, the tool's own name with each character outside
  ``[A-Za-z0-9_]`` replaced by an underscore. Two tools of a server that come to one name are
  both dropped, and so is a name over 64 characters. The call to the server uses the name
  the server gave.
* **Its description** has terminal controls removed (tab and newline are kept) before
  anything else, and is cut to 2,048 bytes on a character boundary, with a marker.
* **Its input schema** is an object schema, at most 64 KiB serialized and 32 levels deep, with no
  ``$ref`` that leaves the schema: a reference is never followed, and one that is not
  local is refused outright.
* **Its fingerprint**, a hash of the two as the model is given them, is compared with the pin
  from an earlier session (:mod:`nanoclaude.mcp.pins`). A tool whose definition is not what it was
  is *changed* for the rest of the session: the policy asks before every call to it, in
  every mode, and answering "always" renews the pin for the next session.

:func:`adapt_tools` does this for one server's tools, from a connected client, and is what a
session calls at start; :func:`combine_tools` puts several servers' results together. A
:class:`McpTool` does no I/O when it is built: it is the one tool in the registry that
reaches a process that is not ncc, and it does so only in :meth:`McpTool.run`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from nanoclaude.config.schema import McpServerConfig
from nanoclaude.mcp.client import CallResult, McpError, McpRequestError, RemoteTool
from nanoclaude.mcp.pins import PinStore, PinStoreError, fingerprint
from nanoclaude.mcp.result import result_text, wrap_result
from nanoclaude.permissions.policy import PermissionRequest
from nanoclaude.permissions.rules import is_mcp_tool_name
from nanoclaude.providers.base import ToolSpec
from nanoclaude.tools.base import (
    ToolContext,
    ToolOutcome,
    cut_bytes,
    failed,
    ok,
    sanitize,
    without_lone_surrogates,
)

#: The most a description may be, in bytes of UTF-8, before it is cut and marked.
MAX_DESCRIPTION_BYTES = 2048

#: Written after a description that was cut.
TRUNCATED = "…(truncated)"

#: The most an input schema may be serialized, and how deeply nested.
MAX_SCHEMA_BYTES = 64 * 1024
MAX_SCHEMA_DEPTH = 32

#: The longest a registry name may be: the limit of the providers' tool names.
MAX_REGISTRY_NAME_CHARS = 64

#: The keys whose text in a schema is something the model reads as a description.
_DESCRIPTIVE_KEYS = frozenset({"description", "title"})

_NOT_A_NAME_CHARACTER = re.compile(r"[^A-Za-z0-9_]")

#: The longest a name or a value of a server's is shown in a warning.
_SHOWN_CHARS = 60


class ToolServer(Protocol):
    """What the adapter needs of a connected client: its tools, and a way to call one.

    :class:`~nanoclaude.mcp.client.McpClient` is one.
    """

    async def list_tools(self) -> list[RemoteTool]: ...

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        timeout: float | None = None,  # noqa: ASYNC109 - the client's own parameter
    ) -> CallResult: ...


# ---- what a server wrote, made fit to show


def registry_name(server: str, tool: str) -> str:
    """The name a tool of ``server`` is registered under: ``mcp__<server>__<tool>``, the tool's
    name with every character outside ``[A-Za-z0-9_]`` replaced by an underscore."""
    return f"mcp__{server}__{_NOT_A_NAME_CHARACTER.sub('_', tool)}"


def clean_description(text: str) -> str:
    """``text`` as a description the model and a terminal may be given.

    Terminal controls and ANSI sequences are removed first (tab and newline stay), and a lone
    surrogate, which cannot be written as UTF-8, becomes U+FFFD; then what is over
    :data:`MAX_DESCRIPTION_BYTES` bytes is cut, on a character boundary, and :data:`TRUNCATED`
    is written after what is left. Removing first means that controls do not use up the limit
    and that nothing the cut keeps differs from what was checked.
    """
    kept, cut = cut_bytes(without_lone_surrogates(sanitize(text)), MAX_DESCRIPTION_BYTES)
    return kept + TRUNCATED if cut else kept


def _shown(text: object) -> str:
    """A server's text as one short line for a warning: controls removed, cut, quoted."""
    line = " ".join(without_lone_surrogates(sanitize(str(text))).split())
    if len(line) > _SHOWN_CHARS:
        line = line[: _SHOWN_CHARS - 1] + "…"
    return f"'{line}'"


def _listed(names: Sequence[str]) -> str:
    quoted = [_shown(name) for name in names]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


# ---- input schemas


def schema_problem(schema: object) -> str | None:
    """Why ``schema`` may not be given to the model, or None when it may.

    It must be an object whose ``type`` is ``"object"``, at most :data:`MAX_SCHEMA_DEPTH`
    levels deep (a list is a level, as an object is), at most :data:`MAX_SCHEMA_BYTES`
    serialized, and hold no ``$ref`` whose value is not text beginning with ``#``: the
    specification forbids a client to fetch what a reference names, and a reference that
    cannot be local has no use here. The answer is a clause that starts with "its input
    schema", for a warning.

    The walk keeps its own stack and does not recurse, so that a schema nested a thousand
    levels deep is a problem to report and not a crash.
    """
    if schema is None:
        return "it has no input schema"
    if not isinstance(schema, dict):
        return "its input schema is not a JSON object"
    if schema.get("type") != "object":
        return 'its input schema does not have "type": "object"'
    stack: list[tuple[object, int]] = [(schema, 1)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, dict | list) and depth > MAX_SCHEMA_DEPTH:
            return f"its input schema is nested more than {MAX_SCHEMA_DEPTH} levels deep"
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "$ref" and not (isinstance(value, str) and value.startswith("#")):
                    return (
                        f"its input schema has a $ref that does not point inside the schema "
                        f"({_shown(value)}), and only references that begin with # are allowed"
                    )
                stack.append((value, depth + 1))
        elif isinstance(node, list):
            stack.extend((item, depth + 1) for item in node)
    size = len(
        json.dumps(schema, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8", "surrogatepass"
        )
    )
    if size > MAX_SCHEMA_BYTES:
        return f"its input schema is {size} bytes, over the limit of {MAX_SCHEMA_BYTES} bytes"
    return None


def clean_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``schema`` that is fit to give the model: the text of each ``description``
    and ``title`` has controls removed, and no string anywhere holds a lone surrogate.

    Only for a schema that :func:`schema_problem` passed, whose depth bounds the recursion.
    A property that is called ``description`` is a schema and not a string, and is left to
    the recursion.
    """

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                without_lone_surrogates(key): (
                    without_lone_surrogates(sanitize(value))
                    if key in _DESCRIPTIVE_KEYS and isinstance(value, str)
                    else clean(value)
                )
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [clean(item) for item in node]
        if isinstance(node, str):
            return without_lone_surrogates(node)
        return node

    cleaned: dict[str, Any] = clean(schema)
    return cleaned


# ---- the tool


class McpTool:
    """One tool of one server, as the registry holds it.

    Built from what :func:`adapt_tools` accepted, and does no I/O to be built. It is never
    read-only (nothing says it only reads), and its permission request is that of a call to
    somebody else's code: no path, a subject that names the server, and
    ``definition_changed`` when its pin says it is not what the person last approved.
    """

    read_only = False

    def __init__(
        self,
        client: ToolServer,
        *,
        server: McpServerConfig,
        remote_name: str,
        description: str,
        schema: Mapping[str, Any],
        pins: PinStore,
        output_cap: int,
        pinned: str | None = None,
    ) -> None:
        self._client = client
        self._server = server
        self._remote_name = remote_name
        self._description = description
        self._schema = schema
        self._pins = pins
        self._cap = output_cap
        self.name = registry_name(server.name, remote_name)
        self.fingerprint = fingerprint(description, schema)
        #: Fixed when the tool is built, for the whole session: renewing the pin by an answer of
        #: "always" is for the next session, and does not make this one any less changed.
        self.changed = pinned is not None and pinned != self.fingerprint

    @property
    def server_name(self) -> str:
        return self._server.name

    @property
    def remote_name(self) -> str:
        """The name the server gave the tool, which is the one a call to it uses."""
        return self._remote_name

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self._description, self._schema)

    def permission_request(
        self, _ctx: ToolContext, _arguments: Mapping[str, Any]
    ) -> PermissionRequest:
        # A write, so that plan mode refuses it: nothing says that a tool of somebody else's
        # does not change anything.
        return PermissionRequest(
            self.name,
            f"server {self._server.name}",
            (),
            is_write=True,
            definition_changed=self.changed,
        )

    def on_grant(self) -> None:
        """The person answered "always" to a question about this tool: renew its pin.

        Only a changed tool has anything to renew, and it is for the next session: this one
        keeps asking. A pin that cannot be written is not an error for the call that was just
        approved; the tool is then asked about again next time, which is the safe way to fail.
        """
        if not self.changed:
            return
        self._pins.record(self._server.name, self._remote_name, self.fingerprint)
        try:
            self._pins.save()
        except (OSError, PinStoreError):
            return

    async def run(
        self, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        server, tool = self._server.name, self.name.removeprefix(f"mcp__{self._server.name}__")
        try:
            result = await self._client.call_tool(
                self._remote_name, arguments, timeout=float(self._server.timeout)
            )
        except McpRequestError as exc:
            # The server's own words, in an answer that is an error: as much its text as a result.
            wrapped = wrap_result(server, tool, str(exc), cap=self._cap, redactor=ctx.redactor)
            return failed(
                call_id,
                f"The MCP server {server} answered {self.name} with an error:\n{wrapped}",
            )
        except McpError as exc:
            text, _ = ctx.redactor.scrub(sanitize(f"{self.name} failed: {exc}"))
            return failed(call_id, text)
        wrapped = wrap_result(
            server, tool, result_text(result), cap=self._cap, redactor=ctx.redactor
        )
        return failed(call_id, wrapped) if result.is_error else ok(call_id, wrapped)


# ---- adapting a server's tools


@dataclass(frozen=True, slots=True)
class AdaptedTools:
    """The tools a session may register from one or more servers, and why others were left out.

    Every warning is one line of text that is safe to print: it says which server and which
    tool, and why. Nothing here is an error: a session with no tool from a server is a session
    that works.
    """

    tools: tuple[McpTool, ...]
    warnings: tuple[str, ...]


def _by_registry_name(remote: Sequence[RemoteTool], server: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for tool in remote:
        groups.setdefault(registry_name(server, tool.name), []).append(tool.name)
    return groups


async def adapt_tools(
    client: ToolServer, server: McpServerConfig, pins: PinStore, *, output_cap: int
) -> AdaptedTools:
    """The tools of ``server`` that may be registered, from a connected ``client``.

    Lists the server's tools and checks each (see the module docstring). A tool seen for the
    first time is pinned, and the pins are saved when any was; a tool that differs from its
    pin comes back marked changed, and its pin is left alone until the person approves it
    with "always". ``output_cap`` is the session's output-cap limit, which a result is held
    to. A failure to list the tools is the client's (:class:`~nanoclaude.mcp.client.McpError`)
    and is raised; so is an unreadable pins file, from loading it.
    """
    remote = await client.list_tools()
    name = server.name
    warnings: list[str] = []
    collisions = {n: orig for n, orig in _by_registry_name(remote, name).items() if len(orig) > 1}
    for registered, originals in collisions.items():
        warnings.append(
            f"MCP server '{name}': the tools {_listed(originals)} come to the same name, "
            f"{registered}, since a character outside letters, digits and underscores becomes an "
            "underscore; none of them is registered"
        )
    accepted: list[McpTool] = []
    for tool in remote:
        registered = registry_name(name, tool.name)
        if registered in collisions:
            continue
        reason = _reason_to_drop(tool, registered)
        if reason is not None:
            warnings.append(f"MCP server '{name}': tool {_shown(tool.name)} is dropped: {reason}")
            continue
        description = clean_description(tool.description or "")
        pinned = pins.get(name, tool.name)
        built = McpTool(
            client,
            server=server,
            remote_name=tool.name,
            description=description,
            schema=clean_schema(tool.input_schema),
            pins=pins,
            output_cap=output_cap,
            pinned=pinned,
        )
        if pinned is None:
            pins.record(name, tool.name, built.fingerprint)
        accepted.append(built)
    try:
        pins.save()
    except (OSError, PinStoreError) as exc:
        reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
        warnings.append(
            f"MCP server '{name}': could not write {pins.path} ({reason}); the definitions of "
            "its tools are not remembered, so a change to one will not be noticed next time"
        )
    return AdaptedTools(tuple(accepted), tuple(warnings))


def _reason_to_drop(tool: RemoteTool, registered: str) -> str | None:
    """Why ``tool`` is not registered, or None when it is."""
    if not is_mcp_tool_name(registered):
        return "it has no name to build a registry name from"
    if len(registered) > MAX_REGISTRY_NAME_CHARS:
        return (
            f"its registry name is {len(registered)} characters, over the limit of "
            f"{MAX_REGISTRY_NAME_CHARS}"
        )
    if tool.description is not None and not isinstance(tool.description, str):
        return "its description is not text"
    return schema_problem(tool.input_schema)


def combine_tools(results: Sequence[AdaptedTools]) -> AdaptedTools:
    """The tools and warnings of several servers together, with a name shared by two servers
    dropped from both.

    Names are made of the server's name and the tool's, joined by ``__``, and both may hold
    ``__``: the tool ``b__t`` of server ``a`` and the tool ``t`` of server ``a__b`` are the
    same name, which the registry would refuse to hold twice. A name that only one tool of
    all of them comes to is kept as it is.
    """
    groups: dict[str, list[McpTool]] = {}
    for result in results:
        for tool in result.tools:
            groups.setdefault(tool.name, []).append(tool)
    warnings = [warning for result in results for warning in result.warnings]
    for name, same in groups.items():
        if len(same) > 1:
            servers = sorted({tool.server_name for tool in same})
            warnings.append(
                f"MCP servers {_listed(servers)} have tools that come to the same name, {name}; "
                "none of them is registered"
            )
    kept = tuple(tool for result in results for tool in result.tools if len(groups[tool.name]) == 1)
    return AdaptedTools(kept, tuple(warnings))
