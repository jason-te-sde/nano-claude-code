"""Building a wired-up Session from a script, for tests and for examples.

Kept in the package rather than in tests/ for the same reason ScriptedModel is:
someone writing a tool wants to exercise it inside a real session, and this is
the shortest path to one.

Everything except the model is real: the router resolves capabilities and
prices the calls, the policy and sandbox are the shipped ones, the store is a
SQLite file. Nothing here reads the real home directory, waits out a retry
back-off, or reaches a network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from nanoclaude.agent.router import Router
from nanoclaude.agent.session import Session, new_session_id
from nanoclaude.agent.ui import UI, AutoApprove
from nanoclaude.config.schema import Config, LimitsConfig, ModelConfig, RolesConfig
from nanoclaude.conversation.store import Store
from nanoclaude.permissions.audit import AuditLog
from nanoclaude.permissions.policy import PermissionMode, Policy
from nanoclaude.permissions.redact import SECRET_PATH_PATTERNS
from nanoclaude.permissions.rules import RuleSet
from nanoclaude.permissions.sandbox import Sandbox
from nanoclaude.providers.base import ModelClient, ModelError, ModelReply, ModelRequest
from nanoclaude.providers.capabilities import Capabilities, CapabilityCache
from nanoclaude.providers.retry import RetryPolicy
from nanoclaude.testing.scripted import ScriptedModel
from nanoclaude.tools.registry import default_registry
from nanoclaude.tools.todo import TodoState

#: The model a session built here talks to unless it is told otherwise: priced,
#: with a table row of its own, and native tool calling.
DEFAULT_MODEL = "claude-sonnet-5"

#: The model id given to a session built with ``capabilities``. The capability
#: table has no row for it and the price book has no price, so its capabilities
#: are exactly the ones passed in and its cost is unknown.
SCRIPTED_MODEL = "scripted"


class ScriptedClient(ScriptedModel):
    """A ScriptedModel whose script may hold a ModelError, raised in place of a reply.

    That is how a test plays a provider that is overloaded, or refuses a key,
    at a chosen point in a conversation. The request is recorded either way.
    """

    def __init__(self, script: Sequence[ModelReply | ModelError]) -> None:
        super().__init__([])
        self._script = list(script)
        #: True once the router has closed this client.
        self.closed = False

    @property
    def exhausted(self) -> bool:
        return not self._script

    async def complete(self, request: ModelRequest) -> ModelReply:
        self.requests.append(request)
        if not self._script:
            raise ModelError(
                f"scripted model ran out of replies after {len(self.requests)} requests"
            )
        entry = self._script.pop(0)
        if isinstance(entry, ModelError):
            raise entry
        return entry

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class ScriptedSession(Session):
    """A Session that also holds the scripted clients it was built with."""

    #: The client every role but ``compact`` talks to, with the requests it was sent.
    model: ScriptedClient = field(kw_only=True)
    #: The client for the ``compact`` role, when it was given a script of its own.
    compact_model: ScriptedClient | None = field(default=None, kw_only=True)


def build_session(
    root: Path,
    script: Sequence[ModelReply | ModelError],
    *,
    compact_script: Sequence[ModelReply | ModelError] | None = None,
    model: str | None = None,
    compact_model: str = "claude-haiku-4-5",
    capabilities: Capabilities | None = None,
    compact_capabilities: Capabilities | None = None,
    context_window: int | None = None,
    max_turns: int = 40,
    compact_soft: float = 0.70,
    compact_hard: float = 0.85,
    keep_recent_turns: int = 3,
    ui: UI | None = None,
    retry: RetryPolicy | None = None,
    home: Path | None = None,
) -> ScriptedSession:
    """A session in ``root`` whose model replies from ``script``, in order.

    An entry that is a ModelError is raised instead of returned. A script that
    runs out raises a ModelError too, so a turn that asks for more replies than
    were written fails by name instead of hanging.

    ``compact_script`` gives the ``compact`` role a model of its own
    (``compact_model``) with its own script. Without it every role shares one.
    ``compact_capabilities`` are that model's, with the same caveat as
    ``capabilities`` below: they only take effect for a model id the table does not
    know.

    ``model`` is the id the roles are routed to. ``capabilities`` replace what the
    capability table says for it: they are read through the capability cache,
    which the table outranks, so they only take effect for a model the table has
    no row for. When they are given and ``model`` is not, the id is
    ``SCRIPTED_MODEL``. ``context_window`` is the config override of the same name,
    applied the way the loader's value is, with the output bound it implies.

    Retries wait no time unless a ``retry`` policy says otherwise, and global
    instructions are looked for under ``home`` (a directory that does not exist,
    by default) and never in the real home directory.
    """
    state_dir = root / ".nanoclaude"
    model_id = model or (SCRIPTED_MODEL if capabilities is not None else DEFAULT_MODEL)
    cache = CapabilityCache(state_dir / "capabilities.json")
    if capabilities is not None:
        cache.put("anthropic", model_id, capabilities)

    main_client = ScriptedClient(script)
    models = {"m": ModelConfig("anthropic", model_id, context_window=context_window)}
    clients: dict[str, ModelClient] = {"m": main_client}
    compact_client: ScriptedClient | None = None
    compact_alias = "m"
    if compact_script is not None:
        compact_client = ScriptedClient(compact_script)
        models["c"] = ModelConfig("anthropic", compact_model)
        if compact_capabilities is not None:
            cache.put("anthropic", compact_model, compact_capabilities)
        clients["c"] = compact_client
        compact_alias = "c"

    config = Config(
        models=models,
        roles=RolesConfig("m", "m", "m", "m", compact_alias, "m"),
        limits=LimitsConfig(
            max_turns=max_turns,
            compact_soft=compact_soft,
            compact_hard=compact_hard,
            keep_recent_turns=keep_recent_turns,
        ),
    )
    store = Store(state_dir / "sessions.db")
    store.open()
    todo = TodoState()
    return ScriptedSession(
        root=str(root),
        config=config,
        router=Router(config, cache, clients),
        registry=default_registry(todo),
        policy=Policy(
            sandbox=Sandbox((str(root),)),
            rules=RuleSet.build(
                allow=list(config.permissions.allow),
                ask=list(config.permissions.ask),
                deny=list(config.permissions.deny),
            ),
            mode=PermissionMode.DEFAULT,
            secret_paths=SECRET_PATH_PATTERNS,
        ),
        ui=ui or AutoApprove(),
        store=store,
        audit=AuditLog(store),
        session_id=new_session_id(),
        todo_state=todo,
        retry=retry or RetryPolicy(base_delay_s=0.0, max_delay_s=0.0),
        home=str(home if home is not None else state_dir / "home"),
        model=main_client,
        compact_model=compact_client,
    )
