"""What a model can do, and what to assume when we do not know.

Every degradation in the loop keys off this record, so the safe direction is
always "assume less". A model wrongly believed to support parallel tool calls
produces a confusing failure halfway through a task; one wrongly believed not to
just runs its calls one at a time.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

CacheStyle = Literal["explicit", "automatic", "none"]
ReasoningStyle = Literal["none", "thinking", "opaque"]


@dataclass(frozen=True, slots=True)
class Capabilities:
    native_tools: bool
    parallel_tools: bool
    cache: CacheStyle
    context_window: int
    max_output: int
    reasoning: ReasoningStyle = "none"
    vision: bool = False


#: Used for any model not in the table below and not successfully probed.
CONSERVATIVE_DEFAULT = Capabilities(
    native_tools=False,
    parallel_tools=False,
    cache="none",
    context_window=8_192,
    max_output=2_048,
)

_KNOWN: dict[tuple[str, str], Capabilities] = {
    ("anthropic", "claude-opus-5"): Capabilities(
        True, True, "explicit", 200_000, 32_000, "thinking", True
    ),
    ("anthropic", "claude-sonnet-5"): Capabilities(
        True, True, "explicit", 200_000, 32_000, "thinking", True
    ),
    ("anthropic", "claude-haiku-4-5-20251001"): Capabilities(
        True, True, "explicit", 200_000, 16_000, "none", True
    ),
}

#: Prefix rules for families whose members all behave alike. Checked in order.
_FAMILIES: tuple[tuple[str, str, Capabilities], ...] = (
    (
        "anthropic",
        "claude-",
        Capabilities(True, True, "explicit", 200_000, 8_192, "thinking", True),
    ),
    ("openai_compat", "gpt-", Capabilities(True, True, "automatic", 128_000, 16_384, "none", True)),
    ("openai_compat", "deepseek", Capabilities(True, True, "automatic", 64_000, 8_192)),
)


def capabilities_for(adapter: str, model: str) -> Capabilities:
    exact = _KNOWN.get((adapter, model))
    if exact is not None:
        return exact
    for family_adapter, prefix, caps in _FAMILIES:
        if adapter == family_adapter and model.startswith(prefix):
            return caps
    return CONSERVATIVE_DEFAULT


class CapabilityCache:
    """Probe results, remembered between runs.

    Probing costs a round trip and, for a cold local model, can cost a load. It
    is fine to be a little stale: a model's tool support does not change without
    the model changing, and `ncc init` clears the file.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, object]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            # Missing file, unreadable file, or text that is not valid JSON at
            # all -- every one of these means "nothing cached yet", not a
            # crash. The cache is advisory; it must never take the process
            # down for being absent or stale.
            return {}
        if not isinstance(data, dict):
            # Valid JSON need not be an object: a half-written or hand-edited
            # file could hold a bare list or string. Same treatment as a
            # missing file -- empty, not an error.
            return {}
        return data

    def get(self, adapter: str, model: str) -> Capabilities | None:
        raw = self._load().get(f"{adapter}/{model}")
        if not isinstance(raw, dict):
            return None
        try:
            return Capabilities(**raw)
        except TypeError:
            # The key exists but its value is not shaped like a Capabilities
            # record -- missing fields, extra ones, or both, which a hand
            # edit or a field added in a newer version could produce. Treated
            # the same as a cache miss rather than a crash.
            return None

    def put(self, adapter: str, model: str, caps: Capabilities) -> None:
        # Read, modify, replace. The replace is atomic, so a reader never sees a
        # torn file; but two processes putting different keys at the same moment
        # can each write from a stale read, and the later one drops the earlier
        # entry. That loses a cached answer, not correctness: the next session
        # simply probes that model again.
        data = self._load()
        data[f"{adapter}/{model}"] = {
            "native_tools": caps.native_tools,
            "parallel_tools": caps.parallel_tools,
            "cache": caps.cache,
            "context_window": caps.context_window,
            "max_output": caps.max_output,
            "reasoning": caps.reasoning,
            "vision": caps.vision,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Written atomically: a sibling temp file is fully flushed first, then
        # swapped into place with a single rename. A reader can only ever see
        # the old complete file or the new complete file, never a half-written
        # one from a process that died mid-write. The pid qualifies the temp
        # name so two `ncc` processes racing on the same cache file write
        # their own temp file rather than clobbering each other's partial
        # write before either gets to replace().
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(self.path)


async def resolve_capabilities(
    adapter: str,
    model: str,
    *,
    cache: CapabilityCache,
    probe: Callable[[], Awaitable[Capabilities]] | None = None,
    overrides: dict[str, object] | None = None,
) -> Capabilities:
    """Table, then cache, then probe, then the conservative default."""
    known = capabilities_for(adapter, model)
    if known is not CONSERVATIVE_DEFAULT:
        return _apply(known, overrides)
    cached = cache.get(adapter, model)
    if cached is not None:
        return _apply(cached, overrides)
    if probe is None:
        return _apply(CONSERVATIVE_DEFAULT, overrides)
    probed = await probe()
    cache.put(adapter, model, probed)
    return _apply(probed, overrides)


def _apply(caps: Capabilities, overrides: dict[str, object] | None) -> Capabilities:
    """Config wins over anything discovered: the user may know better than us."""
    if not overrides:
        return caps
    kwargs = {k: v for k, v in overrides.items() if v is not None}
    return replace(caps, **kwargs)  # type: ignore[arg-type]
