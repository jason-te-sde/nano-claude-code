"""What a model can do, and what to assume when we do not know.

Every degradation in the loop keys off this record, so the safe direction is
always "assume less". A model wrongly believed to support parallel tool calls
produces a confusing failure halfway through a task; one wrongly believed not to
just runs its calls one at a time.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from nanoclaude.private import write_private

#: The cache's file, inside the directory ncc keeps its state in. Named once, so that the
#: command that reads the cache and the one that clears it cannot disagree about where it is.
CACHE_FILENAME = "capabilities.json"

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

# Checked 2026-10-02 against each model's page on platform.claude.com: the
# context window and the synchronous Messages API's max output.
_CURRENT_CLAUDE = Capabilities(True, True, "explicit", 1_000_000, 128_000, "thinking", True)
_HAIKU_4_5 = Capabilities(True, True, "explicit", 200_000, 64_000, "thinking", True)

_KNOWN: dict[tuple[str, str], Capabilities] = {
    ("anthropic", "claude-fable-5-1"): _CURRENT_CLAUDE,
    ("anthropic", "claude-opus-5-5"): _CURRENT_CLAUDE,
    ("anthropic", "claude-opus-5"): _CURRENT_CLAUDE,
    ("anthropic", "claude-sonnet-5-5"): _CURRENT_CLAUDE,
    ("anthropic", "claude-sonnet-5"): _CURRENT_CLAUDE,
    # The alias and the dated id it points to.
    ("anthropic", "claude-haiku-4-5"): _HAIKU_4_5,
    ("anthropic", "claude-haiku-4-5-20251001"): _HAIKU_4_5,
    # The model ncc init offers for OpenRouter. Checked 2026-10-07 against OpenRouter's public
    # model list (https://openrouter.ai/api/v1/models): tools and tool_choice are supported
    # and the entry prices cached reads, so the capabilities are the "deepseek" family's;
    # what differs is the window, 1_024_000 against the family's 64_000. Without this row the
    # model would be compacted as if its window were 64K. The listing's own output ceiling is
    # 384_000, but OpenRouter can route a request to another host that caps lower, so the
    # budget is the lower figure 64_000, which a routed host is more likely to meet.
    ("openai_compat", "deepseek/deepseek-v4-pro"): Capabilities(
        True, True, "automatic", 1_024_000, 64_000
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

    def clear(self) -> None:
        """Forget every model, so that the next run probes each one again.

        A cache that was never written is already clear. A path that cannot be removed
        raises ``OSError``, naming it: the stale answers would stay in force, and which
        file to delete by hand is for the caller to say.
        """
        self.path.unlink(missing_ok=True)

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
        # Written atomically and privately, by the one routine ncc keeps files under its
        # home with: a sibling temp file, made private as it is made and fully flushed
        # first, is swapped into place with a single rename. A reader can only ever see
        # the old complete file or the new complete file, never a half-written one from a
        # process that died mid-write. The pid qualifies the temp name so two `ncc`
        # processes racing on the same cache file write their own temp file rather than
        # clobbering each other's partial write before either gets to replace().
        write_private(self.path, json.dumps(data, indent=2, sort_keys=True), replace=True)


async def resolve_capabilities(
    adapter: str,
    model: str,
    *,
    cache: CapabilityCache,
    probe: Callable[[], Awaitable[Capabilities | None]] | None = None,
    overrides: dict[str, object] | None = None,
) -> Capabilities:
    """Table, then cache, then probe, then the conservative default.

    A probe returns None when it could not find out (the server is not up yet, the
    request timed out). That is not an answer, so it is not remembered: this call
    assumes the conservative default and the next one asks again.
    """
    known = capabilities_for(adapter, model)
    if known is not CONSERVATIVE_DEFAULT:
        return _apply(known, overrides)
    cached = cache.get(adapter, model)
    if cached is not None:
        return _apply(cached, overrides)
    if probe is None:
        return _apply(CONSERVATIVE_DEFAULT, overrides)
    probed = await probe()
    if probed is None:
        return _apply(CONSERVATIVE_DEFAULT, overrides)
    cache.put(adapter, model, probed)
    return _apply(probed, overrides)


def _apply(caps: Capabilities, overrides: dict[str, object] | None) -> Capabilities:
    """Config wins over anything discovered: the user may know better than us.

    A window cut below the model's own also bounds the output, to a quarter of
    the new window, the share CONSERVATIVE_DEFAULT keeps. The budget holds
    max_output back for the reply, so the cut would otherwise leave little or
    nothing for the conversation. A window restated or raised changes nothing
    else, and neither does one given together with its own max_output.
    """
    if not overrides:
        return caps
    kwargs = {k: v for k, v in overrides.items() if v is not None}
    applied = replace(caps, **kwargs)  # type: ignore[arg-type]
    window = kwargs.get("context_window")
    if isinstance(window, int) and window < caps.context_window and "max_output" not in kwargs:
        applied = replace(applied, max_output=min(applied.max_output, window // 4))
    return applied
