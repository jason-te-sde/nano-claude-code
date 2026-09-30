"""What a model can do, and what to assume when we do not know.

Every degradation in the loop keys off this record, so the safe direction is
always "assume less". A model wrongly believed to support parallel tool calls
produces a confusing failure halfway through a task; one wrongly believed not to
just runs its calls one at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
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
