"""How much room is left, and what to do about it.

Thresholds are fractions of the model's own window rather than a fixed number
of tokens, because the same session runs against a 200k model and an 8k one
and a constant would be wrong for both.

The fourth verdict is the one that matters. If the system prompt, tools and
project map alone do not fit, compaction cannot help -- there is no
conversation to compact -- and saying "compacting..." in a loop is the worst
possible behaviour. The error names the fixed part that is too big and the
flag that changes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from nanoclaude.conversation.transcript import (
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Transcript,
)
from nanoclaude.providers.capabilities import Capabilities

Verdict = Literal["ok", "micro", "full", "impossible"]

#: Averaged over source code and English prose. Used only for budget decisions;
#: the authoritative count comes back in the provider's response.
CHARS_PER_TOKEN = 3.5


class ContextTooSmallError(RuntimeError):
    """The fixed part of the request does not fit in the model's window."""


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicCounter:
    def count(self, text: str) -> int:
        return estimate_tokens(text)


def transcript_text(transcript: Transcript) -> str:
    parts: list[str] = []
    for message in transcript.messages:
        for block in message.blocks:
            if isinstance(block, (TextBlock, ThinkingBlock)):
                parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                parts.append(f"{block.name}{block.arguments}")
            elif isinstance(block, ToolResultBlock):
                parts.append(block.content)
    return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class Budget:
    capabilities: Capabilities
    soft: float = 0.70
    hard: float = 0.85
    reserve_fraction: float = 0.10
    # HeuristicCounter has no fields and no mutable state, so every Budget
    # that does not pass its own counter sharing this one instance is safe --
    # unlike the list/dict/set defaults this rule exists to catch. A
    # default_factory would instead hand each Budget its own instance, which
    # stops two default-constructed Budgets from comparing or hashing equal
    # (HeuristicCounter has no __eq__, so equality falls back to identity)
    # for no behavioural benefit.
    counter: TokenCounter = HeuristicCounter()  # noqa: RUF009

    def available(self) -> int:
        window = self.capabilities.context_window
        return int(window - self.capabilities.max_output - window * self.reserve_fraction)

    def fixed_cost(self, system: str, tools: str) -> int:
        return self.counter.count(system) + self.counter.count(tools)

    def used(self, transcript: Transcript, system: str, tools: str) -> int:
        return self.fixed_cost(system, tools) + self.counter.count(transcript_text(transcript))

    def pressure(self, transcript: Transcript, system: str, tools: str) -> float:
        available = self.available()
        return 0.0 if available <= 0 else self.used(transcript, system, tools) / available

    def verdict(self, transcript: Transcript, system: str, tools: str) -> Verdict:
        available = self.available()
        if available <= 0 or self.fixed_cost(system, tools) >= available:
            return "impossible"
        ratio = self.pressure(transcript, system, tools)
        if ratio >= self.hard:
            return "full"
        if ratio >= self.soft:
            return "micro"
        return "ok"

    def require_fits(self, transcript: Transcript, system: str, tools: str) -> None:
        if self.verdict(transcript, system, tools) != "impossible":
            return
        fixed = self.fixed_cost(system, tools)
        raise ContextTooSmallError(
            f"the system prompt and tools alone need about {fixed} tokens, and this "
            f"model's window is {self.capabilities.context_window}. Compacting cannot "
            "help. Use a model with a larger window (--model), or reduce the project "
            "map depth in your config."
        )
