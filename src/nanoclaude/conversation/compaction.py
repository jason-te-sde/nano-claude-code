"""Making room, without breaking the transcript.

Two strategies. Micro-compaction replaces the *content* of old tool results
with a one-line note, leaving every block in place, so the structure the
providers validate is untouched. Full compaction replaces a stretch of history
with a single summary message.

Three things must survive both, and each has a test:

* the last user message, because it is the actual request;
* the pairing of every tool_use with its tool_result, because the transcript
  must stay valid for every provider;
* the set of files the session has touched, because losing that is the failure
  that costs real work -- the model re-reads what it already changed and edits
  on top of its own edit.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from nanoclaude.conversation.budget import TokenCounter, transcript_text
from nanoclaude.conversation.transcript import (
    Block,
    Message,
    ToolResultBlock,
    ToolUseBlock,
    Transcript,
    user_text,
    validate,
)

SUMMARY_TEMPLATE = """\
Summarise this coding session so work can continue from the summary alone.
Use exactly these headings, and be specific -- paths, names, error text:

GOAL: what the user asked for, in one sentence.
DECISIONS: choices made and why, one per line.
CHANGED: every file modified so far, with a one-line description each.
OPEN: what is still unresolved.
FAILED: approaches already tried that did not work, so they are not retried.

Session:
"""

_PATH = re.compile(r"[\w./-]+\.[A-Za-z0-9]{1,6}")


def micro_compact(transcript: Transcript, *, keep_recent: int, counter: TokenCounter) -> Transcript:
    """Shrink the content of old tool results. Structure is untouched."""
    messages = list(transcript.messages)
    cutoff = max(0, len(messages) - keep_recent * 2)
    for index in range(cutoff):
        message = messages[index]
        if not message.tool_results():
            continue
        blocks: list[Block] = []
        for block in message.blocks:
            if isinstance(block, ToolResultBlock) and counter.count(block.content) > 50:
                first = block.content.splitlines()[0][:120] if block.content else ""
                blocks.append(
                    ToolResultBlock(
                        block.tool_use_id,
                        f"[compacted: {len(block.content)} chars] {first}",
                        block.is_error,
                    )
                )
            else:
                blocks.append(block)
        messages[index] = Message(message.role, tuple(blocks))
    compacted = Transcript(tuple(messages))
    validate(compacted)
    return compacted


async def full_compact(
    transcript: Transcript,
    *,
    keep_recent: int,
    summarise: Callable[[str], Awaitable[str]],
) -> Transcript:
    """Replace old history with one summary message, keeping recent turns intact."""
    if transcript.pending_tool_uses():
        raise ValueError("cannot compact while tool calls are unanswered")
    messages = list(transcript.messages)
    keep = keep_recent * 2
    if len(messages) <= keep + 1:
        return transcript

    head, tail = messages[:-keep], messages[-keep:]
    # The summary that will be prepended is itself a user message (index 0),
    # so for roles to keep alternating, tail[0] must land at index 1 as
    # "assistant". A naive slice can instead start tail on the user message
    # that holds the tool_result for a tool_use the slice left in head -- the
    # cut landing between a call and its result. Shift that leading message
    # back into head (it is summarised, not dropped) until tail starts with
    # "assistant", which also keeps the pair together on the head side.
    while tail and tail[0].role != "assistant":
        head.append(tail.pop(0))
    if not tail:
        return transcript

    summary = await summarise(transcript_text(Transcript(tuple(head))))
    touched = sorted(_paths_in(Transcript(tuple(head))))
    if touched:
        summary += "\nFILES SEEN: " + ", ".join(touched)

    compacted = Transcript((user_text(f"[earlier conversation, compacted]\n{summary}"), *tail))
    validate(compacted)
    return compacted


def _paths_in(transcript: Transcript) -> set[str]:
    found: set[str] = set()
    for message in transcript.messages:
        for block in message.blocks:
            if isinstance(block, ToolUseBlock):
                for value in block.arguments.values():
                    if isinstance(value, str):
                        found.update(_PATH.findall(value))
    return found
