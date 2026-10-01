# Cassettes

The Anthropic cassettes in this directory (`anthropic_text.jsonl`,
`anthropic_tool_use.jsonl`, `anthropic_parallel_tools.jsonl`,
`anthropic_truncated.jsonl`) are **synthetic**. They were hand-written, event
for event, in the documented Messages streaming format -- the same shapes
`message_start`, `content_block_start`, `content_block_delta`,
`content_block_stop`, `message_delta`, and `message_stop` that
`nanoclaude.providers.anthropic.iter_sse` and `StreamAccumulator` parse -- not
captured from a real request. `anthropic_truncated.jsonl` was then derived by
hand from `anthropic_tool_use.jsonl`: the same opening, cut off mid
`input_json_delta`, with no closing `content_block_stop`, `message_delta`, or
`message_stop`.

Each line is one JSON object, `{"event": ..., "data": ...}`, matching what
`tests/providers/test_anthropic.py`'s `replay()` helper reads back.

`scripts/record-cassettes.py` records the real thing: run it with a live
`ANTHROPIC_API_KEY` and it overwrites these four files with actual recordings
from the API, redacted on the way out. Nobody has run it yet -- these files
are still the hand-written originals. When it is run, replace this note's
"synthetic" description of the affected files with where and when they were
recorded.
