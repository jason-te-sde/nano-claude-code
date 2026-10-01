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

The openai_compat cassettes in this directory (`openai_text.jsonl`,
`openai_tool_use.jsonl`) are **also synthetic**, hand-written the same way: one
JSON value per line, in the documented chat-completions streaming shape
(`choices[].delta.content`, `delta.tool_calls[]` with `index`/`id`/
`function.name`/incrementally streamed `function.arguments`, `finish_reason`,
a final `usage` chunk) that `nanoclaude.providers.openai_compat.ChunkAccumulator`
parses. The one difference from the Anthropic files: there is no `event`
field to wrap, since a chat-completions chunk never carries one, so each line
is just the chunk object itself. The last line of each file is the JSON
string `"[DONE]"`, standing in for the literal `data: [DONE]` sentinel real
servers send to close the stream; `tests/providers/test_openai_compat.py`'s
`replay()` helper skips it, and its `wire_body()` helper turns it back into
the bare, unquoted `data: [DONE]` line when reconstructing real wire text for
`OpenAICompatClient.complete()`.

`scripts/record-cassettes.py` records the real thing: run it with a live
`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` and it overwrites the matching files
with actual recordings from the respective API, redacted on the way out.
Nobody has run it yet -- all six files here are still the hand-written
originals. When it is run for a given provider, replace this note's
"synthetic" description of the affected files with where and when they were
recorded.
