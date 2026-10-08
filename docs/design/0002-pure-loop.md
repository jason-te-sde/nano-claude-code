# 0002: The loop is a state machine with no I/O

Status: accepted, implemented in `agent/loop.py`.

One turn of an agent is: ask the model, take what it said, run the tools it asked for, hand the results back. `loop.py` computes what should happen next and nothing else. `start` and `resume` make a state from a prompt, `step(state, reply)` returns either `Done` or `RunTools`, and `observe(state, outcomes)` records the results of the calls. Every one of them takes a value and returns a new value.

## Why

Everything that waits, reads a clock or touches a disk lives in `Session` and `Executor`: the awaits on the provider, the retry policy, the decision to compact, the writes to the store. That is the only reason a whole conversation can be run from a scripted model with no key, no network and no clock. `nanoclaude.testing.ScriptedModel` plays back what it is told to say, and the suite uses it to play models that behave badly: one that edits a file it never read, one that walks out of the sandbox, one that sends arguments that do not match the schema. A recording of a well-behaved model contains none of these, and they are what the safety rules exist for.

The rule is enforced and not only stated. `tests/test_boundaries.py` parses `loop.py` and fails if it imports `os`, `subprocess`, `time`, `random`, `uuid`, `asyncio`, `sqlite3` or `httpx`. The list of modules held to this is one entry long, on purpose: `executor.py` needs `asyncio` to run tools together and `time` to time them, so the rule is `loop.py`'s and not the package's. A test pins that the list can never quietly become empty and pass having checked nothing.

Two invariants live in the state machine because they have to hold whatever drives it. Every tool call the model makes is answered exactly once, in the order requested, or `observe` raises (`test_results_must_answer_every_pending_call_in_order`). And when the turn limit is reached, `step` answers the calls it will not run with a result saying so, and ends on text, because a transcript that ends with an unanswered call is rejected by every provider on the next request (`test_turn_limit_closes_the_transcript_rather_than_leaving_calls_unanswered`).

## Costs

The decision about what happens next is in two places. A reader who wants to know why a turn ended has to read `loop.py` for the normal stops and `session.py` for the stops that need I/O: the text-protocol retries, a context overflow, a stream that broke. `Session._drive` is a larger loop than the pure one it wraps.

"Pure" covers one module, not the `agent/` package, and the documentation has to keep saying so. The executor, the router and the session all do I/O.

A new stop reason is a new member of an enumeration, and the places that handle one are `match` statements that end in `assert_never`, so the type checker names each one that was missed. That protection is only as good as the habit of writing them that way.

## Rejected alternatives

- A loop that awaits the model itself. It is the shortest code, and it makes every test of turn logic a test of a fake network, with retries, compaction and cancellation interleaved with it.
- A class hierarchy with a hook for each step of the turn. It spreads the turn across overrides, and what a turn does next could then depend on which subclass is running.
- Making the whole of `agent/` pure. The executor genuinely needs concurrency and a clock; forcing it through a port would add a layer to hide two imports.
- A message bus or actor model between the loop and the tools. Nothing here is distributed, and a bus adds ordering questions the direct calls do not have.
