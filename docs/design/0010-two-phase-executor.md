# 0010: Decide everything, then run

Status: accepted, implemented in `agent/executor.py`.

A model can ask for several tool calls in one reply. The executor handles them in two phases. In the first it decides every call and runs none: resolve the paths, classify any command, evaluate the policy, tell the front end and write the decision to the audit table, and for each call that needs confirming, ask the person. In the second it runs the calls that were allowed.

## Why

The person is asked about every call that needs it before anything that was asked for has run, and in the order the model asked. With one phase, a confirmation appears for the third write after the first two have landed, and a person who refuses the first finds the others already done. Deciding first means there is nothing half done behind a question.

The decision is on disk before the call runs. The audit row for a call (tool, redacted arguments, decision and the rule that made it) is written in phase one and filled in with the outcome after, so a crash leaves a row saying "this was allowed and we do not know what happened", which is the honest state (`test_the_decision_is_recorded_before_the_tool_runs`). A call that is refused or declined never runs and gets its outcome recorded at once, so a row with no outcome means exactly one thing.

Running is in request order, with reads together. Consecutive read-only calls run concurrently, up to eight at a time. A call that changes something is a barrier: everything requested before it finishes first, it runs alone, and nothing after it starts until it is done. A `Read` placed right after a `Write` in the same batch sees what the write produced (`test_a_write_immediately_followed_by_a_read_sees_the_write`), and two writes never overlap, whatever paths they name (`test_writes_to_different_paths_still_run_one_at_a_time`). A refused call between two reads does not split the group.

A grant takes effect at the next batch. Answering "always" to the first call of a batch does not turn a later call in the same batch into one that no longer asks, because that call was decided before the answer was given and phase one promises that no call's outcome depends on how a later one turns out (`test_an_always_grant_does_not_apply_within_the_same_batch`, `test_a_session_grant_from_an_earlier_batch_is_honoured_by_a_later_one`).

A bug in one tool costs one call and not the batch. An exception while running, or while a tool is being asked what a call would touch, becomes an error result for that call under the rule id `tool.internal-error`, with its text stripped of terminal control sequences and scrubbed of anything shaped like a credential; the other calls go on. Cancellation and `KeyboardInterrupt` are not exceptions in this sense and still stop the batch.

## Costs

Everything waits for the slowest question. A batch with five reads and one write asks about the write first, though the reads could have started, and the person's delay in answering is the batch's.

Phase one decides against the disk as it is, before any call has run. A call whose path or meaning depends on what an earlier call in the same batch creates is decided without that, and the model's remedy is to ask for it in the next turn.

Confirming does not cancel the rest of the batch. If the person declines the first of five calls and accepts the others, the others run, and the declined one tells the model not to look for another way to do the same thing.

Writes never run together, so a batch of independent writes is as slow as the sum of them. The design allowed for a lock per path and concurrent writes to different paths; the barrier is simpler and stricter, and the difference has not been needed.

If the person cancels with Ctrl+C, the batch stops and the results of calls that had finished are not handed back to the model: the next prompt starts after the session has closed what the interruption left open (the unanswered calls are answered with a note that they may or may not have run). That is a known gap in v0.1.

## Rejected alternatives

- Decide and run each call in turn. A confirmation can then be preceded by the side effects of the calls before it, and a refusal arrives after the damage.
- Run everything that was allowed at once, writes included, with a lock per path. It gives a speed-up on writes that is rarely wanted, and it needs the locking, and without a barrier a read after a write in the same batch can see the file as it was.
- Ask before the model's turn is even complete, as each call streams in. The arguments of a call are not known until it is whole, and a person asked about half a command is asked about nothing.
- Treat a refusal in phase one as ending the batch. The model needs an answer for every call it made, and a transcript with a call that has no result is rejected by the provider.
