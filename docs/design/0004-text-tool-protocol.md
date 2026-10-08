# 0004: Tool calling for models that cannot call tools

Status: accepted, implemented in `providers/texttools.py` and `agent/session.py`.

A model whose capability record says `native_tools` is false is not sent tool definitions as a parameter. The definitions go into the system prompt, with a grammar to answer in:

```
<tool name="Read">
{"path": "src/main.py", "limit": 50}
</tool>
```

The reply is parsed back into the same blocks a native call would have produced, and nothing above `texttools.py` knows the difference.

## Why

Most local models cannot call functions, and a coding agent that works only with those that can leaves out most of the models people run on their own machines. The grammar is XML-like and not a JSON envelope because a model that cannot be trusted to emit a well-formed function call cannot be trusted to emit a well-formed envelope, and a missing closing tag is easier to detect than a missing brace.

What the parser tolerates, and what it reports. Code fences around a block are stripped, and a closing tag in capitals closes the block. Four things are reported back to the model as a complaint: arguments that are not valid JSON, arguments that are JSON but not an object, a block with no closing tag, and an opening tag in the wrong form. A reply that holds complaints and no usable call is asked for again, with the complaints as the next message, and this does not use up a turn. It is asked again twice at most (`MAX_PARSE_RETRIES`). After that the session stops with the reason `model_unsuitable` and tells the person which model failed and to choose one with native tool calling. A reply with one good call and one bad one runs the good one and keeps the complaint in the reply text, so the model learns the other did not run.

Pinned by `test_a_reply_of_only_complaints_is_retried_at_most_the_limit_and_then_stops`, `test_asking_again_does_not_use_up_turns`, `test_a_model_that_never_writes_a_valid_call_is_given_up_on_by_name`, `test_a_malformed_call_beside_a_valid_one_is_reported_not_dropped` and, for the cost of scanning a hostile reply, `test_a_long_run_of_unclosed_tags_is_reported_once_and_scanned_once`.

## Costs

It is the least reliable path in the project. Small models write malformed JSON, forget the closing tag and wrap everything in fences, and the parser forgives two of those three. How often a given model manages is not known: the table that would say so comes with `ncc bench`, which is not built, and nothing in this repository has measured a success rate.

Every request carries the definitions of every tool in the system prompt, and the budget counts them, so a model with a small window pays for them on every turn.

The reply of a model on this path is not shown while it arrives, since a half-written tag cannot be shown as prose and parsed later. The person sees it once it is whole.

There is no start-up banner saying that a model is in text-tool mode; the only sign is that the system prompt names the protocol, and a session that is failing is told which model to replace.

Which models get this path is a consequence of 0003: any model the table does not know and the probe could not vouch for. That includes every OpenAI-compatible model that is not in the table, whether or not its server could call tools, until `native_tools = true` is set for it.

## Rejected alternatives

- Refuse models that cannot call tools. It is the simplest and most defensible, and it removes the project's reason to exist for the people running small local models.
- A JSON envelope for the whole reply. A model that cannot be relied on to write one object cannot be relied on to write the envelope around it, and a broken brace is harder to find than a broken tag.
- Constrained decoding, where the server only samples tokens that fit a grammar. It works on the servers that offer it and on none of the rest, and it would turn the one OpenAI-compatible adapter into several.
- Taking the first malformed call as an error and stopping. The model has no way to know that a call vanished, and a person watching sees a session that stopped for no reason; a complaint fed back lets most models correct themselves on the next reply.
- Retrying without a limit. A model that has failed three times in a row is not going to succeed on the fourth, and each attempt is billed.
