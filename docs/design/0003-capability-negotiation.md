# 0003: Degrade against a descriptor, never branch on a provider name

Status: accepted, partly implemented. The costs say which part.

Providers differ in whether a model can call tools natively, how large its window is, how much it can write in one reply and how it caches a prompt. The code keeps one record of this, `Capabilities`, a frozen dataclass with seven fields: `native_tools`, `parallel_tools`, `cache`, `context_window`, `max_output`, `reasoning` and `vision`. Anything that has to behave differently for one model than for another asks that record, and not the name of the provider.

## Why

The alternative is `if provider == "anthropic"` spread through a session, and every new provider then touches every one of those places. With a descriptor, a model that gains or loses a feature is a change to one row of one table, and the code that degrades has one thing to read.

Where the record comes from, in order: a table of models the project knows (an exact match, then a prefix rule for a family), then what an earlier run learned and kept in `capabilities.json`, then a probe, then the conservative default. The probe exists for Ollama only: `/api/show` returns the model's template, and whether the template mentions tools says whether it can call them. The conservative default assumes the least: no native tool calls, no parallel calls, no caching, an 8,192-token window and 2,048 tokens of output. A model wrongly believed to do more fails confusingly in the middle of a task; one wrongly believed to do less is merely slower, so the default errs towards less. `native_tools` and `context_window` in a model's configuration entry override whatever was found. A probe that could not find out (the server was not up) is not remembered, so the next run asks again, and `ncc init` clears the cache.

Pinned by `test_unknown_model_falls_back_to_the_conservative_default`, `test_conservative_default_never_assumes_capabilities_it_has_not_seen`, `test_what_the_config_says_beats_what_the_table_says` and `test_a_probe_that_found_nothing_out_is_not_remembered_between_sessions`.

## Costs

Three of the seven fields are consulted and four are not. The session reads `native_tools` to decide between native tool calls and the text protocol (0004), and the budget reads `context_window` and `max_output` to decide when to compact. `parallel_tools`, `cache`, `reasoning` and `vision` are recorded and nothing acts on them. The Anthropic adapter puts cache breakpoints on the system prompt and the tool list whatever the record says, the executor does not change what it does for a model that cannot call tools in parallel, and thinking blocks are sent back by the Anthropic adapter whatever the record says and by neither of the other two. They are in the record because that is where the decision will be keyed when it matters. Today they inform nothing, and a reader of the table should not take them for working switches.

No code outside the adapters, the router's client factory and the capability probe names a provider, and that holds by convention and review. The lint rule that was meant to lock it in does not exist.

A model the table does not know and that is not on Ollama is not probed, because only Ollama has a probe. An OpenAI-compatible model that is not in the table therefore gets the conservative default and the text protocol, even when the server could do native calls. The person's remedy is `native_tools = true` in that model's entry, and the documentation says so where it describes the entry.

The table goes stale. Its rows carry the date they were checked against the provider's own page, and a test refuses a row added without that line, but nothing notices a provider changing a limit.

## Rejected alternatives

- Branch on the provider name where behaviour differs. It is the least code on the first day and the most places to change on the tenth.
- Probe every unknown model with a small tool-use request. It costs a request, and on a hosted model money, the first time each is used; a failed probe cannot tell a model that cannot call tools from a server that is down. Ollama's `/api/show` answers without generating anything, which is why only it is probed.
- Assume every model supports everything and fail when it does not. It makes the first session with a small local model the one that finds out, in the middle of a task.
- A single configuration flag per feature, set by the person for every model. It puts the burden of knowing on the person, who does not know, and keeps the knowledge out of the table where it could be shared.
