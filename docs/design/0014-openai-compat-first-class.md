# 0014: One adapter for every OpenAI-compatible endpoint

Status: accepted, implemented in `providers/openai_compat.py`.

There are three adapters: `anthropic`, `ollama` and `openai_compat`. The third is written against the chat-completions format and not against OpenAI, and one model entry with `adapter = "openai_compat"` and a `base_url` reaches OpenAI, OpenRouter, Groq, DeepSeek, Together, Azure, vLLM, LM Studio and llama.cpp's server alike. The model entry names the variable that holds its key, so one session can use several of them with a key each.

## Why

The set of servers that speak this format is large and keeps growing, and almost none of them is worth an adapter of its own. Writing against the format gets the reach of all of them for the code of one, and a vendor's name appears in the logic in a single place: which request parameter carries the output cap. OpenAI's own hosts (`api.openai.com`, with its regional subdomains, and Azure's `openai.azure.com`) take `max_completion_tokens`, and refuse `max_tokens` for reasoning models. Every other host gets `max_tokens`, which is what most of them know. The match is on the host name, case-insensitive and exact, so a look-alike host gets the old name (`test_openai_itself_is_sent_max_completion_tokens`, `test_other_compatible_servers_keep_max_tokens`).

The adapter is careful where real servers differ from the documentation. A tool call's arguments are documented as a JSON string and are sometimes an object already, and both are read. Reasoning text arrives under three different names depending on who serves it (`reasoning_content`, `reasoning`, `thinking`) and is kept as a thinking block. A call that arrives without an id is given a random one, so that a second tool round on such a server does not collide with the first. A stream that ends without its end marker is a reply that was cut off, and is kept as one and not mistaken for a finished one.

A request sends no temperature unless one is set, and nothing sets one. Current OpenAI reasoning models refuse any value but the default, and a local server's own default is the one its model's authors tuned.

The retry policy is shared with the other two adapters and written once (`providers/retry.py`): a status of 408, 425, 429, 500, 502, 503, 504 or 529 is retried, up to five attempts in all with a delay that doubles from one second to a ceiling of thirty and is jittered; a connection that fails is retried up to three attempts in all; 401 and 403 are not retried, and are reported as a credentials problem; a 400 is not retried and the provider's own words are passed on. The wording of these is pinned in `tests/providers/test_retry.py`.

## Costs

"Compatible" is a spectrum. Servers differ in whether they stream usage, whether a given model calls tools at all, and which parameters they refuse. This adapter handles the differences found so far and no more, and every handled one was found in a document and then written down, not seen on a live server (the cassettes in this repository are hand-written).

Capability is guessed. No probe exists for these servers, so a model that is not in the capability table gets the conservative default, and with it the text protocol of 0004 and an 8,192-token window, until `native_tools` and `context_window` are set in its entry. A server that has a larger window than the table believes is merely compacted early. One that has a smaller window than the entry says is compacted too late, which the error for a conversation that does not fit explains.

The key is required. The adapter sends it as a bearer token and refuses to start without one, so a local server that wants none still needs its variable set to something.

The price book has no entry for these models, so `/cost` shows a dash and not a number for them.

## Rejected alternatives

- An adapter for each vendor. It is more code for each, and each is a place for the retry policy and the error handling to drift apart.
- An adapter for OpenAI only, with the rest left to a proxy the person runs. It puts a second program between the person and the model, and the tool's reason to exist is that one configuration line is enough.
- The vendors' own client libraries. The stream parsing, the error classification and the retry policy are shared by three adapters here and are what the tests exercise; a library would put them behind someone else's abstraction, and add a dependency for each.
- Native tool calling assumed for every compatible server. It fails in the middle of a task, and the capability record exists to avoid it.
