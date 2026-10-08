# Testing

## Running it

```
scripts/preflight.sh
```

runs everything CI runs, in CI's order: the hygiene check, `ruff check`, `ruff format --check`, `mypy` in strict mode, and the whole suite with coverage. CI runs the suite on Python 3.11 and 3.13, on Ubuntu and on macOS, and a separate job installs the package from a clean checkout, outside the source tree, to see that what was packaged includes what the code reads (the price table).

The figures the README quotes (how many tests there are, how much of the code they cover) come from one script, `python scripts/measure.py`, and a test fails if the README's figures that can be recomputed have drifted from it.

## The layers

The suite puts a test in the cheapest layer that can catch its bug. These are the layers that exist.

| Layer | What it covers | Where |
| --- | --- | --- |
| Unit | Pure functions and algorithms, table-driven, and with `hypothesis` where an invariant can be stated: the edit algorithm, permission decisions, the rule grammar, the sandbox, redaction, the budget, compaction, the text-protocol parser, pricing, the retry policy, configuration loading. | `tests/permissions`, `tests/conversation`, `tests/config`, `tests/providers`, `tests/tools/test_edit_algorithm.py` |
| Tool | A tool against a real temporary directory, including symlinks, files that change between a read and an edit, and a named pipe where a file was expected. | `tests/tools` |
| Adapter | Each adapter fed streams in its provider's documented format over `httpx.MockTransport`, with assertions that the request body has the shape the provider documents, and that a stream which breaks midway is kept and not retried. | `tests/providers`, `tests/cassettes` |
| Session | A scripted model drives the whole agent loop with the real router, policy, sandbox, executor and SQLite store, including scripts that behave badly: an edit before a read, a path that leaves the sandbox, arguments that do not match the schema, a huge output, an interrupted turn, a conversation that overflows. | `tests/agent`, `tests/testing` |
| End to end | `ncc` run through its own `main()`, and in a few cases as a subprocess, against a fixture project and a fake provider, asserting the exit code, what is on stdout and what is on stderr. A few tests run the prompt and `ncc init` on a pseudo-terminal, to see the keys and the hidden key entry as a terminal sees them. | `tests/cli` |
| Docs and repository | The hygiene check, the import boundaries, the documentation, and `scripts/measure.py`. | `tests/test_hygiene.py`, `tests/test_boundaries.py`, `tests/test_docs.py`, `tests/test_measure.py` |

Two layers that the design describes do not exist yet: the differential layer, which runs the regex and syntax-tree danger classifiers over the same corpus of commands and checks every disagreement, and the benchmark layer, which runs real models against a fixture repository and scores them by that repository's own tests. Neither can exist before the shell tool and the syntax-tree classifier do.

## What the helpers are

- `nanoclaude.testing.ScriptedModel` says what it is told to say, with the usage and the stop reason it is told, and can raise the errors a provider raises. `nanoclaude.testing.session.build_session` wires one into a real session. Both are shipped in the package so that someone writing a tool can use them.
- The cassettes in `tests/cassettes` are streams in each provider's documented format. They are hand-written. See the next section for what that means.
- `tests/cli/screen.py` lays what Rich wrote out as a terminal would show it, so that a streaming test can say what is on the screen and not what bytes were written, and `tests/cli/helpers.py` holds the doubles the CLI tests share.

## Conventions

The rules are in [CONTRIBUTING.md](../CONTRIBUTING.md): a change in behaviour comes with a test that fails without it; a test is named for the behaviour, as a sentence; an assertion says the state it found and not only the one it wanted; a test does not sleep to wait for something that can be polled or driven by a clock; anything random takes a seed and prints it.

Coverage is a smoke alarm and not a target. A figure near 100 says which lines ran and nothing about whether anything was checked about them. The suite is not mutation-tested in CI: whether a test would notice the defect it guards against is checked by hand when the test is written, by making the defect and watching the test fail, and that is a practice and not a record.

## What these tests do not cover

A document about testing that lists only strengths is marketing. These are the places where a green suite says little or nothing.

- **Real provider behaviour.** No test sends a request to a real provider. The eight cassettes (four Anthropic, two OpenAI-compatible, two Ollama) are hand-written, event by event, in each provider's documented streaming format; none was captured from a real request, and `scripts/record-cassettes.py`, which would replace them with recordings, has not been run. What a live server does beyond the documented formats (a field that is missing, a reordered event, a rate limit that arrives as a different status) is outside what has been tested. The adapters handle the differences that were found in documentation and nothing found in the field.
- **Windows.** The suite is run on Linux and macOS and the product does not support Windows.
- **Concurrent sessions against one working directory.** Two `ncc` processes in the same project are not tested together. What is tested is narrower: two processes resuming one stored session notice that the rows differ, and an edit refuses a file that changed since it was read. Nothing arbitrates edits to the same file by two live sessions.
- **A repository whose contents are hostile.** Prompt injection, a README or a file or a tool's output that tells the model to do something, is not tested as an attack. What is tested is the structure that should hold whatever the model is told: the policy never reads the conversation, a call is judged by what it touches, a path outside the sandbox is refused, credentials paths are refused and what a tool returns is scrubbed. No test puts a persuasive injection in front of a model, since the suite has no real model, and whether a model resists one is not known.
- **Models other than those scripted.** There is no model compatibility table yet; it comes with `ncc bench`. So nothing measures how well a real model uses the tools, how often a small model writes a usable call under the text protocol, or whether compaction preserves what a real model needed. The capability table says what the documentation of each provider says, and a model it does not list is not tested at all.
- **The danger classifier is only as good as its corpus, and there is no corpus yet.** The regex classifier has unit tests of the commands it was written for and a list of its known blind spots, each checked to be one. No count says how many dangerous commands it misses, because no corpus of them exists. The syntax-tree classifier that is meant to be the authority is not built. And nothing calls either classifier in a session, since the shell tool is not built; the first end-to-end test of a refused command waits for it.
- **The content layer of redaction.** The patterns are tested on the shapes they were written for and on what must not be hit. What the content layer does not see is whatever is in a form it has no pattern for, and what a person types or the model writes.
- **Retry timing.** The retry policy is tested by what it does with the limits it is given: it retries what it should, stops at the limit, caps the delay and uses the network limit for a failure with no status. The values it ships with (five attempts, three for a failed connection, a first delay of one second, a ceiling of thirty) are not asserted by any test.
- **The project map and credentials.** The map's depth, entry cap and symlink handling are tested. That it lists the names of credentials files is not, and is a known gap, not a specified behaviour.
- **Terminals.** Rendering is tested against a pseudo-terminal and against plain consoles, not against the terminal emulators people use. A terminal that handles an escape sequence unlike the ones tested is not covered.
- **Time and scale.** Long sessions, very large repositories and slow networks are not tested. A few bounds are tested on purpose: the cost of scanning a hostile reply for tool tags, a symlink loop in the project map, and a configuration file of absurd size.
- **The coverage figure itself.** It is measured by `scripts/measure.py` on one interpreter and one operating system, and the README's figure cannot be recomputed from inside the suite, so nothing checks it. CI computes coverage in each of its four cells and nothing reads or compares what it computes.
- **Whether the documentation is true.** `tests/test_docs.py` checks that what the documents name exists: tools, commands, keys, defaults, exit codes, rule ids, examples that load, and the figures that can be recomputed. It cannot check a sentence. The statements in these documents about behaviour were read from the code, and the design notes name the test that pins a property where there is one.
