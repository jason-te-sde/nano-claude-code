# Contributing

## Prerequisites

Python 3.11 or newer. macOS or Linux — Windows is not supported (see the README).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,shell,openai]"
```

## Before you push

```bash
scripts/preflight.sh
```

This runs everything CI runs, in CI's order: the hygiene check, `ruff check`, `ruff format
--check`, `mypy`, then the test suite with coverage. CI additionally runs the test suite across
a matrix of Python 3.11 and 3.13 on both Ubuntu and macOS, so a pass locally on one interpreter
is not a guarantee — it is the fast check you run before pushing, not a substitute for CI.

## Workflow

1. Branch from `main`.
2. Commit in small, reviewable steps.
3. Open a pull request. CI must be green before it merges.
4. Squash merge, so `main` keeps one commit per change.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/), with the module as the scope:

```
<type>(<scope>): <imperative summary, no trailing period>

<body: why this change, not just what it does>
```

Scopes are the packages under `src/nanoclaude/`: `cli`, `agent`, `conversation`, `tools`,
`permissions`, `providers`, `context`, `checkpoint`, `mcp`, `hooks`, `config`. Use `build` for
packaging, CI and tooling changes that are not specific to one module, or omit the scope
entirely for a change that spans the whole repository. Common types: `feat`, `fix`, `refactor`,
`test`, `docs`, `ci`, `build`, `chore`.

Commit messages and everything else in this repository are English only — see Hygiene below.

## Tests

Every behavioural change needs a test that fails without it. The suite has seven layers, per the
design spec's own testing section; put a new test in the cheapest layer that can catch its bug.

| Layer | What it covers | Speed |
| --- | --- | --- |
| 1. Unit | Pure functions and algorithms — the edit algorithm, permission decisions, dangerous-command classification, compaction, budget packing, secret redaction — table-driven and with `hypothesis` | milliseconds |
| 2. Tool | A tool against a real temporary directory, a real subprocess, or a real git fixture repository | seconds |
| 3. Adapter | Recorded-cassette replay for each provider adapter, plus an assertion that request bodies match that provider's documented shape. Cassettes are recorded once against a real key by `scripts/record-cassettes.py`, redacted, and committed; each provider covers at least a plain-text reply, a single tool call, parallel tool calls, a streaming interruption, and a 429 | seconds |
| 4. Session | A scripted fake model driving the whole agent loop, including adversarial scripts: an edit without reading first, a path escape, malformed arguments, a 10MB output | seconds |
| 5. End-to-end | `ncc -p` against a fixture repository, asserting the exit code and the shape of its JSON output | seconds |
| 6. Differential | The regex and AST dangerous-command classifiers, run side by side over the same few hundred commands; every disagreement is either a documented gap or a fixed bug | seconds |
| 7. Bench | Real models, scored by whether a fixture repository's own test suite passes, not by human or LLM judgment. Not part of regular CI — a nightly job runs only the local-Ollama rows | minutes |

Today, only two of these have any code behind them: a handful of unit-style tests (the hygiene
checks) and the CLI smoke test. Tool, adapter, session, end-to-end, differential and bench all
arrive with the subsystem they exercise — a provider adapter needs to exist before it has
cassettes, a dangerous-command classifier before it has a differential corpus, and so on.

Rules that matter in practice:

- Tests must not sleep to wait for something to happen. Drive a fake clock, or poll a condition
  with a deadline, so the suite behaves the same on a loaded CI runner as on a laptop.
- Anything randomized takes a seed and prints it on failure.
- An assertion failure should identify the state, not just the expected value.

## Hygiene

Two rules are enforced by `scripts/check-hygiene.sh` — run by `preflight.sh` and by CI's
`hygiene` job — on every tracked file and the full commit history, from the first commit
onward:

1. **No Chinese characters anywhere in a tracked file.** This project's design discussions live
   outside the repository; only the English implementation lives inside it.
2. **No AI attribution anywhere** — not in a commit message, not in a code comment, not in
   documentation. This is the author's own work. The check looks for attribution *shapes*: a
   commit trailer crediting a model, a line claiming a change was produced by one, and similar
   constructions — not the word "Claude" on its own. The project is named nano-claude-code,
   `NOTICE` carries a required trademark statement, and `providers/anthropic.py` is a real
   module name; all three must keep passing.
