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

<body: why this change, what was considered and rejected, what it does not do, and how far
it was verified>
```

Scopes are the packages under `src/nanoclaude/`: `cli`, `agent`, `conversation`, `tools`,
`permissions`, `providers`, `context`, `config`, and `checkpoint`, `mcp` and `hooks` once they
exist. Use `build` for packaging, CI and tooling changes that are not specific to one module,
or omit the scope entirely for a change that spans the whole repository. Common types: `feat`,
`fix`, `refactor`, `test`, `docs`, `ci`, `build`, `chore`.

Commit messages and everything else in this repository are English only — see Hygiene below.

## Tests

Every behavioural change needs a test that fails without it: write the test, watch it fail for
the reason you expect, then make it pass. Put a new test in the cheapest layer that can catch its
bug. [`docs/testing.md`](docs/testing.md) describes the layers that exist, what the helpers are,
and, in a section of its own, what the suite does not cover.

Rules that matter in practice:

- Tests must not sleep to wait for something to happen. Drive a fake clock, or poll a condition
  with a deadline, so the suite behaves the same on a loaded CI runner as on a laptop.
- Anything randomized takes a seed and prints it on failure.
- An assertion failure should identify the state, not just the expected value.
- A test that depends on something in the environment asserts that it ran. A skipped dependency
  that turns the suite green looks exactly like a pass.
- A check that loops over a set of things asserts that the set is not empty.
- Coverage is a smoke alarm and not a target.

## Documentation

The documents make claims, and `tests/test_docs.py` checks the ones a test can: the commands,
keys, defaults, exit codes, rule ids and tools a document names exist, every TOML example is
loaded by the real configuration loader, every link resolves, and the figures in the README's
numbers table are the ones `scripts/measure.py` produces.

- When a number in the README goes stale the docs test says so. Run `python scripts/measure.py`
  and copy its output into the table. Never write a figure there that the script did not print.
- A decision that a reader would otherwise have to reconstruct from the code gets a note in
  `docs/design/`, numbered next after the last, with exactly three sections: `## Why`,
  `## Costs` and `## Rejected alternatives`. The costs say where the code falls short of the
  idea. A change to a tool's description, which is a prompt, needs a note too.
- A document that says a thing is not built is checked by a test that fails when it is built.
  When one fails for that reason, the document is what to correct.
- Every hosted model id in an example is checked against the provider's own list on the day it
  is written, and the date is said.
- An example that shows `allow`, `base_url` or `api_key_env` belongs in the home file's
  example, and says so on its first line.

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
   module name; all three must keep passing. The shapes are described, and why, in
   [`docs/design/0016-attribution-policy.md`](docs/design/0016-attribution-policy.md), in words
   that do not themselves match.

The check reads tracked files only, so run `git add` on a new file before you run it. Tools that
write commit messages may add their own trailer by default; delete it before you commit.
