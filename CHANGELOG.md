# Changelog

All notable changes to this project are recorded here. There has been no release; the version in
the repository, `0.1.0.dev0`, is a development version.

## Unreleased

What is on `main`, for the first release (v0.1). Not yet built, and so not yet in a release:
the `Bash` and `Git` tools, and the syntax-tree danger classifier with its corpus.

### Added

- The agent loop as a state machine with no I/O, a session that wires it to the model, the tools
  and the store, and a two-phase executor that decides every call of a turn before it runs any.
- The tools `Read`, `Edit` (a batch of exact replacements that applies whole or not at all),
  `Write`, `Glob`, `Grep` and `TodoWrite`.
- Four permission modes (`default`, `plan`, `accept-edits`, `bypass`), `allow`, `ask` and `deny`
  rules, a sandbox that follows symlinks, and a fixed order of checks whose rule ids are recorded
  with every decision. `--dangerously-skip-permissions` skips the questions and not the sandbox,
  the credentials rule or your `deny` rules.
- Three provider adapters (Anthropic, OpenAI-compatible, Ollama), capability negotiation, a text
  protocol for models that cannot call tools, one retry policy, cost accounting with a dated
  price table, and a conservative default for a model nobody has described.
- Roles, with `main` and `compact` consulted so far, and `--model` and `--role`.
- Compaction (old tool results shrunk at 70% of the room, older history summarised at 85%), and
  SQLite sessions with `--continue` and `--resume`, a message archive and an audit record.
- Credentials kept out of the transcript: refused by path, and scrubbed by shape.
- Configuration in two files, with the project's file treated as untrusted: it may add `deny`
  and `ask` rules and choose models, and may not add `allow` rules, name a `base_url` or an
  `api_key_env`, re-point one of your models or raise the turn, timeout and output limits.
- The `ncc` command: an interactive REPL with history, completion, vi keys on request and
  `/help`, `/clear`, `/compact`, `/status`, `/cost`, `/model`, `/mode`, `/tools`, `/resume`,
  `/export`, `/init` and `/exit`; `-p` for one prompt with text or JSON output and the exit codes
  0, 1, 2, 3, 4, 130 and 141; and `ncc init`, which writes the configuration and then checks the
  key.
- Documentation: a configuration reference, an architecture overview, a testing document with a
  section on what the tests do not cover, and a design note for each decision. `scripts/measure.py` is the
  one source of the figures the README quotes.
- A hygiene check, run in CI from the first commit, for non-English characters and for
  attribution.

### Known gaps

- A retry of a provider request is shown and not stored.
- A small `compact` model can fail to summarise a long history; the conversation is left as it
  was, and the error names the role.
- The diff that `Edit` returns is not scrubbed of credentials.
- The roles `explore`, `plan`, `verify` and `title` are accepted and no request is sent to them.
- Models that are not in the capability table, other than local ones that can be probed, use the
  text protocol until `native_tools` is set.
- Every provider cassette in the tests is hand-written; none was recorded from a real server.
