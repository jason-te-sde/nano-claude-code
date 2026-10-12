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
  the protected paths, the credentials rule or your `deny` rules.
- A rule id, `sandbox.protected-path`, that refuses a write into a `.git` or a `.nanoclaude`
  directory in every mode, since git runs what `.git/config` and `.git/hooks` name and `ncc` obeys
  what `.nanoclaude` holds.
- Rules that match case-folded, NFC-normalized spellings where they refuse: `deny` rules, the
  credentials list and the protected-path row. `allow` and `ask` rules match as written. A rule
  that names a tool that does not exist is refused when the file is loaded; `Bash` and `Git` are
  reserved names, for the tools that are not built.
- Three provider adapters (Anthropic, OpenAI-compatible, Ollama), capability negotiation, a text
  protocol for models that cannot call tools, one retry policy, cost accounting with a dated
  price table, and a conservative default for a model nobody has described.
- A price book that prices by prompt length. Claude Haiku 5.5 costs $0.10 per million input
  tokens and $0.50 per million output tokens for a call whose prompt is up to 100,000 tokens, and
  $0.50 and $2.50 above that. A price row that lacks a price is refused, by name.
- Roles, with `main` and `compact` sent requests so far, `--model` and `--role`, and a check when
  a session starts that the key of every role's model is set, which names the first role that
  lacks one.
- Compaction (old tool results shrunk at 70% of the room, older history summarised at 85%), and
  SQLite sessions with `--continue` and `--resume`, a message archive and an audit record that
  only grows: the database refuses a delete, a replacement and any update but an outcome set once,
  and a call cancelled at its confirmation ends `cancelled`.
- Credentials kept out of the transcript: a credentials list that is refused by path, in any case,
  for `Read`, `Grep`, `@` mentions, the project map and the instruction files; private-key blocks
  masked whole wherever `Read`, `Grep` and the diff `Edit` returns show a file; and other keys and
  tokens scrubbed by shape.
- `Read` `deny` rules that hold for every way a file reaches the model, through one question to the
  policy, and a search that stays in the project: neither backend of `Grep` follows a symlink, a
  refused file is dropped before the result limit is applied, and `Glob` does not list a symlink
  that leads outside the sandbox.
- Git run from one place and only as `git rev-parse`, so that a repository cannot make `ncc` run a
  program through it. The environment block names the branch.
- Instruction files layered from the top level of the repository down to the project root, each
  contained in its boundary, checked as a `Read` of it would be, scrubbed and cut at 32,000 bytes,
  with 64,000 bytes for the chain, shared nearest the project root first.
- What `ncc` creates under its home made readable by you alone from the moment it exists, a
  session store, history or capability cache that an older version left open narrowed when it
  is opened, and a `config.toml` that is there never changed. `ncc` run in `/`, in a home
  directory or in a directory that holds one is refused unless `--root` names it.
- Configuration in two files, with the project's file treated as untrusted: it may add `deny`
  and `ask` rules and choose models, and may not add `allow` rules, name a `base_url` or an
  `api_key_env`, re-point one of your models or raise the turn, timeout and output limits.
  `bash_timeout_s`, `output_cap_bytes` and the `Bash` and `Git` entries in the rule lists are
  reserved for tools this version does not have, and the configuration `ncc init` writes says so.
- The `ncc` command: an interactive REPL with history, completion, vi keys on request and
  `/help`, `/clear`, `/compact`, `/status`, `/cost`, `/model`, `/mode`, `/tools`, `/resume`,
  `/export`, `/init` and `/exit`; `-p` for one prompt with text or JSON output and the exit codes
  0, 1, 2, 3, 4, 130 and 141; and `ncc init`, which writes the configuration and then checks the
  key.
- Under `-p`, a line on stderr for each call that was declined because nobody could be asked,
  naming the rule that asked. A call the policy refuses outright has none: the model receives the
  refusal as the tool's result.
- With `--output-format json`, exactly one object on every exit, a failure included, with
  `is_error`, `stop_reason`, `error` and `exit_code` beside the keys of a finished run.
- `--mode` offers `default`, `plan` and `accept-edits`. `bypass` is reached only with
  `--dangerously-skip-permissions`, and the error for `--mode bypass` names that flag. Every flag
  has help text.
- A system prompt that tells the model to run the tests only when one of its tools runs commands.
- Documentation: a configuration reference, an architecture overview, a testing document with a
  section on what the tests do not cover, and a design note for each decision. `scripts/measure.py` is the
  one source of the figures the README quotes.
- A hygiene check, run in CI from the first commit, for non-English characters and for
  attribution, and a test that fails when a tracked file holds a credential-shaped literal.

### Removed

- The `shell` and `openai` extras, since nothing imports what they installed. CI and the
  contributing guide install `.[dev]`; a `shell` extra comes back with the syntax-tree classifier.

### Known gaps

- A retry of a provider request is shown and not stored.
- A small `compact` model can fail to summarise a long history; the conversation is left as it
  was, and the error names the role.
- `Glob` lists the names of files that the credentials list and `Read` `deny` rules refuse, and
  the project map lists a directory whose contents are refused by its name.
- The limits of the defences that keep credentials out are in `SECURITY.md`: what `Grep` says
  about a private-key block, an approved edit to the first line of one, short and plain values in
  assignments, a rule's glob that is also matched against the absolute path without its leading
  slash (a known issue, to be fixed), and a line number that `Read` and `Grep` can count
  differently.
- The roles `explore`, `plan`, `verify` and `title` are accepted and no request is sent to them; routing them is v0.3 scope, with sub-agents.
- Models that are not in the capability table, other than local ones that can be probed, use the
  text protocol until `native_tools` is set; the REPL says so when it starts.
- Every provider cassette in the tests is hand-written; none was recorded from a real server.
