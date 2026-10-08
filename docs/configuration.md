# Configuration reference

This describes `main` as it is built. Bash and Git are not built yet, so everything below about them is about what the configuration accepts for when they exist, and says so.

Every TOML example in this document is loaded by the real configuration loader in the test suite, so an example that stops being valid fails the build. An example that shows `allow`, `base_url` or `api_key_env` is always one for the home file, for the reason in the next section.

## Where things live

| What | Where |
| --- | --- |
| Your configuration | `~/.nanoclaude/config.toml` |
| A project's configuration | `.nanoclaude/config.toml` in the project root (the directory `ncc` runs in, or `--root`) |
| Stored sessions and the audit record | `~/.nanoclaude/sessions.db` |
| What `ncc` learned about a model | `~/.nanoclaude/capabilities.json` |
| What you typed at the prompt | `~/.nanoclaude/history` |
| Your instructions for every project | `~/.nanoclaude/NANO.md` |
| Instructions for one project | `NANO.md` in its root, and in any directory below it that you work in; `CLAUDE.md` is read in a directory that has no `NANO.md` |

`NANOCLAUDE_HOME` moves the home. Set it to a directory and `ncc` keeps `.nanoclaude` inside that directory instead of inside your home directory: the configuration, the session store, the history, the capability cache and the global `NANO.md` all move together. It may start with `~`.

A key is never read from a file. A model entry names the environment variable that holds its key, and an `api_key` in a file is refused with a message saying so.

## Two files, one trust rule

A project's `.nanoclaude/config.toml` arrives with a repository, so it is untrusted input. It can narrow what `ncc` does and choose among models. It cannot widen anything, and it cannot say where your key goes. A cloned repository must not be able to grant itself permissions or redirect your API key.

| A project file may | A project file may not |
| --- | --- |
| add `deny` rules, and `ask` rules | add `allow` rules |
| choose models and roles that are already defined, and define a model of its own with the `anthropic` or `ollama` adapter | set `base_url` or `api_key_env` on any model; for that reason it cannot define an `openai_compat` model, which needs a `base_url`, though it can select one your home file defines |
| change the `model`, `context_window` or `native_tools` of an alias your home file defines | change the `adapter` of an alias your home file defines |
| lower `max_turns`, `bash_timeout_s` and `output_cap_bytes`, or leave them as they are | raise any of those three above the value in effect |
| set `compact_soft`, `compact_hard` and `keep_recent_turns`, which bound nothing | |

Each thing on the right is refused with an error that names the file and says where the setting belongs, and nothing is silently ignored. A project's lists add to yours: its `deny` and `ask` rules are appended to the ones in effect, each rule once, and an empty list removes nothing. Your own `deny` rules therefore always apply. When `ncc` is run from your home directory, the project file is your own file and is read once as that.

Everything else about the two files is the same. Tables merge key by key, so a project that wants a tighter turn limit does not have to restate its models.

```toml
# .nanoclaude/config.toml (in the project)
[roles]
explore = "local"

[permissions]
deny = ["Read(**/*.pem)", "Bash(curl:*)"]
ask  = ["Edit(migrations/**)"]

[limits]
max_turns = 20
```

## Models

```toml
# ~/.nanoclaude/config.toml
[models.sonnet]
adapter = "anthropic"
model   = "claude-sonnet-5-5"

[models.opus]
adapter = "anthropic"
model   = "claude-opus-5-5"

[models.haiku]
adapter = "anthropic"
model   = "claude-haiku-4-5"

[models.local]
adapter = "ollama"
model   = "qwen3-coder"

[models.cheap]
adapter     = "openai_compat"
base_url    = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
model       = "deepseek/deepseek-v4-pro"
```

An alias is a name you choose. Roles and `--model` refer to aliases. At least one model must be defined, in one of the two files.

| Key | Meaning |
| --- | --- |
| `adapter` | Required. One of `anthropic`, `openai_compat`, `ollama`. |
| `model` | Required. The id the provider knows the model by. |
| `base_url` | Where requests go. Required for `openai_compat`; optional for the others (the defaults are `https://api.anthropic.com` and `http://localhost:11434`). Must start with `http://` or `https://`. Home file only. |
| `api_key_env` | The name of the environment variable that holds the key. The default is `ANTHROPIC_API_KEY` for `anthropic` and `OPENAI_API_KEY` for `openai_compat`; `ollama` needs no key. Home file only. |
| `context_window` | The model's window in tokens, a whole number of at least 1. It overrides what `ncc` knows. If it is lower than the model's own window it also caps the length of a reply at a quarter of the window (a smaller window leaves little room for a long reply, and the budget holds the reply's room back); a window that is raised or restated changes nothing else. |
| `native_tools` | `true` or `false`. It overrides whether the model is believed to call tools natively. |

What the adapters mean in practice:

- `anthropic` talks to the Anthropic API, and caches the system prompt and the tool list.
- `openai_compat` talks to anything that speaks the chat-completions format: OpenAI, OpenRouter, Groq, DeepSeek, Together, Azure, vLLM, LM Studio, llama.cpp's server. It always sends a key as a bearer token and will not build a client without one, so a local server that wants none still needs its variable set to something. OpenAI's own hosts get `max_completion_tokens` and every other host `max_tokens`.
- `ollama` talks to a local Ollama server over its native API, so that `/api/show` can say whether the model's template supports tools.

A request sends no temperature, whatever the model: current Claude models and OpenAI's reasoning models refuse any value but the default, and a local server's default is the one its model's authors tuned.

**What a model can do.** `ncc` decides from, in order: a table of models it knows, what it learned about the model on an earlier run, a probe (Ollama only), and then the conservative default: no native tool calls, an 8,192-token window, 2,048 tokens of output. A model that cannot call tools natively is given the tool definitions in its system prompt and writes its calls as text, and the REPL says so when it starts (`⚠ text-tool mode (model lacks native tool calling)`); that works less well, and a model that fails to write a usable call three times in a row ends the prompt with an error naming it. An OpenAI-compatible model that is not in the table is not probed, so it gets the text protocol until you set `native_tools = true` for it, and its window is believed to be 8,192 tokens until you set `context_window`.

**Prices.** `/cost` shows what each role spent, with a price for the Anthropic models in the price book and a dash for any model without one (a local model is free). A dash is not zero. The table is dated, and `/cost` says so when it is more than 90 days old.

## Roles

```toml
# ~/.nanoclaude/config.toml (excerpt)
[roles]
main    = "sonnet"
explore = "local"
plan    = "opus"
verify  = "cheap"
compact = "haiku"
title   = "local"
```

There are six roles: `main`, `explore`, `plan`, `verify`, `compact` and `title`. Each names a model alias. A role that is not set follows `main`, and `main`, if not set, is the first model defined. A role that names a model nobody defined is refused when the file is loaded.

In this version two roles are used. `main` answers the conversation, and `compact` writes the summary when a conversation is compacted, so that `compact` is where a cheaper model already saves money. `explore`, `plan`, `verify` and `title` are checked and routed and nothing yet sends them a request; routing them is v0.3 scope, with sub-agents. `--model <alias>` replaces `main` for one run, `--role <role>=<alias>` replaces one role and may be given more than once, and `/model <alias>` replaces `main` in a session.

The summary request carries the whole older history to the `compact` model, so a model with a small window can fail to write it. The conversation is then left as it was, and the error names the compact role and says how to route it elsewhere.

## Permissions

```toml
# ~/.nanoclaude/config.toml (excerpt)
[permissions]
allow = ["Read", "Grep", "Glob", "Bash(npm test:*)", "Bash(git status)"]
ask   = ["Bash", "Write", "Edit"]
deny  = ["Read(**/.env*)", "Bash(curl:*)", "Edit(migrations/**)"]
```

The defaults are `allow = ["Read", "Grep", "Glob"]`, `ask = ["Bash", "Write", "Edit", "Git", "TodoWrite"]` and no `deny`. A list in your own file replaces the default for that key. A rule may name a tool that does not exist yet, which is why `Bash` and `Git` are accepted today.

**Rules** have three shapes: `Tool`, which matches every call to that tool; `Tool(subject)`, which matches one subject exactly or, for a path, as a glob; and `Tool(prefix:*)`, which matches a subject that begins with the prefix at a word boundary, so `Bash(npm test:*)` covers `npm test -- --watch` and not `npm testify`. The subject is the resolved path for `Read`, `Edit` and `Write`, the pattern for `Glob` and `Grep`, and, when they are built, the command for `Bash` and the subcommand for `Git`. A glob is matched against both the absolute path and the path relative to the sandbox root, in the syntax of a `.gitignore` file. A rule with an empty subject or unbalanced parentheses is refused when the file is loaded.

**The order of the checks** is fixed, and the first that matches decides. The rule id is what appears in a refusal and in the audit record.

1. `rule.deny` is your own `deny` rule. It refuses in every mode.
2. `secret.path` refuses a path that is a credentials file by convention (`.env`, `*.pem`, `*.key`, SSH keys, `.aws/**`, `.ssh/**` and the like), unless `--allow-secrets` is given.
3. `sandbox.outside-root` and `sandbox.symlink-escape` refuse a path outside the sandbox, and a write through a symlink that leaves it.
4. `bash.dangerous` and `bash.unparseable` refuse a command the danger classifier blocks or cannot read.
5. `mode.plan-read-only` refuses any write in `plan` mode.
6. `mode.bypass` allows everything left, in `bypass` mode.
7. `mode.accept-edits` allows `Edit` and `Write`, in `accept-edits` mode.
8. `rule.allow` allows what your `allow` rule names.
9. `grant.session` allows a tool you answered "always" to, for the rest of the session.
10. `tool.read-only` allows `Read`, `Grep`, `Glob` and `TodoWrite`.
11. `rule.ask` asks about what your `ask` rule names.
12. `default.ask` asks about everything else.

The first five are the hard denials: no mode, rule or grant gets past them. It follows from the order that an `ask` rule has no effect on a tool that is allowed without asking, and that an `allow` rule cannot open a credentials file. To keep something from being read, use `deny`.

**Modes.** `default` asks before changing anything. `plan` refuses every write, so the model can read and reason and not change. `accept-edits` applies `Edit` and `Write` without asking and still asks about anything else. `bypass` asks about nothing, and is reached only with `--dangerously-skip-permissions`; it does not turn off the hard denials, so the sandbox, credentials paths, dangerous commands and your `deny` rules are enforced in it. `--mode` takes `default`, `plan` or `accept-edits`, and `--mode bypass` alone is refused; so is the flag together with another mode, and no configuration file can select it. `/mode` switches among the first three and refuses `bypass`.

**Confirmations.** A call that needs confirming is shown (an `Edit` as a diff, a `Write` as a diff or the first lines and the size) and the prompt is `[y]es / [n]o / [a]lways`. "Always" grants the tool, not the command, for the rest of the session, and is kept in memory only. A call declined tells the model not to try another way to do the same thing. Without a terminal, `ncc -p` declines everything that would have asked.

**The sandbox** is the project root and each `--add-dir`. A path is judged by where it resolves to, symlinks followed, and a write also checks that its parent directory resolves inside. A root given with `~` or relative to where `ncc` was run is made absolute first. An empty value is an error, never the current directory.

## Limits

```toml
# ~/.nanoclaude/config.toml (excerpt)
[limits]
max_turns         = 40
compact_soft      = 0.70
compact_hard      = 0.85
keep_recent_turns = 3
```

| Key | Default | Meaning |
| --- | --- | --- |
| `max_turns` | 40 | The most model round trips one prompt may take, a whole number of at least 1. A prompt that needs more stops with a message saying how to go on; the next prompt gets the whole count again. `--max-turns` overrides it. |
| `bash_timeout_s` | 120 | Seconds a shell command may run, above 0. Read by the shell tool, which is not built yet; today it is checked and has no effect. |
| `output_cap_bytes` | 100000 | The most output of a command kept, a whole number of at least 1. Not read yet, for the same reason. |
| `compact_soft` | 0.7 | The share of the room for a conversation at which old tool results are shrunk to one line each. Above 0, at most 1. |
| `compact_hard` | 0.85 | The share at which everything older than the recent turns is replaced by a summary. |
| `keep_recent_turns` | 3 | How many recent turns a summary leaves as they were, at least 1. A model whose window is under 32,000 tokens keeps at most 2. |

The room for a conversation is the model's window, less the longest reply it may write, less a tenth of the window. If the system prompt and the tool definitions alone do not fit, compaction cannot help, and `ncc` says so with exit code 3. `compact_soft`, `compact_hard` and `keep_recent_turns` are the limits a project file may set freely, since they bound nothing.

## The `[ui]` section

`theme` (`auto`, `light` or `dark`) and `diff_style` (`unified`) are reserved. They are checked when the file is loaded and nothing reads them in this version. `NO_COLOR` in the environment, or `--no-color`, turns colour off.

## Environment variables

| Variable | Effect |
| --- | --- |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | The default key variables of the `anthropic` and `openai_compat` adapters. A model entry can name any other with `api_key_env`. |
| `NANOCLAUDE_HOME` | Where `.nanoclaude` is kept, in place of your home directory. |
| `NANOCLAUDE_EDITING_MODE` | `vi` (or `vim`) gives vi keys at the prompt; anything else is emacs keys. |
| `NO_COLOR` | Any non-empty value turns colour off. |

## The command line

| Invocation | Does |
| --- | --- |
| `ncc` | Starts the interactive REPL. |
| `ncc init` | Chooses a provider, takes a key, writes the configuration. |
| `ncc -p "<prompt>"` | Runs one prompt and exits. |

| Flag | Meaning |
| --- | --- |
| `-p`, `--print PROMPT` | Run once and exit. |
| `--output-format text\|json` | With `-p`: the model's words, or a JSON object. |
| `--root DIR` | The project directory, in place of the current one. |
| `--model ALIAS` | The model for the `main` role, for this run. |
| `--role ROLE=ALIAS` | The model for one role. May be repeated. |
| `--mode default\|plan\|accept-edits` | The permission mode. |
| `--add-dir DIR` | Add a directory to the sandbox. May be repeated. |
| `--max-turns N` | The turn limit for each prompt. |
| `-c`, `--continue` | Take up the latest session started in this directory. |
| `-r`, `--resume ID` | Take up a session by its id. |
| `--allow-secrets` | Read credentials files, and do not redact. Prints a warning. |
| `--dangerously-skip-permissions` | Never ask. The hard denials stay. Prints a warning. |
| `--no-color` | No colour. |
| `--version` | Print the version. |

Flags are spelled out in full; an abbreviation is not accepted. `ncc init` takes no flag but `--no-color`, and needs a terminal. `--continue` and `--resume` together are refused. A session can only be taken up from the directory it was started in, and `ncc` says which directory to run from when it was not. A `--continue` with nothing started here is an error and does not start a new session.

**Streams.** In text mode stdout carries the model's reply exactly as written and nothing else, with a newline added only if it had none. Everything that went wrong is one line on stderr in the form `error: <what happened> — <what to do>`. With no `-p` and no terminal, `ncc` says so and exits with code 2 and does nothing.

**JSON.** `--output-format json` prints one object with `session_id`, `result`, `stop_reason`, `turns`, `usage` (`input_tokens`, `output_tokens`, `cache_read_tokens`), `cost_usd` and `is_error`. `cost_usd` is `null` when any model used has no price, never `0`. `stop_reason` is one of `completed`, `turn_limit`, `refusal`, `max_tokens`, `model_unsuitable` and `cut_off`.

**Exit codes.**

| Code | Meaning |
| --- | --- |
| `0` | The prompt finished, or the REPL was left normally. |
| `1` | The prompt stopped before the task was done (the turn limit, a refusal, a reply cut at the model's output limit, a model that could not write a tool call), or something failed that is neither the configuration nor the provider: the session store could not be opened or is busy, another process changed the session, a bug. |
| `2` | `ncc` was used wrongly: a bad flag, no prompt and no terminal, a session that is not in the store. |
| `3` | The configuration is missing or wrong, including a missing or rejected API key, or the model's window is too small for the prompt. |
| `4` | The provider failed: the network, a server error after the retries, a rejected request, or a reply that was cut off midway. What arrived of a cut-off reply is kept and printed. |
| `130` | Interrupted with Ctrl+C. Nothing is written to stdout. |
| `141` | Whoever was reading the output closed it, as `ncc ... \| head -1` does. Not an error; the shell's own number for it. |

## Slash commands

| Command | Does |
| --- | --- |
| `/help` | Lists the commands. |
| `/clear` | Starts a fresh conversation. The old one stays in the session archive; it is not deleted. |
| `/compact [what to keep]` | Summarises the older conversation now. Words after it say what to keep. |
| `/status` | The model for each role, the mode, the sandbox, the number of tools, the session id, turns used against the limit, and how full the context is. |
| `/cost` | Tokens and money this run, by role. A resumed session shows only what this run spent. |
| `/model [alias]` | Lists the models, or switches the `main` role. |
| `/mode [mode]` | Shows the mode, or switches among `default`, `plan` and `accept-edits`. `bypass` is refused. |
| `/tools` | Lists the tools and marks those that cannot change anything. |
| `/resume` | Says to restart with `ncc --resume <id>`. |
| `/export [file]` | Writes the conversation as markdown. It never overwrites a file; the default name is `session-<id>.md` in the project, and the file is readable by you alone. Tool results are cut at 2,000 characters and the model's private reasoning is left out. |
| `/init` | Says to run `ncc init`. |
| `/exit` | Leaves. Ctrl+D does the same. |

## The prompt

Enter sends a line. A second line is started with Esc then Enter (with emacs keys), or by ending a line with a backslash. Completion offers the slash commands at the start of a line, and after `@` the files in the project; a file named with `@` is inlined into the prompt, up to 100,000 bytes, unless it is outside the project or a credentials file. Ctrl+C cancels what is running and keeps the session; two within a second leave. History is kept between sessions in `~/.nanoclaude/history`.

## ncc init

`ncc init` is the first run. It asks which provider (Anthropic, OpenAI, OpenRouter or Ollama), for the key if one is needed and is not already in the environment, and for the model, and then:

1. It writes `~/.nanoclaude/config.toml`, readable by you alone, with every role on the model you chose and a comment on each choice. A `.nanoclaude` directory that it has to create is one only you can enter. If a configuration is already there it asks first, and the default answer is no.
2. It forgets what `ncc` learned about models, so each is probed again.
3. It checks the key with one small request, sent with the configuration it just wrote. A check that fails is a warning, not an error: the file stays, and your choices are not lost to a mistyped key.

The key is used for that one request and kept nowhere: not in the file, not in the environment of `ncc`, not on the screen. The file names the environment variable and `ncc init` prints the line to add to your shell profile. A key with a space, a line break or a character that is not plain ASCII in it is not sent, and the message says what is wrong and where. It needs a terminal.

| Choice | Adapter | Default model | Key variable |
| --- | --- | --- | --- |
| 1 | `anthropic` | `claude-sonnet-5-5` | `ANTHROPIC_API_KEY` |
| 2 | `openai_compat` (`https://api.openai.com/v1`) | `gpt-5` | `OPENAI_API_KEY` |
| 3 | `openai_compat` (`https://openrouter.ai/api/v1`) | `deepseek/deepseek-v4-pro` | `OPENROUTER_API_KEY` |
| 4 | `ollama` | `qwen3-coder` | none |

It does not write a `NANO.md`.

## Sessions

Every conversation is stored in `~/.nanoclaude/sessions.db`, as it happens. `--continue` and `--resume` load one. Nothing lists them; `sqlite3` does. The tables are `sessions`, `messages` (the conversation the model is shown), `messages_archive` (every message a compaction or `/clear` replaced, with when) and `tool_calls` (the audit record: each call, what was decided and by which rule, and what came of it). A decision is written before the call runs. Arguments are stored with credentials scrubbed.

What resuming does not restore is what the conversation had read: the model reads a file again before it edits it.

## Known gaps

- A retry of a provider request is shown as it happens and is not stored, because the audit table is keyed by tool call and has no row for a request.
- The summary that compaction asks for is written by the `compact` role's model from the whole older history, so a small compact model can fail with the history kept intact. The error names the role.
- After Ctrl+C the model is told that calls which had not reported back may or may not have run; the results of calls that had finished are not handed back, and it may repeat them.

## Model ids in the examples

Each hosted model id in this document and in the README was checked against its provider's live list on the day noted. The Anthropic ids (`claude-sonnet-5-5`, `claude-opus-5-5`, `claude-haiku-4-5`) were checked on 2026-10-02. The OpenRouter id (`deepseek/deepseek-v4-pro`) was checked against OpenRouter's public model list on 2026-10-07. The OpenAI id that `ncc init` offers, `gpt-5`, was not: OpenAI's list needs a key to read. Local model names are whatever your server knows them as.
