<h1 align="center">nano-claude-code</h1>

<p align="center">
  A terminal coding agent that works with Anthropic, OpenAI-compatible and local models.
</p>

<p align="center">
  <a href="https://github.com/jason-te-sde/nano-claude-code/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/jason-te-sde/nano-claude-code/actions/workflows/ci.yml/badge.svg">
  </a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

---

> **Status: version `0.1.0.dev0`, not released.** This describes `main` as it is. Built: the agent loop, the tools `Read`, `Edit`, `Write`, `Glob`, `Grep` and `TodoWrite`, four permission modes with a sandbox, the provider adapters, compaction, stored sessions, cost accounting and the REPL. **Not built yet: `Bash` and `Git`.** Until they are, the agent can read and change files in a project and cannot run a command or a test, and v0.1 will not be tagged before they land.

## Install

```bash
pipx install git+https://github.com/jason-te-sde/nano-claude-code.git
ncc init
ncc
```

Python 3.11 or newer, on macOS or Linux. Windows is not supported. There is no package on PyPI yet; that release is planned for v0.4. Check an install with:

```bash
ncc --version
```

`ncc init` asks which provider you use (Anthropic, OpenAI, OpenRouter or a local Ollama server), takes your key, and writes `~/.nanoclaude/config.toml`. It writes the file first and checks the key afterwards with one small request, so a key you mistype does not cost you your choices: a check that fails is a warning, and the file stays. The key itself is never saved. The file names the environment variable that holds it, and `ncc init` prints the line to add to your shell profile.

## Why this and not Claude Code

It has the same shape as Claude Code, a loop with file tools, permission modes and a REPL, and it takes your choice of model: the Anthropic API, any OpenAI-compatible endpoint (OpenAI, OpenRouter, Groq, DeepSeek, Together, Azure, vLLM, LM Studio, llama.cpp), or a local Ollama server. A local model that cannot call tools natively is not turned away: it is given the tool definitions in its prompt and writes its calls as text, which works less well and is the price of using it. What it gives up is listed below, and it is young. It is an independent project; [`NOTICE`](NOTICE) says what "Claude" in its name does and does not mean.

## Roles

A session does not have to use one model. Each job has a slot, called a role, and each slot names a model you defined.

```toml
# ~/.nanoclaude/config.toml
[models.sonnet]
adapter = "anthropic"
model   = "claude-sonnet-5-5"

[models.local]
adapter = "ollama"
model   = "qwen3-coder"

[models.cheap]
adapter     = "openai_compat"
base_url    = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
model       = "deepseek/deepseek-v4-pro"

[roles]
main    = "sonnet"
explore = "local"
verify  = "cheap"
compact = "cheap"
```

The idea is that reading a codebase to work out what to change is much of the tokens in a session and little of its difficulty, so `explore` is the role meant for a cheap or a local model while `main`, which writes the change, stays strong. A role you leave out follows `main`. `--model <alias>` replaces `main` for one run, and `--role <role>=<alias>` replaces one role.

**Today two roles are used.** `main` answers the conversation, and `compact` writes the summary when a long conversation is compacted; that request carries the whole older history, so a cheap model there already saves money. `explore`, `plan`, `verify` and `title` are accepted and checked, and nothing sends them a request yet: routing them is v0.3 scope, with sub-agents. Neither that premise nor what routing exploration to a cheap model would save, or cost in the quality of what comes back, has been measured. `/cost` shows what each role spent.

## Safety

| Mode | Behaviour |
| --- | --- |
| `default` | Reads run on their own; anything that changes a file asks first |
| `plan` | Every write is refused; the agent can read and reason and change nothing |
| `accept-edits` | `Edit` and `Write` apply without asking |
| `bypass` | Nothing asks. Needs `--dangerously-skip-permissions`, and prints a warning |

`bypass` skips the question and not the protections. The sandbox, the credentials rule, the dangerous-command rule and your own `deny` rules are checked first, in every mode, and nothing gets past them.

| Tool | Runs without asking in `default` mode |
| --- | --- |
| `Read` | yes |
| `Glob` | yes |
| `Grep` | yes |
| `TodoWrite` | yes (it changes only the session's own task list) |
| `Edit` | no, it shows a diff first |
| `Write` | no |

`Bash` and `Git` are not built yet. They are planned to work like this: `Bash` asks first, `Git` reads without asking and asks before it changes anything, and a shell command is classified before it can run.

| Defence | State |
| --- | --- |
| The sandbox: a path is judged by where it resolves to, and a write through a symlink that leaves the project is refused; `--add-dir` adds a root | Built |
| Credentials: paths such as `.env`, `*.pem` and SSH keys are refused, and what looks like a key is scrubbed from what tools return, before it is stored | Built, with gaps listed in [`SECURITY.md`](SECURITY.md) |
| Your own `deny`, `ask` and `allow` rules, with `deny` always winning | Built |
| A record of every call, written before the call runs, in `~/.nanoclaude/sessions.db` | Built; nothing reads it back yet |
| Classifying a shell command as dangerous | **Not measured yet.** The regex classifier is written and has known blind spots, each checked by a test, and there is no corpus of dangerous commands to say how many it misses (`tests/corpus/dangerous.jsonl` does not exist), so there is no figure. The syntax-tree classifier, an optional extra meant to be the authority, is not built, and nothing calls either until `Bash` exists. A command the regex classifier finds nothing in is not treated as cleared. |
| Undoing an edit | Not built. A write is confirmed first, and `git` is how you undo one. |

## Configuration

Two files: `~/.nanoclaude/config.toml` is yours, and `.nanoclaude/config.toml` in a project narrows it. A project's file arrives with a repository, so it is untrusted: it may add `deny` and `ask` rules and choose among models and roles, and it may not add `allow` rules, set a `base_url` or an `api_key_env`, change the adapter of one of your models, or raise `max_turns`, `bash_timeout_s` or `output_cap_bytes`. Each of those is refused with an error. They belong in your own file, because a cloned repository must not be able to grant itself permissions or redirect your API key.

The reference is [`docs/configuration.md`](docs/configuration.md): every key and its default, the rules and the order they are checked in, the flags, the exit codes and what `ncc init` does. [`docs/architecture.md`](docs/architecture.md) is how it is built, [`docs/testing.md`](docs/testing.md) is how it is tested and what the tests do not cover, and [`docs/design/`](docs/design/0001-scope.md) holds a note for each decision, with its costs and the alternatives that were rejected.

## What v0.1 does not do

- **Checkpoints and rewind (v0.2).** Undoing a model's edit means a snapshot before every write, kept where it cannot touch the person's own index, branches or stash. That is a subsystem of its own. Until it exists, an edit is undone with git, and every write is confirmed first.
- **Hooks (v0.2).** A hook runs the person's command around a tool call. A feature that runs commands needs its trust rules designed with it, so a `hooks` section in a configuration file is refused rather than ignored.
- **Post-edit verification (v0.2).** Running the project's own checker after an edit, and returning its diagnostics to the model, reuses the command-running machinery that hooks need. A `verify` section is refused for the same reason.
- **Background commands and BashOutput (v0.2).** A command that outlives its call needs the shell tool to exist first, and a way to read what it printed since.
- **Sub-agents and the Task tool (v0.3).** A sub-agent is a nested session with a narrower sandbox. Roles already let a session use several models; the nesting waits until a single session is solid.
- **MCP client (v0.3).** The description a server gives a tool is read by the model, so it is an injection surface. That boundary (namespaced names, truncated descriptions, confirmation by default) has to be built and tested as a whole, not added to a first release.
- **Tool plugins (v0.3).** A plugin is arbitrary Python, and waits with the MCP trust work for the same reason.
- **WebFetch and AskUser (v0.3).** WebFetch needs protection against requests to private addresses before it is safe to offer, and AskUser needs a new method on the UI protocol.
- **`ncc bench` and the model compatibility table (v0.4).** The table is the only honest form of "works with many models", and building it means running real models against a fixture repository. Until then nothing here says which models do well beyond what the code itself knows.
- **`ncc audit` and `ncc gc` (not scheduled).** The audit table is written in v0.1, and nothing reads it back; open `sessions.db` with `sqlite3`.
- **A package on PyPI (v0.4).** Until then, install from the repository.
- **Windows.** Process groups, signals and path semantics all differ enough that pretending to support it would be worse than saying so.
- **Agent teams and autonomous multi-agent coordination.** A different project.
- **Worktree isolation.** A single session in a single directory is enough.
- **LSP integration.** It is large and open-ended. Verifying an edit by running the affected check gets most of the value at a fraction of the cost.
- **A plugin marketplace.** An ecosystem problem, not a tooling one.
- **A skill system.** `NANO.md` already covers project-level instructions.
- **IDE extensions.** Not a goal.
- **OAuth sign-in and billing.** You bring your own API key.
- **Telemetry.** An open-source tool should not phone home.
- **Voice, notebook and PowerShell support.** Not a goal.
- **An MCP server.** The project is a client only; letting others plug into it is a different project from plugging it into others.
- **Automatic retrieval-augmented injection.** Thin automatic context plus explicit `@` references and search tools does better, and 0013 says why.

## Numbers

Each figure is produced by `python scripts/measure.py`, which also writes `metrics.json`, and nothing here is written from memory: a test fails if a figure that can be recomputed has drifted. Coverage needs the whole suite to run, so no test can recompute it; it was measured on Python 3.11 on macOS, and CI measures it for each platform and interpreter and compares nothing.

| Figure | Key | Value | Produced by |
| --- | --- | --- | --- |
| Tests collected by pytest | `tests` | 2,911 | `python scripts/measure.py` |
| Share of `nanoclaude` the tests run | `coverage_percent` | 99.9% | `python scripts/measure.py` |
| Lines of Python in `src/` | `lines.src` | 11,401 | `python scripts/measure.py` |
| Lines of Python in `tests/` | `lines.tests` | 30,349 | `python scripts/measure.py` |
| Tools in the default registry | `tools` | 6 | `python scripts/measure.py` |
| Adapters the configuration accepts | `adapters` | 3 | `python scripts/measure.py` |
| Dangerous commands the regex classifier misses | `danger_corpus` | not measured yet | `python scripts/measure.py` |

---

[MIT](LICENSE). See [`NOTICE`](NOTICE) for the trademark and affiliation statement.
