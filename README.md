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

> **Status: version `0.1.0.dev0`, not released.** This describes `main` as it is. Built: the agent loop, the tools `Read`, `Edit`, `Write`, `Glob`, `Grep`, `TodoWrite` and `Task`, four permission modes with a sandbox, the provider adapters, MCP servers over stdio, compaction, stored sessions, cost accounting and the REPL. **Not built yet: `Bash` and `Git`.** Until they are, the agent can read and change files in a project and cannot run a command or a test, and v0.1 will not be tagged before they land.

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

**Today three roles are used.** `main` answers the conversation. `explore` answers the `explore` sub-agent, which the `Task` tool starts: it searches and reads the project with `Read`, `Glob` and `Grep` and hands back a written answer, which is the work a cheap or a local model is meant for. `compact` writes the summary when a long conversation is compacted; that request carries the whole older history, so a cheap model there already saves money. `plan`, `verify` and `title` are accepted and checked, and nothing sends them a request yet. Neither that premise nor what routing exploration to a cheap model would save, or cost in the quality of what comes back, has been measured. `/cost` shows what each role spent. A session does check, when it starts, that the key of every role's model is set, those three included, and names the first role whose key is not.

## MCP servers

`ncc` is a client of MCP servers that it starts itself and talks to over stdio. When a session starts it lists each server's tools and gives them to the model as `mcp__<server>__<tool>`; a call goes to the server, and what comes back is one tool result. That is all of the protocol it uses: no resources, no prompts, no sampling or elicitation, no HTTP transport (`transport` can only be `"stdio"`), and a server's tool list is read once and is not followed when it changes. It speaks the protocol of 2026-07-28 and falls back to the three versions before it, which is the kind most servers in use still speak. No test runs a server somebody else wrote: they run a fake one, and [`docs/testing.md`](docs/testing.md) says what that leaves untried.

Servers are configured in your own file, and only there:

```toml
# ~/.nanoclaude/config.toml (excerpt)
[mcp.notes]
command = "/home/you/bin/notes-server"
args    = ["--stdio"]
env     = { NOTES_TOKEN = "write the token here" }
timeout = 30
```

Each is a program that is started directly, not through a shell, in the project directory, with `PATH`, `HOME`, `USER`, `LOGNAME`, `LANG`, `LC_ALL`, `LC_CTYPE`, `TMPDIR` and `TERM` from your environment and what its `env` adds, so no provider key reaches it unless you write it there. `env` is plain text in the file, and `ncc` never changes the permissions of a configuration file that exists, so make a file that holds a credential readable by you alone. `enabled = false` keeps a server configured and not started. A project's own `.nanoclaude/config.toml` cannot define a server and is refused if it tries: a server is a program that runs with your authority, and a cloned repository must not be able to start one. Servers are started together when a session starts, `ncc` waits for all of them for at most fifteen seconds, and one that fails is a single warning line and no tools; the session goes on without it. `/mcp` lists each server with its state, its protocol era and how many tools it has registered, and `/mcp <server>` lists the tools of one.

A tool of a server asks before it runs, whatever it says it does: it is somebody else's code, and nothing says it only reads. `accept-edits` does not allow it, `bypass` does, and `allow = ["mcp__notes__search"]` in your own file does for the one tool it names in full; a wildcard is refused, since it would also allow what a later version of the server adds. Under `-p` a tool that nothing allows is declined, like any other. The model is told that the results of these tools are data and not instructions, and each result is wrapped in `<mcp-result>` tags that nothing inside it can close, stripped of terminal controls, scrubbed of credentials and cut to `output_cap_bytes`.

The description a server gives a tool is read by the model, so it is an injection surface, and it is treated as one. It has its terminal controls removed and is cut at 2,048 bytes, the input schema must be an object within limits that refers to nothing outside itself, and the name is made safe and namespaced. What each tool looked like when you first saw it is written down in `~/.nanoclaude/mcp-pins.json`, as a fingerprint. A tool whose description or schema is different at a later start is marked changed, and a changed tool asks before every call, `bypass` and your `allow` rules included (`plan` mode refuses every MCP tool, and a `deny` rule of yours refuses it, before that is asked). Answering "always" to it approves the new definition for the next session and not for this one, and under `-p` a changed tool is declined, whatever your rules say. A pin does not help against a server that was hostile the first time it was seen. A call to a server carries no path and no command, so the sandbox and the credentials rules do not apply to it, and the server can do what your account can do. [`SECURITY.md`](SECURITY.md) has the rest, and [0018](docs/design/0018-mcp-client.md) has the reasons and what was left out.

## Safety

| Mode | Behaviour |
| --- | --- |
| `default` | Reads run on their own; anything that changes a file asks first |
| `plan` | Every write is refused; the agent can read and reason and change nothing |
| `accept-edits` | `Edit` and `Write` apply without asking |
| `bypass` | Nothing asks, except about an MCP tool whose definition changed. Needs `--dangerously-skip-permissions`, and prints a warning |

`bypass` skips the question and not the protections. The sandbox, the rule for `.git` and `.nanoclaude` directories, the credentials rule, the dangerous-command rule and your own `deny` rules are checked first, in every mode, and nothing gets past them but `--allow-secrets`, which turns the credentials rule and the redaction off together and prints a warning.

With `-p` nothing can answer a question, so every call that would have asked is declined, and each is one line on stderr that names the rule that asked. A call the policy refuses outright gets no such line: the model receives the refusal, with its rule id, as the tool's result.

| Tool | Runs without asking in `default` mode |
| --- | --- |
| `Read` | yes |
| `Glob` | yes |
| `Grep` | yes |
| `TodoWrite` | yes (it changes only the session's own task list) |
| `Task` | no, it asks first; the sub-agent it starts can only read |
| `Edit` | no, it shows a diff first |
| `Write` | no |
| A tool of an MCP server | no, it asks first, and a changed one asks in `bypass` too; `plan` mode refuses it |

A `Task` call starts a sub-agent in a session of its own, and the question you are asked is whether to spend the requests. The sub-agent has `Read`, `Glob` and `Grep` and no other tool, works in your sandbox and your permission mode with every `deny` rule of yours in force, holds the grants you have already given and cannot add one, and cannot ask you anything: a call that would have asked is declined. It is stopped after 30 model calls, an estimated 200,000 input tokens or ten minutes, whichever comes first, and what it returns is one message, cut to `output_cap_bytes`: a model's account of files, as untrusted as the files are. `allow = ["Task"]` in your own file lets it run without asking.

`Bash` and `Git` are not built yet. They are planned to work like this: `Bash` asks first, `Git` reads without asking and asks before it changes anything, and a shell command is classified before it can run.

| Defence | State |
| --- | --- |
| The sandbox: a path is judged by where it resolves to, a write through a symlink that leaves the project is refused, and no search, listing or `@` mention follows a symlink out; `--add-dir` adds a root | Built |
| Protected paths: a write into `.git` or a `.nanoclaude` directory is refused in every mode, since git and `ncc` run or obey what is kept there | Built |
| Credentials: paths such as `.env`, `*.pem`, SSH keys and `.aws/` are refused, whatever their case, and a private-key block is masked whole wherever a file is shown; other keys and tokens are scrubbed from what tools return, before it is stored. `--allow-secrets` turns the refusal and the scrubbing off together | Built, with gaps listed in [`SECURITY.md`](SECURITY.md) |
| Your own `deny`, `ask` and `allow` rules, with `deny` always winning, and `deny` holding for `Grep`, `@` mentions and the project map as it does for `Read`. A rule that names a tool there is not is refused when the file loads | Built |
| Instruction files: `NANO.md` and `CLAUDE.md` from the top of the repository down to the project root go into the prompt only if their real path stays inside the repository and a `Read` of them would be allowed, scrubbed and capped | Built |
| A repository cannot make `ncc` run a program through git: git runs only as `git rev-parse`, which never reads the working tree | Built |
| A record of every call, written before the call runs, in `~/.nanoclaude/sessions.db`; rows are only added, and an outcome is set once | Built; nothing reads it back yet |
| What `ncc` creates under `~/.nanoclaude` is readable by you alone, and `ncc` run in `/` or in a home directory is refused unless `--root` says it was meant | Built |
| Classifying a shell command as dangerous | **Not measured yet.** The regex classifier is written and has eight known blind spots, of which a test shows four getting through and four have no test, and there is no corpus of dangerous commands to say how many it misses (`tests/corpus/dangerous.jsonl` does not exist), so there is no figure. The syntax-tree classifier, meant to be the authority and to be installed as an optional extra, is not built, and nothing calls either until `Bash` exists. A command the regex classifier finds nothing in is not treated as cleared. |
| MCP servers: a tool asks by default and is named exactly in a rule, its name, description and schema are held to limits, its definition is pinned and a change asks, `bypass` included, and its results are wrapped, scrubbed and capped | Built, with the limits in [`SECURITY.md`](SECURITY.md): a server is not inside the sandbox |
| Undoing an edit | Not built. A write is confirmed first, and `git` is how you undo one. |

## Configuration

Two files: `~/.nanoclaude/config.toml` is yours, and `.nanoclaude/config.toml` in a project narrows it. A project's file arrives with a repository, so it is untrusted: it may add `deny` and `ask` rules and choose among models and roles, and it may not add `allow` rules, set a `base_url` or an `api_key_env`, define an MCP server, change the adapter of one of your models, or raise `max_turns`, `bash_timeout_s` or `output_cap_bytes`. Each of those is refused with an error. They belong in your own file, because a cloned repository must not be able to grant itself permissions or redirect your API key.

The reference is [`docs/configuration.md`](docs/configuration.md): every key and its default, the rules and the order they are checked in, the flags, the exit codes and what `ncc init` does. [`docs/architecture.md`](docs/architecture.md) is how it is built, [`docs/testing.md`](docs/testing.md) is how it is tested and what the tests do not cover, and [`docs/design/`](docs/design/0001-scope.md) holds a note for each decision, with its costs and the alternatives that were rejected. The numbers 0008, 0009 and 0011 are left free for the notes on the shell tool, the Git tool and the classification of dangerous commands, which are written when those are built.

## What v0.1 does not do

- **Checkpoints and rewind (v0.2).** Undoing a model's edit means a snapshot before every write, kept where it cannot touch the person's own index, branches or stash. That is a subsystem of its own. Until it exists, an edit is undone with git, and every write is confirmed first.
- **Hooks (v0.2).** A hook runs the person's command around a tool call. A feature that runs commands needs its trust rules designed with it, so a `hooks` section in a configuration file is refused rather than ignored.
- **Post-edit verification (v0.2).** Running the project's own checker after an edit, and returning its diagnostics to the model, reuses the command-running machinery that hooks need. A `verify` section is refused for the same reason.
- **Background commands and BashOutput (v0.2).** A command that outlives its call needs the shell tool to exist first, and a way to read what it printed since.
- **More sub-agents, and nesting (v0.3).** The `Task` tool starts one built-in sub-agent, `explore`, which can only read and cannot start another. Custom sub-agents from `~/.nanoclaude/agents/`, the `plan` and `general` sub-agents, and a sub-agent that starts one of its own are not built: each gives a delegate more reach than `explore` has, and that is settled one sub-agent at a time.
- **The rest of MCP (not scheduled).** The client is built, for servers that `ncc` starts and speaks to over stdio, and it calls their tools. The HTTP transports, resources, prompts, sampling, elicitation and a tool list that changes during a session are not built: each lets a server reach the model or the person in a way that tools over stdio do not, and each needs its own trust decision, which 0018 records.
- **Tool plugins (v0.3).** A plugin is arbitrary Python that would run inside this process. The MCP client keeps a server in a process of its own, with a narrow environment and a pin on what it says, and a plugin has none of that, so its trust rules have to be designed on their own.
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
| Tests collected by pytest | `tests` | 4,955 | `python scripts/measure.py` |
| Share of `nanoclaude` the tests run | `coverage_percent` | 99.9% | `python scripts/measure.py` |
| Lines of Python in `src/` | `lines.src` | 17,311 | `python scripts/measure.py` |
| Lines of Python in `tests/` | `lines.tests` | 48,422 | `python scripts/measure.py` |
| Tools in the default registry | `tools` | 7 | `python scripts/measure.py` |
| Adapters the configuration accepts | `adapters` | 3 | `python scripts/measure.py` |
| Dangerous commands the regex classifier misses | `danger_corpus` | not measured yet | `python scripts/measure.py` |

---

[MIT](LICENSE). See [`NOTICE`](NOTICE) for the trademark and affiliation statement.
