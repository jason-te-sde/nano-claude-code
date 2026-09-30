<h1 align="center">nano-claude-code</h1>

<p align="center">
  A terminal coding agent that brings its own models.<br>
  The shape of Claude Code, pointed at the Anthropic API, any OpenAI-compatible endpoint, or a
  local Ollama server.
</p>

<p align="center">
  <a href="https://github.com/jason-te-sde/nano-claude-code/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/jason-te-sde/nano-claude-code/actions/workflows/ci.yml/badge.svg">
  </a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue">
  <a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

---

**Status:** `0.1.0.dev0`, an early scaffold. `ncc --version` and a usage message are the whole
surface today. Everything below — the agent loop, the tools, the provider adapters, the
permission engine — is the target design and lands incrementally over the tasks that follow.
See [`NOTICE`](NOTICE) for what "Claude" in the name does and does not mean.

## Install

```bash
pipx install nano-claude-code
# or
uvx nano-claude-code
```

Requires Python 3.11 or newer, on macOS or Linux. Windows is not supported — process groups,
signals and path semantics differ enough that pretending to support it would be worse than
saying so.

## A quick example

This is the target interaction (see Status above for what exists today):

```bash
$ pipx install nano-claude-code
$ ncc init
# pick a provider, paste a key, ncc sends a one-token request to check it before saving anything

$ ncc
> fix the failing test in tests/test_thing.py
```

`ncc` reads the failing test and the file it exercises, proposes an edit, asks for confirmation
before writing it, then re-runs the test to check that the fix actually took — the difference
between the model *claiming* it fixed something and a checker *agreeing* that it did.

## Models

Three adapters, chosen so that "supports N providers" means something concrete:

| Adapter | Covers |
| --- | --- |
| `anthropic` | The Anthropic API |
| `openai_compat` | Any OpenAI-compatible endpoint: OpenAI, OpenRouter, Groq, DeepSeek, Together, Azure, vLLM, LM Studio, llama.cpp's server |
| `ollama` | A local Ollama server, over its native API, so capability probing can use `/api/show` |

```toml
[models.sonnet]
adapter = "anthropic"
model   = "claude-sonnet-5"

[models.local]
adapter = "ollama"
model   = "qwen3-coder:30b"

[models.cheap]
adapter  = "openai_compat"
base_url = "https://openrouter.ai/api/v1"
model    = "deepseek/deepseek-v3"
```

A model is not one slot but several: the `main` role writes code, but `explore`, `plan`,
`verify`, `compact` and `title` are routed separately. Exploring a codebase — reading files to
figure out what to change — is a large share of the tokens in a coding session and needs far
less capability than writing the change does, so it can go to a cheap or local model instead of
the one doing the real work. `--model <alias>` overrides `main` for a single run; `--role
<role>=<alias>` overrides one role at a time.

## Safety

| Mode | Behaviour |
| --- | --- |
| `default` | Reads run on their own; writes and `Bash` ask for confirmation |
| `plan` | Every write is refused; the agent can read and reason but not change anything |
| `acceptEdits` | File edits apply automatically; `Bash` still asks |
| `bypass` | Everything runs without asking. Requires `--dangerously-skip-permissions` and prints a warning at startup |

`bypass` only skips the *confirmation prompt* — the sandbox boundary, the secret-path denials
and the dangerous-command classifier sit underneath every mode, including `bypass`, and are not
something a permission mode can turn off. See [`SECURITY.md`](SECURITY.md) for the full threat
model, including what this project does not defend against.

## What this does not do

Deliberately, and by comparison with the project it is modelled after:

| Not in scope | Why |
| --- | --- |
| Windows | Process groups, signals and path semantics all differ enough that pretending to support it would be worse than saying so |
| Agent teams and autonomous multi-agent coordination | A different project |
| Worktree isolation | A single session in a single directory is enough for v1 |
| LSP integration | The upstream implementation is large and open-ended; verifying an edit by re-running the affected check gets most of the value at a fraction of the cost |
| A plugin marketplace | An ecosystem problem, not a tooling one |
| A skill system | `NANO.md` already covers the core need: project-level instructions |
| IDE extensions | — |
| OAuth sign-in and billing | You bring your own API key |
| Telemetry | An open-source tool should not phone home |
| Voice, notebook and PowerShell support | — |
| An MCP server (client only) | Letting others plug into nano-claude-code is a different project than plugging nano-claude-code into others |
| A persistent background shell | Each `Bash` call is an independent process with a fixed working directory and no `cd` or `export` carried over — it trades a slower calling convention for a class of hidden cross-call state bugs it cannot have |
| Automatic retrieval-augmented injection | Thin automatic context plus explicit `@` references and search tools beats injecting "related" code automatically — most of what that would inject is noise, and the model steers `Grep`/`Glob` itself better than a retrieval step can guess on its behalf |

## License

[MIT](LICENSE). See [`NOTICE`](NOTICE) for the trademark and affiliation statement.
