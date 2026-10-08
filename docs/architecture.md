# Architecture

This describes `main` as it is built: six tools (Read, Edit, Write, Glob, Grep, TodoWrite), three provider adapters, a permission engine with a sandbox, and a REPL and a headless mode over one session. Bash and Git are not built yet. Where the code and an earlier plan disagree, this document follows the code, and the last section lists the differences.

Comments in the code cite sections of a design document ("spec 6.2"). That document is not in the repository. What matters from it is restated here, and the reasons behind the choices are in [the design notes](design/0001-scope.md).

## The shape

```
 cli/       ncc: flags, exit codes, the REPL, rendering, the slash commands, ncc init
              |
              |   the UI protocol (agent/ui.py): confirm, on_reply, on_text, on_outcome, ...
              v
 agent/     Session   one conversation, wired to everything below
            loop      the turn as a pure state machine (no I/O)
            Router    role -> model client, and the ledger of what each role spent
            Executor  decide every call, then run them
              |
   +----------+-----------+--------------+------------+--------------+
   |          |           |              |            |              |
 conversation tools    permissions     context      providers      config
 transcript   registry policy          assemble     base           schema
 budget       read     rules           instructions capabilities   load
 compaction   edit     sandbox         mentions     retry
 store        write    redact          projectmap   texttools
 (SQLite)     glob     audit                        pricing
              grep     danger/regex                 anthropic
              todo                                  openai_compat
                                                    ollama
```

`nanoclaude.testing` is also shipped: `ScriptedModel` plays back what it is told to say and `build_session` wires a real session around it. They are in the package and not in `tests/` so that anyone writing a tool can test it inside a real session.

One boundary is enforced by the build. Nothing under `agent/` imports from `cli/`, and `agent/loop.py` imports none of `os`, `subprocess`, `time`, `random`, `uuid`, `asyncio`, `sqlite3` or `httpx` (`tests/test_boundaries.py`). Reasons are in [0002](design/0002-pure-loop.md) and [0006](design/0006-ui-protocol.md). The other directions (that `tools/` or `providers/` do not import `agent/`) hold today and are not enforced.

## What happens to one prompt

1. `ncc` parses its flags, loads the configuration (your file, then the project's, with the trust rules of [configuration.md](configuration.md)), and builds a `Policy` (the sandbox roots, the rules, the mode, the credentials path patterns), a `Store`, a `Router`, the tool registry, one `Redactor`, an `AuditLog` and a `Session`. For a REPL it also builds a `ConsoleUI`; for `-p` the UI is one that declines every question.
2. `Session.follow_up(prompt)` first closes what an interrupted earlier turn left open (a call nobody answered, a request nobody replied to), then expands `@` mentions in the prompt, then hands the prompt to `loop.resume`.
3. The session then repeats the following until the loop says it is done.
   1. The conversation is stored.
   2. The capabilities of the `main` role's model are resolved, the system prompt and tool definitions are assembled ([0013](design/0013-thin-context.md)), and the budget says whether the conversation is within bounds, needs old tool results shrunk (micro-compaction), needs a summary (full compaction), or cannot fit at all.
   3. The main model is asked, under the retry policy, with the reply streamed to the UI when the model has native tool calls. A provider that says the request does not fit is answered with one full compaction and one more request.
   4. A model without native tool calls has its reply parsed for text-protocol calls ([0004](design/0004-text-tool-protocol.md)); one that wrote none that work is asked again, up to twice, without using a turn.
   5. `loop.step` returns `Done` (the reply held no calls, or the turn limit was reached) or `RunTools`.
   6. The executor runs the calls in two phases ([0010](design/0010-two-phase-executor.md)): it decides every call and asks about those that need it, then runs them. `loop.observe` records the results, which become the next user message.
4. On `Done`, the session records what it used and spent in the store, and the front end prints the reply.

## Tools

A tool has a name, a `read_only` flag, a `spec()` (what the model is told) and two methods that must stay separate. `permission_request(ctx, arguments)` says what a call would touch (resolved paths, the command) and changes nothing; `run(ctx, call_id, arguments)` does the work, and is reached only after the policy has seen the answer to the first. A tool that resolved its own paths inside `run` would have moved the sandbox check to after the fact.

| Tool | Read-only | Notes |
| --- | --- | --- |
| `Read` | yes | Numbered lines, 2,000 lines by default, a file over 1 MB or not UTF-8 text is refused. Records a stamp of what it saw. |
| `Edit` | no | Exact string replacement, a batch that applies whole or not at all ([0007](design/0007-edit-by-string-not-lineno.md)). |
| `Write` | no | Creates a file, or replaces one that was read in this session and has not changed since. Atomic. |
| `Glob` | yes | Most recently modified first, at most 200, skips what `.gitignore` skips. |
| `Grep` | yes | ripgrep when it is on `PATH`, otherwise a slower search written in Python that answers the same. Three output modes. Matches in credentials files are dropped. |
| `TodoWrite` | no | The session's own task list, at most one task in progress. It changes nothing outside the session, and is allowed without asking. |

What a tool returns is stripped of terminal control sequences before it goes anywhere, so that a file cannot move the cursor or retitle the window of whoever reads the screen.

## Permissions

`permissions/policy.py` holds one pure function, `evaluate(request, policy, grants)`. It does no I/O: the paths in a request have already been resolved, because following a symlink after the check is how sandboxes are escaped. The checks run in a fixed order and the first that matches decides, which is the security model, so it is written once, as a sequence of early returns. The five before `mode.bypass` are the hard denials, which no mode, rule or grant gets past ([0015](design/0015-bypass-does-not-disable-the-sandbox.md)).

The rule id of the decision is what a refusal quotes (`refused (sandbox.outside-root): ...`), what `/status` shows and what the audit table stores, so the ids are a contract. These seventeen exist:

| Rule id | Decision | Meaning |
| --- | --- | --- |
| `rule.deny` | deny | Your own `deny` rule matched. |
| `secret.path` | deny | A path that is a credentials file by convention. |
| `sandbox.outside-root` | deny | The resolved path is outside every sandbox root. |
| `sandbox.symlink-escape` | deny | A write whose parent directory resolves outside the sandbox. |
| `bash.dangerous` | deny | The danger classifier blocked the command. |
| `bash.unparseable` | deny | The command could not be parsed, so it cannot be judged. |
| `mode.plan-read-only` | deny | A write in `plan` mode. |
| `mode.bypass` | allow | `bypass` mode, after the hard denials. |
| `mode.accept-edits` | allow | `Edit` or `Write` in `accept-edits` mode. |
| `rule.allow` | allow | Your own `allow` rule matched. |
| `grant.session` | allow | You answered "always" for this tool earlier in the session. |
| `tool.read-only` | allow | `Read`, `Grep`, `Glob` or `TodoWrite`. |
| `rule.ask` | ask | Your own `ask` rule matched. |
| `default.ask` | ask | Nothing else decided; changing state asks. |
| `tool.unknown` | deny | The tool name is not registered. Decided before the policy runs. |
| `tool.bad-arguments` | deny | The arguments do not make a request. Decided before the policy runs. |
| `tool.internal-error` | deny | The tool failed while it was being asked what the call would touch. |

A rule id that the original plan reserved, `hook.blocked`, is not in the table: it arrives with hooks, in a later version.

**Two extensions of the plan.** The plan listed seventeen rule ids, of which `hook.blocked` is the one not yet built, so sixteen are here. `tool.internal-error` is an eighteenth, added because none of the others describes a bug: a call whose tool broke while its permission request was being computed never ran, and the model should not be told that its arguments were wrong when they were not. And the store has a table the plan did not list, `messages_archive(session_id, seq, role, blocks_json, created_at, archived_at)`, which holds every stored message that a compaction or `/clear` replaced, so that the promise that the record is complete and is never rewritten holds. (The plan's fourth table, `checkpoints`, waits for checkpoints.) `messages` is the conversation the model is shown, which a resume loads, and compaction has to replace its rows; the replaced ones are moved to the archive, with when, in the same transaction.

**The sandbox.** A path is resolved (symlinks followed, `~` expanded) and compared with the roots by path component and not by string prefix, so a directory called `proj-evil` is not inside one called `proj`. A write also resolves its parent. The comparison is case-sensitive, so on a case-insensitive file system a differently-cased spelling of a path that is inside can be refused; it cannot be made to admit one that is outside.

**The danger classifier.** The interface is `classify(command) -> DangerVerdict`, with levels `safe`, `blocked` and `unparseable`. Only the regex classifier exists: sixteen patterns for the commands that destroy things (`rm -rf`, `dd` to a device, `mkfs`, a fork bomb, `git push --force`, `git reset --hard`, piping a download to a shell, `sudo`, and so on), run over the command with quoted strings blanked out. It lists its own blind spots in `KNOWN_BLIND_SPOTS` (word splitting, command substitution that builds the verb, variable indirection, quoting, an interpreter's `-c`) and a test checks that each really is one. Nothing calls it yet, because Bash is not built. When it is, a verdict of `safe` from this classifier has to mean "no pattern matched" and not "this is safe": the policy refuses to let `allow` rules or session grants promote a `safe` verdict that is marked not authoritative, but the regex classifier does not mark its verdicts so yet, which the shell tool's task has to do.

**Redaction** is in [0012](design/0012-redaction-before-transcript.md).

**The audit record** is the `tool_calls` table. The decision is written before the call runs and the outcome (`ok`, `error`, `refused` or `declined`, a duration, the bytes returned) is filled in afterwards, so a row without an outcome means the process ended between the two. Arguments are scrubbed before they are stored. Nothing reads the table back in this version.

## Conversation, compaction and the store

A transcript is a sequence of messages of blocks (text, thinking, tool use, tool result), and its validity is a property of the data, checked wherever a transcript crosses a boundary: every tool use is answered by a tool result in the next message, and roles alternate. It is immutable.

**Budget.** The room for a conversation is the model's window, less the longest reply it may write, less a tenth of the window. Tokens are estimated at 3.5 characters each, for the budget only; the real counts come back from the provider and are what `/cost` shows. At `compact_soft` (0.70 of the room) the content of old tool results is replaced by one line, leaving every block in place; at `compact_hard` (0.85) everything older than the last `keep_recent_turns` turns is replaced by a summary. If the system prompt and tool definitions do not fit in the room alone, nothing can be done and the error says which part is too big.

**The summary** is requested from the `compact` role's model, with this structure: GOAL, DECISIONS, CHANGED, OPEN, FAILED, followed by a line `FILES SEEN` that lists the paths found in the tool arguments of what was replaced. Three things survive both kinds of compaction, and each has a test: the last user message, the pairing of tool uses and results, and the set of files touched. A summary that cannot be had stops the turn with an error and leaves the history as it was; there is no fallback text.

**The store** is SQLite in WAL mode, with the tables `sessions`, `messages`, `messages_archive` and `tool_calls`. A session is written as it happens: at every turn the store holds what the model is about to be shown. A resumed session adds to the stored rows and rewrites none; a second process that resumes the same session finds out at the first message that differs, and stops with a message instead of splicing the rows. `total_cost_usd` is `NULL` when it is not known, and never `0` in its place.

## Providers

A provider adapter turns a `ModelRequest` into the vendor's request and a stream of the vendor's events into one `ModelReply`, through one method, `complete(request, on_text=None)`. Nothing above the adapters branches on a provider name ([0003](design/0003-capability-negotiation.md)). The three are `anthropic` (Messages API, cache breakpoints on the system prompt and the tool list), `openai_compat` ([0014](design/0014-openai-compat-first-class.md)) and `ollama` (native API, with a probe of `/api/show`).

What a model can do is a `Capabilities` record, resolved from a table, a cache, a probe and a default, and overridable in the configuration. A request sends no temperature.

**Errors** are classified once, in `providers/retry.py`, for all three adapters.

| What happened | What is done |
| --- | --- |
| 408, 425, 429, 500, 502, 503, 504, 529 | Retried, up to five attempts in all, after a delay that starts at one second, doubles, is jittered and is capped at thirty. The UI is told of each retry. |
| The connection failed or timed out before any reply | Retried, up to three attempts in all. |
| 401 or 403 | Not retried. Reported as a credentials problem, pointing at `ncc init`; exit code 3. |
| 3xx | Not followed. The address is probably wrong, or something is answering in the provider's place. |
| 404 | Not retried. |
| 400 that says the request does not fit the window | One full compaction and one more request, then the error. |
| Any other 4xx | Not retried. The provider's own words are passed on. |
| The stream broke after part of the reply arrived, or ended without its end marker | The text that arrived is kept as a message marked cut off. Not retried: the request may already have had effects. |

A retry is shown as it happens and is not stored: the audit table is keyed by tool call and has no row for a request.

## Context

[0013](design/0013-thin-context.md) describes what a request carries. `NANO.md` is layered from your home, then the project root, then each directory down to where you are; `CLAUDE.md` is read in a directory with no `NANO.md`.

## Configuration

Two files, merged key by key, validated on load, with the trust rules of [configuration.md](configuration.md). Each file is checked on its own, before the merge, so that a message can name the file that holds the mistake. Nothing in a file is silently ignored: a key or section that nothing reads is an error.

## Where v0.1 differs from the plan

These are differences between the original design and what is built. The code is right.

- Redaction and the `--output-format json` mode are here and were scheduled for later versions.
- No `ncc audit`, `ncc gc`, `ncc bench`, checkpoints table or banner for text-tool mode. Only the `main` and `compact` roles are sent requests.
- Search uses ripgrep from `PATH` or a pure Python fallback. No ripgrep is bundled.
- Only Ollama is probed; an OpenAI-compatible model the table does not list gets the conservative default.
- Four of the seven capability fields are not acted on ([0003](design/0003-capability-negotiation.md)).
- Writes are serialised by a barrier and not by a lock for each path ([0010](design/0010-two-phase-executor.md)).
- An `allow` rule for one path cannot open a credentials file; only `--allow-secrets` can. The entropy floor of the generic secret rule is 3.5 bits per character.
- "Always" grants a tool, not a command prefix.
- A flag `--root` exists for the project directory.
- A slash command `/exit` exists. The commands for agents, MCP servers, rewinding, diffs and checkpoints do not.
- `TodoWrite` is in the default `ask` list of the configuration and is allowed all the same, because `tool.read-only` is checked before `rule.ask`.
