# 0001: Scope

Status: accepted. Applies to v0.1.

This is what v0.1 does not do. Each line says why it waits, or why it is not a goal at all. The README repeats this list word for word, and a test keeps the two the same.

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

## Why

A release is judged by whether a stranger can install it, use it, and not lose a repository. The number of features it lists does not enter into that. v0.1 therefore keeps what that needs: the agent loop, tools that read and change files, four permission modes with a sandbox under them, three provider adapters with capability negotiation, compaction, stored sessions, cost accounting and a terminal front end.

What waits does so for one of three reasons. It needs something that is not built yet (background commands need the shell tool). It opens a trust boundary that has to be designed whole (hooks, MCP, plugins, WebFetch). Or it is a product of its own (agent teams, a marketplace). The last group is not scheduled at all: those items were weighed and left out, and the list says so in the same words, so that nobody reads an omission as an oversight.

Two things are decided by absence rather than by a flag. A configuration file that names a section this version does not read is refused (`hooks`, `mcp`, `verify` and `tools` are all refused today), because a guard that is silently ignored is worse than one that is not there. And a slash command for a subsystem that does not exist is not stubbed: `/rewind` and `/mcp` are not commands at all.

## Costs

Someone who wants undo, formatting after every edit or a second model reviewing a change cannot have it from this release. The first is the largest gap: with no checkpoints, the protection against a bad edit is the confirmation prompt and the sandbox, plus whatever the person's own git history holds.

The list is also a promise to keep it current. Each release that builds one of these items has to remove it from here and from the README in the same change, and nothing but review makes that happen.

Dropping the marketplace, the skill system and IDE extensions costs reach. Each is a way people find and extend a tool, and this one will be found by `pip` and by word of mouth.

## Rejected alternatives

- Ship every item in one release. Each version is budgeted in weeks, and a release that arrives late, or arrives with half of a trust boundary, serves nobody.
- Stub the missing commands, so that `/rewind` prints "not implemented". A command that exists and does nothing teaches people to stop reading the help, and the stub then has to be removed.
- Accept configuration for features that are not built and ignore it. A person who writes a hook and sees nothing happen has been told something false. Refusing the section is louder and costs one line.
- Leave the list out of the README and let the version history say it. A person deciding whether to try the tool reads the README and not the changelog.
