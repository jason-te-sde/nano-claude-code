# 0013: Thin automatic context, and no automatic retrieval

Status: accepted, implemented in `context/`.

A request carries a small, fixed amount of context that the agent assembles itself, and everything else is something the model asks for. The system prompt is static. Project instructions come next: `NANO.md`, or `CLAUDE.md` in a directory that has no `NANO.md`, from the project root down to the working directory, preceded by `NANO.md` in the `.nanoclaude` directory of the home (each file is cut at 32,000 bytes). Then an environment block: the working directory, the git branch with a count of files that have uncommitted changes, and a map of the project. The map goes three levels deep, hides what `.gitignore` hides, stops at 200 entries and says that it stopped, and does not enter a symlink that points outside the project. After that, the conversation, and a file the person names with `@path` in a prompt, which is inlined where it was mentioned, up to 100,000 bytes, scrubbed first, and only when it is inside the project and is not a credentials file.

## Why

The model decides what to read. It knows better than any retrieval step what the next question is, and `Grep` and `Glob` let it ask. An automatic step that injects "related" code is mostly wrong: it fills the window with plausible text that dilutes attention, it cannot be steered by the model, it costs tokens on every turn, and it is one more subsystem that has to be built and kept right. So the context is thin, the search tools are strong, and the person has a direct way to say "this file", with `@`.

The first part of the request does not change from turn to turn, and that is deliberate. A provider that caches a prompt by prefix, explicitly or automatically, only helps if the prefix is byte for byte what it was. So the system prompt has no timestamp, and the tools are listed in a fixed order (`ToolRegistry.specs` sorts them by name). `test_the_system_prompt_is_identical_across_two_assemblies` and `test_git_state_is_in_the_environment_block_not_the_system_prompt` pin the first, and `test_the_tool_list_is_byte_stable_across_turns` and `test_the_request_prefix_is_byte_stable_across_turns` in the Anthropic tests pin the second.

The mention rules are about what must not happen. An `@` that is not a file inside the project is left as the person wrote it, since an unexpanded mention is confusing and a silently expanded `/etc/passwd` is worse (`test_a_mention_outside_the_root_is_not_expanded`, `test_a_symlink_inside_the_project_to_a_file_outside_is_not_expanded`). A mention of a credentials path is left alone unless `--allow-secrets` is on, because the content rules recognise shapes and not every line a credentials file can hold (`test_a_secret_path_mention_is_left_as_written_by_default`). An email address is not a mention.

## Costs

The model has to search, and searching is turns: finding where something is defined takes a `Grep` and then a `Read`, which a retrieval step would have saved. Nothing finds code by meaning; a query that does not match the text of the code finds nothing.

The environment block is joined to the same system string as the rest, so when the git state changes (a file edited, a branch switched) the system text changes and a provider's cache of it is missed. The tool list, which comes first in the request, keeps its own breakpoint and still hits. The separation in `context/assemble.py` keeps the block out of the instructions; it does not keep it out of the system string.

The project map costs tokens on every request whether or not it is needed, up to 200 lines, and a repository with a deep layout is shown only to the third level. It also names credentials files: it hides what `.gitignore` hides and applies no rule of its own, so a `.env` or an `id_rsa` that is not ignored appears in the map by name (run against a directory holding both, and a `.aws/credentials`, it lists all of them). Names only, never content, and the model still cannot read them; but the names go to the provider.

Instructions are read from the disk at every request. A change to `NANO.md` mid-session takes effect at the next prompt, which is a feature, and an enormous file is truncated and not refused, with a marker.

## Rejected alternatives

- Embeddings and a retrieval step that injects the nearest files. It is the usual answer and it needs an index, a model and a policy for stale files, to produce context the model did not ask for.
- A model-written summary of the repository, kept in the context. It goes stale on the first edit and makes the first request expensive.
- No project map at all. The model would then start every session with `Glob` calls to find out what is there, and the map is two hundred lines at most.
- Inlining every file the model has read into every later request. It is the conversation already; `Read` results are in the transcript and compaction shrinks the old ones.
