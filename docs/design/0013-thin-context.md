# 0013: Thin automatic context, and no automatic retrieval

Status: accepted, implemented in `context/`.

A request carries a small, fixed amount of context that the agent assembles itself, and everything else is something the model asks for. The system prompt is static for a given set of tools: it tells the model to run the project's tests only when one of its tools runs commands, and v0.1 has none. Project instructions come next: `NANO.md`, or `CLAUDE.md` in a directory that has no `NANO.md`, for each directory from the top level of the repository down to the project root, preceded by `NANO.md` in the `.nanoclaude` directory of the home. Each file is cut at 32,000 bytes and the chain at 64,000 bytes, and each has to pass the checks in [configuration.md](../configuration.md#instruction-files) that keep a hostile repository from putting a file of its choosing in the prompt. Then an environment block: the working directory, the git branch, and a map of the project. The map goes three levels deep, hides what `.gitignore` hides and what the policy would not let `Read` show, stops at 200 entries and says that it stopped, and does not enter a symlink that points outside the project. After that, the conversation, and a file the person names with `@path` in a prompt, which is inlined where it was mentioned, up to 100,000 bytes, scrubbed first, and only when it is inside the project and the policy would let `Read` show it.

## Why

The model decides what to read. It knows better than any retrieval step what the next question is, and `Grep` and `Glob` let it ask. An automatic step that injects "related" code is mostly wrong: it fills the window with plausible text that dilutes attention, it cannot be steered by the model, it costs tokens on every turn, and it is one more subsystem that has to be built and kept right. So the context is thin, the search tools are strong, and the person has a direct way to say "this file", with `@`.

The first part of the request does not change from turn to turn, and that is deliberate. A provider that caches a prompt by prefix, explicitly or automatically, only helps if the prefix is byte for byte what it was. So the system prompt has no timestamp, and the tools are listed in a fixed order (`ToolRegistry.specs` sorts them by name). `test_the_system_prompt_is_identical_across_two_assemblies` and `test_git_state_is_in_the_environment_block_not_the_system_prompt` pin the first, and `test_the_tool_list_is_byte_stable_across_turns` and `test_the_request_prefix_is_byte_stable_across_turns` in the Anthropic tests pin the second.

The mention rules are about what must not happen. An `@` that is not a file inside the project is left as the person wrote it, since an unexpanded mention is confusing and a silently expanded `/etc/passwd` is worse (`test_a_mention_outside_the_root_is_not_expanded`, `test_a_symlink_inside_the_project_to_a_file_outside_is_not_expanded`). A mention of a file the policy would refuse a `Read` of (a credentials path, unless `--allow-secrets` is on, or a file a `Read` `deny` rule covers) is not inlined, and one line after it says that it was not and which rule refused it. The content rules recognise shapes and not every line a credentials file can hold, and the person who wrote `@path` believed the model would see the file (`test_a_credentials_path_mention_says_why_it_was_not_inlined`, `test_a_file_a_read_deny_rule_covers_is_not_inlined_and_the_text_says_why`). An email address is not a mention.

## Costs

The model has to search, and searching is turns: finding where something is defined takes a `Grep` and then a `Read`, which a retrieval step would have saved. Nothing finds code by meaning; a query that does not match the text of the code finds nothing.

The environment block is joined to the same system string as the rest, so when the git branch or the project map changes (a branch switched, a file added) the system text changes and a provider's cache of it is missed. The tool list, which comes first in the request, keeps its own breakpoint and still hits. The separation in `context/assemble.py` keeps the block out of the instructions; it does not keep it out of the system string.

The project map costs tokens on every request whether or not it is needed, up to 200 lines, and a repository with a deep layout is shown only to the third level. It leaves out what the policy would not let `Read` show, so the name of a `.env` or an `id_rsa` is not in it. A directory whose contents are refused (`.ssh/`, `.aws/`, one that a `Read(dir/**)` rule covers) is listed by its own name with nothing in it, and `Glob`, which the model can call, asks the policy nothing and lists the names of those files too. Names only, never content; but the names go to the provider.

Instructions are read from the disk at every request. A change to `NANO.md` mid-session takes effect at the next prompt, which is a feature, and an enormous file is truncated and not refused, with a marker. Each request also starts git twice, for the branch and for the top of the repository, each held to five seconds; when git cannot answer there is no branch line and only the project root's own instruction file is read.

## Rejected alternatives

- Embeddings and a retrieval step that injects the nearest files. It is the usual answer and it needs an index, a model and a policy for stale files, to produce context the model did not ask for.
- A model-written summary of the repository, kept in the context. It goes stale on the first edit and makes the first request expensive.
- No project map at all. The model would then start every session with `Glob` calls to find out what is there, and the map is two hundred lines at most.
- Inlining every file the model has read into every later request. It is the conversation already; `Read` results are in the transcript and compaction shrinks the old ones.
