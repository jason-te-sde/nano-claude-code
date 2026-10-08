# 0005: Six roles, and why explore is the one that pays

Status: accepted. The slots and their routing are built; only two of them are consulted yet.

A session does not use one model but a cast of them. A role is a slot a model fills: `main` for the conversation, and `explore`, `plan`, `verify`, `compact` and `title` for the jobs around it. A configuration file maps each role to a model entry in its `[roles]` table, a role it leaves out follows `main`, and `main` is the first model defined when the file does not say. `--model <alias>` replaces `main` for one run, and `--role <role>=<alias>` replaces one role, as many times as it is given.

## Why

Reading a codebase to find what to change is much of the tokens in a session and little of its difficulty. Sending that to a cheap or a local model, while the model that writes the change stays strong, is the saving that a tool with a neutral provider boundary can offer and one wired to a single vendor cannot. The router holds one client for each model and not each role, so two roles on one model share a connection, and it keeps a ledger by role: `/cost` shows what each role used and cost, priced at the model that made each call, so switching a model mid-session does not reprice what was already spent.

A role is not a sub-agent. A role is a model slot ("which model does this job"). A sub-agent, when it arrives in v0.3, will be a persona: a system prompt, a set of tools and the role whose model it uses. The compaction summary uses the `compact` role's model although there is no sub-agent called compact.

A typo is caught early. A role that points at a model nobody defined is refused when the configuration is loaded, and a misspelled role name is refused with the list of real ones, because a mistyped `explore` would otherwise leave exploration on the expensive model without a word. `--role` is checked again when the router is built, before the first request.

Pinned by `test_each_role_can_be_routed_to_its_own_model`, `test_every_role_defaults_to_main_when_unset`, `test_a_role_pointing_at_an_undefined_model_is_rejected_at_load`, `test_a_misspelled_role_is_refused_with_the_roles_that_exist`, `test_usage_is_attributed_per_role` and `test_a_role_that_changes_model_is_priced_call_by_call`.

## Costs

In v0.1 two roles are consulted. `main` answers the conversation and `compact` writes the summary when a conversation is compacted. `explore`, `plan`, `verify` and `title` are accepted, checked and routed, and then nothing sends them a request: exploring is done by the main model's own calls to Grep and Glob, and the sub-agents, the post-edit verification and the session titles that would use the other four are later work. Pointing `explore` at a cheap model today changes nothing. Pointing `compact` at one already saves money, because the summary request carries the whole older history.

That request can also fail for a reason the main model would not hit: a summary asked of a model with a small window may not fit the history. The session reports it as the compact role's error and names the role, and the history is kept intact. Pinned by `test_a_summary_that_fails_stops_the_turn_and_leaves_the_history_alone` and `test_an_overflow_on_the_summary_request_is_the_compact_roles_error_not_a_retry`.

The saving has not been measured. How much of a session's tokens go to reading, and how much a cheaper reader costs in the quality of what it reports back to the main model, are open questions that `ncc bench` is meant to answer.

More roles mean more entries to write and more ways for a configuration to be wrong, which is why `ncc init` writes every role on one model and comments on the rest.

## Rejected alternatives

- One model per session. It is the simplest thing to explain, and it gives up the saving that is the reason to route at all.
- Routing by tool, so that Grep and Glob always go to a cheap model. A tool does not call a model; the model decides to use the tool, and the question is who reads the results.
- A classifier that picks the model for each request. It adds a model call, and so a cost and a failure mode, to every request, to decide how to spend less on the others.
- Making a role the same thing as a sub-agent. They will meet when sub-agents arrive, but compaction needs a model slot and no persona.
