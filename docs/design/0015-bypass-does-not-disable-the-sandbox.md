# 0015: Bypass skips the question, not the protections

Status: accepted. A deliberate difference from the tool this project is modelled on.

`--dangerously-skip-permissions` puts a session in `bypass` mode. In that mode nothing asks for confirmation. What it does not do is switch off the checks that decide whether a call may happen at all. The policy is a fixed list of checks and the first one that matches decides. Bypass is the sixth, and the five before it are the hard-denial band: a rule of the person's own with `deny`, a credentials path, a path outside the sandbox or a write through a symlink that leaves it, a command the danger classifier refuses or cannot parse, and a write in plan mode. No mode and no rule gets past them.

## Why

The flag exists for one use: running in CI, where nobody can answer a prompt. It does not exist to let the model delete the machine, and the two wishes should not share a switch. If "I do not want to be asked" also meant "I have turned off every protection", a person who only wanted to stop being interrupted would have switched off the path sandbox, the credentials rule and the refusal of destructive commands in the same breath, without being told. The three are separate on purpose. Prompts are skipped by the mode. The sandbox is widened by naming a directory with `--add-dir`, an explicit act with a path in it. The credentials rule is switched off by `--allow-secrets`, which has its own warning.

Reaching bypass is made hard to do by accident, and each of the routes below is pinned.

- The long flag is the only way in. `--mode bypass` alone is refused with a message that names the flag, and no abbreviation of the flag is accepted (`test_mode_bypass_alone_is_refused_and_nothing_runs`, `test_dangerous_flag_requires_the_long_spelling`, `test_no_other_spelling_of_the_flag_turns_permission_prompts_off`).
- The flag beside a mode that asks for protection is two answers to one question, and is refused and not settled in favour of the flag (`test_a_mode_that_asks_for_protection_cannot_be_combined_with_turning_it_off`).
- Neither the home configuration nor a project's can turn it on (`test_a_configuration_file_cannot_turn_permission_prompts_off`, `test_a_project_configuration_cannot_either`).
- `/mode bypass` inside a session is refused and says to restart with the flag (`test_bypass_cannot_be_entered_from_inside_a_session`).
- Starting in it prints a warning to stderr saying what is still enforced (`test_turning_prompts_off_says_so_on_stderr_and_still_keeps_the_sandbox`).

What stays on is pinned too: `test_bypass_skips_confirmation_but_not_the_hard_denials` for the policy, and `test_a_deny_rule_in_the_configuration_is_enforced_even_with_prompts_off` and `test_a_deny_rule_in_the_projects_own_configuration_is_enforced_too` for a person's own rules.

For comparison: Claude Code's documentation, as read on 2026-10-07, describes its bypass mode as running everything without asking, with deny rules still applied, and recommends it for isolated containers and virtual machines. This project makes a narrower promise and keeps the sandbox on in that mode.

## Costs

A person who wants a bypassed run to touch another directory has to say which, with `--add-dir`. A person who really does want a destructive command run has to run it themselves: the classifier's refusal is not something a flag overrides, and "refused" is the final answer for that call even in CI.

An `ask` rule has no effect in bypass, and neither has an `allow` rule: nothing asks, and the allowance is moot. A person who wrote `ask` for something as a precaution does not get it back by starting with the flag.

Today the protection that matters most under bypass is the sandbox and the credentials rule, since the tools that exist are the ones that read and write files. The command classifier, the other half of the hard-denial band, has nothing to classify until the shell tool is built, and the regex classifier that exists is the weaker of the two planned (see the safety section of the README).

## Rejected alternatives

- Bypass as a switch for everything, so that it skips the sandbox, the credentials rule and the classifier as well as the prompt. It is the simplest definition and the one in which "stop interrupting me" and "turn the guards off" cannot be told apart.
- Bypass as a configuration setting. A cloned repository could then turn off its own confirmation prompts, which is the thing the flag's long name is there to prevent.
- A flag that skips prompts for edits only, and keeps asking about commands. It is `accept-edits`, which is a mode of its own.
- A confirmation, "are you sure?", when the flag is given. In CI there is no one to answer it, which is the case the flag is for.
