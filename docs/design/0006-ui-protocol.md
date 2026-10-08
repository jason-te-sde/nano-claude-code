# 0006: Why agent/ cannot import cli/

Status: accepted, enforced by a test.

Everything the agent needs to say to a person, or to ask one, goes through the `UI` protocol in `agent/ui.py`: `confirm` to ask whether a call may run, `on_decision` and `on_outcome` to report what was decided and what happened, `on_reply` and `on_text` for what the model said, `on_retry` for a provider that said no, `on_request_start` and `on_request_end` around a request, and `on_output`. The session is handed an implementation and calls it. The terminal is one implementation.

## Why

There are four implementations and the number is the argument. `ConsoleUI` in `cli/render.py` is the REPL's. `AutoDecline` is headless mode's: it answers every question with no, so that anything the policy wanted confirmed is refused when nobody can be asked. `SilentUI` and `AutoApprove` are the doubles the suite uses. The fourth is anybody who imports `nanoclaude` and writes a front end of their own. If `agent/` reached for `print` or a prompt, each of those would need a terminal, and a headless run in CI would need a fake one.

It is enforced and not only declared. `tests/test_boundaries.py` parses every module under `agent/` and fails if any of them imports from `nanoclaude.cli` (`test_agent_never_imports_the_cli`). A rule that lives in a docstring is broken three months later by the person who wrote it, for a reason that seemed good then.

Headless mode is not a reduced REPL. It is the same `Session` with a different UI, one that never asks, and that is the right behaviour where nobody is watching: approving writes because nobody is there to say no is exactly backwards. `test_headless_declines_what_the_policy_wanted_confirmed` pins it, and `test_accept_edits_lets_headless_write_files` pins the way to say yes in advance.

## Costs

Every method added to the protocol is a method that all four implementations have to grow, and one that a library user's does not have yet breaks them. `test_a_ui_that_does_not_take_the_streaming_calls_is_not_a_ui` pins that the protocol's check is real, and the suite iterates over the built-in implementations, but a front end outside the repository learns about a new method when it fails.

The executor hands the UI the request as the tool built it, with the path or the command exactly as the model wrote it, because the policy has to judge what would really run and a rule matched against a stripped copy is matched against something the shell never receives. So a UI must render what it is given safely itself; `ConsoleUI` shows each control character in a confirmation by name and does not print it (`test_the_subject_line_of_a_confirmation_shows_every_control_by_name`).

A question can only be asked between a model's reply and the tools it asked for. There is no way for a tool to ask something in the middle of its work, and the protocol would have to grow to allow one.

## Rejected alternatives

- Let `agent/` print and prompt, and give it a flag for non-interactive use. It is less code at the start, and every test and every library user then inherits the terminal.
- Pass the callbacks to the session one by one. The protocol is the callbacks, and passing them separately lets a front end supply some of them and forget the rest.
- An event stream the front end reads, in place of calls the session makes. A confirmation is a question with an answer the session must wait for, and a stream turns that into a conversation about which event answers which.
- Let the CLI import `agent/` and `agent/` import the CLI's types for convenience. The one type a front end has to return, `Approval`, lives in `agent/ui.py` and not in the CLI.
