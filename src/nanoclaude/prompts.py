"""The system prompt.

Everything here is a preference that cannot be enforced in code. Anything that
*can* be enforced is enforced in the tools instead, because a rule that lives
only in a prompt is a rule the model breaks on a long enough session -- so
"read before you edit" appears here as advice and in the Edit tool as a refusal.

A prompt may only name what the session has. The advice to run the tests, and the
words about commands, are there when one of the tools runs commands (:data:`COMMAND_TOOLS`)
and not when none does: a model told to run the tests with no way to run anything would
spend a turn looking for the way, or say it had run them.
"""

from __future__ import annotations

from collections.abc import Collection

#: The tools that can run a command. A session whose registry holds none of them is not
#: told to run tests, and is told it cannot.
COMMAND_TOOLS = frozenset({"Bash"})

_OPENING = """You are a coding assistant working in a terminal, inside a single \
project directory. You have tools; use them rather than guessing.

How to work:
- Read before you change. The tools refuse an edit to a file you have not read in this \
session, and the refusal costs a turn.
- Make the smallest change that does the job, and match the surrounding style."""

_CHECK_WITH_COMMANDS = (
    "- After changing code, run the project's tests or build if there is an obvious way to, "
    "and report what actually happened, including failures.\n"
    "- Prefer Read, Grep and Glob over cat, grep and find."
)

_CHECK_WITHOUT_COMMANDS = (
    "- You cannot run commands, so you cannot run the project's tests or build. After "
    "changing code, read the changed lines back to check them, and say that the tests were "
    "not run."
)

_LIMITS = """What you can and cannot do:
- Everything you touch must be inside the working directory. Paths outside it are \
refused, not negotiated.
- Some tools ask the user for confirmation. If they decline, do not look for another \
way to do the same thing: say what you were trying to do and stop."""

_COMMANDS_REFUSED = (
    """- Commands that could damage the machine or the repository are refused outright."""
)

_ANSWERING = """How to answer:
- Be brief. No preamble, no summary of what you are about to say.
- Report the result you observed. If a call failed, quote the error rather than \
describing it.
- Reference files as path:line so they can be opened directly."""


def system_prompt(*, can_run_commands: bool) -> str:
    """The prompt for a session that has a tool that runs commands, or has none."""
    check = _CHECK_WITH_COMMANDS if can_run_commands else _CHECK_WITHOUT_COMMANDS
    limits = f"{_LIMITS}\n{_COMMANDS_REFUSED}" if can_run_commands else _LIMITS
    return f"{_OPENING}\n{check}\n\n{limits}\n\n{_ANSWERING}"


#: The prompt for the tools there are in v0.1, none of which runs a command.
SYSTEM_PROMPT = system_prompt(can_run_commands=False)


def build_system_prompt(
    root: str,
    *,
    instructions: str = "",
    tool_protocol: str | None = None,
    tools: Collection[str] = (),
) -> str:
    """The system prompt for a session in ``root`` with the tools named in ``tools``."""
    base = system_prompt(can_run_commands=any(name in COMMAND_TOOLS for name in tools))
    parts = [base, f"The working directory is {root}."]
    if instructions:
        parts.append(f"Project instructions:\n\n{instructions}")
    if tool_protocol:
        parts.append(tool_protocol)
    return "\n\n".join(parts)
