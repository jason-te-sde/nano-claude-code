"""The rule grammar used by the ``allow`` / ``ask`` / ``deny`` config lists.

Three shapes:

* ``Tool`` -- every call to that tool.
* ``Tool(subject)`` -- an exact match on the tool's rule subject: the path for
  file tools, the whole command for ``Bash``, the subcommand for ``Git``.
* ``Tool(prefix:*)`` -- the subject begins with ``prefix`` at a word boundary,
  so ``Bash(npm test:*)`` covers ``npm test -- --watch`` but not ``npm testify``.

For path-taking tools the subject is also treated as a glob and matched against
both the absolute resolved path and the path relative to the sandbox root, so a
user can write either and be understood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pathspec

RuleKind = Literal["allow", "ask", "deny"]


@dataclass(frozen=True, slots=True)
class Rule:
    tool: str
    subject: str | None
    prefix: bool
    source: str

    @staticmethod
    def parse(text: str) -> Rule:
        raw = text.strip()
        if "(" not in raw:
            if ")" in raw:
                raise ValueError(f"unbalanced parentheses in rule {text!r}")
            return Rule(raw, None, False, raw)
        if not raw.endswith(")"):
            raise ValueError(f"unbalanced parentheses in rule {text!r}")
        tool, _, rest = raw.partition("(")
        subject = rest[:-1]
        if subject.endswith(":*"):
            return Rule(tool.strip(), subject[:-2], True, raw)
        return Rule(tool.strip(), subject, False, raw)

    def matches(self, tool: str, subject: str, relative_subject: str) -> bool:
        if tool != self.tool:
            return False
        if self.subject is None:
            return True
        if self.prefix:
            return _prefix_match(subject, self.subject)
        if subject == self.subject:
            return True
        return _glob_match(self.subject, (subject, relative_subject))


def _prefix_match(subject: str, prefix: str) -> bool:
    """Prefix at a word boundary: `npm test` covers `npm test -w`, not `npm testify`."""
    if not subject.startswith(prefix):
        return False
    rest = subject[len(prefix) :]
    return rest == "" or rest[0].isspace()


def _glob_match(pattern: str, candidates: tuple[str, ...]) -> bool:
    spec = pathspec.PathSpec.from_lines("gitwildmatch", [pattern])
    return any(spec.match_file(candidate.lstrip("/")) for candidate in candidates) or any(
        spec.match_file(candidate) for candidate in candidates
    )


@dataclass(frozen=True, slots=True)
class RuleSet:
    allow: tuple[Rule, ...]
    ask: tuple[Rule, ...]
    deny: tuple[Rule, ...]

    @staticmethod
    def build(
        *,
        allow: list[str] | None = None,
        ask: list[str] | None = None,
        deny: list[str] | None = None,
    ) -> RuleSet:
        return RuleSet(
            tuple(Rule.parse(r) for r in allow or []),
            tuple(Rule.parse(r) for r in ask or []),
            tuple(Rule.parse(r) for r in deny or []),
        )

    def first_match(
        self, kind: RuleKind, tool: str, subject: str, relative_subject: str
    ) -> Rule | None:
        # Annotated rather than left for inference: getattr's return type is
        # Any, and returning that straight from a function declared -> Rule |
        # None is exactly the kind of leak mypy --strict's warn_return_any
        # exists to catch. The annotation below is a type assertion, not a
        # behavior change -- kind is restricted to "allow" / "ask" / "deny" by
        # the RuleKind literal, and each of those names a tuple[Rule, ...]
        # field on this dataclass.
        rules: tuple[Rule, ...] = getattr(self, kind)
        for rule in rules:
            if rule.matches(tool, subject, relative_subject):
                return rule
        return None
