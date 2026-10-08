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

import functools
from dataclasses import dataclass
from typing import Literal

import pathspec
from pathspec.pattern import Pattern as PathspecPattern

RuleKind = Literal["allow", "ask", "deny"]

#: The pathspec pattern-matching dialect every glob in this project uses, named
#: once rather than repeated as a literal at each call site. pathspec >=1.0
#: renamed this factory to ``"gitignore"`` and turned uses of this name into a
#: DeprecationWarning (narrowly silenced in pyproject.toml's filterwarnings --
#: see that entry's comment); migrating to the new name is a pending decision
#: for the final review, alongside raising the pathspec floor, and the point of
#: this constant is to make that a one-line change instead of a search across
#: every file that matches a glob.
GLOB_PATTERN_FACTORY = "gitwildmatch"


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
        # Tool() parses without error and silently matches nothing, ever --
        # glob_matches_any on an empty pattern compiles to a spec with no
        # patterns at all, since a blank line is not a pattern in gitignore
        # syntax. For an allow or ask rule that is merely useless; for a deny
        # rule it is the dangerous direction, a rule a user believes is
        # refusing something that in fact never fires. Rejected here
        # alongside the unbalanced-paren cases rather than left to be
        # discovered at match time.
        if not subject:
            raise ValueError(f"empty subject in rule {text!r}")
        if subject.endswith(":*"):
            prefix = subject[:-2]
            if not prefix:
                raise ValueError(f"empty subject in rule {text!r}")
            return Rule(tool.strip(), prefix, True, raw)
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
        return glob_matches_any((self.subject,), (subject, relative_subject))


def _prefix_match(subject: str, prefix: str) -> bool:
    """Prefix at a word boundary: `npm test` covers `npm test -w`, not `npm testify`."""
    if not subject.startswith(prefix):
        return False
    rest = subject[len(prefix) :]
    return rest == "" or rest[0].isspace()


@functools.cache
def _compiled(patterns: tuple[str, ...]) -> pathspec.PathSpec[PathspecPattern]:
    # Cached rather than recompiled per call: both RuleSet and Policy are
    # frozen, so the set of distinct ``patterns`` tuples a process will ever
    # ask for is fixed once they are built -- typically once per CLI session
    # -- and compiling the same gitignore-style spec on every single
    # evaluate() call was pure waste. Keyed on ``patterns`` alone, not on the
    # candidates being matched, since the same compiled spec is reused across
    # every candidate checked against it.
    return pathspec.PathSpec.from_lines(GLOB_PATTERN_FACTORY, patterns)


def glob_matches_any(patterns: tuple[str, ...], candidates: tuple[str, ...]) -> bool:
    """True if any of ``candidates`` matches any of ``patterns``.

    Shared by :meth:`Rule.matches` (one pattern, a rule's own subject) and
    policy.py's secret-path check (many patterns, ``Policy.secret_paths``) --
    the one place this project turns a list of gitignore-style globs into a
    yes/no answer. Each candidate is tried both as given and with a leading
    slash stripped, so a caller working with absolute and root-relative
    spellings of the same path does not need to pick one.
    """
    if not patterns:
        return False
    spec = _compiled(patterns)
    # A newline becomes NUL, which no path holds and a glob treats as any other
    # character. The regular expression a leading ``**/`` becomes starts with ``.+``,
    # and ``.`` does not match a newline: without this a directory named ``a<newline>b``
    # hid ``a<newline>b/.env`` from ``**/.env``, and so from the credentials list and from
    # every deny rule written that way, with a name a repository can hold.
    flattened = [c.replace("\n", "\0") for c in candidates]
    return any(spec.match_file(c.lstrip("/")) or spec.match_file(c) for c in flattened)


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
