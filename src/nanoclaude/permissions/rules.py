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

A rule compares exactly, except when it is asked to refuse something: ``deny`` rules and
the credentials list compare exactly and then after :func:`fold_spelling` as well, because
the filesystems people use most open one file under every spelling that folds to the same
string, and a refusal that only knew one of them would be a refusal a different spelling
walks past. Folding only ever adds refusals. It is tried on top of the exact comparison and
not instead of it: ``?`` and a character class hold one character, and a name that folds to
two (the sharp s, a ligature, the dotted capital I) is one a pattern matched before it was
folded and does not match after. ``allow`` and ``ask`` rules compare exactly: a looser match
there would widen what runs without asking, which is the wrong direction to err in.
"""

from __future__ import annotations

import functools
import unicodedata
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


#: Every tool name a rule may name: the six registered tools and two that are being built.
#: ``Bash`` and ``Git`` are reserved, so that a rule written for them today does not have to
#: be rewritten the day they arrive. Names are case-sensitive, as the registry's are: a rule
#: that names ``read`` matches no call to ``Read``, and for a deny rule that is a refusal
#: the person believes is in force and is not. A test pins this list to the registry.
KNOWN_TOOLS = ("Bash", "Edit", "Git", "Glob", "Grep", "Read", "TodoWrite", "Write")


def fold_spelling(text: str) -> str:
    """``text`` in the form two spellings of one file share, for comparing in a refusal.

    The default macOS volume (APFS) is case-insensitive and normalization-insensitive:
    ``.ENV`` opens ``.env``, and a name in NFD opens the file made under its NFC spelling.
    Both sides of a comparison go through this, so any pair of spellings the filesystem
    treats as one compare equal. It is Unicode's canonical caseless matching (the
    decomposed form is case-folded, then composed again), and it is only ever used to refuse:
    on a case-sensitive filesystem it can refuse a file that is distinct from the one a rule
    names, which errs towards refusing. It is never the only comparison a refusal makes: a
    folded name can be longer than the one it came from, and a pattern that held the letter
    in one character (``?``, a class) no longer fits it, so what is compared as written is
    compared as well.
    """
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", text).casefold())


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
            return Rule(_known_tool(raw), None, False, raw)
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
            return Rule(_known_tool(tool.strip()), prefix, True, raw)
        return Rule(_known_tool(tool.strip()), subject, False, raw)

    def matches(
        self, tool: str, subject: str, relative_subject: str, *, fold: bool = False
    ) -> bool:
        """Whether this rule covers the call. ``fold`` also compares the subject (in all three
        of the rule's forms) after :func:`fold_spelling`, on top of comparing it as it is, so
        that it only ever covers more; the tool's name is compared as it is, since the
        registry's names are exact."""
        if tool != self.tool:
            return False
        if self.subject is None:
            return True
        if self._covers(self.subject, subject, relative_subject):
            return True
        return fold and self._covers(
            fold_spelling(self.subject), fold_spelling(subject), fold_spelling(relative_subject)
        )

    def _covers(self, rule_subject: str, subject: str, relative_subject: str) -> bool:
        """The rule's subject, in whatever spelling it is given, against the call's two."""
        if self.prefix:
            return _prefix_match(subject, rule_subject)
        if subject == rule_subject:
            return True
        return glob_matches_any((rule_subject,), (subject, relative_subject))


def _known_tool(name: str) -> str:
    """``name``, if a rule may name it. The same hazard as an empty subject, from the other
    side: ``read(secrets/**)`` parses, never matches a call to ``Read``, and is a deny rule
    that denies nothing."""
    if name not in KNOWN_TOOLS:
        raise ValueError(
            f"unknown tool {name!r}; tool names are case-sensitive and are {', '.join(KNOWN_TOOLS)}"
        )
    return name


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


def glob_matches_any(
    patterns: tuple[str, ...], candidates: tuple[str, ...], *, fold: bool = False
) -> bool:
    """True if any of ``candidates`` matches any of ``patterns``.

    ``fold`` also compares both after :func:`fold_spelling`: what a refusal asks for, so that
    every spelling of a file the filesystem treats as one matches the pattern that names it.
    It is on top of the comparison as given and not instead of it, so that folding only ever
    adds matches: ``?`` and a class hold one character, and what a pattern matched before the
    name was folded (a sharp s, a ligature) can be two characters after.

    Shared by :meth:`Rule.matches` (one pattern, a rule's own subject) and
    policy.py's secret-path check (many patterns, ``Policy.secret_paths``) --
    the one place this project turns a list of gitignore-style globs into a
    yes/no answer. Each candidate is tried both as given and with a leading
    slash stripped, so a caller working with absolute and root-relative
    spellings of the same path does not need to pick one.
    """
    if not patterns:
        return False
    if _spec_matches(patterns, candidates):
        return True
    if not fold:
        return False
    folded_patterns = tuple(fold_spelling(pattern) for pattern in patterns)
    folded_candidates = tuple(fold_spelling(candidate) for candidate in candidates)
    if folded_patterns == patterns:
        # The same patterns, so the same compiled spec, and only a spelling that folding
        # changed is a question that has not been asked: a credentials list is in lower case
        # and most paths are in it too, and asking about all of them twice doubled the cost
        # of every call that is judged.
        changed = tuple(c for c in folded_candidates if c not in candidates)
        return bool(changed) and _spec_matches(patterns, changed)
    return _spec_matches(folded_patterns, folded_candidates)


def _spec_matches(patterns: tuple[str, ...], candidates: tuple[str, ...]) -> bool:
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
        # Only what refuses is compared by what the filesystem would treat as the same
        # file; allow and ask rules are compared exactly (see the module docstring).
        fold = kind == "deny"
        for rule in rules:
            if rule.matches(tool, subject, relative_subject, fold=fold):
                return rule
        return None
