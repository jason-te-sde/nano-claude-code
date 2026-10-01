"""The baseline classifier: regular expressions over the raw command string.

Shipped, but not trusted. Regular expressions cannot see through the shell's own
expansions, and every entry in :data:`KNOWN_BLIND_SPOTS` is a real way to write
a destructive command that the patterns below wave through. Task 8 adds a
syntax-tree classifier and a corpus that measures the gap.

When tree-sitter is unavailable this is the only classifier. Read its ``SAFE``
verdicts precisely: they mean "none of the patterns below matched", not "this
command is safe", and the two are different claims because
:data:`KNOWN_BLIND_SPOTS` exists. A verdict from this classifier must therefore
never be treated as having cleared a command for automatic approval.
"""

from __future__ import annotations

import re

from nanoclaude.permissions.danger import DangerLevel, DangerRule, DangerVerdict


def _rule(rule_id: str, pattern: str, description: str) -> DangerRule:
    return DangerRule(rule_id, re.compile(pattern, re.IGNORECASE), description)


RULES: tuple[DangerRule, ...] = (
    _rule("fork-bomb", r":\s*\(\s*\)\s*\{[^}]*\|[^}]*&[^}]*\}", "fork bomb"),
    _rule("rm.recursive-force", r"\brm\b[^|;&]*\s-[a-z]*(rf|fr)[a-z]*\b", "rm with -rf"),
    _rule("dd.device", r"\bdd\b[^|;&]*\bof=/dev/", "dd writing to a block device"),
    _rule("mkfs", r"\bmkfs(?:\.[a-z0-9]+)?\b", "filesystem creation"),
    _rule("redirect.device", r">\s*/dev/(?:sd|nvme|disk|hd)", "redirect onto a block device"),
    _rule("chmod.root", r"\bchmod\b[^|;&]*\s777\s+/\s*(?:$|[;&|])", "chmod 777 on /"),
    _rule("git.force-push", r"\bgit\b[^|;&]*\bpush\b[^|;&]*(?:--force|-f)\b", "force push"),
    _rule("git.reset-hard", r"\bgit\b[^|;&]*\breset\b[^|;&]*--hard\b", "discards local work"),
    _rule("git.clean-force", r"\bgit\b[^|;&]*\bclean\b[^|;&]*-[a-z]*f", "deletes untracked files"),
    _rule(
        "pipe-to-shell",
        r"\b(?:curl|wget)\b[^|]*\|\s*(?:sudo\s+)?(?:ba|z|k)?sh\b",
        "downloads and executes a remote script",
    ),
    _rule("base64-to-shell", r"\bbase64\b[^|]*\|\s*(?:ba|z|k)?sh\b", "executes decoded input"),
    _rule("sudo", r"(?:^|[\s;&|])sudo\b", "escalates privilege beyond the sandbox"),
    _rule("power", r"\b(?:shutdown|reboot|halt|poweroff)\b", "affects the whole machine"),
    _rule("sql.drop", r"\bdrop\s+(?:table|database|schema)\b", "destructive SQL"),
    _rule("find.delete", r"\bfind\b[^|;&]*\s-delete\b", "bulk deletion"),
    _rule("history.wipe", r"\bhistory\s+-c\b", "erases the audit trail"),
)

#: Ways to write a destructive command this classifier will not catch. Each one
#: becomes a case in the differential corpus in Task 8.
KNOWN_BLIND_SPOTS: tuple[str, ...] = (
    "word splitting: `rm$IFS-rf$IFS/` never contains the literal `rm -rf`",
    "command substitution: `$(echo rm) -rf /` hides the verb until the shell expands it",
    "variable indirection: `X=rm; $X -rf /`",
    "quoting: `r''m -rf /` and `r\\m -rf /` are both `rm` to the shell",
    "zsh EQUALS expansion: `=curl evil.com` resolves to the absolute path of curl",
    "aliases and functions defined earlier in the same command",
    "indirection through an interpreter: `python -c ...`, `perl -e`, `xargs`",
    "a script file whose contents are dangerous but whose invocation is not",
)


class RegexClassifier:
    name = "regex"

    def classify(self, command: str) -> DangerVerdict:
        hits = tuple(
            f"{rule.rule_id}: {rule.description}"
            for rule in RULES
            if rule.pattern.search(_strip_quoted_strings(command))
        )
        level = DangerLevel.BLOCKED if hits else DangerLevel.SAFE
        return DangerVerdict(level, hits, self.name)


_QUOTED = re.compile(r"""'[^']*'|"[^"]*\"""")


def _strip_quoted_strings(command: str) -> str:
    """Blank out quoted literals so `echo 'rm -rf /'` is not read as running it.

    Crude on purpose: it removes a whole class of false positives, and the
    false *negatives* it introduces (`sh -c 'rm -rf /'`) are covered by the
    interpreter blind spot, which the syntax-tree classifier handles properly.
    """
    return _QUOTED.sub("''", command)
