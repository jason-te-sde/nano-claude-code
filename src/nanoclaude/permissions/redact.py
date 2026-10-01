"""Keeping credentials out of the transcript.

Two layers. Paths that are credentials by convention are refused outright by the
policy (rule ``secret.path``). Everything a tool returns then passes through
:meth:`Redactor.scrub` *before* it becomes a transcript block, so the persisted
session and anything restored by ``--resume`` are clean too -- redacting at
render time would leave the secret in the database and in the next request.

The generic rule is deliberately narrow: high entropy alone matches base64
images, git hashes and minified JavaScript. It fires only next to an assignment
whose name says secret. Everything else is a specific, recognisable shape.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from nanoclaude.permissions.rules import glob_matches_any

SECRET_PATH_PATTERNS: tuple[str, ...] = (
    "**/.env",
    "**/.env.*",
    "**/*.pem",
    "**/*.key",
    "**/id_rsa",
    "**/id_dsa",
    "**/id_ecdsa",
    "**/id_ed25519",
    "**/credentials",
    "**/credentials.*",
    "**/.aws/**",
    "**/.ssh/**",
    "**/*.p12",
    "**/*.pfx",
    "**/secrets.*",
)

_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b")),
    ("github-token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,}\b")),
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google-key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
)

#: NAME=value where NAME says secret and value looks random.
_ASSIGNED = re.compile(
    r"\b([A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|APIKEY|API_KEY|PRIVATE_KEY)[A-Z0-9_]*)"
    r"\s*[=:]\s*[\"']?([A-Za-z0-9+/=_\-]{16,})[\"']?"
)

ENTROPY_FLOOR = 3.5


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {c: value.count(c) for c in set(value)}
    total = len(value)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


@dataclass(frozen=True, slots=True)
class Redactor:
    enabled: bool = True

    def scrub(self, text: str) -> tuple[str, int]:
        if not self.enabled or not text:
            return text, 0
        replacements = 0
        for label, pattern in _SHAPES:
            text, hits = pattern.subn(f"[redacted:{label}]", text)
            replacements += hits

        def _assigned(match: re.Match[str]) -> str:
            nonlocal replacements
            value = match.group(2)
            if shannon_entropy(value) < ENTROPY_FLOOR:
                return match.group(0)
            replacements += 1
            return f"{match.group(1)}=[redacted:assigned-secret]"

        text = _ASSIGNED.sub(_assigned, text)
        return text, replacements

    def is_secret_path(self, absolute: str, relative: str) -> bool:
        # The shared matcher rather than a pathspec of its own: it already tries
        # each candidate with and without leading slashes, and it keeps the
        # deprecated pathspec factory name in exactly one place.
        return glob_matches_any(SECRET_PATH_PATTERNS, (relative, absolute))
