"""Keeping credentials out of the transcript.

Two layers. Paths that are credentials by convention are refused outright by the
policy (rule ``secret.path``). Everything a tool returns then passes through
:meth:`Redactor.scrub` *before* it becomes a transcript block, so the persisted
session and anything restored by ``--resume`` are clean too -- redacting at
render time would leave the secret in the database and in the next request.

The generic rule is deliberately narrow: high entropy alone matches base64
images, git hashes and minified JavaScript. It fires only next to an assignment
whose name says secret. Everything else is a specific, recognisable shape.

What this does *not* cover, so that nobody relies on it for more:

* A value is only recognised after a name that says secret, and only when it is at
  least 16 characters of letters, digits and ``+/=_-``, and random enough
  (:data:`ENTROPY_FLOOR`). A short password, one with punctuation in it (``.``, ``!``,
  ``@``), or one with spaces in it is left as it is.
* A secret in a position no name labels -- a bare string in a list, an argument on a
  command line, a line in a file of values -- is only caught if it has one of the
  recognisable shapes above.
* A name is read as a secret's when it is in capitals and has the keyword anywhere in
  it, or in any other case and has the keyword as a word of its own (``api_key``,
  ``clientSecret``, ``Password``). ``dbpassword`` in lower case, with no break to find the
  word by, is not read as one.
* A private key block that is cut off is masked as far as what is left looks like a key
  (its base64 lines and its two headers); a key whose lines were rewrapped, indented or
  interleaved with other text is not.
* The paths in :data:`SECRET_PATH_PATTERNS` are conventions, not a scan: a credentials
  file under another name is read like any other file, and its contents are then
  scrubbed by the shapes above and nothing more.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import pairwise

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
    # Files that hold a login for a tool people run every day.
    "**/.envrc",
    "**/.netrc",
    "**/.npmrc",
    "**/.pypirc",
    "**/.git-credentials",
    "**/.docker/config.json",
    "**/.kube/config",
    # Keystores, and the state of an infrastructure that holds its secrets in the clear.
    # Certificates (*.crt) and public keys (id_*.pub) are public and are not here.
    "**/*.jks",
    "**/*.keystore",
    "**/*.tfstate",
    "**/*.tfstate.backup",
)

_PEM_BEGIN = r"-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----"
_PEM_END = r"-----END [A-Z ]*PRIVATE KEY(?: BLOCK)?-----"
#: One break between lines of a block, as a file has it or as a string holding it does
#: (a JSON value has ``\n`` written out).
_PEM_BREAK = r"(?:[ \t]*\r?\n|\\r\\n|\\n)[ \t]*"
#: One line of the body of a key: base64, or one of the two headers an encrypted one carries.
_PEM_LINE = r"(?:[A-Za-z0-9+/=]{16,}|(?:Proc-Type|DEK-Info): [^\n\\]*)"
#: A key from its BEGIN line to its END line, whole, whatever is written in front of its lines
#: (indentation, a comment marker, a quote). The search for the END line stops at the next
#: BEGIN, so that a text with many BEGIN lines and no END is read once and not once per
#: line. Without an END line (what Grep -A, or a Read window, shows of one) the body is
#: masked as far as it still looks like a key: its base64 lines and the two headers an
#: encrypted one carries.
_PRIVATE_KEY = re.compile(
    _PEM_BEGIN
    + rf"(?:(?:(?!-----BEGIN ).)*?{_PEM_END}"
    # The blank line an encrypted key has after its headers is part of the block when
    # what follows it is.
    + rf"|(?:{_PEM_BREAK}(?:{_PEM_LINE}|(?={_PEM_BREAK}{_PEM_LINE})))*)",
    re.DOTALL,
)

_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b")),
    ("github-token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,}\b")),
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google-key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("private-key", _PRIVATE_KEY),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
)

#: NAME=value where NAME says secret and value looks random. The name may be in quotes
#: (JSON, a Python dict) and in any case; whether it *says* secret is decided by
#: :func:`_names_a_secret`, which needs the name as the pattern found it.
_ASSIGNED = re.compile(
    r"\b(?P<name>[A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY|PRIVATE_?KEY)[A-Z0-9_]*)"
    r"(?P<sep>[\"']?\s*[=:]\s*[\"']?)"
    r"(?P<value>[A-Za-z0-9+/=_\-]{16,})",
    re.IGNORECASE,
)

_NAME_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
_SECRET_WORDS = frozenset({"secret", "token", "password", "passwd", "apikey"})
_SECRET_WORD_PAIRS = frozenset({("api", "key"), ("private", "key")})


def _names_a_secret(name: str) -> bool:
    """Whether ``name``, which contains a secret keyword somewhere, is a secret's name.

    In capitals there are no word breaks to go by, so the keyword anywhere in it counts
    (``AUTHTOKEN``, ``DBPASSWORD``), as it always has. In any other case it has to be a
    word of the name, split at underscores and at humps: ``api_key``, ``clientSecret`` and
    ``Password`` are, ``tokenizer``, ``max_tokens`` and ``secretary`` are not, and a
    redactor that called them one would mangle every machine-learning script it read.
    """
    if name.isupper():
        return True
    words = [word.lower() for word in _NAME_WORD.findall(name)]
    return any(word in _SECRET_WORDS for word in words) or any(
        pair in _SECRET_WORD_PAIRS for pair in pairwise(words)
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
            if not _names_a_secret(match["name"]):
                return match.group(0)
            if shannon_entropy(match["value"]) < ENTROPY_FLOOR:
                return match.group(0)
            replacements += 1
            # Only the value goes: the name, the quotes and the separator are how a reader
            # (or the model) tells what was there.
            return f"{match['name']}{match['sep']}[redacted:assigned-secret]"

        text = _ASSIGNED.sub(_assigned, text)
        return text, replacements

    def is_secret_path(self, absolute: str, relative: str) -> bool:
        # The shared matcher rather than a pathspec of its own: it already tries
        # each candidate with and without leading slashes, and it keeps the
        # deprecated pathspec factory name in exactly one place.
        return glob_matches_any(SECRET_PATH_PATTERNS, (relative, absolute))
