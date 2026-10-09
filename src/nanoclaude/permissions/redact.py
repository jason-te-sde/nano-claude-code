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
* A name is looked for within 64 characters on each side of its keyword (a name longer than
  that is not read as a secret's), so that scrubbing a line takes time in proportion to its
  length.
* A name is read as a secret's when it is in capitals and has the keyword anywhere in
  it, or in any other case and has the keyword as a word of its own (``api_key``,
  ``clientSecret``, ``Password``). ``dbpassword`` in lower case, with no break to find the
  word by, is not read as one.
* ``_KEY`` ends a secret's name only in capitals (``STRIPE_KEY``): ``stripe_key`` is left
  alone, because ``cache_key`` and ``primary_key`` are everywhere and are not secrets.
* A bearer token written as one case of letters and separators with no digit
  (``YOUR_API_TOKEN_HERE``) is read as a placeholder, and so is the password of a URL that is a
  reference (``${DB_PASSWORD}``, ``<password>``); a real token or password of that shape is
  left as it is.
* A name that says secret, followed by a bare word, is redacted when the word is random
  enough, because an identifier cannot be told from a password typed without quotes
  (``password = a_long_function_name`` looks like ``password = Xy7Kp2mQ9vL4nR8t``). That is
  known over-redaction. A call (``name(``), a subscript (``name[``), an attribute chain
  (``name.other``) and the words ``None``, ``null``, ``nil``, ``undefined``, ``True``,
  ``False``, ``true`` and ``false`` are code, and are left alone.
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
    # Files named for what they hold, wherever a tool or a person left them.
    "**/*.env",
    "**/.env-*",
    "**/.env_*",
    "**/*.tfvars",
    "**/.pgpass",
    "**/.htpasswd",
    "**/.vault-token",
    "**/.my.cnf",
    "**/.gnupg/**",
)

_PEM_BEGIN = r"-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----"
_PEM_END = r"-----END [A-Z ]*PRIVATE KEY(?: BLOCK)?-----"
#: The two ends of a block, for what has to find them without running the whole recogniser
#: (``tools/keyblocks.py`` reads a file as a stream, and runs it over each block it gathers).
PRIVATE_KEY_BEGIN = re.compile(_PEM_BEGIN)
PRIVATE_KEY_END = re.compile(_PEM_END)
#: What a private key block is replaced with: the whole of it by :meth:`Redactor.scrub`, and the
#: lines of it that a window of a file shows by ``tools/keyblocks.py``.
PRIVATE_KEY_MARKER = "[redacted:private-key]"
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

#: The lengths below are the ones the vendors document where there is one to find (an AWS
#: key id is twenty characters, a SendGrid key is ``SG.`` and two parts of 22 and 43, an npm
#: token ``npm_`` and 36, a Hugging Face token ``hf_`` and 34, a GitLab token ``glpat-`` and
#: 20); where there is none (a PyPI token is a long macaroon, a Stripe live key is at least 24
#: after its prefix, a project key of OpenAI's has been about 150) the figure is a conservative
#: floor, low enough to catch a real one and high enough that the prefix alone, in the name of a
#: package or a function, is left alone.
_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("npm-token", re.compile(r"\bnpm_[A-Za-z0-9]{36,}\b")),
    ("gitlab-token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])")),
    ("stripe-key", re.compile(r"\bsk_live_[A-Za-z0-9]{24,}\b")),
    ("sendgrid-key", re.compile(r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}(?![A-Za-z0-9_-])")),
    ("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{34,}\b")),
    ("pypi-token", re.compile(r"\bpypi-[A-Za-z0-9_-]{50,}(?![A-Za-z0-9_-])")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b")),
    ("github-token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,}\b")),
    ("openai-key", re.compile(r"\bsk-proj-[A-Za-z0-9_-]{32,}(?![A-Za-z0-9_-])")),
    ("openai-key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google-key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("private-key", _PRIVATE_KEY),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
)

#: How many characters of a name the assignment rule looks at on each side of its keyword.
#: Unbounded, a line that is one long identifier with the keyword in it many times is read
#: from every occurrence to the end of the line and back, which squares the work.
_NAME_REACH = 64

#: NAME=value where NAME says secret and value looks random. The name may be in quotes
#: (JSON, a Python dict) and in any case; whether it *says* secret is decided by
#: :func:`_names_a_secret`, which needs the name as the pattern found it.
_ASSIGNED = re.compile(
    rf"\b(?P<name>[A-Z0-9_-]{{0,{_NAME_REACH}}}"
    r"(?:SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY|PRIVATE_?KEY|[_-]KEY)"
    rf"[A-Z0-9_-]{{0,{_NAME_REACH}}})"
    r"(?P<sep>[\"']?\s*[=:]\s*[\"']?)"
    r"(?P<value>[A-Za-z0-9+/=_\-]{16,})",
    re.IGNORECASE,
)

#: ``Authorization: Bearer <token>``, in a header, a curl argument, a JSON object or a dict. The
#: header's name and the scheme are what a reader needs to see, so only the token goes.
_BEARER = re.compile(
    r"(?P<prefix>\bAuthorization[\"']?\]?\s*[=:]\s*[\"']?Bearer\s+)(?P<token>[A-Za-z0-9._~+/=-]{16,})",
    re.IGNORECASE,
)
#: What documentation writes where a token goes (``YOUR_API_TOKEN_HERE``, ``my-token-goes-here``):
#: one case of letters and the separators between words, and no digit. A token somebody was
#: issued has digits, or both cases, in sixteen characters or more.
_PLACEHOLDER = re.compile(r"[A-Z_-]+|[a-z_-]+")

#: ``scheme://user:password@host``: the password. A password in a URL has to have its ``/``,
#: ``?``, ``#`` and ``@`` written as ``%xx``, so none of them is in one, and the user may be
#: empty (``redis://:password@host``). Each part ends at a character the next may not hold, so
#: a line of colons or of schemes costs no more than its length.
_URL_PASSWORD = re.compile(
    r"(?P<prefix>\b[A-Za-z][A-Za-z0-9]{1,31}://[^\s:/@?#]*:)(?P<password>[^\s@/?#]+)(?=@)"
)

#: Words that are a value in code, not a secret: what a language writes for nothing, for yes
#: and for no. With the length a value needs to be taken for a secret at all these cannot be
#: one, and are listed so that the rule does not depend on that number staying where it is.
_CODE_WORDS = frozenset({"None", "null", "nil", "undefined", "True", "False", "true", "false"})
#: What follows a value that is an identifier and not a secret: a call (``name(``), a subscript
#: (``name[``) or an attribute (``name.other``). A full stop that ends a sentence is not one.
_CODE_AFTER = re.compile(r"\(|\[|\.[A-Za-z_]")


def _is_code_expression(value: str, after: str) -> bool:
    """Whether an unquoted ``value`` is code (a call, a subscript, an attribute chain, or one
    of the words for nothing and for yes and no) and not something somebody typed as a secret.

    ``after`` is what follows the value in the text, as far as the first two characters. The
    value pattern stops at the first character a secret would not have, which is exactly where
    ``get_password_from_env()`` has its parenthesis, so the pattern alone redacts the name of
    the function and leaves the brackets, and an Edit can no longer match the line the model
    read.
    """
    return value in _CODE_WORDS or _CODE_AFTER.match(after) is not None


#: The keywords a name in capitals is read for anywhere in it. ``_KEY`` is not among them: a
#: name has to end in it (``STRIPE_KEY``), so that ``STRIPE_KEY_ID`` and ``MONKEY`` are not read
#: as secrets', and the pattern above cannot say that, since it only finds the keyword.
_UPPER_KEYWORD = re.compile(r"SECRET|TOKEN|PASSWORD|PASSWD|API[_-]?KEY|PRIVATE[_-]?KEY")

_NAME_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
_SECRET_WORDS = frozenset({"secret", "token", "password", "passwd", "apikey"})
_SECRET_WORD_PAIRS = frozenset({("api", "key"), ("private", "key")})


def _names_a_secret(name: str) -> bool:
    """Whether ``name``, which contains a secret keyword somewhere, is a secret's name.

    In capitals there are no word breaks to go by, so the keyword anywhere in it counts
    (``AUTHTOKEN``, ``DBPASSWORD``), as it always has, and so does ending in ``_KEY``
    (``STRIPE_KEY``). In any other case it has to be a word of the name, split at
    underscores, hyphens and humps: ``api_key``, ``client-secret`` and ``Password`` are,
    ``tokenizer``, ``max_tokens`` and ``secretary`` are not, and a redactor that called them
    one would mangle every machine-learning script it read.
    """
    if name.isupper():
        return _UPPER_KEYWORD.search(name) is not None or name.endswith(("_KEY", "-KEY"))
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
            # A value in quotes is a string, whatever follows it; one without is code when
            # it is a call or an access, and is taken for a secret when it is a bare word.
            quoted = match["sep"].endswith(("'", '"'))
            if not quoted and _is_code_expression(
                match["value"], match.string[match.end() : match.end() + 2]
            ):
                return match.group(0)
            replacements += 1
            # Only the value goes: the name, the quotes and the separator are how a reader
            # (or the model) tells what was there.
            return f"{match['name']}{match['sep']}[redacted:assigned-secret]"

        def _bearer(match: re.Match[str]) -> str:
            nonlocal replacements
            token = match["token"]
            if shannon_entropy(token) < ENTROPY_FLOOR or _PLACEHOLDER.fullmatch(token):
                return match.group(0)
            replacements += 1
            return f"{match['prefix']}[redacted:bearer-token]"

        def _url_password(match: re.Match[str]) -> str:
            nonlocal replacements
            # A reference to where the password lives, not the password.
            if match["password"].startswith(("$", "{", "<")):
                return match.group(0)
            replacements += 1
            return f"{match['prefix']}[redacted:url-password]"

        text = _BEARER.sub(_bearer, text)
        text = _URL_PASSWORD.sub(_url_password, text)
        text = _ASSIGNED.sub(_assigned, text)
        return text, replacements

    def private_key_spans(self, text: str) -> tuple[tuple[int, int], ...]:
        """The ranges of ``text`` (start, end, as ``re`` gives them) that are private key blocks.

        These are exactly the ranges :meth:`scrub` masks as ``private-key``, found by the same
        regular expression, and none when redaction is off. A window of a file is cut from the
        whole of it and its lines masked by these ranges: the extent of a block cannot be told
        from a window that begins or ends inside it.
        """
        if not self.enabled:
            return ()
        return tuple(match.span() for match in _PRIVATE_KEY.finditer(text))

    def is_secret_path(self, absolute: str, relative: str) -> bool:
        # The shared matcher rather than a pathspec of its own: it already tries
        # each candidate with and without leading slashes, and it keeps the
        # deprecated pathspec factory name in exactly one place.
        return glob_matches_any(SECRET_PATH_PATTERNS, (relative, absolute), fold=True)
