"""Tests for nanoclaude.permissions.redact.

Credential-shaped samples are assembled from parts at import time rather than
written as contiguous literals: this repository is public, and GitHub push
protection rejects a push containing a string shaped like a live credential,
even a fake one. The AWS documented example key and a bare PEM header are
published, non-secret values and stay literal.
"""

from collections.abc import Hashable

import pytest

from nanoclaude.permissions.redact import SECRET_PATH_PATTERNS, Redactor, shannon_entropy

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
GITHUB_CLASSIC = "ghp_" + "A" * 36
GITHUB_FINE_GRAINED = "github_pat_" + "B" * 25
ANTHROPIC_KEY = "sk-ant-" + "C" * 25
OPENAI_KEY = "sk-" + "D" * 40
SLACK_TOKEN = "xoxb-" + "1" * 12 + "-" + "E" * 20
GOOGLE_KEY = "AIza" + "F" * 35
PRIVATE_KEY_HEADER = "-----BEGIN RSA PRIVATE KEY-----"
JWT = "eyJ" + "G" * 12 + "." + "H" * 12 + "." + "I" * 12
ASSIGNED = "API_SECRET=" + "ABCDEFGHIJKLMNOP" + "0129"


# One case per row of _SHAPES (nine rows; the two github-token rows match
# different prefixes, so each has its own sample) plus the assignment rule.
@pytest.mark.parametrize(
    ("text", "label"),
    [
        pytest.param(AWS_KEY, "aws-key", id="aws-key"),
        pytest.param(GITHUB_CLASSIC, "github-token", id="github-classic"),
        pytest.param(GITHUB_FINE_GRAINED, "github-token", id="github-fine-grained"),
        pytest.param(ANTHROPIC_KEY, "anthropic-key", id="anthropic-key"),
        pytest.param(OPENAI_KEY, "openai-key", id="openai-key"),
        pytest.param(SLACK_TOKEN, "slack-token", id="slack-token"),
        pytest.param(GOOGLE_KEY, "google-key", id="google-key"),
        pytest.param(PRIVATE_KEY_HEADER, "private-key", id="private-key"),
        pytest.param(JWT, "jwt", id="jwt"),
        pytest.param(ASSIGNED, "assigned-secret", id="assigned-secret"),
    ],
)
def test_known_secret_shapes_are_replaced(text, label):
    cleaned, count = Redactor().scrub(f"prefix {text} suffix")
    assert count == 1
    assert text not in cleaned
    assert f"[redacted:{label}]" in cleaned


@pytest.mark.parametrize(
    "text",
    [
        "commit 9f2a1c4d8e7b6a5f4e3d2c1b0a9f8e7d6c5b4a39",  # a git sha
        "id: 3f2504e0-4f89-11d3-9a0c-0305e82c3301",  # a uuid
        "from very.long.module.path.that.goes.on.forever import x",  # an import
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8",  # base64 png
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "export DEBUG=true",
    ],
)
def test_ordinary_content_is_left_alone(text):
    """The must-not-fire set. A redactor that mangles normal output is unusable."""
    cleaned, count = Redactor().scrub(text)
    assert (cleaned, count) == (text, 0)


def test_a_low_entropy_assignment_is_left_alone():
    # The assignment rule fires only when the value also looks random.
    text = "API_SECRET=" + "a" * 20
    assert Redactor().scrub(text) == (text, 0)


def test_several_secrets_in_one_blob_are_all_replaced():
    cleaned, count = Redactor().scrub(f"{AWS_KEY} and {GITHUB_CLASSIC}")
    assert count == 2
    assert "AKIA" not in cleaned
    assert "ghp_" not in cleaned


def test_redaction_can_be_switched_off_wholesale():
    assert Redactor(enabled=False).scrub(AWS_KEY) == (AWS_KEY, 0)


def test_empty_text_is_returned_unchanged():
    assert Redactor().scrub("") == ("", 0)


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        "svc/.env.production",
        "id_rsa",
        ".ssh/id_ed25519",
        "credentials.json",
        "key.pem",
        "certs/server.key",
    ],
)
def test_secret_paths_are_recognised(path):
    assert Redactor().is_secret_path(f"/p/{path}", path)


@pytest.mark.parametrize("path", ["src/environment.py", "docs/env.md", "keyboard.py"])
def test_paths_that_merely_look_similar_are_not(path):
    assert not Redactor().is_secret_path(f"/p/{path}", path)


def test_secret_path_patterns_is_a_tuple_and_includes_dotenv():
    assert isinstance(SECRET_PATH_PATTERNS, tuple)
    assert "**/.env" in SECRET_PATH_PATTERNS


def test_redactor_is_hashable():
    # Its only field is a bool, so frozen=True's generated __hash__ is not a
    # trap here, unlike ToolUseBlock (conversation/transcript.py), which holds
    # a Mapping. Pinned so a later unhashable field is caught.
    assert isinstance(Redactor(), Hashable)
    assert hash(Redactor()) == hash(Redactor())


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", 0.0), ("aaaa", 0.0), ("ab", 1.0), ("abcd", 2.0)],
)
def test_shannon_entropy_is_bits_per_character(value, expected):
    # scrub() never passes an empty value (the assignment rule needs 16+
    # characters), so the empty case is pinned here directly.
    assert shannon_entropy(value) == pytest.approx(expected)
