"""Tests for nanoclaude.permissions.redact.

Credential-shaped samples are assembled from parts at import time rather than
written as contiguous literals: this repository is public, and GitHub push
protection rejects a push containing a string shaped like a live credential,
even a fake one. The AWS documented example key and a bare PEM header are
published, non-secret values and stay literal.
"""

import base64
import hashlib
import time
from collections.abc import Hashable

import pytest

from nanoclaude.permissions.redact import (
    ENTROPY_FLOOR,
    SECRET_PATH_PATTERNS,
    Redactor,
    shannon_entropy,
)

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


@pytest.mark.parametrize(
    "path", [".ENV", "svc/.Env.Local", "ID_RSA", ".SSH/known_hosts", "Key.PEM"]
)
def test_secret_paths_are_recognised_whatever_their_letter_case(path):
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


# ---- the assignment rule, whatever the case of the name


def fake_secret(seed: str, length: int = 40) -> str:
    """A value that looks random to the scanner and is not anybody's: a slice of a hash of
    ``seed``, the first of them random enough to be caught. Built when the test runs so
    that no credential-shaped literal is in the file."""
    for attempt in range(1000):
        value = hashlib.sha256(f"{seed}/{attempt}".encode()).hexdigest()[:length]
        if shannon_entropy(value) > ENTROPY_FLOOR + 0.1:
            return value
    raise AssertionError("no sample of that length is random enough")


@pytest.mark.parametrize(
    "template",
    [
        'aws_secret_access_key = "{v}"',
        "api_key = '{v}'",
        '{{"PASSWORD": "{v}"}}',
        "{{'api_key': '{v}'}}",
        "password: {v}",
        "Password = {v}",
        "client_secret={v}",
        'apiKey: "{v}"',
        "accessToken = '{v}'",
        'PrivateKey = "{v}"',
        'db_passwd="{v}"',
        "export GITHUB_TOKEN={v}",
        "AUTHTOKEN={v}",
    ],
)
def test_an_assigned_secret_is_redacted_whatever_the_case_of_its_name(template):
    value = fake_secret(template)
    text = template.format(v=value)
    cleaned, count = Redactor().scrub(text)
    assert value not in cleaned, cleaned
    assert count == 1, cleaned
    assert "[redacted:assigned-secret]" in cleaned


def test_the_shape_of_what_surrounds_a_redacted_value_is_kept():
    value = fake_secret("shape")
    cleaned, _ = Redactor().scrub(f'{{"PASSWORD": "{value}", "user": "ada"}}')
    assert cleaned == '{"PASSWORD": "[redacted:assigned-secret]", "user": "ada"}'
    cleaned, _ = Redactor().scrub(f"api_key = '{value}'  # rotate monthly")
    assert cleaned == "api_key = '[redacted:assigned-secret]'  # rotate monthly"


@pytest.mark.parametrize(
    "text",
    [
        'tokenizer_name = "sentence-transformers/all-MiniLM-L6-v2"',
        'tokenizer = "bert-base-uncased-vocabulary-v1"',
        "max_tokens = 4096000000000000000000",
        'secretary = "Alexandria-Montgomery-Wellington"',
        'password_field_label = "Enter your password here, please"',
        "token_count = 1234567890123456789",
        'api_key = "a" * 40',
        "api_key = os.environ['API_KEY_FOR_THE_SERVICE_NAME']",
    ],
)
def test_a_name_that_only_contains_a_secret_word_is_not_a_secret(text):
    """Letting the name be in any case would catch ``tokenizer``, ``secretary`` and the
    rest of the words that merely start with one: the keyword has to be a word of the
    name. A name in capitals keeps the older, broader reading, where there are no word
    breaks to go by."""
    assert Redactor().scrub(text) == (text, 0)


def test_a_value_that_is_too_short_or_too_regular_is_still_left_alone():
    for text in ("password = 'hunter2'", "api_key = '" + "ab" * 10 + "'"):
        assert Redactor().scrub(text) == (text, 0)


# ---- a private key block is masked whole


def _pem(kind: str = "RSA PRIVATE KEY", lines: int = 4) -> tuple[str, list[str]]:
    """A block with a body no scanner would take for a key (the bytes are 0..255), made when
    the test runs. Returns the block and its body lines."""
    body = base64.b64encode(bytes(range(256)) * 2).decode()
    rows = [body[i : i + 64] for i in range(0, 64 * lines, 64)]
    block = "\n".join(["-----BEGIN " + kind + "-----", *rows, "-----END " + kind + "-----"])
    return block, rows


@pytest.mark.parametrize(
    "kind",
    [
        "RSA PRIVATE KEY",
        "PRIVATE KEY",
        "OPENSSH PRIVATE KEY",
        "ENCRYPTED PRIVATE KEY",
        "EC PRIVATE KEY",
    ],
)
def test_a_private_key_block_is_masked_from_begin_to_end(kind):
    block, rows = _pem(kind)
    cleaned, count = Redactor().scrub(f"before\n{block}\nafter")
    assert cleaned == "before\n[redacted:private-key]\nafter"
    assert count == 1
    assert all(row not in cleaned for row in rows)


def test_a_pgp_private_key_block_is_masked_whole():
    block, _ = _pem("PGP PRIVATE KEY BLOCK")
    cleaned, count = Redactor().scrub(f"x\n{block}\ny")
    assert (cleaned, count) == ("x\n[redacted:private-key]\ny", 1)


def test_two_blocks_are_two_masks_and_the_text_between_them_stays():
    first, _ = _pem(lines=2)
    second, _ = _pem("EC PRIVATE KEY", lines=3)
    cleaned, count = Redactor().scrub(f"{first}\nbetween\n{second}\n")
    assert cleaned == "[redacted:private-key]\nbetween\n[redacted:private-key]\n"
    assert count == 2


def test_a_block_with_windows_line_endings_is_masked_whole():
    block, rows = _pem()
    cleaned, _ = Redactor().scrub(block.replace("\n", "\r\n"))
    assert all(row not in cleaned for row in rows)


def test_a_block_inside_a_json_string_is_masked_whole():
    block, _ = _pem()
    cleaned, _ = Redactor().scrub('{"key": "' + block.replace("\n", "\\n") + '", "id": 7}')
    assert cleaned == '{"key": "[redacted:private-key]", "id": 7}'


def test_a_block_whose_lines_are_commented_or_indented_is_masked_whole():
    block, rows = _pem()
    for prefix in ("# ", "    ", "// ", "> "):
        text = "\n".join(prefix + line for line in block.splitlines())
        cleaned, count = Redactor().scrub(text)
        assert all(row not in cleaned for row in rows), prefix
        assert (cleaned, count) == (prefix + "[redacted:private-key]", 1), prefix


def test_a_text_with_thousands_of_begin_lines_and_no_end_is_scrubbed_in_linear_time():
    """Looking for the END line from every BEGIN to the end of the text would square the
    work; the search stops at the next BEGIN."""
    text = ("-----BEGIN " + "PRIVATE KEY-----\n") * 20_000
    started = time.perf_counter()
    cleaned, count = Redactor().scrub(text)
    elapsed = time.perf_counter() - started
    assert count == 20_000 and "BEGIN" not in cleaned
    assert elapsed < 5, f"{elapsed:.1f}s to scrub 20,000 BEGIN lines"


def test_a_long_line_of_keywords_is_scrubbed_in_linear_time():
    """The name part of the assignment rule used to be unbounded on both sides of the
    keyword: in one long identifier with the keyword in it many times, each occurrence
    scanned to the end of the line and back, which squares the work (about fifteen seconds
    for the 60,000 characters here; two minutes for 200,000)."""
    text = "TOKEN" * 12_000
    started = time.perf_counter()
    cleaned, count = Redactor().scrub(text)
    elapsed = time.perf_counter() - started
    assert (cleaned, count) == (text, 0)
    assert elapsed < 2, f"{elapsed:.1f}s to scrub {len(text):,} characters"


def test_a_long_line_with_no_keyword_and_a_long_line_of_separators_are_scrubbed_in_linear_time():
    for text in ("a" * 200_000, "token" + " " * 200_000 + "x", "token=" + "a" * 200_000):
        started = time.perf_counter()
        Redactor().scrub(text)
        elapsed = time.perf_counter() - started
        assert elapsed < 2, f"{elapsed:.1f}s to scrub {len(text):,} characters"


def test_a_name_longer_than_the_bound_on_either_side_of_its_keyword_is_not_read_as_one():
    """Known and documented: the name is looked for within 64 characters of the keyword."""
    value = fake_secret("long-name")
    long_prefix = "A" * 70 + "_TOKEN"
    long_suffix = "TOKEN_" + "B" * 70
    for name in (long_prefix, long_suffix):
        assert Redactor().scrub(f"{name}={value}") == (f"{name}={value}", 0)
    within = "A" * 60 + "_TOKEN_" + "B" * 60
    assert Redactor().scrub(f"{within}={value}")[1] == 1


def test_a_block_cut_off_before_its_end_has_its_body_masked_anyway():
    """What Grep -A or a Read window shows is a block with no end line, and the body is the
    key."""
    block, rows = _pem(lines=4)
    cut = block.rsplit("\n", 1)[0]  # no END line
    cleaned, count = Redactor().scrub(f"{cut}\nnot part of the key\n")
    assert all(row not in cleaned for row in rows)
    assert cleaned == "[redacted:private-key]\nnot part of the key\n"
    assert count == 1


def test_a_cut_off_encrypted_block_is_masked_past_its_two_headers():
    block, rows = _pem(lines=3)
    begin, *body = block.split("\n")[:-1]
    headers = ["Proc-Type: 4,ENCRYPTED", "DEK-Info: AES-128-CBC," + "0123456789ABCDEF" * 2, ""]
    cleaned, count = Redactor().scrub("\n".join([begin, *headers, *body]) + "\nnot part of it\n")
    assert all(row not in cleaned for row in rows)
    assert (cleaned, count) == ("[redacted:private-key]\nnot part of it\n", 1)


def test_a_cut_off_block_inside_a_json_string_has_its_body_masked_too():
    block, rows = _pem(lines=4)
    cut = block.rsplit("\n", 1)[0].replace("\n", "\\n")  # the \\n a JSON string has
    cleaned, _ = Redactor().scrub('{"key": "' + cut)
    assert all(row not in cleaned for row in rows)
    assert cleaned == '{"key": "[redacted:private-key]'


def test_text_that_merely_follows_a_lone_header_line_is_left_alone():
    cleaned, _ = Redactor().scrub("-----BEGIN RSA PRIVATE KEY-----\nthe key is stored elsewhere\n")
    assert cleaned == "[redacted:private-key]\nthe key is stored elsewhere\n"


def test_a_public_key_block_and_a_certificate_are_left_alone():
    for kind in ("PUBLIC KEY", "CERTIFICATE"):
        block, _ = _pem(kind)
        assert Redactor().scrub(block) == (block, 0)


# ---- the list of paths that are credentials by convention


@pytest.mark.parametrize(
    "path",
    [
        ".envrc",
        "svc/.envrc",
        ".netrc",
        "home/.npmrc",
        "pkg/.npmrc",
        ".pypirc",
        ".git-credentials",
        ".docker/config.json",
        "home/.docker/config.json",
        ".kube/config",
        "home/.kube/config",
        "release/app.jks",
        "release/debug.keystore",
        "infra/terraform.tfstate",
        "infra/terraform.tfstate.backup",
    ],
)
def test_more_credentials_by_convention_are_secret_paths(path):
    assert Redactor().is_secret_path(f"/p/{path}", path)


@pytest.mark.parametrize(
    "path",
    [
        "certs/server.crt",
        "certs/ca.crt",
        "keys/id_rsa.pub",
        "keys/id_ed25519.pub",
        "keys/id_ecdsa.pub",
        "docs/envrc.md",
        "src/npmrc.py",
        "docker/config.yaml",
        ".docker/Dockerfile",
        ".kube/README.md",
        "kubeconfig-template.yaml",
        "infra/main.tf",
        "infra/tfstate.md",
        "infra/notes.tfstate.md",
        "keystore.py",
    ],
)
def test_public_material_and_things_that_only_look_similar_stay_readable(path):
    assert not Redactor().is_secret_path(f"/p/{path}", path)
