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


# ---- an assignment whose value is code is shown as it is


@pytest.mark.parametrize(
    "line",
    [
        "password = get_password_from_env()",
        "token = self.token",
        "api_key = config.api_key",
        'secret = os.environ["APP_SECRET"]',
        "password = None",
        "token = null",
        "secret = True",
        # the same shapes with a value long enough to have been taken for a secret
        "api_key = vault_client_secret_provider.api_key",
        "client_secret = settings_loader_instance[0]",
        "token = ProductionSecretsManager.token",
        "access_token = fetch_access_token_from_vault('main')",
        "PASSWORD = get_database_password_from_env()  # rotated by ops",
        "password = app_configuration_registry.database.password",
        "password: self_service_portal_config['db']",
    ],
)
def test_an_assignment_whose_value_is_code_is_shown_unchanged(line):
    assert Redactor().scrub(line) == (line, 0)


def test_a_quoted_value_is_redacted_even_when_it_has_a_dot_or_a_bracket_in_it():
    """What follows a value only says it is code when the value is not in quotes: a
    password in quotes that has a full stop after its first sixteen characters is one."""
    value = fake_secret("quoted-dot")
    for text in (f'password = "{value}.tail"', f"api_key = '{value}[1]'", f'token = "{value}(x)"'):
        cleaned, count = Redactor().scrub(text)
        assert value not in cleaned and count == 1, cleaned


def test_an_unquoted_value_is_redacted_when_what_follows_is_not_a_call_or_an_access():
    value = fake_secret("unquoted-end")
    for tail in ("", ".", " ", ", next", ";", ") # end", ". The next sentence", ".\n"):
        text = f"password: {value}{tail}"
        cleaned, count = Redactor().scrub(text)
        assert value not in cleaned and count == 1, text


@pytest.mark.parametrize(
    "template",
    [
        "PASSWORD={v}",
        "password = {v}",
        "[database]\npassword = {v}\n",
        'password = "{v}"',
        "db_password: {v}",
    ],
)
def test_a_bare_word_or_a_quoted_value_is_still_redacted(template):
    """An unquoted value that is a bare word cannot be told from an identifier, so what
    looks random enough is taken for the secret it most likely is."""
    value = "hunter2abc" + fake_secret(template, 14)
    text = template.format(v=value)
    cleaned, count = Redactor().scrub(text)
    assert value not in cleaned and count == 1, cleaned


def test_a_bare_identifier_that_looks_random_is_redacted_and_that_is_known():
    """The known over-redaction: ``password = <identifier>`` where the identifier is long
    and mixed enough, with nothing after it to say it is code."""
    cleaned, count = Redactor().scrub("api_key = xK9mQ2vL7nR4tY1wZ3pB8s")
    assert count == 1 and "xK9mQ" not in cleaned


@pytest.mark.parametrize(
    ("value", "after", "is_code"),
    [
        ("get_password_from_env", "()", True),
        ("get_password_from_env", "(arg)", True),
        ("application_settings", ".api_key", True),
        ("application_settings", "._private", True),
        ("credential_store_entries", '["main"]', True),
        ("None", "", True),
        ("null", "", True),
        ("nil", "", True),
        ("undefined", "", True),
        ("True", "", True),
        ("False", "", True),
        ("true", "", True),
        ("false", "", True),
        ("get_password_from_env", "", False),
        ("get_password_from_env", ".", False),
        ("get_password_from_env", ". Next", False),
        ("get_password_from_env", ".5", False),
        ("get_password_from_env", " (note)", False),
        ("Nonexistent", "", False),
        ("TrueColour", "", False),
    ],
)
def test_what_counts_as_code_after_an_assignment(value, after, is_code):
    from nanoclaude.permissions.redact import _is_code_expression

    assert _is_code_expression(value, after) is is_code


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


# ---- more credentials by path


@pytest.mark.parametrize(
    "path",
    [
        "prod.env",
        "config/local.env",
        "staging.Env",
        ".env-production",
        "svc/.env_local",
        "infra/prod.tfvars",
        "terraform.tfvars",
        "home/.pgpass",
        ".htpasswd",
        "web/.htpasswd",
        ".vault-token",
        "home/.vault-token",
        ".my.cnf",
        "home/.my.cnf",
        ".gnupg/pubring.kbx",
        "home/.gnupg/private-keys-v1.d/ABCD.key",
        ".gnupg/trustdb.gpg",
    ],
)
def test_the_credentials_paths_of_the_second_list_are_secret_paths(path):
    assert Redactor().is_secret_path(f"/p/{path}", path)


@pytest.mark.parametrize(
    "path",
    [
        "src/envelope.py",
        "docs/dotenv.md",
        ".environment",
        ".envision/notes.txt",
        "infra/tfvars.md",
        "infra/main.tf",
        "my.cnf",
        "etc/my.cnf.d/readme.md",
        "docs/pgpass.md",
        ".htpasswd.md.txt/readme",
        "vault-token.py",
        "docs/gnupg.md",
        "gnupg/readme.md",
        ".gnupgrc",
    ],
)
def test_names_that_only_look_like_the_second_list_stay_readable(path):
    assert not Redactor().is_secret_path(f"/p/{path}", path)


def test_every_path_pattern_the_second_list_names_is_in_the_list():
    for pattern in (
        "**/*.env",
        "**/.env-*",
        "**/.env_*",
        "**/*.tfvars",
        "**/.pgpass",
        "**/.htpasswd",
        "**/.vault-token",
        "**/.my.cnf",
        "**/.gnupg/**",
    ):
        assert pattern in SECRET_PATH_PATTERNS, pattern


# ---- more credential shapes


def _hex(seed: str, length: int) -> str:
    """``length`` characters of a hash of ``seed``: letters and digits no token generator
    would mistake for anything, built when the test runs so that no credential-shaped literal
    is in the file."""
    digest = hashlib.sha256(seed.encode()).hexdigest()
    assert length <= len(digest)
    return digest[:length]


def _urlsafe(seed: str, length: int) -> str:
    """Letters, digits, ``-`` and ``_``, in the mix a token has them."""
    raw = base64.urlsafe_b64encode(hashlib.sha256(seed.encode()).digest() * 4).decode()
    return raw[:length].replace("=", "x")


AWS_TEMPORARY = "ASIA" + _hex("asia", 16).upper()
NPM_TOKEN = "npm_" + _hex("npm", 36)
GITLAB_TOKEN = "glpat-" + _hex("glpat", 20)
STRIPE_LIVE = "sk_live_" + _hex("stripe", 24)
SENDGRID_KEY = "SG." + _hex("sg-id", 22) + "." + _hex("sg-secret", 43)
HUGGING_FACE = "hf_" + _hex("hf", 34)
PYPI_TOKEN = "pypi-" + _urlsafe("pypi", 70).replace("=", "x")
OPENAI_PROJECT = (
    "sk-proj-" + _hex("proj-a", 24) + "_" + _hex("proj-b", 20) + "-" + _hex("proj-c", 20)
)


@pytest.mark.parametrize(
    ("text", "label"),
    [
        pytest.param(AWS_TEMPORARY, "aws-key", id="aws-temporary"),
        pytest.param(NPM_TOKEN, "npm-token", id="npm"),
        pytest.param(GITLAB_TOKEN, "gitlab-token", id="gitlab"),
        pytest.param(STRIPE_LIVE, "stripe-key", id="stripe-live"),
        pytest.param(SENDGRID_KEY, "sendgrid-key", id="sendgrid"),
        pytest.param(HUGGING_FACE, "huggingface-token", id="hugging-face"),
        pytest.param(PYPI_TOKEN, "pypi-token", id="pypi"),
        pytest.param(OPENAI_PROJECT, "openai-key", id="openai-project"),
    ],
)
def test_more_known_secret_shapes_are_replaced(text, label):
    cleaned, count = Redactor().scrub(f"prefix {text} suffix")
    assert cleaned == f"prefix [redacted:{label}] suffix"
    assert count == 1


@pytest.mark.parametrize(
    "text",
    [
        "ASIA" + _hex("short", 15).upper(),  # one character short
        "ASIA" + _hex("lowercase-body", 16),  # the body of a key id is capitals and digits
        "ASIA" + _hex("long", 17).upper(),  # one character too long
        "ASIAN_MARKETS_REPORT",  # a name that starts the same
        "asia" + _hex("lower", 16),  # key ids are in capitals
        "AKIA" + _hex("short", 15).upper(),
        "npm_" + _hex("npm-short", 35),
        "npm_config_registry",
        "npm_package_version",
        "glpat-" + _hex("glpat-short", 19),
        "glpat-token-name",
        "sk_live_" + _hex("stripe-short", 23),
        "sk_live_mode_enabled",
        "sk_test_" + _hex("stripe-test", 24),  # a test key is not a live one
        "SG." + _hex("sg-short", 22) + "." + _hex("sg-short2", 42),
        "SG" + _hex("sg-no-dots", 22) + _hex("sg-no-dots-2", 43),  # the dots are part of it
        "SG. Smith joined the Singapore office",
        "see SG.docs.example",
        "hf_" + _hex("hf-short", 33),
        "hf_hub_download",
        "hf_model_name_or_path_for_tokenizer",
        "pypi-" + _urlsafe("pypi-short", 49),
        "pypi-server",
        "pypi-mirror-url-for-the-internal-index-hosts-in-ci",
        "sk-proj-" + _hex("proj-short", 31),
        "sk-project-name-for-the-billing-team",
    ],
)
def test_text_that_only_looks_like_one_of_the_new_shapes_is_left_alone(text):
    assert Redactor().scrub(f"x {text} y") == (f"x {text} y", 0)


def test_an_openai_project_key_with_underscores_and_hyphens_is_redacted_whole():
    """The old pattern stopped at the first underscore, so the rest was left behind."""
    cleaned, _ = Redactor().scrub(OPENAI_PROJECT)
    assert cleaned == "[redacted:openai-key]"


# ---- an Authorization header, and the password in a URL


@pytest.mark.parametrize(
    "template",
    [
        "Authorization: Bearer {v}",
        "authorization: bearer {v}",
        "AUTHORIZATION: BEARER {v}",
        "curl -H 'Authorization: Bearer {v}' https://api.example.test/x",
        '{{"Authorization": "Bearer {v}"}}',
        "headers['Authorization'] = 'Bearer {v}'",
        "Authorization=Bearer {v}",
    ],
)
def test_a_bearer_token_after_authorization_is_redacted(template):
    value = fake_secret(template, 32)
    cleaned, count = Redactor().scrub(template.format(v=value))
    assert value not in cleaned and count == 1, cleaned
    assert "[redacted:bearer-token]" in cleaned
    assert "earer" in cleaned.lower() and "uthorization" in cleaned.lower()


def test_what_surrounds_a_redacted_bearer_token_is_kept():
    value = fake_secret("surround", 32)
    cleaned, _ = Redactor().scrub(f"curl -H 'Authorization: Bearer {value}' https://x.test")
    assert cleaned == "curl -H 'Authorization: Bearer [redacted:bearer-token]' https://x.test"


@pytest.mark.parametrize(
    "text",
    [
        "Authorization: Bearer ${API_TOKEN}",
        "Authorization: Bearer <your-token-here>",
        "Authorization: Bearer YOUR_API_TOKEN_HERE",
        "Authorization: Bearer my-token-goes-here-please",
        "Authorization: Bearer " + "a1" * 16,  # long enough, and nothing like random
        "Authorization: Bearer " + _hex("fifteen", 15),  # one short of what a token needs
        "Authorization: Bearer token",
        "Authorization: Basic dXNlcjpwYXNzd29yZA==",
        "the bearer of this letter is authorized to collect it",
        "Bearer " + "a" * 40,  # no header name: only the header is looked for
        "Authorization: " + "a" * 40,
    ],
)
def test_text_that_only_looks_like_an_authorization_header_is_left_alone(text):
    assert Redactor().scrub(text) == (text, 0)


@pytest.mark.parametrize(
    "template",
    [
        "postgres://admin:{v}@db.internal:5432/prod",
        "DATABASE_URL=mysql://root:{v}@localhost/app",
        "https://user:{v}@example.test/path?x=1",
        "redis://:{v}@cache.internal:6379/0",
        "git+https://ci:{v}@git.example.test/org/repo.git",
        "see amqps://svc:{v}@queue.example.test, and",
    ],
)
def test_the_password_in_a_url_is_redacted_and_the_rest_of_the_url_is_not(template):
    value = "hunter2" + _hex(template, 6)
    cleaned, count = Redactor().scrub(template.format(v=value))
    assert value not in cleaned and count == 1, cleaned
    assert "[redacted:url-password]@" in cleaned
    # the scheme, the user and the host are what a reader needs to see which service it was
    assert cleaned == template.format(v="[redacted:url-password]")


@pytest.mark.parametrize(
    "text",
    [
        "https://user@example.test/path",
        "ssh://git@github.com/org/repo.git",
        "http://localhost:8080/path",
        "https://example.test:443/a:b@c",
        "https://example.test/users/ada:lovelace@home",
        "postgres://admin:${DB_PASSWORD}@db.internal/prod",
        "postgres://admin:{password}@db.internal/prod",
        "postgres://admin:<password>@db.internal/prod",
        "mailto:ada@example.test",
        "scheme://",
        "xy://b:c d@e",
        "go to xy://host and mail ada:hunter2@example.test",  # a space ends what could be a URL
        "C://data:hunter2@host",  # one letter is a drive, not a scheme
    ],
)
def test_a_url_with_a_user_and_no_password_or_with_no_userinfo_is_left_alone(text):
    assert Redactor().scrub(text) == (text, 0)


# ---- names with hyphens, and names ending in _KEY


@pytest.mark.parametrize(
    "template",
    [
        "api-key: {v}",
        "client-secret = {v}",
        "X-API-KEY: {v}",
        "--api-key={v}",
        '"api-key": "{v}"',
        "access-token: {v}",
        "db-password: '{v}'",
        "private-key = {v}",
    ],
)
def test_an_assignment_to_a_name_with_hyphens_is_redacted(template):
    value = fake_secret(template)
    cleaned, count = Redactor().scrub(template.format(v=value))
    assert value not in cleaned and count == 1, cleaned
    assert "[redacted:assigned-secret]" in cleaned


@pytest.mark.parametrize(
    "template",
    [
        "tokenizer-config: {v}",
        "max-tokens: {v}",
        "secretary-general = {v}",
        "api-keys-count: {v}",
        "the-key: {v}",
    ],
)
def test_a_hyphenated_name_that_only_contains_a_secret_word_is_not_a_secret(template):
    text = template.format(v=fake_secret(template))
    assert Redactor().scrub(text) == (text, 0)


@pytest.mark.parametrize(
    "template",
    [
        "STRIPE_KEY={v}",
        "export SENDGRID_KEY={v}",
        "OPENAI_KEY: {v}",
        'GOOGLE_MAPS_KEY = "{v}"',
        "X-STRIPE-KEY: {v}",
    ],
)
def test_a_name_in_capitals_ending_in_key_is_a_secret(template):
    value = fake_secret(template)
    cleaned, count = Redactor().scrub(template.format(v=value))
    assert value not in cleaned and count == 1, cleaned


@pytest.mark.parametrize(
    "template",
    [
        "MONKEY={v}",
        "KEYBOARD={v}",
        "KEY={v}",
        "STRIPE_KEY_ID={v}",
        "KEY_ID={v}",
        "stripe_key = {v}",
        "cache_key = {v}",
        "primaryKey = {v}",
        "TURKEY_SANDWICH={v}",
    ],
)
def test_a_name_that_merely_has_key_in_it_is_not_a_secret(template):
    text = template.format(v=fake_secret(template))
    assert Redactor().scrub(text) == (text, 0)


def test_a_bearer_token_in_one_case_with_digits_in_it_is_not_taken_for_a_placeholder():
    value = _hex("lowercase-bearer", 32)
    assert value == value.lower() and any(c.isdigit() for c in value)
    cleaned, count = Redactor().scrub(f"Authorization: Bearer {value}")
    assert (cleaned, count) == ("Authorization: Bearer [redacted:bearer-token]", 1)


@pytest.mark.parametrize(
    "text",
    [
        "TOKEN-" * 10_000,
        "api-" * 20_000,
        "x://" + "a:" * 100_000,
        "x://" * 50_000,
        "x://" + "a" * 200_000,
        "Authorization: Bearer " * 10_000,
        "Authorization: Bearer " + "a." * 100_000,
        "_KEY" * 50_000,
        "A-" * 30_000 + "_KEY",
    ],
    ids=lambda text: f"{text[:12]!r}x{len(text):,}",
)
def test_the_new_rules_scrub_adversarial_lines_in_linear_time(text):
    """Each of these takes well under a second; the bound is a wide one, for a slow machine,
    and a pattern that squares the work on them takes minutes."""
    started = time.perf_counter()
    Redactor().scrub(text)
    elapsed = time.perf_counter() - started
    assert elapsed < 5, f"{elapsed:.1f}s to scrub {len(text):,} characters"


@pytest.mark.parametrize(
    ("make", "label"),
    [
        (lambda n: "pypi-" + _urlsafe("floor-pypi", n), "pypi-token"),
        (lambda n: "sk-proj-" + _urlsafe("floor-proj", n), "openai-key"),
        (lambda n: "glpat-" + _urlsafe("floor-gl", n), "gitlab-token"),
    ],
    ids=["pypi", "openai-project", "gitlab"],
)
def test_a_token_exactly_as_long_as_the_floor_is_redacted_and_one_less_is_not(make, label):
    floor = {"pypi-token": 50, "openai-key": 32, "gitlab-token": 20}[label]
    assert Redactor().scrub(make(floor)) == (f"[redacted:{label}]", 1)
    assert Redactor().scrub(make(floor - 1))[1] == 0


@pytest.mark.parametrize("last", ["-", "_"])
@pytest.mark.parametrize(
    ("prefix", "body", "label"),
    [
        ("pypi-", 59, "pypi-token"),
        ("sk-proj-", 39, "openai-key"),
        ("glpat-", 39, "gitlab-token"),
    ],
)
def test_a_token_that_ends_in_a_hyphen_or_an_underscore_is_redacted_to_its_last_character(
    prefix, body, label, last
):
    token = prefix + _hex(prefix, body) + last
    assert Redactor().scrub(f"x {token} y") == (f"x [redacted:{label}] y", 1)


def test_a_sendgrid_key_whose_last_character_is_a_hyphen_is_redacted_whole():
    key = "SG." + _hex("sg-a", 22) + "." + _hex("sg-b", 42) + "-"
    assert Redactor().scrub(f"x {key} y") == ("x [redacted:sendgrid-key] y", 1)


def test_a_bearer_token_of_sixteen_characters_is_redacted_and_one_of_fifteen_is_not():
    sixteen = "AbCdEfGhIjKlMnO1"
    assert len(set(sixteen)) == 16
    assert Redactor().scrub(f"Authorization: Bearer {sixteen}")[1] == 1
    assert Redactor().scrub(f"Authorization: Bearer {sixteen[:15]}")[1] == 0


@pytest.mark.parametrize(
    "template",
    [
        "AUTH-TOKEN-VALUE: {v}",
        "auth-token-value: {v}",
        "X-API-KEY-ID: {v}",
        "PRIVATE-KEY-PASSPHRASE = {v}",
        "access-secret-id: {v}",
    ],
)
def test_a_hyphenated_name_with_the_keyword_in_the_middle_is_a_secret_as_it_is_with_underscores(
    template,
):
    value = fake_secret(template)
    cleaned, count = Redactor().scrub(template.format(v=value))
    assert value not in cleaned and count == 1, cleaned
    underscored = template.replace("-", "_").format(v=value)
    assert Redactor().scrub(underscored)[1] == 1, underscored


# ---- the ranges the scrubber masks, for what has to cut a window out of a whole file


def test_private_key_spans_are_the_ranges_scrub_masks():
    block, _ = _pem(lines=3)
    text = f"before\n{block}\nafter"
    ((start, end),) = Redactor().private_key_spans(text)
    assert text[start:end] == block
    assert Redactor().scrub(text)[0] == text[:start] + "[redacted:private-key]" + text[end:]


def test_there_is_a_span_for_each_key_and_none_for_a_public_key():
    first, _ = _pem(lines=2)
    second, _ = _pem("EC PRIVATE KEY", lines=2)
    public, _ = _pem("PUBLIC KEY", lines=2)
    spans = Redactor().private_key_spans(f"{first}\n{public}\n{second}\n")
    assert len(spans) == 2


def test_a_key_cut_off_before_its_end_has_a_span_as_far_as_it_looks_like_one():
    block, _ = _pem(lines=3)
    cut = "\n".join(block.split("\n")[:-1])
    text = f"{cut}\nnot part of it"
    ((start, end),) = Redactor().private_key_spans(text)
    assert text[start:end] == cut


def test_there_are_no_spans_when_redaction_is_off_or_there_is_no_key():
    block, _ = _pem()
    assert Redactor(enabled=False).private_key_spans(block) == ()
    assert Redactor().private_key_spans("just text\nmore text") == ()
    assert Redactor().private_key_spans("") == ()


def test_the_begin_and_end_patterns_name_what_the_recogniser_starts_and_ends_on():
    from nanoclaude.permissions.redact import PRIVATE_KEY_BEGIN, PRIVATE_KEY_END, PRIVATE_KEY_MARKER

    for kind in ("RSA PRIVATE KEY", "PRIVATE KEY", "OPENSSH PRIVATE KEY", "PGP PRIVATE KEY BLOCK"):
        assert PRIVATE_KEY_BEGIN.search(f"x -----BEGIN {kind}----- y")
        assert PRIVATE_KEY_END.search(f"x -----END {kind}----- y")
    assert not PRIVATE_KEY_BEGIN.search("-----BEGIN PUBLIC KEY-----")
    assert not PRIVATE_KEY_END.search("-----END CERTIFICATE-----")
    assert Redactor().scrub(_pem()[0])[0] == PRIVATE_KEY_MARKER
