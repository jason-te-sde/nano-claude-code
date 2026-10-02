import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nanoclaude.config.load import (
    CONFIG_FILENAME,
    ConfigError,
    check_model,
    check_roles,
    expand_root,
    load_config,
)
from nanoclaude.config.schema import (
    DIFF_STYLES,
    KNOWN_ADAPTERS,
    ROLES,
    THEMES,
    Config,
    LimitsConfig,
    ModelConfig,
    PermissionsConfig,
    RolesConfig,
    UiConfig,
)
from nanoclaude.permissions.rules import Rule
from nanoclaude.permissions.sandbox import Sandbox

EM_DASH = "—"

# The smallest config that loads: one model, and nothing else.
BASE = """
[models.m]
adapter = "anthropic"
model = "claude-sonnet-5"
"""


@pytest.fixture(autouse=True)
def _a_home_directory_that_is_not_the_real_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing here may read the person's own home. Anything that expands a tilde
    # without being told which home lands in an empty directory instead.
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-real-home"))


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def home_file(tmp_path: Path) -> Path:
    return tmp_path / "home" / ".nanoclaude" / "config.toml"


def project_file(tmp_path: Path) -> Path:
    return tmp_path / "proj" / ".nanoclaude" / "config.toml"


def load(
    tmp_path: Path,
    home_text: str | None = None,
    project_text: str | None = None,
    env: dict[str, str] | None = None,
) -> Config:
    """Write whichever files are given, then load them the way the CLI does."""
    if home_text is not None:
        write(home_file(tmp_path), home_text)
    if project_text is not None:
        write(project_file(tmp_path), project_text)
    return load_config(
        home=str(tmp_path / "home"),
        project=str(tmp_path / "proj") if project_text is not None else None,
        env=env or {},
    )


def refusal(tmp_path: Path, home_text: str | None = None, project_text: str | None = None) -> str:
    """The message of the ConfigError the given files produce.

    Every refusal is held to spec 17.9's form for what a person is shown: what
    went wrong, an em-dash, what to do about it; lower case; no exclamation mark.
    """
    with pytest.raises(ConfigError) as caught:
        load(tmp_path, home_text, project_text)
    message = str(caught.value)
    assert f" {EM_DASH} " in message, (
        f"not in the '<what> {EM_DASH} <what to do>' form: {message!r}"
    )
    assert message[:1].islower(), f"does not start in lower case: {message!r}"
    assert "!" not in message, f"shouts: {message!r}"
    return message


# --------------------------------------------------------------------------
# The brief's own tests
# --------------------------------------------------------------------------


def test_a_minimal_config_loads(tmp_path):
    home = tmp_path / "home"
    write(
        home / ".nanoclaude" / "config.toml",
        """
[models.sonnet]
adapter = "anthropic"
model = "claude-sonnet-5"

[roles]
main = "sonnet"
""",
    )
    config = load_config(home=str(home), project=None, env={})
    assert config.roles.main == "sonnet"
    assert config.models["sonnet"].adapter == "anthropic"


def test_project_config_overlays_user_config(tmp_path):
    home, project = tmp_path / "home", tmp_path / "proj"
    write(
        home / ".nanoclaude" / "config.toml",
        """
[models.sonnet]
adapter = "anthropic"
model = "claude-sonnet-5"
[roles]
main = "sonnet"
[limits]
max_turns = 40
""",
    )
    write(
        project / ".nanoclaude" / "config.toml",
        """
[limits]
max_turns = 5
[permissions]
deny = ["Bash(curl:*)"]
""",
    )
    config = load_config(home=str(home), project=str(project), env={})
    assert config.limits.max_turns == 5
    assert config.roles.main == "sonnet"  # inherited
    assert "Bash(curl:*)" in config.permissions.deny


def test_every_role_defaults_to_main_when_unset(tmp_path):
    home = tmp_path / "home"
    write(
        home / ".nanoclaude" / "config.toml",
        """
[models.m]
adapter = "anthropic"
model = "claude-sonnet-5"
[roles]
main = "m"
""",
    )
    config = load_config(home=str(home), project=None, env={})
    assert config.roles.explore == "m" and config.roles.title == "m"


def test_a_role_pointing_at_an_undefined_model_is_rejected_at_load(tmp_path):
    home = tmp_path / "home"
    write(
        home / ".nanoclaude" / "config.toml",
        """
[models.m]
adapter = "anthropic"
model = "claude-sonnet-5"
[roles]
main = "m"
explore = "does-not-exist"
""",
    )
    with pytest.raises(ConfigError, match="does-not-exist"):
        load_config(home=str(home), project=None, env={})


def test_an_unknown_adapter_is_rejected_with_the_list_of_known_ones(tmp_path):
    home = tmp_path / "home"
    write(
        home / ".nanoclaude" / "config.toml",
        """
[models.m]
adapter = "telepathy"
model = "x"
[roles]
main = "m"
""",
    )
    with pytest.raises(ConfigError, match="anthropic"):
        load_config(home=str(home), project=None, env={})


def test_api_keys_come_from_the_environment_and_never_from_the_file(tmp_path):
    home = tmp_path / "home"
    write(
        home / ".nanoclaude" / "config.toml",
        """
[models.m]
adapter = "anthropic"
model = "claude-sonnet-5"
api_key_env = "MY_KEY"
[roles]
main = "m"
""",
    )
    config = load_config(home=str(home), project=None, env={"MY_KEY": "secret"})
    assert config.api_key_for("m") == "secret"


def test_no_config_at_all_gives_a_message_pointing_at_ncc_init(tmp_path):
    with pytest.raises(ConfigError, match="ncc init"):
        load_config(home=str(tmp_path), project=None, env={})


# --------------------------------------------------------------------------
# Layering: defaults, then the home file, then the project file, then the
# environment. Each rule is pinned in both directions.
# --------------------------------------------------------------------------


def test_the_config_file_is_named_config_toml():
    assert CONFIG_FILENAME == "config.toml"


def test_with_no_file_setting_them_every_limit_and_default_is_the_specs(tmp_path):
    config = load(tmp_path, BASE)
    assert config.limits == LimitsConfig(
        max_turns=40,
        bash_timeout_s=120.0,
        output_cap_bytes=100_000,
        compact_soft=0.70,
        compact_hard=0.85,
        keep_recent_turns=3,
    )
    assert config.permissions == PermissionsConfig(
        allow=("Read", "Grep", "Glob"),
        ask=("Bash", "Write", "Edit", "Git", "TodoWrite"),
        deny=(),
    )
    assert config.ui == UiConfig(theme="auto", diff_style="unified")


def test_the_home_file_beats_the_defaults_and_leaves_the_rest_alone(tmp_path):
    config = load(
        tmp_path,
        BASE + '[limits]\nmax_turns = 10\n[ui]\ntheme = "dark"\n[permissions]\nallow = ["Read"]\n',
    )
    assert config.limits.max_turns == 10
    assert config.limits.keep_recent_turns == 3  # not mentioned, so still the default
    assert config.ui.theme == "dark"
    assert config.ui.diff_style == "unified"
    assert config.permissions.allow == ("Read",)  # a list that is set replaces the default
    assert config.permissions.ask == ("Bash", "Write", "Edit", "Git", "TodoWrite")


def test_the_project_file_beats_the_home_file(tmp_path):
    config = load(tmp_path, BASE + "[limits]\nmax_turns = 10\n", "[limits]\nmax_turns = 5\n")
    assert config.limits.max_turns == 5


def test_a_key_the_project_does_not_set_keeps_the_home_value(tmp_path):
    # Merged key by key, not section by section: the project tightens one limit
    # without restating the others.
    home = BASE + "[limits]\nmax_turns = 10\nbash_timeout_s = 30\n"
    config = load(tmp_path, home, "[limits]\nmax_turns = 5\n")
    assert config.limits.max_turns == 5
    assert config.limits.bash_timeout_s == 30.0


def test_a_section_the_project_does_not_mention_keeps_the_home_values(tmp_path):
    home = BASE + '[ui]\ntheme = "light"\n'
    config = load(tmp_path, home, "[limits]\nmax_turns = 5\n")
    assert config.ui.theme == "light"


def test_a_project_file_alone_is_enough(tmp_path):
    write(project_file(tmp_path), BASE + "[limits]\nmax_turns = 7\n")
    config = load_config(home=str(tmp_path / "home"), project=str(tmp_path / "proj"), env={})
    assert config.limits.max_turns == 7
    assert config.roles.main == "m"


def test_a_project_changes_one_key_of_a_model_the_home_file_defines(tmp_path):
    home = BASE + 'api_key_env = "MY_KEY"\n'
    config = load(tmp_path, home, '[models.m]\nmodel = "claude-opus-5"\n')
    assert config.models["m"].model == "claude-opus-5"  # the project's
    assert config.models["m"].adapter == "anthropic"  # the home file's
    assert config.models["m"].api_key_env == "MY_KEY"  # the home file's


# A repository's config is not the person's own, so it may add rules but never
# take one away: a list the project sets stacks onto the home list (spec 6.2, "the
# project config overlays the user's").
RULES_DEFAULTS = {
    "allow": ("Read", "Grep", "Glob"),
    "ask": ("Bash", "Write", "Edit", "Git", "TodoWrite"),
    "deny": (),
}


@pytest.mark.parametrize("key", ["allow", "ask", "deny"])
def test_a_list_the_project_sets_is_added_to_the_home_list_not_put_in_its_place(tmp_path, key):
    home = BASE + f'[permissions]\n{key} = ["Read(**/.env*)"]\n'
    config = load(tmp_path, home, f'[permissions]\n{key} = ["Bash(curl:*)"]\n')
    assert getattr(config.permissions, key) == ("Read(**/.env*)", "Bash(curl:*)")


def test_a_home_deny_rule_still_applies_when_the_project_has_its_own_deny_list(tmp_path):
    home = BASE + '[permissions]\ndeny = ["Read(~/.ssh/**)"]\n'
    config = load(tmp_path, home, '[permissions]\ndeny = ["Bash(curl:*)"]\n')
    assert "Read(~/.ssh/**)" in config.permissions.deny
    assert "Bash(curl:*)" in config.permissions.deny


@pytest.mark.parametrize("key", ["allow", "ask", "deny"])
def test_an_empty_list_in_the_project_removes_nothing_from_the_home_list(tmp_path, key):
    home = BASE + f'[permissions]\n{key} = ["Read(**/.env*)"]\n'
    config = load(tmp_path, home, f"[permissions]\n{key} = []\n")
    assert getattr(config.permissions, key) == ("Read(**/.env*)",)


def test_a_project_allow_or_ask_rule_is_added_to_the_home_rules(tmp_path):
    home = BASE + '[permissions]\nallow = ["Read"]\nask = ["Bash"]\n'
    project = '[permissions]\nallow = ["Bash(npm test:*)"]\nask = ["Write(src/**)"]\n'
    config = load(tmp_path, home, project)
    assert config.permissions.allow == ("Read", "Bash(npm test:*)")
    assert config.permissions.ask == ("Bash", "Write(src/**)")


def test_stacking_keeps_the_home_order_and_drops_duplicates(tmp_path):
    home = BASE + '[permissions]\ndeny = ["Read", "Bash(curl:*)", "Read"]\n'
    config = load(tmp_path, home, '[permissions]\ndeny = ["Bash(curl:*)", "Write", "Read"]\n')
    assert config.permissions.deny == ("Read", "Bash(curl:*)", "Write")


@pytest.mark.parametrize("key", ["allow", "ask"])
def test_when_the_home_file_sets_no_list_the_project_adds_to_the_default_one(tmp_path, key):
    config = load(tmp_path, BASE, f'[permissions]\n{key} = ["WebFetch"]\n')
    assert getattr(config.permissions, key) == (*RULES_DEFAULTS[key], "WebFetch")


@pytest.mark.parametrize("key", ["allow", "ask", "deny"])
def test_an_empty_project_list_leaves_the_default_alone_too(tmp_path, key):
    config = load(tmp_path, BASE, f"[permissions]\n{key} = []\n")
    assert getattr(config.permissions, key) == RULES_DEFAULTS[key]


def test_a_home_list_still_replaces_the_default_instead_of_stacking_onto_it(tmp_path):
    # Stacking is what a project does. The person's own file says what they want
    # and the defaults are only what applies when it says nothing.
    config = load(tmp_path, BASE + '[permissions]\nallow = ["Read"]\n', "[limits]\nmax_turns = 5\n")
    assert config.permissions.allow == ("Read",)


def test_a_list_the_project_does_not_set_keeps_the_home_list(tmp_path):
    home = BASE + '[permissions]\ndeny = ["Read(**/.env*)"]\n'
    config = load(tmp_path, home, '[permissions]\nallow = ["WebFetch"]\n')
    assert config.permissions.deny == ("Read(**/.env*)",)
    assert config.permissions.allow == ("Read", "Grep", "Glob", "WebFetch")


TWO_MODELS = """
[models.a]
adapter = "anthropic"
model = "claude-sonnet-5"
[models.b]
adapter = "anthropic"
model = "claude-opus-5"
"""


def test_a_role_set_in_the_project_beats_the_one_set_at_home(tmp_path):
    config = load(
        tmp_path,
        TWO_MODELS + '[roles]\nmain = "a"\nexplore = "a"\nplan = "a"\n',
        '[roles]\nexplore = "b"\n',
    )
    assert config.roles.explore == "b"
    assert config.roles.plan == "a"  # the project said nothing about it


def test_a_role_nobody_sets_follows_the_final_main_not_the_one_at_home(tmp_path):
    config = load(tmp_path, TWO_MODELS + '[roles]\nmain = "a"\n', '[roles]\nmain = "b"\n')
    assert config.roles.main == "b"
    assert config.roles.explore == "b"
    assert config.roles.title == "b"


def test_a_role_set_explicitly_does_not_follow_main_when_main_changes(tmp_path):
    config = load(
        tmp_path, TWO_MODELS + '[roles]\nmain = "a"\nexplore = "a"\n', '[roles]\nmain = "b"\n'
    )
    assert config.roles.main == "b"
    assert config.roles.explore == "a"


def test_main_is_the_first_model_defined_when_the_roles_do_not_say(tmp_path):
    config = load(tmp_path, TWO_MODELS)
    assert config.roles.main == "a"
    assert config.roles.explore == "a"


def test_an_explicit_main_beats_the_first_model_defined(tmp_path):
    config = load(tmp_path, TWO_MODELS + '[roles]\nmain = "b"\n')
    assert config.roles.main == "b"
    assert config.roles.verify == "b"


def test_each_role_can_be_routed_to_its_own_model(tmp_path):
    # The headline feature: six roles, six different models, each one read into
    # the right slot.
    models = "".join(
        f'[models.{role}-model]\nadapter = "anthropic"\nmodel = "claude-{role}"\n'
        for role in ("main", "explore", "plan", "verify", "compact", "title")
    )
    roles = "[roles]\n" + "".join(
        f'{role} = "{role}-model"\n'
        for role in ("main", "explore", "plan", "verify", "compact", "title")
    )
    config = load(tmp_path, models + roles)
    assert config.roles.main == "main-model"
    assert config.roles.explore == "explore-model"
    assert config.roles.plan == "plan-model"
    assert config.roles.verify == "verify-model"
    assert config.roles.compact == "compact-model"
    assert config.roles.title == "title-model"


def test_the_environment_supplies_the_key_the_file_only_names(tmp_path):
    home = BASE + 'api_key_env = "MY_KEY"\n'
    named = load(tmp_path, home, env={"MY_KEY": "from-the-environment", "ANTHROPIC_API_KEY": "x"})
    assert named.api_key_for("m") == "from-the-environment"
    # With no variable named, the adapter's own variable is the one used.
    default = load(tmp_path, BASE, env={"MY_KEY": "not-this", "ANTHROPIC_API_KEY": "the-default"})
    assert default.api_key_for("m") == "the-default"


def test_the_config_keeps_only_the_variables_that_hold_a_models_key(tmp_path):
    text = """
[models.a]
adapter = "anthropic"
model = "claude-sonnet-5"
api_key_env = "MY_KEY"
[models.b]
adapter = "openai_compat"
model = "gpt-5"
base_url = "https://api.openai.com/v1"
[models.c]
adapter = "ollama"
model = "qwen3-coder"
"""
    environment = {
        "MY_KEY": "k1",
        "OPENAI_API_KEY": "k2",
        "ANTHROPIC_API_KEY": "not named by any model, so not kept",
        "AWS_SECRET_ACCESS_KEY": "unrelated",
        "PATH": "/usr/bin",
    }
    config = load(tmp_path, text, env=environment)
    assert config.env == {"MY_KEY": "k1", "OPENAI_API_KEY": "k2"}


def test_a_model_only_the_project_defines_has_its_key_kept_too(tmp_path):
    project = '[models.p]\nadapter = "anthropic"\nmodel = "claude-opus-5"\napi_key_env = "P_KEY"\n'
    config = load(tmp_path, BASE, project, env={"P_KEY": "pk", "OTHER": "no"})
    assert config.api_key_for("p") == "pk"
    assert config.env == {"P_KEY": "pk"}


def test_the_config_does_not_share_the_dictionary_it_was_given(tmp_path):
    environment = {"ANTHROPIC_API_KEY": "k"}
    config = load(tmp_path, BASE, env=environment)
    config.env.clear()
    assert environment == {"ANTHROPIC_API_KEY": "k"}


def test_every_setting_the_config_types_have_can_be_set_from_the_file(tmp_path):
    # A different value for every key, so that one read into another's field, or
    # a key the loader forgot, shows up as a wrong number rather than a pass.
    text = """
[models.m]
adapter        = "openai_compat"
model          = "deepseek/deepseek-v3"
base_url       = "https://openrouter.ai/api/v1"
api_key_env    = "OPENROUTER_API_KEY"
context_window = 64000
native_tools   = false

[roles]
main    = "m"
explore = "m"
plan    = "m"
verify  = "m"
compact = "m"
title   = "m"

[permissions]
allow = ["Read"]
ask   = ["Bash"]
deny  = ["Bash(curl:*)"]

[limits]
max_turns         = 7
bash_timeout_s    = 9.5
output_cap_bytes  = 1234
compact_soft      = 0.5
compact_hard      = 0.6
keep_recent_turns = 2

[ui]
theme      = "dark"
diff_style = "unified"
"""
    config = load(tmp_path, text)
    assert config.models["m"] == ModelConfig(
        adapter="openai_compat",
        model="deepseek/deepseek-v3",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        context_window=64_000,
        native_tools=False,
    )
    assert config.limits == LimitsConfig(
        max_turns=7,
        bash_timeout_s=9.5,
        output_cap_bytes=1234,
        compact_soft=0.5,
        compact_hard=0.6,
        keep_recent_turns=2,
    )
    assert config.permissions == PermissionsConfig(
        allow=("Read",), ask=("Bash",), deny=("Bash(curl:*)",)
    )
    assert config.ui == UiConfig(theme="dark", diff_style="unified")


# --------------------------------------------------------------------------
# Refusals. Each one is a ConfigError in spec 17.9's form (see ``refusal``) and
# is pinned by a fragment that no other refusal produces, so deleting the branch
# that raises it leaves a test red rather than another branch answering for it.
# --------------------------------------------------------------------------


def test_no_config_names_the_file_ncc_init_would_write(tmp_path):
    message = refusal(tmp_path)
    assert "no configuration found" in message
    assert f"run: ncc init (it writes {home_file(tmp_path)})" in message


def test_a_config_with_no_models_says_so_and_names_the_files_it_read(tmp_path):
    message = refusal(tmp_path, "[limits]\nmax_turns = 5\n")
    assert "no [models.<alias>] sections found" in message
    assert str(home_file(tmp_path)) in message


def test_an_empty_config_file_is_not_the_same_as_no_config_file(tmp_path):
    message = refusal(tmp_path, "# nothing here yet\n")
    assert "no [models.<alias>] sections found" in message
    assert "no configuration found" not in message


def test_a_config_path_that_cannot_be_read_is_reported_not_raised(tmp_path):
    home_file(tmp_path).mkdir(parents=True)  # a directory where the file should be
    message = refusal(tmp_path)
    assert f"could not read {home_file(tmp_path)}" in message
    assert "Is a directory" in message


def test_a_syntax_error_is_reported_with_the_file_and_where_in_it(tmp_path):
    message = refusal(tmp_path, '[models.m\nadapter = "anthropic"\n')
    assert f"the config file {home_file(tmp_path)} is not valid TOML" in message
    assert "line 1" in message


def test_a_file_that_is_not_utf8_is_reported_as_that(tmp_path):
    home_file(tmp_path).parent.mkdir(parents=True)
    home_file(tmp_path).write_bytes(b"[models.m]\nmodel = '\xff\xfe'\n")
    message = refusal(tmp_path)
    assert f"the config file {home_file(tmp_path)} is not valid UTF-8 text" in message


@pytest.mark.parametrize("section", ["hooks", "mcp", "verify", "tools", "limit"])
def test_a_section_this_version_does_not_read_is_refused_rather_than_ignored(tmp_path, section):
    # A hooks section that is silently ignored is a guard that never runs, and a
    # typo in a section name is a setting that never applies. Either way the
    # person believes something is in force that is not.
    message = refusal(tmp_path, BASE + f"[{section}]\nx = 1\n")
    assert f"unknown section '{section}' in {home_file(tmp_path)}" in message
    assert "the sections are models, roles, permissions, limits, ui" in message


def test_a_stray_key_outside_any_section_is_refused_the_same_way(tmp_path):
    message = refusal(tmp_path, "theme = 'dark'\n" + BASE)
    assert f"unknown section 'theme' in {home_file(tmp_path)}" in message


WRONG_TYPES = [
    pytest.param(
        "limits = 5\n" + BASE, "limits in", "must be a table, not a whole number", id="limits"
    ),
    pytest.param("models = 'x'\n", "models in", "must be a table, not a string", id="models"),
    pytest.param("roles = 'm'\n" + BASE, "roles in", "must be a table, not a string", id="roles"),
    pytest.param("ui = 1\n" + BASE, "ui in", "must be a table, not a whole number", id="ui"),
    pytest.param(
        "permissions = ['Read']\n" + BASE,
        "permissions in",
        "must be a table, not a list",
        id="permissions",
    ),
    pytest.param("[models]\nm = 'x'\n", "models.m in", "must be a table, not a string", id="alias"),
    pytest.param(
        BASE + '[limits]\nmax_turns = "forty"\n',
        "limits.max_turns in",
        "must be a whole number, not a string",
        id="max_turns-string",
    ),
    pytest.param(
        BASE + "[limits]\nmax_turns = 40.5\n",
        "limits.max_turns in",
        "must be a whole number, not a number",
        id="max_turns-float",
    ),
    pytest.param(
        BASE + "[limits]\nmax_turns = true\n",
        "limits.max_turns in",
        "must be a whole number, not true or false",
        id="max_turns-bool",
    ),
    pytest.param(
        BASE + "[limits]\nmax_turns = 2026-10-02\n",
        "limits.max_turns in",
        "must be a whole number, not a date or time",
        id="max_turns-date",
    ),
    pytest.param(
        BASE + "[limits]\nmax_turns = { a = 1 }\n",
        "limits.max_turns in",
        "must be a whole number, not a table",
        id="max_turns-table",
    ),
    pytest.param(
        BASE + "[limits]\noutput_cap_bytes = 1.5\n",
        "limits.output_cap_bytes in",
        "must be a whole number, not a number",
        id="output_cap_bytes",
    ),
    pytest.param(
        BASE + '[limits]\nkeep_recent_turns = "3"\n',
        "limits.keep_recent_turns in",
        "must be a whole number, not a string",
        id="keep_recent_turns",
    ),
    pytest.param(
        BASE + '[limits]\nbash_timeout_s = "fast"\n',
        "limits.bash_timeout_s in",
        "must be a number, not a string",
        id="bash_timeout_s-string",
    ),
    pytest.param(
        BASE + "[limits]\nbash_timeout_s = true\n",
        "limits.bash_timeout_s in",
        "must be a number, not true or false",
        id="bash_timeout_s-bool",
    ),
    pytest.param(
        BASE + '[limits]\ncompact_soft = "high"\n',
        "limits.compact_soft in",
        "must be a number, not a string",
        id="compact_soft",
    ),
    pytest.param(
        BASE + "[limits]\ncompact_hard = [1]\n",
        "limits.compact_hard in",
        "must be a number, not a list",
        id="compact_hard",
    ),
    pytest.param(
        BASE + 'native_tools = "yes"\n',
        "models.m.native_tools in",
        "must be true or false, not a string",
        id="native_tools",
    ),
    pytest.param(
        BASE + 'context_window = "big"\n',
        "models.m.context_window in",
        "must be a whole number, not a string",
        id="context_window",
    ),
    pytest.param(
        '[models.m]\nadapter = 3\nmodel = "x"\n',
        "models.m.adapter in",
        "must be a string, not a whole number",
        id="adapter",
    ),
    pytest.param(
        '[models.m]\nadapter = "anthropic"\nmodel = ["x"]\n',
        "models.m.model in",
        "must be a string, not a list",
        id="model",
    ),
    pytest.param(
        BASE + "base_url = 5\n",
        "models.m.base_url in",
        "must be a string, not a whole number",
        id="base_url",
    ),
    pytest.param(
        BASE + "api_key_env = 5\n",
        "models.m.api_key_env in",
        "must be a string, not a whole number",
        id="api_key_env",
    ),
    pytest.param(
        BASE + '[permissions]\nallow = "Read"\n',
        "permissions.allow in",
        "must be a list of strings, not a string",
        id="allow-string",
    ),
    pytest.param(
        BASE + '[permissions]\nask = { a = "b" }\n',
        "permissions.ask in",
        "must be a list of strings, not a table",
        id="ask-table",
    ),
    pytest.param(
        BASE + '[permissions]\ndeny = ["Read", 7]\n',
        "permissions.deny[1] in",
        "must be a string, not a whole number",
        id="deny-item",
    ),
    pytest.param(
        BASE + "[roles]\nmain = 5\n",
        "roles.main in",
        "must be a string, not a whole number",
        id="role",
    ),
    pytest.param(
        BASE + "[ui]\ntheme = 3\n",
        "ui.theme in",
        "must be a string, not a whole number",
        id="theme",
    ),
    pytest.param(
        BASE + "[ui]\ndiff_style = true\n",
        "ui.diff_style in",
        "must be a string, not true or false",
        id="diff_style",
    ),
]


@pytest.mark.parametrize(("text", "where", "problem"), WRONG_TYPES)
def test_a_value_of_the_wrong_type_is_refused_with_its_key_its_file_and_what_it_is(
    tmp_path, text, where, problem
):
    message = refusal(tmp_path, text)
    assert f"{where} {home_file(tmp_path)} {problem}" in message


UNKNOWN_KEYS = [
    pytest.param(
        BASE + "[limits]\nmax_turn = 5\n",
        "unknown key 'max_turn' in [limits]",
        "the keys are max_turns, bash_timeout_s, output_cap_bytes, compact_soft, "
        "compact_hard, keep_recent_turns",
        id="limits",
    ),
    pytest.param(
        BASE + '[ui]\ncolour = "red"\n',
        "unknown key 'colour' in [ui]",
        "the keys are theme, diff_style",
        id="ui",
    ),
    pytest.param(
        BASE + "[permissions]\nblock = []\n",
        "unknown key 'block' in [permissions]",
        "the keys are allow, ask, deny",
        id="permissions",
    ),
    pytest.param(
        BASE + "temperature = 1\n",
        "unknown key 'temperature' in [models.m]",
        "the keys are adapter, model, base_url, api_key_env, context_window, native_tools",
        id="model",
    ),
]


@pytest.mark.parametrize(("text", "what", "keys"), UNKNOWN_KEYS)
def test_a_key_nothing_reads_is_refused_with_the_keys_that_exist(tmp_path, text, what, keys):
    # A misspelled limit that is quietly dropped is a limit the person believes
    # is in force.
    message = refusal(tmp_path, text)
    assert f"{what} of {home_file(tmp_path)}" in message
    assert keys in message


def test_a_misspelled_role_is_refused_with_the_roles_that_exist(tmp_path):
    # explore = ... misspelled would leave exploration on the expensive model
    # without a word, which is the whole saving this tool exists to make.
    message = refusal(tmp_path, BASE + '[roles]\nexplor = "m"\n')
    assert f"unknown role 'explor' in [roles] of {home_file(tmp_path)}" in message
    assert "the roles are main, explore, plan, verify, compact, title" in message


def test_an_unknown_adapter_names_the_model_and_every_adapter_there_is(tmp_path):
    message = refusal(tmp_path, '[models.m]\nadapter = "telepathy"\nmodel = "x"\n')
    assert "model 'm' uses adapter 'telepathy'" in message
    assert "use one of: anthropic, openai_compat, ollama" in message


def test_a_model_with_no_adapter_is_refused_with_every_adapter_there_is(tmp_path):
    message = refusal(tmp_path, '[models.m]\nmodel = "x"\n')
    assert "model 'm' has no adapter" in message
    assert "set adapter to one of: anthropic, openai_compat, ollama" in message


def test_an_adapter_the_project_names_is_checked_after_the_files_are_merged(tmp_path):
    # Neither file is wrong on its own terms: the home file gives the model, the
    # project file gives a bad adapter for it.
    message = refusal(tmp_path, BASE, '[models.m]\nadapter = "telepathy"\n')
    assert "model 'm' uses adapter 'telepathy'" in message


@pytest.mark.parametrize("model_line", ["", 'model = ""\n'])
def test_a_model_with_no_model_id_is_refused_and_told_where_to_put_one(tmp_path, model_line):
    message = refusal(tmp_path, '[models.m]\nadapter = "anthropic"\n' + model_line)
    assert "model 'm' has no model id" in message
    assert 'set model = "<id>" in [models.m]' in message


def test_a_model_the_project_only_half_defines_is_refused(tmp_path):
    message = refusal(tmp_path, BASE, '[models.p]\nmodel = "claude-opus-5"\n')
    assert "model 'p' has no adapter" in message


def test_openai_compat_needs_a_base_url(tmp_path):
    message = refusal(tmp_path, '[models.m]\nadapter = "openai_compat"\nmodel = "gpt-5"\n')
    assert "model 'm' uses openai_compat but has no base_url" in message
    assert "such as https://api.openai.com/v1" in message


@pytest.mark.parametrize("adapter", ["anthropic", "ollama"])
def test_the_other_adapters_do_not_need_a_base_url(tmp_path, adapter):
    config = load(tmp_path, f'[models.m]\nadapter = "{adapter}"\nmodel = "x"\n')
    assert config.models["m"].base_url is None


@pytest.mark.parametrize("url", ["localhost:11434", "ftp://host/v1", "//host/v1", ""])
def test_a_base_url_without_an_http_scheme_is_refused(tmp_path, url):
    message = refusal(tmp_path, BASE + f'base_url = "{url}"\n')
    assert (
        f"models.m.base_url in {home_file(tmp_path)} must start with http:// or https://" in message
    )


@pytest.mark.parametrize("url", ["http://localhost:8080/v1", "https://api.openai.com/v1"])
def test_an_http_or_https_base_url_is_accepted(tmp_path, url):
    assert load(tmp_path, BASE + f'base_url = "{url}"\n').models["m"].base_url == url


@pytest.mark.parametrize("name", ["MY_KEY", "_X", "a1", "OPENROUTER_API_KEY"])
def test_a_variable_name_is_accepted_for_api_key_env(tmp_path, name):
    assert load(tmp_path, BASE + f'api_key_env = "{name}"\n').models["m"].api_key_env == name


@pytest.mark.parametrize(
    "value", ["sk-ant-api03-abc", "has.dot", "1_LEADING_DIGIT", "two words", ""]
)
def test_api_key_env_that_is_not_a_variable_name_is_refused(tmp_path, value):
    message = refusal(tmp_path, BASE + f'api_key_env = "{value}"\n')
    assert (
        f"models.m.api_key_env in {home_file(tmp_path)} must be the name of an environment variable"
        in message
    )


def test_a_key_pasted_into_api_key_env_is_not_echoed_back(tmp_path):
    # The usual mistake is pasting the key itself where its variable's name goes,
    # and an error message is the one place that would print it to a terminal.
    pasted = "sk-ant-api03-SECRETSECRETSECRET"
    message = refusal(tmp_path, BASE + f'api_key_env = "{pasted}"\n')
    assert "SECRETSECRET" not in message
    assert "sk-ant" not in message


def test_a_key_written_in_the_file_is_refused_rather_than_ignored(tmp_path):
    message = refusal(tmp_path, BASE + 'api_key = "sk-ant-api03-SECRETSECRETSECRET"\n')
    assert f"models.m in {home_file(tmp_path)} has an api_key" in message
    assert "keys never go in a config file" in message
    assert "api_key_env" in message
    assert "SECRETSECRET" not in message


def test_a_role_pointing_at_an_undefined_model_says_what_to_define_or_pick(tmp_path):
    message = refusal(tmp_path, TWO_MODELS + '[roles]\nexplore = "nope"\n')
    assert "role 'explore' points at model 'nope', which is not defined" in message
    assert "define [models.nope] or point the role at one of: a, b" in message


def test_an_undefined_main_is_reported_as_main_not_as_every_role_following_it(tmp_path):
    message = refusal(tmp_path, BASE + '[roles]\nmain = "nope"\n')
    assert "role 'main' points at model 'nope'" in message


def test_a_role_set_to_an_empty_string_is_refused_rather_than_read_as_unset(tmp_path):
    message = refusal(tmp_path, BASE + '[roles]\nexplore = ""\n')
    assert f"roles.explore in {home_file(tmp_path)} is empty" in message
    assert "delete the line to leave it unset" in message


@pytest.mark.parametrize("value", [0, -1, -100])
def test_keep_recent_turns_below_one_is_refused_naming_the_key_and_the_minimum(tmp_path, value):
    message = refusal(tmp_path, BASE + f"[limits]\nkeep_recent_turns = {value}\n")
    assert (
        f"limits.keep_recent_turns in {home_file(tmp_path)} is {value}, below the minimum of 1"
        in message
    )
    assert "set it to 1 or more" in message


def test_keep_recent_turns_of_one_is_accepted(tmp_path):
    config = load(tmp_path, BASE + "[limits]\nkeep_recent_turns = 1\n")
    assert config.limits.keep_recent_turns == 1


async def test_the_minimum_the_loader_enforces_is_the_one_full_compaction_needs(tmp_path):
    # The loader exists so that this never surfaces as a traceback from inside
    # compaction. Tie the two together: 0 is refused by both, 1 is accepted by both.
    from nanoclaude.conversation.compaction import full_compact
    from nanoclaude.conversation.transcript import Transcript

    async def summarise(_text: str) -> str:
        return "unused"

    refused = refusal(tmp_path, BASE + "[limits]\nkeep_recent_turns = 0\n")
    assert "limits.keep_recent_turns" in refused
    with pytest.raises(ValueError, match="keep_recent must be at least 1"):
        await full_compact(Transcript(()), keep_recent=0, summarise=summarise)

    config = load(tmp_path, BASE + "[limits]\nkeep_recent_turns = 1\n")
    kept = Transcript(())
    result = await full_compact(
        kept, keep_recent=config.limits.keep_recent_turns, summarise=summarise
    )
    assert result is kept


@pytest.mark.parametrize("key", ["max_turns", "output_cap_bytes"])
@pytest.mark.parametrize("value", [0, -5])
def test_a_whole_number_limit_below_one_is_refused(tmp_path, key, value):
    message = refusal(tmp_path, BASE + f"[limits]\n{key} = {value}\n")
    assert f"limits.{key} in {home_file(tmp_path)} is {value}, below the minimum of 1" in message


def test_a_whole_number_limit_of_one_is_accepted(tmp_path):
    limits = load(tmp_path, BASE + "[limits]\nmax_turns = 1\noutput_cap_bytes = 1\n").limits
    assert (limits.max_turns, limits.output_cap_bytes) == (1, 1)


@pytest.mark.parametrize("value", ["0", "-1", "-0.5", "nan"])
def test_a_timeout_that_is_not_above_zero_is_refused(tmp_path, value):
    message = refusal(tmp_path, BASE + f"[limits]\nbash_timeout_s = {value}\n")
    assert f"limits.bash_timeout_s in {home_file(tmp_path)} is {value}" in message
    assert "set a positive number of seconds" in message


@pytest.mark.parametrize("value", [0.001, 30, 3600.5])
def test_a_positive_timeout_is_accepted(tmp_path, value):
    config = load(tmp_path, BASE + f"[limits]\nbash_timeout_s = {value}\n")
    assert config.limits.bash_timeout_s == value


@pytest.mark.parametrize("key", ["compact_soft", "compact_hard"])
@pytest.mark.parametrize("value", ["0", "-0.1", "1.01", "2", "nan"])
def test_a_compaction_threshold_outside_zero_to_one_is_refused(tmp_path, key, value):
    message = refusal(tmp_path, BASE + f"[limits]\n{key} = {value}\n")
    assert f"limits.{key} in {home_file(tmp_path)} is {value}" in message
    assert "above 0 and at most 1" in message


@pytest.mark.parametrize("value", ["1", "1.0", "0.0001", "0.5"])
def test_the_edges_of_the_threshold_range_are_accepted(tmp_path, value):
    config = load(tmp_path, BASE + f"[limits]\ncompact_soft = {value}\ncompact_hard = {value}\n")
    assert config.limits.compact_soft == float(value)
    assert config.limits.compact_hard == float(value)


def test_a_context_window_below_one_is_refused(tmp_path):
    message = refusal(tmp_path, BASE + "context_window = 0\n")
    assert (
        f"models.m.context_window in {home_file(tmp_path)} is 0, below the minimum of 1" in message
    )


def test_a_model_may_override_what_the_probe_would_find(tmp_path):
    config = load(tmp_path, BASE + "context_window = 64000\nnative_tools = false\n")
    assert config.models["m"].context_window == 64_000
    assert config.models["m"].native_tools is False


@pytest.mark.parametrize("theme", ["auto", "light", "dark"])
def test_every_theme_the_spec_lists_is_accepted(tmp_path, theme):
    assert load(tmp_path, BASE + f'[ui]\ntheme = "{theme}"\n').ui.theme == theme


def test_a_theme_the_spec_does_not_list_is_refused(tmp_path):
    message = refusal(tmp_path, BASE + '[ui]\ntheme = "blue"\n')
    assert f"ui.theme in {home_file(tmp_path)} is 'blue'" in message
    assert "set it to one of: auto, light, dark" in message


def test_the_only_diff_style_is_unified(tmp_path):
    message = refusal(tmp_path, BASE + '[ui]\ndiff_style = "split"\n')
    assert f"ui.diff_style in {home_file(tmp_path)} is 'split'" in message
    assert "set it to one of: unified" in message


@pytest.mark.parametrize(
    "rule",
    ["Read", "Bash(npm test:*)", "Read(**/.env*)", "Git(status)", "Bash(git status)"],
)
def test_every_shape_of_rule_the_grammar_allows_is_accepted(tmp_path, rule):
    config = load(tmp_path, BASE + f'[permissions]\ndeny = ["{rule}"]\n')
    assert config.permissions.deny == (rule,)


@pytest.mark.parametrize("key", ["allow", "ask", "deny"])
@pytest.mark.parametrize(
    ("rule", "reason"),
    [
        ("Bash(", "unbalanced parentheses"),
        ("Bash)", "unbalanced parentheses"),
        ("Bash()", "empty subject"),
        ("Bash(:*)", "empty subject"),
    ],
)
def test_a_rule_the_grammar_rejects_is_refused_at_load_with_its_reason(tmp_path, key, rule, reason):
    # Rule.parse would raise a bare ValueError from inside session start-up; here
    # it is a message that names the list, the position and the reason.
    message = refusal(tmp_path, BASE + f'[permissions]\n{key} = ["Read", "{rule}"]\n')
    assert f"permissions.{key}[1] in {home_file(tmp_path)} is not a valid rule" in message
    assert reason in message
    assert "rules look like Tool, Tool(subject) or Tool(prefix:*)" in message


def test_a_mistake_in_the_project_file_names_the_project_file_only(tmp_path):
    message = refusal(tmp_path, BASE, '[limits]\nmax_turns = "x"\n')
    assert str(project_file(tmp_path)) in message
    assert str(home_file(tmp_path)) not in message


def test_a_mistake_in_the_home_file_names_it_even_when_the_project_overrides_the_value(tmp_path):
    message = refusal(tmp_path, BASE + '[limits]\nmax_turns = "x"\n', "[limits]\nmax_turns = 5\n")
    assert str(home_file(tmp_path)) in message
    assert str(project_file(tmp_path)) not in message


def test_an_integer_for_a_fractional_limit_is_accepted_and_becomes_a_float(tmp_path):
    limits = load(tmp_path, BASE + "[limits]\nbash_timeout_s = 120\ncompact_soft = 1\n").limits
    assert limits.bash_timeout_s == 120.0 and isinstance(limits.bash_timeout_s, float)
    assert limits.compact_soft == 1.0 and isinstance(limits.compact_soft, float)


# --------------------------------------------------------------------------
# What `ncc init` writes (spec 17.6, rendered the way cli/init.py renders it)
# --------------------------------------------------------------------------

INIT_ANTHROPIC = """\
# nano-claude-code configuration.
# Everything here has a default; delete anything you do not want to change.

[models.anthropic]
adapter     = "anthropic"
model       = "claude-sonnet-5"
api_key_env = "ANTHROPIC_API_KEY"

# Roles let one session use different models for different jobs.
[roles]
main    = "anthropic"
explore = "anthropic"
plan    = "anthropic"
verify  = "anthropic"
compact = "anthropic"
title   = "anthropic"

[permissions]
allow = ["Read", "Grep", "Glob"]
ask   = ["Bash", "Write", "Edit", "Git"]
deny  = ["Read(**/.env*)"]

[limits]
max_turns      = 40
bash_timeout_s = 120
"""

INIT_OPENROUTER = """\
[models.openrouter]
adapter     = "openai_compat"
model       = "deepseek/deepseek-v3"
base_url    = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[roles]
main    = "openrouter"
explore = "openrouter"
plan    = "openrouter"
verify  = "openrouter"
compact = "openrouter"
title   = "openrouter"
"""


def test_the_config_ncc_init_writes_for_anthropic_loads(tmp_path):
    config = load(tmp_path, INIT_ANTHROPIC, env={"ANTHROPIC_API_KEY": "k"})
    assert config.models["anthropic"].model == "claude-sonnet-5"
    assert config.roles.title == "anthropic"
    assert config.permissions.deny == ("Read(**/.env*)",)
    assert config.limits.bash_timeout_s == 120.0  # written as an integer
    assert config.api_key_for("anthropic") == "k"


def test_the_config_ncc_init_writes_for_an_openai_compatible_endpoint_loads(tmp_path):
    config = load(tmp_path, INIT_OPENROUTER, env={"OPENROUTER_API_KEY": "k"})
    assert config.models["openrouter"].base_url == "https://openrouter.ai/api/v1"
    assert config.api_key_env_for("openrouter") == "OPENROUTER_API_KEY"
    assert config.api_key_for("openrouter") == "k"


# --------------------------------------------------------------------------
# Sandbox roots. Sandbox refuses a root that is not absolute, by design, so a
# root never depends on where the process happens to be standing. People write
# "~/src/lib" in a command line and in a config file all the same.
# --------------------------------------------------------------------------


def test_a_tilde_root_is_expanded_before_a_sandbox_sees_it(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "src" / "lib").mkdir(parents=True)
    with pytest.raises(ValueError, match="must be absolute"):
        Sandbox(("~/src/lib",))  # what happens without the expansion
    root = expand_root("~/src/lib")
    assert root == str((tmp_path / "src" / "lib").resolve())
    assert Sandbox((root,)).roots == (root,)


def test_a_relative_root_is_anchored_to_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lib").mkdir()
    assert expand_root("lib") == str((tmp_path / "lib").resolve())
    assert expand_root("./lib/../lib") == str((tmp_path / "lib").resolve())


def test_an_absolute_root_is_only_normalised(tmp_path):
    (tmp_path / "lib").mkdir()
    assert expand_root(str(tmp_path / "lib" / "..")) == str(tmp_path.resolve())


def test_only_a_leading_tilde_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "somewhere-else"))
    root = tmp_path / "a~b"
    root.mkdir()
    assert expand_root(str(root)) == str(root.resolve())


def test_a_tilde_for_a_user_that_does_not_exist_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError) as caught:
        expand_root("~no-such-user-for-nanoclaude/lib")
    message = str(caught.value)
    assert "cannot expand '~no-such-user-for-nanoclaude/lib'" in message
    assert f" {EM_DASH} " in message
    assert message.endswith("write the full path")


def test_home_and_project_given_with_a_tilde_are_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write(tmp_path / ".nanoclaude" / "config.toml", BASE)
    write(tmp_path / "proj" / ".nanoclaude" / "config.toml", "[limits]\nmax_turns = 7\n")
    config = load_config(home="~", project="~/proj", env={})
    assert config.limits.max_turns == 7
    assert config.roles.main == "m"


def test_a_home_that_cannot_be_expanded_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot expand '~no-such-user-for-nanoclaude'"):
        load_config(home="~no-such-user-for-nanoclaude", project=None, env={})


# --------------------------------------------------------------------------
# The two checks the router shares with the loader
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("adapter", "model", "base_url", "fragment"),
    [
        (None, "x", None, "model 'm' has no adapter"),
        ("telepathy", "x", None, "model 'm' uses adapter 'telepathy'"),
        ("anthropic", "", None, "model 'm' has no model id"),
        ("anthropic", None, None, "model 'm' has no model id"),
        ("openai_compat", "gpt-5", None, "model 'm' uses openai_compat but has no base_url"),
        ("openai_compat", "gpt-5", "", "model 'm' uses openai_compat but has no base_url"),
    ],
)
def test_check_model_refuses_an_entry_that_cannot_be_used(adapter, model, base_url, fragment):
    with pytest.raises(ConfigError) as caught:
        check_model("m", adapter, model, base_url)
    assert fragment in str(caught.value)
    assert f" {EM_DASH} " in str(caught.value)


@pytest.mark.parametrize(
    ("adapter", "base_url"),
    [
        ("anthropic", None),
        ("ollama", None),
        ("openai_compat", "https://x/v1"),
        ("anthropic", "https://proxy/v1"),
    ],
)
def test_check_model_accepts_every_usable_entry(adapter, base_url):
    check_model("m", adapter, "some-model", base_url)  # does not raise


def test_check_roles_names_the_first_role_that_points_nowhere():
    # Defined out of alphabetical order, so a list that is not sorted shows.
    models = {"b": ModelConfig("ollama", "y"), "a": ModelConfig("anthropic", "x")}
    roles = RolesConfig("a", "a", "nope", "b", "nope-too", "a")
    with pytest.raises(ConfigError) as caught:
        check_roles(roles, models)
    message = str(caught.value)
    assert "role 'plan' points at model 'nope'" in message  # roles are checked in ROLES order
    assert "one of: a, b" in message


def test_check_roles_accepts_roles_that_all_point_at_defined_models():
    models = {"a": ModelConfig("anthropic", "x")}
    check_roles(RolesConfig("a", "a", "a", "a", "a", "a"), models)  # does not raise


_POOL = ["Read", "Grep", "Bash", "Bash(npm test:*)", "Read(**/.env*)", "Write(src/**)", "Git"]


def _rule_list(rules: list[str]) -> str:
    return "[" + ", ".join(f'"{rule}"' for rule in rules) + "]"


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    key=st.sampled_from(["allow", "ask", "deny"]),
    home=st.one_of(st.none(), st.lists(st.sampled_from(_POOL), max_size=5)),
    project=st.lists(st.sampled_from(_POOL), max_size=5),
)
def test_a_project_file_only_ever_adds_rules(key, home, project):
    """Whatever the lists are, adding a project file removes no rule that applied without it."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        home_text = BASE + (
            f"[permissions]\n{key} = {_rule_list(home)}\n" if home is not None else ""
        )
        write(root / "home" / ".nanoclaude" / "config.toml", home_text)
        write(
            root / "proj" / ".nanoclaude" / "config.toml",
            f"[permissions]\n{key} = {_rule_list(project)}\n",
        )
        without = getattr(
            load_config(home=str(root / "home"), project=None, env={}).permissions, key
        )
        stacked = load_config(home=str(root / "home"), project=str(root / "proj"), env={})
        with_project = getattr(stacked.permissions, key)
    assert all(rule in with_project for rule in without), "a rule that applied was removed"
    assert all(rule in with_project for rule in project), "a project rule was not added"
    assert len(with_project) == len(set(with_project)), "a rule is listed twice"
    kept = [rule for rule in with_project if rule in without]
    assert kept == list(dict.fromkeys(without)), "the order of the rules beneath was disturbed"


# --------------------------------------------------------------------------
# No traceback reaches the user: whatever two files say, the answer is a Config
# the rest of the program can rely on, or a ConfigError in spec 17.9's form.
# --------------------------------------------------------------------------

# For each key, what it accepts and what it refuses. A file is built from the
# first and then given, now and then, one value from the second: files drawn from
# random values alone would be refused before the boundaries were reached, and a
# refusal that is the only thing wrong is the one whose branch is under test.
_Values = tuple[list[str], list[str]]  # (what a key accepts, what it refuses)

_MODEL_KEYS: dict[str, _Values] = {
    "adapter": (['"anthropic"', '"ollama"'], ['"openai_compat"', '"telepathy"', '""', "3"]),
    "model": (['"claude-sonnet-5"', '"x"'], ['""', "[1]"]),
    "base_url": (['"https://x/v1"', '"http://h:1"'], ['"localhost"', '""', "5"]),
    "api_key_env": (['"MY_KEY"', '"A1"'], ['"sk-ant-aaaa"', '""', "5"]),
    "context_window": (["1", "64000"], ["0", "1.5", '"big"']),
    "native_tools": (["true", "false"], ['"yes"']),
    "api_key": ([], ['"sk-ant-aaaa"']),
}
_RULES: _Values = (
    ['["Read"]', '["Bash(npm test:*)"]', "[]"],
    ['["Bash("]', '["Bash()"]', '["Read", 1]', '"Read"'],
)
_SECTIONS: dict[str, dict[str, _Values]] = {
    "models.m": _MODEL_KEYS,
    "roles": {
        **{role: (['"m"'], ['""', "5", '"nope"']) for role in ROLES},
        "reviewer": ([], ['"m"']),
    },
    "permissions": {"allow": _RULES, "ask": _RULES, "deny": _RULES},
    "limits": {
        "max_turns": (["1", "40"], ["0", "-1", "1.5", '"40"', "true"]),
        "bash_timeout_s": (["30", "0.5"], ["0", "-1", "nan", '"30"']),
        "output_cap_bytes": (["1", "100000"], ["0", "2.5"]),
        "compact_soft": (["0.5", "1"], ["0", "1.5", "nan", '"0.5"']),
        "compact_hard": (["0.85", "1"], ["0", "2", "nan", '"x"']),
        "keep_recent_turns": (["1", "3"], ["0", "-1", "2.5", '"3"', "true"]),
    },
    "ui": {
        "theme": (['"auto"', '"dark"'], ['"blue"', "1"]),
        "diff_style": (['"unified"'], ['"split"', "true"]),
    },
    "hooks": {"pre_tool_use": ([], ['["x"]'])},
}  # fmt: skip
_REFUSALS = [
    (header, key, value)
    for header, table in _SECTIONS.items()
    for key, (_, refused) in table.items()
    for value in refused
]
_PRELUDES = [""] * 6 + ["limits = 5\n", 'models = "x"\n', "stray = 1\n"]


@st.composite
def _file_text(draw: st.DrawFn, *, with_model: bool) -> str:
    tables: dict[str, dict[str, str]] = {}
    if with_model:
        tables["models.m"] = {"adapter": '"anthropic"', "model": '"claude-sonnet-5"'}
    for header in draw(st.lists(st.sampled_from(list(_SECTIONS)), max_size=5, unique=True)):
        for key in draw(
            st.lists(st.sampled_from(list(_SECTIONS[header])), max_size=5, unique=True)
        ):
            accepted, _ = _SECTIONS[header][key]
            if accepted:
                tables.setdefault(header, {})[key] = draw(st.sampled_from(accepted))
    if draw(st.booleans()):  # and now and then one value that should be refused
        header, key, value = draw(st.sampled_from(_REFUSALS))
        tables.setdefault(header, {})[key] = value
    lines = [draw(st.sampled_from(_PRELUDES))]
    for header, values in tables.items():
        lines.append(f"[{header}]")
        lines.extend(f"{key} = {value}" for key, value in values.items())
    return "\n".join(lines) + "\n"


@settings(
    max_examples=500,
    deadline=None,
    # The autouse fixture only points HOME somewhere harmless, the same way for
    # every example, so sharing it between them cannot change what one finds.
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    home=_file_text(with_model=True),
    project=_file_text(with_model=False),
    with_project=st.booleans(),
)
def test_any_two_files_give_a_usable_config_or_a_refusal_in_spec_form(home, project, with_project):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write(root / "home" / ".nanoclaude" / "config.toml", home)
        if with_project:
            write(root / "proj" / ".nanoclaude" / "config.toml", project)
        try:
            config = load_config(
                home=str(root / "home"),
                project=str(root / "proj") if with_project else None,
                env={"ANTHROPIC_API_KEY": "k"},
            )
        except ConfigError as refused:
            message = str(refused)
            assert f" {EM_DASH} " in message and message[:1].islower(), message
            return
        # What the rest of the program relies on, for anything that was accepted:
        # the ranges, and the types the dataclasses claim, which nothing else checks
        # of data that came out of a file.
        limits = config.limits
        assert isinstance(limits.max_turns, int) and limits.max_turns >= 1
        assert isinstance(limits.output_cap_bytes, int) and limits.output_cap_bytes >= 1
        assert isinstance(limits.keep_recent_turns, int) and limits.keep_recent_turns >= 1
        assert isinstance(limits.bash_timeout_s, float) and limits.bash_timeout_s > 0
        assert isinstance(limits.compact_soft, float) and 0 < limits.compact_soft <= 1
        assert isinstance(limits.compact_hard, float) and 0 < limits.compact_hard <= 1
        assert config.ui.theme in THEMES and config.ui.diff_style in DIFF_STYLES
        for rules in (config.permissions.allow, config.permissions.ask, config.permissions.deny):
            assert isinstance(rules, tuple)
            for rule in rules:
                assert isinstance(rule, str)
                Rule.parse(rule)  # does not raise
        assert all(config.roles.alias_for(role) in config.models for role in ROLES)
        for entry in config.models.values():
            assert entry.adapter in KNOWN_ADAPTERS and entry.model
            assert entry.adapter != "openai_compat" or entry.base_url
            assert entry.base_url is None or entry.base_url.startswith(("http://", "https://"))
            assert entry.api_key_env is None or entry.api_key_env.isidentifier()
            assert entry.context_window is None or (
                isinstance(entry.context_window, int) and entry.context_window >= 1
            )
            assert entry.native_tools is None or isinstance(entry.native_tools, bool)
