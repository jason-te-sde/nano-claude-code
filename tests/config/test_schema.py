import dataclasses
from collections.abc import Hashable

import pytest

from nanoclaude.config.schema import (
    DEFAULT_KEY_ENV,
    KNOWN_ADAPTERS,
    ROLES,
    Config,
    LimitsConfig,
    ModelConfig,
    PermissionsConfig,
    RolesConfig,
    UiConfig,
)


def roles_all(alias: str) -> RolesConfig:
    return RolesConfig(alias, alias, alias, alias, alias, alias)


def config_for(entry: ModelConfig, env: dict[str, str]) -> Config:
    return Config(models={"m": entry}, roles=roles_all("m"), env=env)


def test_the_roles_are_the_six_slots_in_the_order_reports_list_them():
    assert ROLES == ("main", "explore", "plan", "verify", "compact", "title")


def test_roles_config_has_one_field_per_role_in_the_same_order():
    # RolesConfig is built positionally elsewhere, in this order, and ROLES drives
    # every loop over it; the two lists are one list written twice, so they are
    # locked together here rather than trusted to stay in step.
    assert tuple(field.name for field in dataclasses.fields(RolesConfig)) == ROLES


def test_the_known_adapters_are_the_three_the_spec_names():
    assert KNOWN_ADAPTERS == ("anthropic", "openai_compat", "ollama")


@pytest.mark.parametrize("role", ROLES)
def test_alias_for_returns_the_alias_each_role_is_routed_to(role):
    # Six different aliases, so a role answered from the wrong field is visible.
    roles = RolesConfig(*(f"alias-{name}" for name in ROLES))
    assert roles.alias_for(role) == f"alias-{role}"


@pytest.mark.parametrize("name", ["reviewer", "", "alias_for", "__class__"])
def test_alias_for_refuses_a_name_that_is_not_a_role(name):
    # The last two are attributes of the object, which a bare getattr would hand
    # back as if they were aliases.
    with pytest.raises(ValueError, match="unknown role"):
        roles_all("m").alias_for(name)


@pytest.mark.parametrize(
    ("adapter", "variable"),
    [("anthropic", "ANTHROPIC_API_KEY"), ("openai_compat", "OPENAI_API_KEY")],
)
def test_an_adapter_that_needs_a_key_defaults_to_its_own_variable(adapter, variable):
    config = config_for(
        ModelConfig(adapter, "x", base_url="https://x/v1"),
        {variable: "the-key", "SOMETHING_ELSE": "not-it"},
    )
    assert config.api_key_env_for("m") == variable
    assert config.api_key_for("m") == "the-key"


def test_a_local_adapter_has_no_key_variable_and_no_key():
    config = config_for(ModelConfig("ollama", "qwen3-coder"), {"ANTHROPIC_API_KEY": "k"})
    assert config.api_key_env_for("m") == ""
    assert config.api_key_for("m") == ""


def test_a_model_whose_adapter_is_not_known_has_no_key_variable():
    # Refusing such a model is check_model's job. Asking what variable it uses
    # must still answer, not raise a KeyError from the table.
    config = config_for(ModelConfig("telepathy", "x"), {"ANTHROPIC_API_KEY": "k"})
    assert config.api_key_env_for("m") == ""
    assert config.api_key_for("m") == ""


def test_a_named_variable_beats_the_adapters_default():
    config = config_for(
        ModelConfig("anthropic", "x", api_key_env="MY_KEY"),
        {"MY_KEY": "mine", "ANTHROPIC_API_KEY": "default"},
    )
    assert config.api_key_env_for("m") == "MY_KEY"
    assert config.api_key_for("m") == "mine"


def test_a_named_variable_that_is_unset_does_not_fall_back_to_the_default():
    # Falling back would send one provider's key to another: a model pointed at
    # a third-party endpoint with its own variable must not quietly pick up the
    # Anthropic key just because that one happens to be set.
    config = config_for(
        ModelConfig("openai_compat", "x", base_url="https://x/v1", api_key_env="THEIR_KEY"),
        {"OPENAI_API_KEY": "someone-elses", "ANTHROPIC_API_KEY": "also-not-this"},
    )
    assert config.api_key_for("m") == ""


def test_a_variable_missing_from_the_environment_gives_an_empty_key():
    config = config_for(ModelConfig("anthropic", "x"), {})
    assert config.api_key_for("m") == ""


def test_the_default_variables_cannot_be_edited():
    with pytest.raises(TypeError):
        DEFAULT_KEY_ENV["anthropic"] = "SOMETHING_ELSE"  # type: ignore[index]


def test_printing_a_config_does_not_print_the_keys_in_its_environment():
    config = config_for(ModelConfig("anthropic", "x"), {"ANTHROPIC_API_KEY": "sk-secret-value"})
    assert "sk-secret-value" not in repr(config)
    assert "sk-secret-value" not in str(config)


@pytest.mark.parametrize(
    "value",
    [
        ModelConfig("anthropic", "x"),
        roles_all("m"),
        LimitsConfig(),
        PermissionsConfig(),
        UiConfig(),
    ],
    ids=lambda value: type(value).__name__,
)
def test_the_plain_value_types_are_hashable_because_they_hold_only_scalars(value):
    # Strings, numbers and tuples of strings: nothing in them can make hash()
    # raise, so the generated __hash__ is honest. Pinned as a decision.
    clone = dataclasses.replace(value)
    assert isinstance(value, Hashable)
    assert hash(value) == hash(clone)


def test_config_is_not_hashable():
    # It holds two dicts (models and env). Left to frozen=True's default,
    # dataclass would generate a real __hash__ that reports Hashable and only
    # raises when called, naming "dict" rather than this class.
    config = config_for(ModelConfig("anthropic", "x"), {})
    assert not isinstance(config, Hashable)  # type: ignore[unreachable]
    with pytest.raises(TypeError, match="unhashable type: 'Config'"):
        hash(config)
