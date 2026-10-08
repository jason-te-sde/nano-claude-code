import dataclasses
import datetime as dt
from collections.abc import AsyncIterator, Callable, Hashable
from pathlib import Path
from typing import Any

import httpx
import pytest

from nanoclaude.agent.router import RoleUsage, Router
from nanoclaude.config.load import ConfigError
from nanoclaude.config.schema import ROLES, Config, ModelConfig, RolesConfig
from nanoclaude.providers.anthropic import AnthropicClient
from nanoclaude.providers.base import (
    CredentialsError,
    ModelClient,
    ModelError,
    ModelReply,
    ModelRequest,
    Usage,
)
from nanoclaude.providers.capabilities import CONSERVATIVE_DEFAULT, Capabilities, CapabilityCache
from nanoclaude.providers.ollama import OllamaClient
from nanoclaude.providers.openai_compat import OpenAICompatClient
from nanoclaude.providers.pricing import Price, PriceBook


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """A router builds clients; nothing here may send a request with one."""

    async def refuse(self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> None:
        raise AssertionError(f"a router test reached for the network: {request.url}")

    monkeypatch.setattr(httpx.AsyncClient, "send", refuse)


def config_with(**roles: str) -> Config:
    models = {
        "big": ModelConfig("anthropic", "claude-sonnet-5"),
        "small": ModelConfig("ollama", "qwen3-coder"),
    }
    base = {"main": "big", "explore": "big", "plan": "big", "verify": "big", "compact": "big"}
    base["title"] = "big"
    base.update(roles)
    return Config(models=models, roles=RolesConfig(**base), env={"ANTHROPIC_API_KEY": "k"})


def roles_all(alias: str) -> RolesConfig:
    return RolesConfig(alias, alias, alias, alias, alias, alias)


class FakeClient:
    """A ModelClient that records being closed, and can fail to close."""

    def __init__(
        self, model_id: str, *, log: list[str] | None = None, fail_on_close: bool = False
    ) -> None:
        self._model_id = model_id
        self._log = log if log is not None else []
        self._fail_on_close = fail_on_close  # once: closing again afterwards works

    @property
    def model_id(self) -> str:
        return self._model_id

    async def complete(
        self,
        request: ModelRequest,
        *,
        on_text: Callable[[str], None] | None = None,  # noqa: ARG002 - the protocol's keyword
    ) -> ModelReply:
        raise AssertionError(f"a router test must not send a request: {request}")

    async def aclose(self) -> None:
        self._log.append(self._model_id)
        if self._fail_on_close:
            self._fail_on_close = False
            raise RuntimeError(f"{self._model_id} would not close")


class ClientWithoutClose:
    """A ModelClient with nothing to close, like the scripted model."""

    model_id = "no-close"

    async def complete(
        self,
        request: ModelRequest,
        *,
        on_text: Callable[[str], None] | None = None,  # noqa: ARG002 - the protocol's keyword
    ) -> ModelReply:
        raise AssertionError(f"a router test must not send a request: {request}")


def spy_on(
    monkeypatch: pytest.MonkeyPatch, adapter_class: str
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Replace one adapter class in the router with one that records how it was built."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class Spy:
        model_id = "spy"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(f"nanoclaude.agent.router.{adapter_class}", Spy)
    return calls


@pytest.fixture
async def make(tmp_path: Path) -> AsyncIterator[Callable[..., Router]]:
    """Build routers that share one capability cache file, and close them afterwards."""
    routers: list[Router] = []

    def build(config: Config, *args: Any, **kwargs: Any) -> Router:
        router = Router(config, CapabilityCache(tmp_path / "caps.json"), *args, **kwargs)
        routers.append(router)
        return router

    yield build
    for router in routers:
        await router.aclose()


# --------------------------------------------------------------------------
# The brief's own tests
# --------------------------------------------------------------------------


async def test_each_role_resolves_to_its_configured_model(tmp_path):
    router = Router(config_with(explore="small"), CapabilityCache(tmp_path / "c.json"))
    assert (await router.client_for("main")).model_id == "claude-sonnet-5"
    assert (await router.client_for("explore")).model_id == "qwen3-coder"


async def test_clients_are_reused_across_calls(tmp_path):
    router = Router(config_with(), CapabilityCache(tmp_path / "c.json"))
    assert await router.client_for("main") is await router.client_for("main")


async def test_usage_is_attributed_per_role(tmp_path):
    router = Router(config_with(explore="small"), CapabilityCache(tmp_path / "c.json"))
    router.record("main", Usage(input_tokens=100, output_tokens=10), "anthropic", "claude-sonnet-5")
    router.record("explore", Usage(input_tokens=9000), "ollama", "qwen3-coder")
    by_role = router.by_role()
    assert by_role["main"].usage.input_tokens == 100
    assert by_role["explore"].usage.input_tokens == 9000


async def test_total_cost_adds_up_and_local_models_contribute_nothing(tmp_path):
    router = Router(config_with(explore="small"), CapabilityCache(tmp_path / "c.json"))
    router.record("main", Usage(input_tokens=1_000_000), "anthropic", "claude-sonnet-5")
    router.record("explore", Usage(input_tokens=5_000_000), "ollama", "qwen3-coder")
    assert router.total_cost() == pytest.approx(2.0, rel=0.01)


async def test_total_cost_is_none_when_any_model_has_no_price(tmp_path):
    """One unknown model makes the total a guess, so it is reported as unknown."""
    config = config_with()
    config.models["big"] = ModelConfig("openai_compat", "mystery", base_url="https://x/v1")
    router = Router(config, CapabilityCache(tmp_path / "c.json"))
    router.record("main", Usage(input_tokens=1000), "openai_compat", "mystery")
    assert router.total_cost() is None


async def test_a_missing_api_key_names_the_variable_and_the_role(tmp_path):
    config = config_with()
    config.env.clear()
    router = Router(config, CapabilityCache(tmp_path / "c.json"))
    with pytest.raises(Exception, match="ANTHROPIC_API_KEY"):
        await router.client_for("main")


# --------------------------------------------------------------------------
# Routing: each role to the model it names, one client per model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("role", ROLES)
async def test_every_role_is_routed_to_its_own_model(make, role):
    # Six different models, so a role answered from another role's slot is visible.
    models = {f"{name}-alias": ModelConfig("ollama", f"model-for-{name}") for name in ROLES}
    config = Config(models=models, roles=RolesConfig(*(f"{name}-alias" for name in ROLES)))
    client = await make(config).client_for(role)
    assert client.model_id == f"model-for-{role}"


async def test_roles_that_share_a_model_share_one_client(make):
    router = make(config_with())
    assert await router.client_for("explore") is await router.client_for("main")


async def test_roles_on_different_models_get_different_clients(make):
    router = make(config_with(explore="small"))
    assert await router.client_for("explore") is not await router.client_for("main")


async def test_each_adapter_builds_the_client_that_matches_it(make):
    models = {
        "a": ModelConfig("anthropic", "claude-sonnet-5"),
        "o": ModelConfig("openai_compat", "gpt-5", base_url="https://api.openai.com/v1"),
        "l": ModelConfig("ollama", "qwen3-coder"),
    }
    env = {"ANTHROPIC_API_KEY": "k1", "OPENAI_API_KEY": "k2"}
    router = make(Config(models=models, roles=RolesConfig("a", "o", "l", "a", "a", "a"), env=env))
    anthropic = await router.client_for("main")
    openai = await router.client_for("explore")
    ollama = await router.client_for("plan")
    assert isinstance(anthropic, AnthropicClient) and anthropic.model_id == "claude-sonnet-5"
    assert isinstance(openai, OpenAICompatClient) and openai.model_id == "gpt-5"
    assert isinstance(ollama, OllamaClient) and ollama.model_id == "qwen3-coder"
    assert all(isinstance(client, ModelClient) for client in (anthropic, openai, ollama))


async def test_the_configured_model_endpoint_and_variable_reach_every_adapter(make, monkeypatch):
    anthropic = spy_on(monkeypatch, "AnthropicClient")
    openai = spy_on(monkeypatch, "OpenAICompatClient")
    ollama = spy_on(monkeypatch, "OllamaClient")
    models = {
        "a": ModelConfig("anthropic", "claude-opus-5", base_url="https://gateway.example/claude"),
        "o": ModelConfig(
            "openai_compat",
            "deepseek/deepseek-v3",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        ),
        "l": ModelConfig("ollama", "qwen3-coder", base_url="http://gpu-box:11434"),
    }
    env = {"ANTHROPIC_API_KEY": "anthropic-key", "OPENROUTER_API_KEY": "openrouter-key"}
    router = make(Config(models=models, roles=RolesConfig("a", "o", "l", "a", "a", "a"), env=env))
    for role in ("main", "explore", "plan"):
        await router.client_for(role)
    assert anthropic == [
        (
            ("anthropic-key",),
            {"model": "claude-opus-5", "base_url": "https://gateway.example/claude"},
        )
    ]
    assert openai == [
        (
            ("openrouter-key",),
            {
                "model": "deepseek/deepseek-v3",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
            },
        )
    ]
    assert ollama == [((), {"model": "qwen3-coder", "base_url": "http://gpu-box:11434"})]


async def test_an_endpoint_nobody_configured_is_the_adapters_own_default(make, monkeypatch):
    anthropic = spy_on(monkeypatch, "AnthropicClient")
    openai = spy_on(monkeypatch, "OpenAICompatClient")
    ollama = spy_on(monkeypatch, "OllamaClient")
    models = {
        "a": ModelConfig("anthropic", "claude-sonnet-5"),
        "o": ModelConfig("openai_compat", "gpt-5", base_url="https://api.openai.com/v1"),
        "l": ModelConfig("ollama", "qwen3-coder"),
    }
    env = {"ANTHROPIC_API_KEY": "k1", "OPENAI_API_KEY": "k2"}
    router = make(Config(models=models, roles=RolesConfig("a", "o", "l", "a", "a", "a"), env=env))
    for role in ("main", "explore", "plan"):
        await router.client_for(role)
    assert anthropic[0][1]["base_url"] == "https://api.anthropic.com"
    assert ollama[0][1]["base_url"] == "http://localhost:11434"
    assert openai[0][1]["api_key_env"] == "OPENAI_API_KEY"


async def test_each_model_is_given_the_key_from_its_own_variable(make, monkeypatch):
    openai = spy_on(monkeypatch, "OpenAICompatClient")
    models = {
        "x": ModelConfig("openai_compat", "m1", base_url="https://x/v1", api_key_env="X_KEY"),
        "y": ModelConfig("openai_compat", "m2", base_url="https://y/v1", api_key_env="Y_KEY"),
    }
    config = Config(
        models=models,
        roles=RolesConfig("x", "y", "x", "y", "x", "y"),
        env={"X_KEY": "key-for-x", "Y_KEY": "key-for-y", "OPENAI_API_KEY": "not-for-either"},
    )
    router = make(config)
    await router.client_for("main")
    await router.client_for("explore")
    assert [args[0] for args, _ in openai] == ["key-for-x", "key-for-y"]


# --------------------------------------------------------------------------
# A key that is not there: spec 17.9, naming the variable that would fix it
# --------------------------------------------------------------------------


async def test_a_missing_anthropic_key_names_the_role_the_model_and_the_variable(make):
    config = config_with()
    config.env.clear()
    with pytest.raises(CredentialsError) as caught:
        await make(config).client_for("verify")
    assert str(caught.value) == (
        "role 'verify' uses model 'big', which needs ANTHROPIC_API_KEY — set it, or run: ncc init"
    )
    assert caught.value.retryable is False


async def test_a_missing_anthropic_key_names_the_configured_variable_not_the_default(make):
    models = {"big": ModelConfig("anthropic", "claude-sonnet-5", api_key_env="WORK_KEY")}
    config = Config(models=models, roles=roles_all("big"), env={"ANTHROPIC_API_KEY": "personal"})
    with pytest.raises(ModelError) as caught:
        await make(config).client_for("main")
    message = str(caught.value)
    assert "which needs WORK_KEY" in message
    assert "ANTHROPIC_API_KEY" not in message


async def test_a_missing_openai_compatible_key_names_the_models_own_variable(make):
    # The adapter's message names the variable it is told to, and the router tells
    # it the one this model's config names. Telling an OpenRouter user to set
    # OPENAI_API_KEY would send them to the wrong place; the other key being set
    # must not satisfy this model either.
    models = {
        "or": ModelConfig(
            "openai_compat",
            "deepseek/deepseek-v3",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        )
    }
    config = Config(models=models, roles=roles_all("or"), env={"OPENAI_API_KEY": "someone-elses"})
    with pytest.raises(CredentialsError) as caught:
        await make(config).client_for("main")
    assert "set OPENROUTER_API_KEY or run: ncc init" in str(caught.value)
    assert "OPENAI_API_KEY" not in str(caught.value)
    assert caught.value.retryable is False


async def test_a_missing_openai_compatible_key_names_the_default_variable_when_none_is_configured(
    make,
):
    models = {"o": ModelConfig("openai_compat", "gpt-5", base_url="https://api.openai.com/v1")}
    with pytest.raises(ModelError, match="set OPENAI_API_KEY or run: ncc init"):
        await make(Config(models=models, roles=roles_all("o"))).client_for("main")


async def test_a_local_model_needs_no_key(make):
    config = config_with(explore="small")
    config.env.clear()
    assert (await make(config).client_for("explore")).model_id == "qwen3-coder"


# --------------------------------------------------------------------------
# Refusing what cannot be routed
# --------------------------------------------------------------------------


async def test_a_role_pointing_at_an_undefined_model_is_refused_when_the_router_is_built(make):
    # What a --role flag does after the file was loaded: replace an alias, with
    # nothing having checked it. The run should end before the first request.
    config = config_with()
    broken = dataclasses.replace(config, roles=dataclasses.replace(config.roles, explore="typo"))
    with pytest.raises(ConfigError, match="role 'explore' points at model 'typo'"):
        make(broken)


async def test_a_hand_built_model_with_an_unknown_adapter_is_refused_not_sent_to_ollama(make):
    router = make(Config(models={"t": ModelConfig("telepathy", "x")}, roles=roles_all("t")))
    with pytest.raises(ConfigError, match="model 't' uses adapter 'telepathy'"):
        await router.client_for("main")


async def test_an_openai_compatible_model_built_by_hand_without_a_base_url_is_refused(make):
    models = {"o": ModelConfig("openai_compat", "gpt-5")}
    router = make(Config(models=models, roles=roles_all("o"), env={"OPENAI_API_KEY": "k"}))
    with pytest.raises(ConfigError, match="model 'o' uses openai_compat but has no base_url"):
        await router.client_for("main")


async def test_a_name_that_is_not_a_role_is_refused_everywhere_a_role_is_taken(make):
    router = make(config_with())
    with pytest.raises(ValueError, match="unknown role 'reviewer'"):
        await router.client_for("reviewer")
    with pytest.raises(ValueError, match="unknown role 'reviewer'"):
        await router.capabilities_for("reviewer")
    with pytest.raises(ValueError, match="unknown role 'reviewer'"):
        router.record("reviewer", Usage(input_tokens=1), "ollama", "qwen3-coder")
    assert router.by_role() == {}  # and nothing was recorded under it


# --------------------------------------------------------------------------
# Clients handed in
# --------------------------------------------------------------------------


async def test_clients_handed_to_the_router_are_used_instead_of_building_adapters(make):
    config = config_with(explore="small")
    config.env.clear()  # a real Anthropic client could not be built without a key
    scripted = FakeClient("scripted")
    given = {"big": scripted}
    router = make(config, given)  # positional, as Router(config, cache, clients)
    assert await router.client_for("main") is scripted
    assert await router.client_for("title") is scripted
    assert isinstance(await router.client_for("explore"), OllamaClient)  # not handed in: built
    assert given == {"big": scripted}  # the caller's dictionary is not added to


async def test_a_router_can_be_subclassed_with_just_a_config_and_a_cache(tmp_path):
    scripted = FakeClient("scripted")

    class FixedRouter(Router):
        def __init__(self, config: Config, cache: CapabilityCache) -> None:
            super().__init__(config, cache)

        async def client_for(self, role: str) -> ModelClient:  # noqa: ARG002 - same signature
            return scripted

    router = FixedRouter(config_with(), CapabilityCache(tmp_path / "c.json"))
    assert await router.client_for("main") is scripted
    router.record("main", Usage(input_tokens=1_000_000), "anthropic", "claude-sonnet-5")
    assert router.total_cost() == pytest.approx(2.0)


# --------------------------------------------------------------------------
# What each role cost
# --------------------------------------------------------------------------


async def test_usage_accumulates_across_calls_within_a_role(make):
    router = make(config_with())
    router.record("main", Usage(100, 10, 5, 3), "anthropic", "claude-sonnet-5")
    router.record("main", Usage(50, 5, 2, 1), "anthropic", "claude-sonnet-5")
    assert router.by_role()["main"].usage == Usage(150, 15, 7, 4)
    # (150*2 + 15*10 + 7*0.20 + 4*2.50) / 1e6 dollars at sonnet-5's rates
    assert router.by_role()["main"].cost == pytest.approx(461.4e-6)


async def test_roles_are_listed_in_the_order_reports_show_them(make):
    router = make(config_with())
    for role in ("title", "compact", "main", "plan"):
        router.record(role, Usage(input_tokens=1), "ollama", "qwen3-coder")
    assert list(router.by_role()) == ["main", "plan", "compact", "title"]


async def test_a_role_reports_its_model_its_adapter_and_what_it_cost(make):
    router = make(config_with())
    router.record("main", Usage(input_tokens=1_000_000), "anthropic", "claude-sonnet-5")
    router.record("explore", Usage(input_tokens=1_000_000), "ollama", "qwen3-coder")
    router.record("plan", Usage(input_tokens=1_000_000), "openai_compat", "mystery")
    by_role = router.by_role()
    assert by_role["main"].adapter == "anthropic"
    assert by_role["main"].model == "claude-sonnet-5"
    assert by_role["main"].cost == pytest.approx(2.0)
    assert by_role["explore"].cost == 0.0  # local: known to be free
    assert by_role["plan"].cost is None  # no price: not known, and not shown as free


async def test_total_usage_is_the_sum_over_every_role(make):
    router = make(config_with())
    router.record("main", Usage(100, 10, 1, 2), "anthropic", "claude-sonnet-5")
    router.record("title", Usage(5, 6, 7, 8), "ollama", "qwen3-coder")
    assert router.total_usage() == Usage(105, 16, 8, 10)


async def test_the_total_is_the_sum_of_what_each_role_cost(make):
    router = make(config_with())
    router.record("main", Usage(input_tokens=1_000_000), "anthropic", "claude-sonnet-5")  # $2
    router.record("plan", Usage(output_tokens=1_000_000), "anthropic", "claude-opus-5")  # $25
    router.record("explore", Usage(input_tokens=9_000_000), "ollama", "qwen3-coder")  # free
    assert router.total_cost() == pytest.approx(27.0)


async def test_one_unpriced_role_makes_the_total_unknown_even_though_the_others_are_priced(make):
    # A partial sum presented as the total is the misleading answer: two of the
    # three roles cost $27, and the total must not say so.
    router = make(config_with())
    router.record("main", Usage(input_tokens=1_000_000), "anthropic", "claude-sonnet-5")
    router.record("plan", Usage(output_tokens=1_000_000), "anthropic", "claude-opus-5")
    router.record("verify", Usage(input_tokens=10), "openai_compat", "mystery")
    assert router.by_role()["main"].cost == pytest.approx(2.0)  # the known parts stay known
    assert router.total_cost() is None


async def test_nothing_recorded_costs_nothing_and_that_is_known(make):
    router = make(config_with())
    assert router.by_role() == {}
    assert router.total_usage() == Usage()
    assert router.total_cost() == 0.0


async def test_a_role_that_changes_model_is_priced_call_by_call(make):
    # The tokens spent on the first model stay priced at its rate: pricing the
    # whole total at the latest model's rate would report $10, not $7.
    router = make(config_with())
    router.record("main", Usage(input_tokens=1_000_000), "anthropic", "claude-sonnet-5")  # $2
    router.record("main", Usage(input_tokens=1_000_000), "anthropic", "claude-opus-5")  # $5
    entry = router.by_role()["main"]
    assert entry.cost == pytest.approx(7.0)
    assert entry.model == "claude-opus-5"  # the model in use now
    assert entry.usage.input_tokens == 2_000_000


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("openai_compat", "mystery"), ("anthropic", "claude-sonnet-5")),
        (("anthropic", "claude-sonnet-5"), ("openai_compat", "mystery")),
    ],
    ids=["unpriced then priced", "priced then unpriced"],
)
async def test_a_role_that_ever_used_an_unpriced_model_stays_unknown(make, first, second):
    router = make(config_with())
    router.record("main", Usage(input_tokens=1000), *first)
    router.record("main", Usage(input_tokens=1000), *second)
    assert router.by_role()["main"].cost is None
    assert router.total_cost() is None


async def test_a_supplied_price_book_is_the_one_used(make):
    book = PriceBook({("anthropic", "claude-sonnet-5"): Price(100.0, 0.0)}, dt.date(2026, 10, 2))
    router = make(config_with(), prices=book)
    router.record("main", Usage(input_tokens=1_000_000), "anthropic", "claude-sonnet-5")
    assert router.prices is book
    assert router.total_cost() == pytest.approx(100.0)


async def test_the_shipped_price_book_is_used_when_none_is_supplied(make):
    router = make(config_with())
    assert router.prices.price_for("anthropic", "claude-sonnet-5") == Price(2.0, 10.0, 0.20, 2.50)


def test_a_role_usage_is_a_hashable_snapshot():
    # Frozen with a Usage, two strings and a float or None: nothing in it can make
    # hash() raise, and frozen means a caller cannot edit the router's ledger
    # through the object by_role() hands back.
    entry = RoleUsage(Usage(1, 2, 3, 4), "claude-sonnet-5", "anthropic", 0.5)
    assert isinstance(entry, Hashable)
    assert hash(entry) == hash(RoleUsage(Usage(1, 2, 3, 4), "claude-sonnet-5", "anthropic", 0.5))
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.cost = 0.0  # type: ignore[misc]


# --------------------------------------------------------------------------
# What each model can do
# --------------------------------------------------------------------------


async def test_a_known_models_capabilities_come_from_the_table(make):
    caps = await make(config_with()).capabilities_for("main")
    assert caps.context_window == 1_000_000
    assert caps.max_output == 128_000
    assert caps.native_tools is True


async def test_what_the_config_says_beats_what_the_table_says(make):
    models = {
        "big": ModelConfig(
            "anthropic", "claude-sonnet-5", context_window=64_000, native_tools=False
        )
    }
    config = Config(models=models, roles=roles_all("big"), env={"ANTHROPIC_API_KEY": "k"})
    caps = await make(config).capabilities_for("main")
    assert caps.context_window == 64_000
    assert caps.native_tools is False
    # Not set in the config, but bounded by the window that was: a quarter of it.
    assert caps.max_output == 16_000


async def test_an_ollama_model_is_probed_once_at_its_own_server_and_remembered(make, monkeypatch):
    probes: list[tuple[str, str]] = []

    async def fake_probe(model: str, *, base_url: str) -> Capabilities:
        probes.append((model, base_url))
        return Capabilities(True, False, "none", 32_768, 4_096)

    monkeypatch.setattr("nanoclaude.agent.router.probe_ollama", fake_probe)
    models = {"l": ModelConfig("ollama", "qwen3-coder", base_url="http://gpu-box:11434")}
    config = Config(models=models, roles=roles_all("l"))
    router = make(config)
    first = await router.capabilities_for("main")
    again = await router.capabilities_for("explore")  # another role, the same model
    assert first is again
    assert first.context_window == 32_768
    assert probes == [("qwen3-coder", "http://gpu-box:11434")]
    await make(config).capabilities_for("main")  # a new session, the same cache file
    assert probes == [("qwen3-coder", "http://gpu-box:11434")]


async def test_a_probe_that_found_nothing_out_is_not_remembered_between_sessions(
    make, monkeypatch, tmp_path
):
    # Ollama not up yet: the session assumes the conservative default, once, and the
    # next session asks again. Nothing is written that would make "no native tools"
    # the permanent answer for a model that was merely unreachable.
    probes: list[str] = []

    async def unreachable(model: str, *, base_url: str) -> Capabilities | None:
        probes.append(model)
        return None

    monkeypatch.setattr("nanoclaude.agent.router.probe_ollama", unreachable)
    config = Config(models={"l": ModelConfig("ollama", "qwen3-coder")}, roles=roles_all("l"))
    router = make(config)
    first = await router.capabilities_for("main")
    assert first == CONSERVATIVE_DEFAULT
    assert await router.capabilities_for("explore") is first  # same session: not asked again
    assert probes == ["qwen3-coder"]
    assert not (tmp_path / "caps.json").exists()  # nothing was remembered on disk
    await make(config).capabilities_for("main")  # the next session
    assert probes == ["qwen3-coder", "qwen3-coder"]


async def test_what_the_config_declares_still_applies_when_the_probe_found_nothing_out(
    make, monkeypatch
):
    async def unreachable(model: str, *, base_url: str) -> Capabilities | None:
        return None

    monkeypatch.setattr("nanoclaude.agent.router.probe_ollama", unreachable)
    models = {"l": ModelConfig("ollama", "qwen3-coder", context_window=65_536, native_tools=True)}
    caps = await make(Config(models=models, roles=roles_all("l"))).capabilities_for("main")
    assert caps.context_window == 65_536
    assert caps.native_tools is True


async def test_an_ollama_model_with_no_server_configured_is_probed_at_the_local_default(
    make, monkeypatch
):
    probes: list[tuple[str, str]] = []

    async def fake_probe(model: str, *, base_url: str) -> Capabilities:
        probes.append((model, base_url))
        return CONSERVATIVE_DEFAULT

    monkeypatch.setattr("nanoclaude.agent.router.probe_ollama", fake_probe)
    config = Config(models={"l": ModelConfig("ollama", "qwen3-coder")}, roles=roles_all("l"))
    await make(config).capabilities_for("main")
    assert probes == [("qwen3-coder", "http://localhost:11434")]


async def test_a_model_the_table_does_not_know_is_not_probed_unless_it_is_an_ollama_model(
    make, monkeypatch
):
    async def refuse(model: str, *, base_url: str) -> Capabilities:
        raise AssertionError("only an Ollama server can be asked")

    monkeypatch.setattr("nanoclaude.agent.router.probe_ollama", refuse)
    models = {"o": ModelConfig("openai_compat", "mystery", base_url="https://x/v1")}
    caps = await make(Config(models=models, roles=roles_all("o"))).capabilities_for("main")
    assert caps == CONSERVATIVE_DEFAULT


# --------------------------------------------------------------------------
# Re-routing a role: what /model does, from the next request on
# --------------------------------------------------------------------------


async def test_a_rerouted_role_is_answered_by_the_new_models_client(make):
    router = make(config_with())
    assert (await router.client_for("main")).model_id == "claude-sonnet-5"
    await router.route("main", "small")
    assert (await router.client_for("main")).model_id == "qwen3-coder"


async def test_rerouting_one_role_leaves_the_others_where_they_were(make):
    router = make(config_with())
    await router.route("main", "small")
    assert {role: router.config.roles.alias_for(role) for role in ROLES} == {
        "main": "small",
        "explore": "big",
        "plan": "big",
        "verify": "big",
        "compact": "big",
        "title": "big",
    }


async def test_rerouting_to_an_alias_nobody_defined_names_the_ones_that_exist(make):
    router = make(config_with())
    before = router.config
    with pytest.raises(ConfigError, match="'nope'") as caught:
        await router.route("main", "nope")
    assert "big" in str(caught.value) and "small" in str(caught.value)
    assert router.config is before


async def test_rerouting_a_role_that_does_not_exist_is_a_programming_error(make):
    router = make(config_with())
    before = router.config
    with pytest.raises(ValueError, match="unknown role 'mian'"):
        await router.route("mian", "small")
    assert router.config is before


async def test_a_model_that_cannot_be_used_is_refused_at_once_and_the_old_route_stays(make):
    # The client is built by the command that asked for it, so a missing key is
    # reported there, with the model that was answering still answering, and not
    # on the next prompt.
    config = config_with()
    config.models["keyless"] = ModelConfig("anthropic", "claude-opus-5", api_key_env="OTHER_KEY")
    router = make(config)
    before = router.config
    with pytest.raises(ModelError, match="OTHER_KEY"):
        await router.route("main", "keyless")
    assert router.config is before
    assert (await router.client_for("main")).model_id == "claude-sonnet-5"


# --------------------------------------------------------------------------
# Closing
# --------------------------------------------------------------------------


async def test_closing_the_router_closes_every_client_it_holds(make):
    log: list[str] = []
    router = make(
        config_with(explore="small"),
        {"big": FakeClient("one", log=log), "small": FakeClient("two", log=log)},
    )
    await router.client_for("main")
    await router.aclose()
    assert sorted(log) == ["one", "two"]


async def test_a_client_with_nothing_to_close_is_skipped(make):
    log: list[str] = []
    router = make(
        config_with(explore="small"),
        {"big": ClientWithoutClose(), "small": FakeClient("two", log=log)},
    )
    await router.aclose()
    assert log == ["two"]


async def test_one_client_failing_to_close_does_not_leave_the_others_open(make):
    log: list[str] = []
    router = make(
        config_with(explore="small"),
        {
            "big": FakeClient("stubborn", log=log, fail_on_close=True),
            "small": FakeClient("fine", log=log),
        },
    )
    with pytest.raises(RuntimeError, match="stubborn would not close"):
        await router.aclose()
    assert sorted(log) == ["fine", "stubborn"]
