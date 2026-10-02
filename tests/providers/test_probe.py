import json
from pathlib import Path
from typing import Never

import httpx
import pytest

from nanoclaude.providers.capabilities import (
    CONSERVATIVE_DEFAULT,
    Capabilities,
    CapabilityCache,
    _apply,
    resolve_capabilities,
)
from nanoclaude.providers.ollama import probe_ollama

# -- probe_ollama: the brief's own scenarios ---------------------------------


async def test_a_model_whose_template_declares_tools_gets_native_tools():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "template": "{{ if .Tools }}...{{ end }}",
                "model_info": {"general.context_length": 32768},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        caps = await probe_ollama("qwen3-coder", base_url="http://x", client=http)
    assert caps.native_tools and caps.context_window == 32768


async def test_a_model_without_a_tools_template_falls_back_to_text_tools():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"template": "{{ .Prompt }}", "model_info": {}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        caps = await probe_ollama("llama3", base_url="http://x", client=http)
    assert not caps.native_tools


async def test_a_probe_failure_yields_the_conservative_default():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        caps = await probe_ollama("m", base_url="http://x", client=http)
    assert caps == CONSERVATIVE_DEFAULT


async def test_probe_results_are_cached_on_disk(tmp_path):
    cache = CapabilityCache(tmp_path / "caps.json")
    calls = {"n": 0}

    async def probe() -> Capabilities:
        calls["n"] += 1
        return CONSERVATIVE_DEFAULT

    await resolve_capabilities("ollama", "m", cache=cache, probe=probe)
    await resolve_capabilities("ollama", "m", cache=cache, probe=probe)
    assert calls["n"] == 1, "the second resolve should have used the cache"


async def test_a_known_model_is_never_probed(tmp_path):
    cache = CapabilityCache(tmp_path / "caps.json")

    async def probe() -> Never:
        raise AssertionError("should not be called for a known model")

    caps = await resolve_capabilities("anthropic", "claude-sonnet-5", cache=cache, probe=probe)
    assert caps.native_tools


# -- probe_ollama: lessons from Tasks 17/18, applied here too ----------------


@pytest.mark.parametrize("body", [["unexpected"], "just a string", 42, None])
async def test_a_non_object_probe_response_yields_the_conservative_default(body):
    # Review focus #3: valid JSON need not be an object. A bare list, string,
    # number or null must not raise an AttributeError out of data.get(...).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        caps = await probe_ollama("m", base_url="http://x", client=http)
    assert caps == CONSERVATIVE_DEFAULT


async def test_a_non_object_model_info_is_treated_as_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"template": "{{ .Tools }}", "model_info": ["not", "an", "object"]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        caps = await probe_ollama("m", base_url="http://x", client=http)
    assert caps.native_tools is True
    assert caps.context_window == CONSERVATIVE_DEFAULT.context_window


async def test_a_connection_failure_during_probe_yields_the_conservative_default():
    # Do not assume ollama is running: probing it must degrade the same way
    # a hard probe failure does, not raise out of the adapter.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        caps = await probe_ollama("m", base_url="http://x", client=http)
    assert caps == CONSERVATIVE_DEFAULT


async def test_probe_creates_and_closes_its_own_client_when_none_is_given(monkeypatch):
    # The other half of "owns = client is None": every test above supplies a
    # client, so without this the constructor branch and its matching aclose()
    # in the `finally` would never run. No real network call is made --
    # httpx.AsyncClient itself is monkeypatched to hand back a MockTransport-
    # backed instance instead of constructing a real one.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"template": "{{ .Tools }}", "model_info": {"llama.context_length": 4096}},
        )

    owned_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    closed = {"value": False}
    original_aclose = owned_client.aclose

    async def tracking_aclose() -> None:
        closed["value"] = True
        await original_aclose()

    monkeypatch.setattr(owned_client, "aclose", tracking_aclose)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *_args, **_kwargs: owned_client)

    caps = await probe_ollama("m", base_url="http://x")

    assert caps.native_tools is True
    assert caps.context_window == 4096
    assert closed["value"] is True


# -- CapabilityCache ----------------------------------------------------------
# Every test here uses tmp_path: the cache must never read or write the real
# per-user location, and CapabilityCache takes its path explicitly rather than
# defaulting to one, so there is nothing to monkeypatch.


def test_cache_round_trips_a_value(tmp_path):
    cache = CapabilityCache(tmp_path / "caps.json")
    caps = Capabilities(True, False, "automatic", 16_384, 4_096, "opaque", True)
    cache.put("ollama", "m", caps)
    assert cache.get("ollama", "m") == caps


def test_cache_get_on_an_unknown_key_returns_none(tmp_path):
    cache = CapabilityCache(tmp_path / "caps.json")
    cache.put("ollama", "m", CONSERVATIVE_DEFAULT)
    assert cache.get("ollama", "other-model") is None


def test_cache_handles_a_missing_file(tmp_path):
    cache = CapabilityCache(tmp_path / "nope" / "caps.json")
    assert cache.get("ollama", "m") is None


def test_cache_handles_a_file_that_is_not_valid_json(tmp_path):
    path = tmp_path / "caps.json"
    path.write_text("{not json")
    cache = CapabilityCache(path)
    assert cache.get("ollama", "m") is None


def test_cache_handles_a_file_whose_top_level_is_not_an_object(tmp_path):
    path = tmp_path / "caps.json"
    path.write_text("[1, 2, 3]")
    cache = CapabilityCache(path)
    assert cache.get("ollama", "m") is None


def test_cache_handles_an_entry_that_is_not_an_object(tmp_path):
    path = tmp_path / "caps.json"
    path.write_text('{"ollama/m": "not an object"}')
    cache = CapabilityCache(path)
    assert cache.get("ollama", "m") is None


def test_cache_handles_an_entry_missing_required_fields(tmp_path):
    path = tmp_path / "caps.json"
    path.write_text('{"ollama/m": {"native_tools": true}}')
    cache = CapabilityCache(path)
    assert cache.get("ollama", "m") is None


def test_cache_handles_an_entry_with_an_unexpected_extra_field(tmp_path):
    path = tmp_path / "caps.json"
    data = {
        "native_tools": False,
        "parallel_tools": False,
        "cache": "none",
        "context_window": 8192,
        "max_output": 2048,
        "reasoning": "none",
        "vision": False,
        "from_a_future_version": True,
    }
    path.write_text(json.dumps({"ollama/m": data}))
    cache = CapabilityCache(path)
    assert cache.get("ollama", "m") is None


def test_put_creates_parent_directories(tmp_path):
    cache = CapabilityCache(tmp_path / "nested" / "dir" / "caps.json")
    cache.put("ollama", "m", CONSERVATIVE_DEFAULT)
    assert cache.get("ollama", "m") == CONSERVATIVE_DEFAULT


def test_put_writes_through_a_temp_file_in_the_same_directory_then_replaces(tmp_path, monkeypatch):
    path = tmp_path / "caps.json"
    cache = CapabilityCache(path)
    recorded: list[tuple[Path, Path]] = []
    real_replace = Path.replace

    def spy_replace(self: Path, target: Path) -> Path:
        recorded.append((self, Path(target)))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", spy_replace, raising=True)

    cache.put("ollama", "m", CONSERVATIVE_DEFAULT)

    assert len(recorded) == 1
    tmp_used, final_target = recorded[0]
    assert tmp_used != path
    assert tmp_used.parent == path.parent
    assert final_target == path
    assert path.exists()
    assert not tmp_used.exists()  # renamed away by replace(), not left behind


def test_put_overwrites_an_existing_cache_file(tmp_path):
    cache = CapabilityCache(tmp_path / "caps.json")
    cache.put("ollama", "m", CONSERVATIVE_DEFAULT)
    second = Capabilities(True, True, "explicit", 100_000, 8_000)
    cache.put("ollama", "m", second)
    assert cache.get("ollama", "m") == second


# -- resolve_capabilities: precedence and overrides --------------------------


async def test_the_table_wins_over_a_conflicting_cached_value(tmp_path):
    # Table beats not just a fresh probe (the brief's own test) but a cache
    # entry already sitting there for the same key.
    cache = CapabilityCache(tmp_path / "caps.json")
    cache.put("anthropic", "claude-sonnet-5", CONSERVATIVE_DEFAULT)

    async def probe() -> Never:
        raise AssertionError("should not be called for a known model")

    caps = await resolve_capabilities("anthropic", "claude-sonnet-5", cache=cache, probe=probe)
    assert caps.native_tools  # the table's real value, not the stub just cached


async def test_no_probe_and_no_cache_entry_yields_the_conservative_default(tmp_path):
    # The other half of "if probe is None:" -- the brief's own caching test
    # always supplies a probe.
    cache = CapabilityCache(tmp_path / "caps.json")
    caps = await resolve_capabilities("ollama", "unknown-model", cache=cache)
    assert caps == CONSERVATIVE_DEFAULT


async def test_an_override_wins_over_a_fresh_probe(tmp_path):
    cache = CapabilityCache(tmp_path / "caps.json")

    async def probe() -> Capabilities:
        return CONSERVATIVE_DEFAULT

    caps = await resolve_capabilities(
        "ollama", "m", cache=cache, probe=probe, overrides={"context_window": 99_999}
    )
    assert caps.context_window == 99_999


def test_apply_with_no_overrides_returns_the_value_unchanged():
    assert _apply(CONSERVATIVE_DEFAULT, None) == CONSERVATIVE_DEFAULT
    assert _apply(CONSERVATIVE_DEFAULT, {}) == CONSERVATIVE_DEFAULT


def test_apply_overrides_only_the_given_fields():
    result = _apply(CONSERVATIVE_DEFAULT, {"context_window": 4096})
    assert result.context_window == 4096
    assert result.native_tools == CONSERVATIVE_DEFAULT.native_tools


def test_apply_ignores_none_valued_overrides():
    # A config key present but unset (None) must not blow away a real
    # discovered value.
    result = _apply(CONSERVATIVE_DEFAULT, {"native_tools": None})
    assert result.native_tools == CONSERVATIVE_DEFAULT.native_tools


async def test_the_probe_tolerates_a_trailing_slash_on_base_url():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"template": "", "model_info": {}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await probe_ollama("m", base_url="http://x/", client=http)
    assert seen == ["http://x/api/show"]


async def test_a_programming_error_in_the_probe_is_not_disguised_as_a_capability(monkeypatch):
    # Only the named failure types mean "we do not know". Anything else is a
    # bug, and must surface rather than make every model look tool-less.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"template": "", "model_info": {}})

    def broken(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError("bug")

    monkeypatch.setattr("nanoclaude.providers.ollama.Capabilities", broken)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(RuntimeError, match="bug"):
            await probe_ollama("m", base_url="http://x", client=http)


async def test_an_unexpected_error_inside_the_probe_request_is_not_swallowed(monkeypatch):
    # Distinct from the test above: this one raises inside the try block, so
    # it pins that the except clause names its failure types rather than
    # catching Exception.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"template": "", "model_info": {}})

    def broken_json(_self: httpx.Response) -> Never:
        raise RuntimeError("bug while parsing")

    monkeypatch.setattr(httpx.Response, "json", broken_json)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(RuntimeError, match="bug while parsing"):
            await probe_ollama("m", base_url="http://x", client=http)
