import dataclasses
import json
from pathlib import Path
from typing import Never

import httpx
import pytest

from nanoclaude.conversation.budget import Budget
from nanoclaude.providers.capabilities import (
    CONSERVATIVE_DEFAULT,
    Capabilities,
    CapabilityCache,
    _apply,
    capabilities_for,
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
    assert caps is not None
    assert caps.native_tools and caps.context_window == 32768


async def test_a_model_without_a_tools_template_falls_back_to_text_tools():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"template": "{{ .Prompt }}", "model_info": {}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        caps = await probe_ollama("llama3", base_url="http://x", client=http)
    assert caps is not None
    assert not caps.native_tools


async def test_a_probe_failure_reports_that_it_found_nothing_out():
    # None, not the conservative default: a caller that remembered the default would
    # remember "no native tools" for a model that was only unreachable just now.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        caps = await probe_ollama("m", base_url="http://x", client=http)
    assert caps is None


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
async def test_a_non_object_probe_response_reports_that_it_found_nothing_out(body):
    # Review focus #3: valid JSON need not be an object. A bare list, string,
    # number or null must not raise an AttributeError out of data.get(...).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        caps = await probe_ollama("m", base_url="http://x", client=http)
    assert caps is None


async def test_a_non_object_model_info_is_treated_as_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"template": "{{ .Tools }}", "model_info": ["not", "an", "object"]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        caps = await probe_ollama("m", base_url="http://x", client=http)
    assert caps is not None
    assert caps.native_tools is True
    assert caps.context_window == CONSERVATIVE_DEFAULT.context_window


@pytest.mark.parametrize(
    "failure",
    [httpx.ConnectError("connection refused"), httpx.ReadTimeout("timed out")],
    ids=["not running yet", "timeout"],
)
async def test_a_connection_failure_during_probe_reports_that_it_found_nothing_out(failure):
    # Do not assume ollama is running: probing it must degrade the same way
    # a hard probe failure does, not raise out of the adapter.
    def handler(request: httpx.Request) -> httpx.Response:
        raise failure

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        caps = await probe_ollama("m", base_url="http://x", client=http)
    assert caps is None


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

    assert caps is not None
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


def test_clearing_forgets_every_model_so_the_next_run_probes_again(tmp_path):
    cache = CapabilityCache(tmp_path / "caps.json")
    cache.put("ollama", "a", CONSERVATIVE_DEFAULT)
    cache.put("ollama", "b", CONSERVATIVE_DEFAULT)
    cache.clear()
    assert (cache.get("ollama", "a"), cache.get("ollama", "b")) == (None, None)
    assert not (tmp_path / "caps.json").exists()


def test_clearing_a_cache_that_was_never_written_is_not_an_error(tmp_path):
    CapabilityCache(tmp_path / "nowhere" / "caps.json").clear()


def test_a_cache_that_cannot_be_removed_says_so_and_does_not_pretend(tmp_path):
    # A directory where the file should be: unlink refuses, and the refusal is the
    # caller's to word. Swallowing it would leave the stale answers in force.
    (tmp_path / "caps.json").mkdir()
    with pytest.raises(OSError, match=r"caps\.json"):
        CapabilityCache(tmp_path / "caps.json").clear()


# -- resolve_capabilities: only a probe that found something out is remembered --


async def test_a_probe_that_found_nothing_out_is_not_remembered(tmp_path):
    cache = CapabilityCache(tmp_path / "caps.json")
    calls = {"n": 0}

    async def probe() -> Capabilities | None:
        calls["n"] += 1
        return None

    first = await resolve_capabilities("ollama", "m", cache=cache, probe=probe)
    assert first == CONSERVATIVE_DEFAULT  # what to assume for now
    assert cache.get("ollama", "m") is None
    assert not (tmp_path / "caps.json").exists()  # nothing was written at all
    await resolve_capabilities("ollama", "m", cache=cache, probe=probe)
    assert calls["n"] == 2, "the next session has to ask again"


async def test_what_the_config_says_still_applies_when_the_probe_found_nothing_out(tmp_path):
    async def probe() -> Capabilities | None:
        return None

    caps = await resolve_capabilities(
        "ollama",
        "m",
        cache=CapabilityCache(tmp_path / "caps.json"),
        probe=probe,
        overrides={"context_window": 99_999, "native_tools": True},
    )
    assert caps.context_window == 99_999
    assert caps.native_tools is True
    assert caps.parallel_tools is CONSERVATIVE_DEFAULT.parallel_tools  # the rest: the default


async def test_a_probe_that_succeeded_is_remembered_with_what_it_found(tmp_path):
    cache = CapabilityCache(tmp_path / "caps.json")
    found = Capabilities(True, False, "none", 32_768, 4_096)
    calls = {"n": 0}

    async def probe() -> Capabilities | None:
        calls["n"] += 1
        return found

    assert await resolve_capabilities("ollama", "m", cache=cache, probe=probe) == found
    assert cache.get("ollama", "m") == found
    assert await resolve_capabilities("ollama", "m", cache=cache, probe=probe) == found
    assert calls["n"] == 1


async def test_a_successful_probe_that_matches_the_conservative_default_is_still_remembered(
    tmp_path,
):
    # A model with no tools support and a small window is a real answer that happens
    # to equal the default. Failure is signalled, not inferred from the value, so it
    # is remembered like any other result.
    cache = CapabilityCache(tmp_path / "caps.json")
    same_values = dataclasses.replace(CONSERVATIVE_DEFAULT)
    assert same_values == CONSERVATIVE_DEFAULT and same_values is not CONSERVATIVE_DEFAULT
    calls = {"n": 0}

    async def probe() -> Capabilities | None:
        calls["n"] += 1
        return same_values

    await resolve_capabilities("ollama", "m", cache=cache, probe=probe)
    await resolve_capabilities("ollama", "m", cache=cache, probe=probe)
    assert calls["n"] == 1, "a successful answer was treated as a failed one"


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


def test_apply_overrides_the_given_fields_and_a_cut_window_bounds_the_output():
    result = _apply(CONSERVATIVE_DEFAULT, {"context_window": 4096})
    assert result.context_window == 4096
    assert result.native_tools == CONSERVATIVE_DEFAULT.native_tools
    assert result.max_output == 1_024  # a quarter of the cut window


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


# -- a window set in the config bounds the output too -----------------------


async def test_a_window_cut_in_the_config_cuts_the_output_to_a_quarter_of_it(tmp_path):
    cache = CapabilityCache(tmp_path / "caps.json")
    caps = await resolve_capabilities(
        "anthropic", "claude-opus-5-5", cache=cache, overrides={"context_window": 200_000}
    )
    assert (caps.context_window, caps.max_output) == (200_000, 50_000)


async def test_a_window_raised_in_the_config_leaves_the_output_alone(tmp_path):
    cache = CapabilityCache(tmp_path / "caps.json")
    caps = await resolve_capabilities(
        "anthropic", "claude-haiku-4-5", cache=cache, overrides={"context_window": 1_000_000}
    )
    assert (caps.context_window, caps.max_output) == (1_000_000, 64_000)


async def test_no_window_in_the_config_leaves_the_output_alone(tmp_path):
    # Haiku 4.5's own output is more than a quarter of its own window, so a cut
    # applied whenever anything is overridden would show here.
    cache = CapabilityCache(tmp_path / "caps.json")
    caps = await resolve_capabilities(
        "anthropic", "claude-haiku-4-5", cache=cache, overrides={"native_tools": False}
    )
    assert caps.max_output == 64_000


@pytest.mark.parametrize("window", [8_192, 64_000, 200_000])
async def test_any_window_set_in_the_config_leaves_room_for_the_conversation(tmp_path, window):
    cache = CapabilityCache(tmp_path / "caps.json")
    caps = await resolve_capabilities(
        "anthropic", "claude-opus-5-5", cache=cache, overrides={"context_window": window}
    )
    assert Budget(caps).available() >= window // 2


async def test_a_window_restated_in_the_config_leaves_the_output_alone(tmp_path):
    # Haiku 4.5's output is more than a quarter of its window; restating that
    # window is not a cut.
    cache = CapabilityCache(tmp_path / "caps.json")
    caps = await resolve_capabilities(
        "anthropic", "claude-haiku-4-5", cache=cache, overrides={"context_window": 200_000}
    )
    assert (caps.context_window, caps.max_output) == (200_000, 64_000)


def test_an_output_given_with_a_cut_window_is_kept():
    caps = capabilities_for("anthropic", "claude-opus-5-5")
    result = _apply(caps, {"context_window": 64_000, "max_output": 32_000})
    assert (result.context_window, result.max_output) == (64_000, 32_000)


async def test_the_cache_keeps_what_the_probe_found_not_what_the_config_made_of_it(tmp_path):
    cache = CapabilityCache(tmp_path / "caps.json")
    found = Capabilities(True, False, "none", 131_072, 4_096)

    async def probe() -> Capabilities:
        return found

    caps = await resolve_capabilities(
        "ollama", "m", cache=cache, probe=probe, overrides={"context_window": 8_192}
    )
    assert (caps.context_window, caps.max_output) == (8_192, 2_048)
    assert cache.get("ollama", "m") == found


def test_a_new_cache_makes_its_directory_and_file_private_from_the_start(
    tmp_path, private_from_the_start
):
    from tests.conftest import mode_of

    path = tmp_path / ".nanoclaude" / "capabilities.json"
    CapabilityCache(path).put("ollama", "m", CONSERVATIVE_DEFAULT)
    assert (mode_of(path.parent), mode_of(path)) == (0o700, 0o600)
    assert CapabilityCache(path).get("ollama", "m") == CONSERVATIVE_DEFAULT


def test_a_cache_that_was_open_to_others_is_private_once_it_is_written_again(tmp_path):

    from tests.conftest import mode_of

    path = tmp_path / "capabilities.json"
    path.write_text("{}")
    path.chmod(0o644)
    CapabilityCache(path).put("ollama", "m", CONSERVATIVE_DEFAULT)
    assert mode_of(path) == 0o600


def test_a_cache_is_written_without_a_temporary_file_left_behind(tmp_path):
    CapabilityCache(tmp_path / "capabilities.json").put("ollama", "m", CONSERVATIVE_DEFAULT)
    assert [p.name for p in tmp_path.iterdir()] == ["capabilities.json"]
