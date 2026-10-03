"""Tests for the scripted client and the session builder in nanoclaude.testing.session.

build_session is exercised by every test in tests/agent/test_session.py. These
cover what only a test double can get wrong: the script it plays, and the
promises its docstring makes about what a session built here will and will not do.
"""

import pytest

from nanoclaude.agent.ui import AutoApprove
from nanoclaude.conversation.transcript import TextBlock, Transcript, user_text
from nanoclaude.providers.base import ModelClient, ModelError, ModelRequest
from nanoclaude.providers.capabilities import Capabilities
from nanoclaude.testing.scripted import says
from nanoclaude.testing.session import SCRIPTED_MODEL, ScriptedClient, build_session

# Checked by mypy --strict, as test_scripted.py does for ScriptedModel: a client
# that stopped matching the protocol would otherwise only fail inside a session.
_conforms: ModelClient = ScriptedClient([])


def request() -> ModelRequest:
    return ModelRequest("sys", Transcript((user_text("hi"),)), (), 1024)


async def test_a_scripted_client_serves_replies_and_raises_errors_in_the_order_written():
    client = ScriptedClient(
        [says("one"), ModelError("overloaded", retryable=True, status=529), says("two")]
    )
    first = (await client.complete(request())).blocks[0]
    with pytest.raises(ModelError, match="overloaded") as caught:
        await client.complete(request())
    second = (await client.complete(request())).blocks[0]
    assert isinstance(first, TextBlock) and first.text == "one"
    assert isinstance(second, TextBlock) and second.text == "two"
    # The error is the very object the script held, with its fields intact.
    assert caught.value.retryable and caught.value.status == 529
    # All three requests were recorded, the one that raised included.
    assert len(client.requests) == 3
    assert client.exhausted


async def test_a_script_that_runs_out_fails_by_name_after_the_request_is_recorded():
    client = ScriptedClient([says("only")])
    await client.complete(request())
    assert client.exhausted
    with pytest.raises(ModelError, match="ran out of replies after 2 requests"):
        await client.complete(request())
    assert len(client.requests) == 2


async def test_a_scripted_client_says_when_it_has_been_closed():
    client = ScriptedClient([])
    assert client.closed is False
    await client.aclose()
    assert client.closed is True


async def test_a_script_that_is_not_empty_is_not_exhausted():
    assert not ScriptedClient([says("x")]).exhausted


def test_capabilities_given_to_the_builder_route_the_role_to_a_model_the_table_does_not_know(
    tmp_repo,
):
    capabilities = Capabilities(True, False, "none", context_window=32_000, max_output=4_000)
    session = build_session(tmp_repo, [], capabilities=capabilities)
    assert session.router.config.models["m"].model == SCRIPTED_MODEL
    # An explicit model id wins, so a test can name the model it wants unpriced.
    named = build_session(tmp_repo, [], model="my-local-model", capabilities=capabilities)
    assert named.router.config.models["m"].model == "my-local-model"


async def test_a_session_built_here_retries_with_no_delay_unless_told_otherwise(tmp_repo):
    delays: list[float] = []

    class Recording(AutoApprove):
        def on_retry(self, _attempt: int, delay_s: float, _reason: str) -> None:
            delays.append(delay_s)

    session = build_session(
        tmp_repo,
        [ModelError("overloaded", retryable=True, status=529), says("ok")],
        ui=Recording(),
    )
    await session.run("hello")
    # The default policy would wait between half a second and a second here.
    assert delays == [0.0]


def test_a_session_built_here_keeps_its_state_and_home_inside_the_directory_it_was_given(
    tmp_repo,
):
    session = build_session(tmp_repo, [])
    assert session.store is not None
    assert session.store.path.parent == tmp_repo / ".nanoclaude"
    # Global instructions are looked for here, never in the real home directory.
    assert session.home == str(tmp_repo / ".nanoclaude" / "home")
