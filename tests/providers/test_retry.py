"""Tests for the shared retry policy: spec section 7.4's table, exhaustively,
plus the pieces the table alone does not reach -- message extraction for each
distinct error branch, and the status/no-status split in the attempt limit."""

from collections.abc import Hashable
from typing import Never

import httpx
import pytest

from nanoclaude.conversation.transcript import TextBlock
from nanoclaude.providers.base import ModelError, ModelReply, StopKind, Usage
from nanoclaude.providers.retry import (
    RetryPolicy,
    classify_status,
    classify_transport,
    connection_lost,
    with_retry,
)


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (408, True),
        (425, True),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (529, True),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (422, False),
    ],
)
def test_the_retryable_set_matches_the_spec_table(status, retryable):
    assert classify_status(status, "body").retryable is retryable


def test_a_credentials_error_points_at_the_fix():
    error = classify_status(401, '{"error":{"message":"invalid x-api-key"}}')
    assert "ncc init" in str(error)
    assert not error.retryable


def test_a_bad_request_passes_the_providers_own_message_through():
    """A 400 usually means our adapter is wrong, so the detail has to be visible."""
    error = classify_status(400, '{"error":{"message":"tools.0.name: invalid"}}')
    assert "tools.0.name: invalid" in str(error)


def test_classify_status_names_the_missing_model_or_endpoint_on_404():
    # The parametrize table above only pins this status's retryable=False; the
    # 404 branch is a distinct message from both the credentials branch
    # (401/403) and the generic-rejection branch (400/422), and needs its own
    # check that it is the right message, not just a non-retryable one.
    error = classify_status(404, '{"error":{"message":"model not found"}}')
    assert "does not know that model or endpoint" in str(error)
    assert "model not found" in str(error)
    assert not error.retryable


def test_classify_status_says_the_provider_is_busy_when_retryable():
    error = classify_status(529, '{"error":{"message":"overloaded_error"}}')
    assert "busy" in str(error)
    assert error.retryable
    assert error.status == 529


def test_message_from_falls_back_to_the_raw_body_when_json_has_no_error_key():
    # payload.get("error") is None here, the falsy half of ``error or body`` --
    # the dict-shaped tests above only exercise the truthy, dict-shaped half.
    error = classify_status(500, "{}")
    assert str(error).endswith(": {}")


def test_message_from_uses_a_plain_string_error_value_when_present():
    # "error" decodes but is not a dict: isinstance(error, dict) is False, and
    # error itself (not body) is truthy -- the other half of that same "or",
    # not reached by the falsy case directly above.
    error = classify_status(500, '{"error": "service unavailable"}')
    assert "service unavailable" in str(error)


async def test_a_retryable_failure_is_retried_then_succeeds():
    attempts = {"n": 0}

    async def operation() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ModelError("overloaded", retryable=True, status=529)
        return "ok"

    seen: list[tuple[int, float]] = []
    result = await with_retry(
        operation,
        policy=RetryPolicy(),
        on_retry=lambda attempt, delay, _reason: seen.append((attempt, delay)),
        sleep=_no_sleep,
        jitter=lambda: 1.0,
    )
    assert result == "ok"
    assert [a for a, _ in seen] == [1, 2]
    assert seen[0][1] < seen[1][1]  # backoff grows


async def test_retries_stop_at_the_limit_and_raise_the_last_error():
    async def always_overloaded() -> Never:
        raise ModelError("overloaded", retryable=True, status=529)

    with pytest.raises(ModelError, match="overloaded"):
        await with_retry(
            always_overloaded,
            policy=RetryPolicy(overload_attempts=2),
            on_retry=lambda *_args: None,
            sleep=_no_sleep,
            jitter=lambda: 1.0,
        )


async def test_a_non_retryable_failure_is_raised_immediately():
    calls = {"n": 0}

    async def unauthorised() -> Never:
        calls["n"] += 1
        raise ModelError("bad key", retryable=False, status=401)

    with pytest.raises(ModelError):
        await with_retry(
            unauthorised,
            policy=RetryPolicy(),
            on_retry=lambda *_args: None,
            sleep=_no_sleep,
            jitter=lambda: 1.0,
        )
    assert calls["n"] == 1


async def test_the_delay_is_capped():
    delays: list[float] = []

    async def always() -> Never:
        raise ModelError("x", retryable=True, status=529)

    with pytest.raises(ModelError):
        await with_retry(
            always,
            policy=RetryPolicy(overload_attempts=10, max_delay_s=4.0),
            on_retry=lambda _attempt, d, _reason: delays.append(d),
            sleep=_no_sleep,
            jitter=lambda: 1.0,
        )
    assert max(delays) <= 4.0


async def test_a_network_error_without_a_status_uses_the_network_attempts_limit():
    # error.status is None for a transport-level failure -- anthropic.py's own
    # httpx.TimeoutException/TransportError handlers both raise ModelError
    # with no status= at all. That must select network_attempts, not
    # overload_attempts: the other half of with_retry's "if error.status else"
    # choice, untouched by every test above (all of which use status=529).
    attempts = {"n": 0}

    async def flaky() -> Never:
        attempts["n"] += 1
        raise ModelError("could not reach the provider", retryable=True, status=None)

    with pytest.raises(ModelError, match="could not reach"):
        await with_retry(
            flaky,
            policy=RetryPolicy(network_attempts=2, overload_attempts=5),
            on_retry=lambda *_args: None,
            sleep=_no_sleep,
            jitter=lambda: 1.0,
        )
    # Exactly network_attempts calls, not overload_attempts -- if the limit
    # picked were the wrong one this would be 5, not 2.
    assert attempts["n"] == 2


def test_retry_policy_is_hashable():
    # Every field is an int or a float, so -- unlike ToolSpec, ModelRequest,
    # ModelReply, Message and Transcript, each declared unhashable in their
    # own modules for holding decoded-JSON data -- this frozen dataclass has
    # no way to end up holding anything unhashable. Pinned explicitly, as a
    # decision, rather than left as whatever the default happens to produce.
    policy = RetryPolicy()
    assert isinstance(policy, Hashable)
    assert hash(RetryPolicy()) == hash(policy)


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.mark.parametrize("body", ['"just a string"', "[1, 2, 3]", "42", "null"])
def test_an_error_body_that_is_valid_json_but_not_an_object_still_classifies(body):
    error = classify_status(500, body)
    assert body[:20] in str(error)


def test_a_rejected_key_message_joins_what_and_what_to_do_with_an_em_dash():
    # Spec 17.9's user-facing form is "<what> \u2014 <what to do>".
    assert "\u2014 check the key" in str(classify_status(401, "{}"))


async def test_the_smallest_jitter_halves_the_delay_and_never_goes_below_it():
    delays: list[float] = []

    async def always() -> Never:
        raise ModelError("x", retryable=True, status=529)

    with pytest.raises(ModelError):
        await with_retry(
            always,
            policy=RetryPolicy(overload_attempts=3, base_delay_s=1.0, max_delay_s=30.0),
            on_retry=lambda _attempt, d, _reason: delays.append(d),
            sleep=_no_sleep,
            jitter=lambda: 0.0,
        )
    full = []
    with pytest.raises(ModelError):
        await with_retry(
            always,
            policy=RetryPolicy(overload_attempts=3, base_delay_s=1.0, max_delay_s=30.0),
            on_retry=lambda _attempt, d, _reason: full.append(d),
            sleep=_no_sleep,
            jitter=lambda: 1.0,
        )
    assert delays and all(
        low == pytest.approx(high / 2) for low, high in zip(delays, full, strict=True)
    )


# --- A request that does not fit the model's window ---------------------------
#
# Spec 7.4: a context overflow reported by the provider is answered by one forced
# compaction and one retry. For the session to do that, the adapters' errors have
# to say it is an overflow, and they all get that from classify_status. One test per
# marker: a table where only some rows are exercised leaves the others decorative.

OVERFLOW_BODIES = {
    # Anthropic, a conversation longer than the window.
    "prompt is too long": (
        '{"type":"error","error":{"type":"invalid_request_error",'
        '"message":"prompt is too long: 213456 tokens > 200000 maximum"}}'
    ),
    # OpenAI's error code, which comes with a message of its own wording.
    "context_length_exceeded": (
        '{"error":{"message":"Your input is over the limit for this model.",'
        '"type":"invalid_request_error","param":"messages","code":"context_length_exceeded"}}'
    ),
    # OpenAI's message, and vLLM's.
    "maximum context length": (
        '{"object":"error","message":"This model\'s maximum context length is 4096 tokens. '
        'However, you requested 5000 tokens.","type":"BadRequestError","code":400}'
    ),
    # Anthropic again, when the input and the output allowance together do not fit.
    "exceed context limit": (
        '{"type":"error","error":{"type":"invalid_request_error","message":"input length '
        "and `max_tokens` exceed context limit: 188240 + 21333 > 200000, decrease input "
        'length or `max_tokens` and try again"}}'
    ),
    # llama.cpp's server.
    "exceeds the available context size": (
        '{"error":{"code":400,"message":"the request exceeds the available context size, '
        'try increasing it","type":"exceed_context_size_error"}}'
    ),
}


MARKER_IDS = [marker.replace(" ", "-") for marker in OVERFLOW_BODIES]


@pytest.mark.parametrize("marker", list(OVERFLOW_BODIES), ids=MARKER_IDS)
def test_a_400_that_reports_the_context_window_is_flagged_as_an_overflow(marker):
    body = OVERFLOW_BODIES[marker]
    # The row under test is the only marker this body contains.
    assert [m for m in OVERFLOW_BODIES if m in body.lower()] == [marker]
    error = classify_status(400, body)
    assert error.context_overflow is True
    # Still the same error to everything that does not care: not retried as it
    # stands, reported with the provider's own words.
    assert error.retryable is False and error.status == 400
    assert str(error).startswith("the provider rejected the request (HTTP 400): ")


@pytest.mark.parametrize("marker", list(OVERFLOW_BODIES), ids=MARKER_IDS)
def test_the_markers_are_matched_whatever_the_case(marker):
    assert classify_status(400, OVERFLOW_BODIES[marker].upper()).context_overflow is True


@pytest.mark.parametrize(
    "body",
    [
        '{"error":{"message":"tools.0.name: invalid"}}',
        '{"error":{"message":"max_tokens must be at least 1"}}',
        "the context was fine",  # "context", but nothing about a window
        "",
    ],
    ids=["bad-schema", "bad-parameter", "mentions-context", "empty-body"],
)
def test_a_400_that_is_not_about_the_window_is_a_plain_rejection(body):
    error = classify_status(400, body)
    assert error.context_overflow is False
    assert error.retryable is False


@pytest.mark.parametrize("status", [401, 404, 413, 429, 500, 529])
def test_only_a_400_is_taken_for_an_overflow(status):
    # The same words under another status are not what the session knows how to fix.
    assert classify_status(status, OVERFLOW_BODIES["prompt is too long"]).context_overflow is False


def test_the_marker_is_found_in_the_body_even_past_the_part_of_it_that_is_quoted():
    # The message quoted back to the person is cut at 500 characters. What decides
    # whether it was an overflow is the whole body.
    body = '{"error":{"message":"' + "x" * 600 + ' prompt is too long"}}'
    error = classify_status(400, body)
    assert error.context_overflow is True
    assert "prompt is too long" not in str(error)


def test_an_error_is_not_an_overflow_unless_it_says_so():
    assert ModelError("boom").context_overflow is False
    assert ModelError("boom", context_overflow=True).context_overflow is True


# -- a connection that fails (spec 7.4) ----------------------------------------------


def test_a_connection_lost_after_part_of_the_reply_arrived_is_not_retried_and_keeps_it():
    arrived = ModelReply((TextBlock("half"),), StopKind.CUT_OFF, Usage(5, 0), "m")
    error = classify_transport(httpx.ReadError("connection reset by peer"), arrived)
    assert error.retryable is False
    assert error.partial is arrived
    assert str(error) == "the connection to the provider was lost: connection reset by peer"


@pytest.mark.parametrize(
    ("lost", "said"),
    [
        (httpx.ReadTimeout("no bytes for 600 seconds"), "the provider timed out: no bytes"),
        (httpx.ConnectError("connection refused"), "could not reach the provider: connection"),
    ],
    ids=["timeout", "no connection"],
)
def test_a_connection_lost_before_anything_arrived_is_retried_and_says_which_kind(lost, said):
    error = classify_transport(lost, None)
    assert error.retryable is True
    assert error.partial is None
    assert str(error).startswith(said)


def test_a_connection_lost_with_no_message_is_named_by_its_type():
    # httpx.ReadError is often raised with nothing to say, and "was lost: " ends nowhere.
    arrived = ModelReply((), StopKind.CUT_OFF, Usage(), "m")
    error = connection_lost(httpx.ReadError(""), arrived, peer="ollama")
    assert str(error) == "the connection to ollama was lost: ReadError"
