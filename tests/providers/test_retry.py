"""Tests for the shared retry policy: spec section 7.4's table, exhaustively,
plus the pieces the table alone does not reach -- message extraction for each
distinct error branch, and the status/no-status split in the attempt limit."""

from collections.abc import Hashable
from typing import Never

import pytest

from nanoclaude.providers.base import ModelError
from nanoclaude.providers.retry import RetryPolicy, classify_status, with_retry


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
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
