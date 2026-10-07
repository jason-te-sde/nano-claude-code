"""What to do when a provider says no.

For a tool people run against their own key, provider failures are routine, not
exceptional. The table in spec 7.4 is implemented here once and shared by all
three adapters, because a retry policy that differs per provider is a retry
policy nobody can reason about.

The distinction that matters is retryable versus not. A 429 will pass. A 401
will not, and retrying it five times just makes someone wait thirty seconds to
be told their key is wrong.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import httpx

from nanoclaude.providers.base import CredentialsError, ModelError, ModelReply

T = TypeVar("T")

RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 529})

#: What a provider's error body says when the request does not fit the model's
#: context window, matched case-insensitively. Each is a phrase from the provider's
#: own error, so a new provider or a reworded error is one more line here and one
#: more case in the tests.
OVERFLOW_MARKERS = (
    "prompt is too long",  # Anthropic
    "context_length_exceeded",  # OpenAI's error code
    "maximum context length",  # OpenAI's message, and vLLM's
    "exceed context limit",  # Anthropic, when the input and max_tokens do not fit together
    "exceeds the available context size",  # llama.cpp
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    overload_attempts: int = 5
    network_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0

    # Left to frozen=True's default (eq=True, hash generated from the four
    # fields): every field is an int or a float, both hashable unconditionally,
    # so unlike ToolSpec/ModelRequest/ModelReply/Transcript this class has no
    # way to end up holding something unhashable. A policy is also naturally a
    # value worth using as a dict key or set member (e.g. grouping retries by
    # policy in a test), so the default is the right default here, not merely
    # an unexamined one.


def _message_from(body: str) -> str:
    try:
        payload = json.loads(body)
    except ValueError:
        return body[:500]
    if not isinstance(payload, dict):
        # Valid JSON need not be an object: a proxy can return a bare string or
        # a list. Fall back to the raw text rather than raising from .get().
        return body[:500]
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message", body))[:500]
    return str(error or body)[:500]


def _reports_overflow(body: str) -> bool:
    """Whether the body says the request does not fit the model's window.

    Matched on the whole body, not on the message quoted back to the person: that
    is cut at 500 characters.
    """
    lowered = body.lower()
    return any(marker in lowered for marker in OVERFLOW_MARKERS)


def classify_status(status: int, body: str) -> ModelError:
    detail = _message_from(body)
    if status in (401, 403):
        return CredentialsError(
            f"the provider rejected your credentials (HTTP {status}): {detail} — "
            "check the key, or run: ncc init",
            status=status,
        )
    if status == 404:
        return ModelError(
            f"the provider does not know that model or endpoint (HTTP 404): {detail}",
            retryable=False,
            status=status,
        )
    if status in RETRYABLE_STATUSES:
        return ModelError(
            f"the provider is busy (HTTP {status}): {detail}", retryable=True, status=status
        )
    return ModelError(
        f"the provider rejected the request (HTTP {status}): {detail}",
        retryable=False,
        status=status,
        # Only a 400: the same words under another status are not an overflow
        # the session knows how to fix.
        context_overflow=status == 400 and _reports_overflow(body),
    )


def describe(exc: BaseException) -> str:
    """What a person is told of an exception: its message, or its name when it has none."""
    return str(exc) or type(exc).__name__


def connection_lost(
    exc: BaseException, partial: ModelReply, *, peer: str = "the provider"
) -> ModelError:
    """Spec 7.4, a stream that breaks midway: keep what arrived, and do not ask again.

    The request may already have had effects, and asking again would write the first half
    of the reply a second time, so the error is not retryable and what to do next is left
    to the person. ``partial`` is what the adapter had received when the connection went.
    """
    return ModelError(
        f"the connection to {peer} was lost: {describe(exc)}", retryable=False, partial=partial
    )


def classify_transport(exc: httpx.TransportError, partial: ModelReply | None) -> ModelError:
    """Spec 7.4 for a connection that failed, a timeout included.

    Asked again, as before, unless some of the reply had arrived (``partial`` is not
    None): then it is :func:`connection_lost`, and not retried.
    """
    if partial is not None:
        return connection_lost(exc, partial)
    if isinstance(exc, httpx.TimeoutException):
        return ModelError(f"the provider timed out: {exc}", retryable=True)
    return ModelError(f"could not reach the provider: {exc}", retryable=True)


def unreadable_stream(
    exc: BaseException, partial: ModelReply, *, peer: str = "the provider"
) -> ModelError:
    """Spec 7.4 for a stream that arrived and could not be read: keep what came before.

    A body that does not decode, a line that is not JSON: the connection did not fail, and
    nothing is wrong with asking, but a reply that was being shown and was never finished
    is the same case as one whose connection dropped, and is treated the same way.
    """
    return ModelError(
        f"the response from {peer} could not be read: {describe(exc)}",
        retryable=False,
        partial=partial,
    )


def classify_stream_error(exc: Exception, partial: ModelReply | None) -> ModelError | None:
    """What a failure inside a stream's loop means, or None to raise it as it came.

    A connection that failed is :func:`classify_transport`. Anything else the loop raised
    (httpx.DecodingError, a line that is not JSON, any other ValueError) is kept as a
    cut-off once some of the reply has arrived, and before that is left as it always was.
    """
    if isinstance(exc, httpx.TransportError):
        return classify_transport(exc, partial)
    if partial is None:
        return None
    return unreadable_stream(exc, partial)


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    on_retry: Callable[[int, float, str], None],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: Callable[[], float] = random.random,
) -> T:
    """Run ``operation``, retrying only what the policy says is retryable."""
    attempt = 0
    while True:
        try:
            return await operation()
        except ModelError as error:
            if not error.retryable:
                raise
            limit = policy.overload_attempts if error.status else policy.network_attempts
            attempt += 1
            if attempt >= limit:
                raise
            ceiling = min(policy.max_delay_s, policy.base_delay_s * (2 ** (attempt - 1)))
            delay = ceiling * (0.5 + 0.5 * jitter())
            on_retry(attempt, delay, str(error))
            await sleep(delay)
