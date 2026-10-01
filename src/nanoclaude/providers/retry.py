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

from nanoclaude.providers.base import ModelError

T = TypeVar("T")

RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 529})


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


def classify_status(status: int, body: str) -> ModelError:
    detail = _message_from(body)
    if status in (401, 403):
        return ModelError(
            f"the provider rejected your credentials (HTTP {status}): {detail} — "
            "check the key, or run: ncc init",
            retryable=False,
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
    )


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
