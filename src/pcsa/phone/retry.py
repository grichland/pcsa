"""Retry with exponential backoff, driven by the injected clock."""

from __future__ import annotations

from collections.abc import Callable

from pcsa.common import log
from pcsa.common.clock import Clock
from pcsa.common.config import RetryConfig
from pcsa.common.errors import Transient


def with_retry[T](
    operation: str,
    attempt: Callable[[], T],
    policy: RetryConfig,
    clock: Clock,
    on_failure: Callable[[int, Exception], None] | None = None,
) -> T:
    """Call `attempt` up to `policy.attempts` times. Between attempts: `on_failure` (e.g. reconnect),
    then a backoff sleep. The last failure is re-raised as Transient."""
    delay = policy.base_s
    last: Exception | None = None
    for number in range(1, policy.attempts + 1):
        try:
            return attempt()
        except Transient as error:
            last = error
            log.warning("device_retry", op=operation, attempt=number, of=policy.attempts, reason=str(error))
            if number == policy.attempts:
                break
            if on_failure is not None:
                on_failure(number, error)
            clock.sleep(delay)
            delay = min(delay * 2, policy.max_s)
    raise Transient(f"{operation} failed after {policy.attempts} attempts: {last}") from last
