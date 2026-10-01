"""Fixtures shared by the whole suite."""

from collections.abc import Iterator

import pytest

from nanoclaude.conversation.store import Store


@pytest.fixture(autouse=True)
def _close_every_store(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Close every Store a test opened, when the test ends.

    Python 3.13 emits a ResourceWarning when a sqlite3 connection is garbage
    collected unclosed, and this suite turns warnings into errors. The warning
    fires at collection time, so it is reported against whichever later test
    happens to be running -- a leak in one module surfaces as a failure in an
    unrelated one. Closing deterministically keeps the attribution honest.
    """
    opened: list[Store] = []
    real_open = Store.open

    def tracking_open(self: Store) -> None:
        real_open(self)
        opened.append(self)

    monkeypatch.setattr(Store, "open", tracking_open)
    yield
    for store in opened:
        store.close()
