"""What a session cost, or an honest admission that we do not know.

Prices change and this table is shipped, so it is always slightly wrong. The
design accommodates that: every entry is keyed by adapter and model, the file
carries the date it was checked, /cost warns when that date is old, and a model
with no entry reports nothing rather than zero.

Zero is the dangerous answer. Someone watching $0.00 tick along on a model we
have no price for will keep going, and the bill arrives later.
"""

from __future__ import annotations

import datetime as dt
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from nanoclaude.providers.base import Usage

STALE_AFTER_DAYS = 90


@dataclass(frozen=True, slots=True)
class Price:
    """US dollars per million tokens, one rate per kind of token."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0


@dataclass(frozen=True, slots=True)
class PriceBook:
    """Every price this release knows, and the day they were last checked."""

    prices: dict[tuple[str, str], Price]
    last_updated: dt.date

    # Declared unhashable rather than left to the default: prices is a dict. Left
    # to frozen=True's default eq=True, dataclass would generate a real __hash__
    # -- isinstance(x, Hashable) would say True -- that only raises when actually
    # called, and names "dict" rather than this class.
    __hash__ = None  # type: ignore[assignment]

    @staticmethod
    def load(path: Path | None = None) -> PriceBook:
        """The table shipped with the package, or the one at ``path``."""
        if path is None:
            # A resource of the package rather than a path beside this file: from
            # an installed wheel the two need not be the same thing, and only a
            # non-editable install can show the difference.
            source = resources.files("nanoclaude.providers").joinpath("pricing.toml")
            text = source.read_text(encoding="utf-8")
        else:
            text = path.read_text(encoding="utf-8")
        data = tomllib.loads(text)
        updated = dt.date.fromisoformat(str(data.pop("last_updated")))
        prices: dict[tuple[str, str], Price] = {}
        for adapter, models in data.items():
            for model, entry in models.items():
                prices[(adapter, model)] = Price(
                    entry["input"],
                    entry["output"],
                    entry.get("cache_read", 0.0),
                    entry.get("cache_write", 0.0),
                )
        return PriceBook(prices, updated)

    def age_days(self, today: dt.date | None = None) -> int:
        """Whole days since the table was checked; negative if it is dated ahead."""
        return ((today or dt.date.today()) - self.last_updated).days

    def is_stale(self, today: dt.date | None = None) -> bool:
        """True once the table is older than :data:`STALE_AFTER_DAYS`."""
        return self.age_days(today) > STALE_AFTER_DAYS

    def price_for(self, adapter: str, model: str) -> Price | None:
        """The rates for this model, or None when we were never told them."""
        if adapter == "ollama":
            return Price(0.0, 0.0)  # it runs on your own machine
        return self.prices.get((adapter, model))

    def cost(self, usage: Usage, adapter: str, model: str) -> float | None:
        """Dollars for ``usage``, or None when the model has no price."""
        price = self.price_for(adapter, model)
        if price is None:
            return None
        return (
            usage.input_tokens * price.input_per_mtok
            + usage.output_tokens * price.output_per_mtok
            + usage.cache_read_tokens * price.cache_read_per_mtok
            + usage.cache_write_tokens * price.cache_write_per_mtok
        ) / 1_000_000
