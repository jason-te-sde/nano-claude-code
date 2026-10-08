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
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path
from typing import Any

from nanoclaude.providers.base import Usage

STALE_AFTER_DAYS = 90


@dataclass(frozen=True, slots=True)
class Price:
    """US dollars per million tokens, one rate per kind of token.

    A model priced by prompt length (Claude Haiku 5.5) has a second set of rates: a call
    whose prompt is longer than ``threshold_tokens`` is charged ``above_threshold`` for all
    of its tokens, the output included, and not only for the tokens past the line. Without
    one, ``threshold_tokens`` and ``above_threshold`` are both None and one set serves every
    call.
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0
    threshold_tokens: int | None = None
    above_threshold: Price | None = None

    def for_prompt(self, prompt_tokens: int) -> Price:
        """The rates a call whose prompt is ``prompt_tokens`` long is charged.

        Up to and including the threshold is the first set: "for prompts up to 100,000
        tokens" is the vendor's own wording.
        """
        if (
            self.threshold_tokens is not None
            and self.above_threshold is not None
            and prompt_tokens > self.threshold_tokens
        ):
            return self.above_threshold
        return self


def _rates(entry: dict[str, Any]) -> Price:
    """One set of rates from a table: the four prices, the cache ones optional."""
    return Price(
        entry["input"],
        entry["output"],
        entry.get("cache_read", 0.0),
        entry.get("cache_write", 0.0),
    )


def _price_from(name: str, entry: dict[str, Any]) -> Price:
    """A model's price from its table: one set of rates, or two with the threshold between.

    A row that names a threshold and gives no second set (or the reverse) is refused,
    naming the row. Read as a single price it would charge the first set for every prompt,
    however long, and a price table is not a place to guess.
    """
    second = entry.get("above_threshold")
    threshold = entry.get("threshold_tokens")
    if second is None and threshold is None:
        return _rates(entry)
    if second is None or threshold is None:
        raise ValueError(
            f"{name} has a threshold_tokens or an above_threshold table but not both "
            "\u2014 a price that depends on prompt length needs the length and the second prices"
        )
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
        raise ValueError(
            f"{name} has threshold_tokens = {threshold!r} \u2014 it must be a whole number "
            "of tokens, at least 1"
        )
    return replace(_rates(entry), threshold_tokens=threshold, above_threshold=_rates(second))


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
                prices[(adapter, model)] = _price_from(f"{adapter}/{model}", entry)
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
        """Dollars for one call's ``usage``, or None when the model has no price.

        ``usage`` is a single call's: whether it is over a model's threshold is decided on
        the prompt of that call (its input, cache-read and cache-write tokens), and two
        short calls added together do not make a long one.
        """
        price = self.price_for(adapter, model)
        if price is None:
            return None
        rates = price.for_prompt(
            usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
        )
        return (
            usage.input_tokens * rates.input_per_mtok
            + usage.output_tokens * rates.output_per_mtok
            + usage.cache_read_tokens * rates.cache_read_per_mtok
            + usage.cache_write_tokens * rates.cache_write_per_mtok
        ) / 1_000_000
