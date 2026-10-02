import datetime as dt
from collections.abc import Hashable
from pathlib import Path

import pytest

from nanoclaude.providers import pricing
from nanoclaude.providers.base import Usage
from nanoclaude.providers.pricing import Price, PriceBook

# One row per entry in pricing.toml, written out by hand from the vendor's
# pricing page (verified 2026-10-02). The table is the thing being pinned: a row
# that is deleted, mistyped, or read into the wrong column fails its own case.
SHIPPED_ROWS = [
    ("claude-opus-5-5", Price(4.00, 20.00, 0.20, 5.00)),
    ("claude-opus-5", Price(5.00, 25.00, 0.50, 6.25)),
    ("claude-sonnet-5", Price(2.00, 10.00, 0.20, 2.50)),
    ("claude-haiku-4-5", Price(1.00, 5.00, 0.10, 1.25)),
    ("claude-haiku-4-5-20251001", Price(1.00, 5.00, 0.10, 1.25)),
]


def write_table(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pricing.toml"
    path.write_text(text)
    return path


def test_a_known_model_costs_what_the_table_says():
    book = PriceBook.load()
    cost = book.cost(Usage(input_tokens=1_000_000, output_tokens=0), "anthropic", "claude-sonnet-5")
    assert cost == pytest.approx(2.00)  # $2 per million input tokens, pricing.toml


def test_cache_reads_are_cheaper_than_fresh_input():
    book = PriceBook.load()
    fresh = book.cost(Usage(input_tokens=1_000_000), "anthropic", "claude-sonnet-5")
    cached = book.cost(Usage(cache_read_tokens=1_000_000), "anthropic", "claude-sonnet-5")
    assert cached is not None and fresh is not None and cached < fresh


def test_an_unknown_model_costs_none_not_zero():
    """Review Focus #5. Showing $0.00 lets someone spend a hundred dollars
    believing it was free. Not knowing must look like not knowing."""
    book = PriceBook.load()
    assert book.cost(Usage(input_tokens=1_000_000), "openai_compat", "mystery-model") is None


def test_a_local_model_is_explicitly_free_rather_than_unknown():
    book = PriceBook.load()
    assert book.cost(Usage(input_tokens=1_000_000), "ollama", "anything") == 0.0


def test_the_table_carries_a_date_and_its_age_is_available():
    assert PriceBook.load().age_days() >= 0


@pytest.mark.parametrize(("model", "expected"), SHIPPED_ROWS)
def test_every_shipped_row_carries_its_four_verified_prices(model, expected):
    assert PriceBook.load().price_for("anthropic", model) == expected


def test_each_token_kind_is_priced_at_its_own_rate():
    # Four different token counts against four different rates, so a field read
    # into the wrong slot (output billed at the cache rate, say) changes the sum:
    # 1M*5 + 2M*25 + 3M*0.50 + 4M*6.25 = 5 + 50 + 1.5 + 25 = 81.5 dollars.
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=2_000_000,
        cache_read_tokens=3_000_000,
        cache_write_tokens=4_000_000,
    )
    assert PriceBook.load().cost(usage, "anthropic", "claude-opus-5") == pytest.approx(81.5)


def test_a_price_belongs_to_its_adapter_not_just_its_model_name():
    # The same id served through a different adapter is a different deal (a proxy
    # or reseller sets its own rates), so it must not inherit this row.
    book = PriceBook.load()
    assert book.price_for("openai_compat", "claude-sonnet-5") is None
    assert book.cost(Usage(input_tokens=10), "openai_compat", "claude-sonnet-5") is None


def test_a_local_model_is_free_in_every_token_kind():
    usage = Usage(1_000_000, 1_000_000, 1_000_000, 1_000_000)
    book = PriceBook.load()
    assert book.price_for("ollama", "qwen3-coder") == Price(0.0, 0.0)
    assert book.cost(usage, "ollama", "qwen3-coder") == 0.0


def test_a_row_without_cache_prices_charges_nothing_for_cache_tokens(tmp_path):
    table = write_table(
        tmp_path,
        'last_updated = "2026-10-02"\n[acme."plain"]\ninput = 1.50\noutput = 6.00\n',
    )
    book = PriceBook.load(table)
    assert book.price_for("acme", "plain") == Price(1.50, 6.00, 0.0, 0.0)


def test_a_custom_table_replaces_the_shipped_one_entirely(tmp_path):
    table = write_table(
        tmp_path, 'last_updated = "2026-10-02"\n[acme."plain"]\ninput = 1.50\noutput = 6.00\n'
    )
    book = PriceBook.load(table)
    assert book.price_for("anthropic", "claude-sonnet-5") is None


def test_the_date_may_be_written_as_a_quoted_string_or_a_toml_date(tmp_path):
    quoted = write_table(tmp_path, 'last_updated = "2026-10-02"\n')
    assert PriceBook.load(quoted).last_updated == dt.date(2026, 10, 2)
    bare = write_table(tmp_path, "last_updated = 2026-10-03\n")
    assert PriceBook.load(bare).last_updated == dt.date(2026, 10, 3)


def test_the_shipped_table_is_dated_the_day_it_was_verified():
    assert PriceBook.load().last_updated == dt.date(2026, 10, 2)


def test_age_is_counted_in_days_from_the_date_the_table_was_checked():
    book = PriceBook.load()
    assert book.age_days(dt.date(2026, 10, 2)) == 0
    assert book.age_days(dt.date(2026, 10, 12)) == 10
    # Before the table's own date is a negative age, not an absolute value.
    assert book.age_days(dt.date(2026, 9, 30)) == -2


def test_a_table_is_stale_only_after_ninety_days():
    # Both sides of the boundary: the 90th day is still fresh, the 91st is not.
    book = PriceBook.load()
    checked = dt.date(2026, 10, 2)
    assert book.is_stale(checked) is False
    assert book.is_stale(checked + dt.timedelta(days=90)) is False
    assert book.is_stale(checked + dt.timedelta(days=91)) is True


def test_the_table_loads_without_reading_a_path_relative_to_this_module(monkeypatch):
    # Installed from a wheel, the table is a resource of the package, not
    # necessarily a file next to the source. Loading must not depend on where the
    # module claims to live.
    monkeypatch.setattr(pricing, "__file__", "/nonexistent/place/pricing.py")
    assert PriceBook.load().price_for("anthropic", "claude-sonnet-5") is not None


def test_price_is_hashable_because_it_holds_only_numbers():
    assert isinstance(Price(1.0, 2.0), Hashable)
    assert hash(Price(1.0, 2.0)) == hash(Price(1.0, 2.0))


def test_price_book_is_not_hashable():
    # It holds a dict of prices. Left to frozen=True's default, dataclass would
    # generate a real __hash__ that reports Hashable and then raises, naming
    # "dict" instead of this class.
    book = PriceBook({("acme", "m"): Price(1.0, 2.0)}, dt.date(2026, 10, 2))
    assert not isinstance(book, Hashable)  # type: ignore[unreachable]
    with pytest.raises(TypeError, match="unhashable type: 'PriceBook'"):
        hash(book)
