import datetime as dt
from collections.abc import Hashable
from pathlib import Path

import pytest

from nanoclaude.providers import pricing
from nanoclaude.providers.base import Usage
from nanoclaude.providers.pricing import Price, PriceBook

# One row per entry in pricing.toml, written out by hand from the vendor's
# pricing page (verified 2026-10-07). The table is the thing being pinned: a row
# that is deleted, mistyped, or read into the wrong column fails its own case.
SHIPPED_ROWS = [
    ("claude-fable-5-1", Price(10.00, 50.00, 0.25, 12.50)),
    ("claude-opus-5-5", Price(4.00, 20.00, 0.20, 5.00)),
    ("claude-opus-5", Price(5.00, 25.00, 0.50, 6.25)),
    # Cache reads on Sonnet 5.5 are 0.05x the base input price, not the 0.1x of the rest.
    ("claude-sonnet-5-5", Price(2.00, 10.00, 0.10, 2.50)),
    ("claude-sonnet-5", Price(2.00, 10.00, 0.20, 2.50)),
    # Haiku 5.5 is priced by prompt length: "for prompts up to 100,000 tokens", then
    # "for prompts over 100,000 tokens" (the second set is the one after the threshold).
    (
        "claude-haiku-5-5",
        Price(0.10, 0.50, 0.01, 0.125, 100_000, Price(0.50, 2.50, 0.05, 0.625)),
    ),
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


def test_no_row_ships_without_a_hand_checked_line_above():
    shipped = {model for adapter, model in PriceBook.load().prices if adapter == "anthropic"}
    assert shipped == {model for model, _ in SHIPPED_ROWS}


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
    assert PriceBook.load().last_updated == dt.date(2026, 10, 7)


def test_age_is_counted_in_days_from_the_date_the_table_was_checked():
    book = PriceBook.load()
    assert book.age_days(dt.date(2026, 10, 7)) == 0
    assert book.age_days(dt.date(2026, 10, 17)) == 10
    # Before the table's own date is a negative age, not an absolute value.
    assert book.age_days(dt.date(2026, 10, 5)) == -2


def test_a_table_is_stale_only_after_ninety_days():
    # Both sides of the boundary: the 90th day is still fresh, the 91st is not.
    book = PriceBook.load()
    checked = dt.date(2026, 10, 7)
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


# --- Prices that depend on how long the prompt is (Claude Haiku 5.5) ----------------------

HAIKU = ("anthropic", "claude-haiku-5-5")
MILLION = 1_000_000


def haiku_cost(usage: Usage) -> float:
    cost = PriceBook.load().cost(usage, *HAIKU)
    assert cost is not None
    return cost


def test_a_prompt_of_exactly_the_threshold_is_still_charged_at_the_first_prices():
    # "For prompts up to 100,000 tokens": the limit itself belongs to the cheaper set.
    assert haiku_cost(Usage(input_tokens=100_000)) == pytest.approx(100_000 * 0.10 / MILLION)


def test_one_token_over_the_threshold_moves_the_whole_call_to_the_second_prices():
    # Not only the tokens past the line: the vendor charges the whole prompt, and the
    # output, at the longer-prompt rates.
    assert haiku_cost(Usage(input_tokens=100_001)) == pytest.approx(100_001 * 0.50 / MILLION)
    short = haiku_cost(Usage(input_tokens=100_000, output_tokens=MILLION))
    long = haiku_cost(Usage(input_tokens=100_001, output_tokens=MILLION))
    assert short == pytest.approx(0.01 + 0.50)
    assert long == pytest.approx(100_001 * 0.50 / MILLION + 2.50)


def test_the_prompt_is_the_input_and_the_cache_reads_and_the_cache_writes_together():
    # 1 + 60,000 + 40,000 = 100,001: over, though no one kind of token is anywhere near it.
    over = Usage(input_tokens=1, cache_read_tokens=60_000, cache_write_tokens=40_000)
    assert haiku_cost(over) == pytest.approx((1 * 0.50 + 60_000 * 0.05 + 40_000 * 0.625) / MILLION)
    # 50,000 + 50,000 = 100,000: not over.
    at = Usage(input_tokens=50_000, cache_read_tokens=50_000)
    assert haiku_cost(at) == pytest.approx((50_000 * 0.10 + 50_000 * 0.01) / MILLION)


def test_the_tokens_the_model_writes_are_not_part_of_the_prompt():
    usage = Usage(input_tokens=100_000, output_tokens=500_000)
    assert haiku_cost(usage) == pytest.approx(100_000 * 0.10 / MILLION + 500_000 * 0.50 / MILLION)


def test_a_model_with_one_set_of_prices_is_charged_them_however_long_the_prompt():
    book = PriceBook.load()
    huge = Usage(input_tokens=900_000)
    assert book.cost(huge, "anthropic", "claude-sonnet-5-5") == pytest.approx(
        900_000 * 2.0 / MILLION
    )


def test_a_table_can_give_a_second_set_of_prices_after_a_threshold(tmp_path):
    table = write_table(
        tmp_path,
        'last_updated = "2026-10-07"\n'
        '[acme."long"]\ninput = 1.0\noutput = 2.0\ncache_read = 0.1\ncache_write = 1.25\n'
        "threshold_tokens = 1000\n"
        '[acme."long".above_threshold]\ninput = 3.0\noutput = 6.0\n'
        "cache_read = 0.3\ncache_write = 3.75\n",
    )
    book = PriceBook.load(table)
    assert book.price_for("acme", "long") == Price(
        1.0, 2.0, 0.1, 1.25, 1000, Price(3.0, 6.0, 0.3, 3.75)
    )
    assert book.cost(Usage(input_tokens=1000), "acme", "long") == pytest.approx(
        1000 * 1.0 / MILLION
    )
    assert book.cost(Usage(input_tokens=1001), "acme", "long") == pytest.approx(
        1001 * 3.0 / MILLION
    )


@pytest.mark.parametrize(
    "body",
    [
        "threshold_tokens = 1000\n",
        '[acme."half".above_threshold]\ninput = 3.0\noutput = 6.0\n',
    ],
    ids=["a threshold and no second prices", "second prices and no threshold"],
)
def test_half_of_a_tiered_row_is_refused_by_name_and_not_priced_at_one_set(tmp_path, body):
    # Reading it as a single price would charge the cheap set for every prompt, however long.
    head = 'last_updated = "2026-10-07"\n[acme."half"]\ninput = 1.0\noutput = 2.0\n'
    table = write_table(tmp_path, head + body)
    with pytest.raises(ValueError, match="acme/half"):
        PriceBook.load(table)


def table_of(rates: dict[str, str]) -> str:
    """The lines of one table of prices, from the ones a test wants there."""
    return "".join(f"{key} = {value}\n" for key, value in rates.items())


@pytest.mark.parametrize("missing", ["input", "output"])
def test_a_row_without_an_input_or_an_output_price_is_refused_naming_the_row(tmp_path, missing):
    # A KeyError that says only 'input' leaves the person to find which of the hundred rows.
    rates = {"input": "1.0", "output": "2.0"}
    del rates[missing]
    table = write_table(tmp_path, 'last_updated = "2026-10-07"\n[acme."plain"]\n' + table_of(rates))
    with pytest.raises(ValueError, match="acme/plain") as refused:
        PriceBook.load(table)
    assert repr(missing) in str(refused.value) and "above_threshold" not in str(refused.value)


@pytest.mark.parametrize("missing", ["input", "output"])
def test_a_second_set_without_an_input_or_an_output_price_is_refused_naming_it_and_the_row(
    tmp_path, missing
):
    second = {"input": "3.0", "output": "6.0"}
    del second[missing]
    table = write_table(
        tmp_path,
        'last_updated = "2026-10-07"\n[acme."tiered"]\ninput = 1.0\noutput = 2.0\n'
        "threshold_tokens = 1000\n"
        '[acme."tiered".above_threshold]\n' + table_of(second),
    )
    with pytest.raises(ValueError, match="acme/tiered") as refused:
        PriceBook.load(table)
    assert repr(missing) in str(refused.value) and "above_threshold" in str(refused.value)


def test_the_prices_that_are_optional_may_be_left_out_of_either_table(tmp_path):
    table = write_table(
        tmp_path,
        'last_updated = "2026-10-07"\n[acme."tiered"]\ninput = 1.0\noutput = 2.0\n'
        "threshold_tokens = 1000\n"
        '[acme."tiered".above_threshold]\ninput = 3.0\noutput = 6.0\n',
    )
    assert PriceBook.load(table).price_for("acme", "tiered") == Price(
        1.0, 2.0, 0.0, 0.0, 1000, Price(3.0, 6.0, 0.0, 0.0)
    )


@pytest.mark.parametrize("threshold", ["0", "-5", "1.5", '"many"'])
def test_a_threshold_that_is_not_a_positive_whole_number_is_refused(tmp_path, threshold):
    table = write_table(
        tmp_path,
        'last_updated = "2026-10-07"\n[acme."bad"]\ninput = 1.0\noutput = 2.0\n'
        f"threshold_tokens = {threshold}\n"
        '[acme."bad".above_threshold]\ninput = 3.0\noutput = 6.0\n',
    )
    with pytest.raises(ValueError, match="acme/bad"):
        PriceBook.load(table)


def test_a_price_with_a_second_set_is_still_hashable():
    two = Price(1.0, 2.0, 0.1, 1.25, 1000, Price(3.0, 6.0))
    assert isinstance(two, Hashable) and hash(two) == hash(
        Price(1.0, 2.0, 0.1, 1.25, 1000, Price(3.0, 6.0))
    )
