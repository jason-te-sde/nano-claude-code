from nanoclaude.providers.base import Usage


def test_usage_add_accumulates_all_four_fields():
    # Four distinct values on each side so a transposed or dropped field (e.g. the
    # accumulator crediting cache_write_tokens to cache_read_tokens) shows up as a
    # wrong number instead of accidentally cancelling out.
    total = Usage(100, 20, 5, 3) + Usage(50, 10, 2, 1)
    assert total == Usage(150, 30, 7, 4)


def test_usage_defaults_are_all_zero():
    assert Usage() == Usage(0, 0, 0, 0)
