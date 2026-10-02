import pytest

from nanoclaude.providers.capabilities import _KNOWN, CONSERVATIVE_DEFAULT, capabilities_for


def test_known_anthropic_model_has_explicit_cache_and_native_tools():
    caps = capabilities_for("anthropic", "claude-sonnet-5")
    assert caps.native_tools and caps.parallel_tools
    assert caps.cache == "explicit"
    assert caps.context_window >= 200_000
    # The _KNOWN exact entry and the "claude-" family fallback agree on every
    # other field checked above, so only max_output (128_000 vs 8_192) can tell
    # them apart. Without this line, bypassing _KNOWN entirely and always
    # falling through to _FAMILIES still passes every test in this file.
    assert caps.max_output == 128_000


# One line per Claude row in _KNOWN, written out by hand from the vendor's model
# pages (checked 2026-10-02): context window and synchronous max output.
PUBLISHED = [
    ("claude-fable-5-1", 1_000_000, 128_000),
    ("claude-opus-5-5", 1_000_000, 128_000),
    ("claude-opus-5", 1_000_000, 128_000),
    ("claude-sonnet-5-5", 1_000_000, 128_000),
    ("claude-sonnet-5", 1_000_000, 128_000),
    ("claude-haiku-4-5", 200_000, 64_000),
    ("claude-haiku-4-5-20251001", 200_000, 64_000),
]


@pytest.mark.parametrize(("model", "window", "max_output"), PUBLISHED)
def test_each_known_claude_model_carries_its_published_limits(model, window, max_output):
    caps = capabilities_for("anthropic", model)
    assert (caps.context_window, caps.max_output) == (window, max_output)
    assert caps.native_tools and caps.parallel_tools and caps.vision
    assert caps.cache == "explicit"
    assert caps.reasoning == "thinking"


def test_no_claude_row_ships_without_a_hand_checked_line_above():
    shipped = {model for adapter, model in _KNOWN if adapter == "anthropic"}
    assert shipped == {model for model, _, _ in PUBLISHED}


def test_unknown_model_falls_back_to_the_conservative_default():
    caps = capabilities_for("openai_compat", "some-model-nobody-has-heard-of")
    assert caps == CONSERVATIVE_DEFAULT


def test_family_prefix_match_applies_when_no_exact_entry_exists():
    # Neither of the brief's own two tests reaches the _FAMILIES loop at all: the
    # first hits an exact _KNOWN entry, the second falls all the way through to
    # CONSERVATIVE_DEFAULT. A model that is new but clearly part of a known family
    # (a Claude release not yet added to _KNOWN by name) must still get that
    # family's real capabilities rather than the conservative floor.
    caps = capabilities_for("anthropic", "claude-3-5-haiku-20241022")
    assert caps.native_tools and caps.parallel_tools
    assert caps.cache == "explicit"
    assert caps.reasoning == "thinking"


def test_family_prefix_is_paired_with_its_own_adapter():
    # "claude-" is only a family prefix under the "anthropic" adapter. Matching by
    # prefix alone -- ignoring which adapter asked -- would let an OpenAI-compatible
    # endpoint serving a model that merely happens to be named "claude-..." (a
    # proxy or a re-exported model id) inherit Anthropic's capabilities.
    caps = capabilities_for("openai_compat", "claude-3-5-haiku-20241022")
    assert caps == CONSERVATIVE_DEFAULT


def test_conservative_default_never_assumes_capabilities_it_has_not_seen():
    """An unknown model must not be assumed to support parallel calls or caching."""
    # Every field of Capabilities is pinned here, not just the four the brief's own
    # version of this test checked -- CONSERVATIVE_DEFAULT's whole job is to be wrong
    # in the safe direction, and a field left unchecked is a field a regression could
    # silently flip to something a real adapter would read as a green light.
    assert CONSERVATIVE_DEFAULT.native_tools is False
    assert CONSERVATIVE_DEFAULT.parallel_tools is False
    assert CONSERVATIVE_DEFAULT.cache == "none"
    assert CONSERVATIVE_DEFAULT.context_window <= 32_000
    assert CONSERVATIVE_DEFAULT.max_output <= 4_096
    assert CONSERVATIVE_DEFAULT.reasoning == "none"
    assert CONSERVATIVE_DEFAULT.vision is False
