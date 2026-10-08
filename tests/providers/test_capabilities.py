import pytest

from nanoclaude.providers.capabilities import _KNOWN, CONSERVATIVE_DEFAULT, capabilities_for


def test_known_anthropic_model_has_explicit_cache_and_native_tools():
    caps = capabilities_for("anthropic", "claude-sonnet-5")
    assert caps.native_tools and caps.parallel_tools
    assert caps.cache == "explicit"
    # The _KNOWN entry and the "claude-" family fallback agree on the fields
    # above and differ in these two (1M against 200K, 128K against 8K), so
    # bypassing _KNOWN and falling through to _FAMILIES fails here.
    assert caps.context_window == 1_000_000
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


# One line per OpenRouter row in _KNOWN, written out by hand from OpenRouter's public model
# list (https://openrouter.ai/api/v1/models), checked 2026-10-07: the context window the
# listing gives for the model, and the output the program budgets (see the row's comment).
LISTED = [
    ("deepseek/deepseek-v4-pro", 1_024_000, 64_000),
]


@pytest.mark.parametrize(("model", "window", "max_output"), LISTED)
def test_each_known_openrouter_model_carries_its_listed_limits(model, window, max_output):
    caps = capabilities_for("openai_compat", model)
    assert (caps.context_window, caps.max_output) == (window, max_output)
    # What its listing says (tools, tool_choice, a cache-read price) and nothing it does not:
    # text in, text out, and no reasoning style the adapter would have to read.
    assert caps.native_tools and caps.parallel_tools
    assert caps.cache == "automatic"
    assert caps.reasoning == "none" and not caps.vision


def test_a_listed_model_is_not_given_its_families_smaller_row():
    # The "deepseek" family row is for 64K-window models; compaction is set from the window,
    # so a million-token model that fell through to it would be compacted at about 45K tokens.
    family = capabilities_for("openai_compat", "deepseek-chat")
    listed = capabilities_for("openai_compat", "deepseek/deepseek-v4-pro")
    assert family.context_window == 64_000
    assert listed.context_window > 15 * family.context_window


def test_no_openai_compat_row_ships_without_a_hand_checked_line_above():
    shipped = {model for adapter, model in _KNOWN if adapter == "openai_compat"}
    assert shipped == {model for model, _, _ in LISTED}


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
