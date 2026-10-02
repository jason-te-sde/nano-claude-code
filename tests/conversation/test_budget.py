from collections.abc import Hashable

import pytest

from nanoclaude.conversation.budget import (
    Budget,
    ContextTooSmallError,
    HeuristicCounter,
    estimate_tokens,
    transcript_text,
)
from nanoclaude.conversation.transcript import (
    Message,
    Role,
    TextBlock,
    ThinkingBlock,
    Transcript,
    user_text,
)
from nanoclaude.providers.capabilities import Capabilities

BIG = Capabilities(True, True, "explicit", 200_000, 8_192)
SMALL = Capabilities(False, False, "none", 8_192, 2_048)

#: max_output alone already exceeds the window: available() must go
#: non-positive rather than wrap around or raise.
TINY = Capabilities(False, False, "none", 1_000, 1_100)


def transcript_of(size: int) -> Transcript:
    messages = []
    for index in range(size):
        role: Role = "user" if index % 2 == 0 else "assistant"
        messages.append(Message(role, (TextBlock("word " * 200),)))
    return Transcript(tuple(messages))


def test_estimates_scale_with_length():
    assert estimate_tokens("hello world") < estimate_tokens("hello world " * 100)
    assert estimate_tokens("") == 0


def test_available_reserves_room_for_the_output_and_a_margin():
    budget = Budget(BIG)
    assert budget.available() < BIG.context_window - BIG.max_output


def test_an_empty_conversation_is_ok():
    budget = Budget(BIG)
    assert budget.verdict(Transcript(), "system", "") == "ok"


def test_crossing_the_soft_threshold_asks_for_micro_compaction():
    budget = Budget(SMALL, soft=0.01, hard=0.99)
    assert budget.verdict(transcript_of(2), "system", "") == "micro"


def test_crossing_the_hard_threshold_asks_for_full_compaction():
    budget = Budget(SMALL, soft=0.001, hard=0.002)
    assert budget.verdict(transcript_of(2), "system", "") == "full"


def test_a_context_that_cannot_fit_even_empty_is_impossible_not_full():
    """Review Focus #3. An 8k model with a 6k project map has nothing left.

    Compaction cannot help -- the fixed part alone does not fit -- and the
    error has to say which part is too big, or the user will keep retrying.
    """
    budget = Budget(SMALL)
    enormous_system = "word " * 20_000
    assert budget.verdict(Transcript(), enormous_system, "") == "impossible"
    with pytest.raises(ContextTooSmallError, match="system prompt and tools"):
        budget.require_fits(Transcript(), enormous_system, "")


def test_the_impossible_error_names_the_window_and_what_is_in_it():
    # The brief's own version of this test used a bare try/except with the
    # assertions inside it: if require_fits ever stopped raising, the except
    # body would simply never run and the test would pass having checked
    # nothing (confirmed by deleting the raise -- this test stayed green).
    # pytest.raises fails the test itself ("DID NOT RAISE") when that happens.
    budget = Budget(SMALL)
    with pytest.raises(ContextTooSmallError) as exc_info:
        budget.require_fits(Transcript(), "word " * 20_000, "")
    assert "8192" in str(exc_info.value)
    assert "--model" in str(exc_info.value)  # tells them the way out


# The tests below extend the brief's own cases: every boundary of verdict()'s
# thresholds pinned exactly (not just "crosses, roughly"), both named halves
# of require_fits (raises / is a no-op), pressure()'s own non-positive-window
# guard, and the other Produces names (estimate_tokens, HeuristicCounter) each
# getting their own direct test rather than only incidental exercise.


class ConstantCounter:
    """Returns a fixed count for any non-empty text, zero for empty text.

    HeuristicCounter's char/3.5 estimate would need fragile string-length
    arithmetic to land a pressure ratio on an exact boundary; this test double
    makes ``pressure == n / available`` exact and independent of what text is
    actually passed in.
    """

    def __init__(self, n: int) -> None:
        self.n = n

    def count(self, text: str) -> int:
        return 0 if text == "" else self.n


def test_estimate_tokens_is_never_negative():
    for text in ("", "a", "word " * 5_000):
        assert estimate_tokens(text) >= 0


def test_heuristic_counter_delegates_to_estimate_tokens():
    counter = HeuristicCounter()
    assert counter.count("hello world") == estimate_tokens("hello world")
    assert counter.count("") == 0


def test_transcript_text_includes_thinking_blocks_as_well_as_text_blocks():
    # transcript_text's isinstance check accepts TextBlock *or* ThinkingBlock;
    # pin both halves, not just whichever one the other tests happen to use.
    t = Transcript(
        (
            user_text("hi"),
            Message("assistant", (ThinkingBlock("because X", "sig"), TextBlock("answer"))),
        )
    )
    text = transcript_text(t)
    assert "because X" in text
    assert "answer" in text


def test_fixed_cost_ignores_the_transcript():
    budget = Budget(BIG, counter=ConstantCounter(7))
    assert budget.fixed_cost("system prompt", "tool list") == 14


def test_used_adds_the_transcript_cost_to_the_fixed_cost():
    budget = Budget(BIG, counter=ConstantCounter(10))
    assert budget.used(transcript_of(1), "", "") == 10


def test_pressure_is_zero_when_available_is_non_positive():
    # The dedicated guard in pressure(), not just verdict()'s own check. Uses
    # a non-empty transcript (used > 0) rather than Transcript(): available()
    # is negative here, and 0 / a negative number is also 0.0 in floating
    # point, so a used of exactly 0 would let an unguarded division pass this
    # assertion by coincidence rather than by the guard actually firing.
    assert Budget(TINY).available() <= 0
    assert Budget(TINY).pressure(transcript_of(2), "", "") == 0.0


def test_verdict_is_impossible_when_available_is_non_positive_even_for_an_empty_request():
    # The "available <= 0" half of verdict()'s impossible check, kept distinct
    # from the brief's own "fixed cost too big for a positive window" case.
    assert Budget(TINY).verdict(Transcript(), "", "") == "impossible"


def test_require_fits_does_nothing_when_the_budget_is_fine():
    # The non-raising half of require_fits; the brief only tests the raise.
    # require_fits is annotated -> None, so mypy --strict itself refuses to
    # let anything treat that as a value (func-returns-value) -- called as a
    # bare statement instead; if it raised, the test would error.
    Budget(BIG).require_fits(Transcript(), "system", "")


def test_verdict_is_ok_one_token_under_the_soft_threshold():
    available = Budget(BIG).available()
    n = available // 2
    soft = n / available
    budget = Budget(BIG, soft=soft, hard=0.999_999, counter=ConstantCounter(n - 1))
    assert budget.verdict(transcript_of(2), "", "") == "ok"


def test_verdict_is_micro_exactly_at_the_soft_threshold():
    available = Budget(BIG).available()
    n = available // 2
    budget = Budget(BIG, soft=n / available, hard=0.999_999, counter=ConstantCounter(n))
    assert budget.verdict(transcript_of(2), "", "") == "micro"


def test_verdict_is_micro_one_token_over_the_soft_threshold():
    available = Budget(BIG).available()
    n = available // 2
    soft = n / available
    budget = Budget(BIG, soft=soft, hard=0.999_999, counter=ConstantCounter(n + 1))
    assert budget.verdict(transcript_of(2), "", "") == "micro"


def test_verdict_is_micro_one_token_under_the_hard_threshold():
    available = Budget(BIG).available()
    n = int(available * 0.9)
    hard = n / available
    budget = Budget(BIG, soft=0.0, hard=hard, counter=ConstantCounter(n - 1))
    assert budget.verdict(transcript_of(2), "", "") == "micro"


def test_verdict_is_full_exactly_at_the_hard_threshold():
    available = Budget(BIG).available()
    n = int(available * 0.9)
    budget = Budget(BIG, soft=0.0, hard=n / available, counter=ConstantCounter(n))
    assert budget.verdict(transcript_of(2), "", "") == "full"


def test_verdict_is_full_one_token_over_the_hard_threshold():
    available = Budget(BIG).available()
    n = int(available * 0.9)
    hard = n / available
    budget = Budget(BIG, soft=0.0, hard=hard, counter=ConstantCounter(n + 1))
    assert budget.verdict(transcript_of(2), "", "") == "full"


def test_budget_is_hashable():
    # Unlike transcript.py's blocks -- holding decoded-JSON tool arguments,
    # explicitly declared unhashable there -- a Budget's fields are a
    # Capabilities record, three floats and a stateless counter: nothing
    # mutable, nothing that can hold an unhashable value. The dataclass's
    # generated __hash__ is correct as-is; pinned here rather than left as an
    # unexamined accident of frozen=True.
    budget = Budget(BIG)
    assert isinstance(budget, Hashable)
    assert hash(budget) == hash(Budget(BIG))
