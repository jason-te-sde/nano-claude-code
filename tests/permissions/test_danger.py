"""Structural pins for the danger types.

Step 1 of the permissions brief is three type declarations with no behaviour
of their own -- the policy tests exercise them instead. This file pins only
the one property that has nothing to do with classifier behaviour: whether
the two dataclasses declared here are hashable.
"""

import re
from collections.abc import Hashable

import pytest

from nanoclaude.permissions.danger import DangerClassifier, DangerLevel, DangerRule, DangerVerdict


class _StubClassifier:
    """A minimal conforming DangerClassifier, for the two tests below only.

    Not a classifier implementation for Tasks 7-8 to build on -- it exists
    solely to give DangerClassifier, named in the brief's Produces list with
    no test of its own, an actual object to check isinstance/issubclass
    against.
    """

    @property
    def name(self) -> str:
        return "stub"

    def classify(self, _command: str) -> DangerVerdict:
        return DangerVerdict(DangerLevel.SAFE)


def test_danger_rule_is_hashable():
    # Fields are rule_id: str, pattern: re.Pattern[str], description: str.
    # Compiled regex objects are themselves hashable, so frozen=True's
    # generated __hash__ is not a trap here, unlike ToolUseBlock
    # (conversation/transcript.py), which holds a Mapping. Pinned so a later
    # field of a mapping or other unhashable type gets caught the same way
    # those were.
    rule = DangerRule("rm.recursive-force", re.compile(r"rm\s+-rf"), "recursive force delete")
    other = DangerRule("rm.recursive-force", re.compile(r"rm\s+-rf"), "recursive force delete")
    assert isinstance(rule, Hashable)
    assert hash(rule) == hash(other)


def test_danger_verdict_is_hashable():
    # Fields are level: DangerLevel (a str enum), matches: tuple[str, ...],
    # classifier: str, detail: str -- all already hashable, so this composes
    # cleanly. Pinned for the same reason as test_danger_rule_is_hashable.
    verdict = DangerVerdict(DangerLevel.BLOCKED, ("rm.recursive-force",), "regex")
    other = DangerVerdict(DangerLevel.BLOCKED, ("rm.recursive-force",), "regex")
    assert isinstance(verdict, Hashable)
    assert hash(verdict) == hash(other)


def test_danger_verdict_reason_falls_back_to_joined_matches():
    verdict = DangerVerdict(DangerLevel.BLOCKED, ("a", "b"), "regex")
    assert verdict.reason == "a; b"


def test_danger_verdict_reason_prefers_detail_over_matches():
    verdict = DangerVerdict(DangerLevel.UNPARSEABLE, (), "ast", detail="unbalanced quotes")
    assert verdict.reason == "unbalanced quotes"


def test_danger_verdict_reason_names_no_pattern_matched_when_empty():
    verdict = DangerVerdict(DangerLevel.SAFE)
    assert verdict.reason == "no dangerous pattern matched"


def test_danger_classifier_isinstance_check_works_on_a_conforming_stub():
    assert isinstance(_StubClassifier(), DangerClassifier)


def test_danger_classifier_issubclass_check_is_unusable_because_name_is_a_property():
    """A documented limitation, not a defect to fix: @runtime_checkable only
    supports isinstance() against an actual instance when a Protocol has a
    non-method member -- here, the `name` property. issubclass() against the
    class itself raises instead of returning an answer either way.
    """
    with pytest.raises(TypeError, match="non-method members"):
        # mypy flags this call too, correctly predicting the exact runtime
        # TypeError this test exists to pin -- the ignore is for that static
        # finding, not a disagreement with it.
        issubclass(_StubClassifier, DangerClassifier)  # type: ignore[misc]
