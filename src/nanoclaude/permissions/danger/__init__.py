"""Classifying a shell command as safe, refused, or unparseable."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class DangerLevel(StrEnum):
    SAFE = "safe"
    BLOCKED = "blocked"
    UNPARSEABLE = "unparseable"


@dataclass(frozen=True, slots=True)
class DangerRule:
    rule_id: str
    pattern: re.Pattern[str]
    description: str


@dataclass(frozen=True, slots=True)
class DangerVerdict:
    level: DangerLevel
    matches: tuple[str, ...] = ()
    classifier: str = "regex"
    detail: str = ""

    @property
    def reason(self) -> str:
        if self.detail:
            return self.detail
        if not self.matches:
            return "no dangerous pattern matched"
        return "; ".join(self.matches)


@runtime_checkable
class DangerClassifier(Protocol):
    @property
    def name(self) -> str: ...

    def classify(self, command: str) -> DangerVerdict: ...
