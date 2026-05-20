from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BudgetScope = Literal["agent", "session", "run"]
BudgetDimension = Literal[
    "input_tokens", "output_tokens", "cache_tokens", "dollars", "wall_clock_seconds"
]

# Ordered list used by BudgetChecker to iterate dimensions.
BUDGET_DIMENSIONS: tuple[BudgetDimension, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_tokens",
    "dollars",
    "wall_clock_seconds",
)


@dataclass(frozen=True)
class BudgetThreshold:
    """Limits across all tracked dimensions at a single enforcement level.

    Any field left as None is unconstrained.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_tokens: int | None = None
    dollars: float | None = None
    wall_clock_seconds: float | None = None


@dataclass(frozen=True)
class BudgetConfig:
    """Budget policy for a single scope (agent / session / run-span).

    Both *soft* and *hard* are optional; set neither to disable enforcement
    for that scope entirely.  When both are set, soft < hard is expected but
    not enforced — the checker evaluates them independently.
    """

    scope: BudgetScope
    scope_id: str
    soft: BudgetThreshold | None = None
    hard: BudgetThreshold | None = None


@dataclass(frozen=True)
class UsageSnapshot:
    """Accumulated usage at the point of a budget check."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0
    dollars: float = 0.0
    wall_clock_seconds: float = 0.0


@dataclass(frozen=True)
class BudgetViolation:
    """A single threshold breach recorded during a budget check."""

    scope: BudgetScope
    scope_id: str
    dimension: BudgetDimension
    threshold_type: Literal["soft", "hard"]
    limit: float
    actual: float


@dataclass(frozen=True)
class BudgetCheckResult:
    """Returned when all hard limits pass; carries any soft-threshold warnings."""

    warnings: list[BudgetViolation] = field(default_factory=list)
