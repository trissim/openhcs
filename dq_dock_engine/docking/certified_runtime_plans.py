from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np


def _ensure_nonnegative(value: float, *, field_name: str) -> None:
    if not math.isfinite(value) or value < -1e-6:
        raise ValueError(f"{field_name} must be finite and nonnegative, got {value}")


def _ensure_handle_provenance(handles: tuple[str, ...], *, owner: str) -> None:
    if not handles:
        raise ValueError(f"{owner} must carry theorem provenance handles")


@dataclass(frozen=True)
class CertifiedBudgetBreakdownItem:
    label: str
    value: float
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()
    note: str | None = None

    def __post_init__(self) -> None:
        _ensure_nonnegative(self.value, field_name=f"{self.label}.value")
        _ensure_handle_provenance(self.theorem_handles, owner=self.label)


@dataclass(frozen=True)
class CertifiedPruningDeltaBudget:
    source: str
    shared_base_delta: float
    cutoff_tail_delta: float
    omitted_value_delta: float
    softening_mismatch_delta: float
    total_delta: float
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()
    breakdown: tuple[CertifiedBudgetBreakdownItem, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _ensure_handle_provenance(self.theorem_handles, owner=self.source)
        _ensure_nonnegative(self.shared_base_delta, field_name="shared_base_delta")
        _ensure_nonnegative(self.cutoff_tail_delta, field_name="cutoff_tail_delta")
        _ensure_nonnegative(self.omitted_value_delta, field_name="omitted_value_delta")
        _ensure_nonnegative(
            self.softening_mismatch_delta,
            field_name="softening_mismatch_delta",
        )
        _ensure_nonnegative(self.total_delta, field_name="total_delta")
        expected_total = (
            self.shared_base_delta
            + self.cutoff_tail_delta
            + self.omitted_value_delta
            + self.softening_mismatch_delta
        )
        if not math.isclose(
            self.total_delta,
            expected_total,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "CertifiedPruningDeltaBudget.total_delta must equal the sum of its "
                f"components (expected {expected_total}, got {self.total_delta})"
            )


@dataclass(frozen=True)
class PoseSpecificImprovementBudget:
    pose_index: int | None
    active_torsion_indices: tuple[int, ...]
    active_torsion_mask: tuple[bool, ...]
    per_bond_local_bounds: tuple[float, ...]
    total_budget: float
    active_subset_source: str
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _ensure_handle_provenance(self.theorem_handles, owner=self.active_subset_source)
        if len(self.active_torsion_mask) != len(self.per_bond_local_bounds):
            raise ValueError(
                "active_torsion_mask and per_bond_local_bounds must have matching lengths"
            )
        for index, bound in enumerate(self.per_bond_local_bounds):
            _ensure_nonnegative(bound, field_name=f"per_bond_local_bounds[{index}]")
        _ensure_nonnegative(self.total_budget, field_name="total_budget")
        expected_total = sum(
            bound
            for is_active, bound in zip(
                self.active_torsion_mask,
                self.per_bond_local_bounds,
                strict=True,
            )
            if is_active
        )
        if not math.isclose(
            self.total_budget,
            expected_total,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "PoseSpecificImprovementBudget.total_budget must equal the sum of active "
                f"per-bond bounds (expected {expected_total}, got {self.total_budget})"
            )


@dataclass(frozen=True)
class PoseSpecificImprovementBudgetFamily:
    budgets: tuple[PoseSpecificImprovementBudget, ...]
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _ensure_handle_provenance(
            self.theorem_handles,
            owner="PoseSpecificImprovementBudgetFamily",
        )

    @property
    def totals(self) -> tuple[float, ...]:
        return tuple(budget.total_budget for budget in self.budgets)

    @property
    def max_total_budget(self) -> float:
        return 0.0 if not self.budgets else max(self.totals)

    def totals_array(self) -> np.ndarray:
        return np.asarray(self.totals, dtype=np.dtype("float32"))


class CertifiedRefinementBudgetKind(str, Enum):
    DYADIC_ROUNDS = "dyadic_rounds"
    SE3_ITERATIONS = "se3_iterations"


@dataclass(frozen=True)
class CertifiedRefinementBudget:
    kind: CertifiedRefinementBudgetKind
    n_steps: int
    pose_indices: tuple[int, ...]
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()
    max_total_improvement: float | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        _ensure_handle_provenance(self.theorem_handles, owner=self.kind.value)
        if self.n_steps < 0:
            raise ValueError(f"n_steps must be nonnegative, got {self.n_steps}")
        if self.max_total_improvement is not None:
            _ensure_nonnegative(
                self.max_total_improvement,
                field_name="max_total_improvement",
            )

    @property
    def pose_count(self) -> int:
        return len(self.pose_indices)

    def applies_to_pose(self, pose_index: int) -> bool:
        return pose_index in self.pose_indices


@dataclass(frozen=True)
class CertifiedPipelineCostModel:
    input_pose_count: int
    retained_pose_count: int
    refinement_pose_count: int
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _ensure_handle_provenance(
            self.theorem_handles,
            owner="CertifiedPipelineCostModel",
        )
        for field_name, value in (
            ("input_pose_count", self.input_pose_count),
            ("retained_pose_count", self.retained_pose_count),
            ("refinement_pose_count", self.refinement_pose_count),
        ):
            if value < 0:
                raise ValueError(f"{field_name} must be nonnegative, got {value}")


@dataclass(frozen=True)
class CertifiedPipelineExecutionPlan:
    coarse_scores: object
    lower_bounds: object
    tau: float
    retain_mask: object
    pruning_delta_budget: CertifiedPruningDeltaBudget
    improvement_budget_family: PoseSpecificImprovementBudgetFamily | None = None
    refinement_budget: CertifiedRefinementBudget | None = None
    postfilter_cost_model: CertifiedPipelineCostModel | None = None
    theorem_handles: tuple[str, ...] = ()
    witness_handles: tuple[str, ...] = ()
    final_survivor_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _ensure_handle_provenance(
            self.theorem_handles,
            owner="CertifiedPipelineExecutionPlan",
        )
        if not math.isfinite(self.tau):
            raise ValueError(f"tau must be finite, got {self.tau}")

    @property
    def retain_count(self) -> int:
        return int(np.count_nonzero(np.asarray(self.retain_mask, dtype=bool)))

    @property
    def refinement_pose_indices(self) -> tuple[int, ...]:
        return (
            ()
            if self.refinement_budget is None
            else self.refinement_budget.pose_indices
        )

    def with_improvement_budget_family(
        self,
        improvement_budget_family: PoseSpecificImprovementBudgetFamily,
    ) -> "CertifiedPipelineExecutionPlan":
        return replace(self, improvement_budget_family=improvement_budget_family)

    def with_refinement_budget(
        self,
        refinement_budget: CertifiedRefinementBudget,
        *,
        postfilter_cost_model: CertifiedPipelineCostModel,
    ) -> "CertifiedPipelineExecutionPlan":
        return replace(
            self,
            refinement_budget=refinement_budget,
            postfilter_cost_model=postfilter_cost_model,
        )

    def with_final_survivor_indices(
        self,
        final_survivor_indices: tuple[int, ...],
    ) -> "CertifiedPipelineExecutionPlan":
        return replace(self, final_survivor_indices=final_survivor_indices)


@dataclass(frozen=True)
class CertifiedSeedBudgetCandidate:
    budget: int
    source: str
    adequate: bool
    capture_rmsd: float
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()
    provenance_note: str | None = None
    probe_rank: int | None = None

    def __post_init__(self) -> None:
        _ensure_handle_provenance(
            self.theorem_handles,
            owner=f"CertifiedSeedBudgetCandidate[{self.source}]",
        )
        if self.budget <= 0:
            raise ValueError(f"budget must be positive, got {self.budget}")
        _ensure_nonnegative(self.capture_rmsd, field_name="capture_rmsd")


@dataclass(frozen=True)
class CertifiedSeedBudgetPlan:
    candidates: tuple[CertifiedSeedBudgetCandidate, ...]
    selected_index: int
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()
    engineering_probe_pose_cap: int | None = None
    engineering_probe_top_k: int | None = None

    def __post_init__(self) -> None:
        _ensure_handle_provenance(self.theorem_handles, owner="CertifiedSeedBudgetPlan")
        if not self.candidates:
            raise ValueError("CertifiedSeedBudgetPlan requires at least one candidate")
        if self.selected_index < 0 or self.selected_index >= len(self.candidates):
            raise IndexError(
                f"selected_index {self.selected_index} is outside candidates range"
            )
        if not self.selected_candidate.adequate:
            raise ValueError("selected seed budget candidate must be adequate")

    @property
    def selected_candidate(self) -> CertifiedSeedBudgetCandidate:
        return self.candidates[self.selected_index]

    @property
    def selected_budget(self) -> int:
        return self.selected_candidate.budget

    @property
    def explored_budgets(self) -> tuple[int, ...]:
        return tuple(candidate.budget for candidate in self.candidates)
