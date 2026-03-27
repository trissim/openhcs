from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from dq_dock_engine.docking.core import ReturnedPoseContractDecision


def _ensure_nonnegative(value: float, *, field_name: str) -> None:
    if not math.isfinite(value) or value < -1e-6:
        raise ValueError(f"{field_name} must be finite and nonnegative, got {value}")


def _ensure_handle_provenance(handles: tuple[str, ...], *, owner: str) -> None:
    if not handles:
        raise ValueError(f"{owner} must carry theorem provenance handles")


def _cover_stats_against_selected_subset(
    full_points: np.ndarray,
    selected_points: np.ndarray,
) -> tuple[np.ndarray, float]:
    if full_points.size == 0 or selected_points.size == 0:
        raise ValueError("cover stats require non-empty full and selected point sets")
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(selected_points).query(full_points, k=1)
        nearest = selected_points[np.asarray(indices, dtype=np.int64)]
    except Exception:
        deltas = full_points[:, None, :] - selected_points[None, :, :]
        sq = np.sum(deltas * deltas, axis=2)
        indices = np.argmin(sq, axis=1)
        nearest = selected_points[indices]
        distances = np.sqrt(np.min(sq, axis=1))
    axis_half_widths = np.max(np.abs(full_points - nearest), axis=0)
    return axis_half_widths, float(np.max(distances))


class PoseSpecificImprovementBudgetKind(str, Enum):
    ACTIVE_SUBSET = "active_subset"
    RIGID_LOCAL = "rigid_local"


def _validate_pose_budget_shape(
    *,
    budget_kind: PoseSpecificImprovementBudgetKind,
    active_subset_source: str,
    theorem_handles: tuple[str, ...],
    active_torsion_mask: tuple[bool, ...],
    per_bond_local_bounds: tuple[float, ...],
    total_budget: float,
) -> None:
    _ensure_handle_provenance(theorem_handles, owner=active_subset_source)
    _ensure_nonnegative(total_budget, field_name="total_budget")
    if budget_kind == PoseSpecificImprovementBudgetKind.RIGID_LOCAL:
        if active_torsion_mask or per_bond_local_bounds:
            raise ValueError(
                "rigid_local pose budgets must not carry torsion-mask or per-bond data"
            )
        return
    if len(active_torsion_mask) != len(per_bond_local_bounds):
        raise ValueError(
            "active_torsion_mask and per_bond_local_bounds must have matching lengths"
        )
    for index, bound in enumerate(per_bond_local_bounds):
        _ensure_nonnegative(bound, field_name=f"per_bond_local_bounds[{index}]")
    expected_total = sum(
        bound
        for is_active, bound in zip(
            active_torsion_mask,
            per_bond_local_bounds,
            strict=True,
        )
        if is_active
    )
    if not math.isclose(
        total_budget,
        expected_total,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "total_budget must equal the sum of active per-bond bounds "
            f"(expected {expected_total}, got {total_budget})"
        )


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


class CertifiedPruningDeltaComponentKind(str, Enum):
    SHARED_BASE = "shared_base"
    CUTOFF_TAIL = "cutoff_tail"
    OMITTED_VALUE = "omitted_value"
    COOPERATIVE_CORRECTION = "cooperative_correction"
    WATER_MEDIATED = "water_mediated"
    SOFTENING_MISMATCH = "softening_mismatch"
    RECEPTOR_FLEX = "receptor_flex"


@dataclass(frozen=True)
class CertifiedPruningDeltaComponent:
    label: str
    kind: CertifiedPruningDeltaComponentKind
    active: bool
    delta: float
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()
    note: str | None = None

    def __post_init__(self) -> None:
        _ensure_nonnegative(self.delta, field_name=f"{self.label}.delta")
        _ensure_handle_provenance(self.theorem_handles, owner=self.label)
        if not self.active and not math.isclose(
            self.delta,
            0.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"Inactive pruning component {self.label} must have zero delta"
            )

    def as_breakdown_item(self) -> CertifiedBudgetBreakdownItem:
        return CertifiedBudgetBreakdownItem(
            label=self.label,
            value=self.delta,
            theorem_handles=self.theorem_handles,
            witness_handles=self.witness_handles,
            note=self.note,
        )


def _sum_pruning_components(
    components: tuple[CertifiedPruningDeltaComponent, ...],
    *kinds: CertifiedPruningDeltaComponentKind,
) -> float:
    allowed = set(kinds)
    return sum(
        component.delta
        for component in components
        if component.active and component.kind in allowed
    )


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
    components: tuple[CertifiedPruningDeltaComponent, ...] | None = None

    @classmethod
    def from_components(
        cls,
        *,
        source: str,
        components: tuple[CertifiedPruningDeltaComponent, ...],
        theorem_handles: tuple[str, ...],
        witness_handles: tuple[str, ...] = (),
    ) -> "CertifiedPruningDeltaBudget":
        return cls(
            source=source,
            shared_base_delta=_sum_pruning_components(
                components,
                CertifiedPruningDeltaComponentKind.SHARED_BASE,
            ),
            cutoff_tail_delta=_sum_pruning_components(
                components,
                CertifiedPruningDeltaComponentKind.CUTOFF_TAIL,
            ),
            omitted_value_delta=_sum_pruning_components(
                components,
                CertifiedPruningDeltaComponentKind.OMITTED_VALUE,
                CertifiedPruningDeltaComponentKind.COOPERATIVE_CORRECTION,
                CertifiedPruningDeltaComponentKind.WATER_MEDIATED,
            ),
            softening_mismatch_delta=_sum_pruning_components(
                components,
                CertifiedPruningDeltaComponentKind.SOFTENING_MISMATCH,
            ),
            total_delta=sum(
                component.delta for component in components if component.active
            ),
            theorem_handles=theorem_handles,
            witness_handles=witness_handles,
            breakdown=tuple(
                component.as_breakdown_item()
                for component in components
                if component.active or component.delta > 0.0
            ),
            components=components,
        )

    def __post_init__(self) -> None:
        _ensure_handle_provenance(self.theorem_handles, owner=self.source)
        components = self.components
        if components is None:
            raise ValueError(
                "CertifiedPruningDeltaBudget requires explicit authoritative components; "
                "construct it with CertifiedPruningDeltaBudget.from_components(...)"
            )
        assert components is not None
        if not self.breakdown:
            object.__setattr__(
                self,
                "breakdown",
                tuple(
                    component.as_breakdown_item()
                    for component in components
                    if component.active or component.delta > 0.0
                ),
            )
        _ensure_nonnegative(self.shared_base_delta, field_name="shared_base_delta")
        _ensure_nonnegative(self.cutoff_tail_delta, field_name="cutoff_tail_delta")
        _ensure_nonnegative(self.omitted_value_delta, field_name="omitted_value_delta")
        _ensure_nonnegative(
            self.softening_mismatch_delta,
            field_name="softening_mismatch_delta",
        )
        _ensure_nonnegative(self.total_delta, field_name="total_delta")
        expected_shared_base = _sum_pruning_components(
            components,
            CertifiedPruningDeltaComponentKind.SHARED_BASE,
        )
        expected_cutoff_tail = _sum_pruning_components(
            components,
            CertifiedPruningDeltaComponentKind.CUTOFF_TAIL,
        )
        expected_omitted = _sum_pruning_components(
            components,
            CertifiedPruningDeltaComponentKind.OMITTED_VALUE,
            CertifiedPruningDeltaComponentKind.COOPERATIVE_CORRECTION,
            CertifiedPruningDeltaComponentKind.WATER_MEDIATED,
        )
        expected_softening = _sum_pruning_components(
            components,
            CertifiedPruningDeltaComponentKind.SOFTENING_MISMATCH,
        )
        for field_name, actual, expected in (
            ("shared_base_delta", self.shared_base_delta, expected_shared_base),
            ("cutoff_tail_delta", self.cutoff_tail_delta, expected_cutoff_tail),
            ("omitted_value_delta", self.omitted_value_delta, expected_omitted),
            (
                "softening_mismatch_delta",
                self.softening_mismatch_delta,
                expected_softening,
            ),
        ):
            if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(
                    f"CertifiedPruningDeltaBudget.{field_name} must equal the sum of its authoritative components "
                    f"(expected {expected}, got {actual})"
                )
        if any(
            component.kind == CertifiedPruningDeltaComponentKind.RECEPTOR_FLEX
            and component.active
            and component.delta > 0.0
            for component in components
        ):
            raise ValueError(
                "CertifiedPruningDeltaBudget total_delta must only include theorem-owned pruning components; "
                "receptor-flex corrections belong on the additive correction path"
            )
        expected_total = sum(
            component.delta for component in components if component.active
        )
        if not math.isclose(
            self.total_delta,
            expected_total,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "CertifiedPruningDeltaBudget.total_delta must equal the sum of active authoritative "
                f"components (expected {expected_total}, got {self.total_delta})"
            )

    @property
    def active_components(self) -> tuple[CertifiedPruningDeltaComponent, ...]:
        assert self.components is not None
        return tuple(component for component in self.components if component.active)

    @property
    def component_delta_map(self) -> dict[str, float]:
        return {
            component.label: component.delta for component in self.active_components
        }

    def debug_summary(self) -> dict[str, object]:
        return {
            "source": self.source,
            "total_delta": self.total_delta,
            "shared_base_delta": self.shared_base_delta,
            "cutoff_tail_delta": self.cutoff_tail_delta,
            "omitted_value_delta": self.omitted_value_delta,
            "softening_mismatch_delta": self.softening_mismatch_delta,
            "active_components": tuple(
                {
                    "label": component.label,
                    "kind": component.kind.value,
                    "delta": component.delta,
                    "theorem_handles": component.theorem_handles,
                    "witness_handles": component.witness_handles,
                    "note": component.note,
                }
                for component in self.active_components
            ),
        }


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
    kind: PoseSpecificImprovementBudgetKind = (
        PoseSpecificImprovementBudgetKind.ACTIVE_SUBSET
    )

    def __post_init__(self) -> None:
        _validate_pose_budget_shape(
            budget_kind=self.kind,
            active_subset_source=self.active_subset_source,
            theorem_handles=self.theorem_handles,
            active_torsion_mask=self.active_torsion_mask,
            per_bond_local_bounds=self.per_bond_local_bounds,
            total_budget=self.total_budget,
        )

    @classmethod
    def from_active_subset_budget(
        cls,
        budget: "CertifiedActiveSubsetBudget",
    ) -> "PoseSpecificImprovementBudget":
        return cls(
            pose_index=budget.pose_index,
            kind=PoseSpecificImprovementBudgetKind.ACTIVE_SUBSET,
            active_torsion_indices=budget.active_torsion_indices,
            active_torsion_mask=budget.active_torsion_mask,
            per_bond_local_bounds=budget.per_bond_local_bounds,
            total_budget=budget.total_budget,
            active_subset_source=budget.active_subset_source,
            theorem_handles=budget.theorem_handles,
            witness_handles=budget.witness_handles,
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

    @property
    def pose_indices(self) -> tuple[int | None, ...]:
        return tuple(budget.pose_index for budget in self.budgets)

    @classmethod
    def from_active_subset_family(
        cls,
        family: "CertifiedActiveSubsetBudgetFamily",
    ) -> "PoseSpecificImprovementBudgetFamily":
        return cls(
            budgets=tuple(
                PoseSpecificImprovementBudget.from_active_subset_budget(budget)
                for budget in family.budgets
            ),
            theorem_handles=family.theorem_handles,
            witness_handles=family.witness_handles,
        )

    def debug_summary(self) -> dict[str, object]:
        return {
            "budget_count": len(self.budgets),
            "max_total_budget": self.max_total_budget,
            "totals": self.totals,
            "pose_indices": self.pose_indices,
            "kinds": tuple(dict.fromkeys(budget.kind.value for budget in self.budgets)),
            "theorem_handles": self.theorem_handles,
        }


@dataclass(frozen=True)
class CertifiedActiveSubsetBudget:
    pose_index: int | None
    active_torsion_indices: tuple[int, ...]
    active_torsion_mask: tuple[bool, ...]
    per_bond_local_bounds: tuple[float, ...]
    total_budget: float
    active_subset_source: str
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_pose_budget_shape(
            budget_kind=PoseSpecificImprovementBudgetKind.ACTIVE_SUBSET,
            active_subset_source=self.active_subset_source,
            theorem_handles=self.theorem_handles,
            active_torsion_mask=self.active_torsion_mask,
            per_bond_local_bounds=self.per_bond_local_bounds,
            total_budget=self.total_budget,
        )

    def as_pose_specific_improvement_budget(self) -> PoseSpecificImprovementBudget:
        return PoseSpecificImprovementBudget.from_active_subset_budget(self)


@dataclass(frozen=True)
class CertifiedActiveSubsetBudgetFamily:
    budgets: tuple[CertifiedActiveSubsetBudget, ...]
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _ensure_handle_provenance(
            self.theorem_handles,
            owner="CertifiedActiveSubsetBudgetFamily",
        )

    @property
    def totals(self) -> tuple[float, ...]:
        return tuple(budget.total_budget for budget in self.budgets)

    @property
    def max_total_budget(self) -> float:
        return 0.0 if not self.budgets else max(self.totals)

    def totals_array(self) -> np.ndarray:
        return np.asarray(self.totals, dtype=np.dtype("float32"))

    def as_pose_specific_improvement_family(
        self,
    ) -> PoseSpecificImprovementBudgetFamily:
        return PoseSpecificImprovementBudgetFamily.from_active_subset_family(self)

    def debug_summary(self) -> dict[str, object]:
        return {
            "budget_count": len(self.budgets),
            "max_total_budget": self.max_total_budget,
            "totals": self.totals,
            "theorem_handles": self.theorem_handles,
        }


class CertifiedRefinementBudgetKind(str, Enum):
    DYADIC_ROUNDS = "dyadic_rounds"
    SE3_ITERATIONS = "se3_iterations"


class CertifiedRigidSeedRegionKind(str, Enum):
    BOX = "box"
    CERTIFIED_BINDING_SITE = "certified_binding_site"


@dataclass(frozen=True)
class CertifiedRigidSeedFamilyPlan:
    region_kind: CertifiedRigidSeedRegionKind
    pose_count: int
    adequate_pose_count: int
    translation_point_count: int
    lattice_resolution: int
    quaternion_count: int
    translation_search_volume: float
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()
    box_center: tuple[float, float, float] | None = None
    box_size: tuple[float, float, float] | None = None
    binding_site_center: tuple[float, float, float] | None = None
    binding_site_radius: float | None = None
    translation_full_lattice: bool = False
    target_translation_cover_radius: float | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        _ensure_handle_provenance(self.theorem_handles, owner=self.region_kind.value)
        for field_name, value in (
            ("pose_count", self.pose_count),
            ("adequate_pose_count", self.adequate_pose_count),
            ("translation_point_count", self.translation_point_count),
            ("lattice_resolution", self.lattice_resolution),
            ("quaternion_count", self.quaternion_count),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")
        _ensure_nonnegative(
            self.translation_search_volume,
            field_name="translation_search_volume",
        )
        if self.target_translation_cover_radius is not None:
            _ensure_nonnegative(
                self.target_translation_cover_radius,
                field_name="target_translation_cover_radius",
            )
        if self.translation_point_count * self.quaternion_count < self.pose_count:
            raise ValueError(
                "translation_point_count * quaternion_count must cover pose_count"
            )
        if self.adequate_pose_count > self.pose_count:
            raise ValueError(
                "adequate_pose_count must not exceed the realized pose_count"
            )
        if self.region_kind == CertifiedRigidSeedRegionKind.BOX:
            if self.box_center is None or self.box_size is None:
                raise ValueError(
                    "box rigid seed family plans require box_center and box_size"
                )
            if (
                self.binding_site_center is not None
                or self.binding_site_radius is not None
            ):
                raise ValueError(
                    "box rigid seed family plans must not carry binding-site geometry"
                )
        elif self.region_kind == CertifiedRigidSeedRegionKind.CERTIFIED_BINDING_SITE:
            if self.binding_site_center is None or self.binding_site_radius is None:
                raise ValueError(
                    "binding-site rigid seed family plans require binding_site_center and binding_site_radius"
                )
            if self.box_center is not None or self.box_size is not None:
                raise ValueError(
                    "binding-site rigid seed family plans must not carry box geometry"
                )
            _ensure_nonnegative(
                self.binding_site_radius, field_name="binding_site_radius"
            )

    def debug_summary(self) -> dict[str, object]:
        from dq_dock_engine.docking.formal_sampling import (
            materialize_certified_rigid_seed_family,
        )

        family = materialize_certified_rigid_seed_family(self)
        unique_translations = np.asarray(family.translations, dtype=np.float64)
        unique_quaternions = np.asarray(family.quaternions, dtype=np.float64)

        translation_points = np.unique(np.round(unique_translations, 6), axis=0)
        quaternion_points = np.unique(np.round(unique_quaternions, 6), axis=0)

        if self.region_kind == CertifiedRigidSeedRegionKind.BOX:
            assert self.box_center is not None
            assert self.box_size is not None
            center = np.asarray(self.box_center, dtype=np.float64)
            size = np.asarray(self.box_size, dtype=np.float64)
            lower = center - size / 2.0
            upper = center + size / 2.0
            step = size / float(self.lattice_resolution)
            axis_centers = tuple(
                tuple(
                    float(value)
                    for value in np.linspace(
                        lower[axis] + 0.5 * step[axis],
                        upper[axis] - 0.5 * step[axis],
                        self.lattice_resolution,
                    ).tolist()
                )
                for axis in range(3)
            )
            translation_box_kind = "full_box_lattice"
            full_translation_count = self.lattice_resolution**3
            translation_subset_filtered = (
                self.translation_point_count != full_translation_count
            )
            full_grid = np.stack(
                np.meshgrid(*axis_centers, indexing="ij"), axis=-1
            ).reshape((-1, 3))
            subset_axis_half_widths, subset_cover_radius = (
                _cover_stats_against_selected_subset(full_grid, translation_points)
            )
            full_cell_half_widths = step / 2.0
            total_axis_half_widths = full_cell_half_widths + subset_axis_half_widths
            continuous_cover_radius = float(np.linalg.norm(total_axis_half_widths))
        else:
            assert self.binding_site_center is not None
            assert self.binding_site_radius is not None
            center = np.asarray(self.binding_site_center, dtype=np.float64)
            radius = float(self.binding_site_radius)
            lower = center - radius
            upper = center + radius
            step = np.full((3,), (2.0 * radius) / float(self.lattice_resolution))
            axis_centers = tuple(
                tuple(
                    float(value)
                    for value in np.linspace(
                        lower[axis] + 0.5 * step[axis],
                        upper[axis] - 0.5 * step[axis],
                        self.lattice_resolution,
                    ).tolist()
                )
                for axis in range(3)
            )
            translation_box_kind = (
                "binding_site_full_bounding_box_lattice"
                if self.translation_full_lattice
                else "sphere_bounding_box_lattice"
            )
            full_translation_count = self.lattice_resolution**3
            translation_subset_filtered = (
                self.translation_point_count != full_translation_count
            )
            full_grid = np.stack(
                np.meshgrid(*axis_centers, indexing="ij"), axis=-1
            ).reshape((-1, 3))
            if self.translation_full_lattice:
                subset_axis_half_widths, subset_cover_radius = (
                    _cover_stats_against_selected_subset(full_grid, translation_points)
                )
                full_cell_half_widths = step / 2.0
                total_axis_half_widths = full_cell_half_widths + subset_axis_half_widths
                continuous_cover_radius = float(np.linalg.norm(total_axis_half_widths))
            else:
                in_sphere = (
                    np.linalg.norm(full_grid - center[None, :], axis=1)
                    <= float(self.binding_site_radius) + 1e-9
                )
                sphere_grid = full_grid[in_sphere]
                subset_axis_half_widths, subset_cover_radius = (
                    _cover_stats_against_selected_subset(
                        sphere_grid, translation_points
                    )
                )
                full_cell_half_widths = None
                total_axis_half_widths = None
                continuous_cover_radius = None

        obstructions: list[str] = []
        if translation_subset_filtered:
            obstructions.append(
                "translation support is a selected subset of the lattice, not the full tensor-product grid"
            )
        obstructions.append(
            "rotation support is quaternion_dictionary8, not a tensor-product of three scalar rotation center sets"
        )

        rigid_dictionary_bridge_obstructions: list[str] = []
        if continuous_cover_radius is None:
            rigid_dictionary_bridge_obstructions.append(
                "translation cover radius over the enclosing box is unavailable for this region kind"
            )

        return {
            "region_kind": self.region_kind.value,
            "pose_count": self.pose_count,
            "adequate_pose_count": self.adequate_pose_count,
            "translation_point_count": self.translation_point_count,
            "lattice_resolution": self.lattice_resolution,
            "quaternion_count": self.quaternion_count,
            "translation_search_volume": self.translation_search_volume,
            "translation_grid_kind": translation_box_kind,
            "translation_bounds_lower": tuple(float(v) for v in lower.tolist()),
            "translation_bounds_upper": tuple(float(v) for v in upper.tolist()),
            "translation_axis_centers": axis_centers,
            "translation_axis_half_widths": tuple(
                float(v) for v in (step / 2.0).tolist()
            ),
            "translation_subset_axis_half_widths_over_lattice": (
                None
                if subset_axis_half_widths is None
                else tuple(float(v) for v in subset_axis_half_widths.tolist())
            ),
            "translation_cover_axis_half_widths_over_box": (
                None
                if total_axis_half_widths is None
                else tuple(float(v) for v in total_axis_half_widths.tolist())
            ),
            "translation_subset_cover_radius_over_lattice": subset_cover_radius,
            "translation_cover_radius_over_box": continuous_cover_radius,
            "translation_cover_theorem_handles": (
                () if continuous_cover_radius is None else ("CSC61", "CSC64")
            ),
            "rotation_dictionary_theorem_handles": (
                "CSC65",
                "CSC70",
                "CSC72",
                "CSC74",
                "CSC76",
                "CSC78",
                "CSC79",
                "CSC80",
                "CSC81",
                "CSC82",
                "CSC84",
                "CSC85",
                "CSC86",
                "CSC87",
                "CSC88",
                "CSC89",
                "CSC90",
                "CSC91",
                "CSC92",
            ),
            "rotation_winner_bridge_theorem_handles": (
                "CSC67",
                "CSC69",
                "CSC71",
                "CSC73",
                "CSC75",
                "CSC77",
                "CSC83",
            ),
            "quaternion_signed_distance_witness_radius": 1.0,
            "actual_translation_points": tuple(
                tuple(float(v) for v in row.tolist()) for row in translation_points
            ),
            "actual_quaternion_points": tuple(
                tuple(float(v) for v in row.tolist()) for row in quaternion_points
            ),
            "full_translation_lattice_point_count": full_translation_count,
            "translation_tensor_product_complete": not translation_subset_filtered,
            "rotation_tensor_product_complete": False,
            "csc61_csc62_ready": False,
            "csc61_csc62_obstructions": tuple(obstructions),
            "translation_full_lattice": self.translation_full_lattice,
            "target_translation_cover_radius": self.target_translation_cover_radius,
            "csc63_csc77_ready": len(rigid_dictionary_bridge_obstructions) == 0,
            "csc63_csc77_obstructions": tuple(rigid_dictionary_bridge_obstructions),
            "theorem_handles": self.theorem_handles,
            "witness_handles": self.witness_handles,
            "note": self.note,
        }


@dataclass(frozen=True)
class CertifiedConformerCoveragePlan:
    source: str
    n_torsions: int
    score_lipschitz_constant: float
    per_bond_lipschitz: tuple[float, ...]
    canonical_segments: tuple[int, ...]
    support_size: int
    max_cells: int
    min_cell_radius: float
    support_target_delta: float
    target_delta: float
    target_rmsd: float
    max_arm: float
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _ensure_handle_provenance(self.theorem_handles, owner=self.source)
        if self.n_torsions < 0:
            raise ValueError(f"n_torsions must be nonnegative, got {self.n_torsions}")
        if len(self.per_bond_lipschitz) != self.n_torsions:
            raise ValueError("per_bond_lipschitz length must match n_torsions")
        if len(self.canonical_segments) != self.n_torsions:
            raise ValueError("canonical_segments length must match n_torsions")
        for index, value in enumerate(self.per_bond_lipschitz):
            _ensure_nonnegative(value, field_name=f"per_bond_lipschitz[{index}]")
        for index, value in enumerate(self.canonical_segments):
            if value <= 0:
                raise ValueError(
                    f"canonical_segments[{index}] must be positive, got {value}"
                )
        for field_name, value in (
            ("support_size", self.support_size),
            ("max_cells", self.max_cells),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")
        for field_name, value in (
            ("score_lipschitz_constant", self.score_lipschitz_constant),
            ("min_cell_radius", self.min_cell_radius),
            ("support_target_delta", self.support_target_delta),
            ("target_delta", self.target_delta),
            ("target_rmsd", self.target_rmsd),
            ("max_arm", self.max_arm),
        ):
            _ensure_nonnegative(value, field_name=field_name)

    def as_branch_and_bound_config(
        self,
        *,
        reuse_initial_conformer: bool,
        max_conformers: int = 1,
    ) -> object:
        from dq_dock_engine.docking.conformer_search import BranchAndBoundConfig

        return BranchAndBoundConfig.from_coverage_plan(
            self,
            reuse_initial_conformer=reuse_initial_conformer,
            max_conformers=max_conformers,
        )


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
    active_subset_budget_family: CertifiedActiveSubsetBudgetFamily | None = None
    improvement_budget_family: PoseSpecificImprovementBudgetFamily | None = None
    refinement_budget: CertifiedRefinementBudget | None = None
    postfilter_cost_model: CertifiedPipelineCostModel | None = None
    theorem_handles: tuple[str, ...] = ()
    witness_handles: tuple[str, ...] = ()
    final_survivor_indices: tuple[int, ...] | None = None
    native_witness_debug: dict[str, object] | None = None

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

    def with_active_subset_budget_family(
        self,
        active_subset_budget_family: CertifiedActiveSubsetBudgetFamily,
    ) -> "CertifiedPipelineExecutionPlan":
        return replace(
            self,
            active_subset_budget_family=active_subset_budget_family,
            improvement_budget_family=active_subset_budget_family.as_pose_specific_improvement_family(),
        )

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

    def with_native_witness_debug(
        self,
        native_witness_debug: dict[str, object],
    ) -> "CertifiedPipelineExecutionPlan":
        return replace(self, native_witness_debug=native_witness_debug)

    def debug_summary(self) -> dict[str, object]:
        return {
            "tau": self.tau,
            "retain_count": self.retain_count,
            "final_survivor_indices": self.final_survivor_indices,
            "pruning_delta_budget": self.pruning_delta_budget.debug_summary(),
            "active_subset_budget_family": (
                None
                if self.active_subset_budget_family is None
                else self.active_subset_budget_family.debug_summary()
            ),
            "improvement_budget_family": (
                None
                if self.improvement_budget_family is None
                else self.improvement_budget_family.debug_summary()
            ),
            "refinement_pose_indices": self.refinement_pose_indices,
            "native_witness_debug": self.native_witness_debug,
            "theorem_handles": self.theorem_handles,
        }


@dataclass(frozen=True)
class CertifiedSeedBudgetCandidate:
    budget: int
    source: str
    adequate: bool
    capture_rmsd: float
    theorem_handles: tuple[str, ...]
    rigid_seed_family_plan: CertifiedRigidSeedFamilyPlan | None = None
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
        if self.rigid_seed_family_plan is not None:
            if self.rigid_seed_family_plan.adequate_pose_count != self.budget:
                raise ValueError(
                    "rigid_seed_family_plan.adequate_pose_count must match the candidate budget"
                )

    def debug_summary(self) -> dict[str, object]:
        return {
            "budget": self.budget,
            "source": self.source,
            "adequate": self.adequate,
            "capture_rmsd": self.capture_rmsd,
            "probe_rank": self.probe_rank,
            "theorem_handles": self.theorem_handles,
            "witness_handles": self.witness_handles,
            "provenance_note": self.provenance_note,
            "rigid_seed_family_plan": (
                None
                if self.rigid_seed_family_plan is None
                else self.rigid_seed_family_plan.debug_summary()
            ),
        }


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

    @property
    def selected_family_plan(self) -> CertifiedRigidSeedFamilyPlan | None:
        return self.selected_candidate.rigid_seed_family_plan

    def debug_summary(self) -> dict[str, object]:
        return {
            "selected_index": self.selected_index,
            "selected_budget": self.selected_budget,
            "explored_budgets": self.explored_budgets,
            "engineering_probe_pose_cap": self.engineering_probe_pose_cap,
            "engineering_probe_top_k": self.engineering_probe_top_k,
            "theorem_handles": self.theorem_handles,
            "witness_handles": self.witness_handles,
            "candidates": tuple(
                candidate.debug_summary() for candidate in self.candidates
            ),
        }


@dataclass(frozen=True)
class InactiveConformerReturnedPoseWitness:
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()
    note: str | None = None

    def __post_init__(self) -> None:
        _ensure_handle_provenance(
            self.theorem_handles,
            owner="InactiveConformerReturnedPoseWitness",
        )

    @property
    def conformer_active(self) -> bool:
        return False


@dataclass(frozen=True)
class ActiveConformerReturnedPoseWitness:
    coverage_plan: CertifiedConformerCoveragePlan
    cover_rmsd_radius: float
    cover_gap_budget: float
    basin_mu_coord: float
    target_energy_gap: float
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _ensure_handle_provenance(
            self.theorem_handles,
            owner="ActiveConformerReturnedPoseWitness",
        )
        for field_name, value in (
            ("cover_rmsd_radius", self.cover_rmsd_radius),
            ("cover_gap_budget", self.cover_gap_budget),
            ("basin_mu_coord", self.basin_mu_coord),
            ("target_energy_gap", self.target_energy_gap),
        ):
            _ensure_nonnegative(value, field_name=field_name)

    @property
    def conformer_active(self) -> bool:
        return True


@dataclass(frozen=True)
class ActiveConformerEnergyGapWitness:
    coverage_plan: CertifiedConformerCoveragePlan
    cover_rmsd_radius: float
    cover_gap_budget: float
    certified_energy_gap: float
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _ensure_handle_provenance(
            self.theorem_handles,
            owner="ActiveConformerEnergyGapWitness",
        )
        for field_name, value in (
            ("cover_rmsd_radius", self.cover_rmsd_radius),
            ("cover_gap_budget", self.cover_gap_budget),
            ("certified_energy_gap", self.certified_energy_gap),
        ):
            _ensure_nonnegative(value, field_name=field_name)

    @property
    def conformer_active(self) -> bool:
        return True


CertifiedConformerAwareReturnedPoseWitness = (
    InactiveConformerReturnedPoseWitness
    | ActiveConformerReturnedPoseWitness
    | ActiveConformerEnergyGapWitness
)


class ReturnedPoseProofCase(str, Enum):
    NO_POSE = "no_pose"
    RIGID_AMBIGUITY = "rigid_ambiguity"
    CERTIFIED_SINGLETON = "certified_singleton"
    CERTIFIED_AMBIGUITY_SET = "certified_ambiguity_set"
    CERTIFIED_ENERGY_SINGLETON = "certified_energy_singleton"
    CERTIFIED_ENERGY_AMBIGUITY_SET = "certified_energy_ambiguity_set"
    DOWNGRADED = "downgraded"


@dataclass(frozen=True)
class CertifiedReturnedPoseProofPlan:
    proof_case: ReturnedPoseProofCase
    target_rmsd: float
    winner_index: int | None
    support_indices: tuple[int, ...]
    ambiguity_band_width: float | None
    atom_count: int
    dimension: int
    total_gap_budget: float | None
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...] = ()
    conformer_witness: CertifiedConformerAwareReturnedPoseWitness | None = None
    winner_refinement_certificate_present: bool = False
    winner_basin_witness_source: str = "missing"
    winner_conformer_improved: bool = False
    missing_conformer_requirements: tuple[str, ...] = ()
    downgrade_decision: ReturnedPoseContractDecision | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        _ensure_handle_provenance(
            self.theorem_handles,
            owner="CertifiedReturnedPoseProofPlan",
        )
        _ensure_nonnegative(self.target_rmsd, field_name="target_rmsd")
        if self.ambiguity_band_width is not None:
            _ensure_nonnegative(
                self.ambiguity_band_width,
                field_name="ambiguity_band_width",
            )
        for field_name, value in (
            ("atom_count", self.atom_count),
            ("dimension", self.dimension),
        ):
            if value < 0:
                raise ValueError(f"{field_name} must be nonnegative, got {value}")
        if self.total_gap_budget is not None:
            _ensure_nonnegative(self.total_gap_budget, field_name="total_gap_budget")
        if self.winner_index is None and self.support_indices:
            raise ValueError("support_indices require winner_index")
        if (
            self.winner_index is not None
            and self.support_indices
            and self.winner_index not in self.support_indices
        ):
            raise ValueError("support_indices must contain winner_index")
        active_conformer = isinstance(
            self.conformer_witness,
            (ActiveConformerReturnedPoseWitness, ActiveConformerEnergyGapWitness),
        )
        if active_conformer and self.total_gap_budget is None:
            raise ValueError("Active conformer witnesses require total_gap_budget")
        if self.proof_case == ReturnedPoseProofCase.NO_POSE:
            if self.winner_index is not None or self.support_indices:
                raise ValueError("No-pose proof plans must not carry winner state")
            if self.downgrade_decision is not None:
                raise ValueError(
                    "No-pose proof plans must not carry downgrade_decision"
                )
            return
        if self.winner_index is None:
            raise ValueError("Returned-pose proof plans require winner_index")
        if self.proof_case == ReturnedPoseProofCase.CERTIFIED_SINGLETON:
            if not self.winner_singleton:
                raise ValueError(
                    "Certified singleton proof plans require singleton support"
                )
            if self.total_gap_budget is None:
                raise ValueError(
                    "Certified singleton proof plans require total_gap_budget"
                )
            if self.downgrade_decision is not None:
                raise ValueError(
                    "Certified singleton proof plans must not carry downgrade_decision"
                )
        elif self.proof_case == ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON:
            if not self.winner_singleton:
                raise ValueError(
                    "Certified energy singleton proof plans require singleton support"
                )
            if self.total_gap_budget is None:
                raise ValueError(
                    "Certified energy singleton proof plans require total_gap_budget"
                )
            if not isinstance(self.conformer_witness, ActiveConformerEnergyGapWitness):
                raise ValueError(
                    "Certified energy singleton proof plans require an active conformer energy witness"
                )
            if self.downgrade_decision is not None:
                raise ValueError(
                    "Certified energy singleton proof plans must not carry downgrade_decision"
                )
        elif self.proof_case == ReturnedPoseProofCase.CERTIFIED_AMBIGUITY_SET:
            if self.ambiguity_size <= 1:
                raise ValueError(
                    "Certified ambiguity-set proof plans require multiple support indices"
                )
            if self.total_gap_budget is None:
                raise ValueError(
                    "Certified ambiguity-set proof plans require total_gap_budget"
                )
            if not active_conformer:
                raise ValueError(
                    "Certified ambiguity-set proof plans require an active conformer witness"
                )
            if self.downgrade_decision is not None:
                raise ValueError(
                    "Certified ambiguity-set proof plans must not carry downgrade_decision"
                )
        elif self.proof_case == ReturnedPoseProofCase.CERTIFIED_ENERGY_AMBIGUITY_SET:
            if self.ambiguity_size <= 1:
                raise ValueError(
                    "Certified energy ambiguity-set proof plans require multiple support indices"
                )
            if self.total_gap_budget is None:
                raise ValueError(
                    "Certified energy ambiguity-set proof plans require total_gap_budget"
                )
            if not isinstance(self.conformer_witness, ActiveConformerEnergyGapWitness):
                raise ValueError(
                    "Certified energy ambiguity-set proof plans require an active conformer energy witness"
                )
            if self.downgrade_decision is not None:
                raise ValueError(
                    "Certified energy ambiguity-set proof plans must not carry downgrade_decision"
                )
        elif self.proof_case == ReturnedPoseProofCase.RIGID_AMBIGUITY:
            if self.downgrade_decision is not None:
                raise ValueError(
                    "Rigid-ambiguity proof plans must not carry downgrade_decision"
                )
        elif self.proof_case == ReturnedPoseProofCase.DOWNGRADED:
            if self.downgrade_decision is None:
                raise ValueError("Downgraded proof plans require downgrade_decision")
            if self.downgrade_decision not in (
                ReturnedPoseContractDecision.DOWNGRADED_NO_FINAL_SCORE_CERTIFICATE,
                ReturnedPoseContractDecision.DOWNGRADED_NO_REFINEMENT_CERTIFICATE,
                ReturnedPoseContractDecision.DOWNGRADED_CONFORMER_PATH,
            ):
                raise ValueError(
                    "Downgraded proof plans require a runtime-derivable downgrade decision"
                )

    @property
    def ambiguity_size(self) -> int:
        return len(self.support_indices)

    @property
    def has_winner(self) -> bool:
        return self.winner_index is not None

    @property
    def winner_singleton(self) -> bool:
        return self.has_winner and self.ambiguity_size == 1

    @property
    def ambiguity_set_certified(self) -> bool:
        return self.proof_case in (
            ReturnedPoseProofCase.CERTIFIED_AMBIGUITY_SET,
            ReturnedPoseProofCase.CERTIFIED_ENERGY_AMBIGUITY_SET,
        )

    @property
    def decision(self) -> ReturnedPoseContractDecision:
        if self.proof_case == ReturnedPoseProofCase.NO_POSE:
            return ReturnedPoseContractDecision.DOWNGRADED_NO_POSE
        if self.proof_case == ReturnedPoseProofCase.RIGID_AMBIGUITY:
            return ReturnedPoseContractDecision.DOWNGRADED_RIGID_AMBIGUITY
        if self.proof_case == ReturnedPoseProofCase.CERTIFIED_SINGLETON:
            return ReturnedPoseContractDecision.CERTIFIED_TARGET_RMSD
        if self.proof_case == ReturnedPoseProofCase.CERTIFIED_AMBIGUITY_SET:
            return ReturnedPoseContractDecision.CERTIFIED_AMBIGUITY_SET_TARGET_RMSD
        if self.proof_case == ReturnedPoseProofCase.CERTIFIED_ENERGY_SINGLETON:
            return ReturnedPoseContractDecision.CERTIFIED_ENERGY_GAP
        if self.proof_case == ReturnedPoseProofCase.CERTIFIED_ENERGY_AMBIGUITY_SET:
            return ReturnedPoseContractDecision.CERTIFIED_AMBIGUITY_SET_ENERGY_GAP
        assert self.downgrade_decision is not None
        return self.downgrade_decision

    @property
    def conformer_witness_status(self) -> str:
        witness = self.conformer_witness
        if witness is None:
            return "absent"
        if isinstance(witness, ActiveConformerReturnedPoseWitness):
            return "active_complete"
        if isinstance(witness, ActiveConformerEnergyGapWitness):
            return "active_energy_only"
        return "inactive"

    def debug_summary(self) -> dict[str, object]:
        certified_energy_gap = None
        if isinstance(self.conformer_witness, ActiveConformerEnergyGapWitness):
            certified_energy_gap = self.conformer_witness.certified_energy_gap
        return {
            "proof_case": self.proof_case.value,
            "decision": self.decision.name,
            "winner_index": self.winner_index,
            "support_indices": self.support_indices,
            "ambiguity_size": self.ambiguity_size,
            "winner_singleton": self.winner_singleton,
            "total_gap_budget": self.total_gap_budget,
            "conformer_witness_status": self.conformer_witness_status,
            "winner_refinement_certificate_present": self.winner_refinement_certificate_present,
            "winner_basin_witness_source": self.winner_basin_witness_source,
            "winner_conformer_improved": self.winner_conformer_improved,
            "certified_energy_gap": certified_energy_gap,
            "missing_conformer_requirements": self.missing_conformer_requirements,
            "theorem_handles": self.theorem_handles,
            "note": self.note,
        }
