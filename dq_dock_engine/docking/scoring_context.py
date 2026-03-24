from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import jax.tree_util
import numpy as np
from jax.tree_util import register_pytree_node_class

from dq_dock_engine.docking.core import LigandContext
from dq_dock_engine.docking.explicit_water_placement import (
    WaterPlacementGrid,
    generate_water_grid,
    score_water_bridges,
)
from dq_dock_engine.docking.receptor_flexibility import (
    ReceptorConformation,
    conformational_error_radius,
    ensemble_score_upper_bound,
)
from dq_dock_engine.docking.rich_chemistry import build_certified_rich_chemistry_plan
from dq_dock_engine.docking.scoring import (
    CertifiedBatchResult,
    CertifiedRealSpaceEwaldSpec,
    CertifiedRichChemistryPlan,
    CertifiedSoftenedBatchResult,
    score_certified_batch,
    score_certified_rich_chemistry_batch,
    score_certified_softened_lj,
    score_certified_softened_lj_realspace_ewald,
    score_certified_softened_rich_chemistry_batch,
)
from dq_dock_engine.docking_config import ExactChemistryMode


@register_pytree_node_class
@dataclass(frozen=True)
class CertifiedScoringContext:
    exact_chemistry_mode: ExactChemistryMode
    electrostatics: CertifiedRealSpaceEwaldSpec | None = None
    rich_chemistry_plan: CertifiedRichChemistryPlan | None = None
    water_grid: WaterPlacementGrid | None = None
    receptor_conformations: tuple[ReceptorConformation, ...] | None = None

    def tree_flatten(self):
        water_positions = None
        water_grid_spacing = None
        if self.water_grid is not None:
            water_positions = self.water_grid.positions
            water_grid_spacing = self.water_grid.grid_spacing
        receptor_conf_coords: tuple[jnp.ndarray, ...] = ()
        receptor_conf_radii: tuple[jnp.ndarray, ...] = ()
        has_receptor_conformations = self.receptor_conformations is not None
        if self.receptor_conformations is not None:
            receptor_conf_coords = tuple(c.coords for c in self.receptor_conformations)
            receptor_conf_radii = tuple(c.radii for c in self.receptor_conformations)
        children = (
            self.electrostatics,
            self.rich_chemistry_plan,
            water_positions,
            receptor_conf_coords,
            receptor_conf_radii,
        )
        aux_data = (
            self.exact_chemistry_mode,
            water_grid_spacing,
            has_receptor_conformations,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        exact_chemistry_mode, water_grid_spacing, has_receptor_conformations = aux_data
        electrostatics, rich_chemistry_plan, water_positions, rc_coords, rc_radii = (
            children
        )
        water_grid = None
        if water_positions is not None:
            water_grid = WaterPlacementGrid(
                positions=water_positions, grid_spacing=water_grid_spacing
            )
        receptor_conformations = None
        if has_receptor_conformations:
            receptor_conformations = tuple(
                ReceptorConformation(coords=c, radii=r)
                for c, r in zip(rc_coords, rc_radii, strict=True)
            )
        return cls(
            exact_chemistry_mode=exact_chemistry_mode,
            electrostatics=electrostatics,
            rich_chemistry_plan=rich_chemistry_plan,
            water_grid=water_grid,
            receptor_conformations=receptor_conformations,
        )

    @property
    def uses_extended_rich(self) -> bool:
        return self.exact_chemistry_mode == ExactChemistryMode.EXTENDED_RICH

    def analytic_pruning_delta(self) -> float:
        """Batch-size-independent certified pruning delta.

        Dispatches on exact_chemistry_mode:
          EXTENDED_RICH → analytic 3-tier bounds from rich_chemistry_plan
          Otherwise     → 0.0 (softened LJ is the shared base; no other channels)

        Lean: softened_lj_self_approx_zero (tier 1),
              sum_uniformApprox (tier 2),
              base_plus_omitted_uniformApprox (tier 3).
        """
        if not self.uses_extended_rich:
            # Pure softened LJ or Ewald: exact and coarse share the same
            # softened LJ base → δ = 0 (Lean: softened_lj_self_approx_zero)
            return 0.0
        assert self.rich_chemistry_plan is not None
        n_water_bridges = 0
        if self.water_grid is not None:
            n_water_bridges = int(self.water_grid.positions.shape[0])
        return self.rich_chemistry_plan.analytic_total_delta(
            n_water_bridges=n_water_bridges,
        )

    def receptor_subset(
        self, retained_indices: jnp.ndarray
    ) -> "CertifiedScoringContext":
        rec_confs = None
        if self.receptor_conformations is not None:
            rec_confs = tuple(
                ReceptorConformation(
                    coords=c.coords[retained_indices],
                    radii=c.radii[retained_indices],
                )
                for c in self.receptor_conformations
            )
        return CertifiedScoringContext(
            exact_chemistry_mode=self.exact_chemistry_mode,
            electrostatics=(
                None
                if self.electrostatics is None
                else self.electrostatics.receptor_subset(retained_indices)
            ),
            rich_chemistry_plan=(
                None
                if self.rich_chemistry_plan is None
                else self.rich_chemistry_plan.receptor_subset(retained_indices)
            ),
            water_grid=self.water_grid,  # water grid is receptor-global, not subset
            receptor_conformations=rec_confs,
        )

    def _water_bridge_contribution(
        self,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
    ) -> tuple[jnp.ndarray, float]:
        """Compute additive water bridge score and error bound (EWP3).

        Returns (scores (batch,), error_bound).
        """
        if self.water_grid is None or self.rich_chemistry_plan is None:
            batch_size = poses_coords.shape[0]
            return jnp.zeros(batch_size, dtype=jnp.float32), 0.0
        plan = self.rich_chemistry_plan
        # Scatter H-bond strengths to atom positions via anchor indices
        n_rec = receptor_coords.shape[0]
        n_lig = poses_coords.shape[1]
        rec_strengths = jnp.zeros(n_rec, dtype=jnp.float32)
        lig_strengths = jnp.zeros(n_lig, dtype=jnp.float32)
        # Scatter hbond strengths to receptor atom positions via anchor indices
        rec_donor_idx = plan.hbond_receptor_donor.receptor_anchor_indices
        rec_acc_idx = plan.hbond_ligand_donor.receptor_anchor_indices
        if rec_donor_idx.shape[0] > 0:
            rec_strengths = rec_strengths.at[rec_donor_idx].add(
                plan.hbond_receptor_donor.receptor_strengths
            )
        if rec_acc_idx.shape[0] > 0:
            rec_strengths = rec_strengths.at[rec_acc_idx].add(
                plan.hbond_ligand_donor.receptor_strengths
            )
        lig_donor_idx = plan.hbond_ligand_donor.ligand_anchor_indices
        lig_acc_idx = plan.hbond_receptor_donor.ligand_anchor_indices
        if lig_donor_idx.shape[0] > 0:
            lig_strengths = lig_strengths.at[lig_donor_idx].add(
                plan.hbond_ligand_donor.ligand_strengths
            )
        if lig_acc_idx.shape[0] > 0:
            lig_strengths = lig_strengths.at[lig_acc_idx].add(
                plan.hbond_receptor_donor.ligand_strengths
            )
        result = score_water_bridges(
            receptor_coords=receptor_coords,
            poses_coords=poses_coords,
            water_grid=self.water_grid,
            receptor_hbond_strengths=rec_strengths,
            ligand_hbond_strengths=lig_strengths,
        )
        # Water bridges are favorable (negative energy contribution)
        return -result.bridge_scores, result.grid_error_bound

    def _score_physics_only(
        self,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        ligand_radii: jnp.ndarray,
        target_error: float,
        epsilon: float,
    ) -> CertifiedBatchResult:
        """Score physics only — no water bridges, no ensemble.

        This is the inner kernel used by both the full scoring path and
        the fast rigid-only path (conformer search).
        """
        if self.uses_extended_rich:
            if self.rich_chemistry_plan is None:
                raise ValueError(
                    "Extended-rich exact chemistry mode requires a rich chemistry plan"
                )
            return score_certified_rich_chemistry_batch(
                receptor_coords=receptor_coords,
                poses_coords=poses_coords,
                receptor_radii=receptor_radii,
                ligand_radii=ligand_radii,
                rich_chemistry_plan=self.rich_chemistry_plan,
                target_error=target_error,
                epsilon=epsilon,
            )
        return score_certified_batch(
            receptor_coords=receptor_coords,
            poses_coords=poses_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            target_error=target_error,
            epsilon=epsilon,
            electrostatics=self.electrostatics,
        )

    def _score_softened_physics_only(
        self,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        ligand_radii: jnp.ndarray,
        target_error: float,
        epsilon: float,
        softening_radius: float | None,
    ) -> CertifiedSoftenedBatchResult:
        if self.uses_extended_rich:
            if self.rich_chemistry_plan is None:
                raise ValueError(
                    "Extended-rich exact chemistry mode requires a rich chemistry plan"
                )
            return score_certified_softened_rich_chemistry_batch(
                receptor_coords=receptor_coords,
                poses_coords=poses_coords,
                receptor_radii=receptor_radii,
                ligand_radii=ligand_radii,
                rich_chemistry_plan=self.rich_chemistry_plan,
                target_error=target_error,
                epsilon=epsilon,
                softening_radius=softening_radius,
            )
        if self.electrostatics is not None:
            return score_certified_softened_lj_realspace_ewald(
                receptor_coords=receptor_coords,
                poses_coords=poses_coords,
                receptor_radii=receptor_radii,
                ligand_radii=ligand_radii,
                electrostatics=self.electrostatics,
                target_error=target_error,
                epsilon=epsilon,
                softening_radius=softening_radius,
            )
        return score_certified_softened_lj(
            receptor_coords=receptor_coords,
            poses_coords=poses_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            target_error=target_error,
            epsilon=epsilon,
            softening_radius=softening_radius,
        )

    @staticmethod
    def _posewise_receptor_flex_error(
        reference_scores: jnp.ndarray,
        per_conformation_scores: tuple[jnp.ndarray, ...],
    ) -> jnp.ndarray:
        if not per_conformation_scores:
            return jnp.zeros_like(reference_scores)
        diffs = [
            jnp.abs(scores_k - reference_scores) for scores_k in per_conformation_scores
        ]
        return jnp.max(jnp.stack(diffs, axis=0), axis=0)

    def score_rigid_exact_batch(
        self,
        *,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        ligand_radii: jnp.ndarray,
        target_error: float,
        epsilon: float,
    ) -> CertifiedBatchResult:
        """Fast rigid-only scoring — no ensemble, no water bridges.

        Used by conformer search inner loop where CS7 (isometric kinematics)
        guarantees Lipschitz validity with the rigid receptor alone.
        Water bridges are additive (EWP3) and applied once after search.
        """
        return self._score_physics_only(
            receptor_coords,
            poses_coords,
            receptor_radii,
            ligand_radii,
            target_error,
            epsilon,
        )

    def score_rigid_softened_batch(
        self,
        *,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        ligand_radii: jnp.ndarray,
        target_error: float,
        epsilon: float,
        softening_radius: float | None,
    ) -> CertifiedSoftenedBatchResult:
        return self._score_softened_physics_only(
            receptor_coords,
            poses_coords,
            receptor_radii,
            ligand_radii,
            target_error,
            epsilon,
            softening_radius,
        )

    def posewise_receptor_flex_error_exact_batch(
        self,
        *,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        ligand_radii: jnp.ndarray,
        target_error: float,
        epsilon: float,
    ) -> jnp.ndarray:
        if self.receptor_conformations is None:
            return jnp.zeros((poses_coords.shape[0],), dtype=poses_coords.dtype)
        ref_scores = self.score_rigid_exact_batch(
            receptor_coords=receptor_coords,
            poses_coords=poses_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            target_error=target_error,
            epsilon=epsilon,
        ).scores
        conf_scores = tuple(
            self.score_rigid_exact_batch(
                receptor_coords=conf.coords,
                poses_coords=poses_coords,
                receptor_radii=conf.radii,
                ligand_radii=ligand_radii,
                target_error=target_error,
                epsilon=epsilon,
            ).scores
            for conf in self.receptor_conformations
        )
        return self._posewise_receptor_flex_error(ref_scores, conf_scores)

    def posewise_receptor_flex_error_softened_batch(
        self,
        *,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        ligand_radii: jnp.ndarray,
        target_error: float,
        epsilon: float,
        softening_radius: float | None,
    ) -> tuple[jnp.ndarray, jax.Array | float]:
        ref_scores = self.score_rigid_softened_batch(
            receptor_coords=receptor_coords,
            poses_coords=poses_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            target_error=target_error,
            epsilon=epsilon,
            softening_radius=softening_radius,
        )
        flex_delta: jax.Array | float = ref_scores.softening_error_bound
        if self.uses_extended_rich:
            # In extended-rich mode the theorem-backed pruning delta is the analytic
            # composite bound, not the raw summed softening tail bound from the
            # coarse scorer internals. Using the analytic delta keeps the runtime
            # aligned with BlindConformerPipelineRefinements.lean
            # (coarse_rigid_with_flex_and_conformer_lower_bound).
            flex_delta = self.analytic_pruning_delta()
        if self.receptor_conformations is None:
            return (
                jnp.zeros((poses_coords.shape[0],), dtype=poses_coords.dtype),
                flex_delta,
            )
        conf_scores = tuple(
            self.score_rigid_softened_batch(
                receptor_coords=conf.coords,
                poses_coords=poses_coords,
                receptor_radii=conf.radii,
                ligand_radii=ligand_radii,
                target_error=target_error,
                epsilon=epsilon,
                softening_radius=softening_radius,
            ).scores
            for conf in self.receptor_conformations
        )
        return (
            self._posewise_receptor_flex_error(ref_scores.scores, conf_scores),
            flex_delta,
        )

    def score_exact_batch(
        self,
        *,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        ligand_radii: jnp.ndarray,
        target_error: float,
        epsilon: float,
    ) -> CertifiedBatchResult:
        # Step 1: Score reference receptor (physics only)
        ref_result = self._score_physics_only(
            receptor_coords,
            poses_coords,
            receptor_radii,
            ligand_radii,
            target_error,
            epsilon,
        )
        # Step 2: EWP3 — water bridges compose additively (computed once)
        water_scores, water_error = self._water_bridge_contribution(
            receptor_coords, poses_coords
        )
        scores = ref_result.scores + water_scores
        error = ref_result.error_bound + water_error

        # Step 3: RFE1-RFE3 — ensemble scoring with early termination (RFE2)
        if self.receptor_conformations is not None:
            best_scores = scores  # (batch,) — current best per pose
            all_conf_scores = [ref_result.scores]
            for conf in self.receptor_conformations:
                conf_result = self._score_physics_only(
                    conf.coords,
                    poses_coords,
                    conf.radii,
                    ligand_radii,
                    target_error,
                    epsilon,
                )
                conf_scores_with_water = conf_result.scores + water_scores
                all_conf_scores.append(conf_result.scores)
                best_scores = jnp.minimum(best_scores, conf_scores_with_water)
            # RFE1: conformational error radius
            error_radius = conformational_error_radius(
                ref_result.scores, tuple(all_conf_scores[1:])
            )
            scores = best_scores
            error = error + error_radius

        return CertifiedBatchResult(
            scores=scores,
            error_bound=error,
            target_error=target_error,
            cutoff_radius=ref_result.cutoff_radius,
        )

    def score_softened_batch(
        self,
        *,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        ligand_radii: jnp.ndarray,
        target_error: float,
        epsilon: float,
        softening_radius: float | None,
    ) -> CertifiedSoftenedBatchResult:
        if self.uses_extended_rich:
            if self.rich_chemistry_plan is None:
                raise ValueError(
                    "Extended-rich exact chemistry mode requires a rich chemistry plan"
                )
            base = score_certified_softened_rich_chemistry_batch(
                receptor_coords=receptor_coords,
                poses_coords=poses_coords,
                receptor_radii=receptor_radii,
                ligand_radii=ligand_radii,
                rich_chemistry_plan=self.rich_chemistry_plan,
                target_error=target_error,
                epsilon=epsilon,
                softening_radius=softening_radius,
            )
            water_scores, water_error = self._water_bridge_contribution(
                receptor_coords, poses_coords
            )
            return CertifiedSoftenedBatchResult(
                scores=base.scores + water_scores,
                softening_error_bound=base.softening_error_bound + water_error,
                target_error=base.target_error,
                cutoff_radius=base.cutoff_radius,
                softening_radius=base.softening_radius,
            )
        if self.electrostatics is not None:
            return score_certified_softened_lj_realspace_ewald(
                receptor_coords=receptor_coords,
                poses_coords=poses_coords,
                receptor_radii=receptor_radii,
                ligand_radii=ligand_radii,
                electrostatics=self.electrostatics,
                target_error=target_error,
                epsilon=epsilon,
                softening_radius=softening_radius,
            )
        return score_certified_softened_lj(
            receptor_coords=receptor_coords,
            poses_coords=poses_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            target_error=target_error,
            epsilon=epsilon,
            softening_radius=softening_radius,
        )


def _generate_receptor_conformations(
    receptor_coords: np.ndarray,
    receptor_radii: np.ndarray,
    receptor_elements: tuple[str, ...],
    n_conformations: int = 3,
    seed: int = 42,
) -> tuple[ReceptorConformation, ...]:
    """Generate discrete receptor conformations via element-aware perturbation.

    Backbone-like atoms (C, N in non-terminal positions) get small perturbation
    (~0.1Å), side-chain/polar atoms get larger perturbation (~0.3Å). This
    produces a physically reasonable ensemble for RFE1-RFE6 scoring.

    Returns n_conformations alternative conformations (excluding the reference).
    """
    rng = np.random.RandomState(seed)
    # Element-specific perturbation amplitudes (Å)
    # Backbone-like: small; polar/flexible: larger
    flexible_elements = {"O", "N", "S"}
    amplitudes = np.array(
        [0.3 if e.upper() in flexible_elements else 0.1 for e in receptor_elements],
        dtype=np.float32,
    )
    conformations = []
    for _ in range(n_conformations):
        perturbation = rng.randn(*receptor_coords.shape).astype(np.float32)
        perturbation *= amplitudes[:, None]
        conf_coords = receptor_coords + perturbation
        conformations.append(
            ReceptorConformation(
                coords=jnp.array(conf_coords, dtype=jnp.float32),
                radii=jnp.array(receptor_radii, dtype=jnp.float32),
            )
        )
    return tuple(conformations)


def build_certified_scoring_context(
    *,
    exact_chemistry_mode: ExactChemistryMode,
    electrostatics: CertifiedRealSpaceEwaldSpec | None,
    receptor_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    receptor_elements: tuple[str, ...] | None,
    receptor_file: str | Path | None,
    ligand_source_path: str | Path | None,
    ligand_ctx: LigandContext,
    target_electrostatic_error: float,
) -> CertifiedScoringContext:
    if exact_chemistry_mode == ExactChemistryMode.NONE:
        return CertifiedScoringContext(
            exact_chemistry_mode=exact_chemistry_mode,
            electrostatics=electrostatics,
            rich_chemistry_plan=None,
        )
    if receptor_elements is None:
        raise ValueError(
            "Extended-rich exact chemistry mode requires receptor element annotations"
        )
    receptor_charges = (
        np.asarray(electrostatics.receptor_charges, dtype=np.float32)
        if electrostatics is not None
        else np.zeros((receptor_coords.shape[0],), dtype=np.float32)
    )
    receptor_coords_np = np.asarray(receptor_coords, dtype=np.float32)
    receptor_radii_np = np.asarray(receptor_radii, dtype=np.float32)
    rich_plan = build_certified_rich_chemistry_plan(
        receptor_coords=receptor_coords_np,
        receptor_elements=receptor_elements,
        receptor_charges=receptor_charges,
        ligand_ctx=ligand_ctx,
        receptor_file=receptor_file,
        ligand_source_path=ligand_source_path,
        target_electrostatic_error=target_electrostatic_error,
    )
    # Generate water placement grid from receptor/ligand geometry
    ligand_center = np.mean(
        np.asarray(ligand_ctx.base_coords, dtype=np.float32), axis=0
    )
    water_grid = generate_water_grid(
        receptor_coords=receptor_coords_np,
        ligand_center=ligand_center,
    )
    # Generate receptor conformations for RFE1-RFE6 ensemble scoring
    receptor_conformations = _generate_receptor_conformations(
        receptor_coords_np,
        receptor_radii_np,
        receptor_elements,
    )
    return CertifiedScoringContext(
        exact_chemistry_mode=exact_chemistry_mode,
        electrostatics=electrostatics,
        rich_chemistry_plan=rich_plan,
        water_grid=water_grid if water_grid.positions.shape[0] > 0 else None,
        receptor_conformations=receptor_conformations,
    )
