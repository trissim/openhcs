from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dq_dock_engine.docking import pipeline as docking_pipeline
from dq_dock_engine.docking.certified_runtime_plans import (
    CertifiedPipelineExecutionPlan,
    CertifiedPruningDeltaComponent,
    CertifiedPruningDeltaComponentKind,
    CertifiedPruningDeltaBudget,
    PoseSpecificImprovementBudget,
    PoseSpecificImprovementBudgetFamily,
    PoseSpecificImprovementBudgetKind,
)
from dq_dock_engine.docking.core import (
    CertifiedBindingSite,
    DockingBox,
    LigandContext,
    PoseVector,
)
from dq_dock_engine.docking.formal_handles import (
    rigid_posewise_improvement_budget_theorem_handles,
)
from dq_dock_engine.docking.formal_sampling import (
    derive_certified_rigid_seed_family_plan,
)
from dq_dock_engine.docking.scoring import CertifiedRealSpaceEwaldSpec
from dq_dock_engine.docking.scoring_context import CertifiedScoringContext
from dq_dock_engine.docking_config import (
    CertifiedScoringFamily,
    DockingConfig,
    DockingMode,
    ExactChemistryMode,
    OptimizerBackend,
)


def _make_request(*, target_error: float = 0.001):
    ligand_ctx = LigandContext(
        base_coords=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        base_radii=jnp.array([1.7], dtype=jnp.float32),
        center_of_mass=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        elements=("C",),
    )
    box = DockingBox(
        center=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        size=jnp.array([8.0, 8.0, 8.0], dtype=jnp.float32),
    )
    return docking_pipeline.PipelineDockingRequest(
        protein_coords=jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        receptor_radii=jnp.array([1.7], dtype=jnp.float32),
        ligand_ctx=ligand_ctx,
        box=box,
        n_poses_override=3,
        key=jax.random.PRNGKey(0),
        config=DockingConfig(
            mode=DockingMode.CERTIFIED,
            optimizer_backend=OptimizerBackend.FORMAL,
            certified_scoring_family=CertifiedScoringFamily.LJ,
            target_error=target_error,
            exact_chemistry_mode=ExactChemistryMode.NONE,
        ),
    )


def _make_delta_budget(total_delta: float) -> CertifiedPruningDeltaBudget:
    return CertifiedPruningDeltaBudget.from_components(
        source="test_delta",
        components=(
            CertifiedPruningDeltaComponent(
                label="softening_mismatch",
                kind=CertifiedPruningDeltaComponentKind.SOFTENING_MISMATCH,
                active=total_delta > 0.0,
                delta=total_delta,
                theorem_handles=("LJ10",),
            ),
        ),
        theorem_handles=("LJ10",),
    )


def test_certified_pruning_pass_includes_certified_coarse_delta_in_lower_bound(
    monkeypatch,
) -> None:
    coarse_scores = jnp.array([0.0, 10.0, 0.1], dtype=jnp.float32)
    request = _make_request(target_error=0.1)

    monkeypatch.setattr(
        docking_pipeline,
        "_score_softened_pose_batch",
        lambda request, *, poses_coords, electrostatics, scoring_context: (
            coarse_scores,
            _make_delta_budget(0.1),
            jnp.zeros_like(coarse_scores),
        ),
    )

    execution_plan = docking_pipeline._certified_pruning_pass(
        request,
        poses_coords=jnp.zeros((3, 1, 3), dtype=jnp.float32),
        electrostatics=None,
    )

    assert jnp.array_equal(
        execution_plan.retain_mask,
        jnp.array([True, False, True], dtype=bool),
    )
    assert jnp.array_equal(execution_plan.coarse_scores, coarse_scores)
    assert jnp.allclose(
        execution_plan.lower_bounds,
        jnp.array([-0.1, 9.9, 0.0], dtype=jnp.float32),
    )
    assert execution_plan.pruning_delta_budget.total_delta == 0.1


def test_rigid_local_refinement_plan_uses_seed_family_geometry(monkeypatch) -> None:
    request = _make_request()
    binding_site = CertifiedBindingSite(
        center=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        radius=3.0,
        theorem_handles=("SD10",),
    )
    rigid_seed_family_plan = derive_certified_rigid_seed_family_plan(
        request.box,
        64,
        certified_binding_site=binding_site,
    )
    request = request.with_updates(rigid_seed_family_plan=rigid_seed_family_plan)

    monkeypatch.setattr(
        docking_pipeline,
        "_rigid_local_improvement_bound",
        lambda request, scoring_context, **kwargs: 0.0,
    )

    plan = docking_pipeline._derive_certified_rigid_local_refinement_plan(
        request,
        cast(CertifiedScoringContext, SimpleNamespace()),
    )

    assert plan.translation_cell_width == pytest.approx(
        (2.0 * binding_site.radius) / rigid_seed_family_plan.lattice_resolution
    )


def test_certified_pruning_pass_uses_posewise_softening_correction(monkeypatch) -> None:
    coarse_scores = jnp.array([0.0, 10.0, 0.1], dtype=jnp.float32)
    posewise_correction = jnp.array([0.05, 0.0, 0.15], dtype=jnp.float32)
    request = _make_request(target_error=0.1)

    monkeypatch.setattr(
        docking_pipeline,
        "_score_softened_pose_batch",
        lambda request, *, poses_coords, electrostatics, scoring_context: (
            coarse_scores,
            _make_delta_budget(0.0),
            posewise_correction,
        ),
    )

    execution_plan = docking_pipeline._certified_pruning_pass(
        request,
        poses_coords=jnp.zeros((3, 1, 3), dtype=jnp.float32),
        electrostatics=None,
    )

    assert jnp.array_equal(
        execution_plan.retain_mask,
        jnp.array([True, False, True], dtype=bool),
    )
    assert execution_plan.pruning_delta_budget.total_delta == pytest.approx(0.0)


def test_canonical_retain_mask_falls_back_to_retain_all_on_nonfinite_inputs() -> None:
    scores = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)

    retain_mask, tau, lower_bounds = docking_pipeline._canonical_retain_mask(
        scores,
        delta=0.1,
        additive_correction=jnp.array([0.0, jnp.inf, 0.0], dtype=jnp.float32),
    )

    assert jnp.array_equal(retain_mask, jnp.array([True, True, True], dtype=bool))
    assert tau == pytest.approx(2.0)
    assert jnp.array_equal(lower_bounds, scores)


def test_pose_specific_improvement_budget_allows_rigid_scalar_budget() -> None:
    budget = PoseSpecificImprovementBudget(
        pose_index=7,
        active_torsion_indices=(),
        active_torsion_mask=(),
        per_bond_local_bounds=(),
        total_budget=1.25,
        active_subset_source="rigid_local_refinement",
        theorem_handles=("SH12",),
        kind=PoseSpecificImprovementBudgetKind.RIGID_LOCAL,
    )

    assert budget.kind == PoseSpecificImprovementBudgetKind.RIGID_LOCAL
    assert budget.total_budget == pytest.approx(1.25)


def test_rigid_posewise_improvement_budget_handles_match_reachable_lower_bound_math() -> (
    None
):
    handles = set(rigid_posewise_improvement_budget_theorem_handles())

    assert {
        "LJ24",
        "LJ25",
        "LJ26",
        "LJ27",
        "LJ28",
        "CB5",
        "CB6",
        "CB18",
        "CB19",
        "BCRC4",
        "BCRC5",
        "BCRC6",
        "SH12",
    }.issubset(handles)
    assert {"LJ20", "CB15", "CB16", "CB17"}.isdisjoint(handles)


def test_posewise_rigid_local_improvement_bounds_budget_close_pose_without_clearance_gate() -> (
    None
):
    request = _make_request(target_error=0.1)
    scoring_context = CertifiedScoringContext(
        exact_chemistry_mode=ExactChemistryMode.NONE
    )

    bounds, clearance_safe_mask, clearance = (
        docking_pipeline._posewise_rigid_local_improvement_bounds(
            request,
            scoring_context,
            poses_coords=jnp.zeros((1, 1, 3), dtype=jnp.float32),
            base_translation_step=0.25,
            base_rotation_step_rad=0.0,
            ligand_radius=1.0,
            n_rounds=1,
        )
    )

    assert bounds.shape == (1,)
    assert np.isfinite(bounds[0])
    assert bounds[0] >= 0.0
    assert np.array_equal(clearance_safe_mask, np.array([True], dtype=bool))
    assert clearance.shape == (1,)


def test_posewise_rigid_local_improvement_bounds_supports_realspace_ewald_branch() -> (
    None
):
    request = _make_request(target_error=0.1)
    scoring_context = CertifiedScoringContext(
        exact_chemistry_mode=ExactChemistryMode.NONE,
        electrostatics=CertifiedRealSpaceEwaldSpec(
            receptor_charges=jnp.array([1.0], dtype=jnp.float32),
            ligand_charges=jnp.array([-1.0], dtype=jnp.float32),
            cutoff=12.0,
            alpha=0.2,
            dielectric=4.0,
        ),
    )

    bounds, clearance_safe_mask, clearance = (
        docking_pipeline._posewise_rigid_local_improvement_bounds(
            request,
            scoring_context,
            poses_coords=jnp.array([[[2.5, 0.0, 0.0]]], dtype=jnp.float32),
            base_translation_step=0.25,
            base_rotation_step_rad=0.0,
            ligand_radius=1.0,
            n_rounds=2,
        )
    )

    assert bounds.shape == (1,)
    assert np.isfinite(bounds[0])
    assert bounds[0] >= 0.0
    assert np.array_equal(clearance_safe_mask, np.array([True], dtype=bool))
    assert clearance.shape == (1,)


def test_merge_pruning_delta_budgets_preserves_component_provenance() -> None:
    lhs = CertifiedPruningDeltaBudget.from_components(
        source="lhs",
        components=(
            CertifiedPruningDeltaComponent(
                label="cutoff_tail",
                kind=CertifiedPruningDeltaComponentKind.CUTOFF_TAIL,
                active=True,
                delta=0.1,
                theorem_handles=("A1",),
            ),
        ),
        theorem_handles=("A1",),
    )
    rhs = CertifiedPruningDeltaBudget.from_components(
        source="rhs",
        components=(
            CertifiedPruningDeltaComponent(
                label="water_mediated",
                kind=CertifiedPruningDeltaComponentKind.WATER_MEDIATED,
                active=True,
                delta=0.2,
                theorem_handles=("B1",),
            ),
        ),
        theorem_handles=("B1",),
    )

    merged = docking_pipeline._merge_pruning_delta_budgets(lhs, rhs)

    assert merged.total_delta == pytest.approx(0.3)
    assert merged.cutoff_tail_delta == pytest.approx(0.1)
    assert merged.omitted_value_delta == pytest.approx(0.2)
    assert {component.label for component in merged.active_components} == {
        "cutoff_tail",
        "water_mediated",
    }


def test_pruning_budget_requires_explicit_authoritative_components() -> None:
    with pytest.raises(ValueError, match="requires explicit authoritative components"):
        CertifiedPruningDeltaBudget(
            source="legacy",
            shared_base_delta=0.0,
            cutoff_tail_delta=0.0,
            omitted_value_delta=0.0,
            softening_mismatch_delta=0.1,
            total_delta=0.1,
            theorem_handles=("LJ10",),
        )


def test_pruning_budget_debug_summary_exposes_active_component_map() -> None:
    budget = CertifiedPruningDeltaBudget.from_components(
        source="debug",
        components=(
            CertifiedPruningDeltaComponent(
                label="softening_mismatch",
                kind=CertifiedPruningDeltaComponentKind.SOFTENING_MISMATCH,
                active=True,
                delta=0.25,
                theorem_handles=("LJ10",),
            ),
        ),
        theorem_handles=("LJ10",),
    )

    summary = budget.debug_summary()

    assert summary["total_delta"] == pytest.approx(0.25)
    assert budget.component_delta_map == {"softening_mismatch": 0.25}


def test_score_softened_pose_batch_uses_softening_bound_for_base_physics() -> None:
    request = _make_request(target_error=0.1)

    class FakeBaseContext:
        uses_extended_rich = False

        def pruning_delta_budget(
            self,
            *,
            softening_error_bound=None,
            softening_radius=None,
            use_posewise_softening=False,
        ):
            del softening_radius
            assert softening_error_bound is not None
            return _make_delta_budget(
                0.0 if use_posewise_softening else float(softening_error_bound)
            )

        def score_softened_batch(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                scores=jnp.array([1.0, 2.0], dtype=jnp.float32),
                softening_error_bound=jnp.asarray(0.125, dtype=jnp.float32),
                posewise_softening_error_bound=jnp.array(
                    [0.125, 0.025], dtype=jnp.float32
                ),
            )

    scores, delta_budget, posewise_delta = docking_pipeline._score_softened_pose_batch(
        cast(Any, request),
        poses_coords=jnp.zeros((2, 1, 3), dtype=jnp.float32),
        electrostatics=None,
        scoring_context=cast(CertifiedScoringContext, FakeBaseContext()),
    )

    assert jnp.array_equal(scores, jnp.array([1.0, 2.0], dtype=jnp.float32))
    assert jnp.array_equal(posewise_delta, jnp.array([0.125, 0.025], dtype=jnp.float32))
    assert delta_budget.total_delta == pytest.approx(0.0)
    assert delta_budget.softening_mismatch_delta == pytest.approx(0.0)


def test_certified_pruning_pass_uses_rigid_posewise_improvement_correction(
    monkeypatch,
) -> None:
    coarse_scores = jnp.array([0.0, 0.2, 0.3], dtype=jnp.float32)
    request = _make_request(target_error=0.1)

    monkeypatch.setattr(
        docking_pipeline,
        "_score_softened_pose_batch",
        lambda request, *, poses_coords, electrostatics, scoring_context: (
            coarse_scores,
            _make_delta_budget(0.0),
            jnp.zeros_like(coarse_scores),
        ),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_posewise_rigid_improvement_budget_family",
        lambda request, scoring_context, *, poses_coords, **kwargs: (
            PoseSpecificImprovementBudgetFamily(
                budgets=(
                    PoseSpecificImprovementBudget(
                        pose_index=1,
                        active_torsion_indices=(),
                        active_torsion_mask=(),
                        per_bond_local_bounds=(),
                        total_budget=0.25,
                        active_subset_source="rigid_local_refinement",
                        theorem_handles=("SH12",),
                        kind=PoseSpecificImprovementBudgetKind.RIGID_LOCAL,
                    ),
                ),
                theorem_handles=("SH12",),
            ),
            np.array(
                [0.25 if i == 1 else 0.0 for i in range(int(poses_coords.shape[0]))],
                dtype=np.float32,
            ),
            np.ones((int(poses_coords.shape[0]),), dtype=bool),
            np.ones((int(poses_coords.shape[0]),), dtype=np.float32),
        ),
    )
    rigid_improvement_context = SimpleNamespace(
        score_rigid_exact_batch=lambda **kwargs: SimpleNamespace(
            scores=jnp.asarray(
                [0.0] * int(kwargs["poses_coords"].shape[0]), dtype=jnp.float32
            ),
            error_bound=jnp.asarray(0.0, dtype=jnp.float32),
        )
    )

    execution_plan = docking_pipeline._certified_pruning_pass(
        request,
        poses_coords=jnp.zeros((3, 1, 3), dtype=jnp.float32),
        electrostatics=None,
        rigid_improvement_scoring_context=cast(
            CertifiedScoringContext, rigid_improvement_context
        ),
        rigid_refinement_plan=cast(
            Any,
            SimpleNamespace(
                base_translation_step=0.25,
                base_rotation_step_rad=0.1,
                ligand_radius=1.0,
                n_search_rounds=1,
                local_improvement_bound=0.25,
            ),
        ),
    )

    assert jnp.array_equal(
        execution_plan.retain_mask,
        jnp.array([True, True, False], dtype=bool),
    )
    assert execution_plan.improvement_budget_family is not None
    assert execution_plan.improvement_budget_family.totals == pytest.approx((0.25,))


def test_certified_pruning_pass_retains_rigid_clearance_unsafe_poses(
    monkeypatch,
) -> None:
    coarse_scores = jnp.array([0.0, 0.2, 0.3], dtype=jnp.float32)
    request = _make_request(target_error=0.1)

    monkeypatch.setattr(
        docking_pipeline,
        "_score_softened_pose_batch",
        lambda request, *, poses_coords, electrostatics, scoring_context: (
            coarse_scores,
            _make_delta_budget(0.0),
            jnp.zeros_like(coarse_scores),
        ),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_posewise_rigid_improvement_budget_family",
        lambda request, scoring_context, *, poses_coords, **kwargs: (
            PoseSpecificImprovementBudgetFamily(
                budgets=(),
                theorem_handles=("SH12",),
            ),
            np.zeros((int(poses_coords.shape[0]),), dtype=np.float32),
            np.array(
                [False if i == 1 else True for i in range(int(poses_coords.shape[0]))],
                dtype=bool,
            ),
            np.array(
                [-0.1 if i == 1 else 1.0 for i in range(int(poses_coords.shape[0]))],
                dtype=np.float32,
            ),
        ),
    )
    rigid_improvement_context = SimpleNamespace(
        score_rigid_exact_batch=lambda **kwargs: SimpleNamespace(
            scores=jnp.asarray(
                [0.0] * int(kwargs["poses_coords"].shape[0]), dtype=jnp.float32
            ),
            error_bound=jnp.asarray(0.0, dtype=jnp.float32),
        )
    )

    execution_plan = docking_pipeline._certified_pruning_pass(
        request,
        poses_coords=jnp.zeros((3, 1, 3), dtype=jnp.float32),
        electrostatics=None,
        rigid_improvement_scoring_context=cast(
            CertifiedScoringContext, rigid_improvement_context
        ),
        rigid_refinement_plan=cast(
            Any,
            SimpleNamespace(
                base_translation_step=0.25,
                base_rotation_step_rad=0.1,
                ligand_radius=1.0,
                n_search_rounds=1,
                local_improvement_bound=0.25,
            ),
        ),
    )

    assert jnp.array_equal(
        execution_plan.retain_mask,
        jnp.array([True, True, False], dtype=bool),
    )
    assert {"BCPO1", "BCRP1", "BCRC3"}.isdisjoint(set(execution_plan.theorem_handles))


def test_score_softened_pose_batch_uses_analytic_delta_for_rich_pruning_context() -> (
    None
):
    request = SimpleNamespace(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.array([1.7], dtype=jnp.float32),
        ligand_ctx=SimpleNamespace(base_radii=jnp.array([1.7], dtype=jnp.float32)),
        target_error=0.1,
        softening_radius=9.0,
    )

    class FakeRichPruningContext:
        uses_extended_rich = True

        def pruning_delta_budget(
            self,
            *,
            softening_error_bound=None,
            softening_radius=None,
            use_posewise_softening=False,
        ):
            del softening_error_bound, softening_radius, use_posewise_softening
            return _make_delta_budget(99.0)

        def score_softened_batch(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                scores=jnp.array([3.0], dtype=jnp.float32),
                softening_error_bound=jnp.asarray(0.25, dtype=jnp.float32),
                posewise_softening_error_bound=jnp.array([0.25], dtype=jnp.float32),
            )

    scores, delta_budget, posewise_delta = docking_pipeline._score_softened_pose_batch(
        cast(Any, request),
        poses_coords=jnp.zeros((1, 1, 3), dtype=jnp.float32),
        electrostatics=None,
        scoring_context=cast(CertifiedScoringContext, FakeRichPruningContext()),
    )

    assert jnp.array_equal(scores, jnp.array([3.0], dtype=jnp.float32))
    assert delta_budget.total_delta == pytest.approx(99.0)
    assert jnp.array_equal(posewise_delta, jnp.array([0.0], dtype=jnp.float32))


def test_score_softened_pose_batch_uses_batch_exact_delta_for_pruning_context() -> None:
    request = SimpleNamespace(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.array([1.7], dtype=jnp.float32),
        ligand_ctx=SimpleNamespace(base_radii=jnp.array([1.7], dtype=jnp.float32)),
        target_error=0.1,
        softening_radius=1.5,
    )

    class FakeTightPruningContext:
        uses_extended_rich = True

        def uses_batch_pruning_delta(self):
            return True

        def pruning_softening_matches_exact(self, softening_radius):
            return softening_radius == 1.5

        def pruning_delta_budget(
            self,
            *,
            softening_error_bound=None,
            softening_radius=None,
            use_posewise_softening=False,
        ):
            assert softening_error_bound is not None
            assert softening_radius == 1.5
            return CertifiedPruningDeltaBudget.from_components(
                source="rich_batch_exact_pruning_delta",
                components=(
                    CertifiedPruningDeltaComponent(
                        label="batch_exact_delta",
                        kind=CertifiedPruningDeltaComponentKind.SOFTENING_MISMATCH,
                        active=(not use_posewise_softening)
                        and float(softening_error_bound) > 0.0,
                        delta=(
                            0.0
                            if use_posewise_softening
                            else float(softening_error_bound)
                        ),
                        theorem_handles=("RPG6",),
                    ),
                ),
                theorem_handles=("RPG6",),
            )

        def score_softened_batch(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                scores=jnp.array([3.0], dtype=jnp.float32),
                softening_error_bound=jnp.asarray(0.25, dtype=jnp.float32),
                posewise_softening_error_bound=jnp.array([0.25], dtype=jnp.float32),
            )

    scores, delta_budget, posewise_delta = docking_pipeline._score_softened_pose_batch(
        cast(Any, request),
        poses_coords=jnp.zeros((1, 1, 3), dtype=jnp.float32),
        electrostatics=None,
        scoring_context=cast(CertifiedScoringContext, FakeTightPruningContext()),
    )

    assert jnp.array_equal(scores, jnp.array([3.0], dtype=jnp.float32))
    assert delta_budget.total_delta == pytest.approx(0.0)
    assert jnp.array_equal(posewise_delta, jnp.array([0.0], dtype=jnp.float32))


def test_certified_pipeline_route_scores_only_valid_survivors(monkeypatch) -> None:
    request = _make_request()
    pose_vecs = PoseVector(
        translation=jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32
        ),
        quaternion=jnp.tile(
            jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32), (3, 1)
        ),
    )
    batched_coords = jnp.array(
        [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]], dtype=jnp.float32
    )

    monkeypatch.setattr(
        docking_pipeline,
        "_certified_pruning_pass",
        lambda request, *, poses_coords, electrostatics, **kwargs: (
            CertifiedPipelineExecutionPlan(
                coarse_scores=jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32),
                lower_bounds=jnp.array([0.0, 0.999, 1.999], dtype=jnp.float32),
                tau=0.001,
                retain_mask=jnp.array([True, False, False]),
                pruning_delta_budget=_make_delta_budget(0.001),
                theorem_handles=("TK11",),
            )
        ),
    )

    def fake_score_exact(
        request, *, poses_coords, electrostatics, scoring_context=None
    ):
        del request, electrostatics, scoring_context
        assert poses_coords.shape[0] == 1
        return jnp.array([3.14], dtype=jnp.float32)

    monkeypatch.setattr(docking_pipeline, "_score_exact_pose_batch", fake_score_exact)
    monkeypatch.setattr(
        docking_pipeline,
        "_derive_certified_rigid_local_refinement_plan",
        lambda request, scoring_context: SimpleNamespace(local_improvement_bound=0.0),
    )

    initial_scores = docking_pipeline.CertifiedPipelineRoute().score_pose_batch(
        request,
        batched_coords,
        pose_vecs,
    )

    assert jnp.allclose(
        initial_scores.final_scores,
        jnp.array([3.14, 1e6, 1e6], dtype=jnp.float32),
    )
    assert initial_scores.survivor_pose_vecs is not None
    assert initial_scores.survivor_pose_vecs.translation.shape[0] == 1
    assert initial_scores.survivor_exact_scores is not None
    assert jnp.allclose(
        initial_scores.survivor_exact_scores,
        jnp.array([3.14], dtype=jnp.float32),
    )
    assert initial_scores.execution_plan is not None


def test_certified_pipeline_route_optimizes_all_certified_survivors() -> None:
    request = _make_request()
    survivor_pose_vecs = PoseVector(
        translation=jnp.array(
            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
            dtype=jnp.float32,
        ),
        quaternion=jnp.tile(
            jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32), (3, 1)
        ),
    )
    initial_scores = docking_pipeline.PipelineInitialScores(
        final_scores=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        survivor_pose_vecs=survivor_pose_vecs,
        survivor_exact_scores=jnp.array([0.4, 0.1, 0.2], dtype=jnp.float32),
        valid_survivor_mask=jnp.ones((3,), dtype=bool),
    )

    translations, quaternions, n_to_opt = (
        docking_pipeline.CertifiedPipelineRoute().optimization_inputs(
            request,
            pose_vecs=survivor_pose_vecs,
            initial_scores=initial_scores,
        )
    )

    assert n_to_opt == 3
    assert jnp.allclose(
        translations,
        jnp.array(
            [[20.0, 0.0, 0.0], [30.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=jnp.float32,
        ),
    )
    assert quaternions.shape == (3, 4)


def test_score_exact_pose_batch_scores_large_pose_sets_in_chunks(monkeypatch) -> None:
    request = _make_request()
    chunk_sizes: list[int] = []

    class FakeScoringContext:
        def score_exact_batch(self, **kwargs):
            chunk = kwargs["poses_coords"]
            chunk_sizes.append(int(chunk.shape[0]))
            return SimpleNamespace(scores=chunk[:, 0, 0])

    monkeypatch.setattr(
        docking_pipeline,
        "resolve_request_scoring_context",
        lambda request, *, engine=None: FakeScoringContext(),
    )

    poses_coords = jnp.arange(5, dtype=jnp.float32).reshape(5, 1, 1)
    exact_scores = docking_pipeline._score_exact_pose_batch(
        request,
        poses_coords=poses_coords,
        electrostatics=None,
        chunk_size=2,
    )

    assert chunk_sizes == [2, 2, 1]
    assert jnp.allclose(exact_scores, jnp.array([0, 1, 2, 3, 4], dtype=jnp.float32))


def test_certified_pipeline_route_keeps_all_certified_survivors(
    monkeypatch,
) -> None:
    request = _make_request()
    n_survivors = 7
    pose_vecs = PoseVector(
        translation=jnp.zeros((n_survivors, 3), dtype=jnp.float32),
        quaternion=jnp.tile(
            jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
            (n_survivors, 1),
        ),
    )
    batched_coords = jnp.zeros((n_survivors, 1, 3), dtype=jnp.float32)

    monkeypatch.setattr(
        docking_pipeline,
        "_certified_pruning_pass",
        lambda request, *, poses_coords, electrostatics, **kwargs: (
            CertifiedPipelineExecutionPlan(
                coarse_scores=jnp.zeros((n_survivors,), dtype=jnp.float32),
                lower_bounds=jnp.zeros((n_survivors,), dtype=jnp.float32),
                tau=0.001,
                retain_mask=jnp.ones((n_survivors,), dtype=bool),
                pruning_delta_budget=_make_delta_budget(0.001),
                theorem_handles=("TK11",),
            )
        ),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_score_exact_pose_batch",
        lambda request, *, poses_coords, electrostatics, scoring_context=None: (
            jnp.arange(poses_coords.shape[0], dtype=jnp.float32)
        ),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_derive_certified_rigid_local_refinement_plan",
        lambda request, scoring_context: SimpleNamespace(local_improvement_bound=0.0),
    )

    initial_scores = docking_pipeline.CertifiedPipelineRoute().score_pose_batch(
        request,
        batched_coords,
        pose_vecs,
    )

    assert initial_scores.survivor_pose_vecs is not None
    assert initial_scores.survivor_pose_vecs.translation.shape[0] == n_survivors
    assert initial_scores.survivor_exact_scores is not None
    assert initial_scores.survivor_exact_scores.shape[0] == n_survivors


def test_certified_pipeline_route_exact_repruning_retains_uncertified_clearance_pose(
    monkeypatch,
) -> None:
    request = _make_request()
    pose_vecs = PoseVector(
        translation=jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float32
        ),
        quaternion=jnp.tile(
            jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32), (3, 1)
        ),
    )
    batched_coords = jnp.array(
        [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]], dtype=jnp.float32
    )

    class FakeImprovementFamily:
        def totals_array(self):
            return np.zeros((3,), dtype=np.float32)

    class FakeFullContext:
        uses_extended_rich = False
        receptor_conformations = None

        def pruning_context(self):
            return self

        def optimization_context(self):
            return self

    monkeypatch.setattr(
        docking_pipeline,
        "resolve_request_scoring_context",
        lambda request, *, engine=None: FakeFullContext(),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_certified_pruning_pass",
        lambda request, *, poses_coords, electrostatics, **kwargs: (
            CertifiedPipelineExecutionPlan(
                coarse_scores=jnp.array([0.0, 0.1, 0.2], dtype=jnp.float32),
                lower_bounds=jnp.array([0.0, 0.1, 0.2], dtype=jnp.float32),
                tau=0.0,
                retain_mask=jnp.array([True, True, True], dtype=bool),
                pruning_delta_budget=_make_delta_budget(0.0),
                theorem_handles=("TK11",),
            )
        ),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_request_uses_conformer_search",
        lambda request: True,
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_request_rotatable_bonds",
        lambda request: (),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_derive_certified_rigid_local_refinement_plan",
        lambda request, scoring_context: SimpleNamespace(
            local_improvement_bound=0.0,
            base_translation_step=0.5,
            base_rotation_step_rad=0.25,
            ligand_radius=1.0,
            n_search_rounds=1,
        ),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_score_rigid_exact_pose_batch",
        lambda request, *, poses_coords, electrostatics, scoring_context=None: (
            jnp.array([0.0, 0.2, 0.3], dtype=jnp.float32)
        ),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_build_conformer_search_config",
        lambda request, *, rotatable_bonds: SimpleNamespace(per_bond_lipschitz=()),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_conformer_local_improvement_bounds",
        lambda request, rotatable_bonds, per_bond_lipschitz: jnp.zeros(
            (0,), dtype=jnp.float32
        ),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_posewise_improvement_budget_family",
        lambda *args, **kwargs: FakeImprovementFamily(),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_posewise_rigid_local_improvement_bounds",
        lambda request, scoring_context, *, poses_coords, **kwargs: (
            np.zeros((3,), dtype=np.float32),
            np.array([True, False, True], dtype=bool),
            np.array([1.0, -0.1, 1.0], dtype=np.float32),
        ),
    )

    def fake_score_exact(
        request, *, poses_coords, electrostatics, scoring_context=None
    ):
        del request, electrostatics, scoring_context
        assert poses_coords.shape[0] == 2
        return jnp.array([0.0, 0.2], dtype=jnp.float32)

    monkeypatch.setattr(docking_pipeline, "_score_exact_pose_batch", fake_score_exact)

    initial_scores = docking_pipeline.CertifiedPipelineRoute().score_pose_batch(
        request,
        batched_coords,
        pose_vecs,
    )

    assert initial_scores.survivor_pose_vecs is not None
    assert jnp.allclose(
        initial_scores.survivor_pose_vecs.translation,
        jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32),
    )
    assert initial_scores.survivor_exact_scores is not None
    assert jnp.allclose(
        initial_scores.survivor_exact_scores,
        jnp.array([0.0, 0.2], dtype=jnp.float32),
    )
    assert jnp.allclose(
        initial_scores.final_scores,
        jnp.array([0.0, 0.2, 1e6], dtype=jnp.float32),
    )


def test_certified_pipeline_route_uses_pruning_context_for_pruning(
    monkeypatch,
) -> None:
    request = _make_request()
    pose_vecs = PoseVector(
        translation=jnp.zeros((1, 3), dtype=jnp.float32),
        quaternion=jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
    )
    batched_coords = jnp.zeros((1, 1, 3), dtype=jnp.float32)
    seen: dict[str, object] = {}

    class FakeOptimizationContext:
        uses_extended_rich = False

    class FakePruningContext:
        uses_extended_rich = True

    class FakeFullContext:
        uses_extended_rich = True

        def optimization_context(self):
            return FakeOptimizationContext()

        def pruning_context(self):
            return FakePruningContext()

        def score_exact_batch(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                scores=jnp.array([0.0], dtype=jnp.float32),
                error_bound=jnp.asarray(0.0, dtype=jnp.float32),
            )

        def score_flip_disambiguation_batch(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                scores=jnp.array([0.0], dtype=jnp.float32),
                error_bound=jnp.asarray(0.0, dtype=jnp.float32),
            )

    monkeypatch.setattr(
        docking_pipeline,
        "resolve_request_scoring_context",
        lambda request, *, engine=None: FakeFullContext(),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_certified_pruning_pass",
        lambda request, *, poses_coords, electrostatics, scoring_context, **kwargs: (
            CertifiedPipelineExecutionPlan(
                coarse_scores=jnp.array([0.0], dtype=jnp.float32),
                lower_bounds=jnp.array([0.0], dtype=jnp.float32),
                tau=0.0,
                retain_mask=(
                    seen.setdefault("pruning_context", scoring_context) is not None
                    and jnp.array([True], dtype=bool)
                ),
                pruning_delta_budget=_make_delta_budget(0.0),
                theorem_handles=("TK11",),
            )
        ),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_score_exact_pose_batch",
        lambda request, *, poses_coords, electrostatics, scoring_context=None: (
            jnp.array([0.0], dtype=jnp.float32)
        ),
    )
    monkeypatch.setattr(
        docking_pipeline,
        "_derive_certified_rigid_local_refinement_plan",
        lambda request, scoring_context: SimpleNamespace(local_improvement_bound=0.0),
    )

    initial_scores = docking_pipeline.CertifiedPipelineRoute().score_pose_batch(
        request,
        batched_coords,
        pose_vecs,
    )

    assert seen["pruning_context"] is None
    assert initial_scores.survivor_exact_scores is not None
