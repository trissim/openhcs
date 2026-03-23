from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from dq_dock_engine.docking.core import DockingBox, LigandContext, PoseVector
from dq_dock_engine.docking.pipeline import (
    PipelineDockingRequest,
    PipelineInitialScores,
    PipelinePoseBatch,
    PipelineRoute,
    _probe_seed_budget_certificate,
    derive_seed_budget,
)
from dq_dock_engine.docking.se3_refinement import (
    RefinementCertificate,
    SE3SpectralCertificate,
)
from dq_dock_engine.docking_config import create_config


def _dummy_ligand_context() -> LigandContext:
    coords = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=jnp.float32)
    return LigandContext(
        base_coords=coords,
        base_radii=jnp.array([1.7, 1.7], dtype=jnp.float32),
        center_of_mass=jnp.array([0.75, 0.0, 0.0], dtype=jnp.float32),
        elements=("C", "C"),
        adjacency=((1,), (0,)),
    )


def _dummy_certificate(*, q: float, n_steps: int) -> RefinementCertificate:
    return RefinementCertificate(
        spectral=SE3SpectralCertificate(
            lmin_param=2.0,
            lmax_param=6.0,
            sigma_max_sq=2.0,
            mu_coord=1.0,
        ),
        q=q,
        initial_gap=10.0,
        target_rmsd=0.5,
        n_steps=n_steps,
        mode=create_config("certified").refinement_certification,
    )


def test_derive_seed_budget_uses_formulaic_zero_step_capture() -> None:
    box_size = jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32)
    confidence = 0.9
    target_rmsd = 0.5
    ligand_radius = 2.0

    expected_trans = (4.0 / 3.0) * math.pi * target_rmsd**3
    rot_radius = target_rmsd / ligand_radius
    expected_rot = (4.0 / 3.0) * math.pi * rot_radius**3
    ratio = (expected_trans * expected_rot) / (1000.0 * (2.0 * math.pi**2))
    expected = math.ceil(math.log(1.0 - confidence) / math.log(1.0 - ratio))

    assert (
        derive_seed_budget(
            confidence=confidence,
            box_size=box_size,
            target_rmsd=target_rmsd,
            ligand_radius=ligand_radius,
        )
        == expected
    )


def test_probe_certificate_expands_capture_radius_and_reduces_budget() -> None:
    box_size = jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32)
    base = derive_seed_budget(
        confidence=0.99,
        box_size=box_size,
        target_rmsd=0.5,
        ligand_radius=2.0,
    )
    probed = derive_seed_budget(
        confidence=0.99,
        box_size=box_size,
        target_rmsd=0.5,
        ligand_radius=2.0,
        probe_certificate=_dummy_certificate(q=0.25, n_steps=2),
    )

    assert probed < base


class _FakeRoute(PipelineRoute):
    def generate_pose_batch(self, request: PipelineDockingRequest) -> PipelinePoseBatch:
        n = request.n_poses_override or 1
        pose_vecs = PoseVector(
            translation=jnp.zeros((n, 3), dtype=jnp.float32),
            quaternion=jnp.tile(
                jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32), (n, 1)
            ),
        )
        return PipelinePoseBatch(request=request, pose_vecs=pose_vecs)

    def score_pose_batch(
        self,
        request: PipelineDockingRequest,
        batched_coords: jnp.ndarray,
        pose_vecs: PoseVector,
    ) -> PipelineInitialScores:
        del request, batched_coords
        n = pose_vecs.translation.shape[0]
        return PipelineInitialScores(final_scores=jnp.arange(n, dtype=jnp.float32))


def test_probe_seed_budget_certificate_overrides_request_budget(monkeypatch) -> None:
    cert = _dummy_certificate(q=0.5, n_steps=4)

    monkeypatch.setattr(
        "dq_dock_engine.docking.pipeline._certified_refinement",
        lambda **kwargs: (
            kwargs["initial_translations"],
            kwargs["initial_quaternions"],
            [cert],
        ),
    )

    request = PipelineDockingRequest(
        protein_coords=jnp.zeros((1, 3), dtype=jnp.float32),
        receptor_radii=jnp.ones((1,), dtype=jnp.float32),
        ligand_ctx=_dummy_ligand_context(),
        box=DockingBox(
            center=jnp.zeros((3,), dtype=jnp.float32),
            size=jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32),
        ),
        key=jax.random.PRNGKey(0),
        config=create_config("certified", confidence=0.99, target_rmsd=0.5),
    )

    updated_request, probe_cert = _probe_seed_budget_certificate(request, _FakeRoute())

    assert probe_cert == cert
    assert updated_request.n_poses_override == derive_seed_budget(
        confidence=0.99,
        box_size=request.box.size,
        target_rmsd=0.5,
        ligand_radius=0.75,
        probe_certificate=cert,
    )
