import numpy as np
import jax.numpy as jnp
from pathlib import Path
import tempfile

from dq_dock_engine.docking.scoring import (
    CertifiedRealSpaceEwaldSpec,
    certified_realspace_ewald_error_bound,
    route_scoring,
    score_certified_batch,
    score_certified_lj,
    score_certified_lj_realspace_ewald,
)
from dq_dock_engine.docking.core import ScoringEngine
from dq_dock_engine.docking.core import LigandContext
from dq_dock_engine.docking.core import DockingBox
from dq_dock_engine.docking.pipeline import _resolve_route_scoring_electrostatics
from dq_dock_engine.docking.pipeline import run_docking_pipeline
from dq_dock_engine.docking.charges import ChargeMethod
from dq_dock_engine.docking_config import (
    CertifiedScoringFamily,
    DockingConfig,
    DockingMode,
    OptimizerBackend,
)
import jax
from dq_dock_engine.benchmark.benchmark_pdb import prepare_protein
from dq_dock_engine.benchmark.benchmark_pdb import prepare_ligand


def test_realspace_ewald_error_bound_rejects_invalid_inputs() -> None:
    try:
        certified_realspace_ewald_error_bound(0.0, 0.2, 1.0)
        assert False, "expected cutoff validation"
    except ValueError:
        pass

    try:
        certified_realspace_ewald_error_bound(12.0, 0.0, 1.0)
        assert False, "expected alpha validation"
    except ValueError:
        pass


def test_composite_score_matches_lj_when_charges_zero() -> None:
    receptor_coords = jnp.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    poses_coords = jnp.array(
        [
            [[3.5, 0.0, 0.0], [4.5, 0.0, 0.0]],
            [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ]
    )
    receptor_radii = jnp.array([1.7, 1.7])
    ligand_radii = jnp.array([1.7, 1.7])
    receptor_charges = jnp.zeros((2,))
    ligand_charges = jnp.zeros((2,))

    electrostatics = CertifiedRealSpaceEwaldSpec(
        receptor_charges=receptor_charges,
        ligand_charges=ligand_charges,
    )
    lj_scores, lj_error = score_certified_lj(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
    )
    composite_scores, composite_error = score_certified_lj_realspace_ewald(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        electrostatics=electrostatics,
    )

    np.testing.assert_allclose(np.asarray(composite_scores), np.asarray(lj_scores))
    assert abs(composite_error - lj_error) < 1e-10


def test_composite_score_stays_finite_for_overlapping_atoms() -> None:
    receptor_coords = jnp.array([[0.0, 0.0, 0.0]])
    poses_coords = jnp.array([[[0.0, 0.0, 0.0]]])
    receptor_radii = jnp.array([1.7])
    ligand_radii = jnp.array([1.7])
    receptor_charges = jnp.array([1.0])
    ligand_charges = jnp.array([-1.0])
    electrostatics = CertifiedRealSpaceEwaldSpec(
        receptor_charges=receptor_charges,
        ligand_charges=ligand_charges,
    )

    scores, error_bound = score_certified_lj_realspace_ewald(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        electrostatics=electrostatics,
    )

    assert np.isfinite(float(scores[0]))
    assert np.isfinite(error_bound)


def test_route_scoring_supports_certified_realspace_ewald_engine() -> None:
    receptor_coords = jnp.array([[0.0, 0.0, 0.0]])
    poses_coords = jnp.array([[[2.0, 0.0, 0.0]]])
    receptor_radii = jnp.array([1.7])
    ligand_radii = jnp.array([1.7])
    electrostatics = CertifiedRealSpaceEwaldSpec(
        receptor_charges=jnp.array([1.0]),
        ligand_charges=jnp.array([-1.0]),
    )

    scores = route_scoring(
        ScoringEngine.CERTIFIED_LJ_REALSPACE_EWALD,
        receptor_coords=receptor_coords,
        poses_coords=poses_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        electrostatics=electrostatics,
    )

    assert scores.shape == (1,)
    assert np.isfinite(float(scores[0]))


def test_pipeline_resolves_simple_charges_for_realspace_ewald_engine() -> None:
    ligand_ctx = LigandContext(
        base_coords=jnp.array([[0.0, 0.0, 0.0]]),
        base_radii=jnp.array([1.7]),
        center_of_mass=jnp.array([0.0, 0.0, 0.0]),
        elements=("O",),
        charges=None,
    )

    electrostatics = _resolve_route_scoring_electrostatics(
        ScoringEngine.CERTIFIED_LJ_REALSPACE_EWALD,
        ligand_ctx,
        ("N",),
        ChargeMethod.SIMPLE,
        receptor_file=None,
    )

    assert electrostatics is not None
    np.testing.assert_allclose(
        np.asarray(electrostatics.receptor_charges), np.asarray([-0.3])
    )
    np.testing.assert_allclose(
        np.asarray(electrostatics.ligand_charges), np.asarray([-0.4])
    )


def test_certified_config_defaults_to_realspace_ewald_family() -> None:
    config = DockingConfig(
        mode=DockingMode.CERTIFIED,
        optimizer_backend=OptimizerBackend.FORMAL,
    )
    assert config.certified_scoring_family == CertifiedScoringFamily.LJ_REALSPACE_EWALD


def test_score_certified_batch_uses_composite_path_when_charges_present() -> None:
    receptor_coords = jnp.array([[0.0, 0.0, 0.0]])
    poses_coords = jnp.array([[[2.0, 0.0, 0.0]]])
    receptor_radii = jnp.array([1.7])
    ligand_radii = jnp.array([1.7])
    electrostatics = CertifiedRealSpaceEwaldSpec(
        receptor_charges=jnp.array([1.0]),
        ligand_charges=jnp.array([-1.0]),
    )

    batch = score_certified_batch(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        electrostatics=electrostatics,
    )

    assert batch.scores.shape == (1,)
    assert np.isfinite(float(batch.scores[0]))
    assert np.isfinite(batch.error_bound)


def test_certified_pipeline_accepts_realspace_ewald_scoring_family() -> None:
    protein_coords = jnp.array([[0.0, 0.0, 0.0]])
    receptor_radii = jnp.array([1.7])
    ligand_ctx = LigandContext(
        base_coords=jnp.array([[0.0, 0.0, 0.0]]),
        base_radii=jnp.array([1.7]),
        center_of_mass=jnp.array([0.0, 0.0, 0.0]),
        elements=("O",),
        charges=None,
    )
    box = DockingBox(
        center=jnp.array([2.0, 0.0, 0.0]),
        size=jnp.array([2.0, 2.0, 2.0]),
    )
    config = DockingConfig(
        mode=DockingMode.CERTIFIED,
        optimizer_backend=OptimizerBackend.FORMAL,
        certified_scoring_family=CertifiedScoringFamily.LJ_REALSPACE_EWALD,
    )

    poses, _ = run_docking_pipeline(
        protein_coords=protein_coords,
        receptor_radii=receptor_radii,
        ligand_ctx=ligand_ctx,
        box=box,
        n_poses=1,
        engine=ScoringEngine.INTERNAL_LJ,
        key=jax.random.PRNGKey(0),
        receptor_elements=("N",),
        charge_method=ChargeMethod.SIMPLE,
        config=config,
        top_k=1,
        optimize=True,
        n_opt_steps=1,
        top_k_to_optimize=1,
        include_native=False,
    )

    assert len(poses) == 1
    assert np.isfinite(poses[0].energy)


def test_prepare_protein_filters_non_a_altlocs() -> None:
    pdb_text = (
        "ATOM      1  N  AARG A 145      13.530  27.616  39.774  0.50 34.59           N  \n"
        "ATOM      2  N  BARG A 145      13.400  28.109  40.075  0.50 34.17           N  \n"
        "ATOM      3  CA  ARG A 145      14.521  28.624  40.315  1.00 26.15           C  \n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = Path(tmpdir) / "protein.pdb"
        pdb_path.write_text(pdb_text)
        protein_path = prepare_protein(pdb_path)
        lines = protein_path.read_text().splitlines()

    assert len(lines) == 2
    assert "AARG" in lines[0]
    assert "BARG" not in "\n".join(lines)


def test_prepare_ligand_filters_non_a_altlocs() -> None:
    pdb_text = (
        "HETATM    1  C1 AOLA A1004       7.453   6.475  17.189  0.50 46.96           C  \n"
        "HETATM    2  C1 BOLA A1004      14.789  -5.215  28.955  0.50 60.73           C  \n"
        "HETATM    3  O1 AOLA A1004       6.937   6.655  16.062  0.50 44.55           O  \n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = Path(tmpdir) / "complex.pdb"
        pdb_path.write_text(pdb_text)
        ligand_path = prepare_ligand(pdb_path)
        lines = ligand_path.read_text().splitlines()

    assert len(lines) == 2
    assert "AOLA" in lines[0]
    assert "BOLA" not in "\n".join(lines)
