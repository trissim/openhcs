from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from rdkit import Chem

from dq_dock_engine.docking.chemistry_annotations import _infer_bond_adjacency
from dq_dock_engine.docking.conformer_search import (
    BranchAndBoundConfig,
    search_conformers,
)
from dq_dock_engine.docking.core import ScoringEngine
from dq_dock_engine.docking.pipeline import _build_conformer_score_fns
from dq_dock_engine.docking.pdb_io import build_ligand_context
from dq_dock_engine.physics.kernels import rigid_transform_3d


@dataclass(frozen=True)
class _TraceBatchResult:
    scores: jnp.ndarray


class _TraceScoringContext:
    def score_rigid_exact_batch(
        self,
        *,
        receptor_coords: jnp.ndarray,
        poses_coords: jnp.ndarray,
        receptor_radii: jnp.ndarray,
        ligand_radii: jnp.ndarray,
        target_error: float,
        epsilon: float,
    ) -> _TraceBatchResult:
        del receptor_coords, receptor_radii, ligand_radii, target_error, epsilon
        print(f"score_rigid_exact_batch poses_coords.shape={poses_coords.shape}")
        return _TraceBatchResult(
            scores=jnp.zeros((poses_coords.shape[0],), dtype=poses_coords.dtype)
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_pdb_atoms(pdb_path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    coords: list[list[float]] = []
    elements: list[str] = []
    with pdb_path.open() as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            coords.append(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ]
            )
            element = line[76:78].strip() or line[12:16].strip()[0]
            elements.append(element)
    return np.asarray(coords, dtype=np.float32), tuple(elements)


def _vdw_radii(elements: tuple[str, ...]) -> np.ndarray:
    pt = Chem.GetPeriodicTable()
    radii = []
    for element in elements:
        try:
            radii.append(float(pt.GetRvdw(element)))
        except Exception:
            radii.append(1.5)
    return np.asarray(radii, dtype=np.float32)


def test_1hk4_pipeline_trace_recreates_rigid_score_path() -> None:
    poses_dir = (
        _repo_root()
        / "benchmark_1hk4_doc_repro_known_rdkit_fix1"
        / "pdb_redocking_20260324_133456_poses"
    )
    ligand_pdb = poses_dir / "1hk4_native_ligand.pdb"
    receptor_pdb = poses_dir / "1hk4_receptor.pdb"

    assert ligand_pdb.exists(), ligand_pdb
    assert receptor_pdb.exists(), receptor_pdb

    ligand_mol = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=False)
    assert ligand_mol is not None

    ligand_conf = ligand_mol.GetConformer(0)
    ligand_coords = np.asarray(
        [
            [
                ligand_conf.GetAtomPosition(i).x,
                ligand_conf.GetAtomPosition(i).y,
                ligand_conf.GetAtomPosition(i).z,
            ]
            for i in range(ligand_mol.GetNumAtoms())
        ],
        dtype=np.float32,
    )
    ligand_elements = tuple(atom.GetSymbol() for atom in ligand_mol.GetAtoms())
    ligand_adjacency = _infer_bond_adjacency(
        ligand_coords, ligand_elements, include_hydrogens=False
    )
    ligand_radii = _vdw_radii(ligand_elements)
    ligand_ctx = build_ligand_context(
        ligand_coords,
        ligand_radii,
        elements=list(ligand_elements),
        adjacency=ligand_adjacency,
    )

    receptor_coords, receptor_elements = _load_pdb_atoms(receptor_pdb)
    receptor_radii = _vdw_radii(receptor_elements)

    print(f"ligand_atoms={ligand_coords.shape[0]}")
    print(f"receptor_atoms={receptor_coords.shape[0]}")
    print(f"ligand_adjacency_len={len(ligand_adjacency)}")
    print(f"ligand_base_coords_shape={ligand_ctx.base_coords.shape}")

    identity_q = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    zero_t = jnp.zeros((3,), dtype=jnp.float32)

    single_pose = rigid_transform_3d(ligand_ctx.base_coords, identity_q, zero_t)
    print(f"single_pose_shape={single_pose.shape}")

    pose_batch = jnp.stack([ligand_ctx.base_coords] * 8, axis=0)
    vmap_pose_batch = jax.vmap(lambda c: rigid_transform_3d(c, identity_q, zero_t))(
        pose_batch
    )
    print(f"vmap_pose_batch_shape={vmap_pose_batch.shape}")

    config = BranchAndBoundConfig(
        max_cells=16,
        min_cell_radius=0.1,
        score_lipschitz_constant=1.0,
        max_conformers=4,
    )

    conformer_result = search_conformers(
        base_coords=ligand_ctx.base_coords,
        adjacency=ligand_adjacency,
        elements=ligand_elements,
        score_fn=lambda coords: float(jnp.sum(coords) * 0.0),
        config=config,
    )
    print(f"search_conformers_n={len(conformer_result.conformer_coords)}")
    print(f"search_conformers_energies={conformer_result.conformer_energies}")

    request = SimpleNamespace(
        protein_coords=jnp.asarray(receptor_coords),
        receptor_radii=jnp.asarray(receptor_radii),
        ligand_ctx=ligand_ctx,
        target_error=1.0,
        effective_engine=ScoringEngine.CERTIFIED_LJ_REALSPACE_EWALD,
    )

    scoring_context = _TraceScoringContext()
    score_fn, score_fn_batch = _build_conformer_score_fns(
        request,
        identity_q,
        zero_t,
        scoring_context,
        None,
    )

    single_score = score_fn(ligand_ctx.base_coords)
    print(f"single_score={single_score}")

    batch_scores = score_fn_batch(pose_batch)
    batch_scores.block_until_ready()
    print(f"score_fn_batch_shape={batch_scores.shape}")

    assert ligand_coords.shape == (24, 3)
    assert ligand_ctx.base_coords.shape == (24, 3)
    assert single_pose.shape == (24, 3)
    assert vmap_pose_batch.shape == (8, 24, 3)
    assert len(conformer_result.conformer_coords) == 1
    assert batch_scores.shape == (8,)
