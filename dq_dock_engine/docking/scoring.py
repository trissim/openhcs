"""
Scoring mechanism using strict OpenHCS Enum dispatch.

Separates JAX-native internal physics from impure SMINA external subprocess wrapper.
"""

from typing import List, Dict, Callable
import os
import subprocess
import tempfile
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.core import ScoringEngine, ScoredPose


@jax.jit
def _score_single_lj(receptor_coords: jnp.ndarray, pose_coords: jnp.ndarray) -> float:
    """
    Very crude pure-JAX inter-molecular LJ score.
    receptor_coords: (N_rec, 3)
    pose_coords: (N_lig, 3)
    """
    diffs = receptor_coords[:, None, :] - pose_coords[None, :, :]
    dist_sq = jnp.sum(diffs ** 2, axis=-1)
    
    # Clamp BEFORE inverse — gradient is safe everywhere
    dist_sq_safe = jnp.maximum(dist_sq, 0.5 ** 2)  # r_min = 0.5σ
    
    r6 = dist_sq_safe ** 3
    inv_r6 = 1.0 / r6
    inv_r12 = inv_r6 ** 2
    
    pe = 4.0 * (inv_r12 - inv_r6)
    return jnp.sum(pe)


@jax.jit
def score_internal_lj(receptor_coords: jnp.ndarray, poses_coords: jnp.ndarray) -> jnp.ndarray:
    """
    Pure JAX batched internal LJ score.
    receptor_coords: (N_rec, 3)
    poses_coords: (N_poses, N_lig, 3)
    
    Returns:
        (N_poses,) array of scores
    """
    batched_score = jax.vmap(_score_single_lj, in_axes=(None, 0))
    return batched_score(receptor_coords, poses_coords)


def _write_pdb(coords: np.ndarray, template_pdb: str, output_pdb: str):
    """Write coordinates back to a temporary PDB using a template."""
    with open(template_pdb, 'r') as f:
        lines = f.readlines()
        
    out_lines = []
    atom_idx = 0
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            if atom_idx < len(coords):
                x, y, z = coords[atom_idx]
                # PDB column formatting
                new_line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                out_lines.append(new_line)
                atom_idx += 1
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)
            
    with open(output_pdb, 'w') as f:
        f.writelines(out_lines)


def score_smina_exact(receptor_file: str, ligand_template: str, poses_coords: np.ndarray) -> np.ndarray:
    """
    Impure external wrapper invoking SMINA for accurate scoring.
    
    Args:
        receptor_file: path to receptor PDB
        ligand_template: path to original ligand PDB to use as template
        poses_coords: (N_poses, N_lig, 3) numpy array
        
    Returns:
        np.ndarray of shape (N_poses,) with SMINA affinities
    """
    from dq_dock_engine.benchmark.benchmark_pdb import check_vina
    vina_path = check_vina()
    if not vina_path:
        raise RuntimeError("SMINA/Vina binary not found.")
        
    n_poses = poses_coords.shape[0]
    scores = np.zeros(n_poses)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        
        # We can score sequentially or in parallel here, but for simplicity
        # we iterate over poses sequentially for the wrapper.
        for i in range(n_poses):
            pose_pdb = tmp_path / f"pose_{i}.pdb"
            _write_pdb(poses_coords[i], ligand_template, str(pose_pdb))
            
            cmd = [
                vina_path,
                "--receptor", str(receptor_file),
                "--ligand", str(pose_pdb),
                "--score_only"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                # parse "Affinity: -6.54321 (kcal/mol)"
                for line in result.stdout.split('\n'):
                    if line.startswith("Affinity:"):
                        val = float(line.split()[1])
                        scores[i] = val
                        break
            except Exception as e:
                # Fallback to highly unfavorable score if SMINA fails
                scores[i] = 1000.0
                
    return scores


def route_scoring(engine: ScoringEngine, **kwargs) -> np.ndarray:
    """
    Strict Enum dispatch for scoring.
    
    Required kwargs:
      - INTERNAL_LJ: receptor_coords, poses_coords
      - SMINA_EXACT: receptor_file, ligand_template, poses_coords
    """
    if engine == ScoringEngine.INTERNAL_LJ:
        # Returns jnp.ndarray, we cast to np.ndarray for API consistency
        return np.array(score_internal_lj(kwargs['receptor_coords'], kwargs['poses_coords']))
        
    elif engine == ScoringEngine.SMINA_EXACT:
        return score_smina_exact(
            kwargs['receptor_file'], 
            kwargs['ligand_template'], 
            kwargs['poses_coords']
        )
        
    raise ValueError(f"Unknown ScoringEngine: {engine}")
