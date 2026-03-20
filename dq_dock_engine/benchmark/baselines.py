import subprocess
import time
import json
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

"""
AutoDock Vina baseline runner.
Runs Vina on PDBbind complexes and parses results for comparison.
"""


@dataclass(frozen=True)
class VinaResult:
    """Result from a single Vina run."""

    pdb_code: str
    best_affinity: float  # kcal/mol
    best_rmsd_lb: float  # Å
    best_rmsd_ub: float  # Å
    n_modes: int
    wall_time_s: float
    success: bool
    error: str = ""


@dataclass(frozen=True)
class VinaConfig:
    """Configuration for Vina runs."""

    vina_path: str = "vina"
    exhaustiveness: int = 8
    n_modes: int = 9
    energy_range: float = 3.0
    seed: int = 42
    cpu: int = 1


def run_vina(
    config: VinaConfig,
    receptor_path: str,
    ligand_path: str,
    center: tuple,
    size: tuple = (12.0, 12.0, 12.0),
    output_path: Optional[str] = None,
) -> VinaResult:
    """
    Run AutoDock Vina on a single complex.
    Returns VinaResult with timing and affinity.
    """
    pdb_code = Path(receptor_path).parent.name

    if output_path is None:
        output_path = f"/tmp/vina_{pdb_code}_out.pdbqt"

    cmd = [
        config.vina_path,
        "--receptor",
        receptor_path,
        "--ligand",
        ligand_path,
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(size[0]),
        "--size_y",
        str(size[1]),
        "--size_z",
        str(size[2]),
        "--exhaustiveness",
        str(config.exhaustiveness),
        "--num_modes",
        str(config.n_modes),
        "--energy_range",
        str(config.energy_range),
        "--seed",
        str(config.seed),
        "--cpu",
        str(config.cpu),
        "--out",
        output_path,
    ]

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        wall_time = time.time() - start

        if result.returncode != 0:
            return VinaResult(
                pdb_code=pdb_code,
                best_affinity=0,
                best_rmsd_lb=0,
                best_rmsd_ub=0,
                n_modes=0,
                wall_time_s=wall_time,
                success=False,
                error=result.stderr[:200],
            )

        # Parse output
        affinity, rmsd_lb, rmsd_ub, n_modes = _parse_vina_output(result.stdout)

        return VinaResult(
            pdb_code=pdb_code,
            best_affinity=affinity,
            best_rmsd_lb=rmsd_lb,
            best_rmsd_ub=rmsd_ub,
            n_modes=n_modes,
            wall_time_s=wall_time,
            success=True,
        )
    except subprocess.TimeoutExpired:
        return VinaResult(
            pdb_code=pdb_code,
            best_affinity=0,
            best_rmsd_lb=0,
            best_rmsd_ub=0,
            n_modes=0,
            wall_time_s=600,
            success=False,
            error="Timeout",
        )
    except FileNotFoundError:
        return VinaResult(
            pdb_code=pdb_code,
            best_affinity=0,
            best_rmsd_lb=0,
            best_rmsd_ub=0,
            n_modes=0,
            wall_time_s=0,
            success=False,
            error="Vina binary not found",
        )


def _parse_vina_output(stdout: str) -> tuple:
    """Parse Vina stdout for best mode."""
    lines = stdout.strip().split("\n")
    best_affinity = 0.0
    best_rmsd_lb = 0.0
    best_rmsd_ub = 0.0
    n_modes = 0

    in_results = False
    for line in lines:
        if "-----" in line:
            in_results = True
            continue
        if in_results and line.strip():
            parts = line.split()
            if len(parts) >= 4:
                try:
                    mode = int(parts[0])
                    affinity = float(parts[1])
                    rmsd_lb = float(parts[2])
                    rmsd_ub = float(parts[3])
                    n_modes += 1
                    if mode == 1:
                        best_affinity = affinity
                        best_rmsd_lb = rmsd_lb
                        best_rmsd_ub = rmsd_ub
                except (ValueError, IndexError):
                    continue

    return best_affinity, best_rmsd_lb, best_rmsd_ub, n_modes


def run_batch(config: VinaConfig, complexes: list, centers: dict) -> List[VinaResult]:
    """Run Vina on a batch of complexes."""
    results = []
    for cx in complexes:
        center = centers.get(cx.pdb_code, (0, 0, 0))
        result = run_vina(config, cx.protein_path, cx.ligand_path, center)
        results.append(result)
    return results
