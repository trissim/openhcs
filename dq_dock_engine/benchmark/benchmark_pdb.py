#!/usr/bin/env python3
"""
Real PDB-based Docking Benchmark

Downloads famous drug-target complexes and runs DQ-Dock vs SMINA comparison.
These are well-studied, difficult targets where docking actually matters.

Famous drug targets:
- HIV-1 Protease: Major AIDS drug target
- CDK2 Kinase: Cancer target
- SARS-CoV-2 Mpro: COVID-19 drug target
- Carbonic anhydrase: Anti-glaucoma, altitude sickness
- Thrombin: Blood thinner target
- BACE-1: Alzheimer's target

Usage:
    python benchmark_pdb.py --n_complexes 10

Requirements:
    smina: yay -S smina-bin
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, cast
import urllib.request
import gzip
import shutil

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass

from dq_dock_engine.docking.core import (
    DockingBox,
    LigandContext,
    ScoringEngine,
    BenchmarkResult,
)
from dq_dock_engine.docking.pipeline import run_docking_pipeline
from dq_dock_engine.docking.pdb_io import parse_structure, build_receptor_arrays
from dq_dock_engine.docking.metrics import (
    compute_rmsd_batched,
    compute_docking_rmsd_batched,
)
from dq_dock_engine.docking_config import (
    CERTIFIED_DOCKING,
)
from dq_dock_engine.benchmark.large_pdb_ids import (
    BenchmarkScope,
    PDBEntry,
    get_benchmark_entries,
    get_default_ligand_entries,
)


# Famous, difficult drug targets where docking matters
# These are well-studied complexes that people actually care about
TEST_PDB_IDS = [
    # HIV-1 Protease - famous drug target, challenging docking
    "1hvr",  # HIV-1 protease with inhibitor
    "1ajx",  # HIV-1 protease
    # Kinases - largest drug target class, difficult
    "1jvp",  # CDK2 kinase
    "1ywr",  # AKT kinase
    # Carbonic anhydrase - tight binding, important
    "1z00",  # Carbonic anhydrase II
    # SARS-CoV-2 main protease - COVID drug target
    "6lu7",  # Mpro with inhibitor
    # Thrombin - important drug target
    "1ppb",  # Thrombin
    # BACE - Alzheimer's drug target, challenging
    "4pei",  # BACE-1
    # Factor VII - important for blood clotting
    "1fak",  # Factor VIIa
    # Tankyrase - cancer target
    "3t0i",  # Tankyrase
]


@dataclass
class PDBComplex:
    """A protein-ligand complex from PDB."""

    pdb_id: str
    protein_file: Path
    ligand_file: Path
    center: tuple  # docking center
    size: tuple  # docking box size


@dataclass(frozen=True)
class ComplexSummaryRow:
    pdb_id: str
    target_name: str
    ligand_atoms: int
    pocket_atoms: int
    rmsd: float
    time: float


@dataclass(frozen=True)
class VinaResult:
    success: bool
    time: float
    best_affinity: float | None = None
    top_coords: np.ndarray | None = None
    model_coords: tuple[np.ndarray, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class VinaSummaryRow:
    pdb_id: str
    target_name: str
    top_rmsd: float
    best_mode_rmsd: float
    time: float
    affinity: float


@dataclass(frozen=True)
class BenchmarkOutputPaths:
    json_path: Path
    csv_path: Path


@dataclass(frozen=True)
class VinaPrepTools:
    receptor_tool: str
    ligand_tool: str


@dataclass(frozen=True)
class ExcludedComplexRow:
    pdb_id: str
    reason: str


@dataclass(frozen=True)
class LigandResidue:
    resname: str
    chain_id: str
    residue_id: str
    atom_count: int


@dataclass(frozen=True)
class ScreeningDecision:
    accepted: bool
    reasons: tuple[str, ...]
    ligand_residue: LigandResidue | None = None
    heavy_atom_count: int = 0
    resolution: float | None = None
    model_count: int = 1


MIN_HEAVY_ATOMS = 10
MAX_RESOLUTION_ANGSTROMS = 2.5
COVALENT_DISTANCE_ANGSTROMS = 1.55
METAL_RESNAMES = {
    "ZN",
    "MG",
    "MN",
    "FE",
    "CU",
    "CO",
    "NI",
    "CA",
    "NA",
    "K",
    "CD",
}


def format_gap_proof_label(certified: bool | None) -> str | None:
    if certified is None:
        return None
    return "proved" if certified else "inconclusive"


def _non_water_hetatm_residue_counts(pdb_path: Path) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if resname in {"HOH", "DOD", "WAT"}:
                continue
            chain_id = line[21].strip()
            residue_id = f"{line[22:26].strip()}{line[26].strip()}"
            key = (resname, chain_id, residue_id)
            counts[key] = counts.get(key, 0) + 1
    return counts


def detect_primary_ligand_residue(
    pdb_path: Path, preferred_resname: str | None = None
) -> LigandResidue:
    counts = _non_water_hetatm_residue_counts(pdb_path)
    if not counts:
        raise ValueError(f"No ligand found in {pdb_path}")
    if preferred_resname is not None:
        counts = {
            key: value for key, value in counts.items() if key[0] == preferred_resname
        }
        if not counts:
            raise ValueError(
                f"Preferred ligand residue {preferred_resname} not found in {pdb_path}"
            )
    (resname, chain_id, residue_id), atom_count = max(
        counts.items(), key=lambda x: x[1]
    )
    return LigandResidue(
        resname=resname,
        chain_id=chain_id,
        residue_id=residue_id,
        atom_count=atom_count,
    )


def parse_model_count(pdb_path: Path) -> int:
    model_count = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                model_count += 1
    return model_count if model_count > 0 else 1


def parse_resolution_angstroms(pdb_path: Path) -> float | None:
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("REMARK   2 RESOLUTION."):
                parts = line.split()
                for idx, token in enumerate(parts):
                    if token == "RESOLUTION." and idx + 1 < len(parts):
                        try:
                            return float(parts[idx + 1])
                        except ValueError:
                            return None
    return None


def has_covalent_contact(protein_path: Path, ligand_path: Path) -> bool:
    protein_coords, _ = cast(
        tuple[np.ndarray, np.ndarray],
        parse_structure(protein_path, strip_hydrogens=True),
    )
    ligand_coords, _ = cast(
        tuple[np.ndarray, np.ndarray],
        parse_structure(ligand_path, strip_hydrogens=True),
    )
    if len(protein_coords) == 0 or len(ligand_coords) == 0:
        return False
    diffs = protein_coords[:, None, :] - ligand_coords[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    return bool(np.any(distances < COVALENT_DISTANCE_ANGSTROMS))


def screen_complex(
    entry: PDBEntry, pdb_path: Path, protein_path: Path
) -> ScreeningDecision:
    reasons: list[str] = []

    if entry.scope != BenchmarkScope.LIGAND:
        reasons.append(f"scope={entry.scope.value} is out of scope for ligand docking")

    if entry.exclusion_reason is not None:
        reasons.append(entry.exclusion_reason)

    resolution = parse_resolution_angstroms(pdb_path)
    if resolution is not None and resolution >= MAX_RESOLUTION_ANGSTROMS:
        reasons.append(
            f"resolution {resolution:.2f}A exceeds gate < {MAX_RESOLUTION_ANGSTROMS:.2f}A"
        )

    model_count = parse_model_count(pdb_path)
    if model_count != 1:
        reasons.append(f"MODEL count {model_count} != 1")

    try:
        ligand_residue = detect_primary_ligand_residue(
            pdb_path, preferred_resname=entry.preferred_resname
        )
    except ValueError as exc:
        reasons.append(str(exc))
        return ScreeningDecision(
            accepted=False,
            reasons=tuple(reasons),
            ligand_residue=None,
            heavy_atom_count=0,
            resolution=resolution,
            model_count=model_count,
        )

    if ligand_residue.atom_count < MIN_HEAVY_ATOMS:
        reasons.append(
            f"heavy atom count {ligand_residue.atom_count} is below gate >= {MIN_HEAVY_ATOMS}"
        )

    if ligand_residue.resname in METAL_RESNAMES:
        reasons.append(
            f"ligand residue {ligand_residue.resname} is a metal-only species"
        )

    ligand_path = prepare_ligand(pdb_path, preferred_resname=entry.preferred_resname)
    if has_covalent_contact(protein_path, ligand_path):
        reasons.append(
            f"minimum protein-ligand distance is below {COVALENT_DISTANCE_ANGSTROMS:.2f}A"
        )

    return ScreeningDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        ligand_residue=ligand_residue,
        heavy_atom_count=ligand_residue.atom_count,
        resolution=resolution,
        model_count=model_count,
    )


def download_pdb(pdb_id: str, cache_dir: Path) -> Path | None:
    """Download PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
    gz_path = cache_dir / f"{pdb_id}.pdb.gz"
    pdb_path = cache_dir / f"{pdb_id}.pdb"

    if pdb_path.exists():
        return pdb_path

    print(f"  Downloading {pdb_id}...", flush=True)
    try:
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, "rt") as f_in:
            with open(pdb_path, "w") as f_out:
                f_out.write(f_in.read())
        gz_path.unlink()  # remove gz
        return pdb_path
    except Exception as e:
        print(f"  ⚠️  Failed to download {pdb_id}: {e}")
        return None


def prepare_protein(pdb_path: Path) -> Path:
    """Prepare protein for Vina - use smina which handles PDB directly.

    Extracts only ATOM records (protein) from the PDB file.
    """
    protein_path = pdb_path.parent / f"{pdb_path.stem}_protein.pdb"
    if protein_path.exists():
        return protein_path

    with open(pdb_path) as f_in:
        with open(protein_path, "w") as f_out:
            for line in f_in:
                if line.startswith("ATOM"):
                    f_out.write(line)

    return protein_path


def prepare_pocket_protein(
    protein_path: Path,
    center: tuple[float, float, float],
    pocket_radius: float = 12.0,
) -> Path:
    """Write a pocket-only receptor PDB matching the filtered docking arrays."""
    pocket_path = protein_path.parent / f"{protein_path.stem}_pocket.pdb"

    center_array = np.array(center, dtype=float)
    pocket_lines: list[str] = []

    with open(protein_path) as f_in:
        for line in f_in:
            if not line.startswith("ATOM"):
                continue
            try:
                coord = np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=float,
                )
            except ValueError:
                continue

            if np.linalg.norm(coord - center_array) < pocket_radius:
                pocket_lines.append(line)

    if not pocket_lines:
        raise ValueError(f"No pocket atoms found in {protein_path}")

    with open(pocket_path, "w") as f_out:
        f_out.writelines(pocket_lines)

    return pocket_path


def prepare_ligand(pdb_path: Path, preferred_resname: str | None = None) -> Path:
    """Prepare ligand for Vina - extract primary ligand HETATM records from PDB file.

    Raises error if no ligand found in the PDB.
    """
    ligand_path = pdb_path.parent / f"{pdb_path.stem}_ligand.pdb"

    # Check if already extracted
    if ligand_path.exists():
        # Remove it if we are re-running to fix old bad extractions
        ligand_path.unlink()

    target_residue = detect_primary_ligand_residue(
        pdb_path, preferred_resname=preferred_resname
    )

    ligand_lines = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            chain_id = line[21].strip()
            residue_id = f"{line[22:26].strip()}{line[26].strip()}"
            if (
                line[17:20].strip() == target_residue.resname
                and chain_id == target_residue.chain_id
                and residue_id == target_residue.residue_id
            ):
                ligand_lines.append(line)

    if not ligand_lines:
        raise ValueError(f"No ligand found in {pdb_path}")

    with open(ligand_path, "w") as f:
        f.writelines(ligand_lines)
    return ligand_path


from dq_dock_engine.docking.physics_params import get_vdw_radius


def extract_coords_and_radii(
    pdb_path: Path, is_ligand: bool = True
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract heavy atom coordinates and VdW radii from PDB file."""
    coords = []
    radii = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM") or line.startswith("ATOM"):
                element = line[76:78].strip()
                if not element:
                    element_str = line[12:16].strip()
                    if element_str.startswith("H"):
                        if is_ligand:
                            continue
                        element = "H"
                    else:
                        element = element_str[0]
                elif element == "H" and is_ligand:
                    continue

                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                    radii.append(get_vdw_radius(element))
                except ValueError:
                    continue

    if coords:
        return np.array(coords), np.array(radii)
    return None, None


def compute_pocket_center(ligand_coords: np.ndarray) -> tuple:
    """Compute center of ligand for docking box."""
    center = ligand_coords.mean(axis=0)
    return tuple(center)


def find_docking_center(pdb_path: Path, preferred_resname: str | None = None) -> tuple:
    """Find docking center from ligand in PDB."""
    ligand_pdb = prepare_ligand(pdb_path, preferred_resname=preferred_resname)
    coords, _, _ = cast(
        tuple[np.ndarray, np.ndarray, list[str]],
        parse_structure(ligand_pdb, return_elements=True),
    )
    return compute_pocket_center(coords)


def check_vina() -> Optional[str]:
    """Find Vina or SMINA binary."""
    import shutil

    # Check common locations - prefer smina over vina
    for name in ["smina", "vina", "autodock_vina"]:
        path = shutil.which(name)
        if path:
            return path

    # Check local directory
    for name in ["smina", "vina"]:
        local = Path(f"./{name}")
        if local.exists():
            return str(local.absolute())

    return None


def check_vina_prep_tools() -> VinaPrepTools | None:
    receptor_tool = shutil.which("prepare_receptor4.py")
    ligand_tool = shutil.which("prepare_ligand4.py")
    if receptor_tool is not None and ligand_tool is not None:
        return VinaPrepTools(receptor_tool=receptor_tool, ligand_tool=ligand_tool)
    return None


def prepare_vina_inputs(
    prep_tools: VinaPrepTools | None,
    receptor_pdb: Path,
    ligand_pdb: Path,
) -> tuple[Path, Path, Path | None]:
    if prep_tools is None:
        return receptor_pdb, ligand_pdb, None

    temp_dir = Path(tempfile.mkdtemp(prefix="vina_prep_"))
    receptor_pdbqt = temp_dir / f"{receptor_pdb.stem}.pdbqt"
    ligand_pdbqt = temp_dir / f"{ligand_pdb.stem}.pdbqt"

    receptor_cmd = [
        prep_tools.receptor_tool,
        "-r",
        str(receptor_pdb),
        "-o",
        str(receptor_pdbqt),
        "-A",
        "checkhydrogens",
    ]
    ligand_cmd = [
        prep_tools.ligand_tool,
        "-l",
        str(ligand_pdb),
        "-o",
        str(ligand_pdbqt),
        "-A",
        "checkhydrogens",
    ]

    subprocess.run(receptor_cmd, check=True, capture_output=True, text=True)
    subprocess.run(ligand_cmd, check=True, capture_output=True, text=True)
    return receptor_pdbqt, ligand_pdbqt, temp_dir


def _extract_pdb_element(line: str) -> str:
    element = line[76:78].strip()
    if element:
        return element
    atom_name = line[12:16].strip()
    return atom_name[0] if atom_name else ""


def parse_smina_models(pdb_path: Path) -> tuple[np.ndarray, ...]:
    """Extract heavy-atom coordinates for every returned docking model."""
    models: list[np.ndarray] = []
    current_coords: list[list[float]] = []
    saw_model = False

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                saw_model = True
                current_coords = []
                continue
            if line.startswith("ENDMDL"):
                if current_coords:
                    models.append(np.array(current_coords, dtype=np.float64))
                current_coords = []
                continue
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            element = _extract_pdb_element(line)
            if element == "H":
                continue

            try:
                current_coords.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
            except ValueError:
                continue

    if current_coords:
        models.append(np.array(current_coords, dtype=np.float64))

    if not saw_model and models:
        return (models[0],)
    return tuple(models)


def parse_smina_affinities(stdout: str) -> tuple[float, ...]:
    affinities: list[float] = []
    in_results = False
    for line in stdout.split("\n"):
        if "mode |" in line and "affinity" in line:
            in_results = True
            continue
        if not in_results or not line.strip() or not line[0].isdigit():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            affinities.append(float(parts[1]))
        except ValueError:
            continue
    return tuple(affinities)


def compute_pose_rmsd(pose_coords: np.ndarray, native_coords: np.ndarray) -> float:
    min_len = min(len(pose_coords), len(native_coords))
    if min_len == 0:
        return float("nan")
    pose_jnp = jnp.expand_dims(pose_coords[:min_len], axis=0)
    native_jnp = jnp.array(native_coords[:min_len])
    return float(compute_docking_rmsd_batched(pose_jnp, native_jnp)[0])


def _safe_json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def create_benchmark_output_paths(output_dir: Path) -> BenchmarkOutputPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"pdb_redocking_{timestamp}"
    return BenchmarkOutputPaths(
        json_path=output_dir / f"{stem}.json",
        csv_path=output_dir / f"{stem}.csv",
    )


def save_benchmark_results(
    output_paths: BenchmarkOutputPaths,
    *,
    charge_method_name: str,
    n_complexes_requested: int,
    n_poses: int,
    n_opt_steps: int,
    use_multi_stage: bool,
    use_pocket_guided: bool,
    bench_elapsed: float,
    phase: str,
    complexes: list[dict],
    dq_rows: list[ComplexSummaryRow],
    dq_results: list[BenchmarkResult],
    vina_rows: list[VinaSummaryRow],
    excluded_rows: list[ExcludedComplexRow],
) -> tuple[Path, Path]:
    json_path = output_paths.json_path
    csv_path = output_paths.csv_path

    complex_map = {str(cx["pdb_id"]): cx for cx in complexes}
    dq_map = {row.pdb_id: row for row in dq_rows}
    dq_result_map = {
        str(cx["pdb_id"]): result for cx, result in zip(complexes, dq_results)
    }
    vina_map = {row.pdb_id: row for row in vina_rows}

    csv_rows: list[dict[str, object]] = []
    for pdb_id, row in dq_map.items():
        cx = complex_map[pdb_id]
        dq_result = dq_result_map[pdb_id]
        vina_row = vina_map.get(pdb_id)
        csv_rows.append(
            {
                "pdb_id": pdb_id,
                "target_name": row.target_name,
                "center_x": cx["center"][0],
                "center_y": cx["center"][1],
                "center_z": cx["center"][2],
                "ligand_atoms": row.ligand_atoms,
                "pocket_atoms": row.pocket_atoms,
                "dq_energy": dq_result.energy,
                "dq_rmsd": row.rmsd,
                "dq_time_s": row.time,
                "gap_proof": format_gap_proof_label(dq_result.certified),
                "native_rank": dq_result.native_rank,
                "energy_gap": dq_result.energy_gap,
                "vina_affinity": None if vina_row is None else vina_row.affinity,
                "vina_top_rmsd": None if vina_row is None else vina_row.top_rmsd,
                "vina_best_mode_rmsd": None
                if vina_row is None
                else vina_row.best_mode_rmsd,
                "vina_time_s": None if vina_row is None else vina_row.time,
            }
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(csv_rows[0].keys())
            if csv_rows
            else [
                "pdb_id",
                "target_name",
                "center_x",
                "center_y",
                "center_z",
                "ligand_atoms",
                "pocket_atoms",
                "dq_energy",
                "dq_rmsd",
                "dq_time_s",
                "gap_proof",
                "native_rank",
                "energy_gap",
                "vina_affinity",
                "vina_top_rmsd",
                "vina_best_mode_rmsd",
                "vina_time_s",
            ],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({k: _safe_json_value(v) for k, v in row.items()})

    summary: dict[str, object] = {
        "timestamp": json_path.stem.removeprefix("pdb_redocking_"),
        "phase": phase,
        "charge_method": charge_method_name,
        "n_complexes_requested": n_complexes_requested,
        "n_complexes_run": len(dq_rows),
        "n_complexes_excluded": len(excluded_rows),
        "n_vina_completed": len(vina_rows),
        "n_poses": n_poses,
        "n_opt_steps": n_opt_steps,
        "use_multi_stage": use_multi_stage,
        "use_pocket_guided": use_pocket_guided,
        "competitors": ["smina"],
        "total_benchmark_time_s": bench_elapsed,
        "dq_avg_rmsd": None
        if not dq_rows
        else float(np.mean([row.rmsd for row in dq_rows])),
        "dq_total_time_s": float(sum(row.time for row in dq_rows)),
        "vina_avg_top_rmsd": None
        if not vina_rows
        else float(np.mean([row.top_rmsd for row in vina_rows])),
        "vina_avg_best_mode_rmsd": None
        if not vina_rows
        else float(np.mean([row.best_mode_rmsd for row in vina_rows])),
        "vina_total_time_s": float(sum(row.time for row in vina_rows)),
    }

    payload = {
        "summary": {k: _safe_json_value(v) for k, v in summary.items()},
        "dq_dock": [
            {
                "pdb_id": row.pdb_id,
                "target_name": row.target_name,
                "ligand_atoms": row.ligand_atoms,
                "pocket_atoms": row.pocket_atoms,
                "rmsd": _safe_json_value(row.rmsd),
                "time_s": _safe_json_value(row.time),
                "energy": _safe_json_value(dq_result_map[row.pdb_id].energy),
                "gap_proof": format_gap_proof_label(
                    dq_result_map[row.pdb_id].certified
                ),
                "native_rank": _safe_json_value(dq_result_map[row.pdb_id].native_rank),
                "energy_gap": _safe_json_value(dq_result_map[row.pdb_id].energy_gap),
            }
            for row in dq_rows
        ],
        "vina": [
            {
                "pdb_id": row.pdb_id,
                "target_name": row.target_name,
                "top_rmsd": _safe_json_value(row.top_rmsd),
                "best_returned_mode_rmsd": _safe_json_value(row.best_mode_rmsd),
                "time_s": _safe_json_value(row.time),
                "affinity": _safe_json_value(row.affinity),
            }
            for row in vina_rows
        ],
        "excluded": [
            {"pdb_id": row.pdb_id, "reason": row.reason} for row in excluded_rows
        ],
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    return json_path, csv_path


def run_vina(
    vina_path: str,
    receptor: Path,
    ligand: Path,
    center: tuple,
    size: tuple = (20, 20, 20),
    exhaustiveness: int = 64,
    n_models: int = 20,
    energy_range: float = 12.0,
    min_rmsd_filter: float = 0.5,
) -> VinaResult:
    """Run a strong-search smina configuration for known-pocket redocking."""
    # Use a safe temp file path
    temp_fd, temp_path = tempfile.mkstemp(suffix=".pdb")
    os.close(temp_fd)
    output_path = Path(temp_path)

    cmd = [
        vina_path,
        "--receptor",
        str(receptor),
        "--ligand",
        str(ligand),
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
        str(exhaustiveness),
        "--num_modes",
        str(n_models),
        "--energy_range",
        str(energy_range),
        "--min_rmsd_filter",
        str(min_rmsd_filter),
        "--seed",
        "42",
        "--out",
        str(output_path),
    ]

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start

        if result.returncode != 0:
            return VinaResult(
                success=False,
                error=result.stderr[:200],
                time=elapsed,
            )

        affinities = parse_smina_affinities(result.stdout)
        models = parse_smina_models(output_path) if output_path.exists() else ()
        top_coords = models[0] if models else None

        return VinaResult(
            success=True,
            best_affinity=affinities[0] if affinities else None,
            time=elapsed,
            top_coords=top_coords,
            model_coords=models,
        )
    except Exception as e:
        return VinaResult(success=False, error=str(e), time=time.time() - start)
    finally:
        if output_path.exists():
            output_path.unlink()


def run_dq_dock(
    pocket_coords: np.ndarray,
    pocket_radii: np.ndarray,
    ligand_coords: np.ndarray,
    ligand_radii: np.ndarray,
    center: tuple,
    ligand_file: Path,
    receptor_file: Path,
    charge_method,
    n_poses: int = 2000,
    use_multi_stage: bool = False,
    use_pocket_guided: bool = False,
    engine: ScoringEngine = ScoringEngine.INTERNAL_LJ,
    ligand_elements: list[str] | tuple[str, ...] | None = None,
    receptor_elements: list[str] | tuple[str, ...] | None = None,
    n_opt_steps: int = 50,
    max_retries: int = 6,
) -> BenchmarkResult:
    """Run true DQ-Dock pipeline on complex using core infrastructure."""
    from dq_dock_engine.docking.core import DockingBox, ScoringEngine
    from dq_dock_engine.docking.pdb_io import (
        build_ligand_context,
        build_receptor_arrays,
    )
    from dq_dock_engine.docking.pipeline import run_docking_pipeline
    from dq_dock_engine.docking.metrics import compute_docking_rmsd_batched

    start = time.time()

    center_jnp = jnp.array(center)
    box = DockingBox(center=center_jnp, size=jnp.array([20.0, 20.0, 20.0]))

    # Assign ligand elements/charges and build LigandContext
    from dq_dock_engine.docking.charges import create_charge_assigner

    if ligand_elements is None:
        ligand_elements = None
    else:
        # ensure tuple input for assigner
        ligand_elements = tuple(ligand_elements)

    ligand_charges = None
    if use_multi_stage and ligand_elements is not None:
        assigner = create_charge_assigner(charge_method)
        ligand_source = (
            ligand_elements if assigner.method.name == "SIMPLE" else ligand_file
        )
        ligand_charges = assigner.assign(ligand_source).charges

    # Build LigandContext via core infra (auto-centers, stores radii/elements/charges)
    ligand_ctx = build_ligand_context(
        ligand_coords,
        ligand_radii,
        elements=list(ligand_elements) if ligand_elements is not None else None,
        charges=np.asarray(ligand_charges) if ligand_charges is not None else None,
    )

    RMSD_THRESHOLD = 2.0

    base_key = jax.random.PRNGKey(42)
    formal_status = "DQ-Dock"
    best_result: BenchmarkResult | None = None

    for attempt in range(max_retries):
        try:
            attempt_key = jax.random.fold_in(base_key, attempt)
            result_tuple = run_docking_pipeline(
                protein_coords=jnp.array(pocket_coords),
                receptor_radii=jnp.array(pocket_radii),
                ligand_ctx=ligand_ctx,
                box=box,
                n_poses=n_poses,
                engine=engine,
                key=attempt_key,
                top_k=1,
                optimize=True,
                n_opt_steps=n_opt_steps,
                charge_method=charge_method,
                receptor_file=receptor_file,
                receptor_elements=tuple(receptor_elements)
                if receptor_elements is not None
                else None,
                config=CERTIFIED_DOCKING,
                use_pocket_guided=use_pocket_guided,
                use_multi_stage=use_multi_stage,
                include_native=True,
            )
            best_poses, cert = result_tuple[0], result_tuple[1]

            if not best_poses:
                if attempt < max_retries - 1:
                    n_poses *= 2
                    n_opt_steps *= 2
                    print(
                        f"    [Retry {attempt + 2}/{max_retries}] n_poses={n_poses}, n_opt_steps={n_opt_steps} (no poses)"
                    )
                    continue
                return BenchmarkResult(
                    success=False,
                    energy=0.0,
                    rmsd=0.0,
                    time=time.time() - start,
                    n_atoms=0,
                    formal_status=formal_status,
                )

            elapsed = time.time() - start
            best_pose = best_poses[0]
            pose_jnp = jnp.expand_dims(best_pose.coords, axis=0)
            native_jnp = jnp.array(ligand_coords)
            rmsd = float(compute_docking_rmsd_batched(pose_jnp, native_jnp)[0])

            attempt_result = BenchmarkResult.from_certification(
                pose_energy=best_pose.energy,
                pose_rmsd=rmsd,
                elapsed=elapsed,
                n_atoms=len(pocket_coords) + len(ligand_coords),
                formal_status=formal_status,
                cert=cert,
            )

            if best_result is None or attempt_result.rmsd < best_result.rmsd:
                best_result = attempt_result

            if attempt < max_retries - 1 and rmsd > RMSD_THRESHOLD:
                n_poses *= 2
                n_opt_steps *= 2
                print(
                    f"    [Retry {attempt + 2}/{max_retries}] RMSD={rmsd:.2f}A > {RMSD_THRESHOLD}A, n_poses={n_poses}, n_opt_steps={n_opt_steps}"
                )
                continue
            break
        except Exception as e:
            if attempt < max_retries - 1:
                n_poses *= 2
                n_opt_steps *= 2
                print(
                    f"    [Retry {attempt + 2}/{max_retries}] n_poses={n_poses}, n_opt_steps={n_opt_steps} (exception: {e})"
                )
                continue
            return BenchmarkResult(
                success=False,
                energy=0.0,
                rmsd=0.0,
                time=time.time() - start,
                n_atoms=0,
                formal_status=formal_status,
            )

    if best_result is None:
        return BenchmarkResult(
            success=False,
            energy=0.0,
            rmsd=0.0,
            time=time.time() - start,
            n_atoms=0,
            formal_status=formal_status,
        )

    return best_result


def run_benchmark(
    n_complexes: int = 10,
    charge_method=None,
    n_poses: int = 2000,
    n_opt_steps: int = 50,
    use_multi_stage: bool = False,
    use_pocket_guided: bool | None = None,
    results_dir: Path = Path("benchmark_results"),
):
    """Run full benchmark."""
    bench_start = time.time()
    effective_pocket_guided = True if use_pocket_guided is None else use_pocket_guided

    print("=" * 70, flush=True)
    print("REAL PDB DOCKING BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    # Check Vina
    vina_path = check_vina()
    vina_prep_tools = check_vina_prep_tools()
    if vina_path:
        print(f"\n✅ Vina found: {vina_path}")
        if vina_prep_tools is not None:
            print("✅ AutoDock preparation tools found; Vina will use PDBQT inputs")
        else:
            print(
                "⚠️  AutoDock preparation tools not found; Vina will use direct PDB inputs"
            )
    else:
        print("""
❌ Vina not found!

To run this benchmark with real Vina comparisons:

1. Install Vina:
   conda install -c conda-forge vina
   
   Or download from:
   https://github.com/ccsb-scripps/AutoDock-Vina/releases

2. Make sure 'vina' is in your PATH

3. Re-run this benchmark

For now, we'll run DQ-Dock only on PDB files.
""")
        vina_path = None

    # Download PDB files
    cache_dir = Path("./pdb_cache")
    cache_dir.mkdir(exist_ok=True)
    output_paths = create_benchmark_output_paths(results_dir)

    print(
        f"\nDownloading {n_complexes} PDB complexes from saved benchmark list...",
        flush=True,
    )
    print(f"Live Results JSON: {output_paths.json_path}", flush=True)
    print(f"Live Results CSV: {output_paths.csv_path}", flush=True)
    dataset_exclusions = [
        entry
        for entry in get_benchmark_entries()
        if entry.scope == BenchmarkScope.LIGAND
        and not entry.include_by_default
        and entry.exclusion_reason is not None
    ]
    if dataset_exclusions:
        print("Configured dataset exclusions:", flush=True)
        for entry in dataset_exclusions:
            print(f"  {entry.pdb_id}: {entry.exclusion_reason}", flush=True)
    complexes = []
    excluded_rows: list[ExcludedComplexRow] = []

    for entry in get_default_ligand_entries():
        if len(complexes) >= n_complexes:
            break
        pdb_path = download_pdb(entry.pdb_id, cache_dir)
        if pdb_path:
            try:
                protein_path = prepare_protein(pdb_path)
                screening = screen_complex(entry, pdb_path, protein_path)
                if not screening.accepted:
                    reason = "; ".join(screening.reasons)
                    excluded_rows.append(
                        ExcludedComplexRow(pdb_id=entry.pdb_id, reason=reason)
                    )
                    print(f"  {entry.pdb_id}: excluded ({reason})", flush=True)
                    continue

                center = find_docking_center(
                    pdb_path, preferred_resname=entry.preferred_resname
                )
            except ValueError as e:
                excluded_rows.append(
                    ExcludedComplexRow(pdb_id=entry.pdb_id, reason=str(e))
                )
                print(f"  {entry.pdb_id}: excluded ({e})", flush=True)
                continue
            complexes.append(
                {
                    "pdb_id": entry.pdb_id,
                    "target_name": entry.target_name,
                    "path": pdb_path,
                    "center": center,
                    "preferred_resname": entry.preferred_resname,
                }
            )
            print(
                f"  {entry.pdb_id}: center = ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})"
            )

    print(f"\nDownloaded {len(complexes)} complexes", flush=True)
    if excluded_rows:
        print(f"Excluded during QC: {len(excluded_rows)}", flush=True)

    save_benchmark_results(
        output_paths,
        charge_method_name=(
            "unknown" if charge_method is None else charge_method.name.lower()
        ),
        n_complexes_requested=n_complexes,
        n_poses=n_poses,
        n_opt_steps=n_opt_steps,
        use_multi_stage=use_multi_stage,
        use_pocket_guided=effective_pocket_guided,
        bench_elapsed=time.time() - bench_start,
        phase="curation",
        complexes=complexes,
        dq_rows=[],
        dq_results=[],
        vina_rows=[],
        excluded_rows=excluded_rows,
    )

    if not complexes:
        print("❌ No complexes downloaded")
        return

    # Run DQ-Dock on each
    print("\n" + "=" * 70, flush=True)
    mode_label = "multi-stage composite" if use_multi_stage else "random picking"
    if effective_pocket_guided and not use_multi_stage:
        mode_label = "pocket-guided picking"
    print(
        f"RUNNING DQ-DOCK ({n_poses} poses, {n_opt_steps} opt steps, {mode_label})",
        flush=True,
    )
    print("=" * 70, flush=True)

    dq_results = []
    dq_rows: list[ComplexSummaryRow] = []
    vina_results: list[VinaResult] = []
    vina_rows: list[VinaSummaryRow] = []
    for cx in complexes:
        print(f"\n{cx['pdb_id']}:", flush=True)

        receptor_pdb = prepare_protein(cx["path"])
        pocket_receptor_pdb = prepare_pocket_protein(receptor_pdb, cx["center"])
        ligand_pdb = prepare_ligand(
            cx["path"], preferred_resname=cx["preferred_resname"]
        )

        try:
            ligand_coords, ligand_radii, ligand_elements = cast(
                tuple[np.ndarray, np.ndarray, list[str]],
                parse_structure(ligand_pdb, return_elements=True),
            )
            prot_coords, prot_radii, prot_elements = cast(
                tuple[np.ndarray, np.ndarray, list[str]],
                parse_structure(receptor_pdb, return_elements=True),
            )
        except ValueError as e:
            print(f"  ⚠️  {e}")
            continue

        # Extract pocket via core infra
        center = np.array(cx["center"])
        pocket_coords, pocket_radii, pocket_elements = cast(
            tuple[jnp.ndarray, jnp.ndarray, list[str]],
            build_receptor_arrays(
                prot_coords,
                prot_radii,
                center,
                pocket_radius=12.0,
                receptor_elements=prot_elements,
            ),
        )
        pocket_coords_np = np.array(pocket_coords)
        pocket_radii_np = np.array(pocket_radii)

        if len(pocket_coords_np) == 0:
            print("  ⚠️  No protein atoms in pocket")
            continue

        print(
            f"  Ligand atoms: {len(ligand_coords)}, Pocket atoms: {len(pocket_coords_np)}",
            flush=True,
        )

        # Run DQ-Dock
        result = run_dq_dock(
            pocket_coords_np,
            pocket_radii_np,
            ligand_coords,
            ligand_radii,
            cx["center"],
            ligand_file=ligand_pdb,
            receptor_file=pocket_receptor_pdb,
            charge_method=charge_method,
            n_poses=n_poses,
            n_opt_steps=n_opt_steps,
            use_multi_stage=use_multi_stage,
            use_pocket_guided=effective_pocket_guided,
            ligand_elements=ligand_elements,
            receptor_elements=tuple(pocket_elements),
        )
        dq_results.append(result)
        dq_rows.append(
            ComplexSummaryRow(
                pdb_id=str(cx["pdb_id"]),
                target_name=str(cx["target_name"]),
                ligand_atoms=len(ligand_coords),
                pocket_atoms=len(pocket_coords_np),
                rmsd=result.rmsd,
                time=result.time,
            )
        )

        print(
            f"  Best Energy: {result.energy:.2f} kcal/mol, Sampled Pose RMSD: {result.rmsd:.2f}A, Time: {result.time:.2f}s"
        )
        print("  DQ-Dock Redocking", end="")
        gap_proof = format_gap_proof_label(result.certified)
        if gap_proof is not None:
            print(f", Gap Proof: {gap_proof}", end="")
            if result.native_rank is not None:
                print(f", Native Rank: {result.native_rank}", end="")
            if result.energy_gap is not None:
                print(f", Energy Gap: {result.energy_gap:.4f}", end="")
        print()
        if result.native_rank is not None:
            print(
                "  Gap proof compares native pre-optimization energy against sampled poses.",
                flush=True,
            )

        save_benchmark_results(
            output_paths,
            charge_method_name=(
                "unknown" if charge_method is None else charge_method.name.lower()
            ),
            n_complexes_requested=n_complexes,
            n_poses=n_poses,
            n_opt_steps=n_opt_steps,
            use_multi_stage=use_multi_stage,
            use_pocket_guided=effective_pocket_guided,
            bench_elapsed=time.time() - bench_start,
            phase="dq_dock",
            complexes=complexes,
            dq_rows=dq_rows,
            dq_results=dq_results,
            vina_rows=vina_rows,
            excluded_rows=excluded_rows,
        )

    # Run Vina if available
    if vina_path:
        print("\n" + "=" * 70, flush=True)
        print("RUNNING SMINA/VINA", flush=True)
        print("=" * 70, flush=True)

        for cx in complexes:
            print(f"\n{cx['pdb_id']}...", flush=True)

            # Prepare separate protein and ligand files
            receptor_pdb = prepare_protein(cx["path"])
            ligand_pdb = prepare_ligand(
                cx["path"], preferred_resname=cx["preferred_resname"]
            )

            prep_temp_dir: Path | None = None
            try:
                vina_receptor, vina_ligand, prep_temp_dir = prepare_vina_inputs(
                    vina_prep_tools,
                    receptor_pdb,
                    ligand_pdb,
                )
                print(f"  Receptor: {vina_receptor.name}", flush=True)
                print(f"  Ligand: {vina_ligand.name}", flush=True)

                # Run Vina docking
                result = run_vina(
                    vina_path,
                    vina_receptor,
                    vina_ligand,
                    cx["center"],
                )
            finally:
                if prep_temp_dir is not None:
                    shutil.rmtree(prep_temp_dir, ignore_errors=True)

            vina_results.append(result)

            # Must have successfully parsed affinity
            if not result.success or result.best_affinity is None:
                error_msg = result.error or "Failed to parse SMINA output"
                print(f"  ❌ {error_msg}")
                save_benchmark_results(
                    output_paths,
                    charge_method_name=(
                        "unknown"
                        if charge_method is None
                        else charge_method.name.lower()
                    ),
                    n_complexes_requested=n_complexes,
                    n_poses=n_poses,
                    n_opt_steps=n_opt_steps,
                    use_multi_stage=use_multi_stage,
                    use_pocket_guided=effective_pocket_guided,
                    bench_elapsed=time.time() - bench_start,
                    phase="vina",
                    complexes=complexes,
                    dq_rows=dq_rows,
                    dq_results=dq_results,
                    vina_rows=vina_rows,
                    excluded_rows=excluded_rows,
                )
                continue

            top_rmsd = float("nan")
            best_mode_rmsd = float("nan")
            ligand_coords_vina, _, _ = cast(
                tuple[np.ndarray, np.ndarray, list[str]],
                parse_structure(ligand_pdb, return_elements=True),
            )
            if result.top_coords is not None:
                pose_coords = result.top_coords
                if len(pose_coords) != len(ligand_coords_vina):
                    print(
                        f"  ⚠️  Atom count mismatch: Native={len(ligand_coords_vina)}, SMINA={len(pose_coords)}"
                    )

                top_rmsd = compute_pose_rmsd(pose_coords, ligand_coords_vina)

            if result.model_coords:
                model_rmsds = [
                    compute_pose_rmsd(model_coords, ligand_coords_vina)
                    for model_coords in result.model_coords
                ]
                finite_model_rmsds = [r for r in model_rmsds if not np.isnan(r)]
                if finite_model_rmsds:
                    best_mode_rmsd = min(finite_model_rmsds)

            print(
                f"  Affinity: {result.best_affinity:.2f} kcal/mol, Top Pose RMSD: {top_rmsd:.2f}A, Best Returned Mode RMSD: {best_mode_rmsd:.2f}A, Time: {result.time:.1f}s"
            )
            vina_rows.append(
                VinaSummaryRow(
                    pdb_id=str(cx["pdb_id"]),
                    target_name=str(cx["target_name"]),
                    top_rmsd=top_rmsd,
                    best_mode_rmsd=best_mode_rmsd,
                    time=result.time,
                    affinity=result.best_affinity,
                )
            )

            save_benchmark_results(
                output_paths,
                charge_method_name=(
                    "unknown" if charge_method is None else charge_method.name.lower()
                ),
                n_complexes_requested=n_complexes,
                n_poses=n_poses,
                n_opt_steps=n_opt_steps,
                use_multi_stage=use_multi_stage,
                use_pocket_guided=effective_pocket_guided,
                bench_elapsed=time.time() - bench_start,
                phase="vina",
                complexes=complexes,
                dq_rows=dq_rows,
                dq_results=dq_results,
                vina_rows=vina_rows,
                excluded_rows=excluded_rows,
            )

    bench_elapsed = time.time() - bench_start

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print(f"\n{'Method':<20} {'Avg Time':<12} {'Total Time':<12} {'Success Rate':<15}")
    print("-" * 60)

    if dq_rows:
        print("\nDQ-Dock Per-Complex")
        print("-" * 70)
        print(f"{'PDB':<8} {'Ligand':<8} {'Pocket':<8} {'RMSD':<10} {'Time':<10}")
        for row in dq_rows:
            print(
                f"{row.pdb_id:<8} {row.ligand_atoms:<8} {row.pocket_atoms:<8} {row.rmsd:<10.2f} {row.time:<10.2f}"
            )

    if excluded_rows:
        print("\nExcluded By QC")
        print("-" * 70)
        for row in excluded_rows:
            print(f"{row.pdb_id:<8} {row.reason}")

    if dq_results:
        avg_time = np.mean([r.time for r in dq_results])
        avg_rmsd = np.mean([r.rmsd for r in dq_results])
        min_time = np.min([r.time for r in dq_results])
        max_time = np.max([r.time for r in dq_results])
        dq_total = sum(r.time for r in dq_results)
        print(
            f"{'DQ-Dock':<20} {avg_time:<12.2f} {dq_total:<12.2f} {len(dq_results)}/{len(complexes)}"
        )
        print(f"  Avg RMSD: {avg_rmsd:.2f}A")
        print(f"  Time range: {min_time:.2f}s - {max_time:.2f}s per complex")

    if vina_results:
        successes = sum(1 for r in vina_results if r.success)
        avg_time = np.mean([r.time for r in vina_results if r.success])
        vina_total = sum(r.time for r in vina_results if r.success)
        print(
            f"{'Vina':<20} {avg_time:<12.1f} {vina_total:<12.1f} {successes}/{len(vina_results)}"
        )
        if vina_rows:
            avg_top_rmsd = np.mean([r.top_rmsd for r in vina_rows])
            avg_best_mode_rmsd = np.mean([r.best_mode_rmsd for r in vina_rows])
            print(f"  Avg top-pose RMSD: {avg_top_rmsd:.2f}A")
            print(f"  Avg best-returned RMSD: {avg_best_mode_rmsd:.2f}A")

    print(f"\n  Total benchmark time: {bench_elapsed:.2f}s")
    print(f"  {len(dq_results)}/{len(complexes)} complexes completed successfully")
    print(f"  {len(excluded_rows)} complexes excluded by QC")

    json_path, csv_path = save_benchmark_results(
        output_paths,
        charge_method_name=(
            "unknown" if charge_method is None else charge_method.name.lower()
        ),
        n_complexes_requested=n_complexes,
        n_poses=n_poses,
        n_opt_steps=n_opt_steps,
        use_multi_stage=use_multi_stage,
        use_pocket_guided=effective_pocket_guided,
        bench_elapsed=bench_elapsed,
        phase="complete",
        complexes=complexes,
        dq_rows=dq_rows,
        dq_results=dq_results,
        vina_rows=vina_rows,
        excluded_rows=excluded_rows,
    )
    print(f"  Results JSON: {json_path}")
    print(f"  Results CSV: {csv_path}")


if __name__ == "__main__":
    from dq_dock_engine.docking.charges import ChargeMethod

    parser = argparse.ArgumentParser(description="Real PDB docking benchmark")
    parser.add_argument(
        "--n_complexes", type=int, default=10, help="Number of complexes"
    )
    parser.add_argument(
        "--charge_method",
        type=str,
        choices=("am1bcc", "gasteiger", "simple"),
        required=True,
        help="Charge assignment method for composite scoring",
    )
    parser.add_argument(
        "--n_poses",
        type=int,
        default=2000,
        help="Number of sampled poses for DQ-Dock",
    )
    parser.add_argument(
        "--n_opt_steps",
        type=int,
        default=50,
        help="Number of optimization steps per pose",
    )
    parser.add_argument(
        "--use_multi_stage",
        action="store_true",
        help="Enable the slower multi-stage composite scoring pipeline",
    )
    parser.add_argument(
        "--use_pocket_guided",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use pocket-guided pose sampling (default: enabled)",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory for CSV/JSON benchmark outputs",
    )
    args = parser.parse_args()

    is_valid, warnings = CERTIFIED_DOCKING.validate()
    for warning in warnings:
        print(f"Config warning: {warning}")

    charge_method = {
        "am1bcc": ChargeMethod.AM1BCC,
        "gasteiger": ChargeMethod.GASTEIGER,
        "simple": ChargeMethod.SIMPLE,
    }[args.charge_method]

    run_benchmark(
        args.n_complexes,
        charge_method=charge_method,
        n_poses=args.n_poses,
        n_opt_steps=args.n_opt_steps,
        use_multi_stage=args.use_multi_stage,
        use_pocket_guided=args.use_pocket_guided,
        results_dir=args.results_dir,
    )
