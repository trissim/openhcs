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
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
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
    FormalProofStatus,
)
from dq_dock_engine.docking.pipeline import run_docking_pipeline
from dq_dock_engine.docking.metrics import (
    compute_rmsd_batched,
    compute_docking_rmsd_batched,
)
from dq_dock_engine.docking_config import (
    DockingConfig,
    DockingMode,
    CERTIFIED_DOCKING,
    HEURISTIC_SCREENING,
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


def resolve_formal_status(
    config: DockingConfig,
) -> FormalProofStatus:
    """Return the formal status based on docking configuration mode."""
    match config.mode:
        case DockingMode.CERTIFIED:
            return FormalProofStatus.CERTIFIED
        case DockingMode.HEURISTIC:
            return FormalProofStatus.HEURISTIC


def download_pdb(pdb_id: str, cache_dir: Path) -> Path:
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


def prepare_ligand(pdb_path: Path) -> Path:
    """Prepare ligand for Vina - extract primary ligand HETATM records from PDB file.

    Raises error if no ligand found in the PDB.
    """
    ligand_path = pdb_path.parent / f"{pdb_path.stem}_ligand.pdb"

    # Check if already extracted
    if ligand_path.exists():
        # Remove it if we are re-running to fix old bad extractions
        ligand_path.unlink()

    # Auto-detect drug by finding most common non-water HETATM
    resname_counts = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM"):
                resname = line[17:20].strip()
                if resname not in ["HOH", "DOD", "WAT"]:
                    resname_counts[resname] = resname_counts.get(resname, 0) + 1

    if not resname_counts:
        raise ValueError(f"No ligand found in {pdb_path}")

    target_resname = max(resname_counts.items(), key=lambda x: x[1])[0]

    ligand_lines = []
    with open(pdb_path) as f:
        for line in f:
            if (line.startswith("HETATM") or line.startswith("ATOM")) and line[
                17:20
            ].strip() == target_resname:
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


def find_docking_center(pdb_path: Path) -> tuple:
    """Find docking center from ligand in PDB."""
    from dq_dock_engine.docking.pdb_io import parse_structure

    try:
        coords, _ = parse_structure(pdb_path, return_elements=False)
        return compute_pocket_center(coords)
    except ValueError:
        pass

    # Fallback: center of protein
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])

    if coords:
        return tuple(np.mean(coords, axis=0))

    return (0.0, 0.0, 0.0)


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


def run_vina(
    vina_path: str,
    receptor: Path,
    ligand: Path,
    center: tuple,
    size: tuple = (20, 20, 20),
    exhaustiveness: int = 1,  # Reduced for faster benchmarking
    n_models: int = 3,  # Reduced for faster benchmarking
) -> Dict:
    """Run Vina docking."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as out:
        output_path = Path(out.name)

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
        "--out",
        str(output_path),
    ]

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start

        if result.returncode != 0:
            return {"success": False, "error": result.stderr[:200], "time": elapsed}

        # Parse best affinity - look for mode line after results header
        # Format: "mode |   affinity | dist from best mode" followed by data
        best_affinity = None
        in_results = False
        for line in result.stdout.split("\n"):
            # Skip until we see the results header
            if "mode |" in line and "affinity" in line:
                in_results = True
                continue
            # Skip separator line
            if in_results and line.startswith("-----"):
                continue
            # Parse data lines (start with digit, have affinity value)
            if in_results and line.strip() and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        best_affinity = float(parts[1])
                        break
                    except ValueError:
                        pass
    finally:
        if output_path.exists():
            output_path.unlink()


def parse_smina_output(pdb_path: Path) -> Optional[np.ndarray]:
    """Extract first model's heavy atom coordinates from smina output."""
    from dq_dock_engine.docking.pdb_io import parse_structure

    try:
        # PDB files from smina have multiple models; parse_structure reads first one until ENDMDL/END
        coords, _ = parse_structure(
            pdb_path, strip_hydrogens=True, return_elements=False
        )
        return coords
    except Exception:
        return None


def run_vina(
    vina_path: str,
    receptor: Path,
    ligand: Path,
    center: tuple,
    size: tuple = (20, 20, 20),
    exhaustiveness: int = 1,
    n_models: int = 3,
) -> Dict:
    """Run smina docking."""
    import tempfile

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
        "--out",
        str(output_path),
    ]

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start

        if result.returncode != 0:
            return {"success": False, "error": result.stderr[:200], "time": elapsed}

        best_affinity = None
        in_results = False
        for line in result.stdout.split("\n"):
            if "mode |" in line and "affinity" in line:
                in_results = True
                continue
            if in_results and line.strip() and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        best_affinity = float(parts[1])
                        break
                    except ValueError:
                        pass

        best_coords = None
        if output_path.exists() and best_affinity is not None:
            best_coords = parse_smina_output(output_path)

        return {
            "success": True,
            "best_affinity": best_affinity,
            "time": elapsed,
            "best_coords": best_coords,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "time": time.time() - start}
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
    config: DockingConfig,
    n_poses: int = 2000,
    use_multi_stage: bool = False,
    use_pocket_guided: bool = False,
    engine: ScoringEngine = ScoringEngine.INTERNAL_LJ,
    ligand_elements: list[str] | tuple[str, ...] | None = None,
    receptor_elements: list[str] | tuple[str, ...] | None = None,
) -> Dict:
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

    key = jax.random.PRNGKey(42)
    formal_status = resolve_formal_status(config)
    best_poses = run_docking_pipeline(
        protein_coords=jnp.array(pocket_coords),
        receptor_radii=jnp.array(pocket_radii),
        ligand_ctx=ligand_ctx,
        box=box,
        n_poses=n_poses,
        engine=engine,
        key=key,
        top_k=1,
        optimize=False,
        charge_method=charge_method,
        receptor_file=receptor_file,
        receptor_elements=tuple(receptor_elements)
        if receptor_elements is not None
        else None,
        use_pocket_guided=use_pocket_guided,
        use_multi_stage=use_multi_stage,
    )

    if not best_poses:
        return {"success": False, "error": "No poses generated"}

    best_pose = best_poses[0]

    pose_jnp = jnp.expand_dims(best_pose.coords, axis=0)
    native_jnp = jnp.array(ligand_coords)
    rmsd = float(compute_docking_rmsd_batched(pose_jnp, native_jnp)[0])

    elapsed = time.time() - start

    return {
        "success": True,
        "energy": best_pose.energy,
        "rmsd": rmsd,
        "time": elapsed,
        "n_atoms": len(pocket_coords) + len(ligand_coords),
        "formal_status": formal_status.name,
    }


def run_benchmark(
    n_complexes: int = 10,
    charge_method=None,
    config: DockingConfig = HEURISTIC_SCREENING,
    n_poses: int = 2000,
    use_multi_stage: bool = False,
    use_pocket_guided: bool = False,
):
    """Run full benchmark."""

    print("=" * 70, flush=True)
    print("REAL PDB DOCKING BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    # Check Vina
    vina_path = check_vina()
    if vina_path:
        print(f"\n✅ Vina found: {vina_path}")
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

    print(f"\nDownloading {n_complexes} PDB complexes...", flush=True)
    complexes = []

    for pdb_id in TEST_PDB_IDS[:n_complexes]:
        pdb_path = download_pdb(pdb_id, cache_dir)
        if pdb_path:
            center = find_docking_center(pdb_path)
            complexes.append(
                {
                    "pdb_id": pdb_id,
                    "path": pdb_path,
                    "center": center,
                }
            )
            print(
                f"  {pdb_id}: center = ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})"
            )

    print(f"\nDownloaded {len(complexes)} complexes", flush=True)

    if not complexes:
        print("❌ No complexes downloaded")
        return

    # Run DQ-Dock on each
    print("\n" + "=" * 70, flush=True)
    mode_label = "multi-stage composite" if use_multi_stage else "random picking"
    if use_pocket_guided and not use_multi_stage:
        mode_label = "pocket-guided picking"
    formal_status = resolve_formal_status(config)
    print(
        f"RUNNING DQ-DOCK ({n_poses} batched poses, {mode_label}, {config.mode.name} mode, {formal_status.name})",
        flush=True,
    )
    print("=" * 70, flush=True)

    dq_results = []
    for cx in complexes:
        print(f"\n{cx['pdb_id']}:", flush=True)

        # 1. Use core pdb_io infrastructure for all parsing + atom typing
        from dq_dock_engine.docking.pdb_io import parse_structure, build_receptor_arrays

        receptor_pdb = prepare_protein(cx["path"])
        pocket_receptor_pdb = prepare_pocket_protein(receptor_pdb, cx["center"])
        ligand_pdb = prepare_ligand(cx["path"])

        try:
            ligand_coords, ligand_radii, ligand_elements = parse_structure(
                ligand_pdb, return_elements=True
            )
            prot_coords, prot_radii, prot_elements = parse_structure(
                receptor_pdb, return_elements=True
            )
        except ValueError as e:
            print(f"  ⚠️  {e}")
            continue

        # Extract pocket via core infra
        center = np.array(cx["center"])
        pocket_coords, pocket_radii, pocket_elements = build_receptor_arrays(
            prot_coords,
            prot_radii,
            center,
            pocket_radius=12.0,
            receptor_elements=prot_elements,
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
            config=config,
            n_poses=n_poses,
            use_multi_stage=use_multi_stage,
            use_pocket_guided=use_pocket_guided,
            ligand_elements=ligand_elements,
            receptor_elements=tuple(pocket_elements),
        )
        dq_results.append(result)

        print(
            f"  Best Energy: {result['energy']:.2f} kcal/mol, Native RMSD: {result['rmsd']:.2f}Å, Time: {result['time']:.2f}s"
        )
        print(f"  Formal Status: {result['formal_status']}")

    # Run Vina if available
    vina_results = []
    if vina_path:
        print("\n" + "=" * 70, flush=True)
        print("RUNNING SMINA/VINA", flush=True)
        print("=" * 70, flush=True)

        for cx in complexes:
            print(f"\n{cx['pdb_id']}...", flush=True)

            # Prepare separate protein and ligand files
            receptor_pdb = prepare_protein(cx["path"])
            ligand_pdb = prepare_ligand(cx["path"])

            print(f"  Receptor: {receptor_pdb.name}", flush=True)
            print(f"  Ligand: {ligand_pdb.name}", flush=True)

            # Run Vina docking
            result = run_vina(
                vina_path,
                str(receptor_pdb),  # receptor (PDB)
                str(ligand_pdb),  # ligand (PDB)
                cx["center"],
            )

            vina_results.append(result)

            # Must have successfully parsed affinity
            if not result["success"] or result["best_affinity"] is None:
                error_msg = result.get("error", "Failed to parse SMINA output")
                print(f"  ❌ {error_msg}")
                continue

            # Compute SMINA RMSD
            smina_rmsd = float("nan")
            if result.get("best_coords") is not None and ligand_coords is not None:
                pose_coords = result["best_coords"]
                if len(pose_coords) != len(ligand_coords):
                    print(
                        f"  ⚠️  Atom count mismatch: Native={len(ligand_coords)}, SMINA={len(pose_coords)}"
                    )

                min_len = min(len(pose_coords), len(ligand_coords))
                if min_len > 0:
                    pose_jnp = jnp.expand_dims(pose_coords[:min_len], axis=0)
                    native_jnp = jnp.array(ligand_coords[:min_len])
                    smina_rmsd = float(
                        compute_docking_rmsd_batched(pose_jnp, native_jnp)[0]
                    )

            print(
                f"  Affinity: {result['best_affinity']:.2f} kcal/mol, Native RMSD: {smina_rmsd:.2f}Å, Time: {result['time']:.1f}s"
            )
            result["rmsd"] = smina_rmsd

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print(f"\n{'Method':<20} {'Avg Time':<12} {'Success Rate':<15}")
    print("-" * 50)

    if dq_results:
        avg_time = np.mean([r["time"] for r in dq_results])
        print(f"{'DQ-Dock':<20} {avg_time:<12.2f} {len(dq_results)}/{len(complexes)}")

        print("\nDQ-Dock by Formal Status")
        print("-" * 50)
        grouped_results: dict[str, list[dict]] = {}
        for result in dq_results:
            grouped_results.setdefault(result["formal_status"], []).append(result)
        for status, results in grouped_results.items():
            avg_status_time = np.mean([r["time"] for r in results])
            print(
                f"{status:<20} {avg_status_time:<12.2f} {len(results)}/{len(dq_results)}"
            )

    if vina_results:
        successes = sum(1 for r in vina_results if r["success"])
        avg_time = np.mean([r["time"] for r in vina_results if r["success"]])
        print(f"{'Vina':<20} {avg_time:<12.1f} {successes}/{len(vina_results)}")

    print("""
Note: This is a simplified benchmark. For production use:
- Use prepare_receptor4.py and prepare_ligand4.py for Vina
- Provide separate ligand files
- Use proper scoring functions
""")


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
        "--use_multi_stage",
        action="store_true",
        help="Enable the slower multi-stage composite scoring pipeline",
    )
    parser.add_argument(
        "--use_pocket_guided",
        action="store_true",
        help="Use pocket-guided pose sampling instead of uniform random sampling",
    )
    parser.add_argument(
        "--certified",
        action="store_true",
        help="Use CERTIFIED_DOCKING mode (Lean proofs + NIST constants only)",
    )
    args = parser.parse_args()

    config = CERTIFIED_DOCKING if args.certified else HEURISTIC_SCREENING
    is_valid, warnings = config.validate()
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
        config=config,
        n_poses=args.n_poses,
        use_multi_stage=args.use_multi_stage,
        use_pocket_guided=args.use_pocket_guided,
    )
