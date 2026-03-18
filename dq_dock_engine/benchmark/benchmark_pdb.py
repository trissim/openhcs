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

from dq_dock_engine.docking.core import DockingBox, LigandContext, ScoringEngine
from dq_dock_engine.docking.pipeline import run_docking_pipeline
from dq_dock_engine.docking.metrics import compute_rmsd_batched, compute_docking_rmsd_batched


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
            if (line.startswith("HETATM") or line.startswith("ATOM")) and line[17:20].strip() == target_resname:
                ligand_lines.append(line)
    
    if not ligand_lines:
        raise ValueError(f"No ligand found in {pdb_path}")
    
    with open(ligand_path, "w") as f:
        f.writelines(ligand_lines)
    return ligand_path


def extract_ligand_coords(pdb_path: Path) -> Optional[np.ndarray]:
    """Extract heavy atom coordinates from PDB file (strips hydrogens)."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM") or line.startswith("ATOM"):
                # Strip hydrogens to ensure SMINA and DQ-Dock topologies match
                element = line[76:78].strip()
                if not element:
                    if line[12:16].strip().startswith("H"):
                        continue
                elif element == "H":
                    continue
                    
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    continue

    if coords:
        return np.array(coords)
    return None


def compute_pocket_center(ligand_coords: np.ndarray) -> tuple:
    """Compute center of ligand for docking box."""
    center = ligand_coords.mean(axis=0)
    return tuple(center)


def find_docking_center(pdb_path: Path) -> tuple:
    """Find docking center from ligand in PDB."""
    coords = extract_ligand_coords(pdb_path)
    if coords is not None:
        return compute_pocket_center(coords)
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

        # Parse best coordinates from the output PDB
        best_coords = None
        if output_path.exists() and best_affinity is not None:
            coords = []
            with open(output_path) as f:
                for line in f:
                    if line.startswith("ENDMDL"):
                        break
                    elif line.startswith("ATOM") or line.startswith("HETATM"):
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x, y, z])
                        except ValueError:
                            pass
            if coords:
                best_coords = np.array(coords)

        return {
            "success": True,
            "best_affinity": best_affinity,
            "time": elapsed,
            "output": result.stdout,
            "best_coords": best_coords,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout", "time": 300}
    except Exception as e:
        return {"success": False, "error": str(e), "time": 0}
    finally:
        if output_path.exists():
            output_path.unlink()


def run_dq_dock(
    protein_coords: np.ndarray,
    ligand_coords: np.ndarray,
    center: tuple,
) -> Dict:
    """Run true DQ-Dock pipeline on complex."""
    from dq_dock_engine.docking.core import DockingBox, LigandContext, ScoringEngine
    from dq_dock_engine.docking.pipeline import run_docking_pipeline
    from dq_dock_engine.docking.metrics import compute_docking_rmsd_batched
    
    start = time.time()
    
    # 1. Setup Data Structures
    center_jnp = jnp.array(center)
    box = DockingBox(center=center_jnp, size=jnp.array([20.0, 20.0, 20.0]))
    
    com = jnp.mean(jnp.array(ligand_coords), axis=0)
    # Center ligand at origin for proper rotation sampling
    base_coords = jnp.array(ligand_coords) - com
    ligand_ctx = LigandContext(base_coords=base_coords, center_of_mass=com)
    
    # 2. Run Pipeline (pure JAX batched generation + internal LJ scoring)
    key = jax.random.PRNGKey(42)
    best_poses = run_docking_pipeline(
        protein_coords=jnp.array(protein_coords),
        ligand_ctx=ligand_ctx,
        box=box,
        n_poses=2000,
        engine=ScoringEngine.INTERNAL_LJ,
        key=key,
        top_k=1
    )
    
    if not best_poses:
        return {"success": False, "error": "No poses generated"}
        
    best_pose = best_poses[0]
    
    # 3. Compute True Docking RMSD (absolute Cartesian error) to native crystal structure
    pose_jnp = jnp.expand_dims(best_pose.coords, axis=0)
    native_jnp = jnp.array(ligand_coords)
    # Get RMSD
    rmsd = float(compute_docking_rmsd_batched(pose_jnp, native_jnp)[0])
    
    elapsed = time.time() - start

    return {
        "success": True,
        "energy": best_pose.energy,
        "rmsd": rmsd,
        "time": elapsed,
        "n_atoms": len(protein_coords) + len(ligand_coords),
    }


def run_benchmark(n_complexes: int = 10):
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
    print("RUNNING DQ-DOCK (2000 batched poses/sec, JAX LJ scoring)", flush=True)
    print("=" * 70, flush=True)

    dq_results = []
    for cx in complexes:
        print(f"\n{cx['pdb_id']}:", flush=True)

        # 1. Prepare files and extract coordinates
        receptor_pdb = prepare_protein(cx["path"])
        ligand_pdb = prepare_ligand(cx["path"])
        
        ligand_coords = extract_ligand_coords(ligand_pdb)
        if ligand_coords is None or len(ligand_coords) == 0:
            print("  ⚠️  No ligand atoms extracted")
            continue
            
        prot_coords = []
        with open(receptor_pdb) as f:
            for line in f:
                if line.startswith("ATOM"):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    prot_coords.append([x, y, z])
                    
        prot_coords = np.array(prot_coords)

        # Define pocket (atoms near ligand region)
        center = np.array(cx["center"])
        distances = np.linalg.norm(prot_coords - center, axis=1)
        pocket_mask = distances < 12.0  # 12Å around center
        pocket_coords = prot_coords[pocket_mask]

        if len(pocket_coords) == 0:
            print("  ⚠️  No protein atoms in pocket")
            continue

        print(
            f"  Ligand atoms: {len(ligand_coords)}, Pocket atoms: {len(pocket_coords)}",
            flush=True,
        )

        # Run DQ-Dock
        result = run_dq_dock(pocket_coords, ligand_coords, cx["center"])
        dq_results.append(result)

        print(f"  Best Energy: {result['energy']:.2f} kcal/mol, Native RMSD: {result['rmsd']:.2f}Å, Time: {result['time']:.2f}s")

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
                str(ligand_pdb),    # ligand (PDB)
                cx["center"],
            )

            vina_results.append(result)

            # Must have successfully parsed affinity
            if not result["success"] or result["best_affinity"] is None:
                error_msg = result.get("error", "Failed to parse SMINA output")
                print(f"  ❌ {error_msg}")
                continue

            # Compute SMINA RMSD
            smina_rmsd = float('nan')
            if result.get("best_coords") is not None and ligand_coords is not None:
                if len(result["best_coords"]) != len(ligand_coords):
                    print(f"  ⚠️  Atom count mismatch: Native={len(ligand_coords)}, SMINA={len(result['best_coords'])} (Slicing to min length)")
                
                # Assume original atom order is preserved at the beginning of the file (SMINA usually appends added atoms)
                min_len = min(len(result["best_coords"]), len(ligand_coords))
                if min_len > 0:
                    pose_jnp = jnp.expand_dims(result["best_coords"][:min_len], axis=0)
                    native_jnp = jnp.array(ligand_coords[:min_len])
                    smina_rmsd = float(compute_docking_rmsd_batched(pose_jnp, native_jnp)[0])

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
    parser = argparse.ArgumentParser(description="Real PDB docking benchmark")
    parser.add_argument(
        "--n_complexes", type=int, default=10, help="Number of complexes"
    )
    args = parser.parse_args()

    run_benchmark(args.n_complexes)
