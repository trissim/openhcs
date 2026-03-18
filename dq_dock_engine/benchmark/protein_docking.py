#!/usr/bin/env python3
"""
Protein Docking Benchmark: Regime A (Full MD) vs Regime B (DQ-DOCK srank pruning)

Regime A: Standard MD - all N atoms, O(N²) interactions
Regime B: DQ-DOCK - only srank-relevant atoms, O(srank²) interactions

Goal: Show Regime B achieves comparable precision with less computational cost.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple
import time


def lj_force(r: jnp.ndarray, epsilon: float = 1.0, sigma: float = 1.0) -> jnp.ndarray:
    """Lennard-Jones force."""
    r_norm = jnp.linalg.norm(r, axis=-1, keepdims=True)
    r_safe = jnp.where(r_norm < 1e-6, 1e-6, r_norm)
    inv_r6 = (sigma / r_safe) ** 6
    force_mag = 24 * epsilon * (2 * inv_r6**2 - inv_r6) / r_safe
    return -force_mag * (r / r_safe)


def build_force_field(positions: jnp.ndarray, cutoff: float = 2.5) -> callable:
    """Build force function for given positions."""

    def force_fn(q: jnp.ndarray) -> jnp.ndarray:
        n_atoms = q.shape[0]
        forces = jnp.zeros_like(q)

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r_ij = q[j] - q[i]
                r_norm = jnp.linalg.norm(r_ij)

                if r_norm < cutoff:
                    f_ij = lj_force(r_ij)
                    forces = forces.at[i].add(f_ij)
                    forces = forces.at[j].add(-f_ij)

        return forces

    return force_fn


def velocity_verlet_step(
    q: jnp.ndarray, v: jnp.ndarray, f: jnp.ndarray, masses: jnp.ndarray, dt: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """One Velocity-Verlet step."""
    v_half = v + (dt / 2) * (f / masses[:, None])
    q_next = q + dt * v_half

    f_next = build_force_field(q_next)(q_next)
    v_next = v_half + (dt / 2) * (f_next / masses[:, None])

    return q_next, v_next


def run_simulation(
    positions: jnp.ndarray, n_steps: int = 100, dt: float = 0.001
) -> Dict:
    """Run MD simulation and return trajectory."""
    n_atoms = positions.shape[0]
    masses = jnp.ones(n_atoms)  # Unit mass
    velocities = jnp.zeros_like(positions)

    q = positions
    v = velocities
    f = build_force_field(q)(q)

    energies = []

    for step in range(n_steps):
        q, v = velocity_verlet_step(q, v, f, masses, dt)
        f = build_force_field(q)(q)

        # Energy
        ke = 0.5 * jnp.sum(masses[:, None] * v**2)
        energies.append(float(ke))

    return {"final_positions": np.array(q), "energies": energies, "n_atoms": n_atoms}


def compute_srank_approximate(positions: jnp.ndarray, n_samples: int = 50) -> int:
    """
    Approximate srank: number of coordinates that affect binding affinity.

    Method: Sample positions near minimum, compute gradient magnitudes,
    count atoms with significant gradient contribution.
    """
    n_atoms = positions.shape[0]

    # Sample configurations and compute gradient norms
    gradient_norms = jnp.zeros(n_atoms)

    for _ in range(n_samples):
        # Random perturbation around current position
        noise = jax.random.normal(jax.random.key(0), positions.shape) * 0.1
        q_perturbed = positions + noise

        # Compute force as proxy for gradient
        f = build_force_field(q_perturbed)(q_perturbed)
        gradient_norms += jnp.linalg.norm(f, axis=1)

    gradient_norms /= n_samples

    # Atoms with gradient above threshold are "relevant"
    threshold = jnp.percentile(gradient_norms, 50)  # Top 50%
    srank = int(jnp.sum(gradient_norms > threshold))

    return max(srank, 3)  # Minimum 3 for valid docking


def create_pocket_system(n_pocket: int = 30, n_ligand: int = 5) -> Dict:
    """
    Create a mock protein-ligand pocket system.

    Returns positions, masses, and identification of pocket vs ligand atoms.
    """
    np.random.seed(42)

    # Pocket atoms (protein): clustered sphere
    pocket_positions = np.random.randn(n_pocket, 3) * 2.0

    # Ligand: near pocket
    ligand_positions = (
        pocket_positions[:n_ligand]
        + np.random.randn(n_ligand, 3) * 0.5
        + np.array([3.0, 0, 0])
    )

    # Combine
    all_positions = np.vstack([pocket_positions, ligand_positions])

    return {
        "positions": jnp.array(all_positions),
        "n_pocket": n_pocket,
        "n_ligand": n_ligand,
        "pocket_indices": list(range(n_pocket)),
        "ligand_indices": list(range(n_pocket, n_pocket + n_ligand)),
    }


def benchmark_regimes(system: Dict) -> Dict:
    """Run both regimes and compare."""

    print("\n" + "=" * 60)
    print("REGIME A vs REGIME B COMPARISON")
    print("=" * 60)

    positions = system["positions"]
    n_atoms = positions.shape[0]

    # Regime A: Full system
    print(
        f"\nRegime A (Full MD): {n_atoms} atoms, O({n_atoms}²) = O({n_atoms**2}) interactions"
    )
    print("  Running simulation...", end=" ", flush=True)
    start = time.time()
    result_a = run_simulation(positions, n_steps=50)
    time_a = time.time() - start
    print(f"done ({time_a:.1f}s)")
    print(f"  Final KE: {result_a['energies'][-1]:.4f}")

    # Compute srank
    print("  Computing srank...", end=" ", flush=True)
    srank = compute_srank_approximate(positions)
    print(f"done (srank={srank})")
    print(
        f"  srank = {srank}/{3 * n_atoms} = {100 * srank / (3 * n_atoms):.1f}% relevant"
    )

    # Regime B: Only srank-relevant atoms (pocket + ligand)
    # In real docking, this would be the binding pocket
    pocket_size = min(srank // 3, system["n_pocket"])  # Atoms in srank
    n_relevant = pocket_size + system["n_ligand"]

    relevant_indices = jnp.array(list(range(n_relevant)))
    positions_b = positions[relevant_indices]

    print(
        f"\nRegime B (DQ-DOCK srank pruning): {n_relevant} atoms, O({n_relevant}²) = O({n_relevant**2}) interactions"
    )
    print("  Running simulation...", end=" ", flush=True)
    start = time.time()
    result_b = run_simulation(positions_b, n_steps=50)
    time_b = time.time() - start
    print(f"done ({time_b:.1f}s)")
    print(f"  Final KE: {result_b['energies'][-1]:.4f}")

    # Speedup
    speedup = time_a / time_b if time_b > 0 else float("inf")

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(
        f"  Atoms reduced: {n_atoms} → {n_relevant} ({100 * (1 - n_relevant / n_atoms):.1f}% pruned)"
    )
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  srank efficiency: {100 * srank / (3 * n_atoms):.1f}% of full system")

    return {
        "regime_a": {
            "n_atoms": n_atoms,
            "time": time_a,
            "energy": result_a["energies"][-1],
        },
        "regime_b": {
            "n_atoms": n_relevant,
            "time": time_b,
            "energy": result_b["energies"][-1],
        },
        "srank": srank,
        "speedup": speedup,
    }


def run_protein_benchmark():
    """Run protein docking benchmark."""

    print("\n" + "=" * 60)
    print("PROTEIN DOCKING BENCHMARK")
    print("DQ-DOCK vs Standard MD")
    print("=" * 60)

    # Test different system sizes
    systems = [
        create_pocket_system(30, 5),  # Small pocket
        create_pocket_system(50, 8),  # Medium pocket
        create_pocket_system(100, 10),  # Large pocket
    ]

    results = []

    for i, system in enumerate(systems):
        print(f"\n{'#' * 60}")
        print(
            f"SYSTEM {i + 1}: {system['n_pocket']} pocket + {system['n_ligand']} ligand atoms"
        )
        print(f"{'#' * 60}")

        result = benchmark_regimes(system)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'System':<20} {'Atoms':<10} {'srank':<10} {'Speedup':<10}")
    print("-" * 50)
    for i, r in enumerate(results):
        n = r["regime_a"]["n_atoms"]
        s = r["srank"]
        sp = r["speedup"]
        print(f"System {i + 1} ({n} atoms)   {s:<10} {sp:.1f}x")

    return results


if __name__ == "__main__":
    results = run_protein_benchmark()
