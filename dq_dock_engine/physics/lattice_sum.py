import jax.numpy as jnp

"""
Lattice Sum Convergence for power-law potentials.
Direct translation of Tractability/LatticeSum.lean.

Key theorem (MP1): For s > 3, the 3D lattice sum tail decays as O(1/R^(s-3)).
Justifies cutoff approximation for LJ 6-12 potential.
"""

def lattice_tail_bound(s: float, R: float) -> float:
    """
    Upper bound on the lattice tail sum: Σ_{‖n‖>R} 1/‖n‖^s.
    From LatticeSum.lean::lattice_sum_converges:
      latticeTailSum(s, R) ≤ M / R^(s-3) for s > 3.
    
    For LJ:
    - s=6: tail ≤ C/R³  (lj6_tail_bound)
    - s=12: tail ≤ C/R⁹  (lj12_tail_bound)
    """
    if s <= 3:
        raise ValueError(f"Lattice sum diverges for s ≤ 3, got s={s}")
    if R <= 0:
        raise ValueError(f"R must be positive, got R={R}")
    # Conservative bound constant (4π for sphere surface × safety factor)
    M = 4 * jnp.pi * 2.0  # Depends on lattice geometry
    return float(M / R ** (s - 3))

def lj6_cutoff_error(R: float) -> float:
    """Error from LJ r⁻⁶ cutoff at radius R. O(1/R³)."""
    return lattice_tail_bound(6.0, R)

def lj12_cutoff_error(R: float) -> float:
    """Error from LJ r⁻¹² cutoff at radius R. O(1/R⁹)."""
    return lattice_tail_bound(12.0, R)

def optimal_cutoff(target_error: float, s: float = 6.0) -> float:
    """Find minimum cutoff radius for desired error bound."""
    M = 4 * jnp.pi * 2.0
    return float((M / target_error) ** (1.0 / (s - 3)))
