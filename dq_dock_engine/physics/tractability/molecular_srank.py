import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable

"""
Molecular Structural Rank — the Decision-Theoretic Backdoor.
Direct translation of Tractability/MolecularSrank.lean.

KEY THEOREM: For molecular docking with finite cutoff r_c,
  srank ≤ 3 × |atoms within r_c| + 3 × |ligand atoms|

Small binding pocket → low srank → TRACTABLE.
"""

@dataclass(frozen=True)
class BindingSite:
    """From MolecularSrank.lean::BindingSite."""
    center: jnp.ndarray  # (3,)
    radius: float

def atoms_within_cutoff(
    positions: jnp.ndarray,
    binding_site: BindingSite,
    cutoff: float
) -> jnp.ndarray:
    """
    From MolecularSrank.lean::atomWithinCutoff.
    Returns boolean mask of atoms within cutoff of binding site.
    """
    distances = jnp.linalg.norm(positions - binding_site.center, axis=1)
    return distances < cutoff

def num_relevant_atoms(
    positions: jnp.ndarray,
    binding_site: BindingSite,
    cutoff: float
) -> int:
    """
    From MolecularSrank.lean::numRelevantAtoms.
    Count of atoms within cutoff.
    """
    mask = atoms_within_cutoff(positions, binding_site, cutoff)
    return int(jnp.sum(mask))

def molecular_srank_bound(
    protein_positions: jnp.ndarray,
    ligand_positions: jnp.ndarray,
    binding_site: BindingSite,
    cutoff: float
) -> int:
    """
    MAIN THEOREM (md_srank_bound):
      srank ≤ 3 × numRelevantAtoms + 3 × numLigandAtoms
    
    From MolecularSrank.lean::md_srank_bound.
    """
    K = num_relevant_atoms(protein_positions, binding_site, cutoff)
    L = ligand_positions.shape[0]
    return 3 * K + 3 * L

def compute_molecular_srank(
    potential_fn: Callable,
    protein_positions: jnp.ndarray,
    ligand_positions: jnp.ndarray,
    binding_site: BindingSite,
    cutoff: float
) -> int:
    """
    Compute actual srank via gradient-based relevance detection,
    bounded by the theoretical cutoff bound.
    
    Returns min(gradient_srank, theoretical_bound).
    """
    from dq_dock_engine.physics.srank import compute_srank
    
    all_positions = jnp.concatenate([protein_positions, ligand_positions], axis=0)
    gradient_srank = compute_srank(all_positions, potential_fn)
    theoretical_bound = molecular_srank_bound(
        protein_positions, ligand_positions, binding_site, cutoff
    )
    return min(gradient_srank, theoretical_bound)

@dataclass(frozen=True)
class MolecularSrankResult:
    """Result of molecular srank analysis."""
    srank: int
    theoretical_bound: int   # 3K + 3L
    n_relevant_atoms: int    # K
    n_ligand_atoms: int      # L
    n_total_atoms: int
    cutoff: float
    
    @property
    def speedup_estimate(self) -> float:
        """Speedup from srank reduction vs full dimensionality."""
        full_dim = 3 * self.n_total_atoms
        return (full_dim / max(self.srank, 1)) ** 2

def thermodynamic_lower_bound(
    srank: int,
    kB: float = 1.380649e-23,
    T: float = 300.0
) -> float:
    """
    From MolecularSrank.lean::md_thermodynamic_lower_bound:
      E ≥ srank × kB × T × ln(2)
    """
    return srank * kB * T * jnp.log(2.0)
