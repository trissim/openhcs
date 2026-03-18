import jax.numpy as jnp
import MDAnalysis as mda
from typing import Tuple

"""
PDB parsing utilities using MDAnalysis to extract JAX-compatible arrays.
"""

def parse_pdb(pdb_path: str) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Parse a PDB file and return positions, masses, and atomic elements.
    """
    u = mda.Universe(pdb_path)
    positions = jnp.array(u.atoms.positions)
    masses = jnp.array(u.atoms.masses)
    # Map elements to integers for JAX compatibility
    elements = u.atoms.elements
    unique_elements = sorted(list(set(elements)))
    element_map = {el: i for i, el in enumerate(unique_elements)}
    element_indices = jnp.array([element_map[el] for el in elements])
    
    return positions, masses, element_indices

def get_charges_from_pdb(pdb_path: str) -> jnp.ndarray:
    """
    Extract partial charges. 
    Respecting Architecture: Fail-loud if charges are missing rather than silent fallback.
    """
    u = mda.Universe(pdb_path)
    return jnp.array(u.atoms.charges) # Fails with AttributeError/NoDataError if missing
