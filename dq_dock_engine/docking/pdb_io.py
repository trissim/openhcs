"""
Core PDB I/O and atom-typing infrastructure.

Responsible for:
  1. Parsing PDB files into coordinate arrays + per-atom VdW radii.
  2. Constructing LigandContext and receptor arrays from raw PDB paths.

This is core infrastructure — the benchmark should never do physics-level parsing.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import jax.numpy as jnp

from dq_dock_engine.docking.core import LigandContext, DockingBox
from dq_dock_engine.docking.physics_params import get_vdw_radius


def parse_structure(
    pdb_path: Path, *, strip_hydrogens: bool = True, return_elements: bool = False
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Parse a PDB file into coordinate and VdW radius arrays.

    Atom element is determined from PDB columns 76-78 (standard element field),
    falling back to the first character of the atom name (columns 12-16).

    Args:
        pdb_path: Path to PDB file.
        strip_hydrogens: If True, skip hydrogen atoms entirely.

    Returns:
        (coords, radii) where:
            coords: (N, 3) float64 array of Cartesian coordinates
            radii:  (N,) float64 array of VdW radii in Angstroms
    """
    coords: list[list[float]] = []
    radii: list[float] = []
    elements: list[str] = []

    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            # --- Determine element ---
            element = line[76:78].strip() if len(line) > 77 else ""
            if not element:
                # Fallback: first non-digit character of atom name (cols 12-16)
                atom_name = line[12:16].strip()
                element = atom_name.lstrip("0123456789")[:1]

            if strip_hydrogens and element.upper() == "H":
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue

            coords.append([x, y, z])
            radii.append(get_vdw_radius(element))
            elements.append(element)

    if not coords:
        raise ValueError(f"No atoms parsed from {pdb_path}")

    if return_elements:
        return (
            np.array(coords, dtype=np.float64),
            np.array(radii, dtype=np.float64),
            elements,
        )

    return np.array(coords, dtype=np.float64), np.array(radii, dtype=np.float64)


def build_ligand_context(
    ligand_coords: np.ndarray,
    ligand_radii: np.ndarray,
    elements: list[str] | None = None,
    charges: np.ndarray | None = None,
) -> LigandContext:
    """
    Construct an immutable LigandContext from parsed arrays.

    Centers the ligand at origin for proper rotation sampling.
    """
    coords_jnp = jnp.array(ligand_coords)
    com = jnp.mean(coords_jnp, axis=0)
    base_coords = coords_jnp - com

    el_tuple = tuple(elements) if elements is not None else ()
    jnp_charges = jnp.array(charges) if charges is not None else None

    return LigandContext(
        base_coords=base_coords,
        base_radii=jnp.array(ligand_radii),
        center_of_mass=com,
        elements=el_tuple,
        charges=jnp_charges,
    )


def build_receptor_arrays(
    receptor_coords: np.ndarray,
    receptor_radii: np.ndarray,
    center: np.ndarray,
    pocket_radius: float = 12.0,
    receptor_elements: list[str] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, list[str]]:
    """
    Extract pocket atoms within `pocket_radius` of `center`.

    Returns:
        (pocket_coords, pocket_radii) as JAX arrays.
    """
    distances = np.linalg.norm(receptor_coords - center, axis=1)
    mask = distances < pocket_radius
    coords_out = jnp.array(receptor_coords[mask])
    radii_out = jnp.array(receptor_radii[mask])
    if receptor_elements is not None:
        elems_out = [e for i, e in enumerate(receptor_elements) if mask[i]]
        return coords_out, radii_out, elems_out
    return coords_out, radii_out
