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


def _normalize_element(element: str) -> str:
    return element.strip().upper()


def parse_structure(
    pdb_path: Path,
    *,
    strip_hydrogens: bool = True,
    return_elements: bool = False,
    first_model_only: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Parse a PDB file into coordinate and VdW radius arrays.

    Atom element is determined from PDB columns 76-78 (standard element field),
    falling back to the first character of the atom name (columns 12-16).

    Args:
        pdb_path: Path to PDB file.
        strip_hydrogens: If True, skip hydrogen atoms entirely.
        first_model_only: If True, stop after first MODEL/ENDMDL block.

    Returns:
        (coords, radii) where:
            coords: (N, 3) float64 array of Cartesian coordinates
            radii:  (N,) float64 array of VdW radii in Angstroms
    """
    coords: list[list[float]] = []
    radii: list[float] = []
    elements: list[str] = []
    in_first_model = True

    with open(pdb_path) as f:
        for line in f:
            if first_model_only and (
                "ENDMDL" in line or (in_first_model and line.strip() == "END")
            ):
                break

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
    adjacency: tuple[tuple[int, ...], ...] | None = None,
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
    inferred_adjacency = adjacency
    if inferred_adjacency is None and elements is not None:
        from dq_dock_engine.docking.chemistry_annotations import _infer_bond_adjacency

        inferred_adjacency = _infer_bond_adjacency(
            ligand_coords, tuple(el_tuple), include_hydrogens=False
        )

    return LigandContext(
        base_coords=base_coords,
        base_radii=jnp.array(ligand_radii),
        center_of_mass=com,
        elements=el_tuple,
        charges=jnp_charges,
        adjacency=inferred_adjacency,
    )


def generate_independent_ligand_geometry(
    ligand_source_path: str | Path,
    *,
    expected_elements: tuple[str, ...],
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Generate a non-crystal ligand conformer from chemistry alone.

    Uses RDKit distance geometry to build an independent 3D conformer while
    preserving atom ordering/connectivity for the heavy-atom runtime model.
    """

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ValueError("Blind ligand geometry generation requires RDKit") from exc

    source_path = Path(ligand_source_path)
    mol = Chem.MolFromPDBFile(str(source_path), removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit failed to parse ligand source: {source_path}")

    mol = Chem.AddHs(mol, addCoords=True)
    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xD0C6
    params.useRandomCoords = True
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        fallback = AllChem.ETKDGv2()
        fallback.randomSeed = 0xD0C6
        fallback.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, fallback)
    if status != 0:
        raise ValueError(
            f"RDKit failed to embed an independent conformer for {source_path}"
        )

    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol)
    else:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass

    conformer = mol.GetConformer()
    heavy_indices = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    heavy_elements = tuple(
        _normalize_element(mol.GetAtomWithIdx(atom_idx).GetSymbol())
        for atom_idx in heavy_indices
    )
    normalized_expected = tuple(
        _normalize_element(element) for element in expected_elements
    )
    if heavy_elements != normalized_expected:
        raise ValueError(
            "Independent ligand conformer generation changed heavy-atom ordering; "
            f"expected {normalized_expected} but got {heavy_elements}"
        )

    coords = np.asarray(
        [list(conformer.GetAtomPosition(atom_idx)) for atom_idx in heavy_indices],
        dtype=np.float32,
    )
    runtime_index = {atom_idx: idx for idx, atom_idx in enumerate(heavy_indices)}
    adjacency: list[tuple[int, ...]] = []
    for atom_idx in heavy_indices:
        neighbors = []
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in runtime_index:
                neighbors.append(runtime_index[neighbor_idx])
        adjacency.append(tuple(sorted(neighbors)))
    return coords, tuple(adjacency)


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
