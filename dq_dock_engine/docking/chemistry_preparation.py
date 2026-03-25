from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from dq_dock_engine.docking.chemistry_annotations import (
    LigandStructureModel,
    ReceptorStructureModel,
    _infer_bond_adjacency,
    _normalize_element,
    _normalized,
)
from dq_dock_engine.docking.core import LigandContext
from dq_dock_engine.docking.pdb_io import parse_structure
from dq_dock_engine.docking.rdkit_io import load_rdkit_molecule
from dq_dock_engine.docking.receptor_preparation import (
    group_atom_records_by_residue,
    iter_atom_records,
)


@dataclass(frozen=True)
class ProtonatedSourceStructure:
    coords: np.ndarray
    elements: tuple[str, ...]
    adjacency: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MatchedRuntimeSource:
    runtime_coords: np.ndarray
    runtime_elements: tuple[str, ...]
    source: ProtonatedSourceStructure
    source_index_for_runtime: tuple[int, ...]

    def runtime_adjacency(self) -> tuple[tuple[int, ...], ...]:
        runtime_index_by_source = {
            source_index: runtime_index
            for runtime_index, source_index in enumerate(self.source_index_for_runtime)
        }
        rows: list[tuple[int, ...]] = []
        for source_index in self.source_index_for_runtime:
            rows.append(
                tuple(
                    sorted(
                        runtime_index_by_source[neighbor]
                        for neighbor in self.source.adjacency[source_index]
                        if _normalize_element(self.source.elements[neighbor]) != "H"
                        and neighbor in runtime_index_by_source
                    )
                )
            )
        return tuple(rows)

    def runtime_hydrogen_directions(self) -> tuple[tuple[np.ndarray, ...], ...]:
        directions: list[tuple[np.ndarray, ...]] = []
        for source_index in self.source_index_for_runtime:
            heavy_coord = self.source.coords[source_index]
            directions.append(
                tuple(
                    _normalized(self.source.coords[neighbor] - heavy_coord)
                    for neighbor in self.source.adjacency[source_index]
                    if _normalize_element(self.source.elements[neighbor]) == "H"
                )
            )
        return tuple(directions)


def _match_runtime_atoms_to_source(
    *,
    runtime_coords: np.ndarray,
    runtime_elements: tuple[str, ...],
    source_coords: np.ndarray,
    source_elements: tuple[str, ...],
    structure_label: str,
    max_distance: float = 0.35,
) -> tuple[int, ...]:
    if len(runtime_elements) != len(source_elements):
        raise ValueError(
            f"{structure_label} protonation source does not match runtime atom count"
        )
    remaining = set(range(len(source_elements)))
    mapping: list[int] = []
    for runtime_index, (coord, element) in enumerate(
        zip(runtime_coords, runtime_elements, strict=True)
    ):
        normalized = _normalize_element(element)
        candidates = [
            source_index
            for source_index in remaining
            if _normalize_element(source_elements[source_index]) == normalized
        ]
        if not candidates:
            raise ValueError(
                f"{structure_label} protonation source is missing atom {runtime_index} ({normalized})"
            )
        distances = np.asarray(
            [
                np.linalg.norm(coord - source_coords[source_index])
                for source_index in candidates
            ],
            dtype=np.float32,
        )
        best_local = int(np.argmin(distances))
        best_source_index = candidates[best_local]
        if float(distances[best_local]) > max_distance:
            raise ValueError(
                f"{structure_label} protonation source does not align with runtime atom {runtime_index} ({normalized})"
            )
        mapping.append(best_source_index)
        remaining.remove(best_source_index)
    return tuple(mapping)


def _load_ligand_source_with_protonation(
    ligand_source_path: str | Path | None,
) -> ProtonatedSourceStructure:
    if ligand_source_path is None:
        raise ValueError(
            "Extended-rich chemistry requires ligand_source_path for protonation-backed chemistry"
        )
    source_path = Path(ligand_source_path)
    if source_path.suffix.lower() in {".pdb", ".ent"}:
        coords, _, elements = cast(
            tuple[np.ndarray, np.ndarray, list[str]],
            parse_structure(source_path, strip_hydrogens=False, return_elements=True),
        )
        normalized_elements = tuple(_normalize_element(element) for element in elements)
        if not any(element == "H" for element in normalized_elements):
            try:
                from rdkit import Chem
            except ImportError as exc:
                raise ValueError(
                    "Ligand protonation requires RDKit when the ligand source lacks explicit hydrogens"
                ) from exc
            mol = load_rdkit_molecule(source_path, remove_hs=False, sanitize=False)
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                Chem.SanitizeMol(
                    mol,
                    Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                    | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                    | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                    | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                )
            mol = Chem.AddHs(mol, addCoords=True)
            conformer = mol.GetConformer()
            coords = np.asarray(
                [
                    list(conformer.GetAtomPosition(atom_index))
                    for atom_index in range(mol.GetNumAtoms())
                ],
                dtype=np.float32,
            )
            normalized_elements = tuple(
                _normalize_element(atom.GetSymbol()) for atom in mol.GetAtoms()
            )
            if not any(element == "H" for element in normalized_elements):
                raise ValueError("RDKit protonation did not produce explicit hydrogens")
        else:
            coords = np.asarray(coords, dtype=np.float32)
    else:
        mol = load_rdkit_molecule(source_path, remove_hs=False, sanitize=True)
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise ValueError("Ligand chemistry loading requires RDKit") from exc
        if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
            mol = Chem.AddHs(mol, addCoords=True)
        if mol.GetNumConformers() == 0:
            raise ValueError(
                f"Ligand chemistry source does not contain coordinates: {source_path}"
            )
        conformer = mol.GetConformer()
        coords = np.asarray(
            [
                list(conformer.GetAtomPosition(atom_index))
                for atom_index in range(mol.GetNumAtoms())
            ],
            dtype=np.float32,
        )
        normalized_elements = tuple(
            _normalize_element(atom.GetSymbol()) for atom in mol.GetAtoms()
        )
    return ProtonatedSourceStructure(
        coords=coords,
        elements=normalized_elements,
        adjacency=_infer_bond_adjacency(
            coords,
            normalized_elements,
            include_hydrogens=True,
        ),
    )


def prepare_ligand_chemistry_structure(
    ligand_ctx: LigandContext,
    *,
    ligand_source_path: str | Path | None,
) -> LigandStructureModel:
    coords = np.asarray(ligand_ctx.base_coords, dtype=np.float32)
    elements = tuple(ligand_ctx.elements)
    if len(elements) != coords.shape[0] or len(elements) == 0:
        raise ValueError(
            "Extended-rich chemistry requires ligand element annotations for every runtime atom"
        )
    charges = (
        None
        if ligand_ctx.charges is None
        else np.asarray(ligand_ctx.charges, dtype=np.float32)
    )
    source = _load_ligand_source_with_protonation(ligand_source_path)
    source_heavy_indices = tuple(
        index
        for index, element in enumerate(source.elements)
        if _normalize_element(element) != "H"
    )
    if not source_heavy_indices:
        raise ValueError("Ligand protonation source does not contain heavy atoms")
    heavy_center = np.mean(
        source.coords[list(source_heavy_indices)],
        axis=0,
        keepdims=True,
    )
    centered_source = ProtonatedSourceStructure(
        coords=source.coords - heavy_center,
        elements=source.elements,
        adjacency=source.adjacency,
    )
    source_heavy_elements = tuple(
        centered_source.elements[index] for index in source_heavy_indices
    )
    if tuple(_normalize_element(element) for element in elements) == tuple(
        _normalize_element(element) for element in source_heavy_elements
    ):
        matched_heavy_positions = tuple(range(len(source_heavy_indices)))
    else:
        centered_heavy_coords = centered_source.coords[list(source_heavy_indices)]
        matched_heavy_positions = _match_runtime_atoms_to_source(
            runtime_coords=coords,
            runtime_elements=elements,
            source_coords=centered_heavy_coords,
            source_elements=source_heavy_elements,
            structure_label="Ligand",
        )
    matched_source = MatchedRuntimeSource(
        runtime_coords=coords,
        runtime_elements=elements,
        source=centered_source,
        source_index_for_runtime=tuple(
            source_heavy_indices[position] for position in matched_heavy_positions
        ),
    )
    return LigandStructureModel(
        coords=coords,
        elements=elements,
        adjacency=matched_source.runtime_adjacency(),
        hydrogen_directions=matched_source.runtime_hydrogen_directions(),
        charges=charges,
    )


def _match_protonated_source_to_records(
    matched_source: MatchedRuntimeSource,
    records: tuple,
) -> tuple[tuple[int, object], ...]:
    """Map runtime atoms back to original PDB records when the protonated
    source came from RDKit (different atom ordering than PDB records).

    Matches each runtime heavy atom to its nearest PDB record by coordinate
    distance + element agreement. Fails loud on mismatch.
    """
    from dq_dock_engine.docking.receptor_preparation import PDBAtomRecord

    record_coords = np.asarray([record.coord for record in records], dtype=np.float32)
    record_elements = tuple(_normalize_element(record.element) for record in records)
    remaining = set(range(len(records)))
    result: list[tuple[int, PDBAtomRecord]] = []
    for runtime_index, source_index in enumerate(
        matched_source.source_index_for_runtime
    ):
        source_coord = matched_source.source.coords[source_index]
        source_element = _normalize_element(
            matched_source.source.elements[source_index]
        )
        candidates = [
            record_index
            for record_index in remaining
            if record_elements[record_index] == source_element
        ]
        if not candidates:
            raise ValueError(
                f"Receptor protonation: no PDB record matches runtime atom "
                f"{runtime_index} ({source_element})"
            )
        distances = np.asarray(
            [
                np.linalg.norm(source_coord - record_coords[record_index])
                for record_index in candidates
            ],
            dtype=np.float32,
        )
        best_local = int(np.argmin(distances))
        best_record_index = candidates[best_local]
        if float(distances[best_local]) > 0.5:
            raise ValueError(
                f"Receptor protonation: PDB record {best_record_index} is "
                f"{distances[best_local]:.2f} A from RDKit heavy atom "
                f"{source_index} ({source_element}) — alignment failure"
            )
        result.append((runtime_index, records[best_record_index]))
        remaining.remove(best_record_index)
    return tuple(result)


def _protonate_receptor_with_rdkit(
    receptor_path: Path,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Add explicit hydrogens to a receptor PDB that lacks them.

    Returns (coords, elements) for the full protonated structure.
    Fails loud if RDKit cannot parse or protonate the structure.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ValueError(
            "Receptor protonation requires RDKit when the receptor file lacks explicit hydrogens"
        ) from exc
    mol = Chem.MolFromPDBFile(str(receptor_path), removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(
            f"RDKit failed to parse receptor protonation source: {receptor_path}"
        )
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # Protein PDBs may have partial valence issues; sanitize what we can
        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        )
    mol = Chem.AddHs(mol, addCoords=True)
    conformer = mol.GetConformer()
    coords = np.asarray(
        [
            list(conformer.GetAtomPosition(atom_index))
            for atom_index in range(mol.GetNumAtoms())
        ],
        dtype=np.float32,
    )
    elements = tuple(_normalize_element(atom.GetSymbol()) for atom in mol.GetAtoms())
    if not any(element == "H" for element in elements):
        raise ValueError(
            "RDKit protonation of receptor did not produce explicit hydrogens"
        )
    return coords, elements


def prepare_receptor_chemistry_structure(
    receptor_coords: np.ndarray,
    receptor_elements: tuple[str, ...],
    *,
    receptor_file: str | Path | None,
) -> ReceptorStructureModel:
    if receptor_file is None:
        raise ValueError(
            "Extended-rich chemistry requires receptor_file for protonation-backed chemistry"
        )
    normalized_receptor_elements = tuple(
        _normalize_element(element) for element in receptor_elements
    )
    if len(normalized_receptor_elements) != receptor_coords.shape[0]:
        raise ValueError(
            "Extended-rich chemistry requires receptor element annotations for every runtime atom"
        )
    records = iter_atom_records(Path(receptor_file))
    has_hydrogens = any(_normalize_element(record.element) == "H" for record in records)
    if has_hydrogens:
        full_coords = np.asarray([record.coord for record in records], dtype=np.float32)
        full_elements = tuple(_normalize_element(record.element) for record in records)
    else:
        full_coords, full_elements = _protonate_receptor_with_rdkit(Path(receptor_file))
    source = ProtonatedSourceStructure(
        coords=full_coords,
        elements=full_elements,
        adjacency=_infer_bond_adjacency(
            full_coords,
            full_elements,
            include_hydrogens=True,
        ),
    )
    heavy_record_indices = tuple(
        index for index, element in enumerate(source.elements) if element != "H"
    )
    matched_heavy_positions = _match_runtime_atoms_to_source(
        runtime_coords=np.asarray(receptor_coords, dtype=np.float32),
        runtime_elements=normalized_receptor_elements,
        source_coords=source.coords[list(heavy_record_indices)],
        source_elements=tuple(source.elements[index] for index in heavy_record_indices),
        structure_label="Receptor",
    )
    matched_source = MatchedRuntimeSource(
        runtime_coords=np.asarray(receptor_coords, dtype=np.float32),
        runtime_elements=normalized_receptor_elements,
        source=source,
        source_index_for_runtime=tuple(
            heavy_record_indices[position] for position in matched_heavy_positions
        ),
    )
    if has_hydrogens:
        # Source came from records — direct index lookup
        indexed_records = tuple(
            (runtime_index, records[source_index])
            for runtime_index, source_index in enumerate(
                matched_source.source_index_for_runtime
            )
        )
    else:
        # Source came from RDKit protonation — match RDKit heavy atoms
        # back to original PDB records by coordinate proximity
        indexed_records = _match_protonated_source_to_records(matched_source, records)
    grouped_records = group_atom_records_by_residue(
        tuple(record for _, record in indexed_records)
    )
    record_index_by_identity = {id(record): index for index, record in indexed_records}
    return ReceptorStructureModel(
        coords=np.asarray(receptor_coords, dtype=np.float32),
        elements=normalized_receptor_elements,
        adjacency=matched_source.runtime_adjacency(),
        hydrogen_directions=matched_source.runtime_hydrogen_directions(),
        indexed_records=indexed_records,
        residue_records={
            residue: tuple(
                (record_index_by_identity[id(record)], record)
                for record in residue_records
            )
            for residue, residue_records in grouped_records.items()
        },
    )
