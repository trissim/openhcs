from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import ClassVar, Iterable, TypeVar, cast

import numpy as np

from dq_dock_engine.docking.core import LigandContext
from dq_dock_engine.docking.receptor_preparation import (
    PDBAtomRecord,
    ResidueKey,
    WATER_RESNAMES,
)


HALOGEN_ELEMENTS = frozenset({"CL", "BR", "I"})
POLAR_ELEMENTS = frozenset({"N", "O", "S", "F"})
HBOND_DONOR_ELEMENTS = frozenset({"N", "O", "S"})
HBOND_ACCEPTOR_ELEMENTS = frozenset({"N", "O", "S"})
AROMATIC_RING_ELEMENTS = frozenset({"C", "N"})
POSITIVE_ELEMENTS = frozenset(
    {"NA", "K", "CA", "MG", "MN", "FE", "CO", "NI", "CU", "ZN", "CD"}
)

MONOCATION_SITE_STRENGTH = 1.0

_COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
    "NA": 1.66,
    "K": 2.03,
    "CA": 1.76,
    "MG": 1.41,
    "MN": 1.39,
    "FE": 1.32,
    "CO": 1.26,
    "NI": 1.24,
    "CU": 1.32,
    "ZN": 1.22,
    "CD": 1.44,
}

_AROMATIC_RESIDUE_TEMPLATES: dict[str, tuple[tuple[str, ...], ...]] = {
    "PHE": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
    "TYR": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
    "HIS": (("CG", "ND1", "CE1", "NE2", "CD2"),),
    "HID": (("CG", "ND1", "CE1", "NE2", "CD2"),),
    "HIE": (("CG", "ND1", "CE1", "NE2", "CD2"),),
    "HIP": (("CG", "ND1", "CE1", "NE2", "CD2"),),
    "TRP": (
        ("CG", "CD1", "NE1", "CE2", "CD2"),
        ("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"),
    ),
}

_DONOR_ONLY_RECEPTOR_NITROGENS = frozenset(
    {
        ("ARG", "NE"),
        ("ARG", "NH1"),
        ("ARG", "NH2"),
        ("ASN", "ND2"),
        ("GLN", "NE2"),
        ("LYS", "NZ"),
        ("TRP", "NE1"),
    }
)

_RECEPTOR_O_DONOR_ATOMS = frozenset(
    {
        ("SER", "OG"),
        ("THR", "OG1"),
        ("TYR", "OH"),
    }
)

_RECEPTOR_S_DONOR_ATOMS = frozenset({("CYS", "SG")})


def _normalize_element(element: str) -> str:
    return element.strip().upper()


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return np.zeros((3,), dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _safe_centroid(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] == 0:
        return np.zeros((3,), dtype=np.float32)
    return np.mean(coords, axis=0).astype(np.float32)


def _plane_normal(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] < 3:
        return np.zeros((3,), dtype=np.float32)
    origin = coords[0]
    for idx_b in range(1, coords.shape[0] - 1):
        for idx_c in range(idx_b + 1, coords.shape[0]):
            normal = np.cross(coords[idx_b] - origin, coords[idx_c] - origin)
            if np.linalg.norm(normal) > 1e-6:
                return _normalized(normal)
    return np.zeros((3,), dtype=np.float32)


def _infer_bond_cutoff(element_a: str, element_b: str) -> float:
    radius_a = _COVALENT_RADII.get(_normalize_element(element_a), 0.77)
    radius_b = _COVALENT_RADII.get(_normalize_element(element_b), 0.77)
    return radius_a + radius_b + 0.45


def _infer_bond_adjacency(
    coords: np.ndarray,
    elements: tuple[str, ...],
    *,
    include_hydrogens: bool = False,
) -> tuple[tuple[int, ...], ...]:
    adjacency = [set() for _ in range(len(elements))]
    for idx_a in range(len(elements)):
        element_a = _normalize_element(elements[idx_a])
        if not include_hydrogens and element_a == "H":
            continue
        for idx_b in range(idx_a + 1, len(elements)):
            element_b = _normalize_element(elements[idx_b])
            if not include_hydrogens and element_b == "H":
                continue
            distance = float(np.linalg.norm(coords[idx_a] - coords[idx_b]))
            if 0.4 <= distance <= _infer_bond_cutoff(element_a, element_b):
                adjacency[idx_a].add(idx_b)
                adjacency[idx_b].add(idx_a)
    return tuple(tuple(sorted(neighbors)) for neighbors in adjacency)


def _direction_from_neighbors(
    coords: np.ndarray,
    atom_index: int,
    neighbor_indices: Iterable[int],
) -> np.ndarray:
    neighbors = tuple(int(index) for index in neighbor_indices)
    if not neighbors:
        return np.zeros((3,), dtype=np.float32)
    return _normalized(coords[atom_index] - np.mean(coords[list(neighbors)], axis=0))


def _unique_cycles_of_size(
    adjacency: tuple[tuple[int, ...], ...],
    size: int,
) -> tuple[tuple[int, ...], ...]:
    cycles: set[tuple[int, ...]] = set()

    def canonicalize(cycle: tuple[int, ...]) -> tuple[int, ...]:
        anchor = min(cycle)
        candidates = []
        for series in (list(cycle), list(reversed(cycle))):
            start = series.index(anchor)
            candidates.append(tuple(series[start:] + series[:start]))
        return min(candidates)

    def dfs(start: int, current: int, path: list[int]) -> None:
        if len(path) == size:
            if start in adjacency[current]:
                cycles.add(canonicalize(tuple(path)))
            return
        for neighbor in adjacency[current]:
            if neighbor < start or neighbor in path:
                continue
            dfs(start, neighbor, [*path, neighbor])

    for start in range(len(adjacency)):
        dfs(start, start, [start])
    return tuple(sorted(cycles))


def _is_planar_ring(coords: np.ndarray, atom_indices: tuple[int, ...]) -> bool:
    ring_coords = coords[list(atom_indices)]
    center = np.mean(ring_coords, axis=0)
    normal = _plane_normal(ring_coords)
    return bool(
        np.linalg.norm(normal) > 1e-6
        and np.max(np.abs((ring_coords - center) @ normal)) <= 0.25
    )


class ChemistrySiteRole(Enum):
    AROMATIC_RING = auto()
    CATION = auto()
    HBOND_DONOR = auto()
    HBOND_ACCEPTOR = auto()
    HALOGEN_DONOR = auto()
    HALOGEN_ACCEPTOR = auto()
    POLAR = auto()
    BRIDGE_WATER = auto()


@dataclass(frozen=True)
class ChemistrySiteAnnotation:
    role: ChemistrySiteRole
    anchor_index: int
    position: np.ndarray
    strength: float = 1.0


@dataclass(frozen=True)
class IndexedSiteAnnotation(ChemistrySiteAnnotation):
    atom_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class RingSiteAnnotation(IndexedSiteAnnotation):
    normal: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))


@dataclass(frozen=True)
class DirectionalSiteAnnotation(IndexedSiteAnnotation):
    neighbor_indices: tuple[int, ...] = ()
    direction: np.ndarray = field(
        default_factory=lambda: np.zeros((3,), dtype=np.float32)
    )


SiteT = TypeVar("SiteT", bound=ChemistrySiteAnnotation)
StructureT = TypeVar("StructureT")


@dataclass(frozen=True)
class ChemistryAnnotationCatalog:
    sites: tuple[ChemistrySiteAnnotation, ...] = ()

    def typed_sites(
        self,
        site_type: type[SiteT],
        *,
        role: ChemistrySiteRole | None = None,
    ) -> tuple[SiteT, ...]:
        return tuple(
            cast(SiteT, site)
            for site in self.sites
            if isinstance(site, site_type) and (role is None or site.role == role)
        )


class LigandChemistryCatalog(ChemistryAnnotationCatalog):
    pass


class ReceptorChemistryCatalog(ChemistryAnnotationCatalog):
    pass


@dataclass(frozen=True)
class BondedStructureModel:
    coords: np.ndarray
    elements: tuple[str, ...]
    adjacency: tuple[tuple[int, ...], ...]
    hydrogen_directions: tuple[tuple[np.ndarray, ...], ...]


@dataclass(frozen=True)
class LigandStructureModel(BondedStructureModel):
    charges: np.ndarray | None = None


@dataclass(frozen=True)
class ReceptorStructureModel(BondedStructureModel):
    indexed_records: tuple[tuple[int, PDBAtomRecord], ...]
    residue_records: dict[ResidueKey, tuple[tuple[int, PDBAtomRecord], ...]]


def _ring_site(
    *,
    role: ChemistrySiteRole,
    coords: np.ndarray,
    atom_indices: tuple[int, ...],
    strength: float = 1.0,
) -> RingSiteAnnotation:
    ring_coords = coords[list(atom_indices)]
    return RingSiteAnnotation(
        role=role,
        anchor_index=int(atom_indices[0]),
        atom_indices=atom_indices,
        position=_safe_centroid(ring_coords),
        strength=float(strength),
        normal=_plane_normal(ring_coords),
    )


def _merge_overlapping_ring_sites(
    sites: tuple[RingSiteAnnotation, ...],
    *,
    coords: np.ndarray,
    min_shared_atoms: int,
) -> tuple[RingSiteAnnotation, ...]:
    if len(sites) <= 1:
        return sites
    remaining = list(range(len(sites)))
    merged: list[RingSiteAnnotation] = []
    while remaining:
        seed = remaining.pop(0)
        component = {seed}
        changed = True
        while changed:
            changed = False
            for idx in list(remaining):
                site_atoms = set(sites[idx].atom_indices)
                if any(
                    len(site_atoms & set(sites[other].atom_indices)) >= min_shared_atoms
                    for other in component
                ):
                    component.add(idx)
                    remaining.remove(idx)
                    changed = True
        atom_indices = tuple(
            sorted({atom for idx in component for atom in sites[idx].atom_indices})
        )
        merged.append(
            _ring_site(
                role=sites[seed].role,
                coords=coords,
                atom_indices=atom_indices,
                strength=max(sites[idx].strength for idx in component),
            )
        )
    return tuple(merged)


def _indexed_site(
    *,
    role: ChemistrySiteRole,
    atom_indices: tuple[int, ...],
    coords: np.ndarray,
    strength: float,
) -> IndexedSiteAnnotation:
    atom_index = int(atom_indices[0])
    return IndexedSiteAnnotation(
        role=role,
        anchor_index=atom_index,
        atom_indices=atom_indices,
        position=coords[atom_index].astype(np.float32),
        strength=float(strength),
    )


def _directional_site(
    *,
    role: ChemistrySiteRole,
    atom_indices: tuple[int, ...],
    neighbor_indices: tuple[int, ...],
    coords: np.ndarray,
    strength: float,
    direction: np.ndarray | None = None,
) -> DirectionalSiteAnnotation:
    atom_index = int(atom_indices[0])
    return DirectionalSiteAnnotation(
        role=role,
        anchor_index=atom_index,
        atom_indices=atom_indices,
        position=coords[atom_index].astype(np.float32),
        strength=float(strength),
        neighbor_indices=neighbor_indices,
        direction=(
            _direction_from_neighbors(coords, atom_index, neighbor_indices)
            if direction is None
            else direction.astype(np.float32)
        ),
    )


def _receptor_acceptor_allowed(
    residue: ResidueKey,
    atom_name: str,
    *,
    element: str,
    hydrogen_count: int,
    heavy_degree: int,
) -> bool:
    normalized_element = _normalize_element(element)
    normalized_atom_name = atom_name.upper()
    normalized_resname = residue.resname.upper()
    if normalized_element == "O":
        return True
    if normalized_element == "S":
        return True
    if normalized_element != "N":
        return False
    if normalized_atom_name == "N":
        return False
    if (normalized_resname, normalized_atom_name) in _DONOR_ONLY_RECEPTOR_NITROGENS:
        return False
    if normalized_resname in {"HIS", "HID", "HIE", "HIP"} and normalized_atom_name in {
        "ND1",
        "NE2",
    }:
        return hydrogen_count == 0
    return hydrogen_count == 0 and heavy_degree <= 2


def _ligand_acceptor_allowed(
    structure: LigandStructureModel,
    atom_index: int,
) -> bool:
    element = _normalize_element(structure.elements[atom_index])
    charge = 0.0 if structure.charges is None else float(structure.charges[atom_index])
    hydrogen_count = len(structure.hydrogen_directions[atom_index])
    heavy_degree = len(structure.adjacency[atom_index])
    if element == "O":
        return charge <= 0.4
    if element == "S":
        return charge <= 0.2 and heavy_degree <= 2
    if element != "N":
        return False
    return charge <= 0.2 and hydrogen_count == 0 and heavy_degree <= 2


def _receptor_donor_allowed(
    residue: ResidueKey,
    atom_name: str,
    *,
    element: str,
    hydrogen_count: int,
) -> bool:
    if hydrogen_count == 0:
        return False
    normalized_element = _normalize_element(element)
    normalized_atom_name = atom_name.upper()
    normalized_resname = residue.resname.upper()
    residue_atom = (normalized_resname, normalized_atom_name)
    if normalized_element == "N":
        if normalized_atom_name == "N":
            return normalized_resname != "PRO"
        if residue_atom in _DONOR_ONLY_RECEPTOR_NITROGENS:
            return True
        if normalized_resname in {
            "HIS",
            "HID",
            "HIE",
            "HIP",
        } and normalized_atom_name in {
            "ND1",
            "NE2",
        }:
            return True
        return False
    if normalized_element == "O":
        return residue_atom in _RECEPTOR_O_DONOR_ATOMS
    if normalized_element == "S":
        return residue_atom in _RECEPTOR_S_DONOR_ATOMS
    return False


def _split_donor_capacity(total_strength: float, hydrogen_count: int) -> float:
    if hydrogen_count <= 0:
        return 0.0
    return float(total_strength) / float(hydrogen_count)


class LigandSiteExtractor(ABC):
    _registered_types: ClassVar[list[type["LigandSiteExtractor"]]] = []

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            cls._registered_types.append(cls)

    @classmethod
    def registered_types(cls) -> tuple[type["LigandSiteExtractor"], ...]:
        return tuple(cls._registered_types)

    @abstractmethod
    def extract(
        self, structure: LigandStructureModel
    ) -> tuple[ChemistrySiteAnnotation, ...]:
        raise NotImplementedError


class ReceptorSiteExtractor(ABC):
    _registered_types: ClassVar[list[type["ReceptorSiteExtractor"]]] = []

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            cls._registered_types.append(cls)

    @classmethod
    def registered_types(cls) -> tuple[type["ReceptorSiteExtractor"], ...]:
        return tuple(cls._registered_types)

    @abstractmethod
    def extract(
        self, structure: ReceptorStructureModel
    ) -> tuple[ChemistrySiteAnnotation, ...]:
        raise NotImplementedError


class LigandAromaticRingExtractor(LigandSiteExtractor):
    def extract(
        self, structure: LigandStructureModel
    ) -> tuple[ChemistrySiteAnnotation, ...]:
        sites: list[ChemistrySiteAnnotation] = []
        for ring_size in (5, 6):
            for atom_indices in _unique_cycles_of_size(structure.adjacency, ring_size):
                ring_elements = {
                    _normalize_element(structure.elements[index])
                    for index in atom_indices
                }
                if ring_elements.issubset(AROMATIC_RING_ELEMENTS) and _is_planar_ring(
                    structure.coords, atom_indices
                ):
                    sites.append(
                        _ring_site(
                            role=ChemistrySiteRole.AROMATIC_RING,
                            coords=structure.coords,
                            atom_indices=atom_indices,
                        )
                    )
        return _merge_overlapping_ring_sites(
            tuple(cast(tuple[RingSiteAnnotation, ...], tuple(sites))),
            coords=structure.coords,
            min_shared_atoms=2,
        )


class LigandCationExtractor(LigandSiteExtractor):
    def extract(
        self, structure: LigandStructureModel
    ) -> tuple[ChemistrySiteAnnotation, ...]:
        sites: list[ChemistrySiteAnnotation] = []
        for index, element in enumerate(structure.elements):
            charge = (
                0.0 if structure.charges is None else float(structure.charges[index])
            )
            if _normalize_element(element) in POSITIVE_ELEMENTS or charge >= 0.35:
                sites.append(
                    _indexed_site(
                        role=ChemistrySiteRole.CATION,
                        atom_indices=(index,),
                        coords=structure.coords,
                        strength=max(1.0, charge),
                    )
                )
        return tuple(sites)


class LigandHalogenDonorExtractor(LigandSiteExtractor):
    def extract(
        self, structure: LigandStructureModel
    ) -> tuple[ChemistrySiteAnnotation, ...]:
        sites: list[ChemistrySiteAnnotation] = []
        for index, element in enumerate(structure.elements):
            normalized = _normalize_element(element)
            neighbors = structure.adjacency[index]
            if normalized in HALOGEN_ELEMENTS and neighbors:
                neighbor_index = int(neighbors[0])
                sites.append(
                    _directional_site(
                        role=ChemistrySiteRole.HALOGEN_DONOR,
                        atom_indices=(index,),
                        neighbor_indices=(neighbor_index,),
                        coords=structure.coords,
                        strength={"CL": 0.7, "BR": 0.85, "I": 1.0}.get(normalized, 1.0),
                        direction=_normalized(
                            structure.coords[index] - structure.coords[neighbor_index]
                        ),
                    )
                )
        return tuple(sites)


class LigandHBondDonorExtractor(LigandSiteExtractor):
    def extract(
        self, structure: LigandStructureModel
    ) -> tuple[ChemistrySiteAnnotation, ...]:
        sites: list[ChemistrySiteAnnotation] = []
        for atom_index, element in enumerate(structure.elements):
            if _normalize_element(element) not in HBOND_DONOR_ELEMENTS:
                continue
            hydrogen_directions = structure.hydrogen_directions[atom_index]
            donor_strength = _split_donor_capacity(1.0, len(hydrogen_directions))
            for direction in hydrogen_directions:
                sites.append(
                    _directional_site(
                        role=ChemistrySiteRole.HBOND_DONOR,
                        atom_indices=(atom_index,),
                        neighbor_indices=structure.adjacency[atom_index],
                        coords=structure.coords,
                        strength=donor_strength,
                        direction=direction,
                    )
                )
        return tuple(sites)


class LigandHBondAcceptorExtractor(LigandSiteExtractor):
    def extract(
        self, structure: LigandStructureModel
    ) -> tuple[ChemistrySiteAnnotation, ...]:
        sites: list[ChemistrySiteAnnotation] = []
        for atom_index, element in enumerate(structure.elements):
            if _normalize_element(element) not in HBOND_ACCEPTOR_ELEMENTS:
                continue
            if not _ligand_acceptor_allowed(structure, atom_index):
                continue
            sites.append(
                _directional_site(
                    role=ChemistrySiteRole.HBOND_ACCEPTOR,
                    atom_indices=(atom_index,),
                    neighbor_indices=structure.adjacency[atom_index],
                    coords=structure.coords,
                    strength=1.0,
                )
            )
        return tuple(sites)


class _LigandDirectionalExtractor(LigandSiteExtractor):
    @property
    @abstractmethod
    def role(self) -> ChemistrySiteRole: ...

    def include_atom(self, charge: float) -> bool:
        return True

    def extract(
        self, structure: LigandStructureModel
    ) -> tuple[ChemistrySiteAnnotation, ...]:
        sites: list[ChemistrySiteAnnotation] = []
        for index, element in enumerate(structure.elements):
            if _normalize_element(element) not in POLAR_ELEMENTS:
                continue
            charge = (
                0.0 if structure.charges is None else float(structure.charges[index])
            )
            if not self.include_atom(charge):
                continue
            strength = (
                1.0
                if structure.charges is None
                else max(0.5, min(1.5, abs(charge) + 0.5))
            )
            sites.append(
                _directional_site(
                    role=self.role,
                    atom_indices=(index,),
                    neighbor_indices=structure.adjacency[index],
                    coords=structure.coords,
                    strength=strength,
                )
            )
        return tuple(sites)


class LigandPolarExtractor(_LigandDirectionalExtractor):
    @property
    def role(self) -> ChemistrySiteRole:
        return ChemistrySiteRole.POLAR


class LigandHalogenAcceptorExtractor(_LigandDirectionalExtractor):
    @property
    def role(self) -> ChemistrySiteRole:
        return ChemistrySiteRole.HALOGEN_ACCEPTOR

    def include_atom(self, charge: float) -> bool:
        return charge <= 0.2


class ReceptorAromaticRingExtractor(ReceptorSiteExtractor):
    def extract(
        self, structure: ReceptorStructureModel
    ) -> tuple[ChemistrySiteAnnotation, ...]:
        sites: list[ChemistrySiteAnnotation] = []
        for residue, indexed_records in structure.residue_records.items():
            atoms_by_name = {
                record.atom_name.upper(): index for index, record in indexed_records
            }
            for template in _AROMATIC_RESIDUE_TEMPLATES.get(
                residue.resname.upper(), ()
            ):
                if all(name in atoms_by_name for name in template):
                    sites.append(
                        _ring_site(
                            role=ChemistrySiteRole.AROMATIC_RING,
                            coords=structure.coords,
                            atom_indices=tuple(
                                atoms_by_name[name] for name in template
                            ),
                        )
                    )
        return tuple(sites)


class ReceptorCationExtractor(ReceptorSiteExtractor):
    def extract(
        self, structure: ReceptorStructureModel
    ) -> tuple[ChemistrySiteAnnotation, ...]:
        sites: list[ChemistrySiteAnnotation] = []
        for residue, indexed_records in structure.residue_records.items():
            atoms_by_name = {
                record.atom_name.upper(): index for index, record in indexed_records
            }
            if residue.resname.upper() == "LYS" and "NZ" in atoms_by_name:
                sites.append(
                    _indexed_site(
                        role=ChemistrySiteRole.CATION,
                        atom_indices=(atoms_by_name["NZ"],),
                        coords=structure.coords,
                        strength=MONOCATION_SITE_STRENGTH,
                    )
                )
            if residue.resname.upper() == "ARG" and {"NE", "CZ", "NH1", "NH2"}.issubset(
                atoms_by_name
            ):
                cation_atoms = tuple(
                    atoms_by_name[name] for name in ("NE", "CZ", "NH1", "NH2")
                )
                sites.append(
                    IndexedSiteAnnotation(
                        role=ChemistrySiteRole.CATION,
                        anchor_index=atoms_by_name["CZ"],
                        atom_indices=cation_atoms,
                        position=_safe_centroid(structure.coords[list(cation_atoms)]),
                        strength=MONOCATION_SITE_STRENGTH,
                    )
                )
        return tuple(sites)


class ReceptorHBondDonorExtractor(ReceptorSiteExtractor):
    def extract(
        self, structure: ReceptorStructureModel
    ) -> tuple[ChemistrySiteAnnotation, ...]:
        sites: list[ChemistrySiteAnnotation] = []
        for residue, indexed_records in structure.residue_records.items():
            for atom_index, record in indexed_records:
                element = _normalize_element(record.element)
                if element not in HBOND_DONOR_ELEMENTS:
                    continue
                hydrogen_directions = structure.hydrogen_directions[atom_index]
                if not _receptor_donor_allowed(
                    residue,
                    record.atom_name,
                    element=element,
                    hydrogen_count=len(hydrogen_directions),
                ):
                    continue
                donor_strength = _split_donor_capacity(1.0, len(hydrogen_directions))
                for direction in hydrogen_directions:
                    sites.append(
                        _directional_site(
                            role=ChemistrySiteRole.HBOND_DONOR,
                            atom_indices=(atom_index,),
                            neighbor_indices=structure.adjacency[atom_index],
                            coords=structure.coords,
                            strength=donor_strength,
                            direction=direction,
                        )
                    )
        return tuple(sites)


class ReceptorHBondAcceptorExtractor(ReceptorSiteExtractor):
    def extract(
        self, structure: ReceptorStructureModel
    ) -> tuple[ChemistrySiteAnnotation, ...]:
        sites: list[ChemistrySiteAnnotation] = []
        for residue, indexed_records in structure.residue_records.items():
            for atom_index, record in indexed_records:
                element = _normalize_element(record.element)
                if element not in HBOND_ACCEPTOR_ELEMENTS:
                    continue
                if not _receptor_acceptor_allowed(
                    residue,
                    record.atom_name,
                    element=element,
                    hydrogen_count=len(structure.hydrogen_directions[atom_index]),
                    heavy_degree=len(structure.adjacency[atom_index]),
                ):
                    continue
                sites.append(
                    _directional_site(
                        role=ChemistrySiteRole.HBOND_ACCEPTOR,
                        atom_indices=(atom_index,),
                        neighbor_indices=structure.adjacency[atom_index],
                        coords=structure.coords,
                        strength=1.0,
                    )
                )
        return tuple(sites)


class ReceptorHalogenDonorExtractor(ReceptorSiteExtractor):
    def extract(
        self, structure: ReceptorStructureModel
    ) -> tuple[ChemistrySiteAnnotation, ...]:
        sites: list[ChemistrySiteAnnotation] = []
        for indexed_records in structure.residue_records.values():
            for index, record in indexed_records:
                if record.element not in HALOGEN_ELEMENTS:
                    continue
                neighbor_indices = tuple(
                    other_index
                    for other_index, other_record in indexed_records
                    if other_index != index
                    and 0.4
                    <= float(np.linalg.norm(record.coord - other_record.coord))
                    <= _infer_bond_cutoff(record.element, other_record.element)
                )
                if not neighbor_indices:
                    continue
                neighbor_index = neighbor_indices[0]
                sites.append(
                    _directional_site(
                        role=ChemistrySiteRole.HALOGEN_DONOR,
                        atom_indices=(index,),
                        neighbor_indices=(neighbor_index,),
                        coords=structure.coords,
                        strength={"CL": 0.7, "BR": 0.85, "I": 1.0}.get(
                            record.element, 1.0
                        ),
                        direction=_normalized(
                            structure.coords[index] - structure.coords[neighbor_index]
                        ),
                    )
                )
        return tuple(sites)


class ReceptorHalogenAcceptorExtractor(ReceptorSiteExtractor):
    def extract(
        self, structure: ReceptorStructureModel
    ) -> tuple[ChemistrySiteAnnotation, ...]:
        excluded = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2")}
        sites: list[ChemistrySiteAnnotation] = []
        for residue, indexed_records in structure.residue_records.items():
            for index, record in indexed_records:
                if record.element not in POLAR_ELEMENTS:
                    continue
                if (residue.resname.upper(), record.atom_name.upper()) in excluded:
                    continue
                neighbor_indices = tuple(
                    other_index
                    for other_index, other_record in indexed_records
                    if other_index != index
                    and 0.4
                    <= float(np.linalg.norm(record.coord - other_record.coord))
                    <= _infer_bond_cutoff(record.element, other_record.element)
                )
                sites.append(
                    _directional_site(
                        role=ChemistrySiteRole.HALOGEN_ACCEPTOR,
                        atom_indices=(index,),
                        neighbor_indices=neighbor_indices,
                        coords=structure.coords,
                        strength=1.0,
                    )
                )
        return tuple(sites)


class ReceptorBridgeWaterExtractor(ReceptorSiteExtractor):
    def extract(
        self, structure: ReceptorStructureModel
    ) -> tuple[ChemistrySiteAnnotation, ...]:
        acceptor_sites = [
            cast(DirectionalSiteAnnotation, site)
            for site in ReceptorHalogenAcceptorExtractor().extract(structure)
        ]
        if not acceptor_sites:
            return ()
        acceptor_positions = np.stack(
            [site.position for site in acceptor_sites], axis=0
        )
        sites: list[ChemistrySiteAnnotation] = []
        for residue, indexed_records in structure.residue_records.items():
            if residue.resname.upper() not in WATER_RESNAMES:
                continue
            for index, record in indexed_records:
                if record.element != "O":
                    continue
                deltas = record.coord[None, :] - acceptor_positions
                distances = np.linalg.norm(deltas, axis=1)
                nearest_index = int(np.argmin(distances))
                nearest_distance = float(distances[nearest_index])
                if nearest_distance > 3.5:
                    continue
                sites.append(
                    _directional_site(
                        role=ChemistrySiteRole.BRIDGE_WATER,
                        atom_indices=(index,),
                        neighbor_indices=(),
                        coords=structure.coords,
                        strength=max(0.4, 1.0 - 0.15 * nearest_distance),
                        direction=_normalized(deltas[nearest_index]),
                    )
                )
        return tuple(sites)


def _collect_sites(
    structure: StructureT,
    extractor_types: tuple[type, ...],
) -> tuple[ChemistrySiteAnnotation, ...]:
    return tuple(
        site
        for extractor_type in extractor_types
        for site in extractor_type().extract(structure)
    )


def catalog_from_ligand_structure(
    structure: LigandStructureModel,
) -> LigandChemistryCatalog:
    return LigandChemistryCatalog(
        sites=_collect_sites(structure, LigandSiteExtractor.registered_types())
    )


def catalog_from_receptor_structure(
    structure: ReceptorStructureModel,
) -> ReceptorChemistryCatalog:
    return ReceptorChemistryCatalog(
        sites=_collect_sites(structure, ReceptorSiteExtractor.registered_types())
    )


def build_ligand_chemistry_catalog(
    ligand_ctx: LigandContext | LigandStructureModel,
    *,
    ligand_source_path: str | Path | None = None,
) -> LigandChemistryCatalog:
    if isinstance(ligand_ctx, LigandStructureModel):
        return catalog_from_ligand_structure(ligand_ctx)
    from dq_dock_engine.docking.chemistry_preparation import (
        prepare_ligand_chemistry_structure,
    )

    return catalog_from_ligand_structure(
        prepare_ligand_chemistry_structure(
            ligand_ctx,
            ligand_source_path=ligand_source_path,
        )
    )


def build_receptor_chemistry_catalog(
    receptor_coords: np.ndarray | ReceptorStructureModel,
    receptor_elements: tuple[str, ...] | None = None,
    *,
    receptor_file: str | Path | None = None,
) -> ReceptorChemistryCatalog:
    if isinstance(receptor_coords, ReceptorStructureModel):
        return catalog_from_receptor_structure(receptor_coords)
    if receptor_elements is None:
        raise ValueError(
            "Extended-rich chemistry requires receptor element annotations for every runtime atom"
        )
    from dq_dock_engine.docking.chemistry_preparation import (
        prepare_receptor_chemistry_structure,
    )

    return catalog_from_receptor_structure(
        prepare_receptor_chemistry_structure(
            receptor_coords,
            receptor_elements,
            receptor_file=receptor_file,
        )
    )
