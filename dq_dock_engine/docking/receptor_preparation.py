from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, ClassVar, Iterable, Sequence, cast

import numpy as np

from dq_dock_engine.docking.charges import ChargeMethod
from dq_dock_engine.docking.pdb_io import parse_structure
from dq_dock_engine.docking_config import CertifiedScoringFamily


ALLOWED_ALTLOCS = {" ", "A"}
WATER_RESNAMES = frozenset({"HOH", "DOD", "WAT"})
METAL_ELEMENTS = frozenset(
    {"ZN", "MG", "MN", "FE", "CU", "CO", "NI", "CA", "NA", "K", "CD"}
)


@dataclass(frozen=True)
class ResidueKey:
    resname: str
    chain_id: str
    residue_id: str


@dataclass(frozen=True)
class PDBAtomRecord:
    line: str
    record_name: str
    atom_name: str
    residue: ResidueKey
    element: str
    coord: np.ndarray


class ReceptorComponentRole(Enum):
    POLYMER = auto()
    PRIMARY_LIGAND = auto()
    STRUCTURAL_METAL = auto()
    STRUCTURAL_COFACTOR = auto()
    STRUCTURAL_WATER = auto()
    OTHER_HETATM = auto()


@dataclass(frozen=True)
class ReceptorComponent:
    residue: ResidueKey
    role: ReceptorComponentRole
    atom_count: int
    elements: tuple[str, ...]


@dataclass(frozen=True)
class PreparedReceptorSystem:
    source_pdb: Path
    receptor_pdb: Path
    pocket_receptor_pdb: Path
    retained_components: tuple[ReceptorComponent, ...]
    pocket_components: tuple[ReceptorComponent, ...]
    receptor_coords: np.ndarray
    receptor_radii: np.ndarray
    receptor_elements: tuple[str, ...]
    pocket_coords: np.ndarray
    pocket_radii: np.ndarray
    pocket_elements: tuple[str, ...]
    policy_name: str

    @property
    def has_structural_metals(self) -> bool:
        return any(
            component.role == ReceptorComponentRole.STRUCTURAL_METAL
            for component in self.pocket_components
        )

    @property
    def has_structural_cofactors(self) -> bool:
        return any(
            component.role == ReceptorComponentRole.STRUCTURAL_COFACTOR
            for component in self.pocket_components
        )

    @property
    def has_structural_waters(self) -> bool:
        return any(
            component.role == ReceptorComponentRole.STRUCTURAL_WATER
            for component in self.pocket_components
        )


@dataclass(frozen=True)
class ReceptorCompatibilityReport:
    supported: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ReceptorChemistryPlan:
    protocol_name: str
    charge_method: ChargeMethod
    compatibility: ReceptorCompatibilityReport


@dataclass(frozen=True)
class MetalChargeCompatibility:
    charge_method: ChargeMethod
    incompatibility_reasons: tuple[str, ...] = ()
    _registry: ClassVar[dict[ChargeMethod, "MetalChargeCompatibility"]] = {}

    @classmethod
    def register(
        cls,
        *,
        charge_method: ChargeMethod,
        incompatibility_reasons: tuple[str, ...] = (),
    ) -> None:
        if charge_method in cls._registry:
            raise TypeError(
                f"Duplicate metal charge compatibility registration for {charge_method.name}."
            )
        cls._registry[charge_method] = cls(
            charge_method=charge_method,
            incompatibility_reasons=incompatibility_reasons,
        )

    @classmethod
    def for_charge_method(
        cls, charge_method: ChargeMethod
    ) -> "MetalChargeCompatibility":
        if charge_method not in cls._registry:
            raise TypeError(
                f"No metal charge compatibility policy registered for {charge_method.name}."
            )
        return cls._registry[charge_method]


@dataclass(frozen=True)
class ScoringFamilyCompatibility:
    scoring_family: CertifiedScoringFamily
    structural_water_reasons: tuple[str, ...] = ()
    structural_metal_reasons: tuple[str, ...] = ()
    structural_cofactor_reasons: tuple[str, ...] = ()
    _registry: ClassVar[dict[CertifiedScoringFamily, "ScoringFamilyCompatibility"]] = {}

    @classmethod
    def register(
        cls,
        *,
        scoring_family: CertifiedScoringFamily,
        structural_water_reasons: tuple[str, ...] = (),
        structural_metal_reasons: tuple[str, ...] = (),
        structural_cofactor_reasons: tuple[str, ...] = (),
    ) -> None:
        if scoring_family in cls._registry:
            raise TypeError(
                f"Duplicate scoring-family compatibility registration for {scoring_family.name}."
            )
        cls._registry[scoring_family] = cls(
            scoring_family=scoring_family,
            structural_water_reasons=structural_water_reasons,
            structural_metal_reasons=structural_metal_reasons,
            structural_cofactor_reasons=structural_cofactor_reasons,
        )

    @classmethod
    def for_scoring_family(
        cls, scoring_family: CertifiedScoringFamily
    ) -> "ScoringFamilyCompatibility":
        if scoring_family not in cls._registry:
            raise TypeError(
                f"No scoring-family compatibility policy registered for {scoring_family.name}."
            )
        return cls._registry[scoring_family]


CompatibilityProjector = Callable[
    [ScoringFamilyCompatibility, MetalChargeCompatibility], tuple[str, ...]
]


@dataclass(frozen=True)
class ComponentCompatibilityRule:
    role: ReceptorComponentRole
    incompatibility_projector: CompatibilityProjector
    _registry: ClassVar[dict[ReceptorComponentRole, "ComponentCompatibilityRule"]] = {}

    @classmethod
    def register(
        cls,
        *,
        role: ReceptorComponentRole,
        incompatibility_projector: CompatibilityProjector,
    ) -> None:
        if role in cls._registry:
            raise TypeError(
                f"Duplicate component compatibility registration for {role.name}."
            )
        cls._registry[role] = cls(
            role=role,
            incompatibility_projector=incompatibility_projector,
        )

    @classmethod
    def for_component(
        cls, component: ReceptorComponent
    ) -> "ComponentCompatibilityRule":
        if component.role not in cls._registry:
            raise TypeError(
                f"No component compatibility rule registered for {component.role.name}."
            )
        return cls._registry[component.role]

    def incompatibility_reasons(
        self,
        component: ReceptorComponent,
        *,
        scoring_compatibility: ScoringFamilyCompatibility,
        metal_charge_compatibility: MetalChargeCompatibility,
    ) -> tuple[str, ...]:
        del component
        return self.incompatibility_projector(
            scoring_compatibility,
            metal_charge_compatibility,
        )


MetalChargeCompatibility.register(
    charge_method=ChargeMethod.GASTEIGER,
    incompatibility_reasons=(
        "Gasteiger charge assignment is not a supported metal-aware charge protocol for retained structural metals",
    ),
)
MetalChargeCompatibility.register(charge_method=ChargeMethod.AM1BCC)
MetalChargeCompatibility.register(charge_method=ChargeMethod.SIMPLE)

ScoringFamilyCompatibility.register(
    scoring_family=CertifiedScoringFamily.LJ,
    structural_water_reasons=(
        "retained structural waters require a hydrated docking protocol not provided by the current benchmark scoring",
    ),
    structural_metal_reasons=(
        "retained structural metals require specialized metal-aware scoring beyond the current benchmark scoring families",
    ),
    structural_cofactor_reasons=(
        "retained structural cofactors require electrostatic scoring support; pure LJ benchmark mode is not chemically faithful",
    ),
)
ScoringFamilyCompatibility.register(
    scoring_family=CertifiedScoringFamily.LJ_REALSPACE_EWALD,
    structural_water_reasons=(
        "retained structural waters require a hydrated docking protocol not provided by the current benchmark scoring",
    ),
    structural_metal_reasons=(
        "retained structural metals require specialized metal-aware scoring beyond the current benchmark scoring families",
    ),
)

ComponentCompatibilityRule.register(
    role=ReceptorComponentRole.POLYMER,
    incompatibility_projector=lambda scoring, charge: (),
)
ComponentCompatibilityRule.register(
    role=ReceptorComponentRole.PRIMARY_LIGAND,
    incompatibility_projector=lambda scoring, charge: (),
)
ComponentCompatibilityRule.register(
    role=ReceptorComponentRole.STRUCTURAL_WATER,
    incompatibility_projector=lambda scoring, charge: scoring.structural_water_reasons,
)
ComponentCompatibilityRule.register(
    role=ReceptorComponentRole.STRUCTURAL_METAL,
    incompatibility_projector=lambda scoring, charge: (
        *scoring.structural_metal_reasons,
        *charge.incompatibility_reasons,
    ),
)
ComponentCompatibilityRule.register(
    role=ReceptorComponentRole.STRUCTURAL_COFACTOR,
    incompatibility_projector=lambda scoring, charge: (
        scoring.structural_cofactor_reasons
    ),
)
ComponentCompatibilityRule.register(
    role=ReceptorComponentRole.OTHER_HETATM,
    incompatibility_projector=lambda scoring, charge: (),
)


class ReceptorPreparationPolicy(ABC):
    name: ClassVar[str]

    @abstractmethod
    def classify_component(
        self,
        residue: ResidueKey,
        records: Sequence[PDBAtomRecord],
        primary_ligand: ResidueKey,
    ) -> ReceptorComponentRole:
        """Classify the residue-level component role."""

    @abstractmethod
    def retain_component(self, component: ReceptorComponent) -> bool:
        """Whether the full receptor should retain this component."""

    def retain_pocket_component(self, component: ReceptorComponent) -> bool:
        return self.retain_component(component)


class ProteinOnlyReceptorPolicy(ReceptorPreparationPolicy):
    name = "protein_only"

    def classify_component(
        self,
        residue: ResidueKey,
        records: Sequence[PDBAtomRecord],
        primary_ligand: ResidueKey,
    ) -> ReceptorComponentRole:
        del residue, primary_ligand
        if any(record.record_name == "ATOM" for record in records):
            return ReceptorComponentRole.POLYMER
        return ReceptorComponentRole.OTHER_HETATM

    def retain_component(self, component: ReceptorComponent) -> bool:
        return component.role == ReceptorComponentRole.POLYMER


class EssentialSiteComponentsPolicy(ReceptorPreparationPolicy):
    name = "essential_site_components"

    def classify_component(
        self,
        residue: ResidueKey,
        records: Sequence[PDBAtomRecord],
        primary_ligand: ResidueKey,
    ) -> ReceptorComponentRole:
        if residue == primary_ligand:
            return ReceptorComponentRole.PRIMARY_LIGAND
        if any(record.record_name == "ATOM" for record in records):
            return ReceptorComponentRole.POLYMER
        if residue.resname in WATER_RESNAMES:
            return ReceptorComponentRole.STRUCTURAL_WATER
        if any(record.element.upper() in METAL_ELEMENTS for record in records):
            return ReceptorComponentRole.STRUCTURAL_METAL
        return ReceptorComponentRole.STRUCTURAL_COFACTOR

    def retain_component(self, component: ReceptorComponent) -> bool:
        return component.role in {
            ReceptorComponentRole.POLYMER,
            ReceptorComponentRole.STRUCTURAL_METAL,
            ReceptorComponentRole.STRUCTURAL_COFACTOR,
        }


class HydratedEssentialSitePolicy(EssentialSiteComponentsPolicy):
    name = "hydrated_essential_site"

    def retain_component(self, component: ReceptorComponent) -> bool:
        return component.role in {
            ReceptorComponentRole.POLYMER,
            ReceptorComponentRole.STRUCTURAL_METAL,
            ReceptorComponentRole.STRUCTURAL_COFACTOR,
            ReceptorComponentRole.STRUCTURAL_WATER,
        }


def _extract_element(line: str) -> str:
    element = line[76:78].strip() if len(line) > 77 else ""
    if element:
        return element.upper()
    atom_name = line[12:16].strip()
    return atom_name.lstrip("0123456789")[:1].upper()


def _iter_atom_records(pdb_path: Path) -> tuple[PDBAtomRecord, ...]:
    records: list[PDBAtomRecord] = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[16] not in ALLOWED_ALTLOCS:
                continue
            try:
                coord = np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=np.float64,
                )
            except ValueError:
                continue
            records.append(
                PDBAtomRecord(
                    line=line,
                    record_name=line[:6].strip(),
                    atom_name=line[12:16].strip(),
                    residue=ResidueKey(
                        resname=line[17:20].strip(),
                        chain_id=line[21].strip(),
                        residue_id=f"{line[22:26].strip()}{line[26].strip()}",
                    ),
                    element=_extract_element(line),
                    coord=coord,
                )
            )
    return tuple(records)


def iter_atom_records(pdb_path: Path) -> tuple[PDBAtomRecord, ...]:
    return _iter_atom_records(pdb_path)


def _group_records_by_residue(
    records: Sequence[PDBAtomRecord],
) -> dict[ResidueKey, list[PDBAtomRecord]]:
    grouped: dict[ResidueKey, list[PDBAtomRecord]] = {}
    for record in records:
        grouped.setdefault(record.residue, []).append(record)
    return grouped


def group_atom_records_by_residue(
    records: Sequence[PDBAtomRecord],
) -> dict[ResidueKey, tuple[PDBAtomRecord, ...]]:
    return {
        residue: tuple(residue_records)
        for residue, residue_records in _group_records_by_residue(records).items()
    }


def _component_from_records(
    residue: ResidueKey,
    records: Sequence[PDBAtomRecord],
    *,
    policy: ReceptorPreparationPolicy,
    primary_ligand: ResidueKey,
) -> ReceptorComponent:
    return ReceptorComponent(
        residue=residue,
        role=policy.classify_component(residue, records, primary_ligand),
        atom_count=len(records),
        elements=tuple(record.element for record in records),
    )


def _write_atom_records(records: Iterable[PDBAtomRecord], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w") as f:
        for record in records:
            f.write(record.line)


def prepare_receptor_system(
    pdb_path: Path,
    *,
    center: tuple[float, float, float],
    pocket_radius: float,
    primary_ligand: ResidueKey,
    policy: ReceptorPreparationPolicy,
    receptor_output_path: Path | None = None,
    pocket_output_path: Path | None = None,
) -> PreparedReceptorSystem:
    records = _iter_atom_records(pdb_path)
    grouped = _group_records_by_residue(records)
    component_records: list[tuple[ReceptorComponent, tuple[PDBAtomRecord, ...]]] = []
    for residue, residue_records in grouped.items():
        component = _component_from_records(
            residue,
            residue_records,
            policy=policy,
            primary_ligand=primary_ligand,
        )
        component_records.append((component, tuple(residue_records)))

    retained_component_records = [
        (component, residue_records)
        for component, residue_records in component_records
        if policy.retain_component(component)
    ]
    if not retained_component_records:
        raise ValueError(f"No receptor components retained from {pdb_path}")

    center_array = np.array(center, dtype=np.float64)
    pocket_component_records: list[
        tuple[ReceptorComponent, tuple[PDBAtomRecord, ...]]
    ] = []
    pocket_atom_records: list[PDBAtomRecord] = []
    for component, residue_records in retained_component_records:
        if not policy.retain_pocket_component(component):
            continue
        retained_atoms = tuple(
            record
            for record in residue_records
            if np.linalg.norm(record.coord - center_array) < pocket_radius
        )
        if not retained_atoms:
            continue
        pocket_component_records.append((component, retained_atoms))
        pocket_atom_records.extend(retained_atoms)

    if not pocket_atom_records:
        raise ValueError(f"No pocket atoms found in {pdb_path}")

    receptor_pdb = (
        receptor_output_path
        if receptor_output_path is not None
        else pdb_path.parent / f"{pdb_path.stem}_{policy.name}.pdb"
    )
    pocket_receptor_pdb = (
        pocket_output_path
        if pocket_output_path is not None
        else pdb_path.parent / f"{pdb_path.stem}_{policy.name}_pocket.pdb"
    )
    _write_atom_records(
        (
            record
            for _, residue_records in retained_component_records
            for record in residue_records
        ),
        receptor_pdb,
    )
    _write_atom_records(pocket_atom_records, pocket_receptor_pdb)

    receptor_coords, receptor_radii, receptor_elements = cast(
        tuple[np.ndarray, np.ndarray, list[str]],
        parse_structure(receptor_pdb, return_elements=True),
    )
    pocket_coords, pocket_radii, pocket_elements = cast(
        tuple[np.ndarray, np.ndarray, list[str]],
        parse_structure(pocket_receptor_pdb, return_elements=True),
    )
    return PreparedReceptorSystem(
        source_pdb=pdb_path,
        receptor_pdb=receptor_pdb,
        pocket_receptor_pdb=pocket_receptor_pdb,
        retained_components=tuple(
            component for component, _ in retained_component_records
        ),
        pocket_components=tuple(component for component, _ in pocket_component_records),
        receptor_coords=receptor_coords,
        receptor_radii=receptor_radii,
        receptor_elements=tuple(receptor_elements),
        pocket_coords=pocket_coords,
        pocket_radii=pocket_radii,
        pocket_elements=tuple(pocket_elements),
        policy_name=policy.name,
    )


def assess_receptor_compatibility(
    system: PreparedReceptorSystem,
    *,
    charge_method: ChargeMethod,
    certified_scoring_family: CertifiedScoringFamily,
) -> ReceptorCompatibilityReport:
    scoring_compatibility = ScoringFamilyCompatibility.for_scoring_family(
        certified_scoring_family
    )
    metal_charge_compatibility = MetalChargeCompatibility.for_charge_method(
        charge_method
    )
    reasons = tuple(
        dict.fromkeys(
            reason
            for component in system.pocket_components
            for reason in ComponentCompatibilityRule.for_component(
                component
            ).incompatibility_reasons(
                component,
                scoring_compatibility=scoring_compatibility,
                metal_charge_compatibility=metal_charge_compatibility,
            )
        )
    )
    return ReceptorCompatibilityReport(supported=not reasons, reasons=reasons)


class BenchmarkChemistryProtocol(ABC):
    name: ClassVar[str | None] = None
    _registered_types: ClassVar[list[type["BenchmarkChemistryProtocol"]]] = []

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("name") is not None:
            cls._registered_types.append(cls)

    @classmethod
    def derive_plan(
        cls,
        system: PreparedReceptorSystem,
        *,
        requested_charge_method: ChargeMethod,
        certified_scoring_family: CertifiedScoringFamily,
    ) -> ReceptorChemistryPlan:
        matches = [
            candidate
            for candidate in cls._registered_types
            if candidate.matches_system(system)
        ]
        if len(matches) != 1:
            raise TypeError(
                f"Expected exactly one benchmark chemistry protocol for receptor system, found {[candidate.__name__ for candidate in matches]}"
            )
        protocol = matches[0]()
        resolved_charge_method = protocol.resolve_charge_method(requested_charge_method)
        compatibility = protocol.assess(
            system,
            charge_method=resolved_charge_method,
            certified_scoring_family=certified_scoring_family,
        )
        assert protocol.name is not None
        return ReceptorChemistryPlan(
            protocol_name=protocol.name,
            charge_method=resolved_charge_method,
            compatibility=compatibility,
        )

    @classmethod
    @abstractmethod
    def matches_system(cls, system: PreparedReceptorSystem) -> bool:
        """Return whether this protocol is the nominal chemistry branch for the receptor system."""

    def resolve_charge_method(
        self, requested_charge_method: ChargeMethod
    ) -> ChargeMethod:
        return requested_charge_method

    @abstractmethod
    def assess(
        self,
        system: PreparedReceptorSystem,
        *,
        charge_method: ChargeMethod,
        certified_scoring_family: CertifiedScoringFamily,
    ) -> ReceptorCompatibilityReport:
        """Assess compatibility under this chemistry protocol."""


class HydratedUnsupportedChemistryProtocol(BenchmarkChemistryProtocol):
    name = "hydrated_unsupported"

    @classmethod
    def matches_system(cls, system: PreparedReceptorSystem) -> bool:
        return system.has_structural_waters

    def assess(
        self,
        system: PreparedReceptorSystem,
        *,
        charge_method: ChargeMethod,
        certified_scoring_family: CertifiedScoringFamily,
    ) -> ReceptorCompatibilityReport:
        return assess_receptor_compatibility(
            system,
            charge_method=charge_method,
            certified_scoring_family=certified_scoring_family,
        )


class MetalAwareBenchmarkChemistryProtocol(BenchmarkChemistryProtocol):
    name = "metal_aware"

    @classmethod
    def matches_system(cls, system: PreparedReceptorSystem) -> bool:
        return system.has_structural_metals and not system.has_structural_waters

    def resolve_charge_method(
        self, requested_charge_method: ChargeMethod
    ) -> ChargeMethod:
        return (
            ChargeMethod.SIMPLE
            if requested_charge_method == ChargeMethod.GASTEIGER
            else requested_charge_method
        )

    def assess(
        self,
        system: PreparedReceptorSystem,
        *,
        charge_method: ChargeMethod,
        certified_scoring_family: CertifiedScoringFamily,
    ) -> ReceptorCompatibilityReport:
        reasons: list[str] = []
        if certified_scoring_family == CertifiedScoringFamily.LJ:
            reasons.append(
                "metal-aware protocol requires electrostatic scoring; pure LJ benchmark mode is not chemically faithful"
            )
        if (
            system.has_structural_cofactors
            and certified_scoring_family == CertifiedScoringFamily.LJ
        ):
            reasons.append(
                "retained structural cofactors require electrostatic scoring support; pure LJ benchmark mode is not chemically faithful"
            )
        return ReceptorCompatibilityReport(
            supported=not reasons, reasons=tuple(reasons)
        )


class StandardBenchmarkChemistryProtocol(BenchmarkChemistryProtocol):
    name = "standard"

    @classmethod
    def matches_system(cls, system: PreparedReceptorSystem) -> bool:
        return not system.has_structural_metals and not system.has_structural_waters

    def assess(
        self,
        system: PreparedReceptorSystem,
        *,
        charge_method: ChargeMethod,
        certified_scoring_family: CertifiedScoringFamily,
    ) -> ReceptorCompatibilityReport:
        return assess_receptor_compatibility(
            system,
            charge_method=charge_method,
            certified_scoring_family=certified_scoring_family,
        )


def prepare_protein_only_receptor(
    pdb_path: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    system = prepare_receptor_system(
        pdb_path,
        center=(0.0, 0.0, 0.0),
        pocket_radius=float("inf"),
        primary_ligand=protein_only_residue_key(),
        policy=ProteinOnlyReceptorPolicy(),
        receptor_output_path=output_path,
        pocket_output_path=(
            pdb_path.parent / f"{pdb_path.stem}_protein_only_full.pdb"
            if output_path is None
            else output_path.parent / f"{output_path.stem}_full_copy.pdb"
        ),
    )
    return system.receptor_pdb


def prepare_protein_only_pocket(
    pdb_path: Path,
    *,
    center: tuple[float, float, float],
    pocket_radius: float,
    output_path: Path | None = None,
) -> Path:
    system = prepare_receptor_system(
        pdb_path,
        center=center,
        pocket_radius=pocket_radius,
        primary_ligand=protein_only_residue_key(),
        policy=ProteinOnlyReceptorPolicy(),
        receptor_output_path=(
            pdb_path.parent / f"{pdb_path.stem}_protein_only_source.pdb"
            if output_path is None
            else output_path.parent / f"{output_path.stem}_source.pdb"
        ),
        pocket_output_path=output_path,
    )
    return system.pocket_receptor_pdb


def protein_only_residue_key() -> ResidueKey:
    return ResidueKey(resname="", chain_id="", residue_id="")
