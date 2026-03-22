from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np

from dq_dock_engine.docking.chemistry_annotations import (
    ChemistryAnnotationCatalog,
    ChemistrySiteRole,
    DirectionalSiteAnnotation,
    IndexedSiteAnnotation,
    RingSiteAnnotation,
    build_ligand_chemistry_catalog,
    build_receptor_chemistry_catalog,
)
from dq_dock_engine.docking.chemistry_runtime import (
    AnchoredSiteArray,
    HalogenBondInteractionTerm,
    IndexedSiteArray,
    PiCationInteractionTerm,
    PiStackingInteractionTerm,
    SiteGeometry,
    WaterMediatedHBondInteractionTerm,
)
from dq_dock_engine.docking.core import LigandContext
from dq_dock_engine.docking.receptor_preparation import METAL_ELEMENTS
from dq_dock_engine.docking.scoring import (
    CertifiedContactSurrogateSpec,
    CertifiedDirectionalHBondSpec,
    CertifiedExtendedInteractionBundle,
    CertifiedMetalCoordinationSpec,
    CertifiedRichChemistryPlan,
    CertifiedScreenedCoulombSpec,
    KAPPA_PHYSIOLOGICAL,
    CertifiedOptionalInteractionTerm,
    screened_coulomb_min_cutoff,
)


_DEFAULT_ELECTROSTATIC_TARGET_ERROR = 0.5
KAPPA_METAL = 1.0


def find_k_nearest_neighbors(coords: np.ndarray, k: int) -> np.ndarray:
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dists, np.inf)
    if dists.shape[0] <= k:
        return np.tile(np.arange(dists.shape[0]), (dists.shape[0], 1))[:, :k]
    return np.argsort(dists, axis=-1)[:, :k]


def build_screened_coulomb_spec(
    receptor_charges: np.ndarray | jnp.ndarray,
    ligand_charges: np.ndarray | jnp.ndarray,
    *,
    kappa: float,
    cutoff: float,
    dielectric: float = 4.0,
) -> CertifiedScreenedCoulombSpec:
    return CertifiedScreenedCoulombSpec(
        receptor_charges=jnp.array(receptor_charges, dtype=jnp.float32),
        ligand_charges=jnp.array(ligand_charges, dtype=jnp.float32),
        kappa=kappa,
        cutoff=cutoff,
        dielectric=dielectric,
    )


def build_contact_surrogate_spec(
    receptor_elements: tuple[str, ...],
    ligand_elements: tuple[str, ...],
    *,
    beta: float = 0.6,
    cutoff: float = 6.0,
) -> CertifiedContactSurrogateSpec:
    polar_elements = {"N", "O", "S", "F", "CL", "BR", "I"}
    receptor_weights = np.array(
        [
            1.0 if element.upper() in polar_elements else 0.5
            for element in receptor_elements
        ],
        dtype=np.float32,
    )
    ligand_weights = np.array(
        [
            1.0 if element.upper() in polar_elements else 0.5
            for element in ligand_elements
        ],
        dtype=np.float32,
    )
    return CertifiedContactSurrogateSpec(
        receptor_weights=jnp.array(receptor_weights),
        ligand_weights=jnp.array(ligand_weights),
        beta=beta,
        cutoff=cutoff,
    )


def build_directional_hbond_spec(
    receptor_coords: np.ndarray | jnp.ndarray,
    receptor_elements: tuple[str, ...],
    ligand_coords: np.ndarray | jnp.ndarray,
    ligand_elements: tuple[str, ...],
    *,
    ideal_distance: float = 2.8,
    distance_width: float = 0.8,
    cutoff: float = 4.0,
) -> CertifiedDirectionalHBondSpec:
    receptor_coords = np.asarray(receptor_coords, dtype=np.float32)
    ligand_coords = np.asarray(ligand_coords, dtype=np.float32)
    polar_elements = {"N", "O", "F"}
    receptor_strengths = np.array(
        [
            1.0 if element.upper() in polar_elements else 0.0
            for element in receptor_elements
        ],
        dtype=np.float32,
    )
    ligand_strengths = np.array(
        [
            1.0 if element.upper() in polar_elements else 0.0
            for element in ligand_elements
        ],
        dtype=np.float32,
    )
    if receptor_coords.shape[0] > 0:
        receptor_neighbors = find_k_nearest_neighbors(receptor_coords, k=3)
        receptor_neighbor_coords = receptor_coords[receptor_neighbors]
        receptor_mean_neighbor = np.mean(receptor_neighbor_coords, axis=1)
        receptor_directions = receptor_coords - receptor_mean_neighbor
        norms = np.linalg.norm(receptor_directions, axis=-1, keepdims=True)
        receptor_directions = np.where(norms > 1e-6, receptor_directions / norms, 0.0)
    else:
        receptor_directions = np.zeros((0, 3), dtype=np.float32)
    ligand_neighbors = (
        find_k_nearest_neighbors(ligand_coords, k=3)
        if ligand_coords.shape[0] > 0
        else np.zeros((0, 3), dtype=np.int32)
    )
    return CertifiedDirectionalHBondSpec(
        receptor_directions=jnp.array(receptor_directions, dtype=jnp.float32),
        ligand_neighbor_indices=jnp.array(ligand_neighbors, dtype=jnp.int32),
        receptor_strengths=jnp.array(receptor_strengths, dtype=jnp.float32),
        ligand_strengths=jnp.array(ligand_strengths, dtype=jnp.float32),
        ideal_distance=ideal_distance,
        distance_width=distance_width,
        cutoff=cutoff,
    )


def build_metal_coordination_spec(
    receptor_elements: tuple[str, ...],
    ligand_elements: tuple[str, ...],
) -> CertifiedMetalCoordinationSpec:
    polar_elements = {"N", "O", "S"}
    receptor_strengths = np.array(
        [
            50.0 if element.upper() in METAL_ELEMENTS else 0.0
            for element in receptor_elements
        ],
        dtype=np.float32,
    )
    ligand_strengths = np.array(
        [
            1.0 if element.upper() in polar_elements else 0.0
            for element in ligand_elements
        ],
        dtype=np.float32,
    )
    return CertifiedMetalCoordinationSpec(
        receptor_strengths=jnp.array(receptor_strengths),
        ligand_strengths=jnp.array(ligand_strengths),
    )


def _derive_electrostatic_cutoff(
    receptor_charges: np.ndarray | jnp.ndarray,
    ligand_charges: np.ndarray | jnp.ndarray,
    *,
    kappa: float,
    target_error: float,
) -> float:
    receptor_charges = np.asarray(receptor_charges, dtype=np.float32)
    ligand_charges = np.asarray(ligand_charges, dtype=np.float32)
    max_receptor_charge = (
        float(np.max(np.abs(receptor_charges))) if len(receptor_charges) > 0 else 1.0
    )
    max_ligand_charge = (
        float(np.max(np.abs(ligand_charges))) if len(ligand_charges) > 0 else 1.0
    )
    q_bound = max(
        max_receptor_charge
        * max_ligand_charge
        * max(len(receptor_charges) * len(ligand_charges), 1),
        1.0,
    )
    effective_kappa = kappa if kappa > 0 else KAPPA_PHYSIOLOGICAL
    return screened_coulomb_min_cutoff(
        max_charge_product=q_bound,
        target_error=target_error,
        kappa=effective_kappa,
        min_cutoff=4.0,
    )


def _pad_rows(
    rows: list[tuple[int, ...]], minimum_width: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    width = max(minimum_width, max((len(row) for row in rows), default=0), 1)
    values = np.zeros((len(rows), width), dtype=np.int32)
    mask = np.zeros((len(rows), width), dtype=bool)
    for row_index, row in enumerate(rows):
        limit = min(len(row), width)
        if limit > 0:
            values[row_index, :limit] = np.asarray(row[:limit], dtype=np.int32)
            mask[row_index, :limit] = True
    return values, mask


@dataclass(frozen=True)
class SiteArrayPackingRule:
    field_name: str
    source: Literal["receptor", "ligand"]
    role: ChemistrySiteRole
    geometry: SiteGeometry
    anchored: bool
    site_type: type[IndexedSiteAnnotation]


def _pack_anchored_sites(
    sites: tuple[IndexedSiteAnnotation, ...],
    geometry: SiteGeometry,
) -> AnchoredSiteArray:
    if not sites:
        return AnchoredSiteArray.empty(geometry)
    vector_attr = {
        SiteGeometry.POINT: None,
        SiteGeometry.DIRECTIONAL: "direction",
        SiteGeometry.RING: "normal",
    }[geometry]
    vectors = (
        np.zeros((len(sites), 3), dtype=np.float32)
        if vector_attr is None
        else np.stack([getattr(site, vector_attr) for site in sites], axis=0).astype(
            np.float32
        )
    )
    return AnchoredSiteArray(
        geometry=geometry,
        positions=jnp.array(
            np.stack([site.position for site in sites], axis=0), dtype=jnp.float32
        ),
        vectors=jnp.array(vectors, dtype=jnp.float32),
        strengths=jnp.array([site.strength for site in sites], dtype=jnp.float32),
        anchor_indices=jnp.array(
            [site.anchor_index for site in sites], dtype=jnp.int32
        ),
    )


def _reference_rows(
    site: IndexedSiteAnnotation, geometry: SiteGeometry
) -> tuple[int, ...]:
    if geometry == SiteGeometry.POINT:
        return ()
    if geometry == SiteGeometry.DIRECTIONAL:
        site = site  # narrow for pyright
        return getattr(site, "neighbor_indices", ())
    ring_indices = site.atom_indices[:3]
    if len(ring_indices) >= 3:
        return tuple(ring_indices)
    if not ring_indices:
        return ()
    return tuple(list(ring_indices) + [ring_indices[-1]] * (3 - len(ring_indices)))


def _pack_indexed_sites(
    sites: tuple[IndexedSiteAnnotation, ...],
    geometry: SiteGeometry,
) -> IndexedSiteArray:
    if not sites:
        return IndexedSiteArray.empty(geometry)
    atom_rows, atom_mask = _pad_rows(
        [site.atom_indices for site in sites], minimum_width=1
    )
    reference_rows, reference_mask = _pad_rows(
        [_reference_rows(site, geometry) for site in sites],
        minimum_width=3 if geometry == SiteGeometry.RING else 1,
    )
    return IndexedSiteArray(
        geometry=geometry,
        atom_index_rows=jnp.array(atom_rows, dtype=jnp.int32),
        atom_index_mask=jnp.array(atom_mask),
        reference_index_rows=jnp.array(reference_rows, dtype=jnp.int32),
        reference_index_mask=jnp.array(reference_mask),
        strengths=jnp.array([site.strength for site in sites], dtype=jnp.float32),
    )


def pack_site_array(
    catalog: ChemistryAnnotationCatalog,
    rule: SiteArrayPackingRule,
):
    sites = catalog.typed_sites(rule.site_type, role=rule.role)
    if rule.anchored:
        return _pack_anchored_sites(sites, rule.geometry)
    return _pack_indexed_sites(sites, rule.geometry)


@dataclass(frozen=True)
class InteractionBuildSpec:
    term_type: type[object]
    field_rules: tuple[SiteArrayPackingRule, ...]


INTERACTION_BUILD_SPECS: tuple[InteractionBuildSpec, ...] = (
    InteractionBuildSpec(
        term_type=PiStackingInteractionTerm,
        field_rules=(
            SiteArrayPackingRule(
                "receptor_rings",
                "receptor",
                ChemistrySiteRole.AROMATIC_RING,
                SiteGeometry.RING,
                True,
                RingSiteAnnotation,
            ),
            SiteArrayPackingRule(
                "ligand_rings",
                "ligand",
                ChemistrySiteRole.AROMATIC_RING,
                SiteGeometry.RING,
                False,
                RingSiteAnnotation,
            ),
        ),
    ),
    InteractionBuildSpec(
        term_type=PiCationInteractionTerm,
        field_rules=(
            SiteArrayPackingRule(
                "receptor_rings",
                "receptor",
                ChemistrySiteRole.AROMATIC_RING,
                SiteGeometry.RING,
                True,
                RingSiteAnnotation,
            ),
            SiteArrayPackingRule(
                "receptor_cations",
                "receptor",
                ChemistrySiteRole.CATION,
                SiteGeometry.POINT,
                True,
                IndexedSiteAnnotation,
            ),
            SiteArrayPackingRule(
                "ligand_rings",
                "ligand",
                ChemistrySiteRole.AROMATIC_RING,
                SiteGeometry.RING,
                False,
                RingSiteAnnotation,
            ),
            SiteArrayPackingRule(
                "ligand_cations",
                "ligand",
                ChemistrySiteRole.CATION,
                SiteGeometry.POINT,
                False,
                IndexedSiteAnnotation,
            ),
        ),
    ),
    InteractionBuildSpec(
        term_type=HalogenBondInteractionTerm,
        field_rules=(
            SiteArrayPackingRule(
                "receptor_acceptors",
                "receptor",
                ChemistrySiteRole.HALOGEN_ACCEPTOR,
                SiteGeometry.DIRECTIONAL,
                True,
                DirectionalSiteAnnotation,
            ),
            SiteArrayPackingRule(
                "receptor_donors",
                "receptor",
                ChemistrySiteRole.HALOGEN_DONOR,
                SiteGeometry.DIRECTIONAL,
                True,
                DirectionalSiteAnnotation,
            ),
            SiteArrayPackingRule(
                "ligand_acceptors",
                "ligand",
                ChemistrySiteRole.HALOGEN_ACCEPTOR,
                SiteGeometry.DIRECTIONAL,
                False,
                DirectionalSiteAnnotation,
            ),
            SiteArrayPackingRule(
                "ligand_donors",
                "ligand",
                ChemistrySiteRole.HALOGEN_DONOR,
                SiteGeometry.DIRECTIONAL,
                False,
                DirectionalSiteAnnotation,
            ),
        ),
    ),
    InteractionBuildSpec(
        term_type=WaterMediatedHBondInteractionTerm,
        field_rules=(
            SiteArrayPackingRule(
                "receptor_waters",
                "receptor",
                ChemistrySiteRole.BRIDGE_WATER,
                SiteGeometry.DIRECTIONAL,
                True,
                DirectionalSiteAnnotation,
            ),
            SiteArrayPackingRule(
                "ligand_polar_sites",
                "ligand",
                ChemistrySiteRole.POLAR,
                SiteGeometry.DIRECTIONAL,
                False,
                DirectionalSiteAnnotation,
            ),
        ),
    ),
)


def build_extended_interaction_terms(
    receptor_catalog: ChemistryAnnotationCatalog,
    ligand_catalog: ChemistryAnnotationCatalog,
) -> tuple[CertifiedOptionalInteractionTerm, ...]:
    catalogs = {"receptor": receptor_catalog, "ligand": ligand_catalog}
    built_terms = []
    for spec in INTERACTION_BUILD_SPECS:
        kwargs = {
            rule.field_name: pack_site_array(catalogs[rule.source], rule)
            for rule in spec.field_rules
        }
        built_terms.append(spec.term_type(**kwargs))
    return tuple(built_terms)


def build_certified_rich_chemistry_plan(
    receptor_coords: np.ndarray,
    receptor_elements: tuple[str, ...],
    receptor_charges: np.ndarray,
    ligand_ctx: LigandContext,
    *,
    receptor_file: str | Path | None,
    target_electrostatic_error: float = _DEFAULT_ELECTROSTATIC_TARGET_ERROR,
) -> CertifiedRichChemistryPlan:
    has_metals = any(element.upper() in METAL_ELEMENTS for element in receptor_elements)
    kappa = KAPPA_METAL if has_metals else 0.0
    ligand_charges: np.ndarray = (
        np.asarray(ligand_ctx.charges, dtype=np.float32)
        if ligand_ctx.charges is not None
        else np.zeros((ligand_ctx.base_coords.shape[0],), dtype=np.float32)
    )
    ligand_coords = np.asarray(ligand_ctx.base_coords, dtype=np.float32)
    cutoff = _derive_electrostatic_cutoff(
        receptor_charges=np.asarray(receptor_charges, dtype=np.float32),
        ligand_charges=ligand_charges,
        kappa=kappa,
        target_error=target_electrostatic_error,
    )
    receptor_catalog = build_receptor_chemistry_catalog(
        receptor_coords,
        receptor_elements,
        receptor_file=receptor_file,
    )
    ligand_catalog = build_ligand_chemistry_catalog(ligand_ctx)
    return CertifiedRichChemistryPlan(
        screened_coulomb=build_screened_coulomb_spec(
            np.asarray(receptor_charges, dtype=np.float32),
            ligand_charges,
            kappa=kappa,
            cutoff=cutoff,
        ),
        contact=build_contact_surrogate_spec(receptor_elements, ligand_ctx.elements),
        directional_hbond=build_directional_hbond_spec(
            np.asarray(receptor_coords, dtype=np.float32),
            receptor_elements,
            ligand_coords,
            ligand_ctx.elements,
        ),
        metal_coordination=build_metal_coordination_spec(
            receptor_elements, ligand_ctx.elements
        ),
        extended_terms=CertifiedExtendedInteractionBundle(
            terms=build_extended_interaction_terms(receptor_catalog, ligand_catalog)
        ),
    )
