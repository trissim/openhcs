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
    catalog_from_ligand_structure,
    catalog_from_receptor_structure,
)
from dq_dock_engine.docking.chemistry_preparation import (
    prepare_ligand_chemistry_structure,
    prepare_receptor_chemistry_structure,
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
from dq_dock_engine.docking.physics_params import build_pairwise_sigma_matrix
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
    metal_coordination_min_cutoff,
    screened_coulomb_min_cutoff,
)


_DEFAULT_ELECTROSTATIC_TARGET_ERROR = 0.5
_DEFAULT_METAL_COORDINATION_TARGET_ERROR = 0.01  # kcal/mol
KAPPA_METAL = 1.0


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


@dataclass(frozen=True)
class DirectionalHBondChannelBuildSpec:
    plan_field: str
    receptor_role: ChemistrySiteRole
    ligand_role: ChemistrySiteRole
    receptor_alignment_sign: float
    ligand_alignment_sign: float


HBOND_CHANNEL_BUILD_SPECS: tuple[DirectionalHBondChannelBuildSpec, ...] = (
    DirectionalHBondChannelBuildSpec(
        plan_field="hbond_receptor_donor",
        receptor_role=ChemistrySiteRole.HBOND_DONOR,
        ligand_role=ChemistrySiteRole.HBOND_ACCEPTOR,
        receptor_alignment_sign=1.0,
        ligand_alignment_sign=-1.0,
    ),
    DirectionalHBondChannelBuildSpec(
        plan_field="hbond_ligand_donor",
        receptor_role=ChemistrySiteRole.HBOND_ACCEPTOR,
        ligand_role=ChemistrySiteRole.HBOND_DONOR,
        receptor_alignment_sign=-1.0,
        ligand_alignment_sign=1.0,
    ),
)


def build_directional_hbond_spec(
    receptor_catalog: ChemistryAnnotationCatalog,
    ligand_catalog: ChemistryAnnotationCatalog,
    *,
    ligand_frame_coords: np.ndarray | jnp.ndarray,
    receptor_role: ChemistrySiteRole,
    ligand_role: ChemistrySiteRole,
    receptor_alignment_sign: float,
    ligand_alignment_sign: float,
    ideal_distance: float = 2.8,
    distance_width: float = 0.8,
    cutoff: float = 4.0,
) -> CertifiedDirectionalHBondSpec:
    receptor_sites = receptor_catalog.typed_sites(
        DirectionalSiteAnnotation,
        role=receptor_role,
    )
    ligand_sites = ligand_catalog.typed_sites(
        DirectionalSiteAnnotation,
        role=ligand_role,
    )

    def _anchor_array(sites: tuple[DirectionalSiteAnnotation, ...]) -> jnp.ndarray:
        return jnp.array([site.anchor_index for site in sites], dtype=jnp.int32)

    def _vector_array(sites: tuple[DirectionalSiteAnnotation, ...]) -> jnp.ndarray:
        if not sites:
            return jnp.zeros((0, 3), dtype=jnp.float32)
        return jnp.array(
            np.stack([site.direction for site in sites], axis=0),
            dtype=jnp.float32,
        )

    def _strength_array(sites: tuple[DirectionalSiteAnnotation, ...]) -> jnp.ndarray:
        return jnp.array([site.strength for site in sites], dtype=jnp.float32)

    return CertifiedDirectionalHBondSpec(
        receptor_anchor_indices=_anchor_array(receptor_sites),
        receptor_directions=_vector_array(receptor_sites),
        ligand_anchor_indices=_anchor_array(ligand_sites),
        ligand_local_directions=_vector_array(ligand_sites),
        ligand_frame_coords=jnp.array(ligand_frame_coords, dtype=jnp.float32),
        receptor_strengths=_strength_array(receptor_sites),
        ligand_strengths=_strength_array(ligand_sites),
        receptor_alignment_sign=receptor_alignment_sign,
        ligand_alignment_sign=ligand_alignment_sign,
        ideal_distance=ideal_distance,
        distance_width=distance_width,
        cutoff=cutoff,
    )


def build_metal_coordination_spec(
    receptor_elements: tuple[str, ...],
    ligand_elements: tuple[str, ...],
    *,
    ideal_distance: float = 2.1,
    distance_width: float = 0.3,
    angle_width: float = 0.5,
    target_error: float = _DEFAULT_METAL_COORDINATION_TARGET_ERROR,
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
    # Per-site ideal coordination angle offsets.
    # Zero offset = ideal tetrahedral; nonzero encodes geometry preference.
    receptor_ideal_angles = np.zeros_like(receptor_strengths)
    # Derive cutoff from physical parameters using the Lean-certified formula:
    # rc = ideal + width · √(ln(|w|/ε))  (MetalCoordinationApproximation.lean)
    max_rec = float(np.max(np.abs(receptor_strengths))) if len(receptor_strengths) > 0 else 0.0
    max_lig = float(np.max(np.abs(ligand_strengths))) if len(ligand_strengths) > 0 else 0.0
    max_strength = max_rec * max_lig
    if max_strength > 0:
        cutoff = metal_coordination_min_cutoff(
            max_strength_product=max_strength,
            target_error=target_error,
            ideal_distance=ideal_distance,
            distance_width=distance_width,
        )
    else:
        cutoff = ideal_distance  # No active interactions
    return CertifiedMetalCoordinationSpec(
        receptor_strengths=jnp.array(receptor_strengths),
        ligand_strengths=jnp.array(ligand_strengths),
        receptor_ideal_angles=jnp.array(receptor_ideal_angles),
        ideal_distance=ideal_distance,
        distance_width=distance_width,
        angle_width=angle_width,
        cutoff=cutoff,
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


def _derive_cooperative_alpha(
    hbond_channels: dict[str, CertifiedDirectionalHBondSpec],
) -> float:
    """Derive cooperative coupling constant from H-bond channel strengths.

    Uses the geometric mean of average strengths across both channels,
    clamped to the Lean-specified range [0.1, 0.3].  Returns 0.0 when
    no H-bond sites are present (cooperative correction is vacuous).
    """
    all_strengths: list[float] = []
    for spec in hbond_channels.values():
        rec_s = np.asarray(spec.receptor_strengths)
        lig_s = np.asarray(spec.ligand_strengths)
        if rec_s.size > 0:
            all_strengths.append(float(np.mean(np.abs(rec_s))))
        if lig_s.size > 0:
            all_strengths.append(float(np.mean(np.abs(lig_s))))
    if not all_strengths:
        return 0.0
    geo_mean = float(np.exp(np.mean(np.log(np.clip(all_strengths, 1e-8, None)))))
    # Scale: strength ≈ 1.0 → α ≈ 0.15 (midpoint of [0.1, 0.3])
    alpha = 0.15 * geo_mean
    if alpha < 0.1:
        return 0.0  # No meaningful H-bond interactions → no cooperative correction
    return float(np.clip(alpha, 0.1, 0.3))


def build_certified_rich_chemistry_plan(
    receptor_coords: np.ndarray,
    receptor_elements: tuple[str, ...],
    receptor_charges: np.ndarray,
    ligand_ctx: LigandContext,
    *,
    receptor_file: str | Path | None,
    ligand_source_path: str | Path | None,
    target_electrostatic_error: float = _DEFAULT_ELECTROSTATIC_TARGET_ERROR,
    cooperative_alpha: float | None = None,
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
    receptor_catalog = catalog_from_receptor_structure(
        prepare_receptor_chemistry_structure(
            receptor_coords,
            receptor_elements,
            receptor_file=receptor_file,
        )
    )
    ligand_catalog = catalog_from_ligand_structure(
        prepare_ligand_chemistry_structure(
            ligand_ctx,
            ligand_source_path=ligand_source_path,
        )
    )
    hbond_channels = {
        channel_spec.plan_field: build_directional_hbond_spec(
            receptor_catalog,
            ligand_catalog,
            ligand_frame_coords=ligand_coords,
            receptor_role=channel_spec.receptor_role,
            ligand_role=channel_spec.ligand_role,
            receptor_alignment_sign=channel_spec.receptor_alignment_sign,
            ligand_alignment_sign=channel_spec.ligand_alignment_sign,
        )
        for channel_spec in HBOND_CHANNEL_BUILD_SPECS
    }
    # Derive cooperative_alpha from H-bond strengths when not explicitly set.
    # CHN1 (CooperativeHBondApproximation.lean): α ∈ [0.1, 0.3] typically.
    # We use the geometric mean of average donor/acceptor strengths, clamped to
    # the Lean-specified range.  When no H-bond sites exist, α = 0 (no correction).
    if cooperative_alpha is None:
        cooperative_alpha = _derive_cooperative_alpha(hbond_channels)
    pairwise_sigma = jnp.array(
        build_pairwise_sigma_matrix(receptor_elements, ligand_ctx.elements)
    )
    return CertifiedRichChemistryPlan(
        screened_coulomb=build_screened_coulomb_spec(
            np.asarray(receptor_charges, dtype=np.float32),
            ligand_charges,
            kappa=kappa,
            cutoff=cutoff,
        ),
        contact=build_contact_surrogate_spec(receptor_elements, ligand_ctx.elements),
        hbond_receptor_donor=hbond_channels["hbond_receptor_donor"],
        hbond_ligand_donor=hbond_channels["hbond_ligand_donor"],
        metal_coordination=build_metal_coordination_spec(
            receptor_elements, ligand_ctx.elements
        ),
        pairwise_sigma=pairwise_sigma,
        cooperative_alpha=cooperative_alpha,
        extended_terms=CertifiedExtendedInteractionBundle(
            terms=build_extended_interaction_terms(receptor_catalog, ligand_catalog)
        ).filter_active(),
    )
