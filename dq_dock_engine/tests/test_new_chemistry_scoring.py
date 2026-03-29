import jax
import jax.numpy as jnp
import numpy as np
import pytest
from typing import cast

import dq_dock_engine.docking.scoring as scoring
from dq_dock_engine.docking.chemistry_annotations import (
    ChemistrySiteRole,
    LigandHBondDonorExtractor,
    LigandStructureModel,
    MONOCATION_SITE_STRENGTH,
    ReceptorHBondDonorExtractor,
    ReceptorCationExtractor,
    ReceptorStructureModel,
    RingSiteAnnotation,
    _merge_overlapping_ring_sites,
    _ring_site,
)
from dq_dock_engine.docking.chemistry_runtime import (
    AROMATIC_FACE_OFFSET_WIDTH_ANGSTROM,
    AnchoredSiteArray,
    IndexedSiteArray,
    PiCationInteractionTerm,
    PiStackingInteractionTerm,
    SiteGeometry,
)
from dq_dock_engine.docking.formal_handles import (
    attractive_extended_chemistry_theorem_handles,
    attractive_directional_hbond_theorem_handles,
    attractive_halogen_bond_theorem_handles,
    attractive_pi_cation_theorem_handles,
    attractive_pi_stacking_theorem_handles,
    attractive_water_mediated_hbond_theorem_handles,
    ambiguity_output_contract_theorem_handles,
    conformer_search_theorem_handles,
    conformer_coverage_theorem_handles,
    contact_surrogate_theorem_handles,
    cooperative_hbond_theorem_handles,
    cross_docking_theorem_handles,
    directional_hbond_finite_theorem_handles,
    directional_metal_coordination_theorem_handles,
    explicit_water_placement_theorem_handles,
    extended_rich_chemistry_theorem_handles,
    joint_pruning_budget_optimality_handles,
    ligand_strain_theorem_handles,
    negation_invariance_theorem_handles,
    omitted_channel_bound_theorem_handles,
    optimizer_output_contract_theorem_handles,
    pose_specific_improvement_budget_theorem_handles,
    receptor_flexibility_theorem_handles,
    rigid_seed_family_theorem_handles,
    rigid_seed_runtime_bridge_theorem_handles,
    returned_pose_guarantee_theorem_handles,
    rich_chemistry_theorem_handles,
    screened_coulomb_theorem_handles,
    seed_budget_minimality_theorem_handles,
    softened_lj_shared_base_delta_theorem_handles,
    support_expansion_theorem_handles,
    topk_bridge_theorem_handles,
)
from dq_dock_engine.docking.receptor_preparation import PDBAtomRecord, ResidueKey
from dq_dock_engine.docking_config import ExactChemistryMode
from dq_dock_engine.docking.scoring_context import CertifiedScoringContext
from dq_dock_engine.proof_status import ProofStatus, get_status, get_theorem
from dq_dock_engine.docking.scoring import (
    CertifiedContactSurrogateSpec,
    CertifiedDirectionalHBondSpec,
    CertifiedMetalCoordinationSpec,
    CertifiedRichChemistryPlan,
    CertifiedScreenedCoulombSpec,
    cooperative_hbond_correction_bound,
    cooperative_hbond_correction_bound_sum_bounds,
    cosine_torsion_strain,
    score_certified_attractive_chemistry_batch,
    score_certified_contact_batch,
    score_certified_directional_hbond_batch,
    score_certified_lj_screened_coulomb_batch,
    score_certified_metal_coordination_batch,
    score_certified_rich_chemistry_batch,
    score_certified_screened_coulomb_batch,
    score_certified_softened_rich_chemistry_batch,
    total_torsion_strain,
    total_torsion_strain_bound,
    _cooperative_hbond_correction_batch,
    _score_directional_metal_exact_batch,
    _score_directional_metal_cutoff_batch,
)


def _make_hbond_spec(
    *,
    receptor_alignment_sign: float,
    ligand_alignment_sign: float,
    receptor_strengths: tuple[float, ...] = (0.8, 0.6),
    ligand_strengths: tuple[float, ...] = (0.9, 0.7),
) -> CertifiedDirectionalHBondSpec:
    return CertifiedDirectionalHBondSpec(
        receptor_anchor_indices=jnp.array([0, 1], dtype=jnp.int32),
        receptor_directions=jnp.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
            if receptor_alignment_sign > 0
            else [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=jnp.float32,
        ),
        ligand_anchor_indices=jnp.array([0, 1], dtype=jnp.int32),
        ligand_local_directions=jnp.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
            if ligand_alignment_sign > 0
            else [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=jnp.float32,
        ),
        ligand_frame_coords=jnp.array(
            [[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=jnp.float32
        ),
        receptor_strengths=jnp.array(receptor_strengths, dtype=jnp.float32),
        ligand_strengths=jnp.array(ligand_strengths, dtype=jnp.float32),
        receptor_alignment_sign=receptor_alignment_sign,
        ligand_alignment_sign=ligand_alignment_sign,
        ideal_distance=2.8,
        distance_width=0.7,
        cutoff=4.0,
    )


def _toy_geometry():
    receptor_coords = jnp.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=jnp.float32)
    poses_coords = jnp.array(
        [
            [[2.8, 0.0, 0.0], [3.2, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [4.4, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    receptor_radii = jnp.array([1.7, 1.6], dtype=jnp.float32)
    ligand_radii = jnp.array([1.5, 1.4], dtype=jnp.float32)
    screened = CertifiedScreenedCoulombSpec(
        receptor_charges=jnp.array([0.4, -0.3], dtype=jnp.float32),
        ligand_charges=jnp.array([-0.5, 0.2], dtype=jnp.float32),
        kappa=0.7,
        cutoff=3.5,
        dielectric=4.0,
    )
    contact = CertifiedContactSurrogateSpec(
        receptor_weights=jnp.array([0.8, 0.6], dtype=jnp.float32),
        ligand_weights=jnp.array([0.9, 0.7], dtype=jnp.float32),
        beta=0.55,
        cutoff=4.0,
    )
    metal = CertifiedMetalCoordinationSpec(
        receptor_strengths=jnp.array([1.0, 0.0], dtype=jnp.float32),
        ligand_strengths=jnp.array([1.0, 1.0], dtype=jnp.float32),
        receptor_ideal_angles=jnp.array([0.0, 0.0], dtype=jnp.float32),
    )
    plan = CertifiedRichChemistryPlan(
        screened_coulomb=screened,
        contact=contact,
        hbond_receptor_donor=_make_hbond_spec(
            receptor_alignment_sign=1.0,
            ligand_alignment_sign=-1.0,
        ),
        hbond_ligand_donor=_make_hbond_spec(
            receptor_alignment_sign=-1.0,
            ligand_alignment_sign=1.0,
            receptor_strengths=(0.5, 0.4),
            ligand_strengths=(0.6, 0.5),
        ),
        metal_coordination=metal,
        pairwise_sigma=jnp.full((2, 2), 3.40, dtype=jnp.float32),
    )
    return (
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        plan,
    )


def _toy_geometry_with_inactive_optional_terms():
    receptor_coords, poses_coords, receptor_radii, ligand_radii, plan = _toy_geometry()
    inactive_metal = CertifiedMetalCoordinationSpec(
        receptor_strengths=jnp.zeros_like(plan.metal_coordination.receptor_strengths),
        ligand_strengths=plan.metal_coordination.ligand_strengths,
        receptor_ideal_angles=plan.metal_coordination.receptor_ideal_angles,
        ideal_distance=plan.metal_coordination.ideal_distance,
        distance_width=plan.metal_coordination.distance_width,
        angle_width=plan.metal_coordination.angle_width,
        cutoff=11.0,
    )
    inactive_plan = CertifiedRichChemistryPlan(
        screened_coulomb=plan.screened_coulomb,
        contact=plan.contact,
        hbond_receptor_donor=_make_hbond_spec(
            receptor_alignment_sign=1.0,
            ligand_alignment_sign=-1.0,
            receptor_strengths=(0.0, 0.0),
            ligand_strengths=(0.9, 0.7),
        ),
        hbond_ligand_donor=_make_hbond_spec(
            receptor_alignment_sign=-1.0,
            ligand_alignment_sign=1.0,
            receptor_strengths=(0.0, 0.0),
            ligand_strengths=(0.6, 0.5),
        ),
        metal_coordination=inactive_metal,
        pairwise_sigma=plan.pairwise_sigma,
    )
    return receptor_coords, poses_coords, receptor_radii, ligand_radii, inactive_plan


def test_merge_overlapping_ring_sites_merges_fused_ring_systems() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    sites = (
        _ring_site(
            role=ChemistrySiteRole.AROMATIC_RING,
            coords=coords,
            atom_indices=(0, 1, 2, 3, 4),
        ),
        _ring_site(
            role=ChemistrySiteRole.AROMATIC_RING,
            coords=coords,
            atom_indices=(2, 3, 5, 6, 7),
        ),
    )
    sites = _merge_overlapping_ring_sites(sites, coords=coords, min_shared_atoms=2)
    merged_site = cast(RingSiteAnnotation, sites[0])

    assert len(sites) == 1
    assert set(merged_site.atom_indices) == set(range(8))


def test_aromatic_face_offset_width_is_geometry_derived() -> None:
    expected = 1.39 * np.sqrt(3.0) / 2.0
    assert AROMATIC_FACE_OFFSET_WIDTH_ANGSTROM == pytest.approx(expected)

    pi_stacking = PiStackingInteractionTerm(
        receptor_rings=AnchoredSiteArray.empty(SiteGeometry.RING),
        ligand_rings=IndexedSiteArray.empty(SiteGeometry.RING),
    )
    pi_cation = PiCationInteractionTerm(
        receptor_rings=AnchoredSiteArray.empty(SiteGeometry.RING),
        receptor_cations=AnchoredSiteArray.empty(SiteGeometry.POINT),
        ligand_rings=IndexedSiteArray.empty(SiteGeometry.RING),
        ligand_cations=IndexedSiteArray.empty(SiteGeometry.POINT),
    )

    assert pi_stacking.offset_width == pytest.approx(expected)
    assert pi_cation.offset_width == pytest.approx(expected)


def test_receptor_monocations_use_unit_strength() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.5, 0.5, 0.0],
            [5.0, 0.0, 0.0],
            [4.5, -0.5, 0.0],
        ],
        dtype=np.float32,
    )
    lys = ResidueKey(resname="LYS", chain_id="A", residue_id="1")
    arg = ResidueKey(resname="ARG", chain_id="A", residue_id="2")
    records = (
        (0, PDBAtomRecord("", "ATOM", "NZ", lys, "N", coords[0])),
        (1, PDBAtomRecord("", "ATOM", "NE", arg, "N", coords[1])),
        (2, PDBAtomRecord("", "ATOM", "CZ", arg, "C", coords[2])),
        (3, PDBAtomRecord("", "ATOM", "NH1", arg, "N", coords[3])),
        (4, PDBAtomRecord("", "ATOM", "NH2", arg, "N", coords[4])),
    )
    structure = ReceptorStructureModel(
        coords=coords,
        elements=("N", "N", "C", "N", "N"),
        adjacency=((), (), (), (), ()),
        hydrogen_directions=((), (), (), (), ()),
        indexed_records=records,
        residue_records={
            lys: (records[0],),
            arg: (records[1], records[2], records[3], records[4]),
        },
    )

    sites = ReceptorCationExtractor().extract(structure)

    assert len(sites) == 2
    assert [site.strength for site in sites] == [
        pytest.approx(MONOCATION_SITE_STRENGTH),
        pytest.approx(MONOCATION_SITE_STRENGTH),
    ]


def test_receptor_hbond_donor_extractor_rejects_impossible_backbone_oxygen_donors() -> (
    None
):
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    val = ResidueKey(resname="VAL", chain_id="A", residue_id="29")
    lys = ResidueKey(resname="LYS", chain_id="A", residue_id="33")
    records = (
        (0, PDBAtomRecord("", "ATOM", "O", val, "O", coords[0])),
        (1, PDBAtomRecord("", "ATOM", "NZ", lys, "N", coords[1])),
    )
    structure = ReceptorStructureModel(
        coords=coords,
        elements=("O", "N"),
        adjacency=((1,), (0,)),
        hydrogen_directions=(
            (np.array([0.0, 0.0, 1.0], dtype=np.float32),),
            (np.array([1.0, 0.0, 0.0], dtype=np.float32),),
        ),
        indexed_records=records,
        residue_records={
            val: (records[0],),
            lys: (records[1],),
        },
    )

    sites = ReceptorHBondDonorExtractor().extract(structure)

    assert len(sites) == 1
    assert sites[0].anchor_index == 1


def test_receptor_hbond_donor_strength_is_split_across_explicit_hydrogens() -> None:
    coords = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    lys = ResidueKey(resname="LYS", chain_id="A", residue_id="33")
    record = (0, PDBAtomRecord("", "ATOM", "NZ", lys, "N", coords[0]))
    structure = ReceptorStructureModel(
        coords=coords,
        elements=("N",),
        adjacency=((),),
        hydrogen_directions=(
            (
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
            ),
        ),
        indexed_records=(record,),
        residue_records={lys: (record,)},
    )

    sites = ReceptorHBondDonorExtractor().extract(structure)

    assert len(sites) == 3
    assert [site.strength for site in sites] == [
        pytest.approx(1.0 / 3.0),
        pytest.approx(1.0 / 3.0),
        pytest.approx(1.0 / 3.0),
    ]


def test_ligand_hbond_donor_strength_is_split_across_explicit_hydrogens() -> None:
    coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    structure = LigandStructureModel(
        coords=coords,
        elements=("N",),
        charges=np.array([0.0], dtype=np.float32),
        adjacency=((),),
        hydrogen_directions=(
            (
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            ),
        ),
    )

    sites = LigandHBondDonorExtractor().extract(structure)

    assert len(sites) == 2
    assert [site.strength for site in sites] == [
        pytest.approx(0.5),
        pytest.approx(0.5),
    ]


def test_new_chemistry_handle_helpers_surface_new_theorem_families() -> None:
    assert set(contact_surrogate_theorem_handles()) == {
        "CT1",
        "CT2",
        "CT3",
        "CT4",
        "CT5",
        "CT6",
        "GD7",
    }
    assert {"SC1", "SC6"}.issubset(set(screened_coulomb_theorem_handles()))
    assert {"HB1", "HB9", "HB10", "HB11", "HB12", "HB16"}.issubset(
        set(directional_hbond_finite_theorem_handles())
    )
    assert {"AH1", "AH8"}.issubset(set(attractive_directional_hbond_theorem_handles()))
    assert {"PP6", "PP10"}.issubset(set(attractive_pi_stacking_theorem_handles()))
    assert {"PC6", "PC10"}.issubset(set(attractive_pi_cation_theorem_handles()))
    assert {"XB6", "XB10"}.issubset(set(attractive_halogen_bond_theorem_handles()))
    assert {"WB6", "WB10"}.issubset(
        set(attractive_water_mediated_hbond_theorem_handles())
    )
    assert {"XR1", "XR5"}.issubset(set(attractive_extended_chemistry_theorem_handles()))
    assert {"XR6", "XR10"}.issubset(set(extended_rich_chemistry_theorem_handles()))
    assert {"NG1", "NG5"}.issubset(set(negation_invariance_theorem_handles()))
    assert {"AR1", "AR10"}.issubset(set(rich_chemistry_theorem_handles()))
    assert set(conformer_search_theorem_handles()) == {
        "CS1",
        "CS2",
        "CS3",
        "CS4",
        "CS5",
        "CS6",
        "CS7",
        "CS8",
        "CS9",
        "CS10",
        "CS11",
        "CS12",
        "CS13",
        "CS14",
        "CSC43",
        "CSC44",
        "CSC50",
        "GAP2",
    }
    assert set(cross_docking_theorem_handles()) == {
        "XD1",
        "XD2",
        "XD3",
        "XD4",
        "XD5",
        "XD6",
        "XD7",
        "XD8",
        "XD9",
        "XD10",
        "XD11",
        "XD12",
    }
    assert {"TK13", "TK15"}.issubset(set(topk_bridge_theorem_handles()))
    assert {"SH1", "SH6"}.issubset(set(support_expansion_theorem_handles()))
    assert {"LJ18", "LJ19"} == set(softened_lj_shared_base_delta_theorem_handles())
    assert {"BCRC4", "BCRC6", "EWP6"}.issubset(
        set(omitted_channel_bound_theorem_handles())
    )
    assert {
        "BCRC1",
        "BCRC2",
        "CSC47",
        "LSA10",
        "LJ26",
        "LJ27",
        "LJ28",
        "CB19",
    }.issubset(set(pose_specific_improvement_budget_theorem_handles()))
    assert {"BCPO1", "BCRP1", "BCRC3"} == set(joint_pruning_budget_optimality_handles())
    assert {"SB2", "SB7", "BCRP2", "ERC43"}.issubset(
        set(seed_budget_minimality_theorem_handles())
    )
    assert {"BD7", "SD9", "SD10"}.issubset(set(rigid_seed_family_theorem_handles()))
    assert {
        "CSC61",
        "CSC64",
        "CSC70",
        "CSC72",
        "CSC74",
        "CSC76",
        "CSC78",
        "CSC79",
        "CSC80",
        "CSC81",
        "CSC82",
        "CSC84",
        "CSC85",
        "CSC86",
        "CSC87",
        "CSC88",
        "CSC89",
        "CSC90",
        "CSC91",
        "CSC92",
        "CSC93",
        "CSC94",
        "CSC95",
        "CSC96",
        "CSC97",
    }.issubset(set(rigid_seed_runtime_bridge_theorem_handles()))
    assert {
        "CSC67",
        "CSC69",
        "CSC71",
        "CSC73",
        "CSC75",
        "CSC77",
        "CSC83",
        "CSC96",
        "CSC97",
    }.issubset(set(rigid_seed_family_theorem_handles()))
    assert {"CSC25", "CSC26", "CSC44", "CSC50", "CSC51"}.issubset(
        set(conformer_coverage_theorem_handles())
    )
    assert {"CSC52", "CSC53", "CSC54", "CSC55", "CSC56"}.issubset(
        set(conformer_coverage_theorem_handles())
    )
    assert {"FLO23", "FLO28"}.issubset(set(optimizer_output_contract_theorem_handles()))
    assert {"RPG1", "RPG2", "RPG3", "RPG5"} == set(
        ambiguity_output_contract_theorem_handles()
    )
    assert {"RPG4", "RPG6", "TK16", "ERC39"}.issubset(
        set(returned_pose_guarantee_theorem_handles())
    )


def test_new_chemistry_scoring_helpers_are_jit_safe() -> None:
    (
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        plan,
    ) = _toy_geometry()

    @jax.jit
    def run_all():
        screened_batch = score_certified_screened_coulomb_batch(
            receptor_coords, poses_coords, plan.screened_coulomb
        )
        contact_batch = score_certified_contact_batch(
            receptor_coords, poses_coords, plan.contact
        )
        hbond_receptor_donor_batch = score_certified_directional_hbond_batch(
            receptor_coords, poses_coords, plan.hbond_receptor_donor
        )
        hbond_ligand_donor_batch = score_certified_directional_hbond_batch(
            receptor_coords, poses_coords, plan.hbond_ligand_donor
        )
        metal_batch = score_certified_metal_coordination_batch(
            receptor_coords, poses_coords, plan.metal_coordination
        )
        nonbonded_batch = score_certified_lj_screened_coulomb_batch(
            receptor_coords,
            poses_coords,
            receptor_radii,
            ligand_radii,
            plan.screened_coulomb,
            target_error=0.001,
        )
        rich_batch = score_certified_rich_chemistry_batch(
            receptor_coords=receptor_coords,
            poses_coords=poses_coords,
            receptor_radii=receptor_radii,
            ligand_radii=ligand_radii,
            rich_chemistry_plan=plan,
            target_error=0.001,
        )
        return (
            screened_batch.scores,
            screened_batch.error_bound,
            contact_batch.scores,
            contact_batch.error_bound,
            hbond_receptor_donor_batch.scores,
            hbond_receptor_donor_batch.error_bound,
            hbond_ligand_donor_batch.scores,
            hbond_ligand_donor_batch.error_bound,
            metal_batch.scores,
            metal_batch.error_bound,
            nonbonded_batch.scores,
            nonbonded_batch.error_bound,
            rich_batch.scores,
            rich_batch.error_bound,
        )

    outputs = run_all()
    for value in outputs:
        arr = np.asarray(value)
        assert np.all(np.isfinite(arr))


def test_rich_chemistry_pruning_delta_budget_tracks_named_components() -> None:
    _, _, _, _, plan = _toy_geometry()

    delta_budget = plan.pruning_delta_budget(has_water_bridge_channel=True)

    assert delta_budget.shared_base_delta == pytest.approx(0.0)
    assert delta_budget.cutoff_tail_delta > 0.0
    assert delta_budget.omitted_value_delta >= 2.0
    assert delta_budget.total_delta == pytest.approx(
        delta_budget.cutoff_tail_delta + delta_budget.omitted_value_delta
    )
    assert {component.label for component in delta_budget.active_components} == {
        "cutoff_tail",
        "water_mediated",
    }


def test_rich_chemistry_pruning_delta_budget_omits_inactive_channels() -> None:
    _, _, _, _, plan = _toy_geometry()

    delta_budget = plan.pruning_delta_budget(has_water_bridge_channel=False)

    assert delta_budget.omitted_value_delta == pytest.approx(0.0)
    assert {component.label for component in delta_budget.active_components} == {
        "cutoff_tail",
    }


def test_attractive_chemistry_is_attractive_negative_energy() -> None:
    (
        receptor_coords,
        poses_coords,
        _receptor_radii,
        _ligand_radii,
        plan,
    ) = _toy_geometry()

    contact_batch = score_certified_contact_batch(
        receptor_coords, poses_coords, plan.contact
    )
    hbond_receptor_donor_batch = score_certified_directional_hbond_batch(
        receptor_coords, poses_coords, plan.hbond_receptor_donor
    )
    hbond_ligand_donor_batch = score_certified_directional_hbond_batch(
        receptor_coords, poses_coords, plan.hbond_ligand_donor
    )
    metal_batch = score_certified_metal_coordination_batch(
        receptor_coords, poses_coords, plan.metal_coordination
    )
    attractive_batch = score_certified_attractive_chemistry_batch(
        receptor_coords, poses_coords, plan
    )

    np.testing.assert_allclose(
        np.asarray(attractive_batch.scores),
        -(
            np.asarray(contact_batch.scores)
            + np.asarray(hbond_receptor_donor_batch.scores)
            + np.asarray(hbond_ligand_donor_batch.scores)
        )
        + np.asarray(metal_batch.scores),
        atol=1e-5,
    )
    assert np.isclose(
        float(attractive_batch.error_bound),
        float(
            contact_batch.error_bound
            + hbond_receptor_donor_batch.error_bound
            + hbond_ligand_donor_batch.error_bound
            + metal_batch.error_bound
        ),
    )


def test_rich_chemistry_composes_exact_nonbonded_with_attractive_chemistry() -> None:
    (
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        plan,
    ) = _toy_geometry()

    nonbonded_batch = score_certified_lj_screened_coulomb_batch(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        plan.screened_coulomb,
        target_error=0.001,
    )
    attractive_batch = score_certified_attractive_chemistry_batch(
        receptor_coords, poses_coords, plan
    )
    rich_batch = score_certified_rich_chemistry_batch(
        receptor_coords=receptor_coords,
        poses_coords=poses_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        rich_chemistry_plan=plan,
        target_error=0.001,
    )

    np.testing.assert_allclose(
        np.asarray(rich_batch.scores),
        np.asarray(nonbonded_batch.scores) + np.asarray(attractive_batch.scores),
    )
    np.testing.assert_allclose(
        np.asarray(rich_batch.posewise_error_bound),
        np.asarray(nonbonded_batch.posewise_error_bound)
        + np.asarray(attractive_batch.posewise_error_bound),
    )
    assert np.isclose(
        float(rich_batch.error_bound),
        float(np.max(np.asarray(rich_batch.posewise_error_bound))),
    )


def test_softened_rich_chemistry_composes_softened_nonbonded_with_attractive_chemistry() -> (
    None
):
    (
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        plan,
    ) = _toy_geometry()

    softened_lj = scoring.score_certified_softened_lj(
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        target_error=0.001,
        pairwise_sigma=plan.pairwise_sigma,
        softening_radius=None,
    )
    screened_batch = score_certified_screened_coulomb_batch(
        receptor_coords,
        poses_coords,
        plan.screened_coulomb,
    )
    attractive_batch = score_certified_attractive_chemistry_batch(
        receptor_coords, poses_coords, plan
    )
    softened_rich_batch = score_certified_softened_rich_chemistry_batch(
        receptor_coords=receptor_coords,
        poses_coords=poses_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        rich_chemistry_plan=plan,
        target_error=0.001,
    )

    np.testing.assert_allclose(
        np.asarray(softened_rich_batch.scores),
        np.asarray(softened_lj.scores)
        + np.asarray(screened_batch.scores)
        + np.asarray(attractive_batch.scores),
    )


def test_scoring_context_exact_batch_preserves_posewise_error_bounds_without_flex() -> (
    None
):
    (
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        plan,
    ) = _toy_geometry()

    context = CertifiedScoringContext(
        exact_chemistry_mode=ExactChemistryMode.EXTENDED_RICH,
        rich_chemistry_plan=plan,
    )
    batch = context.score_exact_batch(
        receptor_coords=receptor_coords,
        poses_coords=poses_coords,
        receptor_radii=receptor_radii,
        ligand_radii=ligand_radii,
        target_error=0.001,
        epsilon=0.2,
    )

    assert batch.posewise_error_bound is not None
    assert np.isclose(
        float(batch.error_bound),
        float(np.max(np.asarray(batch.posewise_error_bound))),
    )


def test_inactive_directional_hbond_short_circuits_without_kernel_calls(
    monkeypatch,
) -> None:
    receptor_coords, poses_coords, _receptor_radii, _ligand_radii, plan = (
        _toy_geometry_with_inactive_optional_terms()
    )

    assert not bool(plan.hbond_receptor_donor.is_active)
    assert not bool(plan.hbond_ligand_donor.is_active)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("inactive directional hbond spec should short-circuit")

    monkeypatch.setattr(
        scoring, "_score_directional_hbond_exact_batch", _should_not_run
    )
    monkeypatch.setattr(
        scoring, "_score_directional_hbond_cutoff_batch", _should_not_run
    )

    batch = score_certified_directional_hbond_batch(
        receptor_coords, poses_coords, plan.hbond_receptor_donor
    )

    np.testing.assert_allclose(
        np.asarray(batch.scores), np.zeros(poses_coords.shape[0])
    )
    assert float(batch.error_bound) == 0.0
    assert float(batch.target_error) == 0.0
    assert float(batch.cutoff_radius) == 0.0


def test_inactive_rich_chemistry_terms_are_jit_safe_and_structurally_absent() -> None:
    receptor_coords, poses_coords, _receptor_radii, _ligand_radii, plan = (
        _toy_geometry_with_inactive_optional_terms()
    )
    contact_batch = score_certified_contact_batch(
        receptor_coords, poses_coords, plan.contact
    )

    assert not bool(plan.has_active_directional_hbond)
    assert not bool(plan.has_active_metal_coordination)

    @jax.jit
    def run_inactive():
        hbond_receptor_donor_batch = score_certified_directional_hbond_batch(
            receptor_coords, poses_coords, plan.hbond_receptor_donor
        )
        hbond_ligand_donor_batch = score_certified_directional_hbond_batch(
            receptor_coords, poses_coords, plan.hbond_ligand_donor
        )
        metal_batch = score_certified_metal_coordination_batch(
            receptor_coords, poses_coords, plan.metal_coordination
        )
        attractive_batch = score_certified_attractive_chemistry_batch(
            receptor_coords, poses_coords, plan
        )
        return (
            hbond_receptor_donor_batch.scores,
            hbond_receptor_donor_batch.error_bound,
            hbond_receptor_donor_batch.cutoff_radius,
            hbond_ligand_donor_batch.scores,
            hbond_ligand_donor_batch.error_bound,
            hbond_ligand_donor_batch.cutoff_radius,
            metal_batch.scores,
            metal_batch.error_bound,
            metal_batch.cutoff_radius,
            attractive_batch.scores,
            attractive_batch.error_bound,
            attractive_batch.cutoff_radius,
        )

    (
        hbond_scores,
        hbond_error_bound,
        hbond_cutoff_radius,
        hbond_ligand_scores,
        hbond_ligand_error_bound,
        hbond_ligand_cutoff_radius,
        metal_scores,
        metal_error_bound,
        metal_cutoff_radius,
        attractive_scores,
        attractive_error_bound,
        attractive_cutoff_radius,
    ) = run_inactive()

    np.testing.assert_allclose(
        np.asarray(hbond_scores), np.zeros(poses_coords.shape[0])
    )
    np.testing.assert_allclose(
        np.asarray(hbond_ligand_scores), np.zeros(poses_coords.shape[0])
    )
    np.testing.assert_allclose(
        np.asarray(metal_scores), np.zeros(poses_coords.shape[0])
    )
    np.testing.assert_allclose(
        np.asarray(attractive_scores),
        -np.asarray(contact_batch.scores),
        atol=1e-5,
    )
    assert float(hbond_error_bound) == 0.0
    assert float(hbond_cutoff_radius) == 0.0
    assert float(hbond_ligand_error_bound) == 0.0
    assert float(hbond_ligand_cutoff_radius) == 0.0
    assert float(metal_error_bound) == 0.0
    assert float(metal_cutoff_radius) == 0.0
    assert np.isclose(float(attractive_error_bound), float(contact_batch.error_bound))
    assert np.isclose(
        float(attractive_cutoff_radius), float(contact_batch.cutoff_radius)
    )


def test_directional_hbond_spec_accepts_site_based_geometry() -> None:
    spec = _make_hbond_spec(
        receptor_alignment_sign=1.0,
        ligand_alignment_sign=-1.0,
    )
    spec.validate()

    assert np.asarray(spec.receptor_anchor_indices).shape == (2,)
    assert np.asarray(spec.receptor_directions).shape == (2, 3)
    assert np.asarray(spec.ligand_anchor_indices).shape == (2,)
    assert np.asarray(spec.ligand_local_directions).shape == (2, 3)
    assert np.asarray(spec.ligand_frame_coords).shape == (2, 3)


# =========================================================================
# New theorem family handle tests
# =========================================================================


def test_ligand_strain_handles_exist() -> None:
    handles = ligand_strain_theorem_handles()
    assert len(handles) == 10
    assert handles[0] == "LSA1"


def test_directional_metal_coordination_handles_exist() -> None:
    handles = directional_metal_coordination_theorem_handles()
    assert len(handles) >= 5
    assert handles[0] == "DMC1"
    assert "DMC5" in handles


def test_cooperative_hbond_handles_exist() -> None:
    handles = cooperative_hbond_theorem_handles()
    assert len(handles) == 7
    assert handles[0] == "CHN1"
    assert "CHN5" in handles
    assert "CHN6" in handles
    assert "CHN7" in handles


def test_explicit_water_placement_handles_exist() -> None:
    handles = explicit_water_placement_theorem_handles()
    assert len(handles) == 6
    assert handles[0] == "EWP1"
    assert handles[-1] == "EWP6"


def test_receptor_flexibility_handles_exist() -> None:
    handles = receptor_flexibility_theorem_handles()
    assert len(handles) == 6
    assert handles[0] == "RFE1"


def test_metal_coordination_batch_proof_metadata_uses_directional_handles() -> None:
    assert (
        get_status(scoring.score_certified_metal_coordination_batch)
        == ProofStatus.CONDITIONALLY_CERTIFIED
    )
    theorem = get_theorem(scoring.score_certified_metal_coordination_batch)
    assert "HandleAliases.lean::DMC1" in theorem
    assert (
        "Summary.lean::attractive_metal_coordination_cutoff_certified_top1"
        not in theorem
    )


def test_score_certified_batch_proof_metadata_covers_base_dispatch() -> None:
    assert (
        get_status(scoring.score_certified_batch) == ProofStatus.CONDITIONALLY_CERTIFIED
    )
    theorem = get_theorem(scoring.score_certified_batch)
    assert "LatticeSum.lean::lj6_tail_bound" in theorem
    assert "HandleAliases.lean::CB10" in theorem


def test_scoring_context_exact_batch_proof_metadata_mentions_water_and_flex() -> None:
    assert (
        get_status(CertifiedScoringContext.score_exact_batch)
        == ProofStatus.CONDITIONALLY_CERTIFIED
    )
    theorem = get_theorem(CertifiedScoringContext.score_exact_batch)
    assert "HandleAliases.lean::EWP3" in theorem
    assert "HandleAliases.lean::RFE2" in theorem


def test_scoring_context_softened_batch_proof_metadata_mentions_rich_chain() -> None:
    assert (
        get_status(CertifiedScoringContext.score_softened_batch)
        == ProofStatus.CONDITIONALLY_CERTIFIED
    )
    theorem = get_theorem(CertifiedScoringContext.score_softened_batch)
    assert "HandleAliases.lean::XR6" in theorem
    assert "HandleAliases.lean::EWP3" in theorem


# =========================================================================
# Cosine torsion strain (LSA)
# =========================================================================


def test_cosine_torsion_strain_at_equilibrium() -> None:
    """LSA6: strain vanishes at equilibrium (n·φ = φ₀)."""
    strain = cosine_torsion_strain(
        barrier_height=2.0, multiplicity=1.0, angle=0.5, phase=0.5
    )
    assert np.isclose(strain, 0.0, atol=1e-6)


def test_cosine_torsion_strain_bounded() -> None:
    """LSA1: strain ∈ [0, 2·Vk]."""
    for Vk in [0.5, 1.0, 3.0]:
        for angle in np.linspace(-np.pi, np.pi, 20):
            strain = cosine_torsion_strain(
                barrier_height=Vk, multiplicity=2.0, angle=float(angle), phase=0.0
            )
            assert strain >= -1e-10
            assert strain <= 2.0 * Vk + 1e-10


def test_total_torsion_strain_additive() -> None:
    """LSA5: sum of strains bounded by sum of bounds."""
    barriers = jnp.array([1.0, 2.0, 0.5])
    mults = jnp.array([2.0, 3.0, 1.0])
    angles = jnp.array([1.0, 0.5, -0.3])
    phases = jnp.array([0.0, 0.0, 0.0])
    total = total_torsion_strain(barriers, mults, angles, phases)
    bound = total_torsion_strain_bound(barriers)
    assert total >= -1e-10
    assert total <= bound + 1e-10


# =========================================================================
# Directional metal coordination (DMC)
# =========================================================================


def test_directional_metal_exact_vs_cutoff() -> None:
    """DMC2/DMC3: cutoff variant agrees within cutoff, zero outside."""
    rec_coords = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    lig_coords = jnp.array([[[2.1, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    rec_str = jnp.array([1.0, 0.5])
    lig_str = jnp.array([0.8, 0.6])
    ideal_angles = jnp.array([109.5, 109.5])

    exact = _score_directional_metal_exact_batch(
        rec_coords, lig_coords, rec_str, lig_str, 2.1, 0.3, ideal_angles, 0.5
    )
    cutoff = _score_directional_metal_cutoff_batch(
        rec_coords, lig_coords, rec_str, lig_str, 2.1, 0.3, ideal_angles, 0.5, 4.0
    )
    # Cutoff should differ by at most the tail truncation
    assert exact.shape == cutoff.shape == (1,)
    # With large cutoff, they should be close
    assert np.allclose(exact, cutoff, atol=0.1)


def test_directional_metal_angular_tightens_tail() -> None:
    """DMC2: angular factor ∈ [0,1] can only reduce the score magnitude."""
    rec_coords = jnp.array([[0.0, 0.0, 0.0]])
    lig_coords = jnp.array([[[2.1, 0.0, 0.0]]])
    rec_str = jnp.array([1.0])
    lig_str = jnp.array([1.0])
    ideal_angles = jnp.array([109.5])

    # Wide angle width ≈ radial only
    wide = _score_directional_metal_exact_batch(
        rec_coords, lig_coords, rec_str, lig_str, 2.1, 0.3, ideal_angles, 100.0
    )
    # Narrow angle width — should be tighter (magnitude ≤ wide)
    narrow = _score_directional_metal_exact_batch(
        rec_coords, lig_coords, rec_str, lig_str, 2.1, 0.3, ideal_angles, 0.3
    )
    assert abs(float(narrow[0])) <= abs(float(wide[0])) + 1e-6


# =========================================================================
# Cooperative H-bond correction (CHN)
# =========================================================================


def test_cooperative_correction_bound() -> None:
    """CHN5: |α · (Σf)^2| ≤ |α| · (N·B)^2."""
    bound = cooperative_hbond_correction_bound(alpha=0.2, n_hbonds=5)
    assert np.isclose(bound, 0.2 * 25)


def test_cooperative_correction_bound_with_channel_abs_bound() -> None:
    bound = cooperative_hbond_correction_bound(
        alpha=0.2,
        n_hbonds=2,
        per_channel_abs_bound=3.0,
    )
    assert np.isclose(bound, 0.2 * (2 * 3.0) ** 2)


def test_cooperative_correction_bound_sum_bounds() -> None:
    bound = cooperative_hbond_correction_bound_sum_bounds(
        alpha=0.2,
        per_channel_abs_bounds=(3.0, 1.5),
    )
    assert np.isclose(bound, 0.2 * (3.0 + 1.5) ** 2)


def test_attractive_chemistry_batch_accepts_tighter_support_specific_cooperative_bounds() -> (
    None
):
    (
        receptor_coords,
        poses_coords,
        receptor_radii,
        ligand_radii,
        plan,
    ) = _toy_geometry()
    plan = CertifiedRichChemistryPlan(
        screened_coulomb=plan.screened_coulomb,
        contact=plan.contact,
        hbond_receptor_donor=plan.hbond_receptor_donor,
        hbond_ligand_donor=plan.hbond_ligand_donor,
        metal_coordination=plan.metal_coordination,
        pairwise_sigma=plan.pairwise_sigma,
        cooperative_alpha=0.2,
    )

    default_batch = score_certified_attractive_chemistry_batch(
        receptor_coords=receptor_coords,
        poses_coords=poses_coords,
        rich_chemistry_plan=plan,
    )
    tightened_batch = score_certified_attractive_chemistry_batch(
        receptor_coords=receptor_coords,
        poses_coords=poses_coords,
        rich_chemistry_plan=plan,
        cooperative_channel_abs_bounds=(0.3, 0.1),
    )

    np.testing.assert_allclose(
        np.asarray(default_batch.scores),
        np.asarray(tightened_batch.scores),
        atol=1e-6,
    )
    assert float(tightened_batch.error_bound) < float(default_batch.error_bound)


def test_cooperative_correction_batch_bounded() -> None:
    """CHN1: actual correction bounded by |α|·N² when scores in [0,1]."""
    alpha = 0.2
    n_hbonds = 4
    hbond_scores = jnp.array(
        [[0.8, 0.6, 0.9, 0.3], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
    )
    correction = _cooperative_hbond_correction_batch(hbond_scores, alpha)
    bound = cooperative_hbond_correction_bound(alpha, n_hbonds)
    assert correction.shape == (3,)
    # All corrections must be within bound
    assert jnp.all(jnp.abs(correction) <= bound + 1e-6)
    # Zero scores give zero correction
    assert np.isclose(float(correction[2]), 0.0, atol=1e-8)
    # All-ones gives α · N² (maximum)
    assert np.isclose(float(correction[1]), alpha * n_hbonds**2, atol=1e-6)
