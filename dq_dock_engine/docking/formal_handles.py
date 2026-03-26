from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import re

from dq_dock_engine.generated import formal_handle_aliases as _formal_handle_aliases
from dq_dock_engine.docking_config import CertifiedScoringFamily


for _name in _formal_handle_aliases.__all__:
    globals()[_name] = getattr(_formal_handle_aliases, _name)


_HANDLE_NAME_RE = re.compile(r"^([A-Z_]+?)(\d+|[A-Z]\d+)?$")


def _handle_sort_key(handle: str) -> tuple[str, int, str]:
    match = _HANDLE_NAME_RE.match(handle)
    if match is None:
        return (handle, -1, "")
    prefix, suffix = match.groups()
    if suffix is None:
        return (prefix, -1, "")
    numeric = "".join(ch for ch in suffix if ch.isdigit())
    return (prefix, int(numeric) if numeric else -1, suffix)


def _prefixed_handles(prefix: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                name
                for name in _formal_handle_aliases.__all__
                if name.startswith(prefix)
            ),
            key=_handle_sort_key,
        )
    )


def _selected_handles(*names: str) -> tuple[str, ...]:
    return names


@dataclass(frozen=True)
class FormalHandleBundle:
    theorem_handles: tuple[str, ...]
    witness_handles: tuple[str, ...]


@dataclass(frozen=True)
class CertifiedRuntimeContract:
    name: str
    strategy: "CertifiedRuntimeStrategyName"
    pruning_branch: "Top1PruningBranchName | None"
    pruning_certificate_handle: str
    survivor_set_witness_handle: str
    posterior_theorem_handle: str
    posterior_witness_handle: str
    selection_theorem_handle: str
    selection_witness_handles: tuple[str, ...]
    belief_witness_handles: tuple[str, ...]
    optimizer_witness_handle: str


class SelectionBranchName(str, Enum):
    AMBIGUITY_BAND = "ambiguity_band"
    SUPPORT_FALLBACK = "posterior_support"


class Top1PruningBranchName(str, Enum):
    EXACT_TOP1 = "exact_top1"
    EXACT_SINGLETON_WINNER = "exact_singleton_winner"
    TOP1_COARSE_AMBIGUITY_BAND = "top1_coarse_ambiguity_band"


class CertifiedRuntimeStrategyName(str, Enum):
    EXACT = "exact"
    SINGLETON_HYBRID = "singleton_hybrid"
    STAGED_COARSE_TOP1 = "staged_coarse_top1"
    STAGED_SINGLETON_TOP1 = "staged_singleton_top1"


ACTIVE_EXACT_CERTIFIED_RUNTIME_CONTRACT = CertifiedRuntimeContract(
    name="active_exact_certified_runtime",
    strategy=CertifiedRuntimeStrategyName.EXACT,
    pruning_branch=Top1PruningBranchName.EXACT_TOP1,
    pruning_certificate_handle=CP2,
    survivor_set_witness_handle=CP4,
    posterior_theorem_handle=FLO9,
    posterior_witness_handle=FLO10,
    selection_theorem_handle=FLO8,
    selection_witness_handles=(FLO11, FLO12),
    belief_witness_handles=(FLO13, FLO14),
    optimizer_witness_handle=FLO15,
)


STAGED_COARSE_TOP1_RUNTIME_CONTRACT = CertifiedRuntimeContract(
    name="staged_coarse_top1_runtime",
    strategy=CertifiedRuntimeStrategyName.STAGED_COARSE_TOP1,
    pruning_branch=Top1PruningBranchName.TOP1_COARSE_AMBIGUITY_BAND,
    pruning_certificate_handle=TK11,
    survivor_set_witness_handle=CP5,
    posterior_theorem_handle=FLO9,
    posterior_witness_handle=FLO10,
    selection_theorem_handle=FLO8,
    selection_witness_handles=(FLO11, FLO12),
    belief_witness_handles=(FLO13, FLO14),
    optimizer_witness_handle=FLO16,
)


STAGED_SINGLETON_TOP1_RUNTIME_CONTRACT = CertifiedRuntimeContract(
    name="staged_singleton_top1_runtime",
    strategy=CertifiedRuntimeStrategyName.STAGED_SINGLETON_TOP1,
    pruning_branch=Top1PruningBranchName.EXACT_SINGLETON_WINNER,
    pruning_certificate_handle=TK12,
    survivor_set_witness_handle=CP6,
    posterior_theorem_handle=FLO9,
    posterior_witness_handle=FLO10,
    selection_theorem_handle=FLO8,
    selection_witness_handles=(FLO11, FLO12),
    belief_witness_handles=(FLO13, FLO14),
    optimizer_witness_handle=FLO17,
)


ACTIVE_SINGLETON_HYBRID_RUNTIME_CONTRACT = CertifiedRuntimeContract(
    name="active_singleton_hybrid_certified_runtime",
    strategy=CertifiedRuntimeStrategyName.SINGLETON_HYBRID,
    pruning_branch=None,
    pruning_certificate_handle=CP3,
    survivor_set_witness_handle=CP6,
    posterior_theorem_handle=FLO9,
    posterior_witness_handle=FLO10,
    selection_theorem_handle=FLO8,
    selection_witness_handles=(FLO11, FLO12),
    belief_witness_handles=(FLO13, FLO14),
    optimizer_witness_handle=FLO18,
)


def handle_bundle_from_contracts(
    *contracts: CertifiedRuntimeContract,
    extra_theorem_handles: tuple[str, ...] = (),
    extra_witness_handles: tuple[str, ...] = (),
) -> FormalHandleBundle:
    theorem_handles = []
    witness_handles = []

    for contract in contracts:
        theorem_handles.extend(
            [
                contract.pruning_certificate_handle,
                contract.posterior_theorem_handle,
                contract.selection_theorem_handle,
            ]
        )
        witness_handles.extend(
            [
                contract.survivor_set_witness_handle,
                contract.posterior_witness_handle,
                *contract.selection_witness_handles,
                *contract.belief_witness_handles,
                contract.optimizer_witness_handle,
            ]
        )

    theorem_handles.extend(extra_theorem_handles)
    witness_handles.extend(extra_witness_handles)

    return FormalHandleBundle(
        theorem_handles=tuple(dict.fromkeys(theorem_handles)),
        witness_handles=tuple(dict.fromkeys(witness_handles)),
    )


ACTIVE_CERTIFIED_RUNTIME_HANDLES = handle_bundle_from_contracts(
    ACTIVE_EXACT_CERTIFIED_RUNTIME_CONTRACT,
)


STAGED_COARSE_PRUNING_HANDLES = handle_bundle_from_contracts(
    STAGED_COARSE_TOP1_RUNTIME_CONTRACT,
    STAGED_SINGLETON_TOP1_RUNTIME_CONTRACT,
    extra_theorem_handles=(CP3,),
    extra_witness_handles=(FLO18,),
)


def scoring_family_theorem_handles(
    certified_scoring_family: CertifiedScoringFamily,
) -> tuple[str, ...]:
    if certified_scoring_family == CertifiedScoringFamily.LJ_REALSPACE_EWALD:
        return (
            CB5,
            CB6,
            CB10,
            CB11,
            CB12,
            CB13,
            CB14,
            LJ10,
            LJ11,
            LJ12,
            APX10,
            APX11,
            APX12,
            BD10,
        )
    if certified_scoring_family == CertifiedScoringFamily.LJ:
        return (LJ10, LJ11, LJ12, LJ13, LJ14, APX10, APX11, APX12)
    return ()


def contact_surrogate_theorem_handles() -> tuple[str, ...]:
    return _prefixed_handles("CT")


def screened_coulomb_theorem_handles() -> tuple[str, ...]:
    return _prefixed_handles("SC")


def additive_nonbonded_theorem_handles() -> tuple[str, ...]:
    return _prefixed_handles("NB")


def directional_hbond_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(HB1, HB2, HB3, HB4, HB5, HB6, HB7, HB8, HB13, HB14)


def directional_hbond_finite_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(HB1, HB9, HB10, HB11, HB12)


def attractive_directional_hbond_theorem_handles() -> tuple[str, ...]:
    return _prefixed_handles("AH")


def negation_invariance_theorem_handles() -> tuple[str, ...]:
    return _prefixed_handles("NG")


def rich_chemistry_theorem_handles() -> tuple[str, ...]:
    return _prefixed_handles("AR")


def positive_rich_chemistry_theorem_handles() -> tuple[str, ...]:
    return _prefixed_handles("RC")


def metal_coordination_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(MC1, MC2, MC3, MC4, MC5)


def attractive_metal_coordination_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(MC6, MC7, MC8, MC9, MC10)


def metal_coordination_cutoff_theorem_handles() -> tuple[str, ...]:
    """Physically derived cutoff bounds (MetalCoordinationApproximation.lean).

    MC11: metalCoordination_tail_bound — pointwise Gaussian tail decay
    MC12: metalCoordination_cutoff_sufficient — rc ≥ ideal + width·√(ln(|w|/ε)) ⟹ error ≤ ε
    MC13: metalCoordinationMinCutoff_sufficient — minimum cutoff achieves ε guarantee
    """
    return _selected_handles(MC11, MC12, MC13)


def pi_stacking_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(PP1, PP2, PP3, PP4, PP5)


def attractive_pi_stacking_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(PP6, PP7, PP8, PP9, PP10)


def pi_cation_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(PC1, PC2, PC3, PC4, PC5)


def attractive_pi_cation_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(PC6, PC7, PC8, PC9, PC10)


def halogen_bond_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(XB1, XB2, XB3, XB4, XB5)


def attractive_halogen_bond_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(XB6, XB7, XB8, XB9, XB10)


def water_mediated_hbond_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(WB1, WB2, WB3, WB4, WB5)


def attractive_water_mediated_hbond_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(WB6, WB7, WB8, WB9, WB10)


def attractive_extended_chemistry_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(XR1, XR2, XR3, XR4, XR5)


def extended_rich_chemistry_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(XR6, XR7, XR8, XR9, XR10)


def topk_bridge_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(TK13, TK14, TK15)


def support_expansion_theorem_handles() -> tuple[str, ...]:
    return _prefixed_handles("SH")


def conformer_search_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(
        CS1,
        CS2,
        CS3,
        CS4,
        CS5,
        CS6,
        CS7,
        CS8,
        CS9,
        CS10,
        CS12,
        CS13,
        CS14,
        "CS11",
        "CSC43",
        "CSC44",
        "CSC50",
        "GAP2",
    )


def ligand_strain_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(
        LSA1,
        LSA2,
        LSA3,
        LSA4,
        LSA5,
        LSA6,
        LSA7,
        LSA8,
        LSA9,
        LSA10,
    )


def branch_and_bound_cross_docking_handles() -> tuple[str, ...]:
    return ("XD1", "XD2", "XD3", "XD4")


def strain_augmented_cross_docking_handles() -> tuple[str, ...]:
    return ("XD5", "XD6", "XD7", "XD8")


def pocket_cross_docking_handles() -> tuple[str, ...]:
    return ("XD9", "XD10")


def receptor_flex_cross_docking_handles() -> tuple[str, ...]:
    return ("XD11", "XD12")


def cross_docking_theorem_handles() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            branch_and_bound_cross_docking_handles()
            + strain_augmented_cross_docking_handles()
            + pocket_cross_docking_handles()
            + receptor_flex_cross_docking_handles()
        )
    )


def directional_metal_coordination_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(DMC1, DMC2, DMC3, DMC4, DMC5)


def cooperative_hbond_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(CHN1, CHN2, CHN3, CHN4)


def explicit_water_placement_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(EWP1, EWP2, EWP3, EWP4, EWP5, EWP6)


def softened_lj_shared_base_delta_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(LJ18, LJ19)


def omitted_channel_bound_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(BCRC4, BCRC5, BCRC6, CHN1, CHN2, EWP6)


def pose_specific_improvement_budget_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(
        BCRC1,
        BCRC2,
        CSC47,
        LSA9,
        LSA10,
        LJ20,
        LJ24,
        LJ25,
        LJ26,
        CB5,
        CB6,
    )


def joint_pruning_budget_optimality_handles() -> tuple[str, ...]:
    return _selected_handles(BCPO1, BCRP1, BCRC3)


def seed_budget_minimality_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(SB2, SB5, SB7, BCRP2, ERC43)


def rigid_seed_family_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(BD3, BD4, BD7, BD8, BD9, BD10, SD4, SD5, SD8, SD9, SD10)


def conformer_coverage_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(
        CSC4,
        CSC5,
        CSC25,
        CSC26,
        CSC37,
        CSC43,
        CSC44,
        CSC48,
        CSC50,
        CSC51,
        CSC52,
        CSC53,
        CSC54,
        CSC55,
        CSC56,
    )


def optimizer_output_contract_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(FLO23, FLO24, FLO25, FLO26, FLO27, FLO28)


def ambiguity_output_contract_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(RPG1, RPG2, RPG3, RPG5)


def returned_pose_guarantee_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(RPG4, RPG6, TK16, ERC39, ERC43, DC104)


def ambiguity_output_energy_contract_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(RPG1, RPG7, CSC56)


def returned_pose_energy_guarantee_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(RPG8, CSC56)


def rich_pruning_delta_theorem_handles() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            softened_lj_shared_base_delta_theorem_handles()
            + screened_coulomb_theorem_handles()
            + contact_surrogate_theorem_handles()
            + directional_hbond_finite_theorem_handles()
            + metal_coordination_cutoff_theorem_handles()
            + attractive_extended_chemistry_theorem_handles()
            + omitted_channel_bound_theorem_handles()
        )
    )


def performance_certificate_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(PERF1, PERF2, PERF3, PERF4, PERF5, PERF6)


def receptor_flexibility_theorem_handles() -> tuple[str, ...]:
    return _selected_handles(RFE1, RFE2, RFE3, RFE4, RFE5, RFE6)


def selection_theorem_handle(branch: str) -> str:
    return FLO8


def selection_branch_membership_handle(branch: str) -> str:
    return {
        SelectionBranchName.AMBIGUITY_BAND.value: FLO3,
        SelectionBranchName.SUPPORT_FALLBACK.value: FLO4,
    }[branch]


def posterior_update_theorem_handle() -> str:
    return FLO9


def posterior_update_witness_handle() -> str:
    return FLO10


def selection_witness_handle(branch: str) -> str:
    return {
        SelectionBranchName.AMBIGUITY_BAND.value: FLO11,
        SelectionBranchName.SUPPORT_FALLBACK.value: FLO12,
    }[branch]


def belief_witness_handle(branch: str) -> str:
    return {
        SelectionBranchName.AMBIGUITY_BAND.value: FLO13,
        SelectionBranchName.SUPPORT_FALLBACK.value: FLO14,
    }[branch]


def survivor_set_witness_handle(branch: str) -> str:
    return {
        Top1PruningBranchName.EXACT_TOP1.value: CP4,
        Top1PruningBranchName.TOP1_COARSE_AMBIGUITY_BAND.value: CP5,
        Top1PruningBranchName.EXACT_SINGLETON_WINNER.value: CP6,
    }.get(branch, CP1)


def optimizer_witness_handle(branch: str) -> str:
    return {
        Top1PruningBranchName.EXACT_TOP1.value: FLO15,
        Top1PruningBranchName.TOP1_COARSE_AMBIGUITY_BAND.value: FLO16,
        Top1PruningBranchName.EXACT_SINGLETON_WINNER.value: FLO17,
    }.get(branch, FLO15)


def optimizer_branch_witness_handle() -> str:
    return FLO18


def serialize_dataclass_record(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: serialize_dataclass_record(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return tuple(serialize_dataclass_record(item) for item in value)
    if isinstance(value, list):
        return [serialize_dataclass_record(item) for item in value]
    return value


def runtime_contract_record(contract: CertifiedRuntimeContract) -> dict[str, object]:
    record = serialize_dataclass_record(contract)
    if not isinstance(record, dict):
        raise ValueError("CertifiedRuntimeContract serialization must produce a dict")
    return record
