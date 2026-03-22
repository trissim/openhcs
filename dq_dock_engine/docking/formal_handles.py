from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum

from dq_dock_engine.generated.formal_handle_aliases import (
    AH1,
    AH2,
    AH3,
    AH4,
    AH5,
    AH6,
    AH7,
    AH8,
    APX10,
    APX11,
    APX12,
    APX4,
    AR1,
    AR2,
    AR3,
    AR4,
    AR5,
    AR6,
    AR7,
    AR8,
    AR9,
    AR10,
    BD10,
    CT1,
    CT2,
    CT3,
    CT4,
    CT5,
    CT6,
    CB5,
    CB6,
    CB10,
    CB11,
    CB12,
    CB13,
    CB14,
    CP1,
    CP2,
    CP3,
    CP4,
    CP5,
    CP6,
    FLO10,
    FLO11,
    FLO12,
    FLO13,
    FLO14,
    FLO15,
    FLO16,
    FLO17,
    FLO18,
    FLO3,
    FLO4,
    FLO8,
    FLO9,
    HB1,
    HB2,
    HB3,
    HB4,
    HB5,
    HB6,
    HB7,
    HB8,
    HB9,
    HB10,
    HB11,
    HB12,
    LJ13,
    LJ14,
    LJ10,
    LJ11,
    LJ12,
    NB1,
    NB2,
    NB3,
    NB4,
    NB5,
    NB6,
    NB7,
    NB8,
    NB9,
    NB10,
    NG1,
    NG2,
    NG3,
    NG4,
    NG5,
    RC1,
    RC2,
    RC3,
    RC4,
    RC5,
    RC6,
    RC7,
    RC8,
    RC9,
    RC10,
    SC1,
    SC2,
    SC3,
    SC4,
    SC5,
    SC6,
    SH1,
    SH2,
    SH3,
    SH4,
    SH5,
    SH6,
    SD10,
    TK1,
    TK13,
    TK14,
    TK15,
    TK8,
    TK9A,
    TK11,
    TK12,
    TK4,
)
from dq_dock_engine.docking_config import CertifiedScoringFamily


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
    return (CT1, CT2, CT3, CT4, CT5, CT6)


def screened_coulomb_theorem_handles() -> tuple[str, ...]:
    return (SC1, SC2, SC3, SC4, SC5, SC6)


def additive_nonbonded_theorem_handles() -> tuple[str, ...]:
    return (NB1, NB2, NB3, NB4, NB5, NB6, NB7, NB8, NB9, NB10)


def directional_hbond_theorem_handles() -> tuple[str, ...]:
    return (HB1, HB2, HB3, HB4, HB5, HB6, HB7, HB8)


def directional_hbond_finite_theorem_handles() -> tuple[str, ...]:
    return (HB1, HB9, HB10, HB11, HB12)


def attractive_directional_hbond_theorem_handles() -> tuple[str, ...]:
    return (AH1, AH2, AH3, AH4, AH5, AH6, AH7, AH8)


def negation_invariance_theorem_handles() -> tuple[str, ...]:
    return (NG1, NG2, NG3, NG4, NG5)


def rich_chemistry_theorem_handles() -> tuple[str, ...]:
    return (AR1, AR2, AR3, AR4, AR5, AR6, AR7, AR8, AR9, AR10)


def positive_rich_chemistry_theorem_handles() -> tuple[str, ...]:
    return (RC1, RC2, RC3, RC4, RC5, RC6, RC7, RC8, RC9, RC10)


def topk_bridge_theorem_handles() -> tuple[str, ...]:
    return (TK13, TK14, TK15)


def support_expansion_theorem_handles() -> tuple[str, ...]:
    return (SH1, SH2, SH3, SH4, SH5, SH6)


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
