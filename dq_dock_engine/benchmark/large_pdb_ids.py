"""
Curated benchmark metadata for certified docking.

This file keeps a deliberately small, audited core set of non-covalent ligand
targets for the current benchmark runner, plus explicitly excluded ligand/PPI
candidates that are either out of scope, mislabeled, redundant, or unsuitable
for the current non-covalent protocol.
"""

from dataclasses import dataclass
from enum import Enum


class BenchmarkScope(Enum):
    LIGAND = "ligand"
    PPI = "ppi"


@dataclass(frozen=True)
class PDBEntry:
    pdb_id: str
    target_name: str
    resolution: float | None = None
    category: str = "other"
    scope: BenchmarkScope = BenchmarkScope.LIGAND
    include_by_default: bool = True
    preferred_resname: str | None = None
    exclusion_reason: str | None = None


BENCHMARK_PDB_IDS = [
    # Curated default ligand benchmark: valid, non-trivial, and diverse
    PDBEntry("1hvr", "HIV-1 Protease", 1.8, category="protease"),
    PDBEntry(
        "1fkf", "FKBP12-FK506", 1.7, category="immunophilin", preferred_resname="FK5"
    ),
    PDBEntry(
        "2rh1",
        "beta2-Adrenergic Receptor",
        2.4,
        category="gpcr",
        preferred_resname="CAU",
    ),
    PDBEntry("2xp9", "PIN1", 1.9, category="isomerase", preferred_resname="4G8"),
    PDBEntry("3p5o", "BRD4 Bromodomain 1", 1.6, category="epigenetic_reader"),
    PDBEntry("6w63", "SARS-CoV-2 Main Protease", 2.1, category="viral_protease"),
    PDBEntry("1jvp", "CDK2", 1.53, category="kinase"),
    PDBEntry("2hyy", "ABL1 Kinase", 2.4, category="kinase"),
    PDBEntry(
        "1c5w", "Urokinase-type Plasminogen Activator", 1.94, category="serine_protease"
    ),
    PDBEntry("4d83", "BACE-1", 2.4, category="aspartyl_protease"),
    PDBEntry("1l2i", "Estrogen Receptor alpha", 1.95, category="nuclear_receptor"),
    PDBEntry("2p54", "PPAR alpha", 1.79, category="nuclear_receptor"),
    PDBEntry("1qs4", "HIV-1 Integrase", 2.1, category="metalloenzyme"),
    PDBEntry("3k99", "HSP90 N-terminal Domain", 2.1, category="chaperone"),
    PDBEntry("3g6z", "Renin", 2.0, category="aspartyl_protease"),
    # Valid alternates kept out of the default run to reduce family redundancy
    PDBEntry(
        "1ajx",
        "HIV-1 Protease",
        2.0,
        category="protease",
        include_by_default=False,
        exclusion_reason="redundant with curated HIV protease baseline",
    ),
    PDBEntry(
        "1dmp",
        "HIV-1 Protease",
        2.0,
        category="protease",
        include_by_default=False,
        exclusion_reason="redundant with curated HIV protease baseline",
    ),
    PDBEntry(
        "1pro",
        "HIV-1 Protease",
        1.8,
        category="protease",
        include_by_default=False,
        exclusion_reason="redundant with curated HIV protease baseline",
    ),
    PDBEntry(
        "2p3b",
        "HIV-1 Protease",
        2.1,
        category="protease",
        include_by_default=False,
        exclusion_reason="valid large-ligand alternate, excluded by default for diversity",
    ),
    PDBEntry(
        "5r84",
        "SARS-CoV-2 Main Protease",
        1.83,
        category="viral_protease",
        include_by_default=False,
        exclusion_reason="valid fragment-sized alternate, excluded by default for family diversity",
    ),
    PDBEntry(
        "4i0e",
        "BACE-1",
        1.7,
        category="aspartyl_protease",
        include_by_default=False,
        exclusion_reason="valid alternate, excluded by default for family diversity",
    ),
    PDBEntry(
        "4i0t",
        "SYK Kinase",
        1.7,
        category="kinase",
        include_by_default=False,
        exclusion_reason="valid alternate, excluded by default for family diversity",
    ),
    # Explicit ligand exclusions discovered during audit
    PDBEntry(
        "4er2",
        "Aspartic Proteinase Peptide Inhibitor Complex",
        2.0,
        category="peptide_inhibitor",
        include_by_default=False,
        exclusion_reason="requested as estrogen receptor, but resolves to a linked peptide-like inhibitor complex",
    ),
    PDBEntry(
        "2r0x",
        "Putative Flavin Reductase",
        1.06,
        category="crystallization_artifact",
        include_by_default=False,
        exclusion_reason="requested as carbonic anhydrase II, but resolves to flavin reductase with additives",
    ),
    PDBEntry(
        "1u19",
        "Bovine Rhodopsin",
        2.2,
        category="covalent_membrane_protein",
        include_by_default=False,
        exclusion_reason="requested as HIV-1 reverse transcriptase, but resolves to rhodopsin with covalently linked retinal",
    ),
    PDBEntry(
        "1ppb",
        "alpha-Thrombin with chloromethylketone inhibitor",
        1.92,
        category="covalent_protease",
        include_by_default=False,
        exclusion_reason="covalent chloromethylketone inhibitor is out of scope for non-covalent docking",
    ),
    PDBEntry(
        "6lu7",
        "SARS-CoV-2 Main Protease",
        2.16,
        category="covalent_protease",
        include_by_default=False,
        exclusion_reason="covalent inhibitor split across linked residues",
    ),
    PDBEntry(
        "7btf",
        "SARS-CoV-2 RdRp",
        2.95,
        category="metal_artifact",
        include_by_default=False,
        exclusion_reason="metal-ion pseudo-ligand (two zinc atoms)",
    ),
    # Protein-protein interface targets reserved for a future PPI runner
    PDBEntry(
        "1ay7",
        "Barnase-Barstar",
        2.0,
        category="ppi",
        scope=BenchmarkScope.PPI,
        include_by_default=False,
        exclusion_reason="protein-protein interface benchmark",
    ),
    PDBEntry(
        "1ppe",
        "Trypsin-BPTI",
        2.0,
        category="ppi",
        scope=BenchmarkScope.PPI,
        include_by_default=False,
        exclusion_reason="protein-protein interface benchmark",
    ),
    PDBEntry(
        "1ewy",
        "Ferredoxin:FNR Complex",
        2.3,
        category="ppi",
        scope=BenchmarkScope.PPI,
        include_by_default=False,
        exclusion_reason="protein-protein interface benchmark",
    ),
    PDBEntry(
        "1vfb",
        "Antibody-Antigen Complex",
        2.3,
        category="ppi",
        scope=BenchmarkScope.PPI,
        include_by_default=False,
        exclusion_reason="protein-protein interface benchmark",
    ),
    PDBEntry(
        "1ml0",
        "Viral Chemokine Binding Protein Complex",
        2.2,
        category="ppi",
        scope=BenchmarkScope.PPI,
        include_by_default=False,
        exclusion_reason="protein-protein interface benchmark",
    ),
]


def get_benchmark_entries() -> list[PDBEntry]:
    return BENCHMARK_PDB_IDS


def get_default_ligand_entries() -> list[PDBEntry]:
    return [
        entry
        for entry in BENCHMARK_PDB_IDS
        if entry.scope == BenchmarkScope.LIGAND and entry.include_by_default
    ]


def get_benchmark_pdb_ids() -> list[str]:
    return [entry.pdb_id for entry in get_default_ligand_entries()]


def get_benchmark_info() -> dict[str, PDBEntry]:
    return {entry.pdb_id: entry for entry in BENCHMARK_PDB_IDS}
