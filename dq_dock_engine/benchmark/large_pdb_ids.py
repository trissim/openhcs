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


# =============================================================================
# CASF-2007 core set (PDBbind v2007.2)
# 195 entries from 65 clusters. Canonical redocking benchmark.
# Archive: benchmark/CASF-2007.tar.gz (proteins + ligands included)
# =============================================================================

CASF_2007_RAW_IDS: list[str] = [
    "1hk4",
    "1ha2",
    "1gni",
    "1nhu",
    "2d3z",
    "2d3u",
    "1ajp",
    "1ai5",
    "1ajq",
    "1gpk",
    "1h23",
    "1e66",
    "2rkm",
    "1b9j",
    "1b7h",
    "1u2y",
    "1u33",
    "1xd1",
    "1uwt",
    "2ceq",
    "2cer",
    "2qwb",
    "2qwd",
    "2qwe",
    "2j77",
    "2j78",
    "2cet",
    "1m0n",
    "1zc9",
    "1m0q",
    "2fdp",
    "1fkn",
    "2g94",
    "1v16",
    "1olu",
    "1ols",
    "1tok",
    "1toj",
    "1toi",
    "1n2v",
    "1k4g",
    "1s39",
    "1kv1",
    "2bak",
    "2baj",
    "1ndw",
    "1ndy",
    "1ndz",
    "2hdq",
    "1l2s",
    "1xgj",
    "1q8t",
    "1ydt",
    "1re8",
    "1g7q",
    "1fzj",
    "1fzk",
    "1m2q",
    "1om1",
    "1zoe",
    "2azr",
    "1g7f",
    "1nny",
    "5er1",
    "2er9",
    "4er2",
    "1ppm",
    "1apw",
    "1bxo",
    "1nje",
    "1tsy",
    "1nja",
    "4tln",
    "1tmn",
    "4tmn",
    "1fh7",
    "1fh9",
    "1fh8",
    "2fzc",
    "2h3e",
    "1d09",
    "2ctc",
    "8cpa",
    "7cpa",
    "1e1v",
    "1b39",
    "1pxo",
    "1bcu",
    "1vzq",
    "1sl3",
    "2brb",
    "2brm",
    "1nvq",
    "1jqd",
    "2aov",
    "2aou",
    "1y1m",
    "1pb9",
    "1pbq",
    "1vfn",
    "1v48",
    "1b8o",
    "1p1q",
    "1syh",
    "1ftm",
    "1fcx",
    "1fd0",
    "1fcz",
    "1f4e",
    "1f4f",
    "1f4g",
    "1f5k",
    "1o3p",
    "1sqa",
    "2b1v",
    "2fai",
    "2ayr",
    "1avn",
    "1ttm",
    "1if7",
    "2bok",
    "1nfy",
    "1mq6",
    "2usn",
    "2d1o",
    "1hfs",
    "2flb",
    "2bz6",
    "2b7d",
    "1loq",
    "1lol",
    "1x1z",
    "4tim",
    "1kv5",
    "1trd",
    "1bra",
    "1j16",
    "1j17",
    "1utp",
    "1v2o",
    "1o3f",
    "1jys",
    "1nc1",
    "1y6q",
    "1bma",
    "1ela",
    "1elb",
    "1pr5",
    "1a69",
    "1k9s",
    "3pce",
    "3pch",
    "3pcj",
    "1pz5",
    "2cgr",
    "1flr",
    "2gss",
    "3gss",
    "10gs",
    "6std",
    "2std",
    "3std",
    "1jaq",
    "1zs0",
    "1zvx",
    "2d0k",
    "1dhi",
    "2drc",
    "1slg",
    "1df8",
    "2f01",
    "2g8r",
    "1o0h",
    "1u1b",
    "2c02",
    "1hi4",
    "2bzz",
    "1tyr",
    "1e5a",
    "2g5u",
    "1sv3",
    "1q7a",
    "1jq9",
    "1a08",
    "1a1b",
    "1is0",
    "1a30",
    "2f80",
    "2i0d",
    "1d7j",
    "1fki",
    "1fkb",
    "6rnt",
    "1det",
    "1rnt",
]


def get_casf_2007_ids() -> list[str]:
    """Return the 195-entry CASF-2007 core set from PDBbind v2007.2.

    Canonical redocking benchmark. Pre-prepared proteins and MOL2 ligands
    are included in benchmark/CASF-2007.tar.gz.
    """
    return CASF_2007_RAW_IDS


def get_casf_2007_qc_passed_ids() -> list[str]:
    """Return CASF-2007 entries that pass OpenHCS QC gates.

    Gates: resolution < 2.5A, single MODEL, HAC >= 10, no metals,
    no covalent contacts (< 1.55A), ligand detected in structure.

    Caches results in /tmp/casf2007_pdb_cache after first run.

    Raises:
        RuntimeError: if screening fails for any PDB ID.
    """
    import importlib
    from pathlib import Path

    try:
        benchmark_pdb = importlib.import_module(
            "dq_dock_engine.benchmark.benchmark_pdb"
        )
        screen_complex = benchmark_pdb.screen_complex
        prepare_protein = benchmark_pdb.prepare_protein
        BenchmarkScope = benchmark_pdb.BenchmarkScope
        PDBEntry = benchmark_pdb.PDBEntry
    except ImportError as exc:
        raise RuntimeError("benchmark_pdb module unavailable") from exc

    cache = Path("/tmp/casf2007_pdb_cache")
    if not cache.exists():
        raise RuntimeError(
            f"PDB cache not found at {cache}. "
            "Download PDBs first with the benchmark runner."
        )

    passed = []
    errors = []
    for pdb_id in CASF_2007_RAW_IDS:
        pdb_path = cache / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            errors.append(f"{pdb_id}: PDB not cached")
            continue
        entry = PDBEntry(
            pdb_id=pdb_id, target_name=f"CASF-{pdb_id}", scope=BenchmarkScope.LIGAND
        )
        protein_path = prepare_protein(pdb_path)
        decision = screen_complex(entry, pdb_path, protein_path)
        if decision.accepted:
            passed.append(pdb_id)

    if errors:
        raise RuntimeError(f"QC screening failed for {len(errors)} entries: {errors}")

    return passed
