"""
Curated benchmark metadata for certified docking.

This file keeps a deliberately small, audited core set of non-covalent ligand
targets for the current benchmark runner, plus explicitly excluded ligand/PPI
candidates that are either out of scope, mislabeled, redundant, or unsuitable
for the current non-covalent protocol.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


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


def _casf_data_dir() -> Path:
    return Path(__file__).parent


def get_casf_2007_ids() -> list[str]:
    """Return CASF-2007 PDB IDs, one per line from CASF_2007_ids.lst."""
    ids_path = _casf_data_dir() / "CASF_2007_ids.lst"
    if not ids_path.exists():
        raise FileNotFoundError(
            f"CASF-2007 ID list not found: {ids_path}. "
            "Extract from CASF-2007.tar.gz using extract_casf2007_ids.py"
        )
    with open(ids_path) as f:
        return [line.strip() for line in f if line.strip()]


def _load_casf_resolutions() -> dict[str, float]:
    """Lazily load CASF-2007 resolutions from JSON."""
    res_path = _casf_data_dir() / "CASF_2007_resolutions.json"
    if not res_path.exists():
        raise FileNotFoundError(
            f"CASF-2007 resolution file not found: {res_path}. "
            "Generate it from CASF-2007.tar.gz first."
        )
    import json

    with open(res_path) as f:
        raw = json.load(f)
    return {k: float(v) for k, v in raw.items()}


def _extract_pdb_title(pdb_path: Path, max_length: int = 60) -> str:
    """Extract protein title from PDB file TITLE lines.

    Reads consecutive TITLE records from the PDB file, concatenates them,
    and truncates to max_length without cutting words in half.
    """
    if not pdb_path.exists():
        return pdb_path.stem

    title_parts: list[str] = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("TITLE"):
                title_parts.append(line[10:].strip())
            elif title_parts and not line.startswith("TITLE"):
                break

    if not title_parts:
        return pdb_path.stem

    full_title = " ".join(title_parts)

    if len(full_title) <= max_length:
        return full_title

    truncated = full_title[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.5:
        truncated = truncated[:last_space]

    return truncated.strip()


def get_casf_2007_entries() -> list[PDBEntry]:
    """Return CASF-2007 as PDBEntry objects for the benchmark runner.

    Uses QC-passed IDs. Resolutions come from CASF_2007_resolutions.json.
    Cache: /tmp/casf2007_pdb_cache

    Raises:
        RuntimeError: if QC screening fails or cache is missing.
    """
    import importlib

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

    # Use local repo cache dir to match run_benchmark behavior (./casf2007_pdb_cache)
    cache = Path("./casf2007_pdb_cache")
    if not cache.exists():
        raise RuntimeError(
            f"PDB cache not found at {cache}. "
            "Run with --dataset casf2007 to download and populate."
        )

    pdb_ids = get_casf_2007_ids()
    resolutions = _load_casf_resolutions()
    passed: list[PDBEntry] = []
    errors: list[str] = []

    for pdb_id in pdb_ids:
        pdb_path = cache / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            errors.append(f"{pdb_id}: PDB not cached")
            continue

        entry = PDBEntry(
            pdb_id=pdb_id,
            target_name=_extract_pdb_title(pdb_path),
            resolution=resolutions.get(pdb_id),
            scope=BenchmarkScope.LIGAND,
        )
        protein_path = prepare_protein(pdb_path)
        decision = screen_complex(entry, pdb_path, protein_path)
        if decision.accepted:
            passed.append(entry)

    if errors:
        raise RuntimeError(f"QC screening failed for {len(errors)} entries: {errors}")

    return passed


def get_casf_2007_qc_passed_ids() -> list[str]:
    """Return CASF-2007 entries that pass OpenHCS QC gates.

    Gates: resolution < 2.5A, single MODEL, HAC >= 10, no metals,
    no covalent contacts (< 1.55A), ligand detected in structure.

    Raises:
        RuntimeError: if QC screening fails or cache is missing.
    """
    return [e.pdb_id for e in get_casf_2007_entries()]
