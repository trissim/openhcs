"""
Large-scale PDB benchmark for CERTIFIED docking.

Curated list of ~50 well-known, high-resolution drug-target complexes.
These are famous in the SBDD literature and have crystal structures
with clear binding pockets.
"""

from dataclasses import dataclass


@dataclass
class PDBEntry:
    pdb_id: str
    target_name: str
    resolution: float


BENCHMARK_PDB_IDS = [
    # HIV Protease (classic drug target, ~10 drugs approved)
    PDBEntry("1hvr", "HIV-1 Protease", 2.0),
    PDBEntry("1ajx", "HIV-1 Protease", 2.3),
    PDBEntry("1byw", "HIV-1 Protease", 2.0),
    PDBEntry("1dmp", "HIV-1 Protease", 2.0),
    PDBEntry("1pro", "HIV-1 Protease", 2.0),
    PDBEntry("2p3b", "HIV-1 Protease", 1.7),
    # SARS-CoV-2 / SARS-CoV Main Protease (COVID targets)
    PDBEntry("6lu7", "SARS-CoV-2 Mpro", 2.16),
    PDBEntry("7btf", "SARS-CoV-2 Mpro", 1.45),
    PDBEntry("5r84", "SARS-CoV-2 Mpro", 1.60),
    PDBEntry("6w63", "SARS-CoV Mpro", 1.90),
    # Kinases (largest drug target class)
    PDBEntry("1atp", "CDK2", 2.1),
    PDBEntry("1m17", "CDK2", 2.4),
    PDBEntry("2hyy", "CDK2", 1.9),
    PDBEntry("1jvp", "CDK2", 2.0),
    PDBEntry("3ddi", "CDK2", 2.3),
    PDBEntry("1ywr", "AKT1/PKB", 2.5),
    PDBEntry("3cjs", "ABL1 Kinase", 2.0),
    PDBEntry("2f4j", "SRC Kinase", 1.9),
    PDBEntry("3t5n", "EGFR", 1.7),
    PDBEntry("2ity", "EGFR", 2.05),
    # Carbonic Anhydrases (important, tight binders)
    PDBEntry("1z00", "Carbonic Anhydrase II", 2.0),
    PDBEntry("3ddZ", "Carbonic Anhydrase IX", 1.9),
    PDBEntry("1bn3", "Carbonic Anhydrase I", 2.2),
    # Thrombin / Coagulation (important drug targets)
    PDBEntry("1ppb", "Thrombin", 2.0),
    PDBEntry("1c5w", "Thrombin", 2.3),
    PDBEntry("1ml4", "Factor Xa", 2.0),
    # BACE (Alzheimer's)
    PDBEntry("4pfv", "BACE-1", 1.5),
    PDBEntry("4d83", "BACE-1", 1.95),
    PDBEntry("4i0t", "BACE-1", 1.67),
    PDBEntry("4i0e", "BACE-1", 1.7),
    # Nuclear receptors
    PDBEntry("1dbb", "Estrogen Receptor alpha", 2.25),
    PDBEntry("1r4o", "Estrogen Receptor alpha", 2.0),
    PDBEntry("3e3c", "Androgen Receptor", 2.05),
    PDBEntry("1l2i", "Retinoic Acid Receptor", 2.0),
    PDBEntry("2p54", "Glucocorticoid Receptor", 1.8),
    # HIV Integrase
    PDBEntry("1QS4", "HIV-1 Integrase", 1.30),
    PDBEntry("3ltj", "HIV-1 Integrase", 1.7),
    # Cathepsins / Proteases
    PDBEntry("1QD6", "Cathepsin K", 1.90),
    PDBEntry("3b5t", "Cathepsin L", 1.85),
    PDBEntry("2y0h", "Cathepsin S", 1.7),
    # Phosphodiesterases
    PDBEntry("3myy", "PDE4B", 2.0),
    PDBEntry("1XO2", "PDE5A", 2.0),
    # Heat shock proteins
    PDBEntry("3K99", "HSP90", 1.7),
    # Tankyrases / PARPs
    PDBEntry("3t0i", "Tankyase 2", 2.20),
    PDBEntry("4r16", "PARP-1", 1.96),
    PDBEntry("3l3m", "PARP-1", 1.88),
    # Factor VII
    PDBEntry("1fak", "Factor VIIa", 2.0),
    # Renin
    PDBEntry("3G6Z", "Renin", 2.0),
    # MMPs
    PDBEntry("3AYU", "MMP-12", 1.74),
    PDBEntry("1ZTZ", "MMP-2", 2.05),
    # Viral proteases
    PDBEntry("3k7v", "Hepatitis C NS3/4A Protease", 1.8),
    # Protein phosphatases
    PDBEntry("3l4s", "PTP1B", 1.8),
    # Bromodomains
    PDBEntry("3G0L", "BRD4(1)", 1.94),
    PDBEntry("3Wma", "BRD4(2)", 1.5),
]


def get_benchmark_pdb_ids() -> list[str]:
    return [e.pdb_id.lower() for e in BENCHMARK_PDB_IDS]


def get_benchmark_info() -> dict[str, PDBEntry]:
    return {e.pdb_id.lower(): e for e in BENCHMARK_PDB_IDS}
