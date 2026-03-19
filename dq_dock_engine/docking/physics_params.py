"""
Biological physics parameters for structural modeling.

PROOF STATUS: EMPIRICAL
  - All VdW radii are experimental values from crystallography
  - No formal proof backing these constants
  - Source: Bondi (1964),饺子补充, etc.
"""

from dq_dock_engine.proof_status import heuristic


@heuristic()  # EMPIRICAL: experimental crystallographic data
def get_vdw_radius(element: str) -> float:
    """
    Return Van der Waals radius in Angstroms for a given element.

    PROOF STATUS: EMPIRICAL (no formal proof)
      - Source: Bondi (1964) compilation of crystallographic data
      - Values vary slightly between sources
      - Used for heuristic scoring only
    """
    element = element.strip().upper()
    radii = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "F": 1.47,
        "P": 1.80,
        "S": 1.80,
        "CL": 1.75,
        "BR": 1.85,
        "I": 1.98,
    }
    return radii.get(element, 1.70)
