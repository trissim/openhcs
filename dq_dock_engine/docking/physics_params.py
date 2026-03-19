"""
Biological physics parameters for structural modeling.

PROOF STATUS: EMPIRICAL_CONSTANT
  - All VdW radii are experimental values from crystallography
  - Source: Bondi (1964)
"""

from dq_dock_engine.proof_status import empirical_constant


@empirical_constant("Bondi (1964) J. Phys. Chem. 68, 441-451")
def get_vdw_radius(element: str) -> float:
    """
    Return Van der Waals radius in Angstroms for a given element.

    PROOF STATUS: EMPIRICAL_CONSTANT
      - Source: Bondi (1964) compilation of crystallographic data
      - These are GROUND TRUTH measurements, not heuristics
      - Used for all physics-based scoring
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
