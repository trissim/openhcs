"""
Biological physics parameters for structural modeling.
"""

def get_vdw_radius(element: str) -> float:
    """
    Return realistic Van der Waals radius in Angstroms for a given element.
    Provides parameters for common organic and biological atoms.
    """
    element = element.strip().upper()
    radii = {
        'H': 1.20,
        'C': 1.70,
        'N': 1.55,
        'O': 1.52,
        'F': 1.47,
        'P': 1.80,
        'S': 1.80,
        'CL': 1.75,
        'BR': 1.85,
        'I': 1.98,
        # Default typical heavy atom radius (~3.4A diameter)
    }
    return radii.get(element, 1.70)
