"""
Proof Status Labels for DQ-Dock Engine
=====================================

This module defines the formal proof status for each component of the docking system.

## Status Definitions

### CERTIFIED
Mathematically proven in Lean 4. The theorem guarantees correctness for any input
satisfying the stated hypotheses. No empirical validation needed.

### CONDITIONALLY_CERTIFIED
Proven in Lean 4 subject to stated physical/empirical assumptions. The proof is valid
IF the assumptions hold. Assumptions must be verified empirically or assumed.

### HEURISTIC
No formal proof. Based on empirical observation, physical intuition, or convention.
Must be validated experimentally before use in safety-critical applications.

## Usage

Decorate functions or classes with `@ProofStatus`:
    from dq_dock_engine.proof_status import ProofStatus, CERTIFIED

    @CERTIFIED
    def molecular_srank_bound(...):
        '''From Lean: MolecularSrank.lean::md_srank_bound'''
        ...

"""

from enum import Enum
from functools import wraps
from typing import Callable, Any


class ProofStatus(Enum):
    """Formal proof status of a computational component."""

    #: Mathematically proven in Lean 4
    CERTIFIED = "CERTIFIED"

    #: Proven subject to physical assumptions
    CONDITIONALLY_CERTIFIED = "CONDITIONALLY_CERTIFIED"

    #: Heuristic, no formal proof
    HEURISTIC = "HEURISTIC"


# Decorator factory
def proof_status(status: ProofStatus, theorem: str = "", assumptions: list[str] = None):
    """
    Decorator to mark a function with its proof status.

    Args:
        status: The proof status (CERTIFIED, CONDITIONALLY_CERTIFIED, HEURISTIC)
        theorem: Lean theorem name if applicable (e.g., "MolecularSrank.lean::md_srank_bound")
        assumptions: List of assumed conditions for CONDITIONALLY_CERTIFIED
    """
    assumptions = assumptions or []

    def decorator(fn: Callable) -> Callable:
        fn._proof_status = status
        fn._lean_theorem = theorem
        fn._proof_assumptions = assumptions

        @wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            return fn(*args, **kwargs)

        wrapper._proof_status = status
        wrapper._lean_theorem = theorem
        wrapper._proof_assumptions = assumptions
        return wrapper

    return decorator


# Convenience decorators
def certified(theorem: str = ""):
    """Mark as CERTIFIED."""
    return proof_status(ProofStatus.CERTIFIED, theorem)


def conditionally_certified(theorem: str = "", assumptions: list[str] = None):
    """Mark as CONDITIONALLY_CERTIFIED."""
    return proof_status(ProofStatus.CONDITIONALLY_CERTIFIED, theorem, assumptions)


def heuristic():
    """Mark as HEURISTIC."""
    return proof_status(ProofStatus.HEURISTIC)


# Status accessors
def get_status(fn: Callable) -> ProofStatus:
    """Get proof status of a function."""
    return getattr(fn, "_proof_status", ProofStatus.HEURISTIC)


def get_theorem(fn: Callable) -> str:
    """Get Lean theorem name of a function."""
    return getattr(fn, "_lean_theorem", "")


def get_assumptions(fn: Callable) -> list[str]:
    """Get proof assumptions of a function."""
    return getattr(fn, "_proof_assumptions", [])


# ============================================================================
# PROOF BACKING MAP
# Maps Python functions to their Lean theorem counterparts
# ============================================================================

PROOF_BACKING = {
    # Structural Rank
    "molecular_srank_bound": {
        "theorem": "MolecularSrank.lean::md_srank_bound",
        "status": ProofStatus.CERTIFIED,
        "constant": "srank ≤ 3K + 3L",
    },
    "compute_srank": {
        "theorem": "StructuralRank.lean::srank_eq_relevant_card",
        "status": ProofStatus.CERTIFIED,
        "note": "Gradient-based relevance detection",
    },
    # Lattice Sums
    "lattice_tail_bound": {
        "theorem": "LatticeSum.lean::latticeTailSum6_le_M_div_R3",
        "status": ProofStatus.CERTIFIED,
        "constant": "M = 8π (explicit)",
        "note": "Dyadic shell decomposition with geometric series bound",
    },
    "lj6_cutoff_error": {
        "theorem": "LatticeSum.lean::lj6_tail_bound",
        "status": ProofStatus.CERTIFIED,
        "constant": "O(1/R³)",
    },
    "lj12_cutoff_error": {
        "theorem": "LatticeSum.lean::lj12_tail_bound",
        "status": ProofStatus.CERTIFIED,
        "constant": "O(1/R⁹)",
    },
    # Ewald Summation
    "ewald_real_space_energy": {
        "theorem": "EwaldSummation.lean::ewald_real_space_exponential_decay",
        "status": ProofStatus.CONDITIONALLY_CERTIFIED,
        "assumptions": [
            "Real.erfc is correctly implemented",
            "Minimum image convention is physically appropriate",
        ],
    },
    "ewald_reciprocal_energy": {
        "theorem": "EwaldSummation.lean::ewaldReciprocalCore",
        "status": ProofStatus.CONDITIONALLY_CERTIFIED,
        "assumptions": ["k-space truncation is sufficient"],
    },
    # Physics
    "hamiltonian": {
        "theorem": "SymplecticIntegrator.lean::hamiltonian",
        "status": ProofStatus.CERTIFIED,
    },
    "thermodynamic_lower_bound": {
        "theorem": "MolecularSrank.lean::md_thermodynamic_lower_bound",
        "status": ProofStatus.CONDITIONALLY_CERTIFIED,
        "assumptions": ["Landauer principle holds", "System is at equilibrium"],
    },
    # Potentials
    "lennard_jones_potential": {
        "theorem": "ArrayDSL.lean::lennardJones",
        "status": ProofStatus.CERTIFIED,
        "note": "Core form only; weights are heuristic",
    },
}


def lookup_proof_status(fn_name: str) -> dict:
    """Look up the proof backing for a function."""
    return PROOF_BACKING.get(
        fn_name, {"status": ProofStatus.HEURISTIC, "note": "No Lean backing"}
    )
