from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum, auto

"""
Temporal Integrity.
Direct translation of StochasticSequential/TemporalIntegrity.lean.

Key theorems:
- Claims evolve monotonically (evidence only increases)
- Empty and singleton sequences have integrity
- Temporal integrity = evidence-monotone claim sequence
"""

class ClaimType(Enum):
    """Type of sufficiency claim."""
    SUFFICIENT = auto()
    INSUFFICIENT = auto()

@dataclass(frozen=True)
class Evidence:
    """Evidence supporting a claim."""
    states: List[str]
    proof: bool

@dataclass(frozen=True)
class ClaimWithEvidence:
    """
    A claim with supporting evidence.
    From TemporalIntegrity.lean::ClaimWithEvidence.
    """
    claim_type: ClaimType
    coordinates: frozenset
    evidence: Evidence

@dataclass
class ClaimSequence:
    """
    Sequence of claims over time.
    From TemporalIntegrity.lean::ClaimSequence.
    """
    claims: List[ClaimWithEvidence] = field(default_factory=list)
    
    def add_claim(self, claim: ClaimWithEvidence) -> None:
        """Add a new claim to the sequence."""
        self.claims.append(claim)
    
    def is_monotone(self) -> bool:
        """
        From claimsMonotone:
        Evidence monotonically increases (each claim has at least as much
        evidence as the previous one).
        """
        for i in range(1, len(self.claims)):
            if len(self.claims[i].evidence.states) < len(self.claims[i-1].evidence.states):
                return False
        return True
    
    def temporal_integrity(self) -> bool:
        """
        From temporalIntegrity:
        Claims are monotonically refined.
        
        empty_sequence_integrity: [] has integrity.
        singleton_integrity: [c] has integrity.
        """
        return self.is_monotone()
    
    @property
    def length(self) -> int:
        """sequence_length_finite: len is finite (trivially)."""
        return len(self.claims)
