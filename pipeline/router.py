from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable, Dict, Any, List
import jax.numpy as jnp

class Tractability(Enum):
    """
    Cases derived from StructuralRank.lean for provable efficiency.
    """
    SEPARABLE = auto()      # srank = 0
    TENSOR_RANK = auto()    # Low tensor rank (factorized)
    TREEWIDTH = auto()      # Bounded treewidth
    SYMMETRIC = auto()      # Orbit reduction
    HARD = auto()           # coNP-hard (requires approximation)

    @property
    def algorithm_path(self) -> str:
        """Indirection Minimization: Direct property access instead of dispatch tables."""
        return {
            Tractability.SEPARABLE: "physics.algorithms.separable",
            Tractability.TENSOR_RANK: "physics.algorithms.tensor_rank",
            Tractability.TREEWIDTH: "physics.algorithms.treewidth",
            Tractability.SYMMETRIC: "physics.algorithms.symmetric",
            Tractability.HARD: "physics.algorithms.standard_md",
        }[self]

@dataclass(frozen=True)
class RouterResult:
    """Carries the outcome of srank routing."""
    tractability: Tractability
    srank: int
    estimated_speedup: float

    @property
    def algorithm(self) -> str:
        """Indirection Minimization: Access algorithm path directly from the enum."""
        return self.tractability.algorithm_path

def route_by_srank(srank: int, total_dim: int) -> RouterResult:
    """
    Algebraic decision logic following DQ-Dock principles.
    Uses concise lookup and ternary logic for clarity.
    """
    ratio = srank / total_dim
    
    # Mathematical Simplification: Use a concise conditional chain
    tractability = (
        Tractability.SEPARABLE if srank == 0 else
        Tractability.TENSOR_RANK if ratio < 0.1 else
        Tractability.TREEWIDTH if ratio < 0.5 else
        Tractability.HARD
    )
    
    speedup = float('inf') if srank == 0 else (1.0/ratio if ratio > 0 else 1.0)
    if tractability == Tractability.TENSOR_RANK:
        speedup = speedup ** 2 # Quadratic speedup for factorized kernels
        
    return RouterResult(tractability, srank, speedup)
