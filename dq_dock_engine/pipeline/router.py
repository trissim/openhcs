"""
Router: structural-condition-based tractability dispatch.
Replaces float-threshold heuristics with Lean-derived structural checks.

From StructuralRank.lean, SeparableUtility.lean, TreeStructure.lean,
BoundedActions.lean, MolecularSrank.lean, FPT.lean.
"""
from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable, Optional
import jax.numpy as jnp

class Tractability(Enum):
    """Cases derived from StructuralRank.lean for provable efficiency."""
    SEPARABLE = auto()     # srank = 0: u(a,s) = f(a) + g(s)
    TENSOR_RANK = auto()   # Low tensor rank: CP decomposition tractable
    TREEWIDTH = auto()     # Bounded treewidth: CSP tractable
    BOUNDED_ACTIONS = auto()  # |A| bounded: O(|S|² · k²)
    SYMMETRIC = auto()     # Orbit reduction
    HARD = auto()          # srank ≈ n: coNP-hard under ETH

    @property
    def algorithm_path(self) -> str:
        """Indirection Minimization: direct property on enum."""
        return {
            Tractability.SEPARABLE: "physics.algorithms.run_separable",
            Tractability.TENSOR_RANK: "physics.algorithms.run_factorized",
            Tractability.TREEWIDTH: "physics.algorithms.run_factorized",
            Tractability.BOUNDED_ACTIONS: "physics.algorithms.run_standard_md",
            Tractability.SYMMETRIC: "physics.algorithms.run_factorized",
            Tractability.HARD: "physics.algorithms.run_standard_md",
        }[self]

    def run(self, state, force_fn, dt: float, n_steps: int, **kwargs):
        """Enum-driven dispatch: Tractability.HARD.run(state, ...)."""
        from dq_dock_engine.physics.algorithms import run_standard_md, run_factorized, run_separable
        dispatch = {
            Tractability.SEPARABLE: lambda: run_separable(state, force_fn),
            Tractability.TENSOR_RANK: lambda: run_factorized(state, force_fn, dt, n_steps),
            Tractability.TREEWIDTH: lambda: run_factorized(state, force_fn, dt, n_steps),
            Tractability.BOUNDED_ACTIONS: lambda: run_standard_md(state, force_fn, dt, n_steps, **kwargs),
            Tractability.SYMMETRIC: lambda: run_factorized(state, force_fn, dt, n_steps),
            Tractability.HARD: lambda: run_standard_md(state, force_fn, dt, n_steps, **kwargs),
        }
        return dispatch[self]()

@dataclass(frozen=True)
class RouterResult:
    """Carries the outcome of srank routing."""
    tractability: Tractability
    srank: int
    estimated_speedup: float
    structural_reason: str = ""

    @property
    def algorithm(self) -> str:
        return self.tractability.algorithm_path


def route_by_srank(srank: int, total_dim: int) -> RouterResult:
    """
    Algebraic routing with structural condition checks.
    
    From the Lean proofs:
    - srank = 0 → SEPARABLE (SeparableUtility.lean)
    - srank < total_dim → check conditions for TENSOR_RANK/TREEWIDTH
    - srank = total_dim → HARD
    
    NOTE: This function uses srank alone. For full structural classification
    use route_by_structure() which checks separability, treewidth, etc.
    """
    if srank == 0:
        return RouterResult(Tractability.SEPARABLE, 0, float('inf'),
                          "srank=0: optimal action is state-independent")
    
    if srank == total_dim:
        return RouterResult(Tractability.HARD, srank, 1.0,
                          f"srank={srank}=n: all coordinates relevant")
    
    # srank < total_dim: potentially tractable, speedup exists
    ratio = srank / total_dim
    speedup = (total_dim / srank) ** 2
    
    return RouterResult(Tractability.TREEWIDTH, srank, speedup,
                       f"srank={srank}/{total_dim}: {total_dim - srank} irrelevant coords")


def route_by_structure(
    utility_matrix: Optional[jnp.ndarray] = None,
    positions: Optional[jnp.ndarray] = None,
    potential_fn: Optional[Callable] = None,
    srank: Optional[int] = None,
    total_dim: Optional[int] = None,
    n_actions: Optional[int] = None,
    treewidth_bound: int = 10
) -> RouterResult:
    """
    Full structural-condition routing from Lean proofs.
    Checks in order of tractability strength:
    
    1. Separable? (SeparableUtility.lean) — srank = 0
    2. Low tensor rank? (SeparableUtility.lean) — CP decomposition
    3. Bounded actions? (BoundedActions.lean) — O(|S|²·k²)
    4. Bounded treewidth? (TreeStructure.lean) — CSP on interaction graph
    5. Otherwise: HARD
    """
    # 1. Check separability (utility matrix available)
    if utility_matrix is not None:
        from dq_dock_engine.physics.tractability.separable import detect_separability, detect_low_tensor_rank
        
        if detect_separability(utility_matrix):
            return RouterResult(Tractability.SEPARABLE, 0, float('inf'),
                              "Separable: argmax is state-independent")
        
        rank = detect_low_tensor_rank(utility_matrix, max_rank=10)
        if rank <= 3:  # Low tensor rank
            speedup = (utility_matrix.shape[1] / max(rank, 1)) ** 2
            return RouterResult(Tractability.TENSOR_RANK, rank, speedup,
                              f"Low tensor rank R={rank}")
    
    # 2. Check bounded actions
    if n_actions is not None and n_actions <= 100:
        speedup = float(total_dim or 1) if total_dim else 1.0
        return RouterResult(Tractability.BOUNDED_ACTIONS, srank or 0, speedup,
                          f"Bounded actions |A|={n_actions}")
    
    # 3. Check treewidth (positions + potential available)
    if positions is not None and potential_fn is not None:
        from dq_dock_engine.physics.tractability.tree_structure import (
            build_interaction_graph, estimate_treewidth
        )
        adjacency = build_interaction_graph(potential_fn, positions)
        tw = estimate_treewidth(adjacency)
        if tw <= treewidth_bound:
            speedup = (total_dim or positions.shape[0] * 3) / max(tw + 1, 1)
            return RouterResult(Tractability.TREEWIDTH, srank or tw, speedup,
                              f"Bounded treewidth w={tw}")
    
    # 4. Fallback to srank-based routing
    if srank is not None and total_dim is not None:
        return route_by_srank(srank, total_dim)
    
    # 5. Default: HARD
    return RouterResult(Tractability.HARD, srank or 0, 1.0,
                       "Insufficient information for structural classification")
