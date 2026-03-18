"""Update router.py to use enum-driven dispatch into algorithms."""
from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable, Optional
import jax.numpy as jnp

class Tractability(Enum):
    """Cases derived from StructuralRank.lean for provable efficiency."""
    SEPARABLE = auto()
    TENSOR_RANK = auto()
    TREEWIDTH = auto()
    SYMMETRIC = auto()
    HARD = auto()

    @property
    def algorithm_path(self) -> str:
        """Indirection Minimization: direct property on enum."""
        return {
            Tractability.SEPARABLE: "physics.algorithms.run_separable",
            Tractability.TENSOR_RANK: "physics.algorithms.run_factorized",
            Tractability.TREEWIDTH: "physics.algorithms.run_factorized",
            Tractability.SYMMETRIC: "physics.algorithms.run_factorized",
            Tractability.HARD: "physics.algorithms.run_standard_md",
        }[self]

    def run(self, state, force_fn, dt: float, n_steps: int, **kwargs):
        """Enum-driven dispatch: Tractability.HARD.run(state, ...) instead of if/elif chains."""
        from dq_dock_engine.physics.algorithms import run_standard_md, run_factorized, run_separable
        dispatch = {
            Tractability.SEPARABLE: lambda: run_separable(state, force_fn),
            Tractability.TENSOR_RANK: lambda: run_factorized(state, force_fn, dt, n_steps),
            Tractability.TREEWIDTH: lambda: run_factorized(state, force_fn, dt, n_steps),
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

    @property
    def algorithm(self) -> str:
        return self.tractability.algorithm_path

def route_by_srank(srank: int, total_dim: int) -> RouterResult:
    """Algebraic decision logic. Concise ternary chain."""
    ratio = srank / total_dim
    tractability = (
        Tractability.SEPARABLE if srank == 0 else
        Tractability.TENSOR_RANK if ratio < 0.1 else
        Tractability.TREEWIDTH if ratio < 0.5 else
        Tractability.HARD
    )
    speedup = float('inf') if srank == 0 else (1.0/ratio if ratio > 0 else 1.0)
    if tractability == Tractability.TENSOR_RANK:
        speedup = speedup ** 2
    return RouterResult(tractability, srank, speedup)
