from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List

"""
Substrate Cost κ.
Direct translation of StochasticSequential/SubstrateCost.lean.

Key theorems:
- Verdict is substrate-independent
- Trajectory is substrate-dependent  
- κ is sufficient statistic: same κ values → same trajectory
"""

class SubstrateType(Enum):
    """Hardware substrate types."""
    CPU = auto()
    GPU = auto()
    TPU = auto()
    FPGA = auto()
    QUANTUM = auto()

@dataclass(frozen=True)
class MatrixCell:
    """
    Cell in integrity-competence matrix.
    From SubstrateCost.lean::MatrixCell.
    """
    integrity: bool
    attempted: bool
    competence_available: bool

@dataclass(frozen=True)
class SubstrateConfig:
    """Substrate-specific cost parameters."""
    substrate_type: SubstrateType
    latency_ns: float = 0.0
    thermal_limit_w: float = 0.0
    noise_floor: float = 0.0
    memory_bandwidth_gb_s: float = 0.0
    energy_density_j_gb: float = 0.0

def substrate_cost(
    cell: MatrixCell,
    config: SubstrateConfig
) -> float:
    """
    κ(cell, substrate) — substrate cost function.
    From SubstrateCost.lean::substrateCost.
    """
    base = config.latency_ns * 1e-9
    if config.thermal_limit_w > 0:
        base += config.energy_density_j_gb
    base += config.noise_floor
    if config.memory_bandwidth_gb_s > 0:
        base += 1.0 / config.memory_bandwidth_gb_s
    return base

def trajectory(
    problems: list,
    kappa_fn: Callable[[MatrixCell, SubstrateType], float],
    substrate: SubstrateType
) -> List[tuple]:
    """
    From SubstrateCost.lean::trajectory.
    Problem sequence → list of (cell, cost) pairs.
    """
    cell = MatrixCell(integrity=True, attempted=True, competence_available=True)
    return [(cell, kappa_fn(cell, substrate)) for _ in problems]

def kappa_sufficient_statistic(
    problems: list,
    kappa_fn: Callable,
    tau1: SubstrateType,
    tau2: SubstrateType
) -> bool:
    """
    From kappa_sufficient_statistic:
    If κ(c, τ₁) = κ(c, τ₂) for all cells, then trajectories are equal.
    κ captures all decision-relevant substrate information.
    """
    traj1 = trajectory(problems, kappa_fn, tau1)
    traj2 = trajectory(problems, kappa_fn, tau2)
    return traj1 == traj2

# --- Default configs ---

DEFAULT_CONFIGS: Dict[SubstrateType, SubstrateConfig] = {
    SubstrateType.CPU: SubstrateConfig(SubstrateType.CPU, latency_ns=1.0, memory_bandwidth_gb_s=50.0),
    SubstrateType.GPU: SubstrateConfig(SubstrateType.GPU, latency_ns=0.1, memory_bandwidth_gb_s=900.0),
    SubstrateType.TPU: SubstrateConfig(SubstrateType.TPU, latency_ns=0.05, memory_bandwidth_gb_s=1200.0),
}
