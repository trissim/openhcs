from dataclasses import dataclass, field
from typing import List
from dq_dock_engine.pipeline.router import Tractability

@dataclass(frozen=True)
class PhysicsSidechannel:
    """
    Sidechannel for physics-metadata following DQ-Dock engine plan.
    Carries mathematical invariants alongside the tensor data.
    """
    srank: int = 0
    energy: float = 0.0
    thermodynamic_cost: float = 0.0
    relevant_features: List[int] = field(default_factory=list)
    complexity_bits: float = 0.0
    tractability: Tractability = Tractability.HARD  # Enum, not magic string

@dataclass(frozen=True)
class PhysicsContext:
    """
    Frozen execution context for verifiable physics computations.
    Explicit dependency injection: no hidden state, no dead methods.
    """
    energy_budget: float = 1e-18  # Joules
    temperature: float = 300.0     # K
    sidechannel: PhysicsSidechannel = field(default_factory=PhysicsSidechannel)
