from dataclasses import dataclass, field
from typing import List, Optional
import jax.numpy as jnp

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
    tractability: str = "unknown"

@dataclass
class PhysicsContext:
    """
    Extended execution context for OpenHCS pipelines.
    Provides the environment for verifiable physics computations.
    """
    energy_budget: float = 1e-18  # Default budget in Joules
    temperature: float = 300.0     # K
    sidechannel: PhysicsSidechannel = field(default_factory=PhysicsSidechannel)
    
    def log_physics(self, energy: float, srank: int):
        """Update the sidechannel data (immutable update pattern)."""
        # This is a bit simplified for now
        pass
