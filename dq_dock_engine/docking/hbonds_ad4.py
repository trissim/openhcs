"""
AutoDock4-style hydrogen bond potential.

Reference: AutoDock4.2 User Guide (2019), Section 6.3 "Hydrogen Bond Potential"

OpenHCS Compliance:
- ABC contract for H-bond calculators
- Frozen dataclasses for parameters
- Enum-driven donor types
- Explicit dependency injection
- Fail-loud validation
- Pure JAX functions
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto


# =============================================================================
# ENUM-DRIVEN CONFIGURATION
# =============================================================================


class HBondDonorType(Enum):
    """AD4 hydrogen bond donor types per AutoDock4.2 User Guide."""

    O_DONOR = auto()  # Oxygen donor (stronger H-bond)
    N_DONOR = auto()  # Nitrogen donor (similar to O)
    S_DONOR = auto()  # Sulfur donor (weaker H-bond)


# =============================================================================
# FROZEN DATACLASS PARAMETERS (OpenHCS Pattern)
# =============================================================================


@dataclass(frozen=True)
class HBondParameters:
    """
    AutoDock4 hydrogen bond parameters.

    OpenHCS Compliance:
    - @dataclass(frozen=True) for immutability
    - Explicit types
    - Factory methods for standard configs
    - Fail-loud validation

    Reference: AutoDock4.2 User Guide, Section 6.3 "Hydrogen Bond Potential"
    """

    r_o: float = 1.9
    depth: float = 5.0
    angle_0: float = 180.0
    angle_power: int = 2

    def validate(self) -> None:
        """Fail-loud validation per OpenHCS principles."""
        if not (1.0 <= self.r_o <= 4.0):
            raise ValueError(f"r_o {self.r_o} outside AD4 range [1.0, 4.0]")
        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")
        if self.angle_power not in (1, 2):
            raise ValueError(f"angle_power must be 1 or 2, got {self.angle_power}")

    @staticmethod
    def for_oxygen_nitrogen() -> HBondParameters:
        """Parameters for O/N bonds (standard)."""
        return HBondParameters(r_o=1.9, depth=5.0)

    @staticmethod
    def for_sulfur() -> HBondParameters:
        """Parameters for S-H bonds (weaker)."""
        return HBondParameters(r_o=2.5, depth=1.0)


@dataclass(frozen=True)
class HBondDonor:
    """Hydrogen bond donor group."""

    donor_atom_idx: int
    hydrogen_idx: int
    donor_type: HBondDonorType
    position: jnp.ndarray
    h_position: jnp.ndarray
    direction: jnp.ndarray


@dataclass(frozen=True)
class HBondAcceptor:
    """Hydrogen bond acceptor group."""

    acceptor_idx: int
    position: jnp.ndarray
    lone_pair_direction: jnp.ndarray


# =============================================================================
# ABC CONTRACT FOR H-BOND CALCULATORS (OpenHCS Pattern)
# =============================================================================


class HBondCalculator(ABC):
    """
    ABC contract for hydrogen bond calculators.

    OpenHCS Compliance:
    - ABC enforces explicit contract
    - @abstractmethod for required methods
    """

    @abstractmethod
    def compute_distance_energy(
        self,
        r: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute 12-10 distance term."""

    @abstractmethod
    def compute_angular_energy(
        self,
        donor_direction: jnp.ndarray,
        acceptor_direction: jnp.ndarray,
    ) -> float:
        """Compute cos² angular term."""

    @abstractmethod
    def total_energy(
        self,
        donor: HBondDonor,
        acceptor: HBondAcceptor,
    ) -> float:
        """Total H-bond energy: E_dist * E_angle."""

    @property
    @abstractmethod
    def parameters(self) -> HBondParameters:
        """Direct access to parameters (no defensive getattr)."""


class AD4HBondCalculator(HBondCalculator):
    """
    AutoDock4 12-10 hydrogen bond calculator.

    OpenHCS Compliance:
    - Explicit dependency injection
    - Fail-loud on invalid state
    - Pure functions
    """

    def __init__(
        self,
        parameters: HBondParameters | None = None,
    ) -> None:
        self._params = (
            parameters
            if parameters is not None
            else HBondParameters.for_oxygen_nitrogen()
        )
        self._params.validate()

    @property
    def parameters(self) -> HBondParameters:
        """Direct access per OpenHCS - no defensive getattr."""
        return self._params

    def compute_distance_energy(self, r: jnp.ndarray) -> jnp.ndarray:
        """AutoDock4 12-10 distance term."""
        r_o = self._params.r_o
        depth = self._params.depth
        r_safe = jnp.maximum(r, 0.1)
        r_o_r = r_o / r_safe
        return depth * (5.0 * r_o_r**12 - 6.0 * r_o_r**10)

    def compute_angular_energy(
        self,
        donor_direction: jnp.ndarray,
        acceptor_direction: jnp.ndarray,
    ) -> float:
        """cos²(θ) angular term."""
        cos_theta = jnp.dot(donor_direction, acceptor_direction)
        cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
        theta = jnp.arccos(cos_theta)
        theta_0 = jnp.deg2rad(self._params.angle_0)
        return float(jnp.cos(theta - theta_0) ** self._params.angle_power)

    def total_energy(
        self,
        donor: HBondDonor,
        acceptor: HBondAcceptor,
    ) -> float:
        """Total H-bond energy."""
        h_to_acceptor = acceptor.position - donor.h_position
        r = jnp.linalg.norm(h_to_acceptor)

        if r > 4.0:
            return 0.0

        E_dist = float(jnp.sum(self.compute_distance_energy(jnp.array([r]))))
        E_angle = self.compute_angular_energy(
            donor.direction, acceptor.lone_pair_direction
        )

        return E_dist * E_angle


# =============================================================================
# FACTORY FUNCTION (Explicit Dependency Injection)
# =============================================================================


def create_hbond_calculator(
    donor_type: HBondDonorType,
    parameters: HBondParameters | None = None,
) -> HBondCalculator:
    """
    Factory function with explicit dependency injection.

    OpenHCS Compliance:
    - Explicit factory, not __init__
    - No hidden object creation
    - Enum-driven dispatch
    """
    params = parameters
    if params is None:
        params = (
            HBondParameters.for_sulfur()
            if donor_type == HBondDonorType.S_DONOR
            else HBondParameters.for_oxygen_nitrogen()
        )

    return AD4HBondCalculator(params)


# =============================================================================
# PURE JAX FUNCTIONS (Stateless, No Side Effects)
# =============================================================================


@jax.jit
def hbond_energy_single(
    donor: HBondDonor,
    acceptor: HBondAcceptor,
    calculator: AD4HBondCalculator,
) -> float:
    """H-bond energy for a single donor-acceptor pair."""
    return calculator.total_energy(donor, acceptor)


def total_hbond_energy(
    pose_donors: tuple[HBondDonor, ...],
    pose_acceptors: tuple[HBondAcceptor, ...],
    receptor_donors: tuple[HBondDonor, ...],
    receptor_acceptors: tuple[HBondAcceptor, ...],
    calculator: HBondCalculator,
) -> float:
    """
    Total hydrogen bond energy for a pose.

    OpenHCS Compliance:
    - Pure function (explicit calculator dependency)
    - Tuple inputs (immutable)
    - No side effects
    """
    total = 0.0

    for donor in pose_donors:
        for acceptor in receptor_acceptors:
            total += calculator.total_energy(donor, acceptor)

    for donor in receptor_donors:
        for acceptor in pose_acceptors:
            total += calculator.total_energy(donor, acceptor)

    return total
