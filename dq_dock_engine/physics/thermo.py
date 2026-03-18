from dataclasses import dataclass
import math
import numpy as np

# --- ThermoModel (from ThermodynamicLift.lean) ---


@dataclass(frozen=True)
class ThermoModel:
    """
    Declared thermodynamic conversion model.
    From ThermodynamicLift.lean::ThermoModel
    """

    joules_per_bit: int
    carbon_per_joule: int

    def energy_lower_bound(self, bit_ops: int) -> int:
        """energyLowerBound(M, bits) = M.joulesPerBit * bits"""
        return self.joules_per_bit * bit_ops

    @staticmethod
    def landauer_joules_per_bit(kB: float, T: float) -> float:
        """
        Landauer conversion: kB * T * ln(2)
        From ThermodynamicLift.lean::landauerJoulesPerBit
        """
        return kB * T * math.log(2)

    @staticmethod
    def from_temperature(T: float) -> "ThermoModel":
        """Create ThermoModel at temperature T using Landauer floor."""
        kB = 1.380649e-23
        joules = int(np.ceil(ThermoModel.landauer_joules_per_bit(kB, T)))
        return ThermoModel(joules_per_bit=joules, carbon_per_joule=0)


# --- PhysicalComputer (from PhysicalHardness.lean) ---


@dataclass(frozen=True)
class PhysicalComputer:
    """
    Physical computer with finite energy budget and positive per-bit cost.
    From PhysicalHardness.lean::PhysicalComputer
    """

    E_max: int
    bit_cost: int

    def __post_init__(self):
        if self.bit_cost <= 0:
            raise ValueError("bit_cost must be positive (hbit_pos)")

    def bit_energy_cost(self) -> int:
        """bit_energy_cost(c) = c.bit_cost"""
        return self.bit_cost

    def feasible_work(self, bits: int) -> bool:
        """
        feasibleWork(c, bits) = bits * bit_energy_cost(c) <= c.E_max
        From PhysicalCore.lean::feasibleWork
        """
        return bits * self.bit_energy_cost() <= self.E_max

    def min_energy_for_bits(self, bits: int) -> int:
        """Minimum energy required for given bit operations."""
        return bits * self.bit_energy_cost()
