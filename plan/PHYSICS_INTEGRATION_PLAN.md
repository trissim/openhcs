# Plan: Physics Integration (P1.4 - P1.6) - REVISED

**Status**: Based on full Lean proof reading

---

## Lean Definitions Summary

### BoundedAcquisition.lean

| Lean Definition | Type | Description |
|-----------------|------|-------------|
| `BoundedRegion.diameter` | `ℕ` | Region diameter (positive) |
| `BoundedRegion.signalSpeed` | `ℕ` | Signal propagation speed (positive) |
| `maxAcquisitions(R, T)` | `ℕ` | `R.signalSpeed * T / R.diameter` |
| `maxAcquisitions_bound` | theorem | Rate ≤ c/d |
| `srank_le_resolution_bits` | theorem | BA6: bit ops ≥ srank |
| `energy_ge_srank_cost` | theorem | BA7: energy ≥ srank × joulesPerBit |
| `physical_grounding_bundle` | theorem | BA9: info/time/energy unify at srank |

### DecisionTime.lean

| Lean Definition | Type | Description |
|-----------------|------|-------------|
| `TimedState(S)` | structure | `state : S`, `time : Nat` |
| `DecisionProcess(S, A)` | structure | `decide : S → A`, `transition : S → A → S` |
| `tick(P, x)` | function | Returns new TimedState with time + 1 |
| `run(P, n, x)` | function | Runs n ticks |
| `SubstrateModel(Micro, S, A)` | structure | Maps micro dynamics to interface |

### PhysicalCore.lean

| Lean Definition | Type | Description |
|-----------------|------|-------------|
| `PhysicalComputer` | structure | `E_max : ℕ`, `bit_cost : ℕ`, `hbit_pos : 0 < bit_cost` |
| `bit_energy_cost(c)` | def | `c.bit_cost` |
| `ThermoModel` | structure | `joulesPerBit : ℕ`, `carbonPerJoule : ℕ` |
| `energyLowerBound(M, bits)` | def | `M.joulesPerBit * bits` |
| `feasibleWork(c, bits)` | def | `bits * bit_energy_cost(c) ≤ c.E_max` |
| `landauerJoulesPerBit(kB, T)` | def | `kB * T * log(2)` |

---

## Implementation Plan

### Phase 1: Dependencies (thermo.py)

First implement structures that BoundedAcquisition and PhysicalCore depend on.

```python
# dq_dock_engine/physics/thermo.py
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np

# --- ThermoModel (from ThermodynamicLift.lean) ---

@dataclass(frozen=True)
class ThermoModel:
    """
    Declared thermodynamic conversion model.
    From ThermodynamicLift.lean::ThermoModel
    """
    joules_per_bit: int      # Lower-bound energy cost per bit operation
    carbon_per_joule: int    # Carbon cost per joule
    
    def energy_lower_bound(self, bit_ops: int) -> int:
        """energyLowerBound(M, bits) = M.joulesPerBit * bits"""
        return self.joules_per_bit * bit_ops
    
    @staticmethod
    def landauer_joules_per_bit(kB: float, T: float) -> float:
        """
        Landauer conversion: kB * T * ln(2)
        From ThermodynamicLift.lean::landauerJoulesPerBit
        """
        import math
        return kB * T * math.log(2)
    
    @staticmethod
    def from_temperature(T: float) -> 'ThermoModel':
        """Create ThermoModel at temperature T using Landauer floor."""
        kB = 1.380649e-23  # Boltzmann constant
        joules = int(np.ceil(ThermoModel.landauer_joules_per_bit(kB, T)))
        return ThermoModel(joules_per_bit=joules, carbon_per_joule=0)


# --- PhysicalComputer (from PhysicalHardness.lean) ---

@dataclass(frozen=True)
class PhysicalComputer:
    """
    Physical computer with finite energy budget and positive per-bit cost.
    From PhysicalHardness.lean::PhysicalComputer
    """
    E_max: int               # Maximum energy budget (Joules)
    bit_cost: int           # Energy cost per bit operation
    
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
```

### Phase 2: BoundedAcquisition (bounded_acquisition.py)

```python
# dq_dock_engine/physics/bounded_acquisition.py
from dataclasses import dataclass
from typing import NamedTuple

# --- BoundedRegion (from BoundedAcquisition.lean) ---

@dataclass(frozen=True)
class BoundedRegion:
    """
    Bounded region of spacetime.
    From BoundedAcquisition.lean::BoundedRegion
    """
    diameter: int       # Diameter (positive natural units)
    signal_speed: int   # Signal propagation speed (positive)
    
    def __post_init__(self):
        if self.diameter <= 0:
            raise ValueError("diameter must be positive (hDiameter)")
        if self.signal_speed <= 0:
            raise ValueError("signal_speed must be positive (hSpeed)")
    
    def max_acquisitions(self, T: int) -> int:
        """
        maxAcquisitions(R, T) = R.signalSpeed * T / R.diameter
        From BoundedAcquisition.lean::BA2
        """
        return (self.signal_speed * T) // self.diameter
    
    def acquisition_rate(self) -> float:
        """Rate = signalSpeed / diameter"""
        return self.signal_speed / self.diameter


# --- Core Theorems as Functions ---

def srank_le_resolution_bits(srank: int, sufficient_card: int) -> bool:
    """
    BA6: srank <= |I|
    From BoundedAcquisition.lean::srank_le_resolution_bits
    """
    return srank <= sufficient_card


def energy_ge_srank_cost(
    srank: int,
    joules_per_bit: int
) -> int:
    """
    BA7: energy >= srank * joulesPerBit
    From BoundedAcquisition.lean::energy_ge_srank_cost
    """
    return srank * joules_per_bit


@dataclass(frozen=True)
class AcquisitionBounds:
    """
    Physical grounding bundle (BA9).
    From BoundedAcquisition.lean::physical_grounding_bundle
    """
    srank: int
    region: BoundedRegion
    thermo: 'ThermoModel'  # Forward ref
    
    def minimum_time_steps(self, T: int) -> int:
        """Maximum acquisitions in time T."""
        return self.region.max_acquisitions(T)
    
    def minimum_energy(self) -> int:
        """E_min = srank * joulesPerBit (BA7)"""
        return energy_ge_srank_cost(self.srank, self.thermo.joules_per_bit)
    
    def can_complete_in_time(self, time_steps: int) -> bool:
        """Check if decision can complete within time steps."""
        required_ops = self.srank  # BA6: need at least srank operations
        return self.region.max_acquisitions(time_steps) >= required_ops
    
    def can_complete_with_energy(self, available_energy: int) -> bool:
        """Check if decision can complete with available energy."""
        return available_energy >= self.minimum_energy()
```

### Phase 3: DecisionTime (decision_time.py)

```python
# dq_dock_engine/physics/decision_time.py
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, List
from enum import Enum, auto

S = TypeVar('S')
A = TypeVar('A')

# --- TimedState (from DecisionTime.lean) ---

@dataclass(frozen=True)
class TimedState(Generic[S]):
    """
    Interface state paired with discrete time index.
    From DecisionTime.lean::TimedState
    """
    state: S
    time: int
    
    def tick(self) -> 'TimedState[S]':
        """Advance time by one."""
        return TimedState(state=self.state, time=self.time + 1)


# --- DecisionProcess (from DecisionTime.lean) ---

@dataclass
class DecisionProcess(Generic[S, A]):
    """
    Deterministic decision process at interface level.
    From DecisionTime.lean::DecisionProcess
    """
    decide: Callable[[S], A]           # S → A
    transition: Callable[[S, A], S]   # S → A → S
    
    def step(self, x: TimedState[S]) -> TimedState[S]:
        """
        tick(P, x) - one interface tick
        From DecisionTime.lean::tick
        """
        a = self.decide(x.state)
        new_state = self.transition(x.state, a)
        return TimedState(state=new_state, time=x.time + 1)
    
    def run(self, x: TimedState[S], n: int) -> TimedState[S]:
        """
        run(P, n, x) - run n ticks
        From DecisionTime.lean::run
        """
        result = x
        for _ in range(n):
            result = self.step(result)
        return result
    
    def trace(self, x: TimedState[S], n: int) -> List[A]:
        """
        decisionTrace(P, n, x) - decisions emitted during n ticks
        From DecisionTime.lean::decisionTrace
        """
        decisions = []
        result = x
        for _ in range(n):
            a = self.decide(result.state)
            decisions.append(a)
            result = self.step(result)
        return decisions


# --- SubstrateModel (from DecisionTime.lean) ---

@dataclass
class SubstrateKind(Enum):
    """Substrate type for regime-invariant statements."""
    CLASSICAL = auto()
    QUANTUM = auto()
    CHEMICAL = auto()
    HYBRID = auto()


@dataclass
class SubstrateModel(Generic['Micro', S, A]):
    """
    Substrate-level realization whose interface obeys decision ticks.
    From DecisionTime.lean::SubstrateModel
    """
    kind: SubstrateKind
    micro_step: Callable[['Micro'], 'Micro']
    observe: Callable[['Micro'], S]
    interface: DecisionProcess[S, A]
    
    def consistency_check(self, micro: 'Micro', t: int) -> bool:
        """
        Verify: observe(microStep(μ)) = interface.transition(observe(μ), interface.decide(observe(μ)))
        From DecisionTime.lean::SubstrateModel.consistency
        """
        current_state = self.observe(micro)
        decision = self.interface.decide(current_state)
        next_micro = self.micro_step(micro)
        next_state = self.observe(next_micro)
        expected_next = self.interface.transition(current_state, decision)
        return next_state == expected_next
```

### Phase 4: PhysicalCore (physical_core.py)

```python
# dq_dock_engine/physics/physical_core.py
from dataclasses import dataclass
from typing import Optional, Callable, TYPE_CHECKING
from .thermo import ThermoModel, PhysicalComputer
from .bounded_acquisition import AcquisitionBounds

# Constants
MAX_INSTANCE_SIZE_SEARCH = 10000  # Cap for n0 search in finite_budget_blocks

# --- Finite Budget Theorems ---

def finite_budget_blocks_certification(
    computer: PhysicalComputer,
    required_ops: Callable[[int], int],  # req.required_ops
    instance_size: int
) -> tuple[bool, Optional[int]]:
    """
    finite_budget_blocks_exact_certification_beyond_threshold
    From PhysicalCore.lean::finite_budget_blocks_exact_certification_beyond_threshold
    
    Returns: (blocks_certification, threshold_n0)
    If threshold_n0 is not None, for all n >= threshold_n0, 
    required energy exceeds available energy.
    """
    energy_per_op = computer.bit_energy_cost()
    available = computer.E_max
    
    # Find n0 where required energy exceeds budget
    # From PhysicalCore.lean: ∃ n₀, ∀ n ≥ n₀ → c.E_max < req.required_ops(n) * bit_energy_cost(c)
    for n0 in range(MAX_INSTANCE_SIZE_SEARCH):
        required_energy = required_ops(n0) * energy_per_op
        if required_energy > available:
            return (True, n0)
    
    return (False, None)  # Doesn't block within search range


def feasible_work(computer: PhysicalComputer, bits: int) -> bool:
    """
    feasibleWork(c, bits) = bits * bit_energy_cost(c) <= c.E_max
    From PhysicalCore.lean::feasibleWork
    """
    return computer.feasible_work(bits)


# --- Substrate Transport (theoretical framework) ---

@dataclass
class PhysicalEncoding(Generic['PInst', A, S]):
    """
    Encoding from physical instances to decision problems.
    From PhysicalCore.lean::PhysicalEncoding (simplified)
    """
    encode: Callable[['PInst'], 'DecisionProblem[A, S]']


def substrate_transport_sound(
    encoding: PhysicalEncoding['PInst', A, S],
    core_assm: Callable[['DecisionProblem[A, S]'], bool],
    core_claim: Callable[['DecisionProblem[A, S]'], bool],
    phys_assm: Callable[['PInst'], bool],
    instance: 'PInst'
) -> bool:
    """
    substrate_transport_sound_conditional
    From PhysicalCore.lean::substrate_transport_sound_conditional
    
    If core assumptions transfer to physical instance, 
    then core claim holds for encoded instance.
    """
    if not phys_assm(instance):
        return False
    dp = encoding.encode(instance)
    if not core_assm(dp):
        return False
    return core_claim(dp)
```

### Phase 5: Integration with Router

```python
# dq_dock_engine/pipeline/router.py - Add time-budgeted routing

@dataclass
class BudgetConstraints:
    """Combined time and energy budget constraints."""
    time_budget: float          # seconds (converted to steps internally)
    energy_budget: float        # Joules
    region: 'BoundedRegion'
    thermo: 'ThermoModel'
    computer: 'PhysicalComputer'
    
    def check(self, srank: int) -> tuple[bool, str]:
        """
        Check if srank-sized decision can complete within budgets.
        Returns (can_complete, failure_reason).
        From BA9 physical grounding bundle.
        """
        time_steps = int(self.time_budget)
        bounds = AcquisitionBounds(
            srank=srank,
            region=self.region,
            thermo=self.thermo
        )
        
        # Time constraint: BA2 maxAcquisitions
        max_ops = self.region.max_acquisitions(time_steps)
        if max_ops < srank:
            return (False, f"time: need {srank} ops, have {max_ops} in {time_steps} steps")
        
        # Energy constraint: BA7 energy >= srank * joulesPerBit
        min_energy = bounds.minimum_energy()
        if min_energy > self.energy_budget:
            return (False, f"energy: need {min_energy}J, have {self.energy_budget}J")
        
        return (True, "")


def route_with_budget(
    srank: int,
    constraints: BudgetConstraints,
    positions: jnp.ndarray,
    potential_fn: Callable,
) -> RouterResult:
    """
    Route with time and energy constraints.
    Uses BA9 physical grounding bundle for feasibility checks.
    """
    can_complete, reason = constraints.check(srank)
    
    if not can_complete:
        return RouterResult(
            tractability=Tractability.HARD,
            srank=srank,
            estimated_speedup=1.0,
            structural_reason=reason
        )
    
    # Budget satisfied - route normally based on srank
    return route_by_structure(
        positions=positions,
        potential_fn=potential_fn,
        srank=srank
    )


# --- Convenience factory ---

def create_budget_constraints(
    time_budget_seconds: float,
    energy_budget_joules: float,
    diameter_nm: float = 10.0,
    temperature_kelvin: float = 300.0,
) -> BudgetConstraints:
    """Create BudgetConstraints with sensible defaults."""
    import math
    
    # SR: signal speed = c (speed of light in vacuum, simplified to nm/s)
    signal_speed = int(3e8)  # nm/s
    
    # Convert time to steps (assuming 1 step = 1 tick = minimum region traversal time)
    time_steps = int(time_budget_seconds * signal_speed / diameter_nm)
    
    region = BoundedRegion(
        diameter=int(diameter_nm * 1e9),  # Convert to nm (natural units)
        signal_speed=signal_speed
    )
    
    thermo = ThermoModel.from_temperature(temperature_kelvin)
    
    computer = PhysicalComputer(
        E_max=int(energy_budget_joules),
        bit_cost=thermo.joules_per_bit
    )
    
    return BudgetConstraints(
        time_budget=time_steps,
        energy_budget=energy_budget_joules,
        region=region,
        thermo=thermo,
        computer=computer
    )
```

---

## Files to Create

| File | Lines (est) | Dependencies |
|------|-------------|--------------|
| `physics/thermo.py` | 80 | None |
| `physics/physical_core.py` | 100 | thermo |
| `physics/bounded_acquisition.py` | 120 | thermo, physical_core |
| `physics/decision_time.py` | 150 | None |

---

## Tests

```python
# tests/physics/test_bounded_acquisition.py
import pytest
from dq_dock_engine.physics.bounded_acquisition import BoundedRegion, AcquisitionBounds
from dq_dock_engine.physics.thermo import ThermoModel

def test_bounded_region_max_acquisitions():
    """BA2: maxAcquisitions = signalSpeed * T / diameter"""
    region = BoundedRegion(diameter=10, signal_speed=100)
    assert region.max_acquisitions(50) == 500  # 100 * 50 / 10

def test_acquisition_bounds_minimum_energy():
    """BA7: energy >= srank * joulesPerBit"""
    region = BoundedRegion(diameter=10, signal_speed=100)
    thermo = ThermoModel(joules_per_bit=1000, carbon_per_joule=0)
    bounds = AcquisitionBounds(srank=5, region=region, thermo=thermo)
    assert bounds.minimum_energy() == 5000  # 5 * 1000

def test_can_complete_in_time():
    """Check if srank operations can complete in time T"""
    region = BoundedRegion(diameter=10, signal_speed=100)
    thermo = ThermoModel(joules_per_bit=1000, carbon_per_joule=0)
    bounds = AcquisitionBounds(srank=5, region=region, thermo=thermo)
    # With T=1: max_acquisitions = 100 * 1 / 10 = 10 >= 5 ✓
    assert bounds.can_complete_in_time(1) == True
    # With T=0: max_acquisitions = 0 < 5 ✗
    assert bounds.can_complete_in_time(0) == False
```

```python
# tests/physics/test_decision_time.py
import pytest
from dq_dock_engine.physics.decision_time import TimedState, DecisionProcess, SubstrateKind

def test_tick_increments_time():
    """tick increments time by exactly one"""
    state = TimedState(state=0, time=0)
    new_state = state.tick()
    assert new_state.time == 1

def test_run_n_ticks():
    """run(P, n, x) advances time by n"""
    def decide(s): return s + 1
    def transition(s, a): return a
    
    proc = DecisionProcess(decide=decide, transition=transition)
    state = TimedState(state=0, time=0)
    result = proc.run(state, 5)
    assert result.time == 5

def test_decision_trace_length():
    """decisionTrace length equals n ticks"""
    def decide(s): return s
    def transition(s, a): return s
    
    proc = DecisionProcess(decide=decide, transition=transition)
    state = TimedState(state=0, time=0)
    trace = proc.trace(state, 10)
    assert len(trace) == 10
```

---

## Summary

This plan is based on exact Lean definitions:

1. **thermo.py** - ThermoModel, PhysicalComputer, landauer_joules_per_bit
2. **bounded_acquisition.py** - BoundedRegion, AcquisitionBounds, BA2-BA9 theorems
3. **decision_time.py** - TimedState, DecisionProcess, tick, run, trace, SubstrateModel
4. **physical_core.py** - feasibleWork, finite_budget_blocks (with MAX_INSTANCE_SIZE_SEARCH constant), substrate_transport

All functions map directly to Lean definitions with the same names. Tests verify the mathematical properties from the Lean proofs.

**Key improvements from review:**
- Extracted `MAX_INSTANCE_SIZE_SEARCH = 10000` constant (was magic `1000`)
- Created `BudgetConstraints` dataclass to eliminate duplication in router integration
- Added `create_budget_constraints()` factory for common use cases
