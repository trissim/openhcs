from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, List
from enum import Enum, auto

"""
Decision Time — discrete-time decision process model.
Direct translation of Physics/DecisionTime.lean.

Key theorems:
- tick increments time by exactly 1
- decision count = elapsed time
- substrate realization preserves interface ticks
"""

S = TypeVar('S')
A = TypeVar('A')

# --- TimedState ---

@dataclass(frozen=True)
class TimedState:
    """
    Interface state paired with discrete time index.
    From DecisionTime.lean::TimedState.
    """
    state: object
    time: int

# --- DecisionProcess ---

@dataclass(frozen=True)
class DecisionProcess:
    """
    Deterministic decision process at interface level.
    From DecisionTime.lean::DecisionProcess.
    """
    decide_fn: Callable  # S → A
    transition_fn: Callable  # (S, A) → S

def tick(process: DecisionProcess, timed_state: TimedState) -> TimedState:
    """
    One interface tick: emit decision, advance time by 1.
    From DecisionTime.lean::tick.
    
    tick_increments_time: (tick P x).time = x.time + 1
    """
    action = process.decide_fn(timed_state.state)
    new_state = process.transition_fn(timed_state.state, action)
    return TimedState(state=new_state, time=timed_state.time + 1)

def run(process: DecisionProcess, n_steps: int, initial: TimedState) -> TimedState:
    """
    Run n ticks from initial state.
    From DecisionTime.lean::run.
    
    run_time_exact: (run P n x).time = x.time + n
    """
    state = initial
    for _ in range(n_steps):
        state = tick(process, state)
    return state

def decision_trace(
    process: DecisionProcess, n_steps: int, initial: TimedState
) -> list:
    """
    Trace of decisions emitted during n ticks.
    From DecisionTime.lean::decisionTrace.
    
    decisionTrace_length_eq_ticks: len(trace) = n
    decision_count_equals_elapsed_time: len(trace) = elapsed time
    """
    trace = []
    state = initial
    for _ in range(n_steps):
        action = process.decide_fn(state.state)
        trace.append(action)
        state = tick(process, state)
    return trace

# --- Substrate Model ---

class SubstrateKind(Enum):
    """From DecisionTime.lean::SubstrateKind."""
    CLASSICAL = auto()
    QUANTUM = auto()
    CHEMICAL = auto()
    HYBRID = auto()

@dataclass(frozen=True)
class SubstrateModel:
    """
    Substrate-level realization whose interface obeys decision ticks.
    From DecisionTime.lean::SubstrateModel.
    
    Key property (consistency):
      observe(microStep(μ)) = transition(observe(μ), decide(observe(μ)))
    
    Theorem (substrate_step_realizes_decision_event):
      Any substrate step yields a decision event at the interface.
    
    Theorem (time_unit_law_substrate_invariant):
      Time-unit law is independent of substrate tag.
    """
    kind: SubstrateKind
    micro_step: Callable  # Micro → Micro
    observe: Callable     # Micro → S
    interface: DecisionProcess
    
    def verify_consistency(self, micro_state) -> bool:
        """
        Verify substrate consistency axiom:
        observe(microStep(μ)) = transition(observe(μ), decide(observe(μ)))
        """
        observed = self.observe(micro_state)
        action = self.interface.decide_fn(observed)
        expected = self.interface.transition_fn(observed, action)
        actual = self.observe(self.micro_step(micro_state))
        return actual == expected

# --- Theorems (verified by construction) ---

def verify_tick_increments_time(process: DecisionProcess, state: TimedState) -> bool:
    """tick_increments_time: (tick P x).time = x.time + 1"""
    result = tick(process, state)
    return result.time == state.time + 1

def verify_run_time_exact(process: DecisionProcess, n: int, state: TimedState) -> bool:
    """run_time_exact: (run P n x).time = x.time + n"""
    result = run(process, n, state)
    return result.time == state.time + n

def verify_decision_count_equals_elapsed_time(
    process: DecisionProcess, n: int, state: TimedState
) -> bool:
    """decision_count_equals_elapsed_time: len(trace) = elapsed time"""
    trace = decision_trace(process, n, state)
    result = run(process, n, state)
    elapsed = result.time - state.time
    return len(trace) == elapsed == n
