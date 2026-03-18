from dataclasses import dataclass
from typing import Set

"""
Heisenberg Strong Binding — measurement uncertainty bridge.
Direct translation of Physics/HeisenbergStrong.lean.

Key theorem: A single physical instance can realize two interface states
with different optimal actions → decisions require uncertainty.
"""

@dataclass(frozen=True)
class NoisyPhysicalEncoding:
    """
    From HeisenbergStrong.lean::NoisyPhysicalEncoding.
    A physical encoder with compatibility predicate:
    physical instance p may be observed as interface state s.
    """
    compatible_states: frozenset  # Set of compatible (p, s) pairs

@dataclass(frozen=True)
class HeisenbergBinding:
    """
    HeisenbergStrongBinding: ∃ p, s, s' such that
    - p compatible with s and s'
    - s ≠ s'
    - Opt(s) ≠ Opt(s')
    
    A single physical reality maps to distinct decisions.
    """
    physical_instance: object
    state1: object
    state2: object
    opt1: frozenset  # Optimal actions at state1
    opt2: frozenset  # Optimal actions at state2
    
    @property
    def is_valid(self) -> bool:
        """
        From HeisenbergStrongBinding:
        States are distinct and have different optimal actions.
        """
        return self.state1 != self.state2 and self.opt1 != self.opt2
    
    def implies_core_nontrivial(self) -> bool:
        """
        From strong_binding_implies_core_nontrivial:
        ∃ d, s, s', d.Opt(s) ≠ d.Opt(s').
        """
        return self.is_valid

def check_heisenberg_binding(
    utility_fn,
    state1,
    state2,
    actions
) -> bool:
    """
    Check if two states could constitute a Heisenberg binding:
    different optimal action sets.
    """
    opt1 = {a for a in actions if utility_fn(a, state1) >= max(utility_fn(a2, state1) for a2 in actions)}
    opt2 = {a for a in actions if utility_fn(a, state2) >= max(utility_fn(a2, state2) for a2 in actions)}
    return opt1 != opt2
