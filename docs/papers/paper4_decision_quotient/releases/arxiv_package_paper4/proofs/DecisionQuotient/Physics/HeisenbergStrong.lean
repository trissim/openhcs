import DecisionQuotient.Physics.ClaimTransport
import DecisionQuotient.Physics.Uncertainty

/-!
# Strong Heisenberg Binding (mechanized)

## Dependency Graph
- **Nontrivial source:** Heisenberg uncertainty principle (physics)
- **This module:** provides a mechanized, explicit bridge asserting that a
  single physical instance can be compatible with multiple interface states
  whose decision-optima differ. This is the strongest *mechanized* binding
  between quantum-style measurement uncertainty and the decision problem
  semantics that we can add without formalizing quantum mechanics.

## Role
This module introduces a small, explicit interface for noisy physical
encodings and an explicit assumption predicate `HeisenbergStrongBinding` that asserts the
existence of a physically encoded instance `p` and two interface states
`s`, `s'` such that:

  - the physical instance `p` is compatible with both `s` and `s'` (a
    mechanized "observation ambiguity"), and
  - the encoded core decision problem assigns different optimizer sets to
    `s` and `s'` (i.e. `Opt s ≠ Opt s'`).

In short: a single physical instance can realize two interface states that
lead to different decisions. This is the clearest mechanized statement of
"decisions require uncertainty" we can add without committing to any
particular physical derivation of Heisenberg.

## Triviality Level
NONTRIVIAL — this is an assumption-typed physics bridge that directly connects
substrate ambiguity to decision-nontriviality. The closest nontrivial
connector in the repo is `DecisionQuotient.Physics.ClaimTransport`
(physical → core encodings).
-/

namespace DecisionQuotient
namespace Physics
namespace HeisenbergStrong

open ClaimTransport

/-- A noisy physical encoding bundles a core physical encoder together with
    a predicate `compatible p s` modelling that physical instance `p` may
    be observed (or interpreted) as interface state `s`. -/
structure NoisyPhysicalEncoding (PInst : Type) (A : Type) (S : Type) where
  coreEnc : ClaimTransport.PhysicalEncoding PInst A S
  compatible : PInst → S → Prop

/-- Strong mechanized Heisenberg-style assumption:

    There exists a noisy physical encoding `E` and a physical instance `p`
    such that `p` is compatible with two distinct interface states `s` and
    `s'`, and those states induce different optimizer sets in the encoded
    `DecisionProblem`. This expresses that a single physical reality may
    correspond to multiple interface descriptions that lead to distinct
    decisions. -/
def HeisenbergStrongBinding : Prop :=
  ∃ (PInst : Type) (A : Type) (S : Type)
    (E : NoisyPhysicalEncoding PInst A S)
    (p : PInst) (s s' : S),
    E.compatible p s ∧ E.compatible p s' ∧ s ≠ s' ∧ (E.coreEnc.encode p).Opt s ≠ (E.coreEnc.encode p).Opt s'

/-- The strong binding implies the existence of a core `DecisionProblem`
    with non-constant `Opt` (a direct, machine-checkable witness). -/
theorem strong_binding_implies_core_nontrivial
    (hBind : HeisenbergStrongBinding) :
    ∃ (A : Type) (S : Type) (d : DecisionProblem A S) (s s' : S),
      d.Opt s ≠ d.Opt s' := by
  rcases hBind with ⟨PInst, A, S, E, p, s, s', hcomp⟩
  rcases hcomp with ⟨_hcs, _hcs', _hne, hoptdiff⟩
  exact ⟨A, S, E.coreEnc.encode p, s, s', hoptdiff⟩

/-- The strong binding also provides a witness in the original (core)
    physical-encoding interface used elsewhere in the repo. -/
theorem strong_binding_yields_core_encoding_witness
    (hBind : HeisenbergStrongBinding) :
    Uncertainty.PhysicalNontrivialOptAssumption := by
  rcases hBind with ⟨PInst, A, S, E, p, s, s', hcomp⟩
  rcases hcomp with ⟨_hcs, _hcs', _hne, hoptdiff⟩
  exact ⟨PInst, A, S, E.coreEnc, p, s, s', hoptdiff⟩

/-- The strong binding implies the physical nontrivial-`Opt` assumption
    used by the uncertainty bridge module. -/
theorem strong_binding_implies_physical_nontrivial_opt_assumption
    (hBind : HeisenbergStrongBinding) :
    Uncertainty.PhysicalNontrivialOptAssumption :=
  strong_binding_yields_core_encoding_witness hBind

/-- Composed bridge:
    Heisenberg-strong binding yields a core decision problem
    with non-constant optimizer map via `Physics.Uncertainty`. -/
theorem strong_binding_implies_nontrivial_opt_via_uncertainty
    (hBind : HeisenbergStrongBinding) :
    ∃ (A : Type) (S : Type) (d : DecisionProblem A S) (s s' : S),
      d.Opt s ≠ d.Opt s' := by
  exact Uncertainty.exists_decision_problem_with_nontrivial_opt_of_physical
    (strong_binding_implies_physical_nontrivial_opt_assumption hBind)

end HeisenbergStrong
end Physics
end DecisionQuotient
