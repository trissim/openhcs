/- 
# Heisenberg Uncertainty (abstract bridge)

## Dependency Graph
- **Nontrivial source:** physical quantum theory (Heisenberg uncertainty principle)
- **This module:** introduces a lightweight assumption schema plus a derived
  general witness theorem for non-constant optimizer sets.
- **Used by:** (optional) `Information.lean`, `PhysicalHardness.lean` as a
  physics grounding for the claim that nontrivial uncertainty exists.

## Role
This file separates:
1. a general, derived nontrivial-uncertainty witness, and
2. a physical-to-general bridge theorem under an explicit physical assumption.

## Triviality Level
NONTRIVIAL — introduces a bridge from physical assumptions to core decision
uncertainty statements while keeping the general theorem derived.
-/

import DecisionQuotient.Physics.ClaimTransport

namespace DecisionQuotient
namespace Physics
namespace Uncertainty

open ClaimTransport

/-- A small explicit decision problem whose optimizer set is state-dependent. -/
def binaryIdentityProblem : DecisionProblem Bool Bool where
  utility := fun a s => if a = s then 1 else 0

lemma binaryIdentityProblem_opt_true :
    binaryIdentityProblem.Opt true = ({true} : Set Bool) := by
  ext a
  constructor
  · intro hopt
    cases a with
    | true => simp
    | false =>
      have h1 := hopt true
      simp [binaryIdentityProblem] at h1
      linarith
  · intro ha
    have ha' : a = true := by simpa using ha
    subst ha'
    intro a'
    cases a' <;> simp [binaryIdentityProblem]

lemma binaryIdentityProblem_opt_false :
    binaryIdentityProblem.Opt false = ({false} : Set Bool) := by
  ext a
  constructor
  · intro hopt
    cases a with
    | false => simp
    | true =>
      have h1 := hopt false
      simp [binaryIdentityProblem] at h1
      linarith
  · intro ha
    have ha' : a = false := by simpa using ha
    subst ha'
    intro a'
    cases a' <;> simp [binaryIdentityProblem]

/-- General (derived) existence of a decision problem with non-constant `Opt`. -/
theorem exists_decision_problem_with_nontrivial_opt :
    ∃ (A : Type) (S : Type) (d : DecisionProblem A S) (s s' : S),
      d.Opt s ≠ d.Opt s' := by
  refine ⟨Bool, Bool, binaryIdentityProblem, true, false, ?_⟩
  intro hEq
  have hmem : true ∈ binaryIdentityProblem.Opt true := by
    rw [binaryIdentityProblem_opt_true]
    simp
  have hmem' : true ∈ binaryIdentityProblem.Opt false := by simpa [hEq] using hmem
  rw [binaryIdentityProblem_opt_false] at hmem'
  simp at hmem'

/-- Physical assumption schema: some encoded physical instance has non-constant `Opt`. -/
def PhysicalNontrivialOptAssumption : Prop :=
  ∃ (PInst : Type) (A : Type) (S : Type)
    (E : ClaimTransport.PhysicalEncoding PInst A S)
    (p : PInst) (s s' : S),
    (E.encode p).Opt s ≠ (E.encode p).Opt s'

/-- Physical-to-general bridge under an explicit physical assumption. -/
theorem exists_decision_problem_with_nontrivial_opt_of_physical
    (hPhys : PhysicalNontrivialOptAssumption) :
    ∃ (A : Type) (S : Type) (d : DecisionProblem A S) (s s' : S),
      d.Opt s ≠ d.Opt s' := by
  rcases hPhys with ⟨_PInst, A, S, E, p, s, s', hne⟩
  exact ⟨A, S, E.encode p, s, s', hne⟩

end Uncertainty
end Physics
end DecisionQuotient
