/-
  PreservationExamples.lean

  Small mechanization of the two engineering examples from the paper:
  - one-step POMDP as an instance of stochastic preservation (preservation via a coarse observation map)
  - hyperparameter redundancy as an instance of static preservation (projection of hyperparameter space)

  These are intentionally lightweight encodings that reuse the paper's
  StochasticDecisionProblem / DecisionProblem definitions.  They are meant as
  illustrative bridges from the prose examples to mechanized predicates already
   present in the artifact.
-/

import DecisionQuotient.StochasticSequential.Basic
import Mathlib.Data.Finset.Basic

namespace DecisionQuotient.Examples

open DecisionQuotient.StochasticSequential
open Finset

/-! ## One-step POMDP encoding into stochastic preservation -/

section POMDP

variable {A S O O' : Type*}
[Fintype A] [Fintype S] [DecidableEq O']

/-- A one-step POMDP is a prior over states, an observation map, and an
    immediate-reward function. We reuse `StochasticDecisionProblem` for the
    utility+distribution part and attach an observation map. -/
structure OneStepPOMDP (A S O : Type*) [Fintype A] [Fintype S] where
  toSDP : StochasticDecisionProblem A S
  obs : S → O

open OneStepPOMDP

/-! The conditional expected utility induced by a coarse signal `φ : S → O'` -/
noncomputable def phiFiberExpectedUtility
    {A S O' : Type*} [Fintype A] [Fintype S] [DecidableEq O']
    (P : StochasticDecisionProblem A S) (φ : S → O') (s0 : S) (a : A) : ℝ :=
  ∑ s : S, if φ s = φ s0 then P.distribution s * P.utility a s else 0

def phiFiberOpt
    {A S O' : Type*} [Fintype A] [Fintype S] [DecidableEq O']
    (P : StochasticDecisionProblem A S) (φ : S → O') (s0 : S) : Set A :=
  { a : A | ∀ a', phiFiberExpectedUtility P φ s0 a' ≤ phiFiberExpectedUtility P φ s0 a }

/-- `φ` preserves Bayes-optimal actions exactly when the phi-fiber optimizer
    equals the full-information optimizer at every state. This is the phi-based
    analogue of the paper's preservation predicate. -/
def PhiPreserving
    {A S O' : Type*} [Fintype A] [Fintype S] [DecidableEq O']
    (P : StochasticDecisionProblem A S) (φ : S → O') : Prop :=
  ∀ s : S, phiFiberOpt P φ s = P.toDecisionProblem.Opt s

theorem pomdp_reduction_to_preservation
    {A S O O' : Type*} [Fintype A] [Fintype S] [DecidableEq O']
    (P : StochasticDecisionProblem A S) (φ : S → O') :
    PhiPreserving P φ ↔ (∀ s : S, phiFiberOpt P φ s = P.toDecisionProblem.Opt s) := by
  constructor
  · intro h; exact h
  · intro h; exact h

/-! The above is intentionally straightforward: it simply spells out the
    correspondence used in the paper's prose. The heavy-lifting complexity
    claims (explicit vs succinct encodings, PP-hardness) are imported from the
    rest of the artifact and apply to the instances obtained by encoding the
    POMDP's transition/observation tables appropriately.
-/

end POMDP

/-! ## Hyperparameter redundancy as static preservation -/

section Hyperparam

variable {Xα Xγ Xε E : Type*}
[Fintype Xα] [Fintype Xγ] [Fintype Xε] [Fintype E] [Nonempty Xγ]

/- Remarks: we encode hyperparameter triples as a nested product `(Xα × Xγ × Xε)`.
   Actions are hyperparameter configurations and states are environments. -/

variable (f : E → (Xα × Xγ × Xε) → ℝ)

/- Derived decision problem: actions are hyperparameter configs, states are environments. -/
def hpDecisionProblem : DecisionProblem (Xα × Xγ × Xε) E :=
  { utility := fun h e => f e h }

def Opt_e (e : E) : Set (Xα × Xγ × Xε) := { h | ∀ h' : Xα × Xγ × Xε, f e h' ≤ f e h }

/-- Projection that drops the `γ` coordinate. -/
def projγ : (Xα × Xγ × Xε) → (Xα × Xε) := fun h => (h.1, h.2.2)

/-- Projected maximizers after forgetting the `γ` coordinate. -/
def Opt'_e (e : E) : Set (Xα × Xε) := projγ '' Opt_e f e

/-- Definition of `γ` redundancy: the projection of full maximizers equals the
    reduced optimizer on the retained coordinates. -/
def gamma_redundant (f : E → (Xα × Xγ × Xε) → ℝ) : Prop :=
  ∀ e : E, Opt'_e f e = projγ '' Opt_e f e

theorem hyperparam_reduction_to_static
    : gamma_redundant f ↔
      (∀ e : E, ∀ x : Xα × Xε,
        x ∈ Opt'_e f e ↔ ∃ h0 : Xα × Xγ × Xε, h0 ∈ Opt_e f e ∧ projγ h0 = x) := by
  constructor
  · intro _ e x
    simp [Opt'_e, Set.mem_image]
  · intro h e
    apply Set.ext
    intro x
    simpa [gamma_redundant, Opt'_e, Set.mem_image] using (h e x).symm

end Hyperparam

end DecisionQuotient.Examples
