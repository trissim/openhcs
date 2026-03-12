/-
  Paper 4b: Stochastic and Sequential Regimes

  Economics.lean - Gap as price signal; decision resolution value

  The decision gap is the price signal for observation:
  - gap > 0: prior uniquely resolves the decision; observation wastes resources
  - gap = 0: prior is indifferent (tie); pay to observe if I is sufficient

  The Bayesian picture: truth and memory both drift over time. The gap measures
  how well the prior (memory) still resolves the decision (truth). When entropy
  has degraded the gap to zero, observation is rational. A sufficient coordinate
  set I is worth the price when it restores resolution that the prior has lost.

  Key theorems:
  1. shouldObserve_iff_gap_zero: gap = 0 ↔ prior indifferent ↔ rational to observe
  2. no_observation_when_prior_sufficient: gap > 0 → skip observation
  3. observation_contains_information: if prior fails and I resolves, I ≠ ∅
  4. observation_refines_prior: I-sufficient + prior-insufficient → I gives strictly
     better resolution (some fiber-optimal set differs from the global non-singleton)
-/

import DecisionQuotient.StochasticSequential.Basic
import DecisionQuotient.StochasticSequential.Information
import Mathlib.Data.Real.Basic

namespace DecisionQuotient.StochasticSequential

open Classical
open StochAction

/-! ## Observation Rationality

The gap between the two actions' expected utilities is the price signal.
A positive gap means the prior already resolves the decision; no observation needed.
A zero gap means the prior is indifferent; the controller must observe to act rationally.
-/

/-- Observation is rational when the prior fails to resolve the decision.
    Equivalently (by `shouldObserve_iff_gap_zero`): the decision gap is zero. -/
def shouldObserve {S : Type*} [Fintype S] {n : ℕ} [CoordinateSpace S n]
    (P : StochasticDecisionProblem StochAction S) : Prop :=
  ¬StochasticSufficient P ∅

/-- The decision gap is the price signal for observation: rational to observe iff gap = 0.

    - gap > 0: prior has a uniquely dominant action → don't observe (already resolved)
    - gap = 0: prior is tied → pay to observe (need external information to act) -/
theorem shouldObserve_iff_gap_zero {S : Type*} [Fintype S] [Nonempty S] {n : ℕ}
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem StochAction S) :
    shouldObserve P ↔ decisionGap P = 0 := by
  unfold shouldObserve
  rw [stochastic_sufficient_iff_gap_pos]
  constructor
  · intro h
    exact le_antisymm (not_lt.mp h) (decisionGap_nonneg P)
  · intro h hpos
    linarith

/-- When the prior is sufficient (gap > 0), no observation is needed.
    The controller already has a unique optimal action under the prior; observing wastes resources. -/
theorem no_observation_when_prior_sufficient {S : Type*} [Fintype S] {n : ℕ}
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem StochAction S) :
    StochasticSufficient P ∅ → ¬shouldObserve P := by
  intro h
  exact fun hObs => hObs h

/-! ## Decision Resolution Value

When the prior is indifferent (gap = 0), a sufficient coordinate set I provides
decision resolution that the prior alone cannot supply. The "value" of I is this
resolution — it converts a tie into a unique optimal action on every fiber.
-/

/-- When observation is needed and I resolves the decision:
    I must be nonempty — it contains genuinely decision-relevant information.

    If I were empty, StochasticSufficient P I = StochasticSufficient P ∅, which
    contradicts shouldObserve. So any resolution-providing I has positive content. -/
theorem observation_contains_information {S : Type*} [Fintype S] [Nonempty S] {n : ℕ}
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem StochAction S) (I : Finset (Fin n)) :
    shouldObserve P → StochasticSufficient P I → I.Nonempty := by
  intro hObs hSuff
  rcases Finset.eq_empty_or_nonempty I with hEmpty | hNonempty
  · exact absurd (hEmpty ▸ hSuff) hObs
  · exact hNonempty

/-- When observation is needed and I is sufficient:
    I strictly refines the prior — some fiber-optimal set differs from the global
    non-singleton optimal set.

    Proof: shouldObserve means stochasticOpt is not a singleton ({accept,reject} tie).
    StochasticSufficient P I means every fiber gives a singleton {a}. Since {a} ≠ {accept,reject},
    every fiber already refines the prior. We exhibit any s₀ as the witness. -/
theorem observation_refines_prior {S : Type*} [Fintype S] [Nonempty S] {n : ℕ}
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem StochAction S) (I : Finset (Fin n)) :
    shouldObserve P → StochasticSufficient P I →
    ∃ s₀ : S, fiberOpt P I s₀ ≠ P.stochasticOpt := by
  intro hObs hSuff
  obtain ⟨s₀⟩ := ‹Nonempty S›
  refine ⟨s₀, fun heq => ?_⟩
  obtain ⟨a, ha⟩ := hSuff s₀
  -- ha : fiberOpt P I s₀ = {a}, heq : fiberOpt P I s₀ = P.stochasticOpt
  -- so stochasticOpt = {a}, meaning ∅ is also sufficient — contradicts shouldObserve
  exact hObs ((stochasticSufficient_empty_iff P).mpr ⟨a, ha ▸ heq.symm⟩)

/-! ## Observation Threshold

The gap decays over time as entropy corrupts memory. The observation threshold
captures the budget-constrained decision: observe when gap ≤ tolerance.
At tolerance = 0, this collapses to `shouldObserve`.
-/

/-- Whether the gap has fallen below a tolerance threshold -/
def belowObservationThreshold {S : Type*} [Fintype S]
    (P : StochasticDecisionProblem StochAction S) (tolerance : ℝ) : Prop :=
  decisionGap P ≤ tolerance

/-- At zero tolerance, the threshold coincides with `shouldObserve` -/
theorem zero_tolerance_iff_should_observe {S : Type*} [Fintype S] [Nonempty S] {n : ℕ}
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem StochAction S) :
    belowObservationThreshold P 0 ↔ shouldObserve P := by
  unfold belowObservationThreshold
  rw [shouldObserve_iff_gap_zero]
  exact ⟨fun h => le_antisymm h (decisionGap_nonneg P), fun h => le_of_eq h⟩

/-- Positive tolerance admits observation even when gap > 0 (budget-constrained) -/
theorem positive_tolerance_relaxes_threshold {S : Type*} [Fintype S]
    (P : StochasticDecisionProblem StochAction S)
    (ε : ℝ) (hε : 0 < ε) :
    belowObservationThreshold P 0 → belowObservationThreshold P ε := by
  intro h
  unfold belowObservationThreshold at *
  linarith

/-! ## Resolution Value

The resolution value of observing I is 1 when I converts a prior indifference
into a resolved decision (I sufficient, ∅ not sufficient), else 0.
-/

/-- Resolution value: 1 if I resolves what the prior cannot, else 0 -/
noncomputable def resolutionValue {S : Type*} [Fintype S] {n : ℕ} [CoordinateSpace S n]
    (P : StochasticDecisionProblem StochAction S) (I : Finset (Fin n)) : ℝ :=
  if StochasticSufficient P I ∧ ¬StochasticSufficient P ∅ then 1 else 0

/-- Resolution value is 1 exactly when I resolves what the prior cannot -/
theorem resolutionValue_eq_one_iff {S : Type*} [Fintype S] {n : ℕ} [CoordinateSpace S n]
    (P : StochasticDecisionProblem StochAction S) (I : Finset (Fin n)) :
    resolutionValue P I = 1 ↔ StochasticSufficient P I ∧ ¬StochasticSufficient P ∅ := by
  unfold resolutionValue
  constructor
  · intro h
    by_contra hNot
    rw [if_neg hNot] at h
    norm_num at h
  · intro h
    rw [if_pos h]

/-- Resolution value is nonneg -/
theorem resolutionValue_nonneg {S : Type*} [Fintype S] {n : ℕ} [CoordinateSpace S n]
    (P : StochasticDecisionProblem StochAction S) (I : Finset (Fin n)) :
    0 ≤ resolutionValue P I := by
  unfold resolutionValue
  split_ifs <;> norm_num

/-! ## Observation Cost

The cost of observing I is proportional to the number of coordinates inspected.
Observing more coordinates costs more but can only increase resolution.
-/

/-- Observation cost: proportional to coordinate set cardinality -/
noncomputable def observationCost {n : ℕ} (I : Finset (Fin n)) : ℝ := I.card

/-- Cost is monotone: observing more coordinates costs more -/
theorem observationCost_monotone {n : ℕ} (I J : Finset (Fin n)) (hIJ : I ⊆ J) :
    observationCost I ≤ observationCost J :=
  Nat.cast_le.mpr (Finset.card_le_card hIJ)

/-- Empty observation has zero cost -/
theorem observationCost_empty {n : ℕ} : observationCost (∅ : Finset (Fin n)) = 0 := by
  simp [observationCost]

end DecisionQuotient.StochasticSequential
