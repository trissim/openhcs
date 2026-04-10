import DecisionQuotient.Instances
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.EpsilonUtilityGap

namespace Paper4dFrontier

open Classical
open DecisionQuotient
open DecisionQuotient.Tractability
open DecisionQuotient.Tractability.CoarseApproximation

def singleBoolCoord : Fin 1 := ⟨0, by decide⟩

def allFalseBoolState : Fin 1 → Bool := fun _ => false

def allTrueBoolState : Fin 1 → Bool := fun _ => true

/-- A one-coordinate decision problem whose optimizer tracks the Boolean state. -/
def stateMatchingDecisionProblem (ε : ℝ) : DecisionProblem Bool (Fin 1 → Bool) where
  utility a s := if a = s singleBoolCoord then ε else 0

/-- A flat comparison problem with constant optimizer set. -/
def flatDecisionProblem : DecisionProblem Bool (Fin 1 → Bool) where
  utility _ _ := 0

theorem stateMatchingDecisionProblem_strict_false (hε : 0 < ε) :
    StrictOpt (stateMatchingDecisionProblem ε) false allFalseBoolState := by
  intro a ha
  cases a with
  | false => exact (False.elim (ha rfl))
  | true => simpa [stateMatchingDecisionProblem, allFalseBoolState, singleBoolCoord] using hε

theorem stateMatchingDecisionProblem_strict_true (hε : 0 < ε) :
    StrictOpt (stateMatchingDecisionProblem ε) true allTrueBoolState := by
  intro a ha
  cases a with
  | false => simpa [stateMatchingDecisionProblem, allTrueBoolState, singleBoolCoord] using hε
  | true => exact (False.elim (ha rfl))

theorem stateMatchingDecisionProblem_opt_false (hε : 0 < ε) :
    (stateMatchingDecisionProblem ε).Opt allFalseBoolState = {false} :=
  opt_eq_singleton_of_strict _ _ _ (stateMatchingDecisionProblem_strict_false hε)

theorem stateMatchingDecisionProblem_opt_true (hε : 0 < ε) :
    (stateMatchingDecisionProblem ε).Opt allTrueBoolState = {true} :=
  opt_eq_singleton_of_strict _ _ _ (stateMatchingDecisionProblem_strict_true hε)

theorem flatDecisionProblem_opt (s : Fin 1 → Bool) :
    flatDecisionProblem.Opt s = Set.univ := by
  ext a
  simp [flatDecisionProblem, DecisionProblem.Opt, DecisionProblem.isOptimal]

theorem flatDecisionProblem_irrelevant :
    flatDecisionProblem.isIrrelevant singleBoolCoord := by
  intro s s' _
  rw [flatDecisionProblem_opt, flatDecisionProblem_opt]

theorem stateMatchingDecisionProblem_relevant (hε : 0 < ε) :
    (stateMatchingDecisionProblem ε).isRelevant singleBoolCoord := by
  refine ⟨allFalseBoolState, allTrueBoolState, ?_, ?_⟩
  · intro j hj
    exfalso
    exact hj (by fin_cases j <;> rfl)
  · rw [stateMatchingDecisionProblem_opt_false hε, stateMatchingDecisionProblem_opt_true hε]
    intro hEq
    have hmem : false ∈ ({true} : Set Bool) := by
      have : false ∈ ({false} : Set Bool) := by simp
      rwa [hEq] at this
    simp at hmem

theorem stateMatchingDecisionProblem_uniformApprox_flat {ε : ℝ} (hε : 0 ≤ ε) :
    UniformUtilityApprox (stateMatchingDecisionProblem ε) flatDecisionProblem ε := by
  intro a s
  by_cases h : a = s singleBoolCoord
  · have hself : |ε| ≤ ε := by simpa [abs_of_nonneg hε]
    simpa [stateMatchingDecisionProblem, flatDecisionProblem, h] using hself
  · simpa [stateMatchingDecisionProblem, flatDecisionProblem, h] using hε

/-- A relevance witness survives a uniform approximation when the approximation
error stays below half the strict utility gap at both witness states. -/
theorem relevant_of_uniformApprox_of_strict_gap_witness
    {A S : Type*} [Fintype A] {n : ℕ} [CoordinateSpace S n]
    {exactDP approxDP : DecisionProblem A S} {δ : ℝ}
    (hApprox : UniformUtilityApprox exactDP approxDP δ)
    (hδ : 0 ≤ δ)
    {i : Fin n} {s s' : S} {a a' : A}
    (hAgree : ∀ j : Fin n, j ≠ i → CoordinateSpace.proj s j = CoordinateSpace.proj s' j)
    (hStrict : StrictOpt exactDP a s)
    (hStrict' : StrictOpt exactDP a' s')
    (hNe : a ≠ a')
    (hBound : δ < StrictUtilityGap exactDP a s / 2)
    (hBound' : δ < StrictUtilityGap exactDP a' s' / 2) :
    approxDP.isRelevant i := by
  refine ⟨s, s', hAgree, ?_⟩
  have hs : exactDP.Opt s = approxDP.Opt s :=
    uniform_approx_implies_opt_invariance exactDP approxDP δ hApprox s a hδ hStrict hBound
  have hs' : exactDP.Opt s' = approxDP.Opt s' :=
    uniform_approx_implies_opt_invariance exactDP approxDP δ hApprox s' a' hδ hStrict' hBound'
  have hsing : exactDP.Opt s = {a} := opt_eq_singleton_of_strict _ _ _ hStrict
  have hsing' : exactDP.Opt s' = {a'} := opt_eq_singleton_of_strict _ _ _ hStrict'
  rw [← hs, ← hs', hsing, hsing']
  intro hEq
  have hmem : a ∈ ({a'} : Set A) := by
    have : a ∈ ({a} : Set A) := by simp
    rwa [hEq] at this
  have hEq' : a = a' := by simpa using hmem
  exact hNe hEq'

/-- Uniform closeness alone does not control relevance: for every positive error
budget there are uniformly-close decision problems whose relevance judgment differs.
-/
theorem relevance_can_flip_under_arbitrarily_small_uniform_perturbation
    {ε : ℝ} (hε : 0 < ε) :
    ∃ exactDP approxDP : DecisionProblem Bool (Fin 1 → Bool),
      UniformUtilityApprox exactDP approxDP ε ∧
      exactDP.isRelevant singleBoolCoord ∧
      approxDP.isIrrelevant singleBoolCoord := by
  refine ⟨stateMatchingDecisionProblem ε, flatDecisionProblem, ?_, ?_, ?_⟩
  · exact stateMatchingDecisionProblem_uniformApprox_flat hε.le
  · exact stateMatchingDecisionProblem_relevant hε
  · exact flatDecisionProblem_irrelevant

end Paper4dFrontier
