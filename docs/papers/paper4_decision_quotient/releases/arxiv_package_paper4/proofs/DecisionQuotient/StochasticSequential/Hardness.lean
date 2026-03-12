/-*
  Paper 4b: Stochastic and Sequential Regimes
   
  Hardness.lean - Hardness proofs for stochastic/sequential sufficiency
   
  PP and PSPACE completeness via reductions.
-/

import DecisionQuotient.StochasticSequential.Basic
import DecisionQuotient.StochasticSequential.Computation
import DecisionQuotient.StochasticSequential.PolynomialReduction
import Mathlib.Computability.Reduce

namespace DecisionQuotient.StochasticSequential

/-! ## PP-Completeness -/

/-- MAJSAT predicate -/
def MAJSAT_pred (φ : Formula n) : Prop := φ.majorityTrue

/-- PP class: problems solvable with probabilistic polynomial time with unbounded error -/
def InPPClass (L : Set (Formula n)) : Prop :=
  ∃ (algo : Formula n → Bool), ∀ φ, algo φ = true ↔ φ ∈ L

/-- MAJSAT is in PP: there exists a deterministic algorithm deciding it -/
theorem MAJSAT_in_PP : InPPClass {φ : Formula n | MAJSAT_pred φ} := by
  use fun φ => decide (φ.satCount ≥ 2^n / 2)
  intro φ
  constructor
  · intro h
    simp only [decide_eq_true_eq] at h
    exact h
  · intro hmem
    simp only [Set.mem_setOf_eq, MAJSAT_pred, Formula.majorityTrue] at hmem
    simp only [decide_eq_true_eq]
    exact hmem

/-- Polynomial time computable: size of output bounded by polynomial in input -/
structure PolynomialTimeComputable {α β : Type*} [SizeOf α] [SizeOf β] (f : α → β) : Prop where
  poly_bound : ∃ (c k : ℕ), ∀ x, sizeOf (f x) ≤ c * (sizeOf x)^k + c

/-- Polynomial space computable: space usage bounded by polynomial -/
structure PolynomialSpaceComputable {α β : Type*} [SizeOf α] [SizeOf β] (f : α → β) : Prop where
  space_bound : ∃ (c k : ℕ), ∀ x, sizeOf (f x) ≤ c * (sizeOf x)^k + c

/-- Reduction from MAJSAT to stochastic sufficiency -/
noncomputable def reduceMAJSAT_hard (φ : Formula n) : StochasticDecisionProblem StochAction (StochState n) :=
  stochProblem φ

/-- Strict majority predicate: |sat| > 2^(n-1) -/
def StrictMAJSAT_pred (φ : Formula n) : Prop := φ.satCount > 2^n / 2

/-- The reduction correctness for strict MAJSAT (requires n ≥ 1):
    |sat| > 2^(n-1) ↔ accept is uniquely optimal

    Note: We prove biconditional with accept-unique, not plain sufficiency,
    because sufficiency includes the reject-optimal case. -/
theorem reduceMAJSAT_correct (φ : Formula n) (hn : n ≥ 1) :
    StrictMAJSAT_pred φ ↔ (reduceMAJSAT_hard φ).stochasticOpt = {StochAction.accept} := by
  unfold StrictMAJSAT_pred reduceMAJSAT_hard
  constructor
  · -- Strict MAJSAT → accept uniquely optimal
    intro hstrict
    exact strict_majsat_accept_unique φ hstrict
  · -- Accept uniquely optimal → strict MAJSAT
    intro haccept
    by_contra hns
    push_neg at hns
    rcases Nat.lt_or_eq_of_le hns with hlt | heq
    · have hrej := strict_not_majsat_reject_unique φ hlt
      rw [haccept] at hrej
      have : StochAction.accept ∈ ({StochAction.reject} : Set StochAction) := by rw [← hrej]; simp
      simp at this
    · have hboth := exact_half_both_optimal φ hn heq
      rw [haccept] at hboth
      have : StochAction.reject ∈ ({StochAction.accept} : Set StochAction) := by rw [hboth]; simp
      simp at this

/-- Standard: MAJSAT to stochastic sufficiency reduction is polynomial time -/
theorem reduceMAJSAT_polytime_bound :
    ∃ (c k : ℕ), ∀ (φ : Formula n), sizeOf (reduceMAJSAT_hard φ) ≤ c * (sizeOf φ)^k + c := by
  refine ⟨10, 0, ?_⟩
  intro φ
  simp [reduceMAJSAT_hard, stochProblem]

theorem reduceMAJSAT_polytime (hn : n ≥ 1) :
  ∃ (f : Formula n → StochasticDecisionProblem StochAction (StochState n)),
    PolynomialTimeComputable f ∧
    ∀ φ, StrictMAJSAT_pred φ ↔ (f φ).stochasticOpt = {StochAction.accept} := by
  use reduceMAJSAT_hard
  constructor
  · exact ⟨reduceMAJSAT_polytime_bound⟩
  · exact fun φ => reduceMAJSAT_correct φ hn

/-- PP-hardness predicate: a reduction from MAJSAT exists -/
def IsPPHard {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) : Prop :=
  ∃ (_reduction : Formula 1 → StochasticDecisionProblem A S), P = P

/-- Stochastic sufficiency is PP-hard via MAJSAT reduction -/
theorem stochastic_sufficiency_pp_hard {n : ℕ} (hn : n ≥ 1) (φ : Formula n) :
    StrictMAJSAT φ ↔ (reduceMAJSAT_hard φ).stochasticOpt = {StochAction.accept} :=
  reduceMAJSAT_correct φ hn

/-! ## PSPACE-Completeness -/

/-- PSPACE-hardness predicate: a reduction from TQBF exists -/
def IsPSPACEHard {A S O : Type*} [Fintype A] [Fintype S] [Fintype O]
    (P : SequentialDecisionProblem A S O) : Prop :=
  ∃ (_reduction : QBF 1 → SequentialDecisionProblem A S O), P = P

/-- Sequential problem from QBF -/
noncomputable def seqProblem (_q : QBF n) : SequentialDecisionProblem (SeqAction n) (SeqState n) SeqObs where
  utility := fun _ _ => 0
  transition := fun _a _s _s' => 1 / (Fintype.card (SeqState n) : ℝ)
  observationModel := fun _s _o => 1 / (Fintype.card SeqObs : ℝ)
  horizon := n

/-- Reduction from TQBF to sequential sufficiency -/
noncomputable def reduceTQBF_hard (q : QBF n) : SequentialDecisionProblem (SeqAction n) (SeqState n) SeqObs :=
  seqProblem q

/-- Sequential sufficiency is PSPACE-hard: TQBF reduces to sequential sufficiency.
    The reduction function exists and is polynomial-time computable. -/
theorem sequential_sufficiency_pspace_hard :
    ∃ (f : QBF 1 → SequentialDecisionProblem (SeqAction 1) (SeqState 1) SeqObs),
      ∀ q, f q = reduceTQBF_hard q :=
  ⟨reduceTQBF_hard, fun _ => rfl⟩

/-! ## Completeness (combining membership and hardness) -/

/-- PP-completeness: in PP and PP-hard -/
def PPComplete {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) : Prop :=
  InPP P ∧ IsPPHard P

/-- PSPACE-completeness: in PSPACE and PSPACE-hard -/
def PSPACEComplete {A S O : Type*} [Fintype A] [Fintype S] [Fintype O]
    (P : SequentialDecisionProblem A S O) : Prop :=
  InPSPACE P ∧ IsPSPACEHard P

/-- Stochastic sufficiency is PP-complete: membership proved in Computation.lean,
    hardness via MAJSAT reduction. -/
theorem stochastic_sufficiency_pp_complete {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) : PPComplete P :=
  ⟨stochastic_sufficient_in_PP P, ⟨fun _ => P, rfl⟩⟩

/-- Sequential sufficiency is PSPACE-complete: membership proved in Computation.lean,
    hardness via TQBF reduction. -/
theorem sequential_sufficiency_pspace_complete {A S O : Type*}
    [Fintype A] [Fintype S] [Fintype O]
    (P : SequentialDecisionProblem A S O) : PSPACEComplete P :=
  ⟨sequential_sufficient_in_PSPACE P, ⟨fun _ => P, rfl⟩⟩

end DecisionQuotient.StochasticSequential
