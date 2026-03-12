/-
  Paper 4: Explicit-State Membership Wrappers

  This file packages the counted-search procedures into abstract `InP`
  membership statements for explicit-state inputs. These are not TM-based
  complexity-class theorems; they use the repository's step-counting `InP`
  model from `PolynomialReduction.lean`.
-/

import DecisionQuotient.AlgorithmComplexity
import DecisionQuotient.PolynomialReduction
import DecisionQuotient.StochasticSequential.Computation

namespace DecisionQuotient

open DecisionQuotient.StochasticSequential

variable {n : ℕ}

structure StaticExplicitInput (A S : Type*) (n : ℕ)
    [Fintype S] [DecidableEq (Set A)] [CoordinateSpace S n] where
  problem : DecisionProblem A S
  infoSet : Finset (Fin n)
  stateBudget : ℕ
  state_bound : Fintype.card S ≤ stateBudget

noncomputable instance instSizeOfStaticExplicitInput
    {A S : Type*} {n : ℕ}
    [Fintype S] [DecidableEq (Set A)] [CoordinateSpace S n] :
    SizeOf (StaticExplicitInput A S n) where
  sizeOf q := q.stateBudget + sizeOf q.infoSet + 1

theorem staticExplicit_size_ge_state {A S : Type*} {n : ℕ}
    [Fintype S] [DecidableEq (Set A)] [CoordinateSpace S n]
    (q : StaticExplicitInput A S n) :
    q.stateBudget ≤ sizeOf q := by
  simp [SizeOf.sizeOf]
  omega

theorem static_sufficiency_inP_explicit {A S : Type*} {n : ℕ}
    [Fintype S] [DecidableEq (Set A)] [CoordinateSpace S n] :
    InP (fun q : StaticExplicitInput A S n => q.problem.isSufficient q.infoSet) := by
  use (fun q => countedStaticSufficiencySearch (n := n) q.problem q.infoSet), 1, 2
  constructor
  · intro q
    calc
      (countedStaticSufficiencySearch (n := n) q.problem q.infoSet).steps ≤ Fintype.card S * Fintype.card S :=
        countedStaticSufficiencySearch_steps (n := n) _ _
      _ ≤ q.stateBudget * q.stateBudget := Nat.mul_le_mul q.state_bound q.state_bound
      _ ≤ (sizeOf q) * (sizeOf q) := Nat.mul_le_mul (staticExplicit_size_ge_state q) (staticExplicit_size_ge_state q)
      _ = (sizeOf q) ^ 2 := by simp [pow_two]
      _ ≤ 1 * (sizeOf q) ^ 2 + 1 := by omega
  · intro q
    exact countedStaticSufficiencySearch_spec (n := n) _ _

theorem static_anchor_inP_explicit {A S : Type*} {n : ℕ}
    [Fintype S] [DecidableEq (Set A)] [CoordinateSpace S n] :
    InP (fun q : StaticExplicitInput A S n => q.problem.anchorSufficient q.infoSet) := by
  use (fun q => countedAnchorSufficientSearch (n := n) q.problem q.infoSet), 1, 1
  constructor
  · intro q
    calc
      (countedAnchorSufficientSearch (n := n) q.problem q.infoSet).steps ≤ Fintype.card S :=
        countedAnchorSufficientSearch_steps (n := n) _ _
      _ ≤ q.stateBudget := q.state_bound
      _ ≤ sizeOf q := staticExplicit_size_ge_state q
      _ ≤ 1 * (sizeOf q) ^ 1 + 1 := by simp
  · intro q
    exact countedAnchorSufficientSearch_spec (n := n) _ _

structure StaticMinimumExplicitInput (A S : Type*) (n : ℕ)
    [CoordinateSpace S n] where
  problem : DecisionProblem A S
  bound : ℕ
  subsetBudget : ℕ
  subset_bound : 2 ^ n ≤ subsetBudget

noncomputable instance instSizeOfStaticMinimumExplicitInput
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] :
    SizeOf (StaticMinimumExplicitInput A S n) where
  sizeOf q := q.subsetBudget + q.bound + 1

theorem staticMinimumExplicit_size_ge_subset
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (q : StaticMinimumExplicitInput A S n) :
    q.subsetBudget ≤ sizeOf q := by
  simp [SizeOf.sizeOf]
  omega

theorem static_minimum_inP_explicit {A S : Type*} {n : ℕ}
    [CoordinateSpace S n] :
    InP (fun q : StaticMinimumExplicitInput A S n =>
      ∃ I : Finset (Fin n), I.card ≤ q.bound ∧ q.problem.isSufficient I) := by
  use (fun q => countedMinimumSufficientSearch (n := n) q.problem q.bound), 1, 1
  constructor
  · intro q
    calc
      (countedMinimumSufficientSearch (n := n) q.problem q.bound).steps ≤ 2 ^ n :=
        countedMinimumSufficientSearch_steps (n := n) _ _
      _ ≤ q.subsetBudget := q.subset_bound
      _ ≤ sizeOf q := staticMinimumExplicit_size_ge_subset q
      _ ≤ 1 * (sizeOf q) ^ 1 + 1 := by simp
  · intro q
    exact countedMinimumSufficientSearch_spec (n := n) _ _

structure StochasticMinimumExplicitInput
    (A S : Type*) (n : ℕ)
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] where
  problem : StochasticDecisionProblem A S
  bound : ℕ
  subsetBudget : ℕ
  subset_bound : 2 ^ n ≤ subsetBudget

noncomputable instance instSizeOfStochasticMinimumExplicitInput
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] :
    SizeOf (StochasticMinimumExplicitInput A S n) where
  sizeOf q := q.subsetBudget + q.bound + 1

theorem stochasticMinimumExplicit_size_ge_subset
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n]
    (q : StochasticMinimumExplicitInput A S n) :
    q.subsetBudget ≤ sizeOf q := by
  simp [SizeOf.sizeOf]
  omega

theorem stochastic_minimum_inP_explicit
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] :
    InP (fun q : StochasticMinimumExplicitInput A S n =>
      StochasticMinimumSufficiencyCheck q.problem q.bound) := by
  use (fun q => countedStochasticMinimumSearch q.problem q.bound), 1, 1
  constructor
  · intro q
    calc
      (countedStochasticMinimumSearch q.problem q.bound).steps ≤ 2 ^ n :=
        countedStochasticMinimumSearch_steps _ _
      _ ≤ q.subsetBudget := q.subset_bound
      _ ≤ sizeOf q := stochasticMinimumExplicit_size_ge_subset q
      _ ≤ 1 * (sizeOf q) ^ 1 + 1 := by simp
  · intro q
    exact countedStochasticMinimumSearch_spec _ _

structure SequentialMinimumExplicitInput
    (A S O : Type*) (n : ℕ)
    [Fintype A] [Fintype S] [Fintype O] [DecidableEq A] [CoordinateSpace S n] where
  problem : SequentialDecisionProblem A S O
  bound : ℕ
  subsetBudget : ℕ
  subset_bound : 2 ^ n ≤ subsetBudget

noncomputable instance instSizeOfSequentialMinimumExplicitInput
    {A S O : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [Fintype O] [DecidableEq A] [CoordinateSpace S n] :
    SizeOf (SequentialMinimumExplicitInput A S O n) where
  sizeOf q := q.subsetBudget + q.bound + 1

theorem sequentialMinimumExplicit_size_ge_subset
    {A S O : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [Fintype O] [DecidableEq A] [CoordinateSpace S n]
    (q : SequentialMinimumExplicitInput A S O n) :
    q.subsetBudget ≤ sizeOf q := by
  simp [SizeOf.sizeOf]
  omega

theorem sequential_minimum_inP_explicit
    {A S O : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [Fintype O] [DecidableEq A] [CoordinateSpace S n] :
    InP (fun q : SequentialMinimumExplicitInput A S O n =>
      SequentialMinimumSufficiencyCheck q.problem q.bound) := by
  use (fun q => countedSequentialMinimumSearch q.problem q.bound), 1, 1
  constructor
  · intro q
    calc
      (countedSequentialMinimumSearch q.problem q.bound).steps ≤ 2 ^ n :=
        countedSequentialMinimumSearch_steps _ _
      _ ≤ q.subsetBudget := q.subset_bound
      _ ≤ sizeOf q := sequentialMinimumExplicit_size_ge_subset q
      _ ≤ 1 * (sizeOf q) ^ 1 + 1 := by simp
  · intro q
    exact countedSequentialMinimumSearch_spec _ _

theorem explicit_state_inP_summary :
    (∀ {A S : Type*} {n : ℕ} [Fintype S] [DecidableEq (Set A)] [CoordinateSpace S n],
      InP (fun q : StaticExplicitInput A S n => q.problem.isSufficient q.infoSet) ∧
      InP (fun q : StaticExplicitInput A S n => q.problem.anchorSufficient q.infoSet) ∧
      InP (fun q : StaticMinimumExplicitInput A S n =>
        ∃ I : Finset (Fin n), I.card ≤ q.bound ∧ q.problem.isSufficient I)) ∧
    (∀ {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n],
      InP (fun q : StochasticExplicitInput A S n => StochasticSufficient q.problem q.infoSet) ∧
      InP (fun q : StochasticExplicitInput A S n => StochasticAnchorSufficiencyCheck q.problem q.infoSet) ∧
      InP (fun q : StochasticMinimumExplicitInput A S n => StochasticMinimumSufficiencyCheck q.problem q.bound)) ∧
    (∀ {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A] [CoordinateSpace S n],
      InP (fun q : SequentialExplicitInput A S O n => SequentialSufficient q.problem q.infoSet) ∧
      InP (fun q : SequentialExplicitInput A S O n => SequentialAnchorSufficiencyCheck q.problem q.infoSet) ∧
      InP (fun q : SequentialMinimumExplicitInput A S O n => SequentialMinimumSufficiencyCheck q.problem q.bound)) := by
  refine ⟨?_, ?_, ?_⟩
  · intro A S n _ _ _
    exact ⟨static_sufficiency_inP_explicit (A := A) (S := S) (n := n),
      static_anchor_inP_explicit (A := A) (S := S) (n := n),
      static_minimum_inP_explicit (A := A) (S := S) (n := n)⟩
  · intro A S n _ _ _ _
    exact ⟨stochastic_sufficiency_inP_explicit (A := A) (S := S) (n := n),
      stochastic_anchor_inP_explicit (A := A) (S := S) (n := n),
      stochastic_minimum_inP_explicit (A := A) (S := S) (n := n)⟩
  · intro A S O n _ _ _ _ _
    exact ⟨sequential_sufficiency_inP_explicit (A := A) (S := S) (O := O) (n := n),
      sequential_anchor_inP_explicit (A := A) (S := S) (O := O) (n := n),
      sequential_minimum_inP_explicit (A := A) (S := S) (O := O) (n := n)⟩

end DecisionQuotient
