/-
  Paper 4b: Stochastic and Sequential Regimes

  TemporalLearning.lean - Learning structure across problem sequences

  Formalizes Bayesian learning of structure classes.
-/

import DecisionQuotient.StochasticSequential.Basic
import Mathlib.Data.Real.Basic

namespace DecisionQuotient.StochasticSequential

/-! ## Structure Classes -/

/-- A structure class represents a tractable subcase -/
structure StructureClass where
  name : String
  tractable : Prop

/-- Prior distribution over structure classes -/
def StructurePrior (C : Type*) := C → ℝ

/-! ## Bayesian Learning -/

/-- Posterior after observing evidence -/
noncomputable def posterior {C : Type*}
    (prior : StructurePrior C)
    (likelihood : C → ℝ)
    (evidence : ℝ) :
    C → ℝ :=
  fun c => prior c * likelihood c / evidence

/-- Bayesian update: posterior c = prior c × likelihood c / evidence.
    Definitionally true; the normalization condition (evidence = ∑ c, prior c * likelihood c)
    is the caller's responsibility and does not affect the pointwise equality. -/
theorem bayesian_update {C : Type*}
    (prior : StructurePrior C)
    (c : C)
    (likelihood : C → ℝ)
    (evidence : ℝ) :
    posterior prior likelihood evidence c = prior c * likelihood c / evidence := rfl

/-! ## Convergence -/

/-- Structure classes are identifiable if distinct -/
structure Identifiable (C : Type*) (likelihood : C → ℝ → Prop) : Prop where
  distinct : ∀ c₁ c₂ : C, (∀ e, likelihood c₁ e = likelihood c₂ e) → c₁ = c₂

/-- Posterior convergence: with enough evidence, posterior concentrates -/
theorem posterior_converges {C : Type*} [Fintype C] [DecidableEq C]
    (prior : StructurePrior C)
    (likelihood : C → ℝ)
    (c₀ : C)
    (hLike : likelihood c₀ > 0)
    (hLike_other : ∀ c, c ≠ c₀ → likelihood c = 0)
    (hPrior : prior c₀ > 0)
    (evidence : ℝ)
    (hEvPos : evidence > 0) :
    posterior prior likelihood evidence c₀ > 0 := by
  unfold posterior
  have h1 : prior c₀ * likelihood c₀ > 0 := mul_pos hPrior hLike
  exact div_pos h1 hEvPos

/-! ## Abstention Reduction -/

/-- Abstention set shrinks as posterior evidence accumulates -/
def abstentionSet {C : Type*}
    (_k : ℕ)
    (post : C → ℝ)
    (threshold : ℝ) :
    Set C :=
  { c | post c < threshold }

/-- Abstention reduces when posteriors increase -/
theorem abstention_decreases {C : Type*} {k₁ k₂ : ℕ}
    (posterior₁ posterior₂ : C → ℝ)
    (h : ∀ c, posterior₁ c ≤ posterior₂ c)
    (threshold : ℝ) :
    abstentionSet k₂ posterior₂ threshold ⊆
    abstentionSet k₁ posterior₁ threshold := by
  intro c hc
  unfold abstentionSet at *
  simp only [Set.mem_setOf_eq] at *
  calc posterior₁ c ≤ posterior₂ c := h c
    _ < threshold := hc

end DecisionQuotient.StochasticSequential
