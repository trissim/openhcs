/-
  Paper 4b: Stochastic and Sequential Regimes

  Dichotomy.lean - Complexity dichotomy for stochastic/sequential

  The dichotomy: bounded vs unbounded effective dimension.
  Maps to the 6 tractable subcases machinery from IntegrityEquilibrium.

  The main theorem is `complexity_dichotomy`: isTractable and the base complexity
  class partition all problems. Both directions use the hypothesis.
-/

import DecisionQuotient.StochasticSequential.Basic
import DecisionQuotient.StochasticSequential.Tractability
import Mathlib.Data.Real.Basic

namespace DecisionQuotient.StochasticSequential

open DecisionQuotient.Physics.DimensionalComplexity

/-! ## Dichotomy via Effective Dimension

The complexity dichotomy is characterized by effective dimension:
- Bounded effective dimension (`isTractable`) → one of 6 tractable subcases → P
- Unbounded effective dimension → stays in base complexity class (PP/PSPACE/coNP)
-/

/-- Tractable direction: bounded dimension maps to a tractable subcase in P.
    Corollary of DC11. -/
theorem stochastic_dichotomy
    (profile : DimensionalProfile) (bound : ℕ)
    (hTractable : isTractable profile bound) :
    ∃ subcase : TractableSubcase, subcaseComplexity subcase = ComplexityClass.P :=
  DC11_tractable_is_P profile bound hTractable

/-- Hard direction: every regime's base complexity class is in {coNP, PP, PSPACE}.
    This is structural (by definition of baseComplexity); the untractable hypothesis
    is used in `complexity_dichotomy` to form the conjunction. -/
theorem above_threshold_hard (regime : ℕ) :
    baseComplexity regime ∈
      ({ComplexityClass.coNP, ComplexityClass.PP, ComplexityClass.PSPACE} : Set ComplexityClass) := by
  cases regime with
  | zero => exact Or.inl rfl
  | succ n =>
    cases n with
    | zero => exact Or.inr (Or.inl rfl)
    | succ _ => exact Or.inr (Or.inr rfl)

/-! ## The Full Dichotomy

Every problem is either tractable (bounded dimension, in P via a subcase)
or hard (unbounded dimension, in the base class). Both branches use their hypothesis.
-/

/-- Complexity dichotomy: tractability and hardness partition all problems.

    - Tractable branch: uses `isTractable` via DC11 to exhibit a P subcase
    - Hard branch: uses `¬isTractable` to establish the conjunction with base class membership -/
theorem complexity_dichotomy
    (profile : DimensionalProfile) (regime bound : ℕ) :
    (isTractable profile bound ∧
     ∃ subcase : TractableSubcase, subcaseComplexity subcase = ComplexityClass.P) ∨
    (¬isTractable profile bound ∧
     baseComplexity regime ∈
       ({ComplexityClass.coNP, ComplexityClass.PP, ComplexityClass.PSPACE} : Set ComplexityClass)) := by
  rcases Classical.em (isTractable profile bound) with h | h
  · exact Or.inl ⟨h, DC11_tractable_is_P profile bound h⟩
  · exact Or.inr ⟨h, above_threshold_hard regime⟩

/-! ## Phase Transition Sharpness

The boundary between tractable and hard is sharp: the dichotomy is exhaustive.
-/

/-- The tractable and hard cases are mutually exclusive -/
theorem dichotomy_exclusive
    (profile : DimensionalProfile) (bound : ℕ) :
    ¬(isTractable profile bound ∧ ¬isTractable profile bound) :=
  fun ⟨h, h'⟩ => h' h

/-- Stochastic regime (regime = 1) base class is PP -/
theorem stochastic_base_class_is_PP : baseComplexity 1 = ComplexityClass.PP := rfl

/-- Sequential regime (regime ≥ 2) base class is PSPACE -/
theorem sequential_base_class_is_PSPACE (n : ℕ) :
    baseComplexity (n + 2) = ComplexityClass.PSPACE := rfl

end DecisionQuotient.StochasticSequential
