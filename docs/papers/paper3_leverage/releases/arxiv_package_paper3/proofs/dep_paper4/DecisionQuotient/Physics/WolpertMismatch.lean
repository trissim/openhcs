/-
  Paper 4: Decision-Relevant Uncertainty

  Physics/WolpertMismatch.lean

  The mismatch branch of the Wolpert-facing refinement can be promoted from a
  cited premise to a theorem because the relevant mathematics is already present
  in the Bayes/KL layer:

  - finite strictly-positive distributions
  - Gibbs' inequality (`KL ≥ 0`)
  - equality iff the two distributions coincide

  Wolpert's correction was precise enough that this part of the refinement can
  be proved directly rather than merely imported. In the current artifact, the
  residual side is no longer purely cited either: finite discrete residual
  witnesses are theorem-level in `WolpertResidual.lean`, while the broader
  stopping-time / absolute-irreversibility regime remains imported in
  `WolpertDecomposition.lean`.
-/

import DecisionQuotient.BayesOptimalityProof
import Mathlib.Algebra.Order.Floor.Ring

open scoped BigOperators
open Finset

namespace DecisionQuotient
namespace Physics
namespace WolpertMismatch

open DecisionQuotient.BayesOptimalityProof

/-- A finite distribution with strictly positive mass on every atom. This is
the exact hypothesis surface required by the existing KL theorems. -/
structure StrictFiniteDistribution (α : Type*) [Fintype α] where
  pmf : α → ℝ
  sum_eq_one : (∑ a ∈ univ, pmf a) = 1
  pos : ∀ a, 0 < pmf a

/-- The mismatch cost is the KL divergence between the actual input
distribution and the distribution the system was designed for. -/
noncomputable def mismatchKL {α : Type*} [Fintype α]
    (actual designed : StrictFiniteDistribution α) : ℝ :=
  KL actual.pmf designed.pmf

/-- Gibbs' inequality: mismatch KL is always nonnegative. -/
theorem mismatchKL_nonneg {α : Type*} [Fintype α]
    (actual designed : StrictFiniteDistribution α) :
    0 ≤ mismatchKL actual designed := by
  exact KL_nonneg actual.pmf designed.pmf
    actual.pos designed.pos actual.sum_eq_one designed.sum_eq_one

/-- The mismatch KL vanishes exactly when the actual and designed
distributions coincide pointwise. -/
theorem mismatchKL_eq_zero_iff_eq {α : Type*} [Fintype α]
    (actual designed : StrictFiniteDistribution α) :
    mismatchKL actual designed = 0 ↔ actual.pmf = designed.pmf := by
  simpa [mismatchKL] using
    (KL_eq_zero_iff_eq actual.pmf designed.pmf
      actual.pos designed.pos actual.sum_eq_one designed.sum_eq_one)

/-- Any explicit mismatch witness implies strictly positive mismatch cost. -/
theorem mismatchKL_pos_of_exists_ne {α : Type*} [Fintype α]
    (actual designed : StrictFiniteDistribution α)
    (hNe : ∃ a, actual.pmf a ≠ designed.pmf a) :
    0 < mismatchKL actual designed := by
  have hNonneg : 0 ≤ mismatchKL actual designed :=
    mismatchKL_nonneg actual designed
  have hNeZero : mismatchKL actual designed ≠ 0 := by
    intro hZero
    have hEq : actual.pmf = designed.pmf :=
      (mismatchKL_eq_zero_iff_eq actual designed).1 hZero
    rcases hNe with ⟨a, ha⟩
    exact ha (congrArg (fun f => f a) hEq)
  exact lt_of_le_of_ne hNonneg (by
    intro hEq
    exact hNeZero hEq.symm)

/-- Discrete lower-bound units obtained by conservatively rounding the
real-valued mismatch KL upward. This is the bridge from the real KL theorem to
the nat-valued thermodynamic accounting used by the current artifact. -/
noncomputable def mismatchNatLowerBound {α : Type*} [Fintype α]
    (actual designed : StrictFiniteDistribution α) : ℕ :=
  Nat.ceil (mismatchKL actual designed)

/-- Any explicit mismatch witness yields a positive discrete lower-bound term
after the declared upward rounding. -/
theorem mismatchNatLowerBound_pos_of_exists_ne {α : Type*} [Fintype α]
    (actual designed : StrictFiniteDistribution α)
    (hNe : ∃ a, actual.pmf a ≠ designed.pmf a) :
    0 < mismatchNatLowerBound actual designed := by
  unfold mismatchNatLowerBound
  exact (Nat.ceil_pos).2 (mismatchKL_pos_of_exists_ne actual designed hNe)

end WolpertMismatch
end Physics
end DecisionQuotient
