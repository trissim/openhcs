/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/TopKPreservation.lean

  Conservative survivor-containment theorems for exact top-k-with-ties sets
  under uniform score approximation.
-/
import DecisionQuotient.Tractability.FiniteTopK
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace DecisionQuotient
namespace Tractability
namespace TopKPreservation

open FiniteTopK

variable {A : Type*} [Fintype A] [DecidableEq A]

/-- If every exact top-k action lies at least `delta` above threshold `tau`, and
    the coarse score approximates the exact score within `delta`, then every
    exact top-k action survives the coarse threshold filter. -/
theorem exact_topK_subset_survivorSet_of_margin
    (uExact uCoarse : A → ℝ)
    (k : Nat)
    (tau delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hMargin : ∀ a, a ∈ topKWithTies uExact k → tau + delta ≤ uExact a) :
    topKWithTies uExact k ⊆ survivorSet uCoarse tau := by
  intro a haTop
  have hA := hApprox a
  have hA_right : uExact a - uCoarse a ≤ delta := (abs_le.mp hA).right
  have hLower : uExact a - delta ≤ uCoarse a := by
    linarith
  have hExactMargin : tau + delta ≤ uExact a := hMargin a haTop
  rw [mem_survivorSet_iff]
  linarith

/-- If an exact score lies at least `delta` below threshold `tau`, then the
    coarse score also lies below `tau` and the action is safely excluded from
    the coarse survivor set. -/
theorem exclude_of_exact_below_threshold_margin
    (uExact uCoarse : A → ℝ)
    (tau delta : ℝ)
    (a : A)
    (hApprox : ∀ x, |uExact x - uCoarse x| ≤ delta)
    (hMargin : uExact a + delta < tau) :
    a ∉ survivorSet uCoarse tau := by
  have hA := hApprox a
  have hA_left : -delta ≤ uExact a - uCoarse a := (abs_le.mp hA).left
  have hUpper : uCoarse a ≤ uExact a + delta := by
    linarith
  rw [mem_survivorSet_iff]
  linarith

end TopKPreservation
end Tractability
end DecisionQuotient
