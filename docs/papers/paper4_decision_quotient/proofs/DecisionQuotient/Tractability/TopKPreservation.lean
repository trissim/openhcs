/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/TopKPreservation.lean

  Conservative survivor-containment theorems for exact top-k-with-ties sets
  under uniform score approximation.
-/
import DecisionQuotient.Tractability.FiniteTopK
import DecisionQuotient.Tractability.RankingPreservation
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace DecisionQuotient
namespace Tractability
namespace TopKPreservation

open FiniteTopK
open RankingPreservation

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

/-- Boundary-gap specialization of the conservative survivor theorem. -/
theorem topK_preserved_of_boundary_gap
    (uExact uCoarse : A → ℝ)
    [Nonempty A]
    (k : Nat)
    (hk : 0 < k)
    (tau delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hGap : delta ≤ BoundaryGap uExact k hk tau) :
    topKSet uExact k ⊆ survivorSet uCoarse tau := by
  apply exact_topK_subset_survivorSet_of_margin uExact uCoarse k tau delta hApprox
  intro a ha
  have hBoundary : tau + delta ≤ kthUtility uExact k hk :=
    threshold_plus_delta_le_of_boundaryGap uExact k hk tau delta hGap
  exact le_trans hBoundary (kthUtility_le_of_mem_topKSet uExact k hk a ha)

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
