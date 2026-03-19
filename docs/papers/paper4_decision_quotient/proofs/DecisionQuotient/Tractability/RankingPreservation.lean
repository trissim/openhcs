/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/RankingPreservation.lean

  Pairwise ranking stability under uniform score approximation.
-/
import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import DecisionQuotient.Tractability.FiniteTopK

namespace DecisionQuotient
namespace Tractability
namespace RankingPreservation

open FiniteTopK

variable {A : Type*}

/-- Pairwise utility gap. -/
def PairwiseGap (u : A → ℝ) (a b : A) : ℝ := u a - u b

/-- Boundary gap between the exact top-k utility threshold and a coarse
    threshold `tau`. -/
noncomputable def BoundaryGap {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (u : A → ℝ) (k : Nat) (hk : 0 < k) (tau : ℝ) : ℝ :=
  kthUtility u k hk - tau

theorem threshold_plus_delta_le_of_boundaryGap
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (u : A → ℝ) (k : Nat) (hk : 0 < k) (tau delta : ℝ)
    (hGap : delta ≤ BoundaryGap u k hk tau) :
    tau + delta ≤ kthUtility u k hk := by
  unfold BoundaryGap at hGap
  linarith

/-- If coarse scores approximate exact scores within `delta` uniformly, and the
    exact pairwise gap exceeds `2 * delta`, the pairwise order is preserved. -/
theorem pairwise_order_preserved_of_uniform_error
    (uExact uCoarse : A → ℝ)
    (a b : A)
    (delta : ℝ)
    (hApprox : ∀ x, |uExact x - uCoarse x| ≤ delta)
    (hGap : PairwiseGap uExact a b > 2 * delta) :
    uCoarse b < uCoarse a := by
  have hA := hApprox a
  have hB := hApprox b
  have hA_left : -delta ≤ uExact a - uCoarse a := (abs_le.mp hA).left
  have hA_right : uExact a - uCoarse a ≤ delta := (abs_le.mp hA).right
  have hB_left : -delta ≤ uExact b - uCoarse b := (abs_le.mp hB).left
  have hB_right : uExact b - uCoarse b ≤ delta := (abs_le.mp hB).right
  have hALower : uExact a - delta ≤ uCoarse a := by
    linarith
  have hBUpper : uCoarse b ≤ uExact b + delta := by
    linarith
  calc
    uCoarse b ≤ uExact b + delta := hBUpper
    _ < uExact a - delta := by
      unfold PairwiseGap at hGap
      linarith
    _ ≤ uCoarse a := hALower

/-- A lower-bound corollary phrased as a coarse pairwise gap. -/
theorem coarse_gap_positive_of_exact_gap_margin
    (uExact uCoarse : A → ℝ)
    (a b : A)
    (delta : ℝ)
    (hApprox : ∀ x, |uExact x - uCoarse x| ≤ delta)
    (hGap : PairwiseGap uExact a b > 2 * delta) :
    PairwiseGap uCoarse a b > 0 := by
  unfold PairwiseGap
  have hOrder := pairwise_order_preserved_of_uniform_error uExact uCoarse a b delta hApprox hGap
  linarith

end RankingPreservation
end Tractability
end DecisionQuotient
