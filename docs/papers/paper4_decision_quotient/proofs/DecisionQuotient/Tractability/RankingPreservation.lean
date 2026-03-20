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

/-- If the coarse pairwise gap exceeds `2 * delta`, the exact ordering is also
    preserved in the same direction. -/
theorem exact_order_preserved_of_coarse_gap_margin
    (uExact uCoarse : A → ℝ)
    (a b : A)
    (delta : ℝ)
    (hApprox : ∀ x, |uExact x - uCoarse x| ≤ delta)
    (hGap : PairwiseGap uCoarse a b > 2 * delta) :
    uExact b < uExact a := by
  have hA := hApprox a
  have hB := hApprox b
  have hA_lower : uCoarse a - delta ≤ uExact a := by
    linarith [(abs_le.mp hA).left]
  have hB_upper : uExact b ≤ uCoarse b + delta := by
    linarith [(abs_le.mp hB).right]
  calc
    uExact b ≤ uCoarse b + delta := hB_upper
    _ < uCoarse a - delta := by
      unfold PairwiseGap at hGap
      linarith
    _ ≤ uExact a := hA_lower

/-- A coarse strict winner with margin `> 2 * delta` is also an exact strict
    winner under the same uniform approximation radius. -/
theorem exact_strictOpt_of_coarse_strictOpt_margin
    (uExact uCoarse : A → ℝ)
    (aStar : A)
    (delta : ℝ)
    (hApprox : ∀ x, |uExact x - uCoarse x| ≤ delta)
    (hStrict : ∀ b, b ≠ aStar → PairwiseGap uCoarse aStar b > 2 * delta) :
    ∀ b, b ≠ aStar → uExact b < uExact aStar := by
  intro b hb
  exact exact_order_preserved_of_coarse_gap_margin uExact uCoarse aStar b delta hApprox (hStrict b hb)

/-- If a coarse winner beats every rival by more than `2 * delta`, the exact
    top-1-with-ties set collapses to the singleton `{aStar}`. -/
theorem exact_top1_eq_singleton_of_coarse_gap_margin
    {A : Type*} [Fintype A] [DecidableEq A]
    (uExact uCoarse : A → ℝ)
    (aStar : A)
    (delta : ℝ)
    (hApprox : ∀ x, |uExact x - uCoarse x| ≤ delta)
    (hStrict : ∀ b, b ≠ aStar → PairwiseGap uCoarse aStar b > 2 * delta) :
    topKSet uExact 1 = ({aStar} : Finset A) := by
  classical
  ext a
  rw [mem_topKSet_iff, Finset.mem_singleton]
  constructor
  · intro hTop
    by_contra hne
    have hlt : uExact a < uExact aStar := exact_strictOpt_of_coarse_strictOpt_margin uExact uCoarse aStar delta hApprox hStrict a hne
    have hmem : aStar ∈ (Finset.univ : Finset A).filter (fun x => uExact a < uExact x) := by
      simp [hlt]
    unfold strictBetterCount at hTop
    have hCardPos : 0 < ((Finset.univ : Finset A).filter (fun x => uExact a < uExact x)).card :=
      Finset.card_pos.mpr ⟨aStar, hmem⟩
    omega
  · intro hEq
    subst hEq
    unfold strictBetterCount
    have hEmpty : ((Finset.univ : Finset A).filter (fun b => uExact a < uExact b)).card = 0 := by
      apply Finset.card_eq_zero.mpr
      rw [Finset.filter_eq_empty_iff]
      intro b hbUniv
      by_cases hEq : b = a
      · subst hEq
        exact not_lt_of_ge le_rfl
      · have hlt : uExact b < uExact a :=
          exact_strictOpt_of_coarse_strictOpt_margin uExact uCoarse a delta hApprox hStrict b hEq
        exact not_lt_of_ge (le_of_lt hlt)
    rw [hEmpty]
    omega

end RankingPreservation
end Tractability
end DecisionQuotient
