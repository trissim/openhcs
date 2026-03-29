/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/NearTieBand.lean

  Conservative ambiguity-band objects for near-tie top-k reasoning.
-/
import DecisionQuotient.Tractability.FiniteTopK
import Mathlib.Data.Real.Basic

namespace DecisionQuotient
namespace Tractability
namespace NearTieBand

open FiniteTopK

variable {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]

/-- Certified ambiguity band around the exact kth boundary: any action with
    exact utility at least `kthUtility - eps` is retained in the band. -/
noncomputable def ambiguityBand (u : A → ℝ) (k : Nat) (hk : 0 < k) (eps : ℝ) : Finset A :=
  survivorSet u (kthUtility u k hk - eps)

/-- Posewise ambiguity band around a reference action `aRef`: any action whose
    coarse upper bound overlaps the reference coarse lower bound is retained. -/
noncomputable def posewiseAmbiguityBandAround (u : A → ℝ) (delta : A → ℝ) (aRef : A) : Finset A :=
  (Finset.univ : Finset A).filter fun a => u aRef - delta aRef ≤ u a + delta a

theorem mem_ambiguityBand_iff (u : A → ℝ) (k : Nat) (hk : 0 < k) (eps : ℝ) (a : A) :
    a ∈ ambiguityBand u k hk eps ↔ kthUtility u k hk - eps ≤ u a := by
  rw [ambiguityBand, mem_survivorSet_iff]

theorem mem_posewiseAmbiguityBandAround_iff
    (u : A → ℝ) (delta : A → ℝ) (aRef a : A) :
    a ∈ posewiseAmbiguityBandAround u delta aRef ↔ u aRef - delta aRef ≤ u a + delta a := by
  rw [posewiseAmbiguityBandAround, Finset.mem_filter, Finset.mem_univ, true_and]

/-- If the coarse lower bound of `a` exceeds the coarse upper bound of `b`, then
    the exact ordering is preserved in the same direction. -/
theorem exact_order_preserved_of_posewise_error
    (uExact uCoarse : A → ℝ)
    (delta : A → ℝ)
    (a b : A)
    (hApprox : ∀ x, |uExact x - uCoarse x| ≤ delta x)
    (hGap : uCoarse a - delta a > uCoarse b + delta b) :
    uExact b < uExact a := by
  have hALower : uCoarse a - delta a ≤ uExact a := by
    have h := (abs_le.mp (hApprox a)).left
    linarith
  have hBUpper : uExact b ≤ uCoarse b + delta b := by
    have h := (abs_le.mp (hApprox b)).right
    linarith
  calc
    uExact b ≤ uCoarse b + delta b := hBUpper
    _ < uCoarse a - delta a := by linarith
    _ ≤ uExact a := hALower

/-- Under posewise approximation radii, every exact top-1 action lies in the
    posewise ambiguity band around any reference action. -/
theorem exact_top1_subset_posewiseAmbiguityBandAround_of_posewise_error
    (uExact uCoarse : A → ℝ)
    (delta : A → ℝ)
    (aRef : A)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta a) :
    topKSet uExact 1 ⊆ posewiseAmbiguityBandAround uCoarse delta aRef := by
  intro a ha
  rw [mem_posewiseAmbiguityBandAround_iff]
  have hExactMax : uExact aRef ≤ uExact a := by
    have haCount : strictBetterCount uExact a < 1 := (mem_topKSet_iff uExact 1 a).mp ha
    have hNotBetter : ¬ uExact a < uExact aRef := by
      intro hlt
      have hmem : aRef ∈ (Finset.univ : Finset A).filter (fun x => uExact a < uExact x) := by
        simp [hlt]
      have hCardPos : 1 ≤ strictBetterCount uExact a := by
        unfold strictBetterCount
        exact Nat.succ_le_of_lt (Finset.card_pos.mpr ⟨aRef, hmem⟩)
      omega
    exact le_of_not_gt hNotBetter
  have hRefLower : uCoarse aRef - delta aRef ≤ uExact aRef := by
    have h := (abs_le.mp (hApprox aRef)).left
    linarith
  have hAUpper : uExact a ≤ uCoarse a + delta a := by
    have h := (abs_le.mp (hApprox a)).right
    linarith
  linarith

/-- Posewise coarse interval separation certifies an exact singleton top-1. -/
theorem exact_top1_eq_singleton_of_posewise_coarse_gap_margin
    (uExact uCoarse : A → ℝ)
    (delta : A → ℝ)
    (aStar : A)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta a)
    (hStrict : ∀ b, b ≠ aStar → uCoarse aStar - delta aStar > uCoarse b + delta b) :
    topKSet uExact 1 = ({aStar} : Finset A) := by
  classical
  ext a
  rw [mem_topKSet_iff, Finset.mem_singleton]
  constructor
  · intro hTop
    by_contra hne
    have hlt : uExact a < uExact aStar :=
      exact_order_preserved_of_posewise_error uExact uCoarse delta aStar a hApprox (hStrict a hne)
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
          exact_order_preserved_of_posewise_error uExact uCoarse delta a b hApprox (hStrict b hEq)
        exact not_lt_of_ge (le_of_lt hlt)
    rw [hEmpty]
    omega

/-- Every exact top-k action lies in the certified ambiguity band whenever the
    slack parameter `eps` is nonnegative. This is the conservative replacement
    for exact top-k equality when strict boundary gaps fail. -/
theorem exact_topK_subset_ambiguityBand
    (u : A → ℝ) (k : Nat) (hk : 0 < k) (eps : ℝ)
    (hEps : 0 ≤ eps) :
    topKSet u k ⊆ ambiguityBand u k hk eps := by
  apply topKSet_subset_survivorSet_of_le_kthUtility u k hk (kthUtility u k hk - eps)
  linarith

/-- Under a uniform approximation radius `delta`, every exact top-1 action lies
    in the coarse top-1 ambiguity band of width `2 * delta`. -/
theorem exact_top1_subset_coarse_ambiguityBand_of_uniform_error
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    topKSet uExact 1 ⊆ ambiguityBand uCoarse 1 (by omega) (2 * delta) := by
  intro a ha
  rcases topKSet_nonempty uCoarse (k := 1) (by omega) with ⟨b, hb⟩
  rw [mem_ambiguityBand_iff]
  have hKth : kthUtility uCoarse 1 (by omega) ≤ uCoarse b :=
    kthUtility_le_of_mem_topKSet uCoarse 1 (by omega) b hb
  have hExactMax : uExact b ≤ uExact a := by
    have haCount : strictBetterCount uExact a < 1 := (mem_topKSet_iff uExact 1 a).mp ha
    have hNotBetter : ¬ uExact a < uExact b := by
      intro hlt
      have hmem : b ∈ (Finset.univ : Finset A).filter (fun x => uExact a < uExact x) := by
        simp [hlt]
      have hCardPos : 1 ≤ strictBetterCount uExact a := by
        unfold strictBetterCount
        exact Nat.succ_le_of_lt (Finset.card_pos.mpr ⟨b, hmem⟩)
      omega
    exact le_of_not_gt hNotBetter
  have hA : |uExact a - uCoarse a| ≤ delta := hApprox a
  have hB : |uExact b - uCoarse b| ≤ delta := hApprox b
  have hCoarseLower : uExact a - delta ≤ uCoarse a := by
    linarith [(abs_le.mp hA).right]
  have hCoarseUpper : uCoarse b ≤ uExact b + delta := by
    linarith [(abs_le.mp hB).left]
  have hBand : kthUtility uCoarse 1 (by omega) - 2 * delta ≤ uCoarse a := by
    linarith
  exact hBand

/-- For `k = 1`, zero slack collapses the ambiguity band to the exact top-1 set
    with ties. This is the theorem behind the exact-path shortcut used in the
    active certified runtime. -/
theorem ambiguityBand_zero_eq_top1
    (u : A → ℝ) :
    ambiguityBand u 1 (by omega) 0 = topKSet u 1 := by
  classical
  rcases topKSet_nonempty u (k := 1) (by omega) with ⟨a0, ha0⟩
  have hTop0 : strictBetterCount u a0 = 0 := by
    have hlt : strictBetterCount u a0 < 1 := (mem_topKSet_iff u 1 a0).mp ha0
    omega
  have hMax : ∀ b, u b ≤ u a0 := by
    intro b
    by_contra hgt
    have hlt : u a0 < u b := lt_of_not_ge hgt
    have hmem : b ∈ (Finset.univ : Finset A).filter (fun x => u a0 < u x) := by
      simp [hlt]
    unfold strictBetterCount at hTop0
    have hCardPos : 0 < ((Finset.univ : Finset A).filter (fun x => u a0 < u x)).card :=
      Finset.card_pos.mpr ⟨b, hmem⟩
    omega
  have hTop1_eq : ∀ b, b ∈ topKSet u 1 → u b = u a0 := by
    intro b hb
    have hTopB : strictBetterCount u b = 0 := by
      have hlt : strictBetterCount u b < 1 := (mem_topKSet_iff u 1 b).mp hb
      omega
    have hLe1 : u b ≤ u a0 := hMax b
    have hLe2 : u a0 ≤ u b := by
      by_contra hgt
      have hlt : u b < u a0 := lt_of_not_ge hgt
      have hmem : a0 ∈ (Finset.univ : Finset A).filter (fun x => u b < u x) := by
        simp [hlt]
      unfold strictBetterCount at hTopB
      have hCardPos : 0 < ((Finset.univ : Finset A).filter (fun x => u b < u x)).card :=
        Finset.card_pos.mpr ⟨a0, hmem⟩
      omega
    linarith
  have hkth_eq : kthUtility u 1 (by omega) = u a0 := by
    apply le_antisymm
    · exact kthUtility_le_of_mem_topKSet u 1 (by omega) a0 ha0
    · unfold kthUtility
      apply Finset.le_min'
      intro y hy
      rcases Finset.mem_image.mp hy with ⟨b, hb, rfl⟩
      rw [hTop1_eq b hb]
  ext a
  rw [mem_ambiguityBand_iff, sub_zero, mem_topKSet_iff]
  constructor
  · intro hBand
    rw [hkth_eq] at hBand
    have hEq : u a = u a0 := by
      have hUpper : u a ≤ u a0 := hMax a
      linarith
    have hCountZero : strictBetterCount u a = 0 := by
      unfold strictBetterCount
      apply Finset.card_eq_zero.mpr
      rw [Finset.filter_eq_empty_iff]
      intro b hb
      have hle : u b ≤ u a := by rw [hEq]; exact hMax b
      exact not_lt_of_ge hle
    omega
  · intro hTop
    have haTop : a ∈ topKSet u 1 := (mem_topKSet_iff u 1 a).2 hTop
    have hEq : u a = u a0 := hTop1_eq a haTop
    rw [hkth_eq]
    linarith

end NearTieBand
end Tractability
end DecisionQuotient
