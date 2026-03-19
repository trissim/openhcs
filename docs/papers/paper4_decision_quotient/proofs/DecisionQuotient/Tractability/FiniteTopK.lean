/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/FiniteTopK.lean

  Conservative finite top-k objects for sampled pruning.
  We deliberately use a tie-safe set-based notion: an action belongs to the
  top-k-with-ties set if fewer than k actions are strictly better.
-/
import DecisionQuotient.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Finset.Max
import Mathlib.Data.Finset.Sort
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace DecisionQuotient
namespace Tractability
namespace FiniteTopK

open Classical

variable {A : Type*} [Fintype A] [DecidableEq A]

/-- Number of actions with strictly larger utility than `a`. -/
noncomputable def strictBetterCount (u : A → ℝ) (a : A) : Nat :=
  ((Finset.univ : Finset A).filter (fun b => u a < u b)).card

/-- Conservative top-k set with ties: every action with fewer than `k`
    strictly better competitors survives. -/
noncomputable def topKWithTies (u : A → ℝ) (k : Nat) : Finset A :=
  (Finset.univ : Finset A).filter (fun a => strictBetterCount u a < k)

/-- Canonical finite top-k set used by the pruning layer. -/
noncomputable def topKSet (u : A → ℝ) (k : Nat) : Finset A :=
  topKWithTies u k

/-- Deterministic list representation of the certified top-k set. This is a
    serialization of the set, not a second notion of ranking. -/
noncomputable def topKList [LinearOrder A] (u : A → ℝ) (k : Nat) : List A :=
  (topKSet u k).sort (· ≤ ·)

/-- Threshold survivor set. -/
noncomputable def survivorSet (u : A → ℝ) (tau : ℝ) : Finset A :=
  (Finset.univ : Finset A).filter (fun a => tau ≤ u a)

theorem mem_topKWithTies_iff (u : A → ℝ) (k : Nat) (a : A) :
    a ∈ topKWithTies u k ↔ strictBetterCount u a < k := by
  simp [topKWithTies, strictBetterCount]

theorem mem_topKSet_iff (u : A → ℝ) (k : Nat) (a : A) :
    a ∈ topKSet u k ↔ strictBetterCount u a < k := by
  simpa [topKSet] using mem_topKWithTies_iff u k a

theorem mem_topKList_iff [LinearOrder A] (u : A → ℝ) (k : Nat) (a : A) :
    a ∈ topKList u k ↔ a ∈ topKSet u k := by
  simp [topKList]

theorem not_mem_topKWithTies_iff (u : A → ℝ) (k : Nat) (a : A) :
    a ∉ topKWithTies u k ↔ k ≤ strictBetterCount u a := by
  rw [mem_topKWithTies_iff]
  omega

theorem mem_survivorSet_iff (u : A → ℝ) (tau : ℝ) (a : A) :
    a ∈ survivorSet u tau ↔ tau ≤ u a := by
  simp [survivorSet]

theorem topKWithTies_monotone (u : A → ℝ) {k1 k2 : Nat}
    (hk : k1 ≤ k2) :
    topKWithTies u k1 ⊆ topKWithTies u k2 := by
  intro a ha
  rw [mem_topKWithTies_iff] at ha ⊢
  omega

/-- If at least `k` actions are strictly better than `a`, then `a` is excluded
    from the top-k-with-ties set. -/
theorem exclude_of_strictly_better_count_ge (u : A → ℝ) (k : Nat) (a : A)
    (hCount : k ≤ strictBetterCount u a) :
    a ∉ topKWithTies u k := by
  rw [not_mem_topKWithTies_iff]
  exact hCount

/-- If utility clears the threshold, the action survives the threshold filter. -/
theorem survive_of_threshold_le (u : A → ℝ) (tau : ℝ) (a : A)
    (hTau : tau ≤ u a) :
    a ∈ survivorSet u tau := by
  rw [mem_survivorSet_iff]
  exact hTau

theorem topKSet_nonempty (u : A → ℝ) [Nonempty A] {k : Nat} (hk : 0 < k) :
    (topKSet u k).Nonempty := by
  classical
  have huniv : (Finset.univ : Finset A).Nonempty := by
    rcases ‹Nonempty A› with ⟨a0⟩
    exact ⟨a0, Finset.mem_univ a0⟩
  obtain ⟨a, -, hmax⟩ := Finset.exists_max_image (Finset.univ : Finset A) u huniv
  refine ⟨a, ?_⟩
  rw [mem_topKSet_iff]
  have hCountZero : strictBetterCount u a = 0 := by
    unfold strictBetterCount
    apply Finset.card_eq_zero.mpr
    rw [Finset.filter_eq_empty_iff]
    intro b hb
    have hle : u b ≤ u a := hmax b hb
    exact not_lt_of_ge hle
  rw [hCountZero]
  simpa using hk

/-- Utility value at the conservative top-k boundary. -/
noncomputable def kthUtility (u : A → ℝ) [Nonempty A] (k : Nat) (hk : 0 < k) : ℝ :=
  ((topKSet u k).image u).min' <| by
    rcases topKSet_nonempty u hk with ⟨a, ha⟩
    exact ⟨u a, Finset.mem_image.mpr ⟨a, ha, rfl⟩⟩

theorem kthUtility_le_of_mem_topKSet (u : A → ℝ) [Nonempty A]
    (k : Nat) (hk : 0 < k) (a : A) (ha : a ∈ topKSet u k) :
    kthUtility u k hk ≤ u a := by
  unfold kthUtility
  apply Finset.min'_le
  exact Finset.mem_image.mpr ⟨a, ha, rfl⟩

theorem topKSet_subset_survivorSet_of_le_kthUtility (u : A → ℝ) [Nonempty A]
    (k : Nat) (hk : 0 < k) (tau : ℝ)
    (hTau : tau ≤ kthUtility u k hk) :
    topKSet u k ⊆ survivorSet u tau := by
  intro a ha
  rw [mem_survivorSet_iff]
  exact le_trans hTau (kthUtility_le_of_mem_topKSet u k hk a ha)

end FiniteTopK
end Tractability
end DecisionQuotient
