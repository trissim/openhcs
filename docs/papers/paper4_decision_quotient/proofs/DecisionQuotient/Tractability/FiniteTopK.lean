/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/FiniteTopK.lean

  Conservative finite top-k objects for sampled pruning.
  We deliberately use a tie-safe set-based notion: an action belongs to the
  top-k-with-ties set if fewer than k actions are strictly better.
-/
import DecisionQuotient.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Real.Basic

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

/-- Threshold survivor set. -/
noncomputable def survivorSet (u : A → ℝ) (tau : ℝ) : Finset A :=
  (Finset.univ : Finset A).filter (fun a => tau ≤ u a)

theorem mem_topKWithTies_iff (u : A → ℝ) (k : Nat) (a : A) :
    a ∈ topKWithTies u k ↔ strictBetterCount u a < k := by
  simp [topKWithTies, strictBetterCount]

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

end FiniteTopK
end Tractability
end DecisionQuotient
