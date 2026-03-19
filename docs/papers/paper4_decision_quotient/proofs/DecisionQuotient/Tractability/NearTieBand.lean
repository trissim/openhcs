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

theorem mem_ambiguityBand_iff (u : A → ℝ) (k : Nat) (hk : 0 < k) (eps : ℝ) (a : A) :
    a ∈ ambiguityBand u k hk eps ↔ kthUtility u k hk - eps ≤ u a := by
  rw [ambiguityBand, mem_survivorSet_iff]

/-- Every exact top-k action lies in the certified ambiguity band whenever the
    slack parameter `eps` is nonnegative. This is the conservative replacement
    for exact top-k equality when strict boundary gaps fail. -/
theorem exact_topK_subset_ambiguityBand
    (u : A → ℝ) (k : Nat) (hk : 0 < k) (eps : ℝ)
    (hEps : 0 ≤ eps) :
    topKSet u k ⊆ ambiguityBand u k hk eps := by
  apply topKSet_subset_survivorSet_of_le_kthUtility u k hk (kthUtility u k hk - eps)
  linarith

end NearTieBand
end Tractability
end DecisionQuotient
