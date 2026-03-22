/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/TopKLoweringBridge.lean

  Bridge lemmas tying the ArrayDSL top-k-with-ties primitive to the set-based
  Lean top-k semantics. This formally specifies the threshold/partition-style
  lowering used by the Python/JAX bridge.
-/
import DecisionQuotient.Computation.ArrayDSL
import DecisionQuotient.Tractability.FiniteTopK

namespace DecisionQuotient
namespace Tractability
namespace TopKLoweringBridge

open Computation.ArrayDSL
open FiniteTopK
open Classical

/-- Boolean threshold-mask specification for the top-k-with-ties primitive. -/
noncomputable def thresholdTopKWithTiesMask {n : ℕ}
    [Nonempty (Fin n)]
    (utilities : MDArray n)
    (k : ℕ)
    (hk : 0 < k) : Fin n → Bool :=
  let boundary := kthUtility (fun i : Fin n => utilities i) k hk
  fun i => decide (boundary ≤ utilities i)

theorem topKWithTiesMask_eq_decide_mem_topKSet {n : ℕ}
    (utilities : MDArray n)
    (k : ℕ) :
    topKWithTiesMask utilities k =
      (fun i => decide (i ∈ topKSet (fun j : Fin n => utilities j) k)) := by
  funext i
  simp [topKWithTiesMask, topKSet, topKWithTies, strictBetterCount]

theorem thresholdTopKWithTiesMask_eq_topKWithTiesMask {n : ℕ}
    [Nonempty (Fin n)]
    (utilities : MDArray n)
    (k : ℕ)
    (hk : 0 < k) :
    thresholdTopKWithTiesMask utilities k hk = topKWithTiesMask utilities k := by
  funext i
  rw [topKWithTiesMask_eq_decide_mem_topKSet]
  simp [thresholdTopKWithTiesMask, mem_topKSet_iff_kthUtility_le, hk]

end TopKLoweringBridge
end Tractability
end DecisionQuotient
