import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Data.Finset.Max
import Mathlib.Order.Filter.Tendsto
import Paper1IT.GraphEntropy

namespace Ssot
namespace GraphEntropy

section Asymptotic

variable {α β : Type*} [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β]

/-- Coordinatewise k-block lift of an observation map. -/
def blockObserve (observe : α → β) (k : ℕ) : (Fin k → α) → (Fin k → β) :=
  fun x i => observe (x i)

/-- Coordinatewise k-block lift of a one-shot tagger. -/
def blockTagTuple {n : ℕ} (tag : α → Fin n) (k : ℕ) :
    (Fin k → α) → (Fin k → Fin n) :=
  fun x i => tag (x i)

/-- Encode a tuple-valued block tag into a single `Fin (n^k)` symbol. -/
noncomputable def tupleEncode (k n : ℕ) : (Fin k → Fin n) → Fin (n ^ k) :=
  fun t =>
    Fin.cast (by simp) ((Fintype.equivFin (Fin k → Fin n)) t)

/-- k-block tagger encoded into a single alphabet symbol. -/
noncomputable def blockTag {n : ℕ} (tag : α → Fin n) (k : ℕ) :
    (Fin k → α) → Fin (n ^ k) :=
  fun x => tupleEncode k n (blockTagTuple tag k x)

/-- A one-shot zero-error tagging induces a zero-error k-block tagging. -/
theorem block_zeroErrorTagging_of_zeroErrorTagging
    (observe : α → β) {n : ℕ} (tag : α → Fin n)
    (htag : ZeroErrorTagging observe n tag) (k : ℕ) :
    ZeroErrorTagging (blockObserve observe k) (n ^ k) (blockTag tag k) := by
  intro u v hadj hEq
  rcases hadj with ⟨hneq, hobs⟩
  have hTupleEq : blockTagTuple tag k u = blockTagTuple tag k v := by
    have hEncEq :
        tupleEncode k n (blockTagTuple tag k u) =
          tupleEncode k n (blockTagTuple tag k v) := by
      simpa [blockTag] using hEq
    unfold tupleEncode at hEncEq
    have hFinEq :
        (Fintype.equivFin (Fin k → Fin n)) (blockTagTuple tag k u) =
          (Fintype.equivFin (Fin k → Fin n)) (blockTagTuple tag k v) := by
      exact (Fin.cast_injective (h := by simp)) hEncEq
    exact (Fintype.equivFin (Fin k → Fin n)).injective hFinEq
  have hcoord : ∃ i : Fin k, u i ≠ v i := by
    by_contra hnone
    apply hneq
    funext i
    exact by
      by_contra hne
      exact hnone ⟨i, hne⟩
  rcases hcoord with ⟨i, hne_i⟩
  have hobs_i : observe (u i) = observe (v i) := by
    exact congrArg (fun f => f i) hobs
  have htag_i : tag (u i) = tag (v i) := by
    exact congrArg (fun f => f i) hTupleEq
  exact (htag ⟨hne_i, hobs_i⟩) htag_i

/-- Block-feasibility upper bound induced from a one-shot feasible alphabet. -/
theorem block_tagFeasible_of_tagFeasible
    (observe : α → β) {n : ℕ} (hfeas : TagFeasible observe n) (k : ℕ) :
    TagFeasible (blockObserve observe k) (n ^ k) := by
  rcases hfeas with ⟨tag, htag⟩
  exact ⟨blockTag tag k, block_zeroErrorTagging_of_zeroErrorTagging observe tag htag k⟩

/-- Embed coordinatewise fibers into the block-fiber at a constant observation vector. -/
def constantFiberEmbedding (observe : α → β) (k : ℕ) (b : β) :
    (Fin k → {a : α // observe a = b}) ↪
      {x : Fin k → α // blockObserve observe k x = fun _ => b} where
  toFun f := ⟨fun i => (f i).1, by
    funext i
    exact (f i).2⟩
  inj' := by
    intro f g hfg
    funext i
    apply Subtype.ext
    exact congrArg (fun h : {x : Fin k → α // blockObserve observe k x = fun _ => b} => h.1 i) hfg

/-- Exponential lower bound on block confusability via the worst one-shot fiber. -/
theorem pow_maxFiberCard_le_blockMaxFiberCard
    (observe : α → β) [Nonempty β] (k : ℕ) :
    (maxFiberCard observe) ^ k ≤ maxFiberCard (blockObserve observe k) := by
  classical
  let fb : β → ℕ := fun b => (Finset.univ.filter (fun a : α => observe a = b)).card
  obtain ⟨bmax, hbmem, hbmax⟩ :=
    Finset.exists_max_image (Finset.univ : Finset β) fb Finset.univ_nonempty
  have hle₁ : fb bmax ≤ maxFiberCard observe := by
    simpa [fb] using fiber_card_le_maxFiberCard observe bmax
  have hle₂ : maxFiberCard observe ≤ fb bmax := by
    unfold maxFiberCard
    refine Finset.sup_le ?_
    intro b hb
    exact hbmax b (by simpa using hb)
  have hbEq : fb bmax = maxFiberCard observe := le_antisymm hle₁ hle₂
  have hpow_le_subtype :
      (Fintype.card {a : α // observe a = bmax}) ^ k ≤
        Fintype.card {x : Fin k → α // blockObserve observe k x = fun _ => bmax} := by
    have hcard_le :
        Fintype.card (Fin k → {a : α // observe a = bmax}) ≤
          Fintype.card {x : Fin k → α // blockObserve observe k x = fun _ => bmax} := by
      exact Fintype.card_le_of_embedding (constantFiberEmbedding observe k bmax)
    simpa [Fintype.card_fun] using hcard_le
  have hpow_le_fiber :
      (Fintype.card {a : α // observe a = bmax}) ^ k ≤
        (Finset.univ.filter
            (fun x : Fin k → α => blockObserve observe k x = fun _ => bmax)).card := by
    simpa [Fintype.card_subtype] using hpow_le_subtype
  have hfiber_le_max :
      (Finset.univ.filter
          (fun x : Fin k → α => blockObserve observe k x = fun _ => bmax)).card ≤
        maxFiberCard (blockObserve observe k) := by
    exact fiber_card_le_maxFiberCard (observe := blockObserve observe k) (fun _ => bmax)
  have hleft :
      Fintype.card {a : α // observe a = bmax} = maxFiberCard observe := by
    simpa [fb, Fintype.card_subtype] using hbEq
  calc
    (maxFiberCard observe) ^ k = (Fintype.card {a : α // observe a = bmax}) ^ k := by
      simp [hleft]
    _ ≤
        (Finset.univ.filter
          (fun x : Fin k → α => blockObserve observe k x = fun _ => bmax)).card := hpow_le_fiber
    _ ≤ maxFiberCard (blockObserve observe k) := hfiber_le_max

/-- Any feasible block alphabet must dominate the exponential worst-fiber growth. -/
theorem pow_maxFiberCard_le_of_block_tagFeasible
    (observe : α → β) [Nonempty β] (k : ℕ) {n : ℕ}
    (hfeas : TagFeasible (blockObserve observe k) n) :
    (maxFiberCard observe) ^ k ≤ n := by
  exact (pow_maxFiberCard_le_blockMaxFiberCard observe k).trans
    (maxFiberCard_le_of_tagFeasible (observe := blockObserve observe k) hfeas)

/-- Exact k-block feasibility threshold (finite one-shot confusability regime). -/
theorem block_tagFeasible_iff_pow_maxFiberCard_le
    (observe : α → β) [Nonempty β] (k : ℕ) {n : ℕ} :
    TagFeasible (blockObserve observe k) n ↔ (maxFiberCard observe) ^ k ≤ n := by
  constructor
  · intro hfeas
    exact pow_maxFiberCard_le_of_block_tagFeasible observe k hfeas
  · intro hpow
    have hone : TagFeasible observe (maxFiberCard observe) :=
      tagFeasible_of_maxFiberCard_le observe le_rfl
    have hblock :
        TagFeasible (blockObserve observe k) ((maxFiberCard observe) ^ k) :=
      block_tagFeasible_of_tagFeasible observe hone k
    exact tagFeasible_mono (observe := blockObserve observe k) hpow hblock

/-- Exact k-block worst-fiber growth law. -/
theorem maxFiberCard_blockObserve_eq_pow
    (observe : α → β) [Nonempty β] (k : ℕ) :
    maxFiberCard (blockObserve observe k) = (maxFiberCard observe) ^ k := by
  apply le_antisymm
  · have hone : TagFeasible observe (maxFiberCard observe) :=
      tagFeasible_of_maxFiberCard_le observe le_rfl
    have hblock :
        TagFeasible (blockObserve observe k) ((maxFiberCard observe) ^ k) :=
      block_tagFeasible_of_tagFeasible observe hone k
    exact maxFiberCard_le_of_tagFeasible (observe := blockObserve observe k) hblock
  · exact pow_maxFiberCard_le_blockMaxFiberCard observe k

/-- Minimal feasible alphabet size for k-block zero-error tagging. -/
noncomputable def minBlockFeasibleAlphabet (observe : α → β) [Nonempty β] (k : ℕ) : ℕ :=
  by
    classical
    exact Nat.find <|
      show ∃ n : ℕ, TagFeasible (blockObserve observe k) n from
        ⟨(maxFiberCard observe) ^ k, (block_tagFeasible_iff_pow_maxFiberCard_le observe k).2 le_rfl⟩

/-- The minimal feasible k-block alphabet is exactly `(maxFiberCard observe)^k`. -/
theorem minBlockFeasibleAlphabet_eq_pow
    (observe : α → β) [Nonempty β] (k : ℕ) :
    minBlockFeasibleAlphabet observe k = (maxFiberCard observe) ^ k := by
  classical
  apply le_antisymm
  · exact Nat.find_min' (show ∃ n : ℕ, TagFeasible (blockObserve observe k) n from
      ⟨(maxFiberCard observe) ^ k, (block_tagFeasible_iff_pow_maxFiberCard_le observe k).2 le_rfl⟩)
      ((block_tagFeasible_iff_pow_maxFiberCard_le observe k).2 le_rfl)
  · exact (block_tagFeasible_iff_pow_maxFiberCard_le observe k).1
      (Nat.find_spec (show ∃ n : ℕ, TagFeasible (blockObserve observe k) n from
        ⟨(maxFiberCard observe) ^ k, (block_tagFeasible_iff_pow_maxFiberCard_le observe k).2 le_rfl⟩))

/-- Base-2 log block tag requirement (in bits). -/
noncomputable def blockTagRateBits (observe : α → β) (k : ℕ) : ℝ :=
  Real.logb 2 (maxFiberCard (blockObserve observe k))

/-- One-shot base-2 log requirement (in bits). -/
noncomputable def oneShotTagRateBits (observe : α → β) : ℝ :=
  Real.logb 2 (maxFiberCard observe)

/-- Exact linear growth of block tag-rate in bits. -/
theorem blockTagRateBits_eq_mul_oneShot
    (observe : α → β) [Nonempty β] (k : ℕ) :
    blockTagRateBits observe k = (k : ℝ) * oneShotTagRateBits observe := by
  simp [blockTagRateBits, oneShotTagRateBits, maxFiberCard_blockObserve_eq_pow, Real.logb_pow]

/-- Normalized block rate (bits per coordinate). -/
noncomputable def blockTagRateBitsPerCoordinate (observe : α → β) (k : ℕ) : ℝ :=
  blockTagRateBits observe k / (k : ℝ)

/-- Exact stabilization of normalized block rate to one-shot rate. -/
theorem blockTagRateBitsPerCoordinate_eq_oneShot
    (observe : α → β) [Nonempty β] {k : ℕ} (hk : 0 < k) :
    blockTagRateBitsPerCoordinate observe k = oneShotTagRateBits observe := by
  have hk0 : (k : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hk)
  calc
    blockTagRateBitsPerCoordinate observe k
        = ((k : ℝ) * oneShotTagRateBits observe) / (k : ℝ) := by
            simp [blockTagRateBitsPerCoordinate, blockTagRateBits_eq_mul_oneShot]
    _ = oneShotTagRateBits observe := by
          field_simp [hk0]

/-- Log-bits of the minimal feasible k-block alphabet. -/
noncomputable def minBlockFeasibleBits (observe : α → β) [Nonempty β] (k : ℕ) : ℝ :=
  Real.logb 2 (minBlockFeasibleAlphabet observe k)

/-- The minimal feasible k-block bits coincide with the block tag-rate expression. -/
theorem minBlockFeasibleBits_eq_blockTagRateBits
    (observe : α → β) [Nonempty β] (k : ℕ) :
    minBlockFeasibleBits observe k = blockTagRateBits observe k := by
  simp [minBlockFeasibleBits, blockTagRateBits, minBlockFeasibleAlphabet_eq_pow,
    maxFiberCard_blockObserve_eq_pow]

/-- Asymptotic convergence of normalized block bits (indexed by `k+1`) to one-shot bits. -/
theorem tendsto_blockTagRateBitsPerCoordinate_succ
    (observe : α → β) [Nonempty β] :
    Filter.Tendsto (fun k : ℕ => blockTagRateBitsPerCoordinate observe (k + 1))
      Filter.atTop (nhds (oneShotTagRateBits observe)) := by
  have hconst :
      (fun k : ℕ => blockTagRateBitsPerCoordinate observe (k + 1))
        = fun _ : ℕ => oneShotTagRateBits observe := by
    funext k
    exact blockTagRateBitsPerCoordinate_eq_oneShot observe (Nat.succ_pos k)
  simpa [hconst] using (tendsto_const_nhds : Filter.Tendsto
    (fun _ : ℕ => oneShotTagRateBits observe) Filter.atTop (nhds (oneShotTagRateBits observe)))

/-- Exact operational threshold at blocklength `k+1` under alphabet base `m`. -/
theorem block_tagFeasible_pow_budget_iff
    (observe : α → β) [Nonempty β] (k m : ℕ) :
    TagFeasible (blockObserve observe (k + 1)) (m ^ (k + 1)) ↔
      maxFiberCard observe ≤ m := by
  constructor
  · intro hfeas
    have hpow :
        (maxFiberCard observe) ^ (k + 1) ≤ m ^ (k + 1) :=
      (block_tagFeasible_iff_pow_maxFiberCard_le observe (k + 1)).1 hfeas
    exact (Nat.pow_le_pow_iff_left (Nat.succ_ne_zero k)).1 hpow
  · intro hm
    have hpow :
        (maxFiberCard observe) ^ (k + 1) ≤ m ^ (k + 1) := by
      exact Nat.pow_le_pow_left hm (k + 1)
    exact (block_tagFeasible_iff_pow_maxFiberCard_le observe (k + 1)).2 hpow

/-- Capacity-style operational predicate: fixed base alphabet feasible at every blocklength `k+1`. -/
def FeasibleAtAlphabetBase (observe : α → β) [Nonempty β] (m : ℕ) : Prop :=
  ∀ k : ℕ, TagFeasible (blockObserve observe (k + 1)) (m ^ (k + 1))

/-- Operational characterization of feasible base alphabets by one-shot worst-fiber threshold. -/
theorem feasibleAtAlphabetBase_iff
    (observe : α → β) [Nonempty β] (m : ℕ) :
    FeasibleAtAlphabetBase observe m ↔ maxFiberCard observe ≤ m := by
  constructor
  · intro hbase
    exact (block_tagFeasible_pow_budget_iff observe 0 m).1 (hbase 0)
  · intro hm
    intro k
    exact (block_tagFeasible_pow_budget_iff observe k m).2 hm

end Asymptotic

end GraphEntropy
end Ssot
