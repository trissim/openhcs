import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Combinatorics.Pigeonhole
import Mathlib.Combinatorics.SimpleGraph.Clique
import Mathlib.Data.Fintype.EquivFin
import Mathlib.Data.Finset.Card
import Mathlib.Tactic
import Paper1IT.Entropy

open scoped BigOperators

namespace Ssot
namespace GraphEntropy

structure VertexDist (α : Type*) [Fintype α] where
  prob : α → ℝ
  nonneg : ∀ a, 0 ≤ prob a
  sum_one : ∑ a, prob a = 1

namespace VertexDist

variable {α β : Type*} [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β]

noncomputable def support [DecidableEq α] (p : VertexDist α) : Finset α :=
  Finset.univ.filter fun a => 0 < p.prob a

noncomputable def supportSize [DecidableEq α] (p : VertexDist α) : ℕ :=
  p.support.card

noncomputable def entropy (p : VertexDist α) : ℝ :=
  -∑ a, entropyTerm (p.prob a)

noncomputable def pushforward (p : VertexDist α) (f : α → β) : VertexDist β where
  prob := fun b => Finset.sum (Finset.univ.filter (fun a => f a = b)) fun a => p.prob a
  nonneg b := by
    refine Finset.sum_nonneg ?_
    intro a ha
    exact p.nonneg a
  sum_one := by
    classical
    calc
      ∑ b, Finset.sum (Finset.univ.filter (fun a => f a = b)) (fun a => p.prob a)
          = ∑ b, Finset.sum Finset.univ (fun a => if f a = b then p.prob a else 0) := by
              simp_rw [Finset.sum_filter]
      _ = ∑ a, Finset.sum Finset.univ (fun b => if f a = b then p.prob a else 0) := by
            rw [Finset.sum_comm]
      _ = ∑ a, p.prob a := by
            refine Finset.sum_congr rfl ?_
            intro a ha
            simp [eq_comm]
      _ = 1 := p.sum_one

end VertexDist

section Coloring

variable {α : Type*} [Fintype α] [DecidableEq α]

def IsProperColoring (G : SimpleGraph α) (n : ℕ) (c : α → Fin n) : Prop :=
  ∀ ⦃u v : α⦄, G.Adj u v → c u ≠ c v

noncomputable def coloringEntropy (G : SimpleGraph α)
    (p : VertexDist α) {n : ℕ} (c : α → Fin n) : ℝ :=
  (p.pushforward c).entropy

noncomputable def chromaticEntropySet (G : SimpleGraph α) (p : VertexDist α) : Set ℝ :=
  {r | ∃ n, ∃ c : α → Fin n, IsProperColoring G n c ∧ coloringEntropy G p c = r}

noncomputable def chromaticEntropy (G : SimpleGraph α) (p : VertexDist α) : ℝ :=
  sInf (chromaticEntropySet G p)

theorem properColoring_injOn_clique {G : SimpleGraph α} {n : ℕ}
    (c : α → Fin n) (hc : IsProperColoring G n c)
    {s : Finset α} (hs : G.IsClique (↑s : Set α)) :
    Set.InjOn c (↑s : Set α) := by
  intro u hu v hv huv
  by_cases hneq : u = v
  · exact hneq
  · have hadj : G.Adj u v := hs hu hv hneq
    exact False.elim <| (hc hadj) huv

theorem clique_card_le_colors {G : SimpleGraph α} {n : ℕ}
    (c : α → Fin n) (hc : IsProperColoring G n c)
    {s : Finset α} (hs : G.IsClique (↑s : Set α)) :
    s.card ≤ n := by
  classical
  have hinj : Set.InjOn c (↑s : Set α) := properColoring_injOn_clique c hc hs
  have himage : (s.image c).card = s.card := by
    exact Finset.card_image_of_injOn fun a ha b hb hab => hinj ha hb hab
  have hbound : (s.image c).card ≤ n := by
    simpa using (Finset.card_le_univ (s := s.image c))
  exact himage.ge.trans hbound

theorem complete_graph_needs_all_colors {n m : ℕ} (c : Fin n → Fin m)
    (hc : IsProperColoring (⊤ : SimpleGraph (Fin n)) m c) :
    n ≤ m := by
  classical
  let s : Finset (Fin n) := Finset.univ
  have hs : (⊤ : SimpleGraph (Fin n)).IsClique (↑s : Set (Fin n)) := by
    intro u hu v hv huv
    simpa using huv
  simpa [s] using clique_card_le_colors c hc hs

end Coloring

section Confusability

variable {α β : Type*} [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β]

/-- Confusability graph induced by an observation map. -/
def confusabilityGraph (observe : α → β) : SimpleGraph α where
  Adj u v := u ≠ v ∧ observe u = observe v
  symm := by
    intro u v h
    exact ⟨h.1.symm, h.2.symm⟩
  loopless := by
    intro u h
    exact h.1 rfl

/-- A zero-error tag assignment is exactly a proper coloring of the confusability graph. -/
def ZeroErrorTagging (observe : α → β) (n : ℕ) (tag : α → Fin n) : Prop :=
  IsProperColoring (confusabilityGraph observe) n tag

/-- The fiber of an observation value forms a clique in the confusability graph. -/
theorem fiber_is_clique (observe : α → β) (b : β) :
    (confusabilityGraph observe).IsClique {a : α | observe a = b} := by
  intro u hu v hv huv
  exact ⟨huv, hu.trans hv.symm⟩

/-- Finite observation fibers are cliques. -/
theorem fiber_finset_is_clique (observe : α → β) (b : β) :
    (confusabilityGraph observe).IsClique
      (↑(Finset.univ.filter (fun a : α => observe a = b)) : Set α) := by
  simpa using fiber_is_clique observe b

/-- Any zero-error tagging must use at least as many tags as any confusable fiber size. -/
theorem fiber_card_le_tag_alphabet (observe : α → β) (b : β)
    {n : ℕ} (tag : α → Fin n) (htag : ZeroErrorTagging observe n tag) :
    (Finset.univ.filter (fun a : α => observe a = b)).card ≤ n := by
  exact clique_card_le_colors tag htag (fiber_finset_is_clique observe b)

/-- Worst-case confusable fiber size under the observation map. -/
noncomputable def maxFiberCard (observe : α → β) : ℕ :=
  Finset.sup Finset.univ (fun b : β => (Finset.univ.filter (fun a : α => observe a = b)).card)

/-- Every observation fiber is bounded by the worst-case confusable fiber. -/
theorem fiber_card_le_maxFiberCard (observe : α → β) (b : β) :
    (Finset.univ.filter (fun a : α => observe a = b)).card ≤ maxFiberCard observe := by
  unfold maxFiberCard
  exact Finset.le_sup
    (s := Finset.univ)
    (f := fun b : β => (Finset.univ.filter (fun a : α => observe a = b)).card)
    (by exact Finset.mem_univ b)

/-- Zero-error tag alphabets must dominate the worst-case confusable fiber size. -/
theorem maxFiberCard_le_tag_alphabet (observe : α → β)
    {n : ℕ} (tag : α → Fin n) (htag : ZeroErrorTagging observe n tag) :
    maxFiberCard observe ≤ n := by
  unfold maxFiberCard
  refine Finset.sup_le ?_
  intro b hb
  exact fiber_card_le_tag_alphabet observe b tag htag

/-- Pigeonhole lower bound: some observation fiber has size at least the average occupancy. -/
theorem card_div_le_maxFiberCard (observe : α → β) [Nonempty β] :
    Fintype.card α / Fintype.card β ≤ maxFiberCard observe := by
  let n : ℕ := Fintype.card α / Fintype.card β
  have hmul : Fintype.card β * n ≤ Fintype.card α := by
    dsimp [n]
    exact Nat.mul_div_le (Fintype.card α) (Fintype.card β)
  obtain ⟨b, hb⟩ :=
    Fintype.exists_le_card_fiber_of_mul_le_card (f := observe) (n := n) hmul
  have hb' :
      n ≤ (Finset.univ.filter (fun a : α => observe a = b)).card := by
    simpa using hb
  exact hb'.trans (fiber_card_le_maxFiberCard observe b)

/-- Any zero-error tagging alphabet must dominate the average observation-fiber occupancy. -/
theorem card_div_le_tag_alphabet (observe : α → β) [Nonempty β]
    {n : ℕ} (tag : α → Fin n) (htag : ZeroErrorTagging observe n tag) :
    Fintype.card α / Fintype.card β ≤ n := by
  exact (card_div_le_maxFiberCard observe).trans (maxFiberCard_le_tag_alphabet observe tag htag)

/-- Pigeonhole impossibility: too-small tag alphabets cannot support zero-error tagging. -/
theorem no_zeroErrorTagging_of_mul_lt_card (observe : α → β) [Nonempty β]
    {n : ℕ} (hcard : Fintype.card β * n < Fintype.card α) :
    ¬ ∃ tag : α → Fin n, ZeroErrorTagging observe n tag := by
  obtain ⟨b, hb⟩ :=
    Fintype.exists_lt_card_fiber_of_mul_lt_card (f := observe) (n := n) hcard
  intro htag
  rcases htag with ⟨tag, htag⟩
  have hle :
      (Finset.univ.filter (fun a : α => observe a = b)).card ≤ n :=
    fiber_card_le_tag_alphabet observe b tag htag
  exact (Nat.not_lt_of_ge hle) (by simpa using hb)

/-- Global counting converse: zero-error tagging implies `|α| ≤ |β| * n`. -/
theorem card_le_mul_tag_alphabet (observe : α → β) [Nonempty β]
    {n : ℕ} (tag : α → Fin n) (htag : ZeroErrorTagging observe n tag) :
    Fintype.card α ≤ Fintype.card β * n := by
  by_contra hle
  have hlt : Fintype.card β * n < Fintype.card α := Nat.lt_of_not_ge hle
  exact (no_zeroErrorTagging_of_mul_lt_card observe (n := n) hlt) ⟨tag, htag⟩

/-- Predicate: alphabet size `n` supports a zero-error tagging for `observe`. -/
def TagFeasible (observe : α → β) (n : ℕ) : Prop :=
  ∃ tag : α → Fin n, ZeroErrorTagging observe n tag

/-- Lift a zero-error tagging to a larger alphabet by `Fin.castLE`. -/
theorem lift_zeroErrorTagging (observe : α → β) {m n : ℕ} (hmn : m ≤ n)
    {tag : α → Fin m} (htag : ZeroErrorTagging observe m tag) :
    ZeroErrorTagging observe n (fun a => Fin.castLE hmn (tag a)) := by
  intro u v hadj hEq
  apply htag hadj
  exact Fin.castLE_injective hmn hEq

/-- Feasibility is monotone in the available tag alphabet size. -/
theorem tagFeasible_mono (observe : α → β) {m n : ℕ} (hmn : m ≤ n) :
    TagFeasible observe m → TagFeasible observe n := by
  intro hfeas
  rcases hfeas with ⟨tag, htag⟩
  exact ⟨fun a => Fin.castLE hmn (tag a), lift_zeroErrorTagging observe hmn htag⟩

/-- Any feasible alphabet size must be at least `maxFiberCard`. -/
theorem maxFiberCard_le_of_tagFeasible (observe : α → β) {n : ℕ}
    (hfeas : TagFeasible observe n) :
    maxFiberCard observe ≤ n := by
  rcases hfeas with ⟨tag, htag⟩
  exact maxFiberCard_le_tag_alphabet observe tag htag

/-- Any feasible alphabet size must dominate the average fiber occupancy floor. -/
theorem card_div_le_of_tagFeasible (observe : α → β) [Nonempty β] {n : ℕ}
    (hfeas : TagFeasible observe n) :
    Fintype.card α / Fintype.card β ≤ n := by
  rcases hfeas with ⟨tag, htag⟩
  exact card_div_le_tag_alphabet observe tag htag

/-- Constructive feasibility: if `n` dominates the worst observation fiber, a zero-error tagging exists. -/
theorem tagFeasible_of_maxFiberCard_le (observe : α → β) {n : ℕ}
    (hn : maxFiberCard observe ≤ n) :
    TagFeasible observe n := by
  classical
  let emb : ∀ b : β, {a : α // observe a = b} ↪ Fin n := fun b =>
    Classical.choice <|
      Function.Embedding.nonempty_of_card_le <|
        (by
          have hfiber :
              (Finset.univ.filter (fun a : α => observe a = b)).card ≤ maxFiberCard observe :=
            fiber_card_le_maxFiberCard observe b
          have hcard :
              Fintype.card {a : α // observe a = b} ≤ maxFiberCard observe := by
            simpa [Fintype.card_subtype] using hfiber
          have hcard' : Fintype.card {a : α // observe a = b} ≤ Fintype.card (Fin n) := by
            simpa using hcard.trans hn
          exact hcard')
  let tag : α → Fin n := fun a => emb (observe a) ⟨a, rfl⟩
  refine ⟨tag, ?_⟩
  intro u v hadj hEq
  rcases hadj with ⟨hneq, hobs⟩
  have hEq0 :
      emb (observe u) ⟨u, rfl⟩ = emb (observe v) ⟨v, rfl⟩ := by
    simpa [tag] using hEq
  have hEq1 :
      emb (observe u) ⟨u, rfl⟩ = emb (observe u) ⟨v, hobs.symm⟩ := by
    exact Eq.rec
      (motive := fun b hb => emb (observe u) ⟨u, rfl⟩ = emb b ⟨v, hb⟩)
      hEq0 hobs.symm
  have hsub : (⟨u, rfl⟩ : {a : α // observe a = observe u}) = ⟨v, hobs.symm⟩ :=
    (emb (observe u)).injective hEq1
  have huv : u = v := by
    simpa using congrArg Subtype.val hsub
  exact hneq huv

/-- Exact feasibility characterization by the maximal observation-fiber size. -/
theorem tagFeasible_iff_maxFiberCard_le (observe : α → β) {n : ℕ} :
    TagFeasible observe n ↔ maxFiberCard observe ≤ n := by
  constructor
  · intro hfeas
    exact maxFiberCard_le_of_tagFeasible observe hfeas
  · intro hn
    exact tagFeasible_of_maxFiberCard_le observe hn

/-- If the observation map is constant, every pair of distinct vertices is confusable. -/
theorem confusabilityGraph_eq_top_of_constant
    (observe : α → β) (hconst : ∀ u v : α, observe u = observe v) :
    confusabilityGraph observe = ⊤ := by
  ext u v
  constructor
  · intro h
    simpa using h.1
  · intro h
    exact ⟨by simpa using h, hconst u v⟩

/-- In the maximal-barrier regime, zero-error tagging needs at least one tag per state. -/
theorem constant_observation_needs_all_tags
    (observe : α → β) (hconst : ∀ u v : α, observe u = observe v)
    {n : ℕ} (tag : α → Fin n) (htag : ZeroErrorTagging observe n tag) :
    Fintype.card α ≤ n := by
  let s : Finset α := Finset.univ
  have htop : confusabilityGraph observe = ⊤ :=
    confusabilityGraph_eq_top_of_constant observe hconst
  have hs : (confusabilityGraph observe).IsClique (↑s : Set α) := by
    rw [htop]
    intro u hu v hv huv
    simpa using huv
  simpa [s] using clique_card_le_colors tag htag hs

end Confusability

end GraphEntropy
end Ssot
