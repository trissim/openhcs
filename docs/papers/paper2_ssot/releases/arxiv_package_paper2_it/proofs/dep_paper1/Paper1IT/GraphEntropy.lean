import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Combinatorics.Pigeonhole
import Mathlib.Combinatorics.SimpleGraph.Clique
import Mathlib.Data.Fintype.EquivFin
import Mathlib.Data.Finset.Card
import Mathlib.Data.Nat.Log
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

noncomputable def expectedLength (p : VertexDist α) (ℓ : α → ℕ) : ℝ :=
  ∑ a, p.prob a * (ℓ a : ℝ)

noncomputable def toFiniteDist (p : VertexDist α) : FiniteDist (Fintype.card α) where
  prob i := p.prob ((Fintype.equivFin α).symm i)
  nonneg i := p.nonneg _
  sum_one := by
    classical
    calc
      ∑ i : Fin (Fintype.card α), p.prob ((Fintype.equivFin α).symm i)
          = ∑ a : α, p.prob a := by
              simpa using
                (Fintype.sum_equiv (Fintype.equivFin α)
                  (fun a : α => p.prob a)
                  (fun i : Fin (Fintype.card α) => p.prob ((Fintype.equivFin α).symm i))
                  (by intro a; simp)).symm
      _ = 1 := p.sum_one

theorem entropy_toFiniteDist (p : VertexDist α) : entropy p = _root_.entropy p.toFiniteDist := by
  classical
  unfold entropy _root_.entropy toFiniteDist
  simpa using
    (Fintype.sum_equiv (Fintype.equivFin α)
      (fun a : α => entropyTerm (p.prob a))
      (fun i : Fin (Fintype.card α) => entropyTerm (p.prob ((Fintype.equivFin α).symm i)))
      (by intro a; simp))

theorem expectedLength_toFiniteDist (p : VertexDist α) (ℓ : α → ℕ) :
    expectedLength p ℓ = _root_.expectedLength p.toFiniteDist (fun i => ℓ ((Fintype.equivFin α).symm i)) := by
  classical
  unfold expectedLength _root_.expectedLength
  simpa using
    (Fintype.sum_equiv (Fintype.equivFin α)
      (fun a : α => p.prob a * (ℓ a : ℝ))
      (fun i : Fin (Fintype.card α) => p.prob ((Fintype.equivFin α).symm i) * (ℓ ((Fintype.equivFin α).symm i) : ℝ))
      (by intro a; simp))

theorem finiteDist_supportSize_pos {n : ℕ} (q : FiniteDist n) : 0 < q.supportSize := by
  classical
  by_contra h
  have hsupp : q.supportSize = 0 := Nat.le_zero.mp (not_lt.mp h)
  have hcard : q.support.card = 0 := hsupp
  rw [Finset.card_eq_zero] at hcard
  have hzero : ∀ i, q.prob i = 0 := by
    intro i
    have hi : i ∉ q.support := by simpa [hcard]
    have hnot : ¬ 0 < q.prob i := by
      simpa [FiniteDist.support] using hi
    exact le_antisymm (not_lt.mp hnot) (q.nonneg i)
  have : (∑ i, q.prob i) = 0 := by
    refine Finset.sum_eq_zero ?_
    intro i hi
    exact hzero i
  have hq : (∑ i, q.prob i) = 1 := q.sum_one
  linarith

theorem entropy_le_log_card (p : VertexDist α) [Entropy.ClassicalEntropyAssumptions] :
    entropy p ≤ Real.log (Fintype.card α : ℝ) := by
  have h1 : _root_.entropy p.toFiniteDist ≤ Real.log (p.toFiniteDist.supportSize : ℝ) :=
    Entropy.entropy_le_log_card p.toFiniteDist
  have hsupp : 0 < p.toFiniteDist.supportSize := finiteDist_supportSize_pos p.toFiniteDist
  have hcard : p.toFiniteDist.supportSize ≤ Fintype.card α := by
    unfold FiniteDist.supportSize
    simpa using (Finset.card_le_univ (s := p.toFiniteDist.support))
  rw [entropy_toFiniteDist]
  refine h1.trans ?_
  apply Real.log_le_log
  · exact_mod_cast hsupp
  · exact_mod_cast hcard

theorem exists_codeLengths_le_entropy_bits_plus_one (p : VertexDist α)
    [Entropy.ClassicalCodingAssumptions] :
    ∃ ℓ : α → ℕ, expectedLength p ℓ ≤ entropy p / Real.log 2 + 1 := by
  rcases Coding.expectedLength_le_entropy_bits_plus_one p.toFiniteDist with ⟨ℓfin, hℓ⟩
  refine ⟨fun a => ℓfin (Fintype.equivFin α a), ?_⟩
  rw [expectedLength_toFiniteDist, entropy_toFiniteDist]
  simpa using hℓ

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

/-- Injective observations have no collision block larger than one class. -/
theorem maxFiberCard_le_one_of_injective (observe : α → β)
    (hinj : Function.Injective observe) :
    maxFiberCard observe ≤ 1 := by
  classical
  unfold maxFiberCard
  refine Finset.sup_le ?_
  intro b hb
  rw [Finset.card_le_one]
  intro x hx y hy
  exact hinj ((Finset.mem_filter.mp hx).2.trans (Finset.mem_filter.mp hy).2.symm)

/-- Any nonempty observation space has a fiber of size at least one. -/
theorem one_le_maxFiberCard_of_nonempty [Nonempty α] (observe : α → β) :
    1 ≤ maxFiberCard observe := by
  classical
  let a0 : α := Classical.choice ‹Nonempty α›
  have ha0 : a0 ∈ Finset.univ.filter (fun a : α => observe a = observe a0) := by
    simp [a0]
  have hpos : 0 < (Finset.univ.filter (fun a : α => observe a = observe a0)).card :=
    Finset.card_pos.mpr ⟨a0, ha0⟩
  exact (Nat.succ_le_of_lt hpos).trans (fiber_card_le_maxFiberCard observe (observe a0))

/-- Worst-case fiber size at most one is equivalent to injectivity. -/
theorem maxFiberCard_le_one_iff_injective [Nonempty α] (observe : α → β) :
    maxFiberCard observe ≤ 1 ↔ Function.Injective observe := by
  constructor
  · intro hle u v huv
    have hcard :
        (Finset.univ.filter (fun a : α => observe a = observe u)).card ≤ 1 :=
      (fiber_card_le_maxFiberCard observe (observe u)).trans hle
    rw [Finset.card_le_one] at hcard
    have hu : u ∈ Finset.univ.filter (fun a : α => observe a = observe u) := by
      simp
    have hv : v ∈ Finset.univ.filter (fun a : α => observe a = observe u) := by
      simp [huv]
    exact hcard u hu v hv
  · intro hinj
    exact maxFiberCard_le_one_of_injective observe hinj

/-- In the nonempty case, injectivity is equivalent to exact worst-case fiber size one. -/
theorem maxFiberCard_eq_one_iff_injective [Nonempty α] (observe : α → β) :
    maxFiberCard observe = 1 ↔ Function.Injective observe := by
  constructor
  · intro heq
    exact (maxFiberCard_le_one_iff_injective observe).mp (heq.le)
  · intro hinj
    exact le_antisymm (maxFiberCard_le_one_of_injective observe hinj)
      (one_le_maxFiberCard_of_nonempty observe)

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

/-- Refining the observation channel cannot increase worst-case collision multiplicity. -/
theorem maxFiberCard_mono_of_factors_through
    {γ : Type*} [Fintype γ] [DecidableEq γ]
    (observeFine : α → β) (observeCoarse : α → γ) (factor : β → γ)
    (hfactor : observeCoarse = factor ∘ observeFine) :
    maxFiberCard observeFine ≤ maxFiberCard observeCoarse := by
  classical
  unfold maxFiberCard
  refine Finset.sup_le ?_
  intro b hb
  have hsub :
      (Finset.univ.filter (fun a : α => observeFine a = b)) ⊆
        (Finset.univ.filter (fun a : α => observeCoarse a = factor b)) := by
    intro a ha
    rw [Finset.mem_filter] at ha ⊢
    rcases ha with ⟨hauniv, hobs⟩
    refine ⟨hauniv, ?_⟩
    rw [hfactor, Function.comp_apply, hobs]
  exact (Finset.card_le_card hsub).trans
    (fiber_card_le_maxFiberCard observeCoarse (factor b))

/-- On each observed symbol, a finer representation has no larger fiber than any coarsening that
factors through it. -/
theorem fiberCard_mono_of_factors_through
    {γ : Type*} [Fintype γ] [DecidableEq γ]
    (observeFine : α → β) (observeCoarse : α → γ) (factor : β → γ)
    (hfactor : observeCoarse = factor ∘ observeFine) (b : β) :
    (Finset.univ.filter (fun a : α => observeFine a = b)).card ≤
      (Finset.univ.filter (fun a : α => observeCoarse a = factor b)).card := by
  classical
  have hsub :
      (Finset.univ.filter (fun a : α => observeFine a = b)) ⊆
        (Finset.univ.filter (fun a : α => observeCoarse a = factor b)) := by
    intro a ha
    rw [Finset.mem_filter] at ha ⊢
    rcases ha with ⟨hauniv, hobs⟩
    refine ⟨hauniv, ?_⟩
    rw [hfactor, Function.comp_apply, hobs]
  exact Finset.card_le_card hsub

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

/-- A single-symbol zero-error tag alphabet is feasible exactly for injective observations. -/
theorem tagFeasible_one_iff_injective [Nonempty α] (observe : α → β) :
    TagFeasible observe 1 ↔ Function.Injective observe := by
  rw [tagFeasible_iff_maxFiberCard_le]
  exact maxFiberCard_le_one_iff_injective observe

section TaskRecovery

variable {δ : Type*} [Fintype δ] [DecidableEq δ]

/-- Distinct task labels realizable inside one observation fiber. -/
noncomputable def taskFiberLabels (observe : α → β) (task : α → δ) (b : β) : Finset δ :=
  (Finset.univ.filter (fun a : α => observe a = b)).image task

/-- Largest number of distinct task labels that survive one observation fiber. -/
noncomputable def maxTaskFiberCard (observe : α → β) (task : α → δ) : ℕ :=
  Finset.sup Finset.univ (fun b : β => (taskFiberLabels observe task b).card)

/-- Exact task recovery from the representation alone. -/
def TaskRecoverable (observe : α → β) (task : α → δ) : Prop :=
  ∃ decode : β → δ, ∀ a : α, decode (observe a) = task a

theorem taskFiberCard_le_fiberCard (observe : α → β) (task : α → δ) (b : β) :
    (taskFiberLabels observe task b).card ≤ (Finset.univ.filter (fun a : α => observe a = b)).card := by
  classical
  exact Finset.card_image_le

theorem maxTaskFiberCard_le_maxFiberCard (observe : α → β) (task : α → δ) :
    maxTaskFiberCard observe task ≤ maxFiberCard observe := by
  classical
  unfold maxTaskFiberCard maxFiberCard
  refine Finset.sup_le ?_
  intro b hb
  exact (taskFiberCard_le_fiberCard observe task b).trans (fiber_card_le_maxFiberCard observe b)

theorem taskRecoverable_iff_constant_on_fibers [Nonempty α] (observe : α → β) (task : α → δ) :
    TaskRecoverable observe task ↔ ∀ ⦃u v : α⦄, observe u = observe v → task u = task v := by
  constructor
  · intro hrec u v hEq
    rcases hrec with ⟨decode, hdecode⟩
    calc
      task u = decode (observe u) := (hdecode u).symm
      _ = decode (observe v) := by simpa [hEq]
      _ = task v := hdecode v
  · intro hconst
    let pick : β → α := fun b =>
      if h : ∃ a : α, observe a = b then Classical.choose h else Classical.choice inferInstance
    refine ⟨fun b => task (pick b), ?_⟩
    intro a
    have hex : ∃ x : α, observe x = observe a := ⟨a, rfl⟩
    have hEq : observe (pick (observe a)) = observe a := by
      simpa [pick, hex] using Classical.choose_spec hex
    simpa [pick] using hconst hEq

theorem taskRecoverable_iff_maxTaskFiberCard_le_one [Nonempty α] (observe : α → β) (task : α → δ) :
    TaskRecoverable observe task ↔ maxTaskFiberCard observe task ≤ 1 := by
  constructor
  · intro hrec
    have hconst := (taskRecoverable_iff_constant_on_fibers observe task).mp hrec
    unfold maxTaskFiberCard
    refine Finset.sup_le ?_
    intro b hb
    by_cases hzero : (taskFiberLabels observe task b).card = 0
    · simpa [hzero]
    · have hle : (taskFiberLabels observe task b).card ≤ 1 := by
        by_contra hgt
        have htwo : 2 ≤ (taskFiberLabels observe task b).card := by omega
        have hone : 1 < (taskFiberLabels observe task b).card := by omega
        obtain ⟨y, hy, z, hz, hyz⟩ := (Finset.one_lt_card).1 hone
        rcases Finset.mem_image.mp hy with ⟨ay, hay, hty⟩
        rcases Finset.mem_image.mp hz with ⟨az, haz, htz⟩
        rw [Finset.mem_filter] at hay haz
        have hsame : observe ay = observe az := by simpa [hay.2, haz.2]
        have : task ay = task az := hconst hsame
        exact hyz (by simpa [hty, htz] using this)
      exact hle
  · intro hmax
    have hconst : ∀ ⦃u v : α⦄, observe u = observe v → task u = task v := by
      intro u v hEq
      by_contra hneq
      have hcard : 2 ≤ maxTaskFiberCard observe task := by
        unfold maxTaskFiberCard
        have hy : task u ∈ taskFiberLabels observe task (observe u) := by
          unfold taskFiberLabels
          exact Finset.mem_image.mpr ⟨u, by simp, rfl⟩
        have hz : task v ∈ taskFiberLabels observe task (observe u) := by
          unfold taskFiberLabels
          exact Finset.mem_image.mpr ⟨v, by simp [hEq], rfl⟩
        have htwo : 2 ≤ (taskFiberLabels observe task (observe u)).card := by
          have hone : 1 < (taskFiberLabels observe task (observe u)).card :=
            Finset.one_lt_card.mpr ⟨task u, hy, task v, hz, hneq⟩
          omega
        exact htwo.trans (Finset.le_sup (s := Finset.univ) (f := fun b : β => (taskFiberLabels observe task b).card) (by simp : observe u ∈ Finset.univ))
      have : ¬ 2 ≤ 1 := by decide
      exact this (hcard.trans hmax)
    exact (taskRecoverable_iff_constant_on_fibers observe task).2 hconst

theorem maxTaskFiberCard_mono_of_factors_through
    {γ : Type*} [Fintype γ] [DecidableEq γ]
    (observeFine : α → β) (observeCoarse : α → γ) (task : α → δ) (factor : β → γ)
    (hfactor : observeCoarse = factor ∘ observeFine) :
    maxTaskFiberCard observeFine task ≤ maxTaskFiberCard observeCoarse task := by
  classical
  unfold maxTaskFiberCard
  refine Finset.sup_le ?_
  intro b hb
  have hsub : taskFiberLabels observeFine task b ⊆ taskFiberLabels observeCoarse task (factor b) := by
    intro y hy
    rcases Finset.mem_image.mp hy with ⟨a, ha, rfl⟩
    rw [Finset.mem_filter] at ha
    exact Finset.mem_image.mpr ⟨a, by simpa [hfactor, Function.comp_apply, ha.2] using ha.1, rfl⟩
  exact (Finset.card_le_card hsub).trans
    (Finset.le_sup (s := Finset.univ) (f := fun c : γ => (taskFiberLabels observeCoarse task c).card) (by simp : factor b ∈ Finset.univ))

theorem taskRecoverable_pair_iff [Nonempty α]
    {γ : Type*} [Fintype γ] [DecidableEq γ]
    (observe : α → β) (side : α → γ) (task : α → δ) :
    TaskRecoverable (fun a => (observe a, side a)) task ↔
      ∀ ⦃u v : α⦄, observe u = observe v → side u = side v → task u = task v := by
  constructor
  · intro hrec u v hobs hside
    have hpair : (observe u, side u) = (observe v, side v) := by simpa [hobs, hside]
    exact (taskRecoverable_iff_constant_on_fibers (observe := fun a => (observe a, side a)) (task := task)).mp hrec hpair
  · intro hconst
    refine (taskRecoverable_iff_constant_on_fibers (observe := fun a => (observe a, side a)) (task := task)).2 ?_
    intro u v hpair
    have hobs : observe u = observe v := by simpa using congrArg Prod.fst hpair
    have hside : side u = side v := by simpa using congrArg Prod.snd hpair
    exact hconst hobs hside

theorem maxTaskFiberCard_pair_le_left
    {γ : Type*} [Fintype γ] [DecidableEq γ]
    (observe : α → β) (side : α → γ) (task : α → δ) :
    maxTaskFiberCard (fun a => (observe a, side a)) task ≤ maxTaskFiberCard observe task := by
  simpa using
    maxTaskFiberCard_mono_of_factors_through
      (observeFine := fun a => (observe a, side a))
      (observeCoarse := observe) (task := task) Prod.fst rfl

end TaskRecovery


/-- The finite observation fiber over `b`. -/
noncomputable def observeFiber (observe : α → β) (b : β) : Finset α :=
  Finset.univ.filter (fun a : α => observe a = b)

theorem observeFiber_prod_eq_product
    {γ δ : Type*} [Fintype γ] [Fintype δ] [DecidableEq γ] [DecidableEq δ]
    (observe₁ : α → β) (observe₂ : γ → δ) (b : β) (d : δ) :
    observeFiber (fun p : α × γ => (observe₁ p.1, observe₂ p.2)) (b, d) =
      (observeFiber observe₁ b).product (observeFiber observe₂ d) := by
  ext p
  simp [observeFiber]

theorem observeFiber_prod_card
    {γ δ : Type*} [Fintype γ] [Fintype δ] [DecidableEq γ] [DecidableEq δ]
    (observe₁ : α → β) (observe₂ : γ → δ) (b : β) (d : δ) :
    (observeFiber (fun p : α × γ => (observe₁ p.1, observe₂ p.2)) (b, d)).card =
      (observeFiber observe₁ b).card * (observeFiber observe₂ d).card := by
  rw [observeFiber_prod_eq_product]
  simp

theorem maxFiberCard_prod
    {γ δ : Type*} [Fintype γ] [Fintype δ] [DecidableEq γ] [DecidableEq δ]
    [Nonempty β] [Nonempty δ]
    (observe₁ : α → β) (observe₂ : γ → δ) :
    maxFiberCard (fun p : α × γ => (observe₁ p.1, observe₂ p.2)) =
      maxFiberCard observe₁ * maxFiberCard observe₂ := by
  classical
  have hle : maxFiberCard (fun p : α × γ => (observe₁ p.1, observe₂ p.2)) ≤
      maxFiberCard observe₁ * maxFiberCard observe₂ := by
    unfold maxFiberCard
    refine Finset.sup_le ?_
    intro bd hbd
    rcases bd with ⟨b, d⟩
    have hcardpair :
        (Finset.univ.filter (fun a : α × γ => (observe₁ a.1, observe₂ a.2) = (b, d))).card =
          (observeFiber observe₁ b).card * (observeFiber observe₂ d).card := by
      simpa [observeFiber] using observeFiber_prod_card observe₁ observe₂ b d
    rw [hcardpair]
    exact Nat.mul_le_mul (fiber_card_le_maxFiberCard observe₁ b) (fiber_card_le_maxFiberCard observe₂ d)
  obtain ⟨b, hb, hbmax⟩ := Finset.exists_mem_eq_sup (s := Finset.univ) Finset.univ_nonempty
    (fun b : β => (observeFiber observe₁ b).card)
  obtain ⟨d, hd, hdmax⟩ := Finset.exists_mem_eq_sup (s := Finset.univ) Finset.univ_nonempty
    (fun d : δ => (observeFiber observe₂ d).card)
  have hge : maxFiberCard observe₁ * maxFiberCard observe₂ ≤
      maxFiberCard (fun p : α × γ => (observe₁ p.1, observe₂ p.2)) := by
    calc
      maxFiberCard observe₁ * maxFiberCard observe₂
          = (observeFiber observe₁ b).card * (observeFiber observe₂ d).card := by
              have hleft : maxFiberCard observe₁ = (observeFiber observe₁ b).card := by
                simpa [maxFiberCard] using hbmax
              have hright : maxFiberCard observe₂ = (observeFiber observe₂ d).card := by
                simpa [maxFiberCard] using hdmax
              rw [hleft, hright]
      _ = (observeFiber (fun p : α × γ => (observe₁ p.1, observe₂ p.2)) (b, d)).card := by
            rw [observeFiber_prod_card]
      _ ≤ maxFiberCard (fun p : α × γ => (observe₁ p.1, observe₂ p.2)) :=
            fiber_card_le_maxFiberCard _ (b, d)
  exact le_antisymm hle hge

/-- The classes in fiber `b` decoded correctly by a deterministic tag decoder. -/
noncomputable def correctlyDecodedOnFiber (observe : α → β) {n : ℕ}
    (tag : α → Fin n) (decode : β → Fin n → α) (b : β) : Finset α :=
  (observeFiber observe b).filter (fun a => decode b (tag a) = a)

/-- Uniform success rate on a fixed observation fiber. -/
noncomputable def uniformFiberSuccessRate (observe : α → β) {n : ℕ}
    (tag : α → Fin n) (decode : β → Fin n → α) (b : β) : ℝ :=
  ((correctlyDecodedOnFiber observe tag decode b).card : ℝ) /
    ((observeFiber observe b).card : ℝ)

/-- Uniform error rate on a fixed observation fiber. -/
noncomputable def uniformFiberErrorRate (observe : α → β) {n : ℕ}
    (tag : α → Fin n) (decode : β → Fin n → α) (b : β) : ℝ :=
  1 - uniformFiberSuccessRate observe tag decode b

/-- On a fixed observation fiber, a deterministic decoder can be correct on at most
one class per tag value. -/
theorem correctlyDecodedOnFiber_card_le_tag_alphabet (observe : α → β) {n : ℕ}
    (tag : α → Fin n) (decode : β → Fin n → α) (b : β) :
    (correctlyDecodedOnFiber observe tag decode b).card ≤ n := by
  classical
  let s := correctlyDecodedOnFiber observe tag decode b
  have hinj : Set.InjOn tag (↑s : Set α) := by
    intro u hu v hv hEq
    have hu' : decode b (tag u) = u := by
      exact (Finset.mem_filter.mp hu).2
    have hv' : decode b (tag v) = v := by
      exact (Finset.mem_filter.mp hv).2
    calc
      u = decode b (tag u) := hu'.symm
      _ = decode b (tag v) := by simpa [hEq]
      _ = v := hv'
  have himage : (s.image tag).card = s.card := by
    exact Finset.card_image_of_injOn (fun a ha b hb hab => hinj ha hb hab)
  have hbound : (s.image tag).card ≤ n := by
    simpa using (Finset.card_le_univ (s := s.image tag))
  exact himage.ge.trans hbound

/-- On a nonempty observation fiber, the uniform success probability is bounded by the
ratio of available tag states to fiber size. -/
theorem uniformFiberSuccessRate_le_tag_ratio (observe : α → β) {n : ℕ}
    (tag : α → Fin n) (decode : β → Fin n → α) (b : β)
    (hpos : 0 < (observeFiber observe b).card) :
    uniformFiberSuccessRate observe tag decode b ≤
      (n : ℝ) / ((observeFiber observe b).card : ℝ) := by
  unfold uniformFiberSuccessRate
  exact div_le_div_of_nonneg_right
    (by exact_mod_cast correctlyDecodedOnFiber_card_le_tag_alphabet observe tag decode b)
    (by exact_mod_cast hpos.le)

/-- On a nonempty observation fiber, the uniform error probability is bounded below by
`1 - n / |fiber|`. -/
theorem uniformFiberErrorRate_ge_one_sub_tag_ratio (observe : α → β) {n : ℕ}
    (tag : α → Fin n) (decode : β → Fin n → α) (b : β)
    (hpos : 0 < (observeFiber observe b).card) :
    1 - (n : ℝ) / ((observeFiber observe b).card : ℝ) ≤
      uniformFiberErrorRate observe tag decode b := by
  unfold uniformFiberErrorRate
  linarith [uniformFiberSuccessRate_le_tag_ratio observe tag decode b hpos]

/-- A representation-adaptive bit budget is feasible if every observation fiber fits into the
available binary labels assigned to that fiber. -/
def AdaptiveBitBudgetFeasible (observe : α → β) (ℓ : β → ℕ) : Prop :=
  ∀ b, (observeFiber observe b).card ≤ 2 ^ ℓ b

/-- The pointwise optimal number of extra bits after observing `b`. -/
noncomputable def optimalFiberBitLength (observe : α → β) (b : β) : ℕ :=
  Nat.clog 2 (observeFiber observe b).card

/-- Expected auxiliary bit length under a distribution on observation values. -/
noncomputable def expectedAdaptiveBitLength (p : VertexDist β) (ℓ : β → ℕ) : ℝ :=
  ∑ b, p.prob b * (ℓ b : ℝ)

/-- Expected length of the pointwise optimal adaptive budget. -/
noncomputable def optimalExpectedAdaptiveBitLength (observe : α → β) (p : VertexDist β) : ℝ :=
  expectedAdaptiveBitLength p (optimalFiberBitLength observe)

/-- The pointwise optimal adaptive budget is feasible on every fiber. -/
theorem optimalFiberBitLength_feasible (observe : α → β) :
    AdaptiveBitBudgetFeasible observe (optimalFiberBitLength observe) := by
  intro b
  unfold optimalFiberBitLength
  exact Nat.le_pow_clog Nat.one_lt_two _

/-- Any feasible adaptive bit budget dominates the pointwise optimal budget. -/
theorem optimalFiberBitLength_le_of_feasible (observe : α → β) (ℓ : β → ℕ)
    (hfeas : AdaptiveBitBudgetFeasible observe ℓ) (b : β) :
    optimalFiberBitLength observe b ≤ ℓ b := by
  unfold optimalFiberBitLength
  exact Nat.clog_le_of_le_pow (hfeas b)

/-- Coarsening a representation cannot decrease the pointwise optimal adaptive bit budget. -/
theorem optimalFiberBitLength_mono_of_factors_through
    {γ : Type*} [Fintype γ] [DecidableEq γ]
    (observeFine : α → β) (observeCoarse : α → γ) (factor : β → γ)
    (hfactor : observeCoarse = factor ∘ observeFine) (b : β) :
    optimalFiberBitLength observeFine b ≤ optimalFiberBitLength observeCoarse (factor b) := by
  unfold optimalFiberBitLength
  simpa [observeFiber] using
    Nat.clog_mono_right 2 (fiberCard_mono_of_factors_through observeFine observeCoarse factor hfactor b)

/-- The expected auxiliary bit length of any feasible adaptive scheme is bounded below by the
expectation of the pointwise optimal fiber budget. -/
theorem optimalExpectedAdaptiveBitLength_le (observe : α → β) (p : VertexDist β)
    (ℓ : β → ℕ) (hfeas : AdaptiveBitBudgetFeasible observe ℓ) :
    optimalExpectedAdaptiveBitLength observe p ≤ expectedAdaptiveBitLength p ℓ := by
  unfold optimalExpectedAdaptiveBitLength expectedAdaptiveBitLength
  refine Finset.sum_le_sum ?_
  intro b hb
  have hnat : optimalFiberBitLength observe b ≤ ℓ b :=
    optimalFiberBitLength_le_of_feasible observe ℓ hfeas b
  have hreal : ((optimalFiberBitLength observe b : ℕ) : ℝ) ≤ (ℓ b : ℝ) := by
    exact_mod_cast hnat
  exact mul_le_mul_of_nonneg_left hreal (p.nonneg b)

noncomputable def fiberMass (p : VertexDist α) (observe : α → β) (b : β) : ℝ :=
  (p.pushforward observe).prob b

noncomputable def conditionalOnFiber (p : VertexDist α) (observe : α → β) (b : β)
    (hb : 0 < fiberMass p observe b) : VertexDist {a : α // observe a = b} where
  prob x := p.prob x / fiberMass p observe b
  nonneg x := by
    apply div_nonneg
    · exact p.nonneg x
    · exact le_of_lt hb
  sum_one := by
    classical
    unfold fiberMass
    have hsum :
        ∑ x : {a : α // observe a = b}, p.prob x = (p.pushforward observe).prob b := by
      simpa [VertexDist.pushforward] using
        (Finset.sum_subtype_eq_sum_filter (s := Finset.univ)
          (p := fun a : α => observe a = b)
          (f := fun a : α => p.prob a))
    calc
      ∑ x : {a : α // observe a = b}, p.prob x / (p.pushforward observe).prob b
          = (∑ x : {a : α // observe a = b}, p.prob x) / (p.pushforward observe).prob b := by
              rw [Finset.sum_div]
      _ = (p.pushforward observe).prob b / (p.pushforward observe).prob b := by rw [hsum]
      _ = 1 := by exact div_self hb.ne'

noncomputable def conditionalEntropyGiven (p : VertexDist α) (observe : α → β) : ℝ :=
  ∑ b, if hb : 0 < (p.pushforward observe).prob b then
    (p.pushforward observe).prob b * (conditionalOnFiber p observe b hb).entropy
  else 0

theorem conditionalOnFiber_supportSize_le_fiberCard (p : VertexDist α) (observe : α → β)
    (b : β) (hb : 0 < fiberMass p observe b) :
    (conditionalOnFiber p observe b hb).supportSize ≤ (observeFiber observe b).card := by
  classical
  unfold VertexDist.supportSize VertexDist.support observeFiber
  simpa [Fintype.card_subtype] using
    (Finset.card_le_univ
      (s := ({x : {a : α // observe a = b} | 0 < (conditionalOnFiber p observe b hb).prob x} : Finset _)))

theorem conditionalOnFiber_entropy_le_log_fiberCard (p : VertexDist α) (observe : α → β)
    (b : β) (hb : 0 < fiberMass p observe b) [Entropy.ClassicalEntropyAssumptions] :
    (conditionalOnFiber p observe b hb).entropy ≤ Real.log ((observeFiber observe b).card : ℝ) := by
  simpa [Fintype.card_subtype] using
    (VertexDist.entropy_le_log_card (p := conditionalOnFiber p observe b hb))

theorem conditionalEntropyGiven_le_log2_mul_expectedAdaptiveBitLength
    (p : VertexDist α) (observe : α → β) [Entropy.ClassicalEntropyAssumptions]
    (ℓ : β → ℕ) (hfeas : AdaptiveBitBudgetFeasible observe ℓ) :
    conditionalEntropyGiven p observe ≤ Real.log 2 * expectedAdaptiveBitLength (p.pushforward observe) ℓ := by
  classical
  unfold conditionalEntropyGiven expectedAdaptiveBitLength
  have hsum :
      ∑ b,
        (if hb : 0 < (p.pushforward observe).prob b then
          (p.pushforward observe).prob b * (conditionalOnFiber p observe b hb).entropy
        else 0)
      ≤ ∑ b, (p.pushforward observe).prob b * ((ℓ b : ℝ) * Real.log 2) := by
    refine Finset.sum_le_sum ?_
    intro b hbfin
    by_cases hbpos : 0 < (p.pushforward observe).prob b
    · simp [hbpos]
      have hcardlog :
          (conditionalOnFiber p observe b hbpos).entropy ≤ Real.log ((observeFiber observe b).card : ℝ) :=
        conditionalOnFiber_entropy_le_log_fiberCard p observe b hbpos
      have hpow : (observeFiber observe b).card ≤ 2 ^ ℓ b := hfeas b
      have hfiberpos : 0 < (observeFiber observe b).card := by
        by_contra hzero
        have hempty : observeFiber observe b = ∅ := by
          exact Finset.card_eq_zero.mp (Nat.le_zero.mp (not_lt.mp hzero))
        have hmasszero : fiberMass p observe b = 0 := by
          have hempty' : Finset.univ.filter (fun a : α => observe a = b) = ∅ := by
            simpa [observeFiber] using hempty
          unfold fiberMass VertexDist.pushforward
          simp [hempty']
        exact hbpos.ne' hmasszero
      have hlogpow : Real.log ((observeFiber observe b).card : ℝ) ≤ Real.log ((2 ^ ℓ b : ℕ) : ℝ) := by
        apply Real.log_le_log
        · exact_mod_cast hfiberpos
        · exact_mod_cast hpow
      have hpowlog : Real.log ((2 ^ ℓ b : ℕ) : ℝ) = (ℓ b : ℝ) * Real.log 2 := by
        rw [show ((2 ^ ℓ b : ℕ) : ℝ) = (2 : ℝ) ^ (ℓ b : ℝ) by simp, Real.log_rpow (by norm_num)]
      have hbase :
          (conditionalOnFiber p observe b hbpos).entropy ≤ (ℓ b : ℝ) * Real.log 2 := by
        exact (hcardlog.trans hlogpow).trans_eq hpowlog
      have hmul :
          (p.pushforward observe).prob b * (conditionalOnFiber p observe b hbpos).entropy ≤
          (p.pushforward observe).prob b * ((ℓ b : ℝ) * Real.log 2) :=
        mul_le_mul_of_nonneg_left hbase ((p.pushforward observe).nonneg b)
      simpa [hbpos] using hmul
    · have hnonneg :
          0 ≤ (p.pushforward observe).prob b * ((ℓ b : ℝ) * Real.log 2) := by
        have hlog2pos : 0 < Real.log 2 := by
          exact Real.log_pos (by norm_num : (1 : ℝ) < 2)
        apply mul_nonneg
        · exact (p.pushforward observe).nonneg b
        · apply mul_nonneg
          · exact_mod_cast Nat.zero_le (ℓ b)
          · exact le_of_lt hlog2pos
      simpa [hbpos] using hnonneg
  calc
    conditionalEntropyGiven p observe ≤ ∑ b, (p.pushforward observe).prob b * ((ℓ b : ℝ) * Real.log 2) := hsum
    _ = Real.log 2 * expectedAdaptiveBitLength (p.pushforward observe) ℓ := by
      unfold expectedAdaptiveBitLength
      calc
        ∑ b, (p.pushforward observe).prob b * ((ℓ b : ℝ) * Real.log 2)
            = ∑ b, Real.log 2 * ((p.pushforward observe).prob b * (ℓ b : ℝ)) := by
                refine Finset.sum_congr rfl ?_
                intro b hb
                ring
        _ = Real.log 2 * ∑ b, (p.pushforward observe).prob b * (ℓ b : ℝ) := by
              rw [Finset.mul_sum]

noncomputable def conditionalExpectedCodeLength (p : VertexDist α) (observe : α → β)
    (L : ∀ b, 0 < (p.pushforward observe).prob b → {a : α // observe a = b} → ℕ) : ℝ :=
  ∑ b, if hb : 0 < (p.pushforward observe).prob b then
    (p.pushforward observe).prob b * VertexDist.expectedLength (conditionalOnFiber p observe b hb) (L b hb)
  else 0

theorem exists_conditionalCodes_expectedLength_le_entropy_bits_plus_one
    (p : VertexDist α) (observe : α → β) [Entropy.ClassicalCodingAssumptions] :
    ∃ L : ∀ b, 0 < (p.pushforward observe).prob b → {a : α // observe a = b} → ℕ,
      conditionalExpectedCodeLength p observe L ≤ conditionalEntropyGiven p observe / Real.log 2 + 1 := by
  classical
  choose L hL using fun b hb =>
    VertexDist.exists_codeLengths_le_entropy_bits_plus_one (p := conditionalOnFiber p observe b hb)
  refine ⟨L, ?_⟩
  have hmain :
      ∑ b,
        (if hb : 0 < (p.pushforward observe).prob b then
          (p.pushforward observe).prob b * VertexDist.expectedLength (conditionalOnFiber p observe b hb) (L b hb)
        else 0)
      ≤
      ∑ b,
        (if hb : 0 < (p.pushforward observe).prob b then
          (p.pushforward observe).prob b * ((conditionalOnFiber p observe b hb).entropy / Real.log 2 + 1)
        else 0) := by
    refine Finset.sum_le_sum ?_
    intro b hbfin
    by_cases hbpos : 0 < (p.pushforward observe).prob b
    · simpa [hbpos] using mul_le_mul_of_nonneg_left (hL b hbpos) ((p.pushforward observe).nonneg b)
    · simp [hbpos]
  have hsplit :
      ∑ b,
        (if hb : 0 < (p.pushforward observe).prob b then
          (p.pushforward observe).prob b * ((conditionalOnFiber p observe b hb).entropy / Real.log 2 + 1)
        else 0)
      =
      ∑ b, (if hb : 0 < (p.pushforward observe).prob b then
          (p.pushforward observe).prob b * (conditionalOnFiber p observe b hb).entropy / Real.log 2 else 0)
      +
      ∑ b, (if hb : 0 < (p.pushforward observe).prob b then (p.pushforward observe).prob b else 0) := by
    calc
      ∑ b,
          (if hb : 0 < (p.pushforward observe).prob b then
            (p.pushforward observe).prob b * ((conditionalOnFiber p observe b hb).entropy / Real.log 2 + 1)
          else 0)
        =
        ∑ b,
          ((if hb : 0 < (p.pushforward observe).prob b then
            (p.pushforward observe).prob b * (conditionalOnFiber p observe b hb).entropy / Real.log 2 else 0)
          +
          (if hb : 0 < (p.pushforward observe).prob b then (p.pushforward observe).prob b else 0)) := by
              refine Finset.sum_congr rfl ?_
              intro b hb
              by_cases hbpos : 0 < (p.pushforward observe).prob b
              · simp [hbpos]
                ring
              · simp [hbpos]
      _ =
        ∑ b, (if hb : 0 < (p.pushforward observe).prob b then
            (p.pushforward observe).prob b * (conditionalOnFiber p observe b hb).entropy / Real.log 2 else 0)
        +
        ∑ b, (if hb : 0 < (p.pushforward observe).prob b then (p.pushforward observe).prob b else 0) := by
              rw [Finset.sum_add_distrib]
  have hconddiv :
      ∑ b, (if hb : 0 < (p.pushforward observe).prob b then
        (p.pushforward observe).prob b * (conditionalOnFiber p observe b hb).entropy / Real.log 2 else 0)
      = conditionalEntropyGiven p observe / Real.log 2 := by
    unfold conditionalEntropyGiven
    rw [Finset.sum_div]
    refine Finset.sum_congr rfl ?_
    intro b hb
    by_cases hbpos : 0 < (p.pushforward observe).prob b
    · simp [hbpos]
    · simp [hbpos]
  have hmassone :
      ∑ b, (if hb : 0 < (p.pushforward observe).prob b then (p.pushforward observe).prob b else 0) = 1 := by
    calc
      ∑ b, (if hb : 0 < (p.pushforward observe).prob b then (p.pushforward observe).prob b else 0)
        = ∑ b, (p.pushforward observe).prob b := by
            refine Finset.sum_congr rfl ?_
            intro b hb
            by_cases hbpos : 0 < (p.pushforward observe).prob b
            · simp [hbpos]
            · have hzero : (p.pushforward observe).prob b = 0 :=
                le_antisymm (not_lt.mp hbpos) ((p.pushforward observe).nonneg b)
              simp [hbpos, hzero]
      _ = 1 := (p.pushforward observe).sum_one
  change
    (∑ b,
      if hb : 0 < (p.pushforward observe).prob b then
        (p.pushforward observe).prob b * VertexDist.expectedLength (conditionalOnFiber p observe b hb) (L b hb)
      else 0) ≤ conditionalEntropyGiven p observe / Real.log 2 + 1
  calc
    (∑ b,
        if hb : 0 < (p.pushforward observe).prob b then
          (p.pushforward observe).prob b * VertexDist.expectedLength (conditionalOnFiber p observe b hb) (L b hb)
        else 0)
      ≤ ∑ b,
          (if hb : 0 < (p.pushforward observe).prob b then
            (p.pushforward observe).prob b * ((conditionalOnFiber p observe b hb).entropy / Real.log 2 + 1)
          else 0) := hmain
    _ =
      (∑ b, (if hb : 0 < (p.pushforward observe).prob b then
          (p.pushforward observe).prob b * (conditionalOnFiber p observe b hb).entropy / Real.log 2 else 0))
      +
      ∑ b, (if hb : 0 < (p.pushforward observe).prob b then (p.pushforward observe).prob b else 0) := hsplit
    _ = conditionalEntropyGiven p observe / Real.log 2 + 1 := by
      rw [hconddiv, hmassone]

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

section DistinguishingQueries

variable {α ι : Type*} [Fintype α] [DecidableEq α] [DecidableEq ι]

/-- A finite set of primitive coordinates distinguishes classes if every distinct pair differs on
at least one coordinate from the set. -/
def CoordinateDistinguishing (observe : α → ι → Bool) (S : Finset ι) : Prop :=
  ∀ ⦃u v : α⦄, u ≠ v → ∃ i ∈ S, observe u i ≠ observe v i

instance instDecidableCoordinateDistinguishing (observe : α → ι → Bool) (S : Finset ι) :
    Decidable (CoordinateDistinguishing observe S) := by
  unfold CoordinateDistinguishing
  infer_instance

/-- Inclusion-minimal distinguishing family, checked by failure after removing any used
coordinate. -/
def InclusionMinimalCoordinateDistinguishing (observe : α → ι → Bool) (S : Finset ι) : Prop :=
  CoordinateDistinguishing observe S ∧
    ∀ i ∈ S, ¬ CoordinateDistinguishing observe (S.erase i)

instance instDecidableInclusionMinimalCoordinateDistinguishing
    (observe : α → ι → Bool) (S : Finset ι) :
    Decidable (InclusionMinimalCoordinateDistinguishing observe S) := by
  unfold InclusionMinimalCoordinateDistinguishing
  infer_instance

/-- Finite explicit four-class profile system witnessing that unrestricted inclusion-minimal
distinguishing families need not be equicardinal. -/
def reviewerCounterexampleObserve : Fin 4 → Fin 4 → Bool
  | 0, 0 => false
  | 0, 1 => false
  | 0, 2 => false
  | 0, 3 => false
  | 1, 0 => false
  | 1, 1 => false
  | 1, 2 => false
  | 1, 3 => true
  | 2, 0 => false
  | 2, 1 => true
  | 2, 2 => true
  | 2, 3 => false
  | 3, 0 => true
  | 3, 1 => false
  | 3, 2 => true
  | 3, 3 => true

/-- First inclusion-minimal distinguishing family in the four-class counterexample. -/
def reviewerCounterexampleS1 : Finset (Fin 4) := {2, 3}

/-- Second inclusion-minimal distinguishing family in the four-class counterexample. -/
def reviewerCounterexampleS2 : Finset (Fin 4) := {0, 1, 3}

theorem reviewerCounterexample_s1_minimal :
    InclusionMinimalCoordinateDistinguishing reviewerCounterexampleObserve
      reviewerCounterexampleS1 := by
  native_decide

theorem reviewerCounterexample_s2_minimal :
    InclusionMinimalCoordinateDistinguishing reviewerCounterexampleObserve
      reviewerCounterexampleS2 := by
  native_decide

theorem reviewerCounterexample_s1_card_ne_s2_card :
    reviewerCounterexampleS1.card ≠ reviewerCounterexampleS2.card := by
  native_decide

/-- In the unrestricted coordinate-query model, inclusion-minimal distinguishing families need
not have the same cardinality. -/
theorem inclusionMinimalCoordinateDistinguishing_not_equicardinal :
    ∃ S T : Finset (Fin 4),
      InclusionMinimalCoordinateDistinguishing reviewerCounterexampleObserve S ∧
      InclusionMinimalCoordinateDistinguishing reviewerCounterexampleObserve T ∧
      S.card ≠ T.card := by
  refine ⟨reviewerCounterexampleS1, reviewerCounterexampleS2,
    reviewerCounterexample_s1_minimal, reviewerCounterexample_s2_minimal,
    reviewerCounterexample_s1_card_ne_s2_card⟩

end DistinguishingQueries

end GraphEntropy
end Ssot
