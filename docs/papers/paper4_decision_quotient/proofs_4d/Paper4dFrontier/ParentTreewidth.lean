import Paper4dFrontier.RealTreewidthWitnesses
import Mathlib.Data.Finset.Card
import Mathlib.Combinatorics.SimpleGraph.Finite
import Mathlib.Data.Sym.Sym2
import Mathlib.Data.Fintype.EquivFin
import Mathlib.Tactic

namespace Paper4dFrontier

open Classical
open SimpleGraph

/-- Honest rooted-tree dependency structure: every dependency points to a
strictly smaller coordinate and every coordinate has at most one parent. -/
def ParentTreeStructured {n : ℕ} (deps : Fin n → Finset (Fin n)) : Prop :=
  DecisionQuotient.TreeStructured deps ∧ ∀ c, (deps c).card ≤ 1

theorem parentTreeStructured_implies_treeStructured {n : ℕ}
    {deps : Fin n → Finset (Fin n)} (hpt : ParentTreeStructured deps) :
    DecisionQuotient.TreeStructured deps := hpt.1

theorem treeStructured_and_unique_parent_iff_parentTreeStructured {n : ℕ}
    {deps : Fin n → Finset (Fin n)} :
    ParentTreeStructured deps ↔
      (DecisionQuotient.TreeStructured deps ∧ ∀ c, (deps c).card ≤ 1) := by
  rfl

noncomputable def parentOf {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) (c : Fin n) : Option (Fin n) := by
  classical
  by_cases h : (deps c).Nonempty
  · have hcard : (deps c).card = 1 := by
      have hle : (deps c).card ≤ 1 := hpt.2 c
      have hpos : 0 < (deps c).card := Finset.card_pos.mpr h
      omega
    exact some ((Finset.card_eq_one.mp hcard).choose)
  · exact none

theorem parentOf_eq_none_iff {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) (c : Fin n) :
    parentOf deps hpt c = none ↔ deps c = ∅ := by
  classical
  unfold parentOf
  by_cases h : (deps c).Nonempty
  · simp [h]
    have hcard : (deps c).card = 1 := by
      have hle : (deps c).card ≤ 1 := hpt.2 c
      have hpos : 0 < (deps c).card := Finset.card_pos.mpr h
      omega
    exact Finset.nonempty_iff_ne_empty.mp h
  · have hempty : deps c = ∅ := by
      exact Finset.not_nonempty_iff_eq_empty.mp h
    simp [hempty]

theorem parentOf_mem {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) (c : Fin n) {d : Fin n}
    (hd : parentOf deps hpt c = some d) : d ∈ deps c := by
  classical
  unfold parentOf at hd
  by_cases h : (deps c).Nonempty
  · have hcard : (deps c).card = 1 := by
      have hle : (deps c).card ≤ 1 := hpt.2 c
      have hpos : 0 < (deps c).card := Finset.card_pos.mpr h
      omega
    rcases Finset.card_eq_one.mp hcard with ⟨x, hx⟩
    simp [hx] at hd
    subst hd
    simpa [hx] using (by rw [hx]; simp)
  · simp [h] at hd

theorem parentOf_lt {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) (c d : Fin n)
    (hd : parentOf deps hpt c = some d) : d < c :=
  hpt.1 c d (parentOf_mem deps hpt c hd)

theorem parentOf_eq_some_of_mem {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) (c d : Fin n)
    (hd : d ∈ deps c) : parentOf deps hpt c = some d := by
  classical
  unfold parentOf
  have hne : (deps c).Nonempty := ⟨d, hd⟩
  simp [hne]
  have hcard : (deps c).card = 1 := by
    have hle : (deps c).card ≤ 1 := hpt.2 c
    have hpos : 0 < (deps c).card := Finset.card_pos.mpr hne
    omega
  rcases Finset.card_eq_one.mp hcard with ⟨x, hx⟩
  have hxd : d = x := by
    rw [hx] at hd
    simpa using hd
  simp [hx, hxd]

theorem parentOf_ne_self {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) (c : Fin n) : parentOf deps hpt c ≠ some c := by
  intro h
  exact Fin.ne_of_lt (parentOf_lt deps hpt c c h) rfl

def parentBagGraph {n : ℕ} (deps : Fin n → Finset (Fin n)) (hpt : ParentTreeStructured deps) :
    SimpleGraph (Option (Fin n)) where
  Adj x y :=
    (∃ c : Fin n, x = some c ∧ parentOf deps hpt c = y) ∨
    (∃ c : Fin n, y = some c ∧ parentOf deps hpt c = x)
  symm := by
    intro x y h
    rcases h with h | h
    · exact Or.inr h
    · exact Or.inl h
  loopless := by
    intro x h
    rcases h with ⟨c, hx, hc⟩ | ⟨c, hx, hc⟩
    · subst hx
      exact parentOf_ne_self deps hpt c hc
    · subst hx
      exact parentOf_ne_self deps hpt c hc

theorem parentBagGraph_adj_edgeOf {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) (c : Fin n) :
    (parentBagGraph deps hpt).Adj (some c) (parentOf deps hpt c) := by
  exact Or.inl ⟨c, rfl, rfl⟩

theorem parentBagGraph_root_reachable {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) :
    ∀ c : Fin n, (parentBagGraph deps hpt).Reachable (some c) none := by
  intro c
  let P : ℕ → Prop := fun k => ∀ d : Fin n, d.1 = k → (parentBagGraph deps hpt).Reachable (some d) none
  have hP : ∀ k, P k := by
    intro k
    induction k using Nat.strong_induction_on with
    | h k ih =>
        intro d hd
        cases hpc : parentOf deps hpt d with
        | none =>
            simpa [hpc] using (parentBagGraph_adj_edgeOf deps hpt d).reachable
        | some p =>
            have hlt : p.1 < k := by
              have hlt' : p.1 < d.1 := by simpa using (parentOf_lt deps hpt d p hpc)
              simpa [hd] using hlt'
            have hstep : (parentBagGraph deps hpt).Reachable (some d) (some p) := by
              simpa [hpc] using (parentBagGraph_adj_edgeOf deps hpt d).reachable
            exact hstep.trans (ih p.1 hlt p rfl)
  exact hP c.1 c rfl

theorem parentBagGraph_connected {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) : (parentBagGraph deps hpt).Connected := by
  rw [SimpleGraph.connected_iff_exists_forall_reachable]
  refine ⟨none, ?_⟩
  intro x
  cases x with
  | none => exact SimpleGraph.Reachable.refl _
  | some c => exact (parentBagGraph_root_reachable deps hpt c).symm

noncomputable def parentEdgeOf {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) (c : Fin n) : (parentBagGraph deps hpt).edgeSet :=
  ⟨s(some c, parentOf deps hpt c), (parentBagGraph deps hpt).mem_edgeSet.mpr (parentBagGraph_adj_edgeOf deps hpt c)⟩

theorem parentEdgeOf_injective {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) : Function.Injective (parentEdgeOf deps hpt) := by
  intro c d hEq
  have h := congrArg Subtype.val hEq
  rcases Sym2.mk_eq_mk_iff.mp h with hpair | hswap
  · simpa using congrArg Prod.fst hpair
  · have hc : some c = parentOf deps hpt d := congrArg Prod.fst hswap
    have hd : parentOf deps hpt c = some d := congrArg Prod.snd hswap
    have hlt1 : d < c := parentOf_lt deps hpt c d hd
    have hlt2 : c < d := by simpa using parentOf_lt deps hpt d c hc.symm
    exfalso
    exact Nat.lt_asymm (by simpa using hlt1) (by simpa using hlt2)

theorem parentEdgeOf_surjective {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) : Function.Surjective (parentEdgeOf deps hpt) := by
  intro e
  rcases e with ⟨e, he⟩
  rcases Sym2.mk_surjective e with ⟨⟨x, y⟩, rfl⟩
  rw [(parentBagGraph deps hpt).mem_edgeSet] at he
  rcases he with ⟨c, rfl, hc⟩ | ⟨c, rfl, hc⟩
  · refine ⟨c, Subtype.ext ?_⟩
    simp [parentEdgeOf, hc]
  · refine ⟨c, Subtype.ext ?_⟩
    simpa [parentEdgeOf, hc] using (Sym2.mk_prod_swap_eq (p := (some c, x)))

theorem parentBagGraph_edgeSet_card {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) : Fintype.card (parentBagGraph deps hpt).edgeSet = n := by
  simpa using (Fintype.card_of_bijective (f := parentEdgeOf deps hpt)
    ⟨parentEdgeOf_injective deps hpt, parentEdgeOf_surjective deps hpt⟩).symm

theorem parentBagGraph_isTree {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) : (parentBagGraph deps hpt).IsTree := by
  rw [SimpleGraph.isTree_iff_connected_and_card]
  refine ⟨parentBagGraph_connected deps hpt, ?_⟩
  have hedge : Nat.card (parentBagGraph deps hpt).edgeSet = n := by
    rw [Nat.card_eq_fintype_card]
    exact parentBagGraph_edgeSet_card deps hpt
  have hvert : Nat.card (Option (Fin n)) = n + 1 := by
    rw [Nat.card_eq_fintype_card]
    simp
  omega

noncomputable def optionFinEquiv (n : ℕ) : Option (Fin n) ≃ Fin (n + 1) := by
  exact Fintype.equivOfCardEq (by simp)

def mapInduceSetIso {α β : Type*} (e : α ≃ β) (G : SimpleGraph α) (s : Set α) :
    (G.induce s) ≃g ((G.map e.toEmbedding).induce (e '' s)) where
  toEquiv :=
    { toFun := fun x => ⟨e x.1, by exact ⟨x.1, x.2, rfl⟩⟩
      invFun := fun y => ⟨e.symm y.1, by
        rcases y.2 with ⟨x, hx, hxy⟩
        have hxe : x = e.symm y.1 := by simpa using congrArg e.symm hxy
        simpa [hxe] using hx⟩
      left_inv := by intro x; apply Subtype.ext; simp
      right_inv := by intro y; apply Subtype.ext; simp }
  map_rel_iff' := by
    intro x y
    constructor
    · rintro ⟨u, v, h, hu, hv⟩
      have hu' : u = x.1 := e.injective (by simpa using hu)
      have hv' : v = y.1 := e.injective (by simpa using hv)
      simpa [hu', hv'] using h
    · intro h
      exact ⟨x.1, y.1, h, rfl, rfl⟩

def dependencyGraph {n : ℕ} (deps : Fin n → Finset (Fin n)) : SimpleGraph (Fin n) where
  Adj i j := i ≠ j ∧ (i ∈ deps j ∨ j ∈ deps i)
  symm := by intro i j h; exact ⟨h.1.symm, h.2.symm⟩
  loopless := by intro i h; exact h.1 rfl

def parentBagIndexGraph {n : ℕ} (deps : Fin n → Finset (Fin n)) (hpt : ParentTreeStructured deps) :
    SimpleGraph (Fin (n + 1)) :=
  (parentBagGraph deps hpt).map (optionFinEquiv n).toEmbedding

def oldBagContains {n : ℕ} (deps : Fin n → Finset (Fin n)) (v : Fin n) : Set (Option (Fin n))
  | none => False
  | some c => c = v ∨ v ∈ deps c

theorem oldBagContains_connected {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) (v : Fin n) :
    ((parentBagGraph deps hpt).induce (oldBagContains deps v)).Connected := by
  rw [SimpleGraph.connected_iff_exists_forall_reachable]
  refine ⟨⟨some v, by exact Or.inl rfl⟩, ?_⟩
  intro x
  cases x with
  | mk x hx =>
      cases x with
      | none => cases hx
      | some c =>
          change c = v ∨ v ∈ deps c at hx
          rcases hx with rfl | hvdeps
          · exact SimpleGraph.Reachable.refl _
          · have hparent : parentOf deps hpt c = some v := parentOf_eq_some_of_mem deps hpt c v hvdeps
            have hadj : ((parentBagGraph deps hpt).induce (oldBagContains deps v)).Adj
                ⟨some c, by exact Or.inr hvdeps⟩ ⟨some v, by exact Or.inl rfl⟩ := by
              simpa [hparent] using (parentBagGraph_adj_edgeOf deps hpt c)
            exact hadj.symm.reachable

theorem bagContains_set_eq_image_old {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) (v : Fin n) :
    {t : Fin (n + 1) |
      v ∈ (match (optionFinEquiv n).symm t with | none => (∅ : Finset (Fin n)) | some c => insert c (deps c))}
      = optionFinEquiv n '' oldBagContains deps v := by
  ext t
  constructor
  · intro ht
    refine ⟨(optionFinEquiv n).symm t, ?_, by simp⟩
    cases h : (optionFinEquiv n).symm t with
    | none => simpa [h] using ht
    | some c =>
        have hmem : v ∈ insert c (deps c) := by simpa [h] using ht
        change c = v ∨ v ∈ deps c
        simpa [eq_comm] using hmem
  · rintro ⟨x, hx, rfl⟩
    cases x with
    | none => cases hx
    | some c =>
        have hmem : v ∈ insert c (deps c) := by
          have hx' : c = v ∨ v ∈ deps c := by
            simpa [oldBagContains] using hx
          simpa [eq_comm] using hx'
        simpa using hmem

noncomputable def parentTreeDecomposition {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) :
    TreeDecomposition (n := n) (m := n + 1) (dependencyGraph deps) where
  tree := parentBagIndexGraph deps hpt
  isTree := (SimpleGraph.Iso.map (optionFinEquiv n) (parentBagGraph deps hpt)).isTree_iff.mp
    (parentBagGraph_isTree deps hpt)
  bags := fun t => match (optionFinEquiv n).symm t with | none => ∅ | some c => insert c (deps c)
  cover_vertex := by
    intro v
    exact ⟨optionFinEquiv n (some v), by simp⟩
  cover_edge := by
    intro u v huv
    rcases huv.2 with huv | hvu
    · refine ⟨optionFinEquiv n (some v), ?_, ?_⟩ <;> simp [huv]
    · refine ⟨optionFinEquiv n (some u), ?_, ?_⟩ <;> simp [hvu]
  connected_bags := by
    intro v
    have hconnOld : ((parentBagGraph deps hpt).induce (oldBagContains deps v)).Connected :=
      oldBagContains_connected deps hpt v
    have hIso := mapInduceSetIso (optionFinEquiv n) (parentBagGraph deps hpt) (oldBagContains deps v)
    have hconnNew : ((parentBagIndexGraph deps hpt).induce (optionFinEquiv n '' oldBagContains deps v)).Connected :=
      hIso.connected_iff.mp hconnOld
    rw [bagContains_set_eq_image_old deps hpt v]
    exact hconnNew

theorem parentTree_realTreewidth_one {n : ℕ} (deps : Fin n → Finset (Fin n))
    (hpt : ParentTreeStructured deps) :
    realTreewidth_le (dependencyGraph deps) 1 := by
  refine ⟨n + 1, parentTreeDecomposition deps hpt, ?_⟩
  intro t
  cases h : (optionFinEquiv n).symm t with
  | none => simp [parentTreeDecomposition, h]
  | some c =>
      have hle : (deps c).card ≤ 1 := hpt.2 c
      have hnot : c ∉ deps c := by
        intro hc
        exact Fin.lt_irrefl c (hpt.1 c c hc)
      rw [parentTreeDecomposition]
      simp [h, Finset.card_insert_of_notMem, hnot]
      omega

end Paper4dFrontier
