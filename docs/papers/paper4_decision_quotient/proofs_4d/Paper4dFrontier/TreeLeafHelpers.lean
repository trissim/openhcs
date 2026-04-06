import Paper4dFrontier.TreeHellyHelpers
import Mathlib.Combinatorics.SimpleGraph.Finite

namespace Paper4dFrontier

open SimpleGraph
open Classical

noncomputable def leafNeighborOption {α : Type*} [Fintype α]
    {G : SimpleGraph (Option α)} [Fintype (G.neighborSet none)]
    (hdeg : G.degree none = 1) : Option α :=
  Classical.choose ((SimpleGraph.degree_eq_one_iff_existsUnique_adj).1 hdeg)

theorem leafNeighborOption_adj {α : Type*} [Fintype α]
    {G : SimpleGraph (Option α)} [Fintype (G.neighborSet none)] (hdeg : G.degree none = 1) :
    G.Adj none (leafNeighborOption hdeg) :=
  (Classical.choose_spec ((SimpleGraph.degree_eq_one_iff_existsUnique_adj).1 hdeg)).1

theorem leafNeighborOption_unique {α : Type*} [Fintype α]
    {G : SimpleGraph (Option α)} [Fintype (G.neighborSet none)] (hdeg : G.degree none = 1) {w : Option α}
    (hw : G.Adj none w) : w = leafNeighborOption hdeg :=
  (Classical.choose_spec ((SimpleGraph.degree_eq_one_iff_existsUnique_adj).1 hdeg)).2 w hw

theorem leafNeighborOption_ne_none {α : Type*} [Fintype α]
    {G : SimpleGraph (Option α)} [Fintype (G.neighborSet none)] (hdeg : G.degree none = 1) :
    leafNeighborOption hdeg ≠ none := by
  intro h
  exact G.loopless none (h ▸ leafNeighborOption_adj hdeg)

def eraseNoneSubtypeEquiv {α : Type*} {s : Set (Option α)} (hnone : none ∈ s) :
    ({⟨none, hnone⟩}ᶜ : Set s) ≃ {x : Option α // x ∈ s \ {none}} where
  toFun x := ⟨x.1.1, by
    constructor
    · exact x.1.2
    · intro hx
      exact x.2 (by apply Subtype.ext; exact hx)
    ⟩
  invFun x := ⟨⟨x.1, x.2.1⟩, by
    intro hx
    exact x.2.2 (congrArg Subtype.val hx)
    ⟩
  left_inv := by
    intro x
    cases x with
    | mk x hx =>
        apply Subtype.ext
        apply Subtype.ext
        rfl
  right_inv := by
    intro x
    apply Subtype.ext
    rfl

def eraseNoneInducedIso {α : Type*} [Fintype α]
    {G : SimpleGraph (Option α)} {s : Set (Option α)} (hnone : none ∈ s) :
    ((G.induce s).induce ({⟨none, hnone⟩}ᶜ : Set s)) ≃g (G.induce (s \ {none})) where
  toEquiv := eraseNoneSubtypeEquiv hnone
  map_rel_iff' := by
    intro x y
    rfl

theorem connected_induce_erase_leaf_option {α : Type*} [Fintype α]
    {G : SimpleGraph (Option α)} [Fintype (G.neighborSet none)]
    (hT : G.IsTree) (hdeg : G.degree none = 1)
    {s : Set (Option α)} (hs : (G.induce s).Connected) (hnone : none ∈ s)
    {z : Option α} (hzs : z ∈ s) (hz : z ≠ none) :
    (G.induce (s \ {none})).Connected := by
  let H : SimpleGraph s := G.induce s
  have hpath : ∃ p : H.Walk ⟨none, hnone⟩ ⟨z, hzs⟩, p.IsPath := hs.exists_isPath _ _
  rcases hpath with ⟨p, hp⟩
  have hneq : (⟨none, hnone⟩ : s) ≠ ⟨z, hzs⟩ := by
    intro h
    exact hz (congrArg Subtype.val h).symm
  cases p with
  | nil =>
      cases hneq rfl
  | cons hxy p' =>
    rename_i y
    have hGxy : G.Adj none y.1 := hxy
    have hu : y.1 ∈ s := y.2
    have hy_unique : ∀ w : Option α, G.Adj none w → w = y.1 := by
      intro w hw
      rw [leafNeighborOption_unique hdeg hw, ← leafNeighborOption_unique hdeg hGxy]
    have hdegH : H.degree ⟨none, hnone⟩ = 1 := by
      rw [SimpleGraph.degree_eq_one_iff_existsUnique_adj]
      refine ⟨⟨y.1, hu⟩, hxy, ?_⟩
      intro w hw
      apply Subtype.ext
      exact hy_unique w.1 hw
    have hconnH : H.Connected := hs
    have hconn' := hconnH.induce_compl_singleton_of_degree_eq_one hdegH
    exact (eraseNoneInducedIso hnone).connected_iff.mp hconn'

theorem leafNeighbor_mem_of_connected_intersects_avoiding {α : Type*} [Fintype α]
    {G : SimpleGraph (Option α)} [Fintype (G.neighborSet none)] (hdeg : G.degree none = 1)
    {s t : Set (Option α)} (hs : (G.induce s).Connected) (hnone : none ∈ s)
    (htnone : none ∉ t) (hst : (s ∩ t).Nonempty) :
    leafNeighborOption hdeg ∈ s := by
  rcases hst with ⟨z, hz⟩
  have hznone : z ≠ none := by
    intro hz'
    exact htnone (hz'.symm ▸ hz.2)
  let H : SimpleGraph s := G.induce s
  obtain ⟨p, hp⟩ := hs.exists_isPath ⟨none, hnone⟩ ⟨z, hz.1⟩
  have hneq : (⟨none, hnone⟩ : s) ≠ ⟨z, hz.1⟩ := by
    intro h
    exact hznone (congrArg Subtype.val h).symm
  cases p with
  | nil => cases hneq rfl
  | cons hxy p' =>
      rename_i y
      have hGxy : G.Adj none y.1 := hxy
      have : y.1 = leafNeighborOption hdeg := leafNeighborOption_unique hdeg hGxy
      simpa [this] using y.2

end Paper4dFrontier
