import Paper4dFrontier.TreeLeafHelpers
import Mathlib.Combinatorics.SimpleGraph.Maps

namespace Paper4dFrontier

open SimpleGraph

def someSubtypeEquiv (α : Type*) : α ≃ {x : Option α // x ≠ none} where
  toFun a := ⟨some a, Option.some_ne_none a⟩
  invFun x := by
    cases x with
    | mk x hx =>
        cases x with
        | none => exact False.elim (hx rfl)
        | some a => exact a
  left_inv := by intro a; rfl
  right_inv := by
    intro x
    cases x with
    | mk x hx =>
        cases x with
        | none => cases (hx rfl)
        | some a => rfl

def deleteNoneGraph {α : Type*} (G : SimpleGraph (Option α)) : SimpleGraph α :=
  (G.induce ({none}ᶜ : Set (Option α))).comap (someSubtypeEquiv α).toEmbedding

def deleteNoneInducedIso {α : Type*} (G : SimpleGraph (Option α)) :
    deleteNoneGraph G ≃g (G.induce ({none}ᶜ : Set (Option α))) :=
  SimpleGraph.Iso.comap (someSubtypeEquiv α) (G.induce ({none}ᶜ : Set (Option α)))

def deleteNoneSetEquiv {α : Type*} (s : Set (Option α)) :
    {a : α // some a ∈ s} ≃ {x : Option α // x ∈ s \ {none}} where
  toFun x := ⟨some x.1, ⟨x.2, Option.some_ne_none _⟩⟩
  invFun x := by
    cases x with
    | mk x hx =>
        cases x with
        | none => exact False.elim (hx.2 rfl)
        | some a => exact ⟨a, by simpa using hx.1⟩
  left_inv := by intro x; cases x; rfl
  right_inv := by
    intro x
    cases x with
    | mk x hx =>
        cases x with
        | none => cases (hx.2 rfl)
        | some a => rfl

def deleteNoneInduceSetIso {α : Type*} (G : SimpleGraph (Option α)) (s : Set (Option α)) :
    ((deleteNoneGraph G).induce {a : α | some a ∈ s}) ≃g (G.induce (s \ {none})) := by
  refine (SimpleGraph.Iso.comap (deleteNoneSetEquiv s) (G.induce (s \ {none}))).trans ?_
  refine
    { toEquiv := Equiv.refl _
      map_rel_iff' := by
        intro a b
        rfl }

theorem deleteNoneGraph_connected_of_leaf {α : Type*} [Fintype α]
    {G : SimpleGraph (Option α)} [Fintype (G.neighborSet none)]
    (hT : G.IsTree) (hdeg : G.degree none = 1) :
    (deleteNoneGraph G).Connected := by
  have hconn : (G.induce ({none}ᶜ : Set (Option α))).Connected :=
    hT.isConnected.induce_compl_singleton_of_degree_eq_one hdeg
  exact (deleteNoneInducedIso G).connected_iff.mpr hconn

theorem deleteNoneGraph_isTree_of_leaf {α : Type*} [Fintype α]
    {G : SimpleGraph (Option α)} [Fintype (G.neighborSet none)]
    (hT : G.IsTree) (hdeg : G.degree none = 1) :
    (deleteNoneGraph G).IsTree := by
  refine ⟨deleteNoneGraph_connected_of_leaf hT hdeg, ?_⟩
  have hacyc : (G.induce ({none}ᶜ : Set (Option α))).IsAcyclic := hT.IsAcyclic.induce _
  exact (deleteNoneInducedIso G).isAcyclic_iff.mpr hacyc

theorem deleteNoneInduce_connected_iff {α : Type*} (G : SimpleGraph (Option α)) (s : Set (Option α)) :
    ((deleteNoneGraph G).induce {a : α | some a ∈ s}).Connected ↔ (G.induce (s \ {none})).Connected := by
  exact (deleteNoneInduceSetIso G s).connected_iff

def equivInduceSetIso {α β : Type*} (e : α ≃ β) (G : SimpleGraph β) (s : Set β) :
    ((G.comap e.toEmbedding).induce (e ⁻¹' s)) ≃g (G.induce s) where
  toEquiv :=
    { toFun := fun x => ⟨e x.1, x.2⟩
      invFun := fun y => ⟨e.symm y.1, by simpa using y.2⟩
      left_inv := by intro x; apply Subtype.ext; simp
      right_inv := by intro y; apply Subtype.ext; simp }
  map_rel_iff' := by intro x y; rfl

end Paper4dFrontier
