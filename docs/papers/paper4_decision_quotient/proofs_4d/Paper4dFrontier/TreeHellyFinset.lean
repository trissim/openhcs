import Paper4dFrontier.TreeLeafReduction
import Paper4dFrontier.TreeLeafReduction

namespace Paper4dFrontier

open SimpleGraph
open Classical

def TreeHellyFinsetProp (α : Type*) [Fintype α] : Prop :=
  ∀ (F : Finset (Set α)), F.Nonempty →
    ∀ (G : SimpleGraph α), G.IsTree →
    (∀ s ∈ F, (G.induce s).Connected) →
    (∀ s ∈ F, ∀ t ∈ F, s ≠ t → (s ∩ t).Nonempty) →
    ∃ x : α, ∀ s ∈ F, x ∈ s

theorem treeHelly_none_leaf_finset {α : Type*} [Fintype α]
    (IH : TreeHellyFinsetProp α)
    (F : Finset (Set (Option α))) (hF : F.Nonempty)
    (G : SimpleGraph (Option α)) (hT : G.IsTree) [Fintype (G.neighborSet none)]
    (hdeg : G.degree none = 1)
    (hconn : ∀ s ∈ F, (G.induce s).Connected)
    (hpair : ∀ s ∈ F, ∀ t ∈ F, s ≠ t → (s ∩ t).Nonempty) :
    ∃ x : Option α, ∀ s ∈ F, x ∈ s := by
  by_cases hall : ∀ s ∈ F, none ∈ s
  · exact ⟨none, hall⟩
  · obtain ⟨s0, hs0, hnone0⟩ := by simpa [not_forall] using hall
    let F' : Finset (Set α) := F.image (fun s => {a : α | some a ∈ s})
    have htree0 : (deleteNoneGraph G).IsTree := deleteNoneGraph_isTree_of_leaf hT hdeg
    have hconn' : ∀ t ∈ F', ((deleteNoneGraph G).induce t).Connected := by
      intro t ht
      rcases Finset.mem_image.mp ht with ⟨s, hs, rfl⟩
      simpa [reducedFamily] using reducedFamily_connected hT hdeg F (fun s => s) hconn hpair hs0 hnone0 s hs
    have hpair' : ∀ s ∈ F', ∀ t ∈ F', s ≠ t → (s ∩ t).Nonempty := by
      intro s hs t ht hst
      rcases Finset.mem_image.mp hs with ⟨s0', hs0', rfl⟩
      rcases Finset.mem_image.mp ht with ⟨t0', ht0', rfl⟩
      by_cases hEq : s0' = t0'
      · subst hEq
        exfalso
        exact hst rfl
      · simpa [reducedFamily] using reducedFamily_pairwise hdeg F (fun s => s) hconn hpair hs0 hnone0 s0' hs0' t0' ht0' hEq
    obtain ⟨x, hx⟩ := IH F' (by
      rcases hF with ⟨s, hs⟩
      exact ⟨{a : α | some a ∈ s0}, Finset.mem_image.mpr ⟨s0, hs0, rfl⟩⟩)
      (deleteNoneGraph G) htree0 hconn' hpair'
    exact ⟨some x, by
      intro s hs
      have hs' : {a : α | some a ∈ s} ∈ F' := Finset.mem_image.mpr ⟨s, hs, rfl⟩
      exact hx _ hs'⟩

theorem treeHelly_option_finset {α : Type*} [Fintype α]
    (IH : TreeHellyFinsetProp α) : TreeHellyFinsetProp (Option α) := by
  intro F hF G hT hconn hpair
  by_cases hsub : Subsingleton (Option α)
  · have hnone : ∀ s ∈ F, none ∈ s := by
      intro s hs
      rcases (hconn s hs).nonempty with ⟨x⟩
      have hx : x.1 = none := Subsingleton.elim _ _
      simpa [hx] using x.property
    exact ⟨none, hnone⟩
  · letI : Nontrivial (Option α) := not_subsingleton_iff_nontrivial.mp hsub
    obtain ⟨v, hdegv0⟩ := hT.exists_vert_degree_one_of_nontrivial
    have hdegv : G.degree v = 1 := hdegv0
    let e : Option α ≃ Option α := Equiv.swap none v
    let G' : SimpleGraph (Option α) := G.comap e.toEmbedding
    have hT' : G'.IsTree := (SimpleGraph.Iso.comap e G).isTree_iff.mpr hT
    letI : Fintype (G'.neighborSet none) := Fintype.ofFinite _
    have hdeg' : G'.degree none = 1 := by
      rw [SimpleGraph.degree_eq_one_iff_existsUnique_adj]
      rcases (SimpleGraph.degree_eq_one_iff_existsUnique_adj).1 hdegv with ⟨u, hu, huuniq⟩
      refine ⟨e.symm u, ?_, ?_⟩
      · simpa [G', e] using hu
      · intro w hw
        apply e.injective
        simpa using huuniq (e w) (by simpa [G', e] using hw)
    let F' : Finset (Set (Option α)) := F.image (fun s => e ⁻¹' s)
    have hF' : F'.Nonempty := by
      rcases hF with ⟨s, hs⟩
      exact ⟨e ⁻¹' s, Finset.mem_image.mpr ⟨s, hs, rfl⟩⟩
    have hconn' : ∀ s ∈ F', (G'.induce s).Connected := by
      intro s hs
      rcases Finset.mem_image.mp hs with ⟨t, ht, rfl⟩
      exact (equivInduceSetIso e G t).connected_iff.mpr (hconn t ht)
    have hpair' : ∀ s ∈ F', ∀ t ∈ F', s ≠ t → (s ∩ t).Nonempty := by
      intro s hs t ht hst
      rcases Finset.mem_image.mp hs with ⟨s0, hs0, rfl⟩
      rcases Finset.mem_image.mp ht with ⟨t0, ht0, rfl⟩
      have hst0 : s0 ≠ t0 := by
        intro h
        apply hst
        subst h
        rfl
      rcases hpair s0 hs0 t0 ht0 hst0 with ⟨x, hxs, hxt⟩
      exact ⟨e.symm x, by simpa using hxs, by simpa using hxt⟩
    obtain ⟨x', hx'⟩ := treeHelly_none_leaf_finset IH F' hF' G' hT' hdeg' hconn' hpair'
    refine ⟨e x', ?_⟩
    intro s hs
    have hs' : e ⁻¹' s ∈ F' := Finset.mem_image.mpr ⟨s, hs, rfl⟩
    exact hx' _ hs'

end Paper4dFrontier
