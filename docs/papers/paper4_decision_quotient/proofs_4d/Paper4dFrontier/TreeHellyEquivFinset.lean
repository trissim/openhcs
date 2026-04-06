import Paper4dFrontier.TreeHellyFinset

namespace Paper4dFrontier

open SimpleGraph
open Classical

theorem treeHellyFinset_of_equiv {α β : Type*} [Fintype α] [Fintype β]
    (e : α ≃ β) (hα : TreeHellyFinsetProp α) : TreeHellyFinsetProp β := by
  intro F hF G hT hconn hpair
  let Fα : Finset (Set α) := F.image (fun s : Set β => e ⁻¹' s)
  have hFα : Fα.Nonempty := by
    rcases hF with ⟨s, hs⟩
    exact ⟨e ⁻¹' s, Finset.mem_image.mpr ⟨s, hs, rfl⟩⟩
  let Gα : SimpleGraph α := G.comap e.toEmbedding
  have hTα : Gα.IsTree := (SimpleGraph.Iso.comap e G).isTree_iff.mpr hT
  have hconnα : ∀ s ∈ Fα, (Gα.induce s).Connected := by
    intro s hs
    rcases Finset.mem_image.mp hs with ⟨t, ht, rfl⟩
    exact (equivInduceSetIso e G t).connected_iff.mpr (hconn t ht)
  have hpairα : ∀ s ∈ Fα, ∀ t ∈ Fα, s ≠ t → (s ∩ t).Nonempty := by
    intro s hs t ht hst
    rcases Finset.mem_image.mp hs with ⟨s', hs', rfl⟩
    rcases Finset.mem_image.mp ht with ⟨t', ht', rfl⟩
    have hst' : s' ≠ t' := by
      intro h
      apply hst
      subst h
      rfl
    rcases hpair s' hs' t' ht' hst' with ⟨x, hxs, hxt⟩
    exact ⟨e.symm x, by simpa using hxs, by simpa using hxt⟩
  obtain ⟨x, hx⟩ := hα Fα hFα Gα hTα hconnα hpairα
  refine ⟨e x, ?_⟩
  intro s hs
  have hsα : e ⁻¹' s ∈ Fα := Finset.mem_image.mpr ⟨s, hs, rfl⟩
  exact hx _ hsα

end Paper4dFrontier
