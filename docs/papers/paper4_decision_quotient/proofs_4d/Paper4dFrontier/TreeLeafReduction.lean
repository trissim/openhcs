import Paper4dFrontier.TreeDeletion

namespace Paper4dFrontier

open SimpleGraph
open Classical

def reducedFamily {α ι : Type*} (A : ι → Set (Option α)) : ι → Set α :=
  fun i => {a : α | some a ∈ A i}

theorem reducedFamily_connected {α : Type*} [Fintype α] {ι : Type*} [DecidableEq ι]
    {G : SimpleGraph (Option α)} [Fintype (G.neighborSet none)]
    (hT : G.IsTree) (hdeg : G.degree none = 1)
    (S : Finset ι) (A : ι → Set (Option α))
    (hconn : ∀ i ∈ S, (G.induce (A i)).Connected)
    (hpair : ∀ i ∈ S, ∀ j ∈ S, i ≠ j → (A i ∩ A j).Nonempty)
    {i0 : ι} (hi0 : i0 ∈ S) (hnone0 : none ∉ A i0) :
    ∀ i ∈ S, ((deleteNoneGraph G).induce (reducedFamily A i)).Connected := by
  intro i hi
  by_cases hnonei : none ∈ A i
  · have hInt : (A i ∩ A i0).Nonempty := by
      by_cases h : i = i0
      · subst h
        exact False.elim (hnone0 hnonei)
      · exact hpair i hi i0 hi0 h
    rcases hInt with ⟨z, hzInt⟩
    have hAi' : (G.induce (A i \ {none})).Connected :=
      connected_induce_erase_leaf_option hT hdeg (hconn i hi) hnonei hzInt.1
        (by intro hz; exact hnone0 (hz ▸ hzInt.2))
    exact (deleteNoneInduce_connected_iff G (A i)).2 hAi'
  · have hset : A i \ {none} = A i := by
      ext x
      simp [hnonei]
    have hAi' : (G.induce (A i \ {none})).Connected := by
      rw [hset]
      exact hconn i hi
    exact (deleteNoneInduce_connected_iff G (A i)).2 hAi'

theorem reducedFamily_pairwise {α : Type*} [Fintype α] {ι : Type*} [DecidableEq ι]
    {G : SimpleGraph (Option α)} [Fintype (G.neighborSet none)]
    (hdeg : G.degree none = 1)
    (S : Finset ι) (A : ι → Set (Option α))
    (hconn : ∀ i ∈ S, (G.induce (A i)).Connected)
    (hpair : ∀ i ∈ S, ∀ j ∈ S, i ≠ j → (A i ∩ A j).Nonempty)
    {i0 : ι} (hi0 : i0 ∈ S) (hnone0 : none ∉ A i0) :
    ∀ i ∈ S, ∀ j ∈ S, i ≠ j → (reducedFamily A i ∩ reducedFamily A j).Nonempty := by
  intro i hi j hj hij
  rcases hpair i hi j hj hij with ⟨z, hzi, hzj⟩
  cases hz : z with
  | none =>
      have hiNone : none ∈ A i := by simpa [hz] using hzi
      have hjNone : none ∈ A j := by simpa [hz] using hzj
      have hli : leafNeighborOption hdeg ∈ A i :=
        leafNeighbor_mem_of_connected_intersects_avoiding hdeg (hconn i hi) hiNone hnone0
          (hpair i hi i0 hi0 (by intro h; subst h; exact hnone0 hiNone))
      have hlj : leafNeighborOption hdeg ∈ A j :=
        leafNeighbor_mem_of_connected_intersects_avoiding hdeg (hconn j hj) hjNone hnone0
          (hpair j hj i0 hi0 (by intro h; subst h; exact hnone0 hjNone))
      have hleaf : leafNeighborOption hdeg ≠ none := leafNeighborOption_ne_none hdeg
      cases hleafEq : leafNeighborOption hdeg with
      | none => exact False.elim (hleaf hleafEq)
      | some a =>
          refine ⟨a, ?_, ?_⟩
          · simpa [reducedFamily, hleafEq] using hli
          · simpa [reducedFamily, hleafEq] using hlj
  | some a =>
      refine ⟨a, ?_, ?_⟩
      · simpa [reducedFamily, hz] using hzi
      · simpa [reducedFamily, hz] using hzj

end Paper4dFrontier
