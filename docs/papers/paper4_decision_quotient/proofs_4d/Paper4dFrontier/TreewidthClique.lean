import Paper4dFrontier.TreeHellyTheorem
import Mathlib.Combinatorics.SimpleGraph.Clique

namespace Paper4dFrontier

open SimpleGraph
open Classical

def bagVertexSet {n m : ℕ} {G : SimpleGraph (Fin n)}
    (td : TreeDecomposition (n := n) (m := m) G) (v : Fin n) : Set (Fin m) :=
  {t | v ∈ td.bags t}

theorem clique_has_common_bag {n m : ℕ} {G : SimpleGraph (Fin n)}
    (td : TreeDecomposition (n := n) (m := m) G)
    (K : Finset (Fin n)) (hK : K.Nonempty) (hClique : G.IsClique (↑K : Set (Fin n))) :
    ∃ t : Fin m, ∀ v ∈ K, v ∈ td.bags t := by
  let F : Finset (Set (Fin m)) := K.image (bagVertexSet td)
  have hF : F.Nonempty := by
    rcases hK with ⟨v, hv⟩
    exact ⟨bagVertexSet td v, Finset.mem_image.mpr ⟨v, hv, rfl⟩⟩
  have hconn : ∀ s ∈ F, (td.tree.induce s).Connected := by
    intro s hs
    rcases Finset.mem_image.mp hs with ⟨v, hv, rfl⟩
    simpa [bagVertexSet] using td.connected_bags v
  have hpair : ∀ s ∈ F, ∀ t ∈ F, s ≠ t → (s ∩ t).Nonempty := by
    intro s hs t ht hst
    rcases Finset.mem_image.mp hs with ⟨u, hu, rfl⟩
    rcases Finset.mem_image.mp ht with ⟨v, hv, rfl⟩
    have huv : u ≠ v := by
      intro h
      apply hst
      subst h
      rfl
    obtain ⟨b, hub, hvb⟩ := td.cover_edge (hClique hu hv huv)
    exact ⟨b, hub, hvb⟩
  obtain ⟨t, ht⟩ := treeHellyFinset_all (Fin m) F hF td.tree td.isTree hconn hpair
  refine ⟨t, ?_⟩
  intro v hv
  have hvF : bagVertexSet td v ∈ F := Finset.mem_image.mpr ⟨v, hv, rfl⟩
  exact ht _ hvF

theorem completeGraph_not_realTreewidth_le_of_large (n w : ℕ) (hlarge : w + 2 ≤ n) :
    ¬ realTreewidth_le (⊤ : SimpleGraph (Fin n)) w := by
  intro htw
  rcases htw with ⟨m, td, hbags⟩
  have hClique : (⊤ : SimpleGraph (Fin n)).IsClique (Set.univ : Set (Fin n)) := by
    intro u _ v _ hne
    simpa using hne
  let K : Finset (Fin n) := Finset.univ.map ⟨Fin.castLE hlarge, Fin.castLE_injective hlarge⟩
  have hKnonempty : K.Nonempty := by
    simpa [K] using (Finset.univ_nonempty.map ⟨Fin.castLE hlarge, Fin.castLE_injective hlarge⟩)
  have hKclique : (⊤ : SimpleGraph (Fin n)).IsClique (↑K : Set (Fin n)) := by
    intro u hu v hv hne
    simpa using hne
  obtain ⟨t, ht⟩ := clique_has_common_bag td K hKnonempty hKclique
  have hcard : K.card ≤ (td.bags t).card := by
    exact Finset.card_le_card (by intro x hx; exact ht x (by simpa using hx))
  have hKcard : K.card = w + 2 := by
    simp [K]
  have hcard' : w + 2 ≤ (td.bags t).card := by simpa [hKcard] using hcard
  have hbound : (td.bags t).card ≤ w + 1 := hbags t
  have : w + 2 ≤ w + 1 := le_trans hcard' hbound
  omega

theorem completeGraph_not_realTreewidth_le (w : ℕ) :
    ¬ realTreewidth_le (⊤ : SimpleGraph (Fin (w + 2))) w := by
  exact completeGraph_not_realTreewidth_le_of_large (n := w + 2) (w := w) (by omega)

end Paper4dFrontier
