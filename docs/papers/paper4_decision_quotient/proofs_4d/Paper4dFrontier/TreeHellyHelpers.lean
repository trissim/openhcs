import Paper4dFrontier.RealTreewidth
import Mathlib.Combinatorics.SimpleGraph.Paths
import Mathlib.Combinatorics.SimpleGraph.Walks.Maps

namespace Paper4dFrontier

open SimpleGraph

theorem exists_path_support_subset_of_induce_connected {V : Type*} {G : SimpleGraph V}
    {s : Set V} (hs : (G.induce s).Connected) {u v : V} (hu : u ∈ s) (hv : v ∈ s) :
    ∃ p : G.Walk u v, p.IsPath ∧ ∀ x ∈ p.support, x ∈ s := by
  obtain ⟨p, hp⟩ := hs.exists_isPath ⟨u, hu⟩ ⟨v, hv⟩
  refine ⟨p.map (Embedding.induce s).toHom, ?_, ?_⟩
  · simpa using
      (SimpleGraph.Walk.map_isPath_of_injective (f := (Embedding.induce s).toHom)
        (fun x y hxy => Subtype.ext hxy) hp)
  · intro x hx
    rw [SimpleGraph.Walk.support_map] at hx
    rcases List.mem_map.mp hx with ⟨y, hy, rfl⟩
    exact y.2

theorem path_support_subset_of_induce_connected {V : Type*} {G : SimpleGraph V}
    (hT : G.IsTree) {s : Set V} (hs : (G.induce s).Connected) {u v : V}
    (hu : u ∈ s) (hv : v ∈ s) (p : G.Walk u v) (hp : p.IsPath) :
    ∀ x ∈ p.support, x ∈ s := by
  obtain ⟨q, hq, hqsub⟩ := exists_path_support_subset_of_induce_connected hs hu hv
  have hEq : p = q := (hT.existsUnique_path u v).unique hp hq
  intro x hx
  rw [hEq] at hx
  exact hqsub x hx

theorem connected_induce_inter_of_isTree {V : Type*} {G : SimpleGraph V}
    (hT : G.IsTree) {s t : Set V}
    (hs : (G.induce s).Connected) (ht : (G.induce t).Connected)
    (hst : (s ∩ t).Nonempty) :
    (G.induce (s ∩ t)).Connected := by
  rw [SimpleGraph.connected_iff_exists_forall_reachable]
  rcases hst with ⟨r, hr⟩
  refine ⟨⟨r, hr⟩, ?_⟩
  intro w
  obtain ⟨p, hp⟩ := hT.isConnected.exists_isPath r w.1
  have hsub_s := path_support_subset_of_induce_connected hT hs hr.1 w.2.1 p hp
  have hsub_t := path_support_subset_of_induce_connected hT ht hr.2 w.2.2 p hp
  refine ⟨p.induce (s ∩ t) ?_⟩
  intro x hx
  exact ⟨hsub_s x hx, hsub_t x hx⟩

end Paper4dFrontier
