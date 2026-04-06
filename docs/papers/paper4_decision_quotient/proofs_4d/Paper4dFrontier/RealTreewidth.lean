import Mathlib.Combinatorics.SimpleGraph.Acyclic
import Mathlib.Combinatorics.SimpleGraph.Connectivity.Subgraph
import Mathlib.Combinatorics.SimpleGraph.Clique
import Mathlib.Tactic

namespace Paper4dFrontier

open Classical

/-- A genuine finite tree decomposition of a graph on `Fin n`. The bag-index
graph is a finite tree, every graph edge is covered by some bag, and the bags
containing any fixed graph vertex form a connected induced subgraph. -/
structure TreeDecomposition {n m : ℕ} (G : SimpleGraph (Fin n)) where
  tree : SimpleGraph (Fin m)
  isTree : tree.IsTree
  bags : Fin m → Finset (Fin n)
  cover_vertex : ∀ v : Fin n, ∃ t : Fin m, v ∈ bags t
  cover_edge : ∀ {u v : Fin n}, G.Adj u v → ∃ t : Fin m, u ∈ bags t ∧ v ∈ bags t
  connected_bags : ∀ v : Fin n, (tree.induce {t : Fin m | v ∈ bags t}).Connected

/-- The graph has treewidth at most `w` if it admits a tree decomposition with
all bags of size at most `w+1`. -/
def realTreewidth_le {n : ℕ} (G : SimpleGraph (Fin n)) (w : ℕ) : Prop :=
  ∃ m : ℕ, ∃ td : TreeDecomposition (n := n) (m := m) G,
    ∀ t : Fin m, (td.bags t).card ≤ w + 1

theorem oneBag_tree_isTree : (⊥ : SimpleGraph (Fin 1)).IsTree := by
  exact SimpleGraph.IsTree.of_subsingleton

/-- Any finite graph has the trivial one-bag decomposition of width `n-1`. -/
def oneBagDecomposition {n : ℕ} (G : SimpleGraph (Fin n)) :
    TreeDecomposition (n := n) (m := 1) G where
  tree := ⊥
  isTree := oneBag_tree_isTree
  bags := fun _ => Finset.univ
  cover_vertex := by
    intro v
    refine ⟨0, ?_⟩
    simp
  cover_edge := by
    intro u v huv
    refine ⟨0, ?_, ?_⟩ <;> simp
  connected_bags := by
    intro v
    have hset : ({t : Fin 1 | v ∈ Finset.univ} : Set (Fin 1)) = Set.univ := by
      ext t
      simp
    rw [hset]
    have htree : (((⊥ : SimpleGraph (Fin 1)).induce (Set.univ : Set (Fin 1)))).IsTree := by
      exact SimpleGraph.IsTree.of_subsingleton
    exact htree.isConnected

theorem realTreewidth_le_card_pred {n : ℕ} (G : SimpleGraph (Fin n)) :
    realTreewidth_le G (n - 1) := by
  refine ⟨1, oneBagDecomposition G, ?_⟩
  intro t
  have : n ≤ n - 1 + 1 := by omega
  simpa [oneBagDecomposition, Finset.card_fin] using this

end Paper4dFrontier
