import Paper4dFrontier.ParentTreewidth
import Paper4dFrontier.TreewidthClique

namespace Paper4dFrontier

open DecisionQuotient

def weakTreeDeps : Fin 3 → Finset (Fin 3)
  | 0 => ∅
  | 1 => {0}
  | 2 => {0, 1}

theorem weakTreeDeps_treeStructured : TreeStructured weakTreeDeps := by
  intro c d hd
  fin_cases c <;> fin_cases d <;> simp [weakTreeDeps] at hd ⊢

theorem weakTreeDeps_not_parentTreeStructured : ¬ ParentTreeStructured weakTreeDeps := by
  intro h
  have hcard : (weakTreeDeps 2).card ≤ 1 := h.2 2
  simp [weakTreeDeps] at hcard

theorem weakTree_dependencyGraph_eq_top : dependencyGraph weakTreeDeps = (⊤ : SimpleGraph (Fin 3)) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [dependencyGraph, weakTreeDeps]

theorem weak_tree_structured_not_width_one :
    TreeStructured weakTreeDeps ∧
    ¬ ParentTreeStructured weakTreeDeps ∧
    ¬ realTreewidth_le (dependencyGraph weakTreeDeps) 1 := by
  refine ⟨weakTreeDeps_treeStructured, weakTreeDeps_not_parentTreeStructured, ?_⟩
  simpa [weakTree_dependencyGraph_eq_top] using completeGraph_not_realTreewidth_le 1

end Paper4dFrontier
