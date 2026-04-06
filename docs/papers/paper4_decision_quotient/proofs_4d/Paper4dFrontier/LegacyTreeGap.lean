import Paper4dFrontier.ParentTreewidth
import Paper4dFrontier.TreewidthClique

namespace Paper4dFrontier

open DecisionQuotient

def legacyTreeDeps : Fin 3 → Finset (Fin 3)
  | 0 => ∅
  | 1 => {0}
  | 2 => {0, 1}

theorem legacyTreeDeps_treeStructured : TreeStructured legacyTreeDeps := by
  intro c d hd
  fin_cases c <;> fin_cases d <;> simp [legacyTreeDeps] at hd ⊢

theorem legacyTreeDeps_not_parentTreeStructured : ¬ ParentTreeStructured legacyTreeDeps := by
  intro h
  have hcard : (legacyTreeDeps 2).card ≤ 1 := h.2 2
  simp [legacyTreeDeps] at hcard

theorem legacyTree_dependencyGraph_eq_top : dependencyGraph legacyTreeDeps = (⊤ : SimpleGraph (Fin 3)) := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [dependencyGraph, legacyTreeDeps]

theorem legacy_treeStructured_not_width_one :
    TreeStructured legacyTreeDeps ∧
    ¬ ParentTreeStructured legacyTreeDeps ∧
    ¬ realTreewidth_le (dependencyGraph legacyTreeDeps) 1 := by
  refine ⟨legacyTreeDeps_treeStructured, legacyTreeDeps_not_parentTreeStructured, ?_⟩
  simpa [legacyTree_dependencyGraph_eq_top] using completeGraph_not_realTreewidth_le 1

end Paper4dFrontier
