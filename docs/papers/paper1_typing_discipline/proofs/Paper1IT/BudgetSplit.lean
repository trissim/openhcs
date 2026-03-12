import Mathlib.Data.Finset.Max

namespace Ssot
namespace Paper1IT

section AbstractBudgetSplit

variable {ρ : Type*} [Fintype ρ] [DecidableEq ρ]

/-- Total budget when one component pays for representation complexity and the other pays for
auxiliary identity information. -/
def totalBudget (reprCost tagCost : ρ → Nat) (r : ρ) : Nat :=
  reprCost r + tagCost r

theorem exists_minimizing_split (reprCost tagCost : ρ → Nat) [Nonempty ρ] :
    ∃ rOpt : ρ, ∀ r : ρ, totalBudget reprCost tagCost rOpt ≤ totalBudget reprCost tagCost r := by
  classical
  let costs := (Finset.univ.image (totalBudget reprCost tagCost))
  have hne : costs.Nonempty := by
    classical
    let x : ρ := Classical.choice inferInstance
    exact ⟨totalBudget reprCost tagCost x, by simp [costs, x]⟩
  let m := costs.min' hne
  have hm : m ∈ costs := Finset.min'_mem costs hne
  rcases Finset.mem_image.mp hm with ⟨rOpt, -, hrOpt⟩
  refine ⟨rOpt, ?_⟩
  intro r
  rw [hrOpt]
  exact Finset.min'_le _ _ (by exact Finset.mem_image_of_mem _ (by simp))

/-- Abstract comparison theorem: if a split weakly improves both representation and tag cost, then
it weakly improves the total budget. -/
theorem totalBudget_mono
    (reprCost tagCost : ρ → Nat) {r r' : ρ}
    (hrepr : reprCost r ≤ reprCost r')
    (htag : tagCost r ≤ tagCost r') :
    totalBudget reprCost tagCost r ≤ totalBudget reprCost tagCost r' := by
  unfold totalBudget
  omega

/-- Any finite search space admits an optimal representation-vs-tag split. -/
theorem exists_optimal_budget_split
    (reprCost tagCost : ρ → Nat) [Nonempty ρ] :
    ∃ rOpt : ρ, totalBudget reprCost tagCost rOpt =
      (let costs := (Finset.univ.image (totalBudget reprCost tagCost));
        costs.min' (by
          let x : ρ := Classical.choice inferInstance
          exact ⟨totalBudget reprCost tagCost x, by simp [costs, x]⟩)) := by
  classical
  let costs := (Finset.univ.image (totalBudget reprCost tagCost))
  have hne : costs.Nonempty := by
    let x : ρ := Classical.choice inferInstance
    exact ⟨totalBudget reprCost tagCost x, by simp [costs, x]⟩
  let m := costs.min' hne
  have hm : m ∈ costs := Finset.min'_mem costs hne
  rcases Finset.mem_image.mp hm with ⟨rOpt, -, hrOpt⟩
  refine ⟨rOpt, ?_⟩
  simp [costs, m, hne, hrOpt]

end AbstractBudgetSplit

end Paper1IT
end Ssot
