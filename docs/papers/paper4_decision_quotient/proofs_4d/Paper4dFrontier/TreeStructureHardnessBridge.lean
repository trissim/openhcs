import DecisionQuotient.Tractability.Tightness

namespace Paper4dFrontier

open DecisionQuotient

theorem tree_structure_hardness_bridge {n : ℕ} (φ : Formula n) :
    (reductionProblem φ).isSufficient (∅ : Finset (Fin 1)) ↔ φ.isTautology :=
  cyclic_dependencies_coNP_hard φ

theorem bounded_actions_hardness_bridge {n : ℕ} (φ : Formula n) :
    (reductionProblem φ).isSufficient (∅ : Finset (Fin 1)) ↔ φ.isTautology :=
  two_actions_coNP_hard φ

theorem nonseparable_hardness_bridge {n : ℕ} (φ : Formula n) (hnontriv : ∃ a, φ.eval a = false) :
    ¬∃ (av : ReductionAction → ℝ) (sv : ReductionState n → ℝ),
      ∀ a s, reductionUtility φ a s = av a + sv s :=
  reduction_not_separable φ hnontriv

end Paper4dFrontier
