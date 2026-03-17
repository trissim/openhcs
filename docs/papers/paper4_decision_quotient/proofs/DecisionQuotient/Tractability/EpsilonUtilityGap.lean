/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/EpsilonUtilityGap.lean
  
  Rigorous formalization of the ε-Bounded Suboptimal Margin Invariance.
-/
import DecisionQuotient.Basic
import DecisionQuotient.Sufficiency
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace DecisionQuotient
namespace Tractability

open Classical

/-- 
  The utility of the optimal action.
-/
noncomputable def OptUtility {A S : Type*} (dp : DecisionProblem A S) (s : S) : ℝ :=
  if h : (dp.Opt s).Nonempty then
    dp.utility h.some s
  else 0

/-- A decision problem has a strict optimum at state s. -/
def StrictOpt {A S : Type*} (dp : DecisionProblem A S) (a_star : A) (s : S) : Prop :=
  ∀ a, a ≠ a_star → dp.utility a s < dp.utility a_star s

/-- 
  The minimal numerical margin between the optimal action's utility and any strictly suboptimal action.
-/
noncomputable def StrictUtilityGap {A S : Type*} [Fintype A] (dp : DecisionProblem A S) (a_star : A) (s : S) : ℝ :=
  let subopts := (Finset.univ : Finset A).filter (fun a => a ≠ a_star)
  if h : subopts.Nonempty then
    let maxSubopt := subopts.sup' h (fun a => dp.utility a s)
    dp.utility a_star s - maxSubopt
  else 0

/-- 
  If the utility perturbation δ for all actions is strictly less than half the UtilityGap,
  then the strictly optimal action remains strictly optimal.
-/
theorem strict_margin_invariant {A S : Type*} [Fintype A]
    (dp : DecisionProblem A S) (s s' : S) (a_star : A)
    (δ : ℝ) (hδ : 0 ≤ δ)
    (hStrict : StrictOpt dp a_star s)
    (hPerturb : ∀ a, |dp.utility a s - dp.utility a s'| ≤ δ)
    (hBound : δ < (StrictUtilityGap dp a_star s) / 2) :
    StrictOpt dp a_star s' := by
  intro a ha_ne
  unfold StrictUtilityGap at hBound
  let subopts := (Finset.univ : Finset A).filter (fun x => x ≠ a_star)
  have ha_sub : a ∈ subopts := by
    rw [Finset.mem_filter]
    exact ⟨Finset.mem_univ a, ha_ne⟩
  have h_nonempty : subopts.Nonempty := ⟨a, ha_sub⟩
  
  have h_gap_def : (if h : subopts.Nonempty then dp.utility a_star s - subopts.sup' h (fun x => dp.utility x s) else 0) = 
                   dp.utility a_star s - subopts.sup' h_nonempty (fun x => dp.utility x s) := by
    exact dif_pos h_nonempty

  rw [h_gap_def] at hBound

  let maxSubopt := subopts.sup' h_nonempty (fun x => dp.utility x s)
  have h_max : dp.utility a s ≤ maxSubopt := Finset.le_sup' (fun x => dp.utility x s) ha_sub

  have h_gap_ineq : 2 * δ < dp.utility a_star s - maxSubopt := by linarith

  -- Re-derive |a - b| ≤ δ logic manually
  -- |x - y| ≤ δ means -δ ≤ x - y ≤ δ
  have h_a_pert := hPerturb a
  have h_a_diff_le : dp.utility a s - dp.utility a s' ≤ δ := (abs_le.mp h_a_pert).right
  have h_a_diff_ge : -δ ≤ dp.utility a s - dp.utility a s' := (abs_le.mp h_a_pert).left
  
  have h_ostar_pert := hPerturb a_star
  have h_ostar_diff_le : dp.utility a_star s - dp.utility a_star s' ≤ δ := (abs_le.mp h_ostar_pert).right
  have h_ostar_diff_ge : -δ ≤ dp.utility a_star s - dp.utility a_star s' := (abs_le.mp h_ostar_pert).left

  have h_a_s_prime : dp.utility a s' ≤ dp.utility a s + δ := by linarith
  have h_ostar_s_prime : dp.utility a_star s' ≥ dp.utility a_star s - δ := by linarith

  calc dp.utility a s'
    _ ≤ dp.utility a s + δ := h_a_s_prime
    _ ≤ maxSubopt + δ := by linarith
    _ < dp.utility a_star s - δ := by linarith
    _ ≤ dp.utility a_star s' := h_ostar_s_prime

/-- 
  If an action is strictly optimal, its Opt set contains exactly itself.
-/
theorem opt_eq_singleton_of_strict {A S : Type*}
    (dp : DecisionProblem A S) (a_star : A) (s : S)
    (hStrict : StrictOpt dp a_star s) :
    dp.Opt s = {a_star} := by
  ext a
  simp only [DecisionProblem.Opt, DecisionProblem.isOptimal, Set.mem_setOf_eq, Set.mem_singleton_iff]
  constructor
  · intro hOpt
    by_contra hNe
    have h_lt := hStrict a hNe
    have h_le := hOpt a_star
    linarith
  · rintro rfl
    intro a'
    by_cases h_eq : a' = a
    · rw [h_eq]
    · have h_lt := hStrict a' h_eq
      linarith

/-- 
  Main Theorem: Epsilon-Bounded Suboptimal Margin Invariance.
  If utility perturbation is strictly less than half the gap, the Opt set is invariant.
-/
theorem epsilon_margin_invariance {A S : Type*} [Fintype A]
    (dp : DecisionProblem A S) (s s' : S) (a_star : A)
    (δ : ℝ) (hδ : 0 ≤ δ)
    (hStrict : StrictOpt dp a_star s)
    (hPerturb : ∀ a, |dp.utility a s - dp.utility a s'| ≤ δ)
    (hBound : δ < (StrictUtilityGap dp a_star s) / 2) :
    dp.Opt s = dp.Opt s' := by
  have hStrict_s : dp.Opt s = {a_star} := opt_eq_singleton_of_strict dp a_star s hStrict
  have hStrict_s' : dp.Opt s' = {a_star} := opt_eq_singleton_of_strict dp a_star s' (strict_margin_invariant dp s s' a_star δ hδ hStrict hPerturb hBound)
  rw [hStrict_s, hStrict_s']

end Tractability
end DecisionQuotient
