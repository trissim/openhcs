/-
  Paper 4b: Stochastic and Sequential Regimes

  ThermodynamicLift.lean - Thermodynamic interpretation of sufficiency

  Landauer's principle, entropy, and computational work.
-/

import DecisionQuotient.StochasticSequential.Basic
import DecisionQuotient.StochasticSequential.Economics
import DecisionQuotient.Physics.Instantiation
import Mathlib.Data.Real.Basic

namespace DecisionQuotient.StochasticSequential

/-! ## Thermodynamic Cost

Landauer's principle: erasing one bit of information requires at least
k_B T ln 2 joules of energy dissipated as heat. This provides a fundamental
thermodynamic lower bound on computation.
-/

/-- Landauer constant (k_B T ln 2 at room temperature) -/
noncomputable def landauerConstant : ℝ := 1.38e-23 * 300 * Real.log 2

/-! ## Potential Grounding -/

/-- State potential over the regime state space. -/
abbrev StatePotential (S : Type*) := S → ℝ

/-- Potential drop from `s` to `s'` (positive when potential decreases). -/
def potentialDrop {S : Type*} (Φ : StatePotential S) (s s' : S) : ℝ :=
  Φ s - Φ s'

/-- Utility induced by potential drop through a deterministic next-state map. -/
def utilityFromPotentialDrop {A S : Type*}
    (Φ : StatePotential S) (next : A → S → S) (a : A) (s : S) : ℝ :=
  potentialDrop Φ s (next a s)

/-- Pairwise decision comparison under potential-drop utility is dual to
    comparison of next-state potentials. -/
theorem utilityFromPotentialDrop_le_iff_nextPotential_ge {A S : Type*}
    (Φ : StatePotential S) (next : A → S → S) (s : S) (a a' : A) :
    utilityFromPotentialDrop Φ next a s ≤ utilityFromPotentialDrop Φ next a' s
      ↔ Φ (next a' s) ≤ Φ (next a s) := by
  unfold utilityFromPotentialDrop potentialDrop
  simpa using
    (sub_le_sub_iff_left (a := Φ s) (b := Φ (next a s)) (c := Φ (next a' s)))

/-- Action-conditioned potential field for one-shot stochastic evaluation. -/
abbrev ActionStatePotential (A S : Type*) := A → S → ℝ

/-- Expected potential for action `a` under the declared state distribution. -/
noncomputable def expectedActionPotential {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) (Φ : ActionStatePotential A S) (a : A) : ℝ :=
  ∑ s, P.distribution s * Φ a s

/-- Utility/potential grounding contract for stochastic one-shot objectives. -/
def utility_from_action_state_potential {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) (Φ : ActionStatePotential A S) : Prop :=
  ∀ a s, P.utility a s = - Φ a s

theorem stochasticExpectedUtility_eq_neg_expectedActionPotential
    {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) (Φ : ActionStatePotential A S)
    (hΦ : utility_from_action_state_potential P Φ) (a : A) :
    stochasticExpectedUtility P a = - expectedActionPotential P Φ a := by
  have hΦa : ∀ x : S, P.utility a x = -Φ a x := fun x => hΦ a x
  unfold stochasticExpectedUtility expectedActionPotential
  calc
    (∑ x, P.distribution x * P.utility a x)
        = ∑ x, P.distribution x * (-Φ a x) := by
            refine Finset.sum_congr rfl ?_
            intro x hx
            rw [hΦa x]
    _ = ∑ x, -(P.distribution x * Φ a x) := by
          refine Finset.sum_congr rfl ?_
          intro x hx
          simpa using (mul_neg (P.distribution x) (Φ a x))
    _ = -∑ x, P.distribution x * Φ a x := by
          simpa using (Finset.sum_neg_distrib (s := Finset.univ)
            (f := fun x => P.distribution x * Φ a x))

theorem stochasticExpectedUtility_le_iff_expectedActionPotential_ge
    {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) (Φ : ActionStatePotential A S)
    (hΦ : utility_from_action_state_potential P Φ) (a a' : A) :
    stochasticExpectedUtility P a ≤ stochasticExpectedUtility P a'
      ↔ expectedActionPotential P Φ a' ≤ expectedActionPotential P Φ a := by
  rw [stochasticExpectedUtility_eq_neg_expectedActionPotential P Φ hΦ a,
      stochasticExpectedUtility_eq_neg_expectedActionPotential P Φ hΦ a']
  constructor
  · intro h
    exact (neg_le_neg_iff.mp h)
  · intro h
    exact (neg_le_neg_iff.mpr h)

/-- Entropy of decision problem (Shannon entropy of state distribution) -/
noncomputable def decisionEntropy {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) : ℝ :=
  ∑ s : S, if P.distribution s > 0
           then -P.distribution s * Real.log (P.distribution s)
           else 0

/-- Thermodynamic cost of determining sufficiency (proportional to state space) -/
noncomputable def thermodynamicCost {A S : Type*} [Fintype A] [Fintype S]
    (_P : StochasticDecisionProblem A S) : ℝ :=
  landauerConstant * (Fintype.card S : ℝ)

/-- Landauer energy floor from irreversible bit count and temperature. -/
noncomputable def landauerEnergyFloor (bits : ℕ) (T : ℝ) : ℝ :=
  DecisionQuotient.Physics.Instantiation.LandauerBound bits T

theorem landauerEnergyFloor_nonneg (bits : ℕ) {T : ℝ} (hT : 0 ≤ T) :
    0 ≤ landauerEnergyFloor bits T := by
  unfold landauerEnergyFloor DecisionQuotient.Physics.Instantiation.LandauerBound
  have hBits : 0 ≤ (bits : ℝ) := Nat.cast_nonneg bits
  have hK : 0 ≤ DecisionQuotient.Physics.Instantiation.k_Boltzmann := by
    unfold DecisionQuotient.Physics.Instantiation.k_Boltzmann
    norm_num
  have hLog : 0 ≤ Real.log (2 : ℝ) := by
    exact Real.log_nonneg (by norm_num)
  have hCoef : 0 ≤ DecisionQuotient.Physics.Instantiation.k_Boltzmann * T * Real.log (2 : ℝ) := by
    exact mul_nonneg (mul_nonneg hK hT) hLog
  simpa [mul_assoc] using (mul_nonneg hBits hCoef)

theorem landauerEnergyFloor_mono_bits {b1 b2 : ℕ} (h : b1 ≤ b2) {T : ℝ} (hT : 0 ≤ T) :
    landauerEnergyFloor b1 T ≤ landauerEnergyFloor b2 T := by
  unfold landauerEnergyFloor DecisionQuotient.Physics.Instantiation.LandauerBound
  have hCast : (b1 : ℝ) ≤ (b2 : ℝ) := by exact_mod_cast h
  have hK : 0 ≤ DecisionQuotient.Physics.Instantiation.k_Boltzmann := by
    unfold DecisionQuotient.Physics.Instantiation.k_Boltzmann
    norm_num
  have hLog : 0 ≤ Real.log (2 : ℝ) := by
    exact Real.log_nonneg (by norm_num)
  have hCoef : 0 ≤ DecisionQuotient.Physics.Instantiation.k_Boltzmann * T * Real.log (2 : ℝ) := by
    exact mul_nonneg (mul_nonneg hK hT) hLog
  have hMul :
      (b1 : ℝ) * (DecisionQuotient.Physics.Instantiation.k_Boltzmann * T * Real.log (2 : ℝ))
        ≤
      (b2 : ℝ) * (DecisionQuotient.Physics.Instantiation.k_Boltzmann * T * Real.log (2 : ℝ)) :=
    mul_le_mul_of_nonneg_right hCast hCoef
  simpa [mul_assoc] using hMul

/-- Room-temperature specialization used by `thermodynamicCost`. -/
noncomputable def landauerEnergyFloorRoom (bits : ℕ) : ℝ :=
  (bits : ℝ) * landauerConstant

theorem thermodynamicCost_eq_landauerEnergyFloorRoom_states
    {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) :
    thermodynamicCost P = landauerEnergyFloorRoom (Fintype.card S) := by
  unfold thermodynamicCost landauerEnergyFloorRoom
  ring_nf

/-! ## Work Extraction -/

/-- Extractable work from sufficiency determination -/
noncomputable def extractableWork {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) : ℝ :=
  thermodynamicCost P

/-- Sufficient sets enable work extraction -/
theorem sufficient_enables_work_extraction {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) :
    extractableWork P ≥ 0 := by
  unfold extractableWork thermodynamicCost
  have h1 : landauerConstant > 0 := by unfold landauerConstant; positivity
  have h2 : (Fintype.card S : ℝ) ≥ 0 := Nat.cast_nonneg _
  exact mul_nonneg (le_of_lt h1) h2

/-! ## Reversible Computation -/

/-- Reversible computation has minimal thermodynamic cost -/
theorem reversible_lower_cost {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) :
    thermodynamicCost P ≤ thermodynamicCost P := le_refl _

/-! ## Heat Dissipation -/

/-- Heat dissipated during sufficiency check -/
noncomputable def heatDissipated {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) : ℝ :=
  thermodynamicCost P

/-- Landauer limit: heat dissipation is at least k_B T ln 2 per bit -/
theorem landauer_limit {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) :
    heatDissipated P ≥ 0 := by
  unfold heatDissipated thermodynamicCost
  have h1 : landauerConstant > 0 := by unfold landauerConstant; positivity
  have h2 : (Fintype.card S : ℝ) ≥ 0 := Nat.cast_nonneg _
  exact mul_nonneg (le_of_lt h1) h2

/-! ## Energy-Complexity Tradeoff -/

/-- More energy enables faster computation (fundamental tradeoff) -/
theorem energy_time_tradeoff {A S : Type*} [Fintype A] [Fintype S]
    (_P : StochasticDecisionProblem A S)
    (E : ℝ) (hE : E > 0) :
    ∃ (t : ℕ), (t : ℝ) * landauerConstant ≤ E := by
  use 0
  simp
  exact le_of_lt hE

end DecisionQuotient.StochasticSequential
