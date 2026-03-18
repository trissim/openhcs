/-
  Paper 4: Decision-Relevant Uncertainty
  ContinuousInformation.lean - Measure-Theoretic Decision Entropy
  
  Extends the decision quotient from discrete combinatorial cardinalities
  to continuous probability spaces, defining the Boltzmann measure and
  the continuous decision entropy over Euclidean space.
-/
import DecisionQuotient.Basic
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
import Mathlib.MeasureTheory.Integral.Lebesgue.Basic
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp

namespace DecisionQuotient
namespace Continuous

open MeasureTheory
open scoped ENNReal

/-- 
  A continuous decision problem defined over a measurable state space,
  equipped with a probability measure (e.g., the Boltzmann distribution P(s)).
-/
structure ContinuousDecisionProblem (A S : Type*) [MeasurableSpace S] [MeasureSpace S]
    extends DecisionProblem A S where
  /-- The probability measure governing the state uncertainty. -/
  measure : Measure S
  /-- The measure is a normalized probability distribution. -/
  is_probability_measure : IsProbabilityMeasure measure

variable {A S : Type*} [MeasurableSpace S] [MeasureSpace S] 

/-- 
  The measure of a specific decision-equivalence class (a region of S).
  Given a set of optimal actions O, this is the probability mass of all 
  states s where Opt(s) = O.
-/
noncomputable def decisionClassMeasure 
    (cdp : ContinuousDecisionProblem A S) (O : Set A) : ℝ :=
  (cdp.measure { s : S | cdp.toDecisionProblem.Opt s = O }).toReal

/--
  The continuous measure-theoretic decision entropy.
  H(D) = - ∑_O P(O) ln P(O)
  This replaces the discrete `log(numOptClasses)` with the rigorous Shannon entropy
  of the probability pushforward onto the decision quotient.
-/
noncomputable def continuousQuotientEntropy 
    [Fintype (Set A)] -- Assuming a finite number of possible optimal decision sets
    (cdp : ContinuousDecisionProblem A S) : ℝ :=
  - ∑ O : Set A, 
      let p := decisionClassMeasure cdp O
      if p = 0 then 0 else p * Real.log p

/--
  The partition function Z for a continuous state space.
  Z = ∫ exp(-β * E(s)) ds
-/
noncomputable def partitionFunction (β : ℝ) (E : S → ℝ) : ℝ :=
  ∫ s, Real.exp (-β * E s)

/--
  The Boltzmann probability density function for a given energy landscape E(s).
  P(s) = (1/Z) • exp(-β * E(s))
  
  (Defined using scalar multiplication `•` to match Mathlib's generic measure integrals).
-/
noncomputable def boltzmannDensity 
    (β : ℝ) (E : S → ℝ) (s : S) : ℝ :=
  (1 / partitionFunction β E) • Real.exp (-β * E s)

/--
  Rigorous proof of the Boltzmann normalization over the continuous state space.
  Replaces the previous unverified axiom. Given a physically bounded energy 
  landscape where Z > 0, the continuous Boltzmann density strictly integrates to 1.
-/
theorem boltzmann_is_probability (β : ℝ) (E : S → ℝ) 
    (hZ_pos : partitionFunction β E > 0)
    (hZ_finite : Integrable (fun s => Real.exp (-β * E s))) :
    ∫ s, boltzmannDensity β E s = 1 := by
  unfold boltzmannDensity
  -- Pull the constant partition function out of the Lebesgue integral
  have h_pull : ∫ s, (1 / partitionFunction β E) • Real.exp (-β * E s) =
    (1 / partitionFunction β E) • ∫ s, Real.exp (-β * E s) := by
    exact integral_smul (1 / partitionFunction β E) (fun s => Real.exp (-β * E s))
  rw [h_pull]
  -- Multiply out: (1/Z) * Z = 1
  have h_pos : partitionFunction β E > 0 := hZ_pos
  unfold partitionFunction at h_pos
  have hZ : ∫ (s : S), Real.exp (-β * E s) ≠ 0 := by linarith
  exact div_mul_cancel₀ 1 hZ

end Continuous
end DecisionQuotient
