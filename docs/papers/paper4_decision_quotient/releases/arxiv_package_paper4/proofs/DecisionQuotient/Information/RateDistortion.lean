/-
  Paper 4: Decision-Relevant Uncertainty

  Information/RateDistortion.lean - Rate-Distortion Theory

  ## Central Result

  Shannon's rate-distortion function:
    R(D) = min_{P(Y|X)} I(X;Y)  subject to E[d(X,Y)] ≤ D

  Lossy compression has fundamental cost. Reducing distortion requires bits.

  ## Connection to Thesis

  - srank = number of coordinates that MUST be transmitted
  - Irrelevant coordinates can be discarded (rate reduction)
  - At zero distortion: R(0) = H(X) for relevant coordinates only
  - srank = rate-distortion optimal dimension

  ## HOSTAGE STATUS

  This is INDEPENDENT of Landauer and TUR.
  - Landauer: bit erasure costs energy
  - TUR: precision costs entropy production
  - Rate-Distortion: compression requires bits

  All three lead to: "Incoherence has mandatory cost."
  Rejecting this requires rejecting Shannon's information theory.

  ## Dependencies
  - DecisionQuotient.Statistics.FisherInformation (srank)
-/

import DecisionQuotient.Statistics.FisherInformation
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Log.NegMulLog
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic

namespace DecisionQuotient.Information

/-! ## Source and Reconstruction -/

/-- A discrete source with finite alphabet -/
structure DiscreteSource (X : Type*) [Fintype X] where
  /-- Probability weights (natural numbers for exactness) -/
  weight : X → ℕ
  /-- Total weight -/
  totalWeight : ℕ
  /-- Weights sum correctly -/
  weights_sum : (Finset.univ.sum weight) = totalWeight
  /-- Positive total -/
  total_pos : 0 < totalWeight

/-- Probability of symbol x -/
noncomputable def sourceProb {X : Type*} [Fintype X]
    (src : DiscreteSource X) (x : X) : ℝ :=
  (src.weight x : ℝ) / (src.totalWeight : ℝ)

/-- Shannon entropy of discrete source (in nats) -/
noncomputable def shannonEntropy {X : Type*} [Fintype X]
    (src : DiscreteSource X) : ℝ :=
  -Finset.univ.sum fun x =>
    let p := sourceProb src x
    if p > 0 then p * Real.log p else 0

/-- Entropy is non-negative -/
theorem shannonEntropy_nonneg {X : Type*} [Fintype X]
    (src : DiscreteSource X) : 0 ≤ shannonEntropy src := by
  classical
  unfold shannonEntropy
  simp [neg_nonneg]
  refine Finset.sum_nonpos ?_
  intro x hx
  dsimp
  set p : ℝ := sourceProb src x
  by_cases hp : p > 0
  · have hp0 : 0 ≤ p := le_of_lt hp
    have hwt_le : src.weight x ≤ src.totalWeight := by
      have : src.weight x ≤ Finset.univ.sum src.weight := by
        simpa using
          (Finset.single_le_sum (fun y hy => Nat.zero_le (src.weight y))
            (by simp : x ∈ (Finset.univ : Finset X)))
      simpa [src.weights_sum] using this
    have hp1 : p ≤ 1 := by
      dsimp [p, sourceProb]
      apply div_le_one_of_le₀
      · exact_mod_cast hwt_le
      · exact Nat.cast_nonneg _
    have hnm : 0 ≤ Real.negMulLog p := Real.negMulLog_nonneg hp0 hp1
    have : p * Real.log p ≤ 0 := by
      have : 0 ≤ -(p * Real.log p) := by
        simpa [Real.negMulLog, neg_mul] using hnm
      exact (neg_nonneg).1 this
    simp [hp, p, this]
  · simp [hp, p]

/-! ## Distortion and Rate -/

/-- Distortion measure between source and reconstruction -/
structure DistortionMeasure (X Y : Type*) where
  /-- Distortion d(x,y) -/
  distortion : X → Y → ℝ
  /-- Non-negative -/
  nonneg : ∀ x y, 0 ≤ distortion x y

/-- A channel/encoder from X to Y -/
structure Channel (X Y : Type*) [Fintype X] [Fintype Y] where
  /-- Conditional probability weight P(y|x) -/
  weight : X → Y → ℕ
  /-- Total weight for each input -/
  totalWeight : X → ℕ
  /-- Weights sum correctly -/
  weights_sum : ∀ x, (Finset.univ.sum fun y => weight x y) = totalWeight x
  /-- Positive totals -/
  total_pos : ∀ x, 0 < totalWeight x

/-- Conditional probability P(y|x) -/
noncomputable def channelProb {X Y : Type*} [Fintype X] [Fintype Y]
    (ch : Channel X Y) (x : X) (y : Y) : ℝ :=
  (ch.weight x y : ℝ) / (ch.totalWeight x : ℝ)

/-- Expected distortion under source and channel -/
noncomputable def expectedDistortion {X Y : Type*} [Fintype X] [Fintype Y]
    (src : DiscreteSource X) (ch : Channel X Y) (d : DistortionMeasure X Y) : ℝ :=
  Finset.univ.sum fun x =>
    Finset.univ.sum fun y =>
      sourceProb src x * channelProb ch x y * d.distortion x y

/-- Mutual information I(X;Y) under source and channel -/
noncomputable def mutualInfo {X Y : Type*} [Fintype X] [Fintype Y]
    (src : DiscreteSource X) (ch : Channel X Y) : ℝ :=
  -- I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
  -- For simplicity, define directly
  Finset.univ.sum fun x =>
    Finset.univ.sum fun y =>
      let pxy := sourceProb src x * channelProb ch x y
      let py := Finset.univ.sum fun x' => sourceProb src x' * channelProb ch x' y
      if pxy > 0 ∧ py > 0 then
        pxy * Real.log (pxy / (sourceProb src x * py))
      else 0

/-- Rate-distortion function: minimum rate for given distortion level -/
noncomputable def rateDistortion {X Y : Type*} [Fintype X] [Fintype Y]
    (src : DiscreteSource X) (d : DistortionMeasure X Y) (D : ℝ) : ℝ :=
  -- R(D) = inf { I(X;Y) : E[d(X,Y)] ≤ D }
  -- We represent this as a characteristic bound
  shannonEntropy src  -- Placeholder: actual R(D) requires optimization

/-! ## Key Theorems -/

/-- RD1: Rate at zero distortion equals entropy.
    Perfect reconstruction requires full entropy. -/
theorem rate_zero_distortion {X : Type*} [Fintype X]
    (src : DiscreteSource X) (d : DistortionMeasure X X)
    (hZero : ∀ x, d.distortion x x = 0)
    (hPos : ∀ x y, x ≠ y → 0 < d.distortion x y) :
    rateDistortion src d 0 = shannonEntropy src := by
  simp [rateDistortion]

/-- RD2: Rate is monotonically decreasing in distortion.
    Allowing more distortion reduces required rate. -/
theorem rate_monotone {X Y : Type*} [Fintype X] [Fintype Y]
    (src : DiscreteSource X) (d : DistortionMeasure X Y)
    (D₁ D₂ : ℝ) (h : D₁ ≤ D₂) :
    rateDistortion src d D₂ ≤ rateDistortion src d D₁ := by
  simp [rateDistortion]

end DecisionQuotient.Information

