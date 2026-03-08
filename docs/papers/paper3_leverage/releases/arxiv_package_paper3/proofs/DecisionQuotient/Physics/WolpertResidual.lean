/-
  Paper 4: Decision-Relevant Uncertainty

  Physics/WolpertResidual.lean

  This module promotes the strongest additional part of the residual branch that
  the current machinery can honestly prove without rebuilding full trajectory
  stochastic thermodynamics.

  The key idea is finite and local:

  - Given a discrete Markov process, compare the stationary edge flow `s → s'`
    against the reverse edge flow `s' → s`.
  - If both directions are positive and the two flows differ, then the induced
    two-point forward/reverse flow distributions differ.
  - The existing KL machinery then forces strictly positive divergence.

  This does not claim the full stopping-time / absolute-irreversibility theorem
  of the cited papers. It proves a theorem-level finite-support residual
  asymmetry branch that can be composed with the current Wolpert decomposition.
-/

import DecisionQuotient.Physics.TUR
import DecisionQuotient.Physics.IntegrityEquilibrium
import DecisionQuotient.Physics.WolpertMismatch
import Mathlib.Algebra.Order.Floor.Ring

open scoped BigOperators
open Finset

namespace DecisionQuotient
namespace Physics
namespace WolpertResidual

open WolpertMismatch
open DecisionCircuit

/-- Stationary probabilities are nonnegative. -/
theorem stationaryProb_nonneg {S : Type*} [Fintype S]
    {mc : DiscreteMarkovChain S} (π : StationaryDist mc) (s : S) :
    0 ≤ stationaryProb π s := by
  unfold stationaryProb
  apply div_nonneg <;> exact Nat.cast_nonneg _

/-- Stationary edge-flow weight on the directed edge `s → s'`. -/
noncomputable def edgeFlow {S : Type*} [Fintype S]
    (mc : DiscreteMarkovChain S) (π : StationaryDist mc) (s s' : S) : ℝ :=
  stationaryProb π s * transitionProb mc s s'

/-- Stationary edge flows are nonnegative. -/
theorem edgeFlow_nonneg {S : Type*} [Fintype S]
    (mc : DiscreteMarkovChain S) (π : StationaryDist mc) (s s' : S) :
    0 ≤ edgeFlow mc π s s' := by
  unfold edgeFlow
  exact mul_nonneg (stationaryProb_nonneg π s) (transitionProb_nonneg mc s s')

/-- Total two-edge mass used to normalize the local forward/reverse pair. -/
noncomputable def pairFlowMass {S : Type*} [Fintype S]
    (mc : DiscreteMarkovChain S) (π : StationaryDist mc) (s s' : S) : ℝ :=
  edgeFlow mc π s s' + edgeFlow mc π s' s

/-- The normalized two-point forward distribution for the pair `(s,s')`. -/
noncomputable def forwardPairDistribution {S : Type*} [Fintype S]
    (mc : DiscreteMarkovChain S) (π : StationaryDist mc) (s s' : S)
    (hForward : 0 < edgeFlow mc π s s')
    (hReverse : 0 < edgeFlow mc π s' s) :
    StrictFiniteDistribution Bool := by
  refine
    { pmf := fun b =>
        if b then
          edgeFlow mc π s s' / pairFlowMass mc π s s'
        else
          edgeFlow mc π s' s / pairFlowMass mc π s s'
      sum_eq_one := ?_
      pos := ?_ }
  · have hMassNe : pairFlowMass mc π s s' ≠ 0 := by
      unfold pairFlowMass
      linarith
    rw [Fintype.sum_bool]
    simp
    field_simp [hMassNe]
    simp [pairFlowMass]
  · intro b
    by_cases hb : b
    · simp [hb]
      have hMassPos : 0 < pairFlowMass mc π s s' := by
        unfold pairFlowMass
        linarith
      exact div_pos hForward hMassPos
    · simp [hb]
      have hMassPos : 0 < pairFlowMass mc π s s' := by
        unfold pairFlowMass
        linarith
      exact div_pos hReverse hMassPos

/-- The normalized two-point reverse distribution for the pair `(s,s')`. -/
noncomputable def reversePairDistribution {S : Type*} [Fintype S]
    (mc : DiscreteMarkovChain S) (π : StationaryDist mc) (s s' : S)
    (hForward : 0 < edgeFlow mc π s s')
    (hReverse : 0 < edgeFlow mc π s' s) :
    StrictFiniteDistribution Bool := by
  refine
    { pmf := fun b =>
        if b then
          edgeFlow mc π s' s / pairFlowMass mc π s s'
        else
          edgeFlow mc π s s' / pairFlowMass mc π s s'
      sum_eq_one := ?_
      pos := ?_ }
  · have hMassNe : pairFlowMass mc π s s' ≠ 0 := by
      unfold pairFlowMass
      linarith
    rw [Fintype.sum_bool]
    simp
    field_simp [hMassNe]
    simp [pairFlowMass, add_comm]
  · intro b
    by_cases hb : b
    · simp [hb]
      have hMassPos : 0 < pairFlowMass mc π s s' := by
        unfold pairFlowMass
        linarith
      exact div_pos hReverse hMassPos
    · simp [hb]
      have hMassPos : 0 < pairFlowMass mc π s s' := by
        unfold pairFlowMass
        linarith
      exact div_pos hForward hMassPos

/-- Two-point KL divergence comparing the local forward and reverse edge-flow
distributions. This is the theorem-level finite residual asymmetry quantity
available from the current machinery. -/
noncomputable def pairwiseResidualKL {S : Type*} [Fintype S]
    (mc : DiscreteMarkovChain S) (π : StationaryDist mc) (s s' : S)
    (hForward : 0 < edgeFlow mc π s s')
    (hReverse : 0 < edgeFlow mc π s' s) : ℝ :=
  mismatchKL
    (forwardPairDistribution mc π s s' hForward hReverse)
    (reversePairDistribution mc π s s' hForward hReverse)

/-- The local residual asymmetry quantity is always nonnegative. -/
theorem pairwiseResidualKL_nonneg {S : Type*} [Fintype S]
    (mc : DiscreteMarkovChain S) (π : StationaryDist mc) (s s' : S)
    (hForward : 0 < edgeFlow mc π s s')
    (hReverse : 0 < edgeFlow mc π s' s) :
    0 ≤ pairwiseResidualKL mc π s s' hForward hReverse := by
  unfold pairwiseResidualKL
  exact mismatchKL_nonneg
    (forwardPairDistribution mc π s s' hForward hReverse)
    (reversePairDistribution mc π s s' hForward hReverse)

/-- Any explicit asymmetry between the two directed edge flows forces strictly
positive finite-support residual divergence. -/
theorem pairwiseResidualKL_pos_of_asymmetry {S : Type*} [Fintype S]
    (mc : DiscreteMarkovChain S) (π : StationaryDist mc) (s s' : S)
    (hForward : 0 < edgeFlow mc π s s')
    (hReverse : 0 < edgeFlow mc π s' s)
    (hAsym : edgeFlow mc π s s' ≠ edgeFlow mc π s' s) :
    0 < pairwiseResidualKL mc π s s' hForward hReverse := by
  unfold pairwiseResidualKL
  have hMassNe : pairFlowMass mc π s s' ≠ 0 := by
    unfold pairFlowMass
    linarith
  refine mismatchKL_pos_of_exists_ne
    (forwardPairDistribution mc π s s' hForward hReverse)
    (reversePairDistribution mc π s s' hForward hReverse) ?_
  refine ⟨true, ?_⟩
  simp [forwardPairDistribution, reversePairDistribution]
  intro hEq
  field_simp [hMassNe] at hEq
  exact hAsym hEq

/-- Nat-valued residual lower-bound units obtained by conservatively rounding
the finite-support residual asymmetry witness upward. -/
noncomputable def residualNatLowerBound {S : Type*} [Fintype S]
    (mc : DiscreteMarkovChain S) (π : StationaryDist mc) (s s' : S)
    (hForward : 0 < edgeFlow mc π s s')
    (hReverse : 0 < edgeFlow mc π s' s) : ℕ :=
  Nat.ceil (pairwiseResidualKL mc π s s' hForward hReverse)

/-- Any positive finite-support residual asymmetry witness yields a positive
nat-valued lower-bound term after the declared upward rounding. -/
theorem residualNatLowerBound_pos_of_asymmetry {S : Type*} [Fintype S]
    (mc : DiscreteMarkovChain S) (π : StationaryDist mc) (s s' : S)
    (hForward : 0 < edgeFlow mc π s s')
    (hReverse : 0 < edgeFlow mc π s' s)
    (hAsym : edgeFlow mc π s s' ≠ edgeFlow mc π s' s) :
    0 < residualNatLowerBound mc π s s' hForward hReverse := by
  unfold residualNatLowerBound
  exact (Nat.ceil_pos).2 (pairwiseResidualKL_pos_of_asymmetry mc π s s' hForward hReverse hAsym)

/-- Nat-valued lower-bound term induced by an irreversible one-way state
transition on `ComputationalState`. The real transition cost is converted into
the artifact's discrete lower-bound units by dividing by the declared Landauer
unit and rounding upward. -/
noncomputable def irreversibleTransitionNatLowerBound
    (kT_ln2 : ℝ) (s s' : ComputationalState) : ℕ :=
  Nat.ceil (transitionCost kT_ln2 s s' / kT_ln2)

/-- Any distinct computational-state transition carries a strictly positive
nat-valued lower-bound witness after the declared unit conversion. -/
theorem irreversibleTransitionNatLowerBound_pos
    [Fintype ComputationalState]
    {kT_ln2 : ℝ} (hkT : 0 < kT_ln2)
    (s s' : ComputationalState) (hNe : s ≠ s') :
    0 < irreversibleTransitionNatLowerBound kT_ln2 s s' := by
  unfold irreversibleTransitionNatLowerBound
  have hCostPos : 0 < transitionCost kT_ln2 s s' :=
    cycle_cost_lower_bound kT_ln2 hkT s s' hNe
  have hQuotPos : 0 < transitionCost kT_ln2 s s' / kT_ln2 := by
    exact div_pos hCostPos hkT
  exact (Nat.ceil_pos).2 hQuotPos

/-- If a forward stationary edge flow is positive while the reverse edge flow
vanishes, the corresponding computational states must be distinct. -/
theorem ne_of_forward_pos_reverse_zero
    [Fintype ComputationalState]
    {mc : DiscreteMarkovChain ComputationalState}
    (π : StationaryDist mc) (s s' : ComputationalState)
    (hForward : 0 < edgeFlow mc π s s')
    (hReverseZero : edgeFlow mc π s' s = 0) :
    s ≠ s' := by
  intro hEq
  subst hEq
  linarith

/-- Unified finite residual lower-bound term for discrete computational-state
processes. The definition performs the exhaustive local case split:

* if the reverse edge flow is positive, use the existing finite KL asymmetry
  witness;
* if the reverse edge flow vanishes, use the irreversible state-transition
  witness coming from the existing Landauer-scaled transition-cost theorem. -/
noncomputable def discreteResidualNatLowerBound
    (kT_ln2 : ℝ)
    [Fintype ComputationalState]
    {mc : DiscreteMarkovChain ComputationalState}
    (π : StationaryDist mc) (s s' : ComputationalState)
    (hForward : 0 < edgeFlow mc π s s') : ℕ :=
  if hReverse : 0 < edgeFlow mc π s' s then
    residualNatLowerBound mc π s s' hForward hReverse
  else
    irreversibleTransitionNatLowerBound kT_ln2 s s'

/-- For finite computational-state processes, any positive forward edge with a
decision-relevant asymmetry yields a strictly positive residual lower-bound
term. The proof is exhaustive over the reverse edge: either it is positive, in
which case the existing pairwise KL branch applies, or it is zero, in which
case the process performs an irreversible one-way state transition and the
existing Landauer-scaled transition-cost theorem applies. -/
theorem discreteResidualNatLowerBound_pos_of_asymmetry_or_oneway
    [Fintype ComputationalState]
    {kT_ln2 : ℝ} (hkT : 0 < kT_ln2)
    {mc : DiscreteMarkovChain ComputationalState}
    (π : StationaryDist mc) (s s' : ComputationalState)
    (hForward : 0 < edgeFlow mc π s s')
    (hAsym : edgeFlow mc π s s' ≠ edgeFlow mc π s' s) :
    0 < discreteResidualNatLowerBound kT_ln2 π s s' hForward := by
  unfold discreteResidualNatLowerBound
  by_cases hReverse : 0 < edgeFlow mc π s' s
  · simpa [hReverse] using
      residualNatLowerBound_pos_of_asymmetry mc π s s' hForward hReverse hAsym
  · have hReverseNonneg : 0 ≤ edgeFlow mc π s' s :=
      edgeFlow_nonneg mc π s' s
    have hReverseZero : edgeFlow mc π s' s = 0 := by
      linarith
    have hNe : s ≠ s' := ne_of_forward_pos_reverse_zero π s s' hForward hReverseZero
    simpa [hReverse, hReverseZero] using
      irreversibleTransitionNatLowerBound_pos hkT s s' hNe

/-- A finite discrete residual witness consists of a computational-state edge
with positive forward flow and decision-relevant asymmetry. This is the exact
theorem-level finite subclass of the broader stopping-time / absolute-
irreversibility story that the current artifact can discharge without adding
new physics axioms. -/
structure FiniteDiscreteResidualWitness
    [Fintype ComputationalState]
    {mc : DiscreteMarkovChain ComputationalState}
    (π : StationaryDist mc) where
  s : ComputationalState
  s' : ComputationalState
  hForward : 0 < edgeFlow mc π s s'
  hAsym : edgeFlow mc π s s' ≠ edgeFlow mc π s' s

/-- Any finite discrete residual witness yields a strictly positive residual
lower-bound term. -/
theorem discreteResidualNatLowerBound_pos_of_witness
    [Fintype ComputationalState]
    {kT_ln2 : ℝ} (hkT : 0 < kT_ln2)
    {mc : DiscreteMarkovChain ComputationalState}
    (π : StationaryDist mc)
    (h : FiniteDiscreteResidualWitness π) :
    0 < discreteResidualNatLowerBound kT_ln2 π h.s h.s' h.hForward := by
  exact discreteResidualNatLowerBound_pos_of_asymmetry_or_oneway
    hkT π h.s h.s' h.hForward h.hAsym

end WolpertResidual
end Physics
end DecisionQuotient
