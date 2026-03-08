/-
  Paper 4: Decision-Relevant Uncertainty

  Physics/WolpertDecomposition.lean

  Refined Wolpert-facing interface:

  - Landauer remains the universal floor for irreversible bit erasure.
  - Extra constrained-process dissipation is split explicitly into a mismatch
    term and a residual term.
  - The mismatch branch is now promoted as far as the existing machinery
    supports: it is derived from the finite-distribution KL layer in
    `WolpertMismatch.lean`.
  - We do not claim to rederive the full stochastic-thermodynamics proofs of
    Wolpert et al., Manzano et al., or Yadav et al. here. The broader
    stopping-time / absolute-irreversibility residual regime remains an
    imported cited premise, while a finite discrete residual branch can now be
    constructed from theorem-level edge-flow asymmetry or irreversible one-way
    state-transition witnesses and then composed inside the decision-theoretic
    framework.

  This gives the paper a theorem-level representation of the specific parts of
  the cited work that it uses:
  - PNAS 2024: constrained computation is not characterized by Landauer alone
  - Phys. Rev. X 2024: mismatch / intrinsic dissipation under periodic,
    stopping-time, absolutely irreversible regimes
  - arXiv 2026: structural resource can lower-bound thermodynamic mismatch cost
-/

import DecisionQuotient.Physics.WolpertConstraints
import DecisionQuotient.Physics.WolpertMismatch
import DecisionQuotient.Physics.WolpertResidual

namespace DecisionQuotient
namespace Physics
namespace WolpertDecomposition

open ThermodynamicLift
open WolpertConstraints
open WolpertMismatch
open WolpertResidual
open BoundedAcquisition
open DecisionCircuit

/-- A more explicit constrained-process model in which the extra dissipation
above the Landauer floor is decomposed into two named components:

* `mismatchCostPerBit`: the mismatch / intrinsic-dissipation contribution
* `residualDissipationPerBit`: additional residual entropy production

Both are represented as declared lower-bound terms in the same discrete units as
the existing `ThermoModel`. -/
structure DecomposedProcessModel where
  base : ThermoModel
  mismatchCostPerBit : ℕ
  residualDissipationPerBit : ℕ

/-- Total explicit constrained-process overhead above the base model. -/
def DecomposedProcessModel.totalOverheadPerBit (W : DecomposedProcessModel) : ℕ :=
  W.mismatchCostPerBit + W.residualDissipationPerBit

/-- Forgetful map to the coarser floor-plus-overhead interface. -/
def DecomposedProcessModel.asConstrainedProcessModel
    (W : DecomposedProcessModel) : ConstrainedProcessModel :=
  { base := W.base
    extraDissipationPerBit := W.totalOverheadPerBit }

/-- Effective thermodynamic lower-bound model after adding both decomposition
components. -/
def DecomposedProcessModel.effectiveModel (W : DecomposedProcessModel) : ThermoModel :=
  (W.asConstrainedProcessModel).effectiveModel

/-- The total declared overhead is exactly the sum of the mismatch and residual
components. -/
theorem DecomposedProcessModel.totalOverheadPerBit_eq_sum
    (W : DecomposedProcessModel) :
    W.totalOverheadPerBit = W.mismatchCostPerBit + W.residualDissipationPerBit := rfl

/-- The refined floor statement: if the base declared model dominates the
Landauer floor, then the effective constrained-process model dominates the
Landauer floor plus both declared decomposition terms. -/
theorem landauer_floor_plus_decomposition_lower_bound
    (W : DecomposedProcessModel) {kB T : ℝ}
    (hFloor : landauerJoulesPerBit kB T ≤ (W.base.joulesPerBit : ℝ)) :
    landauerJoulesPerBit kB T + (W.mismatchCostPerBit : ℝ) + (W.residualDissipationPerBit : ℝ) ≤
      ((W.effectiveModel).joulesPerBit : ℝ) := by
  have hBase :=
    landauer_floor_plus_overhead_lower_bound W.asConstrainedProcessModel hFloor
  simpa [DecomposedProcessModel.effectiveModel, DecomposedProcessModel.asConstrainedProcessModel,
    DecomposedProcessModel.totalOverheadPerBit, add_assoc]
    using hBase

/-- The decomposed effective model still dominates the Landauer floor. -/
theorem effective_model_dominates_landauer_floor_decomposition
    (W : DecomposedProcessModel) {kB T : ℝ}
    (hFloor : landauerJoulesPerBit kB T ≤ (W.base.joulesPerBit : ℝ)) :
    landauerJoulesPerBit kB T ≤ ((W.effectiveModel).joulesPerBit : ℝ) := by
  exact effective_model_dominates_landauer_floor W.asConstrainedProcessModel hFloor

/-- Compatibility wrapper preserving the downstream interface of the
periodic / modular mismatch branch. Unlike the residual branch, this positivity
can now be constructed from a theorem-level KL mismatch witness
(`WolpertMismatch.lean`). -/
structure PeriodicModularMismatchHypothesis (W : DecomposedProcessModel) : Prop where
  mismatchPos : 0 < W.mismatchCostPerBit

/-- Imported empirical interface for the stopping-time / absolute-irreversibility
regime of Manzano et al. (Phys. Rev. X 2024). Again, only the theorem-level
consequence used here is recorded: the residual term is strictly positive under
the declared constrained-process regime. -/
structure StoppingTimeResidualHypothesis (W : DecomposedProcessModel) : Prop where
  residualPos : 0 < W.residualDissipationPerBit

/-- Imported structural-scaling interface for repeated circuit families in
Yadav--Yousef--Wolpert (2026): a declared structural resource can lower-bound the
mismatch contribution. This is the exact part that composes with the present
artifact. -/
structure CircuitStructuralScalingHypothesis
    (W : DecomposedProcessModel) (structuralResource : ℕ) : Prop where
  resourceDominated : structuralResource ≤ W.mismatchCostPerBit

/-- The KL-derived mismatch lower bound can be injected into the existing
periodic/modular mismatch interface by declaring the artifact's nat-valued
mismatch term to be the rounded-up KL witness. This keeps the downstream API
stable while promoting the mismatch branch from cited premise to theorem. -/
theorem periodic_modular_mismatch_of_distribution_mismatch
    (W : DecomposedProcessModel)
    {α : Type*} [Fintype α]
    (actual designed : StrictFiniteDistribution α)
    (hUnits : W.mismatchCostPerBit = mismatchNatLowerBound actual designed)
    (hNe : ∃ a, actual.pmf a ≠ designed.pmf a) :
    PeriodicModularMismatchHypothesis W := by
  refine ⟨?_⟩
  rw [hUnits]
  exact mismatchNatLowerBound_pos_of_exists_ne actual designed hNe

/-- Positive mismatch implies positive total overhead. -/
theorem total_overhead_pos_of_positive_mismatch
    (W : DecomposedProcessModel)
    (h : PeriodicModularMismatchHypothesis W) :
    0 < W.totalOverheadPerBit := by
  have hmle : W.mismatchCostPerBit ≤ W.totalOverheadPerBit := by
    simp [DecomposedProcessModel.totalOverheadPerBit]
  exact lt_of_lt_of_le h.mismatchPos hmle

/-- Positive residual dissipation implies positive total overhead. -/
theorem total_overhead_pos_of_positive_residual
    (W : DecomposedProcessModel)
    (h : StoppingTimeResidualHypothesis W) :
    0 < W.totalOverheadPerBit := by
  have hrle : W.residualDissipationPerBit ≤ W.totalOverheadPerBit := by
    simp [DecomposedProcessModel.totalOverheadPerBit]
  exact lt_of_lt_of_le h.residualPos hrle

/-- The theorem-level KL mismatch witness already forces positive total
overhead, without any additional imported premise on the mismatch branch. -/
theorem total_overhead_pos_of_distribution_mismatch
    (W : DecomposedProcessModel)
    {α : Type*} [Fintype α]
    (actual designed : StrictFiniteDistribution α)
    (hUnits : W.mismatchCostPerBit = mismatchNatLowerBound actual designed)
    (hNe : ∃ a, actual.pmf a ≠ designed.pmf a) :
    0 < W.totalOverheadPerBit := by
  have hMismatch : PeriodicModularMismatchHypothesis W :=
    periodic_modular_mismatch_of_distribution_mismatch W actual designed hUnits hNe
  exact total_overhead_pos_of_positive_mismatch W hMismatch

/-- Under the periodic / modular mismatch hypothesis, the effective per-bit
lower bound is strictly above the Landauer floor. -/
theorem effective_model_strictly_exceeds_landauer_of_periodic_modular_mismatch
    (W : DecomposedProcessModel) {kB T : ℝ}
    (hFloor : landauerJoulesPerBit kB T ≤ (W.base.joulesPerBit : ℝ))
    (h : PeriodicModularMismatchHypothesis W) :
    landauerJoulesPerBit kB T < ((W.effectiveModel).joulesPerBit : ℝ) := by
  have hOver : 0 < W.totalOverheadPerBit :=
    total_overhead_pos_of_positive_mismatch W h
  exact effective_model_strictly_exceeds_landauer_of_strict_overhead
    W.asConstrainedProcessModel hFloor hOver

/-- The mismatch branch can now be discharged directly from the existing
finite-distribution KL layer: if the actual and designed distributions differ,
the effective per-bit lower bound is already strictly above the Landauer
floor. -/
theorem effective_model_strictly_exceeds_landauer_of_distribution_mismatch
    (W : DecomposedProcessModel) {kB T : ℝ}
    (hFloor : landauerJoulesPerBit kB T ≤ (W.base.joulesPerBit : ℝ))
    {α : Type*} [Fintype α]
    (actual designed : StrictFiniteDistribution α)
    (hUnits : W.mismatchCostPerBit = mismatchNatLowerBound actual designed)
    (hNe : ∃ a, actual.pmf a ≠ designed.pmf a) :
    landauerJoulesPerBit kB T < ((W.effectiveModel).joulesPerBit : ℝ) := by
  have hMismatch : PeriodicModularMismatchHypothesis W :=
    periodic_modular_mismatch_of_distribution_mismatch W actual designed hUnits hNe
  exact effective_model_strictly_exceeds_landauer_of_periodic_modular_mismatch W hFloor hMismatch

/-- Under the stopping-time / absolute-irreversibility residual hypothesis, the
effective per-bit lower bound is strictly above the Landauer floor. -/
theorem effective_model_strictly_exceeds_landauer_of_stopping_time_residual
    (W : DecomposedProcessModel) {kB T : ℝ}
    (hFloor : landauerJoulesPerBit kB T ≤ (W.base.joulesPerBit : ℝ))
    (h : StoppingTimeResidualHypothesis W) :
    landauerJoulesPerBit kB T < ((W.effectiveModel).joulesPerBit : ℝ) := by
  have hOver : 0 < W.totalOverheadPerBit :=
    total_overhead_pos_of_positive_residual W h
  exact effective_model_strictly_exceeds_landauer_of_strict_overhead
    W.asConstrainedProcessModel hFloor hOver

/-- A theorem-level finite residual asymmetry witness can also discharge the
residual branch: if a pair of stationary edge flows is positive in both
directions and asymmetric, the resulting nat-valued residual lower-bound term
is strictly positive. This is the strongest residual promotion currently proved
inside the artifact without claiming the full stopping-time / absolute-
irreversibility theorems of the cited papers. -/
theorem stopping_time_residual_of_pairwise_flow_asymmetry
    (W : DecomposedProcessModel)
    {S : Type*} [Fintype S]
    (mc : DiscreteMarkovChain S) (π : StationaryDist mc) (s s' : S)
    (hForward : 0 < edgeFlow mc π s s')
    (hReverse : 0 < edgeFlow mc π s' s)
    (hUnits : W.residualDissipationPerBit = residualNatLowerBound mc π s s' hForward hReverse)
    (hAsym : edgeFlow mc π s s' ≠ edgeFlow mc π s' s) :
    StoppingTimeResidualHypothesis W := by
  refine ⟨?_⟩
  rw [hUnits]
  exact residualNatLowerBound_pos_of_asymmetry mc π s s' hForward hReverse hAsym

/-- Finite bidirectional flow asymmetry already forces strict separation above
the Landauer floor through the theorem-level residual branch proved in
`WolpertResidual.lean`. -/
theorem effective_model_strictly_exceeds_landauer_of_pairwise_flow_asymmetry
    (W : DecomposedProcessModel) {kB T : ℝ}
    (hFloor : landauerJoulesPerBit kB T ≤ (W.base.joulesPerBit : ℝ))
    {S : Type*} [Fintype S]
    (mc : DiscreteMarkovChain S) (π : StationaryDist mc) (s s' : S)
    (hForward : 0 < edgeFlow mc π s s')
    (hReverse : 0 < edgeFlow mc π s' s)
    (hUnits : W.residualDissipationPerBit = residualNatLowerBound mc π s s' hForward hReverse)
    (hAsym : edgeFlow mc π s s' ≠ edgeFlow mc π s' s) :
    landauerJoulesPerBit kB T < ((W.effectiveModel).joulesPerBit : ℝ) := by
  have hResidual : StoppingTimeResidualHypothesis W :=
    stopping_time_residual_of_pairwise_flow_asymmetry W mc π s s' hForward hReverse hUnits hAsym
  exact effective_model_strictly_exceeds_landauer_of_stopping_time_residual W hFloor hResidual

/-- The finite discrete stopping-time residual branch can be discharged by an
exhaustive local edge split on computational states. If the reverse edge flow
is positive, the theorem-level pairwise KL branch applies. If the reverse edge
flow vanishes, the process performs an irreversible one-way state transition,
and the existing Landauer-scaled transition-cost theorem yields a positive
residual lower-bound term. -/
theorem stopping_time_residual_of_discrete_edge_split
    (W : DecomposedProcessModel)
    {kT_ln2 : ℝ} (hkT : 0 < kT_ln2)
    [Fintype ComputationalState]
    {mc : DiscreteMarkovChain ComputationalState}
    (π : StationaryDist mc) (s s' : ComputationalState)
    (hForward : 0 < edgeFlow mc π s s')
    (hUnits :
      W.residualDissipationPerBit =
        discreteResidualNatLowerBound kT_ln2 π s s' hForward)
    (hAsym : edgeFlow mc π s s' ≠ edgeFlow mc π s' s) :
    StoppingTimeResidualHypothesis W := by
  refine ⟨?_⟩
  rw [hUnits]
  exact discreteResidualNatLowerBound_pos_of_asymmetry_or_oneway hkT π s s' hForward hAsym

/-- The finite discrete stopping-time residual branch now has a single
theorem-level constructor: any asymmetric computational-state edge with
positive forward flow yields strict separation above the Landauer floor, by the
exhaustive reverse-edge split packaged in
`stopping_time_residual_of_discrete_edge_split`. -/
theorem effective_model_strictly_exceeds_landauer_of_discrete_edge_split
    (W : DecomposedProcessModel) {kB T kT_ln2 : ℝ}
    (hFloor : landauerJoulesPerBit kB T ≤ (W.base.joulesPerBit : ℝ))
    (hkT : 0 < kT_ln2)
    [Fintype ComputationalState]
    {mc : DiscreteMarkovChain ComputationalState}
    (π : StationaryDist mc) (s s' : ComputationalState)
    (hForward : 0 < edgeFlow mc π s s')
    (hUnits :
      W.residualDissipationPerBit =
        discreteResidualNatLowerBound kT_ln2 π s s' hForward)
    (hAsym : edgeFlow mc π s s' ≠ edgeFlow mc π s' s) :
    landauerJoulesPerBit kB T < ((W.effectiveModel).joulesPerBit : ℝ) := by
  have hResidual : StoppingTimeResidualHypothesis W :=
    stopping_time_residual_of_discrete_edge_split W hkT π s s' hForward hUnits hAsym
  exact effective_model_strictly_exceeds_landauer_of_stopping_time_residual W hFloor hResidual

/-- The theorem-level finite discrete residual branch can be packaged as a
single witness statement: if the process exposes at least one computational-
state edge with positive forward flow and decision-relevant asymmetry, then the
residual term is strictly positive. This is the cleanest process-level derived
subclass currently supported by the artifact. -/
theorem stopping_time_residual_of_finite_discrete_witness
    (W : DecomposedProcessModel)
    {kT_ln2 : ℝ} (hkT : 0 < kT_ln2)
    [Fintype ComputationalState]
    {mc : DiscreteMarkovChain ComputationalState}
    (π : StationaryDist mc)
    (h : FiniteDiscreteResidualWitness π)
    (hUnits :
      W.residualDissipationPerBit =
        discreteResidualNatLowerBound kT_ln2 π h.s h.s' h.hForward) :
    StoppingTimeResidualHypothesis W := by
  exact stopping_time_residual_of_discrete_edge_split
    W hkT π h.s h.s' h.hForward hUnits h.hAsym

/-- The process-level finite discrete residual witness is already sufficient to
force strict separation above the Landauer floor. -/
theorem effective_model_strictly_exceeds_landauer_of_finite_discrete_witness
    (W : DecomposedProcessModel) {kB T kT_ln2 : ℝ}
    (hFloor : landauerJoulesPerBit kB T ≤ (W.base.joulesPerBit : ℝ))
    (hkT : 0 < kT_ln2)
    [Fintype ComputationalState]
    {mc : DiscreteMarkovChain ComputationalState}
    (π : StationaryDist mc)
    (h : FiniteDiscreteResidualWitness π)
    (hUnits :
      W.residualDissipationPerBit =
        discreteResidualNatLowerBound kT_ln2 π h.s h.s' h.hForward) :
    landauerJoulesPerBit kB T < ((W.effectiveModel).joulesPerBit : ℝ) := by
  have hResidual : StoppingTimeResidualHypothesis W :=
    stopping_time_residual_of_finite_discrete_witness W hkT π h hUnits
  exact effective_model_strictly_exceeds_landauer_of_stopping_time_residual W hFloor hResidual

/-- Either the derived mismatch branch or the cited residual branch is already
sufficient to force strict separation above the Landauer floor. -/
theorem effective_model_strictly_exceeds_landauer_of_either_cited_component
    (W : DecomposedProcessModel) {kB T : ℝ}
    (hFloor : landauerJoulesPerBit kB T ≤ (W.base.joulesPerBit : ℝ))
    (h :
      PeriodicModularMismatchHypothesis W ∨
      StoppingTimeResidualHypothesis W) :
    landauerJoulesPerBit kB T < ((W.effectiveModel).joulesPerBit : ℝ) := by
  rcases h with hMismatch | hResidual
  · exact effective_model_strictly_exceeds_landauer_of_periodic_modular_mismatch W hFloor hMismatch
  · exact effective_model_strictly_exceeds_landauer_of_stopping_time_residual W hFloor hResidual

/-- The coarse monotonicity theorem remains valid for the decomposed model. -/
theorem energy_lower_bound_mono_under_decomposition
    (W : DecomposedProcessModel) (bitOpsLB : ℕ) :
    energyLowerBound W.base bitOpsLB ≤
      energyLowerBound (W.effectiveModel) bitOpsLB := by
  exact energy_lower_bound_mono_under_overhead W.asConstrainedProcessModel bitOpsLB

/-- If a declared structural resource is lower-bounded by the mismatch term,
then the effective per-bit lower bound also dominates the Landauer floor plus
that structural resource. This is the theorem-level interface to the
thermodynamic circuit-scaling discussion. -/
theorem landauer_floor_plus_structural_resource_lower_bound
    (W : DecomposedProcessModel) (structuralResource : ℕ) {kB T : ℝ}
    (hFloor : landauerJoulesPerBit kB T ≤ (W.base.joulesPerBit : ℝ))
    (hScale : CircuitStructuralScalingHypothesis W structuralResource) :
    landauerJoulesPerBit kB T + (structuralResource : ℝ) ≤
      ((W.effectiveModel).joulesPerBit : ℝ) := by
  have hDecomp := landauer_floor_plus_decomposition_lower_bound W hFloor
  have hScale' : (structuralResource : ℝ) ≤ (W.mismatchCostPerBit : ℝ) := by
    exact_mod_cast hScale.resourceDominated
  have hResidualNonneg : 0 ≤ (W.residualDissipationPerBit : ℝ) := by
    exact_mod_cast Nat.zero_le _
  linarith

/-- If a declared structural resource is absorbed by the mismatch term, then
the induced energy lower bound increases by at least that resource times the
bit-operation lower bound. -/
theorem energy_lower_bound_increases_by_structural_resource
    (W : DecomposedProcessModel) (bitOpsLB structuralResource : ℕ)
    (hScale : CircuitStructuralScalingHypothesis W structuralResource) :
    energyLowerBound W.base bitOpsLB + structuralResource * bitOpsLB ≤
      energyLowerBound (W.effectiveModel) bitOpsLB := by
  rcases hScale with ⟨hScale⟩
  have hMismatchMul :
      structuralResource * bitOpsLB ≤ W.mismatchCostPerBit * bitOpsLB :=
    Nat.mul_le_mul_right _ hScale
  have hMismatchLeTotal : W.mismatchCostPerBit ≤ W.totalOverheadPerBit := by
    simp [DecomposedProcessModel.totalOverheadPerBit]
  have hTotalMul :
      W.mismatchCostPerBit * bitOpsLB ≤ W.totalOverheadPerBit * bitOpsLB :=
    Nat.mul_le_mul_right _ hMismatchLeTotal
  calc
    energyLowerBound W.base bitOpsLB + structuralResource * bitOpsLB
      ≤ energyLowerBound W.base bitOpsLB + W.mismatchCostPerBit * bitOpsLB :=
        Nat.add_le_add_left hMismatchMul _
    _ ≤ energyLowerBound W.base bitOpsLB + W.totalOverheadPerBit * bitOpsLB :=
        Nat.add_le_add_left hTotalMul _
    _ = (W.base.joulesPerBit + W.totalOverheadPerBit) * bitOpsLB := by
        simp [energyLowerBound, Nat.add_mul, Nat.add_assoc]
    _ = (W.effectiveModel).joulesPerBit * bitOpsLB := by
        rfl
    _ = energyLowerBound (W.effectiveModel) bitOpsLB := by
        rfl

/-- The physical grounding bundle remains valid under the fully decomposed
Wolpert-style refinement. -/
theorem physical_grounding_bundle_with_wolpert_decomposition
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n))
    (hI : dp.isSufficient I)
    (hI_pos : 0 < I.card)
    (W : DecomposedProcessModel)
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hFloor : landauerJoulesPerBit kB T ≤ (W.base.joulesPerBit : ℝ)) :
    dp.srank ≤ I.card ∧
    (W.effectiveModel).joulesPerBit * dp.srank ≤ energyLowerBound (W.effectiveModel) I.card ∧
    0 < energyLowerBound (W.effectiveModel) I.card := by
  exact physical_grounding_bundle_with_wolpert_overhead
    dp I hI hI_pos W.asConstrainedProcessModel hkB hT hFloor

end WolpertDecomposition
end Physics
end DecisionQuotient
