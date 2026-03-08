/-
  Paper 4: Decision-Relevant Uncertainty

  Physics/WolpertConstraints.lean

  Explicit interface for the Wolpert-style refinement of Landauer:

  - Landauer gives a universal floor for irreversible bit erasure.
  - Real constrained computational processes can incur additional irreversible
    entropy production above that floor.
  - We do not rederive the full stochastic-thermodynamics machinery here.
    Instead, we represent the relevant empirical refinement as an explicit
    floor-plus-overhead interface and prove that it composes monotonically with
    the existing decision-theoretic lower bounds.

  This is the rigorously scoped bridge for the cited papers:
  PNAS 2024 (Wolpert et al.), Phys. Rev. X 2024 (Manzano et al.),
  arXiv 2026 (Yadav--Yousef--Wolpert), and the stochastic-thermodynamics review
  of Van den Broeck--Esposito 2015.
-/

import DecisionQuotient.ThermodynamicLift
import DecisionQuotient.Physics.BoundedAcquisition

namespace DecisionQuotient
namespace Physics
namespace WolpertConstraints

open ThermodynamicLift
open BoundedAcquisition

/-- A constrained-process thermodynamic model consists of a declared base
lower-bound conversion model together with an explicit nonnegative extra
dissipation term per irreversible bit. The extra term represents the
architecture- or process-induced dissipation (for example periodic driving,
modularity, mismatch cost, or stopping-time effects) that can sit above the
Landauer floor. -/
structure ConstrainedProcessModel where
  /-- Base declared lower-bound conversion model. -/
  base : ThermoModel
  /-- Explicit extra irreversible dissipation term per bit (in the same
  discrete lower-bound units as `joulesPerBit`). -/
  extraDissipationPerBit : ℕ

/-- Effective lower-bound conversion model after adding the explicit extra
dissipation term. -/
def ConstrainedProcessModel.effectiveModel (W : ConstrainedProcessModel) : ThermoModel :=
  { joulesPerBit := W.base.joulesPerBit + W.extraDissipationPerBit
    carbonPerJoule := W.base.carbonPerJoule }

/-- The explicit floor-plus-overhead statement: if the base declared model
dominates the Landauer floor, then the effective constrained-process model
dominates the Landauer floor plus the declared extra dissipation term. -/
theorem landauer_floor_plus_overhead_lower_bound
    (W : ConstrainedProcessModel) {kB T : ℝ}
    (hFloor : landauerJoulesPerBit kB T ≤ (W.base.joulesPerBit : ℝ)) :
    landauerJoulesPerBit kB T + (W.extraDissipationPerBit : ℝ) ≤
      ((W.effectiveModel).joulesPerBit : ℝ) := by
  simpa [ConstrainedProcessModel.effectiveModel]
    using add_le_add_right hFloor (W.extraDissipationPerBit : ℝ)

/-- The constrained-process effective model still dominates the Landauer floor,
even before using any strict-overhead premise. -/
theorem effective_model_dominates_landauer_floor
    (W : ConstrainedProcessModel) {kB T : ℝ}
    (hFloor : landauerJoulesPerBit kB T ≤ (W.base.joulesPerBit : ℝ)) :
    landauerJoulesPerBit kB T ≤ ((W.effectiveModel).joulesPerBit : ℝ) := by
  have hPlus :=
    landauer_floor_plus_overhead_lower_bound W hFloor
  have hNonneg : 0 ≤ (W.extraDissipationPerBit : ℝ) := by exact_mod_cast Nat.zero_le _
  linarith

/-- Strict-overhead specialization: when the declared constrained-process
overhead is strictly positive, the effective per-bit lower bound is strictly
greater than the Landauer floor. This is the exact formal shape of the Wolpert
correction used by the artifact. -/
theorem effective_model_strictly_exceeds_landauer_of_strict_overhead
    (W : ConstrainedProcessModel) {kB T : ℝ}
    (hFloor : landauerJoulesPerBit kB T ≤ (W.base.joulesPerBit : ℝ))
    (hOver : 0 < W.extraDissipationPerBit) :
    landauerJoulesPerBit kB T < ((W.effectiveModel).joulesPerBit : ℝ) := by
  have hPlus :=
    landauer_floor_plus_overhead_lower_bound W hFloor
  have hOver' : 0 < (W.extraDissipationPerBit : ℝ) := by exact_mod_cast hOver
  linarith

/-- Adding an explicit nonnegative extra dissipation term can only increase the
energy lower bound induced by a bit-operation lower bound. -/
theorem energy_lower_bound_mono_under_overhead
    (W : ConstrainedProcessModel) (bitOpsLB : ℕ) :
    energyLowerBound W.base bitOpsLB ≤
      energyLowerBound (W.effectiveModel) bitOpsLB := by
  simp [energyLowerBound, ConstrainedProcessModel.effectiveModel, Nat.add_mul]

/-- The physical grounding bundle remains valid under the Wolpert-style
floor-plus-overhead refinement. We do not need to reprove the grounding chain:
the effective model still dominates the Landauer floor, so the existing
grounding theorem applies directly to the tightened lower-bound model. -/
theorem physical_grounding_bundle_with_wolpert_overhead
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n))
    (hI : dp.isSufficient I)
    (hI_pos : 0 < I.card)
    (W : ConstrainedProcessModel)
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hFloor : landauerJoulesPerBit kB T ≤ (W.base.joulesPerBit : ℝ)) :
    dp.srank ≤ I.card ∧
    (W.effectiveModel).joulesPerBit * dp.srank ≤ energyLowerBound (W.effectiveModel) I.card ∧
    0 < energyLowerBound (W.effectiveModel) I.card := by
  have hEffFloor : landauerJoulesPerBit kB T ≤ ((W.effectiveModel).joulesPerBit : ℝ) :=
    effective_model_dominates_landauer_floor W hFloor
  exact physical_grounding_landauer_floor dp I hI hI_pos (W.effectiveModel) hkB hT hEffFloor

end WolpertConstraints
end Physics
end DecisionQuotient
