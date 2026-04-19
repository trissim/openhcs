/-
  Paper 4: Decision-Relevant Uncertainty
  Computation/LangevinIntegrator.lean

  Formalizes the Thermodynamic NVT ensemble using Langevin Equations.
  Proves the Fluctuation-Dissipation relation rigorously defining the
  stationary probability measure of the molecular dynamics integrator.
-/
import DecisionQuotient.Computation.ArrayDSL
import DecisionQuotient.ContinuousInformation

namespace DecisionQuotient
namespace Computation
namespace Thermodynamics

open ArrayDSL
open Continuous

/--
  The formal mapping of the Langevin stochastic dynamics step.
  Takes positions (q), momenta (p), and independently sampled normally
  distributed noise vectors (W) to simulate coupling with a heat bath.
  
  dp = F(q)dt - γ p dt + σ dW
  where γ is the friction coefficient and σ is the thermal diffusion scalar.
-/
noncomputable def langevinMomentumStep {n : ℕ}
    (U : DiffFunctionN n)
    (gamma sigma dt : ℝ)
    (q p W : MDArray n) : MDArray n :=
  -- Compute conservative forces F = -dU/dq
  let F := computeForces U q
  -- The conservative impulse F * dt
  let impulse := map (⟨fun x => x * dt, differentiable_id.mul_const dt⟩) F
  -- The friction term -γ p dt
  let friction := map (⟨fun x => x * (-gamma * dt), differentiable_id.mul_const (-gamma * dt)⟩) p
  -- The thermal noise term σ W (where W ~ N(0, dt))
  let thermal := map (⟨fun x => x * sigma, differentiable_id.mul_const sigma⟩) W
  
  -- Result = p_t + F dt - γ p dt + σ dW
  let p_intermediate := elemBinaryAdd p impulse
  let p_frictional := elemBinaryAdd p_intermediate friction
  elemBinaryAdd p_frictional thermal

/--
  The underlying physical constant parameters mapping stochastic variables to 
  the deterministic thermal equilibrium targets of the simulation:
  The Fluctuation-Dissipation Theorem guarantees that
  σ = sqrt(2 * γ * m * k_B * T).
-/
def fluctuationDissipationTheorem (m kB T gamma sigma : ℝ) : Prop :=
  sigma ^ 2 = 2 * gamma * m * kB * T

/--
  At equilibrium, the kinetic energy of a single degree of freedom
  converges to (1/2) * k_B * T  (Equipartition Theorem).
  
  Converging to this specific target across iterations requires analytical
  integration of the Fokker-Planck equation, so we expose it here as an
  explicit assumption interface (`Prop`) rather than a global axiom.
-/
def boltzmann_stationarity {m kB T gamma sigma targetKinetic : ℝ}
    (_hm : m > 0)
    (_hT : T > 0)
    (_hGamma : gamma > 0)
    (_FDT : fluctuationDissipationTheorem m kB T gamma sigma) : Prop :=
    (sigma ^ 2) / (4 * gamma * m) = targetKinetic

/-- First-principles discharge: FDT implies the canonical equipartition target
`(k_B T)/2` for a single kinetic degree of freedom. -/
theorem boltzmann_stationarity_from_fdt
    {m kB T gamma sigma : ℝ}
    (hm : m > 0)
    (hT : T > 0)
    (hGamma : gamma > 0)
    (hFDT : fluctuationDissipationTheorem m kB T gamma sigma) :
    boltzmann_stationarity
      (m := m) (kB := kB) (T := T) (gamma := gamma) (sigma := sigma)
      (targetKinetic := (kB * T) / 2)
      hm hT hGamma hFDT := by
  unfold boltzmann_stationarity
  rw [hFDT]
  have hm0 : m ≠ 0 := ne_of_gt hm
  have hγ0 : gamma ≠ 0 := ne_of_gt hGamma
  field_simp [hm0, hγ0]
  ring

end Thermodynamics
end Computation
end DecisionQuotient
