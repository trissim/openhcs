/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/SpectralEnclosureBasin.lean
-/
import Mathlib.Data.Real.Basic
import DecisionQuotient.Tractability.EnergyRMSDConvergence
import DecisionQuotient.Computation.ArrayDSL

namespace DecisionQuotient
namespace Tractability
namespace SpectralEnclosureBasin

open EnergyRMSDConvergence
open Computation.ArrayDSL

def runtime_local_spectral_enclosure_to_basin
  {n : ℕ} {energy : CoordSet n → ℝ} {center : CoordSet n}
  (enc : CertifiedLocalSpectralEnclosure energy center) :
  CertifiedQuadraticBasin energy center :=
  enc.toCertifiedQuadraticBasin

def local_enclosure_dominates_point_probe_failure
  {n : ℕ} {energy : CoordSet n → ℝ} {center : CoordSet n}
  (lmin_point : ℝ)
  (hPointFails : lmin_point ≤ 0)
  (enc : CertifiedLocalSpectralEnclosure energy center) :
  CertifiedQuadraticBasin energy center :=
  enc.toCertifiedQuadraticBasin

def WindowSufficient {n : ℕ} {energy : CoordSet n → ℝ} {center : CoordSet n}
  (enc : CertifiedLocalSpectralEnclosure energy center) (Δ : ℝ) : Prop :=
  ∃ eps : ℝ, 0 ≤ eps ∧ targetEnergyGap enc.lmin n eps ≤ Δ

theorem enclosure_window_sufficient_for_target_gap
  {n : ℕ} {energy : CoordSet n → ℝ} {center : CoordSet n}
  (enc : CertifiedLocalSpectralEnclosure energy center)
  (Δ : ℝ)
  (eps : ℝ) (heps : 0 ≤ eps) (hgap : targetEnergyGap enc.lmin n eps ≤ Δ) :
  WindowSufficient enc Δ := by
  unfold WindowSufficient
  exact ⟨eps, heps, hgap⟩

end SpectralEnclosureBasin
end Tractability
end DecisionQuotient
