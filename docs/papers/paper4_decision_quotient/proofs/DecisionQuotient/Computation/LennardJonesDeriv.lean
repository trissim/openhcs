/-
  Paper 4: Decision-Relevant Uncertainty
  Computation/LennardJonesDeriv.lean
  
  Rigorous formalization of the Lennard-Jones potential derivative.
-/
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Tactic

namespace DecisionQuotient
namespace Computation

open Real

/-- The Lennard-Jones 12-6 potential. -/
noncomputable def lennardJones (ε σ r : ℝ) : ℝ :=
  4 * ε * ((σ * r⁻¹) ^ (12 : ℕ) - (σ * r⁻¹) ^ (6 : ℕ))

/-- 
  Derivative of the Lennard-Jones potential with respect to distance.
  dU/dr = (24ε/r) * [(σ/r)^6 - 2(σ/r)^12]  (Note: physics convention on attractive/repulsive signs)
-/
theorem grad_of_lennardJones (ε σ r : ℝ) (hr : r ≠ 0) :
    HasDerivAt (fun x => lennardJones ε σ x) 
      ((24 * ε * r⁻¹) * ((σ * r⁻¹) ^ (6 : ℕ) - 2 * (σ * r⁻¹) ^ (12 : ℕ))) r := by
  have h_inv : HasDerivAt (fun x : ℝ => x⁻¹) (- (r ^ 2)⁻¹) r :=
    hasDerivAt_inv hr
  
  have h1 : HasDerivAt (fun x : ℝ => σ * x⁻¹) (σ * (- (r ^ 2)⁻¹)) r :=
    HasDerivAt.const_mul σ h_inv

  have h2 : HasDerivAt (fun x : ℝ => (σ * x⁻¹) ^ (12 : ℕ)) 
      (12 * (σ * r⁻¹) ^ (11 : ℕ) * (σ * (- (r ^ 2)⁻¹))) r := by
    have hp := hasDerivAt_pow 12 (σ * r⁻¹)
    exact HasDerivAt.comp r hp h1

  have h3 : HasDerivAt (fun x : ℝ => (σ * x⁻¹) ^ (6 : ℕ)) 
      (6 * (σ * r⁻¹) ^ (5 : ℕ) * (σ * (- (r ^ 2)⁻¹))) r := by
    have hp := hasDerivAt_pow 6 (σ * r⁻¹)
    exact HasDerivAt.comp r hp h1

  have h4 : HasDerivAt (fun x : ℝ => (σ * x⁻¹) ^ (12 : ℕ) - (σ * x⁻¹) ^ (6 : ℕ))
      (12 * (σ * r⁻¹) ^ (11 : ℕ) * (σ * (- (r ^ 2)⁻¹)) - 6 * (σ * r⁻¹) ^ (5 : ℕ) * (σ * (- (r ^ 2)⁻¹))) r :=
    HasDerivAt.sub h2 h3

  have h5 : HasDerivAt (fun x : ℝ => 4 * ε * ((σ * x⁻¹) ^ (12 : ℕ) - (σ * x⁻¹) ^ (6 : ℕ)))
      (4 * ε * (12 * (σ * r⁻¹) ^ (11 : ℕ) * (σ * (- (r ^ 2)⁻¹)) - 6 * (σ * r⁻¹) ^ (5 : ℕ) * (σ * (- (r ^ 2)⁻¹)))) r :=
    HasDerivAt.const_mul (4 * ε) h4

  have h_alg : 
      4 * ε * (12 * (σ * r⁻¹) ^ (11 : ℕ) * (σ * (- (r ^ 2)⁻¹)) - 6 * (σ * r⁻¹) ^ (5 : ℕ) * (σ * (- (r ^ 2)⁻¹))) =
      (24 * ε * r⁻¹) * ((σ * r⁻¹) ^ (6 : ℕ) - 2 * (σ * r⁻¹) ^ (12 : ℕ)) := by
    have h_r2 : (r^2)⁻¹ = r⁻¹ * r⁻¹ := by
      calc (r^2)⁻¹ = (r * r)⁻¹ := by ring_nf
        _ = r⁻¹ * r⁻¹ := mul_inv_rev r r
    rw [h_r2]
    -- Distribute the powers
    have h11 : (σ * r⁻¹) ^ (11 : ℕ) = σ ^ (11 : ℕ) * r⁻¹ ^ (11 : ℕ) := by apply mul_pow
    have h5 : (σ * r⁻¹) ^ (5 : ℕ) = σ ^ (5 : ℕ) * r⁻¹ ^ (5 : ℕ) := by apply mul_pow
    have h12 : (σ * r⁻¹) ^ (12 : ℕ) = σ ^ (12 : ℕ) * r⁻¹ ^ (12 : ℕ) := by apply mul_pow
    have h6 : (σ * r⁻¹) ^ (6 : ℕ) = σ ^ (6 : ℕ) * r⁻¹ ^ (6 : ℕ) := by apply mul_pow
    rw [h11, h5, h12, h6]
    -- Now it's just polynomials in variables ε, σ, and r⁻¹
    ring

  exact HasDerivAt.congr_deriv h5 h_alg
  
end Computation
end DecisionQuotient
