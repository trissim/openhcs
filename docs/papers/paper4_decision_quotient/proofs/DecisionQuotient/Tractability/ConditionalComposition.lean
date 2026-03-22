/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/ConditionalComposition.lean

  Conditional composition theorems for predicate-based branching between certified
  scoring families. When a predicate P determines which scoring function applies,
  the combined scorer is certified with bound max(δA, δB).

  Key applications:
  - System-aware electrostatics: κ=0 (pure Coulomb) for non-metals, κ>0 for metals
  - The κ=0 limit of screened Coulomb is already covered by parametric theorems
-/
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.ScreenedCoulombApproximation

namespace DecisionQuotient
namespace Tractability
namespace ConditionalComposition

open CoarseApproximation
open ScreenedCoulombApproximation
open CertifiedPruning
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer
open Classical

universe u v

/-- Conditional utility: selects between two utilities based on a predicate. -/
def conditionalUtility {A : Type u} {S : Type v}
    (P : S → Prop) [DecidablePred P]
    (uA uB : A → S → ℝ) : A → S → ℝ :=
  fun a s => if P s then uA a s else uB a s

/-- Conditional decision problem: branches on a state predicate. -/
def conditionalDecisionProblem {A : Type u} {S : Type v}
    (P : S → Prop) [DecidablePred P]
    (dpA dpB : DecisionProblem A S) : DecisionProblem A S where
  utility := conditionalUtility P dpA.utility dpB.utility

/-- Core conditional composition theorem: if two approximations hold on
    complementary subsets of the state space, the combined approximation
    holds with the max of the two error bounds. -/
theorem conditional_uniformApprox {A : Type u} {S : Type v}
    (P : S → Prop) [DecidablePred P]
    (exactA coarseA exactB coarseB : DecisionProblem A S)
    (δA δB : ℝ)
    (hA : ∀ a s, P s → |exactA.utility a s - coarseA.utility a s| ≤ δA)
    (hB : ∀ a s, ¬P s → |exactB.utility a s - coarseB.utility a s| ≤ δB) :
    UniformUtilityApprox
      (conditionalDecisionProblem P exactA exactB)
      (conditionalDecisionProblem P coarseA coarseB)
      (max δA δB) := by
  intro a s
  simp only [conditionalDecisionProblem, conditionalUtility]
  by_cases hP : P s
  · simp only [hP, ↓reduceIte]
    exact le_trans (hA a s hP) (le_max_left δA δB)
  · simp only [hP, ↓reduceIte]
    exact le_trans (hB a s hP) (le_max_right δA δB)

/-- Simplified conditional composition when both branches have the same error. -/
theorem conditional_uniformApprox_same_delta {A : Type u} {S : Type v}
    (P : S → Prop) [DecidablePred P]
    (exactA coarseA exactB coarseB : DecisionProblem A S)
    (δ : ℝ)
    (hA : ∀ a s, P s → |exactA.utility a s - coarseA.utility a s| ≤ δ)
    (hB : ∀ a s, ¬P s → |exactB.utility a s - coarseB.utility a s| ≤ δ) :
    UniformUtilityApprox
      (conditionalDecisionProblem P exactA exactB)
      (conditionalDecisionProblem P coarseA coarseB)
      δ := by
  intro a s
  simp only [conditionalDecisionProblem, conditionalUtility]
  by_cases hP : P s
  · simp only [hP, ↓reduceIte]
    exact hA a s hP
  · simp only [hP, ↓reduceIte]
    exact hB a s hP

/-- Certified top-1 survivor set for conditional composition. -/
noncomputable def conditional_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [DecidableEq A] [Nonempty A]
    (P : S → Prop) [DecidablePred P]
    (exactA coarseA exactB coarseB : DecisionProblem A S)
    (δA δB : ℝ)
    (hA : ∀ a s, P s → |exactA.utility a s - coarseA.utility a s| ≤ δA)
    (hB : ∀ a s, ¬P s → |exactB.utility a s - coarseB.utility a s| ≤ δB)
    (hDeltaA : 0 ≤ δA) (hDeltaB : 0 ≤ δB) (s : S) :
    CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => (conditionalDecisionProblem P exactA exactB).utility a s)
    (fun a => (conditionalDecisionProblem P coarseA coarseB).utility a s)
    (max δA δB)
    (fun a => conditional_uniformApprox P exactA coarseA exactB coarseB δA δB hA hB a s)
    (le_max_of_le_left hDeltaA)

theorem conditional_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [DecidableEq A] [Nonempty A]
    (P : S → Prop) [DecidablePred P]
    (exactA coarseA exactB coarseB : DecisionProblem A S)
    (δA δB : ℝ)
    (hA : ∀ a s, P s → |exactA.utility a s - coarseA.utility a s| ≤ δA)
    (hB : ∀ a s, ¬P s → |exactB.utility a s - coarseB.utility a s| ≤ δB)
    (hDeltaA : 0 ≤ δA) (hDeltaB : 0 ≤ δB) (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => (conditionalDecisionProblem P exactA exactB).utility a s)
      (fun a => (conditionalDecisionProblem P coarseA coarseB).utility a s)
      (max δA δB)
      (fun a => conditional_uniformApprox P exactA coarseA exactB coarseB δA δB hA hB a s)
      (le_max_of_le_left hDeltaA)).exactTopK
      ⊆ (conditional_certified_top1 P exactA coarseA exactB coarseB δA δB hA hB hDeltaA hDeltaB s).survivors := by
  simpa [conditional_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => (conditionalDecisionProblem P exactA exactB).utility a s)
      (fun a => (conditionalDecisionProblem P coarseA coarseB).utility a s)
      (max δA δB)
      (fun a => conditional_uniformApprox P exactA coarseA exactB coarseB δA δB hA hB a s)
      (le_max_of_le_left hDeltaA)

/-! ### κ=0 Limit Theorem for Screened Coulomb

The screened Coulomb potential E = q_i q_j * exp(-κr) / r reduces to pure Coulomb
when κ = 0, since exp(0) = 1. The existing `exact_vs_cutoff_screened_coulomb_uniformApprox`
theorem is parametric in κ ≥ 0 and thus covers this case automatically.

This section provides explicit documentation that κ=0 is formally justified.
-/

/-- The screened Coulomb score at κ=0 equals the Coulomb score. -/
theorem screened_coulomb_at_kappa_zero (q_i q_j r : ℝ) :
    exactScreenedCoulombScore q_i q_j 0 r = CoulombApproximation.exactCoulombScore q_i q_j r := by
  unfold exactScreenedCoulombScore CoulombApproximation.exactCoulombScore Ewald.coulombPotential
  simp only [zero_mul, neg_zero, Real.exp_zero, mul_one]

/-! ### Tail Bound Theorems for Screened Coulomb

The screened Coulomb potential E = q_i q_j * exp(-κr) / r has a tail that decays
exponentially for κ > 0. This allows us to derive the minimum cutoff radius R
needed to achieve a given error tolerance ε.

Key formulas (derived from physics):
- Single pair cutoff error at distance r ≥ R: |q_i q_j| * exp(-κR) / R
- Total pairwise tail error: sum over all pairs outside cutoff

Given target error ε and maximum charge product Q_max = max(|q_i q_j|):
- For κ > 0: R ≥ (1/κ) * ln(Q_max * N_pairs / (ε * R))
- For κ = 0: R ≥ Q_max * N_pairs / ε (simpler power-law bound)
-/

/-- Screened Coulomb single-pair cutoff error bound.
    When r ≥ R, the error from ignoring this pair is bounded by |q_i q_j| exp(-κR) / R. -/
theorem screened_coulomb_single_pair_cutoff_error
    (q_i q_j κ R r : ℝ)
    (hR_pos : 0 < R) (hκ_nonneg : 0 ≤ κ) (hr_ge_R : R ≤ r) (hr_pos : 0 < r) :
    |exactScreenedCoulombScore q_i q_j κ r| ≤ |q_i * q_j| * Real.exp (-κ * R) / R := by
  unfold exactScreenedCoulombScore Ewald.coulombPotential
  have hr_ne : r ≠ 0 := ne_of_gt hr_pos
  have hR_ne : R ≠ 0 := ne_of_gt hR_pos
  -- |q_i * q_j / r * exp(-κr)| = |q_i * q_j| / r * exp(-κr)
  rw [abs_mul]
  have h_exp_nonneg : 0 ≤ Real.exp (-κ * r) := Real.exp_pos _ |>.le
  rw [abs_of_nonneg h_exp_nonneg]
  rw [abs_div, abs_of_pos hr_pos]
  -- Now we have |q_i * q_j| / r * exp(-κr) ≤ |q_i * q_j| * exp(-κR) / R
  have h_r_le : 1 / r ≤ 1 / R := by
    exact one_div_le_one_div_of_le hR_pos hr_ge_R
  have h_exp_le : Real.exp (-κ * r) ≤ Real.exp (-κ * R) := by
    apply Real.exp_le_exp_of_le
    exact mul_le_mul_of_nonpos_left hr_ge_R (neg_nonpos.mpr hκ_nonneg)
  have h_abs_nonneg : 0 ≤ |q_i * q_j| := abs_nonneg _
  calc
    |q_i * q_j| / r * Real.exp (-κ * r)
      = |q_i * q_j| * (1 / r) * Real.exp (-κ * r) := by ring
    _ ≤ |q_i * q_j| * (1 / R) * Real.exp (-κ * R) := by
        apply mul_le_mul
        · apply mul_le_mul (le_refl _) h_r_le (by positivity) h_abs_nonneg
        · exact h_exp_le
        · exact Real.exp_pos _ |>.le
        · exact mul_nonneg h_abs_nonneg (by positivity)
    _ = |q_i * q_j| * Real.exp (-κ * R) / R := by ring

/-- For κ > 0, the screened Coulomb cutoff error decays exponentially with R.
    This gives: error ≤ |q_i q_j| * exp(-κR) / R -/
theorem screened_coulomb_exponential_decay
    (q_i q_j κ R : ℝ)
    (hR_pos : 0 < R) (hκ_pos : 0 < κ) :
    ∀ r, R ≤ r → 0 < r →
      |exactScreenedCoulombScore q_i q_j κ r| ≤ |q_i * q_j| * Real.exp (-κ * R) / R := by
  intro r hr_ge hr_pos
  exact screened_coulomb_single_pair_cutoff_error q_i q_j κ R r hR_pos (le_of_lt hκ_pos) hr_ge hr_pos

/-- Exponential bound: Q * exp(-κR) ≤ ε when R ≥ ln(Q/ε) / κ.
    This is the key formula for deriving cutoffs: given target error ε and
    max charge product Q, the minimum cutoff R satisfies R ≥ ln(Q/ε) / κ. -/
theorem screened_coulomb_exp_bound
    (Q ε κ R : ℝ)
    (hQ_pos : 0 < Q) (hε_pos : 0 < ε) (hκ_pos : 0 < κ)
    (hR_bound : Real.log (Q / ε) / κ ≤ R) :
    Q * Real.exp (-κ * R) ≤ ε := by
  have h_arg_pos : 0 < Q / ε := by positivity
  have h_log : Real.log (Q / ε) ≤ κ * R := by
    calc Real.log (Q / ε) = Real.log (Q / ε) / κ * κ := by field_simp
         _ ≤ R * κ := by exact mul_le_mul_of_nonneg_right hR_bound (le_of_lt hκ_pos)
         _ = κ * R := by ring
  have h_exp_log : Real.exp (Real.log (Q / ε)) = Q / ε :=
    Real.exp_log h_arg_pos
  have h_exp_bound : Real.exp (-κ * R) ≤ ε / Q := by
    calc Real.exp (-κ * R) = Real.exp (-(κ * R)) := by ring_nf
         _ ≤ Real.exp (-(Real.log (Q / ε))) := by
             apply Real.exp_le_exp_of_le
             linarith
         _ = (Real.exp (Real.log (Q / ε)))⁻¹ := Real.exp_neg _
         _ = (Q / ε)⁻¹ := by rw [h_exp_log]
         _ = ε / Q := by field_simp
  calc Q * Real.exp (-κ * R)
       ≤ Q * (ε / Q) := mul_le_mul_of_nonneg_left h_exp_bound (le_of_lt hQ_pos)
       _ = ε := by field_simp

/-- Derived cutoff formula: For κ > 0, the minimum cutoff to achieve error ε
    from a charge product Q is R_min = ln(Q/ε) / κ.

    This is the "inverse" of screened_coulomb_exp_bound: given ε, Q, κ,
    we derive R such that Q * exp(-κR) ≤ ε.

    The Python implementation computes: cutoff = max(1.0, log(Q_max / epsilon) / kappa) -/
noncomputable def screenedCoulombMinCutoff (Q ε κ : ℝ) : ℝ :=
  Real.log (Q / ε) / κ

theorem screenedCoulombMinCutoff_sufficient
    (Q ε κ : ℝ) (hQ_pos : 0 < Q) (hε_pos : 0 < ε) (hκ_pos : 0 < κ) :
    Q * Real.exp (-κ * screenedCoulombMinCutoff Q ε κ) ≤ ε := by
  exact screened_coulomb_exp_bound Q ε κ (screenedCoulombMinCutoff Q ε κ)
    hQ_pos hε_pos hκ_pos (le_refl _)

/-- OPTIMALITY: The derived cutoff is the MINIMUM R achieving error ≤ ε.
    Any R < ln(Q/ε)/κ will have error > ε. -/
theorem screenedCoulombMinCutoff_optimal
    (Q ε κ R : ℝ) (hQ_pos : 0 < Q) (hε_pos : 0 < ε) (hκ_pos : 0 < κ)
    (hR_lt : R < screenedCoulombMinCutoff Q ε κ) :
    ε < Q * Real.exp (-κ * R) := by
  unfold screenedCoulombMinCutoff at hR_lt
  have h_arg_pos : 0 < Q / ε := by positivity
  have h_log_bound : κ * R < Real.log (Q / ε) := by
    have h1 : R < Real.log (Q / ε) / κ := hR_lt
    calc κ * R < κ * (Real.log (Q / ε) / κ) := by
             apply mul_lt_mul_of_pos_left h1 hκ_pos
         _ = Real.log (Q / ε) := by field_simp
  have h_exp_bound : ε / Q < Real.exp (-κ * R) := by
    have h1 : Real.exp (-(Real.log (Q / ε))) < Real.exp (-κ * R) := by
      apply Real.exp_strictMono
      linarith
    have h2 : Real.exp (-(Real.log (Q / ε))) = (Q / ε)⁻¹ := by
      rw [Real.exp_neg, Real.exp_log h_arg_pos]
    have h3 : (Q / ε)⁻¹ = ε / Q := by field_simp
    rw [h2, h3] at h1
    exact h1
  calc ε = Q * (ε / Q) := by field_simp
       _ < Q * Real.exp (-κ * R) := by
           apply mul_lt_mul_of_pos_left h_exp_bound hQ_pos

/-- At exactly R = ln(Q/ε)/κ, the error equals ε (tight bound). -/
theorem screenedCoulombMinCutoff_tight
    (Q ε κ : ℝ) (hQ_pos : 0 < Q) (hε_pos : 0 < ε) (hκ_pos : 0 < κ) :
    Q * Real.exp (-κ * screenedCoulombMinCutoff Q ε κ) = ε := by
  unfold screenedCoulombMinCutoff
  have h_arg_pos : 0 < Q / ε := by positivity
  have h_exp : Real.exp (-κ * (Real.log (Q / ε) / κ)) = ε / Q := by
    have h1 : -κ * (Real.log (Q / ε) / κ) = -Real.log (Q / ε) := by field_simp
    rw [h1, Real.exp_neg, Real.exp_log h_arg_pos]
    field_simp
  calc Q * Real.exp (-κ * (Real.log (Q / ε) / κ))
       = Q * (ε / Q) := by rw [h_exp]
     _ = ε := by field_simp

/-! ### Debye-Hückel Screening

In aqueous solution at ionic strength I (in mol/L), electrostatics are screened
with Debye length λ_D = sqrt(ε₀ε_r k_B T / (2 N_A e² I)).

At physiological conditions (37°C, I ≈ 0.15 M):
  λ_D ≈ 7.8 Å  →  κ = 1/λ_D ≈ 0.128 Å⁻¹

This means "pure Coulomb" (κ=0) is physically incorrect for solvated biomolecules.
The correct model uses κ ≈ 0.1 Å⁻¹ (weak screening).

We prove that with weak screening, practical cutoffs achieve reasonable error bounds.
-/

/-- Debye screening parameter at physiological ionic strength (0.15 M, 37°C).
    κ_physiological ≈ 0.128 Å⁻¹ (Debye length ≈ 7.8 Å) -/
noncomputable def κ_physiological : ℝ := 0.128

/-- For a system with max charge product Q, target error ε, using physiological
    screening κ ≈ 0.128 Å⁻¹, the required cutoff is:
    R = ln(Q/ε) / 0.128

    Example: Q = 1000 (large system), ε = 0.5 kcal/mol
    R = ln(2000) / 0.128 ≈ 59 Å

    Example: Q = 100 (medium system), ε = 0.5 kcal/mol
    R = ln(200) / 0.128 ≈ 41 Å

    Example: Q = 10 (small system), ε = 0.5 kcal/mol
    R = ln(20) / 0.128 ≈ 23 Å -/
noncomputable def physiologicalCutoff (Q ε : ℝ) : ℝ :=
  screenedCoulombMinCutoff Q ε κ_physiological

theorem physiological_cutoff_bound
    (Q ε : ℝ) (hQ_pos : 0 < Q) (hε_pos : 0 < ε) :
    Q * Real.exp (-κ_physiological * physiologicalCutoff Q ε) ≤ ε := by
  unfold physiologicalCutoff
  have hκ_pos : 0 < κ_physiological := by unfold κ_physiological; positivity
  exact screenedCoulombMinCutoff_sufficient Q ε κ_physiological hQ_pos hε_pos hκ_pos

/-- 12 Å is justified when Q/ε ≤ exp(12 × 0.128) ≈ 4.6.
    This means for systems with Q ≤ 2.3 kcal/mol (ε = 0.5), 12 Å suffices. -/
theorem cutoff_12_sufficient_condition
    (Q ε : ℝ) (hQ_pos : 0 < Q) (hε_pos : 0 < ε)
    (h_ratio : Q / ε ≤ Real.exp (12 * κ_physiological)) :
    Q * Real.exp (-κ_physiological * 12) ≤ ε := by
  have hκ_pos : 0 < κ_physiological := by unfold κ_physiological; positivity
  have h_log_bound : Real.log (Q / ε) / κ_physiological ≤ 12 := by
    have h_arg_pos : 0 < Q / ε := by positivity
    have h1 : Real.log (Q / ε) ≤ Real.log (Real.exp (12 * κ_physiological)) := by
      apply Real.log_le_log h_arg_pos
      exact h_ratio
    rw [Real.log_exp] at h1
    calc Real.log (Q / ε) / κ_physiological
         ≤ (12 * κ_physiological) / κ_physiological := by
             apply div_le_div_of_nonneg_right h1 (le_of_lt hκ_pos)
       _ = 12 := by field_simp
  exact screened_coulomb_exp_bound Q ε κ_physiological 12 hQ_pos hε_pos hκ_pos h_log_bound

/-! ### Metal Coordination Screening (κ ≈ 1.0 Å⁻¹)

For metal coordination in proteins, the appropriate screening is NOT Debye-Hückel
but rather a "shell separation" model where we want second-shell electrostatics
to be suppressed relative to first-shell coordination bonds.

Physical basis:
- First coordination shell: r₁ ≈ 2.2 Å (metal-ligand bond length)
- Second coordination shell: r₂ ≈ 4.5 Å (next nearest atoms)
- Design goal: V(r₂)/V(r₁) ≤ δ (e.g., 5%)

For screened Coulomb V(r) = exp(-κr)/r:
  V(r₂)/V(r₁) = (r₁/r₂) × exp(-κ(r₂ - r₁))

To achieve suppression ratio δ:
  κ ≥ (ln(r₁/r₂) - ln(δ)) / (r₂ - r₁)
  κ ≥ ln(r₁/(r₂×δ)) / (r₂ - r₁)

For r₁=2.2, r₂=4.5, δ=0.05:
  κ ≥ ln(2.2/(4.5×0.05)) / 2.3
  κ ≥ ln(9.78) / 2.3
  κ ≥ 0.99 Å⁻¹

Thus κ = 1.0 Å⁻¹ achieves ~5% second-shell suppression.
-/

/-- Metal coordination shell distances (empirical, from crystallography) -/
noncomputable def r_first_shell : ℝ := 2.2   -- Å, typical metal-ligand bond
noncomputable def r_second_shell : ℝ := 4.5  -- Å, second coordination shell

/-- Screened Coulomb potential ratio between two distances.
    V(r₂)/V(r₁) = (r₁/r₂) × exp(-κ(r₂ - r₁)) -/
noncomputable def screenedCoulombRatio (r₁ r₂ κ : ℝ) : ℝ :=
  (r₁ / r₂) * Real.exp (-κ * (r₂ - r₁))

/-- The ratio formula is correct: V(r₂)/V(r₁) = (r₁/r₂) × exp(-κ(r₂ - r₁)) -/
theorem screened_coulomb_ratio_formula
    (r₁ r₂ κ : ℝ) (hr₁_pos : 0 < r₁) (hr₂_pos : 0 < r₂) :
    let V := fun r => Real.exp (-κ * r) / r
    V r₂ / V r₁ = screenedCoulombRatio r₁ r₂ κ := by
  unfold screenedCoulombRatio
  simp only []
  have h1 : Real.exp (-κ * r₂) / r₂ / (Real.exp (-κ * r₁) / r₁) =
            (Real.exp (-κ * r₂) / Real.exp (-κ * r₁)) * (r₁ / r₂) := by
    have hr₁_ne : r₁ ≠ 0 := ne_of_gt hr₁_pos
    have hr₂_ne : r₂ ≠ 0 := ne_of_gt hr₂_pos
    have he₁_ne : Real.exp (-κ * r₁) ≠ 0 := Real.exp_ne_zero _
    have he₂_ne : Real.exp (-κ * r₂) ≠ 0 := Real.exp_ne_zero _
    field_simp
  rw [h1]
  have h2 : Real.exp (-κ * r₂) / Real.exp (-κ * r₁) = Real.exp (-κ * r₂ - (-κ * r₁)) := by
    rw [← Real.exp_sub]
  rw [h2]
  ring_nf

/-- Minimum κ to achieve a given suppression ratio δ between shells.
    κ_min = ln(r₁/(r₂×δ)) / (r₂ - r₁) -/
noncomputable def minKappaForSuppression (r₁ r₂ δ : ℝ) : ℝ :=
  Real.log (r₁ / (r₂ * δ)) / (r₂ - r₁)

/-- If κ ≥ κ_min, then the shell ratio is ≤ δ. -/
theorem shell_suppression_achieved
    (r₁ r₂ δ κ : ℝ)
    (hr₁_pos : 0 < r₁) (hr₂_pos : 0 < r₂) (hδ_pos : 0 < δ)
    (hr_order : r₁ < r₂)
    (hκ_bound : minKappaForSuppression r₁ r₂ δ ≤ κ) :
    screenedCoulombRatio r₁ r₂ κ ≤ δ := by
  unfold screenedCoulombRatio minKappaForSuppression at *
  have hΔr_pos : 0 < r₂ - r₁ := by linarith
  have h_arg_pos : 0 < r₁ / (r₂ * δ) := by positivity
  -- κ ≥ ln(r₁/(r₂×δ)) / (r₂ - r₁)
  -- κ × (r₂ - r₁) ≥ ln(r₁/(r₂×δ))
  have h1 : Real.log (r₁ / (r₂ * δ)) ≤ κ * (r₂ - r₁) := by
    calc Real.log (r₁ / (r₂ * δ)) = Real.log (r₁ / (r₂ * δ)) / (r₂ - r₁) * (r₂ - r₁) := by field_simp
         _ ≤ κ * (r₂ - r₁) := mul_le_mul_of_nonneg_right hκ_bound (le_of_lt hΔr_pos)
  -- exp(-κ(r₂ - r₁)) ≤ exp(-ln(r₁/(r₂×δ))) = r₂×δ/r₁
  have h2 : Real.exp (-κ * (r₂ - r₁)) ≤ (r₂ * δ) / r₁ := by
    have h2a : Real.exp (-κ * (r₂ - r₁)) ≤ Real.exp (-Real.log (r₁ / (r₂ * δ))) := by
      apply Real.exp_le_exp_of_le
      linarith
    have h2b : Real.exp (-Real.log (r₁ / (r₂ * δ))) = (r₁ / (r₂ * δ))⁻¹ := by
      rw [Real.exp_neg, Real.exp_log h_arg_pos]
    have h2c : (r₁ / (r₂ * δ))⁻¹ = (r₂ * δ) / r₁ := by field_simp
    calc Real.exp (-κ * (r₂ - r₁)) ≤ Real.exp (-Real.log (r₁ / (r₂ * δ))) := h2a
         _ = (r₁ / (r₂ * δ))⁻¹ := h2b
         _ = (r₂ * δ) / r₁ := h2c
  -- (r₁/r₂) × exp(-κ(r₂-r₁)) ≤ (r₁/r₂) × (r₂×δ/r₁) = δ
  calc r₁ / r₂ * Real.exp (-κ * (r₂ - r₁))
       ≤ r₁ / r₂ * ((r₂ * δ) / r₁) := by apply mul_le_mul_of_nonneg_left h2; positivity
     _ = δ := by field_simp

/-- Metal coordination screening parameter.
    Derived from: 5% second-shell suppression (δ=0.05) with r₁=2.2, r₂=4.5 Å.
    κ_metal = ln(2.2/0.225) / 2.3 ≈ 0.99 ≈ 1.0 Å⁻¹ -/
noncomputable def κ_metal : ℝ := 1.0

/-- The design suppression ratio for metal coordination (5% = 0.05) -/
noncomputable def δ_metal_design : ℝ := 0.05

/-- Given an upper bound on exp(-Δr), we can bound the shell ratio.
    This is the key lemma: if exp(-κΔr) ≤ b, then ratio ≤ (r₁/r₂) × b -/
theorem shell_ratio_from_exp_bound
    (r₁ r₂ κ b : ℝ) (hr₁_pos : 0 < r₁) (hr₂_pos : 0 < r₂)
    (hb_pos : 0 ≤ b) (h_exp_bound : Real.exp (-κ * (r₂ - r₁)) ≤ b) :
    screenedCoulombRatio r₁ r₂ κ ≤ (r₁ / r₂) * b := by
  unfold screenedCoulombRatio
  apply mul_le_mul_of_nonneg_left h_exp_bound
  positivity

/-- The exp function is strictly decreasing for negative arguments.
    So exp(-κΔr) decreases as κ increases (for fixed Δr > 0). -/
theorem exp_decreases_with_kappa
    (κ₁ κ₂ Δr : ℝ) (hΔr_pos : 0 < Δr) (hκ_le : κ₁ ≤ κ₂) :
    Real.exp (-κ₂ * Δr) ≤ Real.exp (-κ₁ * Δr) := by
  apply Real.exp_le_exp_of_le
  have h : -κ₂ * Δr ≤ -κ₁ * Δr := by nlinarith
  exact h

/-- Larger κ gives smaller (better) suppression ratio. -/
theorem larger_kappa_better_suppression
    (r₁ r₂ κ₁ κ₂ : ℝ) (hr₁_pos : 0 < r₁) (hr₂_pos : 0 < r₂)
    (hr_order : r₁ < r₂) (hκ_le : κ₁ ≤ κ₂) :
    screenedCoulombRatio r₁ r₂ κ₂ ≤ screenedCoulombRatio r₁ r₂ κ₁ := by
  unfold screenedCoulombRatio
  apply mul_le_mul_of_nonneg_left
  · apply exp_decreases_with_kappa κ₁ κ₂ (r₂ - r₁) (by linarith) hκ_le
  · positivity

/-- If κ achieves suppression δ, any larger κ also achieves it. -/
theorem suppression_monotone_in_kappa
    (r₁ r₂ δ κ₁ κ₂ : ℝ) (hr₁_pos : 0 < r₁) (hr₂_pos : 0 < r₂)
    (hr_order : r₁ < r₂) (hκ_le : κ₁ ≤ κ₂)
    (h_achieved : screenedCoulombRatio r₁ r₂ κ₁ ≤ δ) :
    screenedCoulombRatio r₁ r₂ κ₂ ≤ δ := by
  calc screenedCoulombRatio r₁ r₂ κ₂
       ≤ screenedCoulombRatio r₁ r₂ κ₁ := larger_kappa_better_suppression r₁ r₂ κ₁ κ₂ hr₁_pos hr₂_pos hr_order hκ_le
     _ ≤ δ := h_achieved

/-- The shell ratio at κ=0 equals r₁/r₂ (no suppression). -/
theorem shell_ratio_at_kappa_zero (r₁ r₂ : ℝ) (hr₂_pos : 0 < r₂) :
    screenedCoulombRatio r₁ r₂ 0 = r₁ / r₂ := by
  unfold screenedCoulombRatio
  simp [Real.exp_zero]

/-- As κ → ∞, the shell ratio → 0 (perfect suppression).
    For any ε > 0, there exists κ₀ such that for all κ ≥ κ₀, the ratio ≤ ε. -/
theorem shell_ratio_limit_zero (r₁ r₂ : ℝ) (hr₁_pos : 0 < r₁) (hr₂_pos : 0 < r₂) (hr_order : r₁ < r₂) :
    ∀ ε > 0, ∃ κ₀ > 0, ∀ κ ≥ κ₀, screenedCoulombRatio r₁ r₂ κ ≤ ε := by
  intro ε hε_pos
  -- Use minKappaForSuppression to get the threshold κ₀
  have hΔr_pos : 0 < r₂ - r₁ := by linarith
  have hδ_pos : 0 < ε * r₂ / r₁ := by positivity
  -- We need (r₁/r₂) × exp(-κΔr) ≤ ε, i.e., ratio ≤ ε
  -- By shell_suppression_achieved, this holds when κ ≥ minKappaForSuppression r₁ r₂ ε
  let κ₀ := max 1 (minKappaForSuppression r₁ r₂ ε + 1)
  use κ₀
  constructor
  · -- κ₀ > 0: max 1 _ ≥ 1 > 0
    have h : (1 : ℝ) ≤ κ₀ := le_max_left 1 _
    linarith
  · intro κ hκ_ge
    -- κ ≥ κ₀ ≥ minKappaForSuppression r₁ r₂ ε + 1 > minKappaForSuppression r₁ r₂ ε
    have hκ_min : minKappaForSuppression r₁ r₂ ε ≤ κ := by
      have h1 : minKappaForSuppression r₁ r₂ ε + 1 ≤ κ₀ := le_max_right _ _
      have h2 : minKappaForSuppression r₁ r₂ ε < minKappaForSuppression r₁ r₂ ε + 1 := by linarith
      linarith
    exact shell_suppression_achieved r₁ r₂ ε κ hr₁_pos hr₂_pos hε_pos hr_order hκ_min

/-! ### Pure Coulomb (κ=0) Tail Bound

For κ=0 (pure Coulomb), the potential is E = q_i q_j / r with no exponential decay.
For finite pairwise interactions, the tail error from cutting off at R is:
  - Single pair at distance r > R: |q_i q_j| / r < |q_i q_j| / R
  - N pairs: tail error ≤ N × Q_max / R

This gives the cutoff formula: R ≥ N × Q_max / ε
-/

/-- Pure Coulomb single-pair contribution bound.
    For r ≥ R > 0, |q_i q_j / r| ≤ |q_i q_j| / R. -/
theorem coulomb_single_pair_bound
    (q_i q_j R r : ℝ)
    (hR_pos : 0 < R) (hr_ge_R : R ≤ r) (hr_pos : 0 < r) :
    |q_i * q_j / r| ≤ |q_i * q_j| / R := by
  have hR_ne : R ≠ 0 := ne_of_gt hR_pos
  have hr_ne : r ≠ 0 := ne_of_gt hr_pos
  rw [abs_div, abs_of_pos hr_pos]
  apply div_le_div_of_nonneg_left (abs_nonneg _) hR_pos hr_ge_R

/-- Pure Coulomb tail bound: N × Q / R ≤ ε when R ≥ N × Q / ε.
    This is the κ=0 analog of screened_coulomb_exp_bound. -/
theorem coulomb_tail_bound
    (N Q ε R : ℝ)
    (hN_pos : 0 < N) (hQ_pos : 0 < Q) (hε_pos : 0 < ε)
    (hR_bound : N * Q / ε ≤ R) :
    N * Q / R ≤ ε := by
  have hR_pos : 0 < R := by
    have h : 0 < N * Q / ε := by positivity
    linarith
  have hNQ_pos : 0 < N * Q := by positivity
  have hNQ_div_ε_pos : 0 < N * Q / ε := by positivity
  -- N * Q / R ≤ N * Q / (N * Q / ε) = ε
  -- div_le_div_of_nonneg_left : 0 ≤ a → 0 < c → c ≤ b → a / b ≤ a / c
  calc N * Q / R
       ≤ N * Q / (N * Q / ε) := by
           apply div_le_div_of_nonneg_left (le_of_lt hNQ_pos) hNQ_div_ε_pos hR_bound
       _ = ε := by field_simp

/-- Minimum cutoff for pure Coulomb to achieve target error ε.
    R_min = N × Q / ε where N is number of pairs and Q is max charge product. -/
noncomputable def coulombMinCutoff (N Q ε : ℝ) : ℝ :=
  N * Q / ε

theorem coulombMinCutoff_sufficient
    (N Q ε : ℝ) (hN_pos : 0 < N) (hQ_pos : 0 < Q) (hε_pos : 0 < ε) :
    N * Q / coulombMinCutoff N Q ε ≤ ε := by
  exact coulomb_tail_bound N Q ε (coulombMinCutoff N Q ε)
    hN_pos hQ_pos hε_pos (le_refl _)

/-- The Coulomb cutoff formula is positive when inputs are positive. -/
theorem coulombMinCutoff_pos
    (N Q ε : ℝ) (hN_pos : 0 < N) (hQ_pos : 0 < Q) (hε_pos : 0 < ε) :
    0 < coulombMinCutoff N Q ε := by
  unfold coulombMinCutoff
  positivity

end ConditionalComposition
end Tractability
end DecisionQuotient

