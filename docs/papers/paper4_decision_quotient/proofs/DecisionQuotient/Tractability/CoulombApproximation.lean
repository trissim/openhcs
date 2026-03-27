/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/CoulombApproximation.lean

  Finite-domain exact/coarse approximation for Coulomb-style scoring.
-/
import DecisionQuotient.Tractability.EwaldSummation
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.CutoffEpsilon
import DecisionQuotient.Tractability.FormalLocalOptimizer
import DecisionQuotient.Tractability.LatticeSum
import Mathlib.Data.Finset.Max

namespace DecisionQuotient
namespace Tractability
namespace CoulombApproximation

open Ewald
open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer
open LatticeSum
open Classical

universe u v

/-- Exact single-pair Coulomb score. -/
noncomputable def exactCoulombScore (q_i q_j r : ℝ) : ℝ :=
  coulombPotential q_i q_j r

/-- Hard-cutoff Coulomb score. -/
noncomputable def cutoffCoulombScore (q_i q_j rc r : ℝ) : ℝ :=
  if r < rc then coulombPotential q_i q_j r else 0

noncomputable def exactCoulombDecisionProblem {A : Type u} {S : Type v}
    (q_i q_j : ℝ) (distance : A → S → ℝ) : DecisionProblem A S where
  utility := fun a s => exactCoulombScore q_i q_j (distance a s)

noncomputable def cutoffCoulombDecisionProblem {A : Type u} {S : Type v}
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) : DecisionProblem A S where
  utility := fun a s => cutoffCoulombScore q_i q_j rc (distance a s)

noncomputable def coulombCutoffErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) : ℝ :=
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S)).image
      (fun p => |exactCoulombScore q_i q_j (distance p.1 p.2) - cutoffCoulombScore q_i q_j rc (distance p.1 p.2)|)
  diffs.max' <| by
    rcases ‹Nonempty A› with ⟨a⟩
    rcases ‹Nonempty S› with ⟨s⟩
    refine ⟨|exactCoulombScore q_i q_j (distance a s) - cutoffCoulombScore q_i q_j rc (distance a s)|, ?_⟩
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩

theorem coulombCutoffErrorRadius_spec {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ)
    (a : A) (s : S) :
    |exactCoulombScore q_i q_j (distance a s) - cutoffCoulombScore q_i q_j rc (distance a s)| ≤
      coulombCutoffErrorRadius q_i q_j rc distance := by
  classical
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S)).image
      (fun p => |exactCoulombScore q_i q_j (distance p.1 p.2) - cutoffCoulombScore q_i q_j rc (distance p.1 p.2)|)
  have hMem : |exactCoulombScore q_i q_j (distance a s) - cutoffCoulombScore q_i q_j rc (distance a s)| ∈ diffs := by
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩
  unfold coulombCutoffErrorRadius
  exact Finset.le_max' diffs _ hMem

theorem exact_vs_cutoff_coulomb_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) :
    UniformUtilityApprox
      (exactCoulombDecisionProblem q_i q_j distance)
      (cutoffCoulombDecisionProblem q_i q_j rc distance)
      (coulombCutoffErrorRadius q_i q_j rc distance) := by
  intro a s
  simpa [exactCoulombDecisionProblem, cutoffCoulombDecisionProblem] using
    coulombCutoffErrorRadius_spec q_i q_j rc distance a s

theorem coulombCutoffErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) :
    0 ≤ coulombCutoffErrorRadius q_i q_j rc distance := by
  rcases ‹Nonempty A› with ⟨a⟩
  rcases ‹Nonempty S› with ⟨s⟩
  exact le_trans (abs_nonneg _) (coulombCutoffErrorRadius_spec q_i q_j rc distance a s)

/-- Exact-vs-cutoff Coulomb induces a theorem-backed certified top-1 survivor set. -/
noncomputable def exact_vs_cutoff_coulomb_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) (s : S) :
    CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactCoulombDecisionProblem q_i q_j distance |>.utility a s)
    (fun a => cutoffCoulombDecisionProblem q_i q_j rc distance |>.utility a s)
    (coulombCutoffErrorRadius q_i q_j rc distance)
    (fun a => exact_vs_cutoff_coulomb_uniformApprox q_i q_j rc distance a s)
    (coulombCutoffErrorRadius_nonneg q_i q_j rc distance)

/-- Soundness of the exact-vs-cutoff Coulomb certified top-1 survivor set. -/
theorem exact_vs_cutoff_coulomb_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactCoulombDecisionProblem q_i q_j distance |>.utility a s)
      (fun a => cutoffCoulombDecisionProblem q_i q_j rc distance |>.utility a s)
      (coulombCutoffErrorRadius q_i q_j rc distance)
      (fun a => exact_vs_cutoff_coulomb_uniformApprox q_i q_j rc distance a s)
      (coulombCutoffErrorRadius_nonneg q_i q_j rc distance)).exactTopK
      ⊆ (exact_vs_cutoff_coulomb_certified_top1 q_i q_j rc distance s).survivors := by
  simpa [exact_vs_cutoff_coulomb_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactCoulombDecisionProblem q_i q_j distance |>.utility a s)
      (fun a => cutoffCoulombDecisionProblem q_i q_j rc distance |>.utility a s)
      (coulombCutoffErrorRadius q_i q_j rc distance)
      (fun a => exact_vs_cutoff_coulomb_uniformApprox q_i q_j rc distance a s)
      (coulombCutoffErrorRadius_nonneg q_i q_j rc distance)

/-- Exact-vs-cutoff Coulomb yields a runtime-facing optimizer witness. -/
noncomputable def exact_vs_cutoff_coulomb_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) (s : S) :
    CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactCoulombDecisionProblem q_i q_j distance |>.utility a s)
    (fun a => cutoffCoulombDecisionProblem q_i q_j rc distance |>.utility a s)
    (coulombCutoffErrorRadius q_i q_j rc distance)
    (fun a => exact_vs_cutoff_coulomb_uniformApprox q_i q_j rc distance a s)
    (coulombCutoffErrorRadius_nonneg q_i q_j rc distance)

noncomputable def exact_vs_cutoff_coulomb_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) (s : S) :
    OptimizerWitness A :=
  (exact_vs_cutoff_coulomb_coherent_optimizer_witness q_i q_j rc distance s).toOptimizerWitness

/-- Coulomb-family packaging theorem: once a Coulomb tail perturbation bound is
    proved for the chosen utility, the finite-gap theorem yields an explicit
    `SatisfiesBoundedPotential` witness. -/
theorem exactCoulomb_satisfiesBoundedPotential_of_tailBound
    (prob : MolecularSrank.MDBindingProblem)
    [Fintype MolecularSrank.MDAction] [Fintype MolecularSrank.MDState] [Nonempty MolecularSrank.MDState]
    (distance : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    (q_i q_j tail_coefficient : ℝ)
    (w : ∀ s : MolecularSrank.MDState,
      { a : MolecularSrank.MDAction // StrictOpt prob.toDecisionProblem a s })
    (hGapPos : 0 < finiteMinimumGap prob w)
    (hUtility : prob.utility = fun a s => exactCoulombScore q_i q_j (distance a s))
    (hTail : ∀ (atomIdx : Nat) (hAtomInProtein : atomIdx < prob.protein.numAtoms)
      (axis : Fin 3) (s s' : MolecularSrank.MDState) (R : ℝ),
      0 < R →
      (∀ j : Fin (MolecularSrank.numMDCoordinates prob),
        j ≠ MolecularSrank.proteinCoordFin prob atomIdx hAtomInProtein axis →
        MolecularSrank.mdProj prob s j = MolecularSrank.mdProj prob s' j) →
      ¬ MolecularSrank.atomWithinCutoff
        (MolecularSrank.proteinAtom prob atomIdx hAtomInProtein) prob.bindingSite R →
      (∀ a : MolecularSrank.MDAction,
        |exactCoulombScore q_i q_j (distance a s) - exactCoulombScore q_i q_j (distance a s')|
          ≤ tail_coefficient * latticeTailSum 6 R)) :
    SatisfiesBoundedPotential prob tail_coefficient (finiteMinimumGap prob w) := by
  apply satisfiesBoundedPotential_of_tailBound_and_finiteGap prob tail_coefficient w hGapPos
  intro atomIdx hAtomInProtein axis s s' R hR hSame hOutside a
  simpa [hUtility] using hTail atomIdx hAtomInProtein axis s s' R hR hSame hOutside a

/-- Exact Real-Space Ewald Coulomb score. The error function forces exponential decay. -/
noncomputable def exactRealEwaldScore (q_i q_j alpha r : ℝ) : ℝ :=
  coulombPotential q_i q_j r * DecisionQuotient.Tractability.Ewald.erfc (alpha * r)

/-- Absolute-value envelope for the exact real-space Ewald score. -/
theorem abs_exactRealEwaldScore_le_charge_envelope
    (q_i q_j alpha r : ℝ) (hr : 0 < r) (ha : 0 < alpha) :
    |exactRealEwaldScore q_i q_j alpha r| ≤
      |q_i * q_j| * ewaldRealSpaceCore r alpha := by
  have hx : 0 < alpha * r := by positivity
  unfold exactRealEwaldScore coulombPotential ewaldRealSpaceCore
  calc
    |(q_i * q_j) / r * DecisionQuotient.Tractability.Ewald.erfc (alpha * r)|
      = |(q_i * q_j) / r| * |DecisionQuotient.Tractability.Ewald.erfc (alpha * r)| := by rw [abs_mul]
    _ ≤ |(q_i * q_j) / r| * Real.exp (-((alpha * r) ^ 2)) := by
      gcongr
      exact DecisionQuotient.Tractability.Ewald.erfc_abs_le_exp_neg_sq hx
    _ = (|q_i * q_j| / r) * Real.exp (-((alpha * r) ^ 2)) := by
      rw [abs_div, abs_of_pos hr]
    _ = |q_i * q_j| * (Real.exp (-((alpha * r) ^ 2)) / r) := by
      field_simp [hr.ne']

/-- Real-space Ewald is nonnegative when the charge product is nonnegative. -/
theorem exactRealEwaldScore_nonneg_of_nonneg_charge_product
    (q_i q_j alpha r : ℝ) (hQ : 0 ≤ q_i * q_j) (hr : 0 < r) (ha : 0 < alpha) :
    0 ≤ exactRealEwaldScore q_i q_j alpha r := by
  have hx : 0 < alpha * r := by positivity
  unfold exactRealEwaldScore coulombPotential
  have hdiv : 0 ≤ (q_i * q_j) / r := by
    exact div_nonneg hQ hr.le
  have herfc : 0 ≤ Ewald.erfc (alpha * r) := Ewald.erfc_nonneg hx
  exact mul_nonneg hdiv herfc

/-- Real-space Ewald is nonpositive when the charge product is nonpositive. -/
theorem exactRealEwaldScore_nonpos_of_nonpos_charge_product
    (q_i q_j alpha r : ℝ) (hQ : q_i * q_j ≤ 0) (hr : 0 < r) (ha : 0 < alpha) :
    exactRealEwaldScore q_i q_j alpha r ≤ 0 := by
  have hx : 0 < alpha * r := by positivity
  unfold exactRealEwaldScore coulombPotential
  have hdiv : (q_i * q_j) / r ≤ 0 := by
    exact div_nonpos_of_nonpos_of_nonneg hQ hr.le
  have herfc : 0 ≤ Ewald.erfc (alpha * r) := Ewald.erfc_nonneg hx
  exact mul_nonpos_of_nonpos_of_nonneg hdiv herfc

/-- Lower bound corollary from the absolute real-space Ewald envelope. -/
theorem exactRealEwaldScore_ge_neg_charge_envelope
    (q_i q_j alpha r : ℝ) (hr : 0 < r) (ha : 0 < alpha) :
    -(|q_i * q_j| * ewaldRealSpaceCore r alpha) ≤ exactRealEwaldScore q_i q_j alpha r := by
  have hAbs := abs_exactRealEwaldScore_le_charge_envelope q_i q_j alpha r hr ha
  have hLower : -(|q_i * q_j| * ewaldRealSpaceCore r alpha) ≤ -|exactRealEwaldScore q_i q_j alpha r| := by
    nlinarith [hAbs]
  exact le_trans hLower (neg_abs_le _)

/-- For attractive charge products, the exact real-space Ewald score is monotone
    nondecreasing in distance, so the left endpoint of a reachable interval gives a
    certified lower bound for the whole interval. -/
theorem exactRealEwaldScore_lower_bound_at_left_endpoint_of_nonpos_charge_product
    (q_i q_j alpha r₀ r : ℝ)
    (hQ : q_i * q_j ≤ 0)
    (ha : 0 < alpha)
    (hr₀ : 0 < r₀)
    (hr : r₀ ≤ r) :
    exactRealEwaldScore q_i q_j alpha r₀ ≤ exactRealEwaldScore q_i q_j alpha r := by
  have hx0 : 0 ≤ alpha * r₀ := by positivity
  have hxy : alpha * r₀ ≤ alpha * r := by gcongr
  have herfc_mono : Ewald.erfc (alpha * r) ≤ Ewald.erfc (alpha * r₀) :=
    Ewald.erfc_antitone hx0 hxy
  have hr_pos : 0 < r := lt_of_lt_of_le hr₀ hr
  have hr_inv : r⁻¹ ≤ r₀⁻¹ := by
    exact (inv_le_inv₀ hr_pos hr₀).2 hr
  have hdiv_nonpos₀ : (q_i * q_j) / r₀ ≤ 0 := by
    exact div_nonpos_of_nonpos_of_nonneg hQ hr₀.le
  unfold exactRealEwaldScore coulombPotential
  have hstep1 : (q_i * q_j) / r₀ * Ewald.erfc (alpha * r₀) ≤ (q_i * q_j) / r₀ * Ewald.erfc (alpha * r) := by
    exact mul_le_mul_of_nonpos_left herfc_mono hdiv_nonpos₀
  have herfc_nonneg : 0 ≤ Ewald.erfc (alpha * r) := Ewald.erfc_nonneg (by positivity)
  have hdiv_mono : (q_i * q_j) / r₀ ≤ (q_i * q_j) / r := by
    simpa [div_eq_mul_inv] using mul_le_mul_of_nonpos_left hr_inv hQ
  have hstep2 : (q_i * q_j) / r₀ * Ewald.erfc (alpha * r) ≤ (q_i * q_j) / r * Ewald.erfc (alpha * r) := by
    exact mul_le_mul_of_nonneg_right hdiv_mono herfc_nonneg
  exact le_trans hstep1 hstep2

/-- For nonnegative charge products, the exact real-space Ewald score is monotone
    nonincreasing in distance, so the right endpoint of a reachable interval gives a
    certified lower bound for the whole interval. -/
theorem exactRealEwaldScore_lower_bound_at_right_endpoint_of_nonneg_charge_product
    (q_i q_j alpha r r₁ : ℝ)
    (hQ : 0 ≤ q_i * q_j)
    (ha : 0 < alpha)
    (hr : 0 < r)
    (hrr : r ≤ r₁)
    (hr₁ : 0 < r₁) :
    exactRealEwaldScore q_i q_j alpha r₁ ≤ exactRealEwaldScore q_i q_j alpha r := by
  have hx0 : 0 ≤ alpha * r := by positivity
  have hxy : alpha * r ≤ alpha * r₁ := by gcongr
  have herfc_mono : Ewald.erfc (alpha * r₁) ≤ Ewald.erfc (alpha * r) :=
    Ewald.erfc_antitone hx0 hxy
  have hr_inv : r₁⁻¹ ≤ r⁻¹ := by
    exact (inv_le_inv₀ hr₁ hr).2 hrr
  have hdiv_nonneg : 0 ≤ (q_i * q_j) / r₁ := by
    exact div_nonneg hQ hr₁.le
  have hdiv_mono : (q_i * q_j) / r₁ ≤ (q_i * q_j) / r := by
    simpa [div_eq_mul_inv] using mul_le_mul_of_nonneg_left hr_inv hQ
  unfold exactRealEwaldScore coulombPotential
  have hstep1 : (q_i * q_j) / r₁ * Ewald.erfc (alpha * r₁) ≤ (q_i * q_j) / r₁ * Ewald.erfc (alpha * r) := by
    exact mul_le_mul_of_nonneg_left herfc_mono hdiv_nonneg
  have herfc_nonneg : 0 ≤ Ewald.erfc (alpha * r) := Ewald.erfc_nonneg (by positivity)
  have hstep2 : (q_i * q_j) / r₁ * Ewald.erfc (alpha * r) ≤ (q_i * q_j) / r * Ewald.erfc (alpha * r) := by
    exact mul_le_mul_of_nonneg_right hdiv_mono herfc_nonneg
  exact le_trans hstep1 hstep2

/-- Explicit far-field error bound for the real-space Ewald correction. -/
noncomputable def realEwaldFarFieldErrorBound (q_i q_j alpha R : ℝ) : ℝ :=
  |q_i * q_j| * ((2 / alpha ^ 4) / R ^ 3)

theorem realEwaldFarFieldErrorBound_nonneg
    (q_i q_j alpha R : ℝ) (ha : 0 < alpha) (hR : 1 ≤ R) :
    0 ≤ realEwaldFarFieldErrorBound q_i q_j alpha R := by
  unfold realEwaldFarFieldErrorBound
  positivity

/--
  On any fixed state whose sampled distances all lie beyond a far-field radius
  `R`, the real-space Ewald correction is uniformly bounded by an explicit
  alpha-dependent tail term.
-/
theorem exactRealEwaldScore_far_field_bound
    {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (q_i q_j alpha R : ℝ)
    (s : S)
    (ha : 0 < alpha)
    (hR : 1 ≤ R)
    (hFar : ∀ a, R ≤ distance a s) :
    ∀ a, |exactRealEwaldScore q_i q_j alpha (distance a s)| ≤ realEwaldFarFieldErrorBound q_i q_j alpha R := by
  intro a
  have hdist_ge_one : 1 ≤ distance a s := le_trans hR (hFar a)
  have hdist_pos : 0 < distance a s := lt_of_lt_of_le zero_lt_one hdist_ge_one
  have hEnv := abs_exactRealEwaldScore_le_charge_envelope q_i q_j alpha (distance a s) hdist_pos ha
  have hTailPoint : ewaldRealSpaceCore (distance a s) alpha ≤ (2 / alpha ^ 4) / (distance a s) ^ 3 := by
    simpa using ewaldRealSpaceCore_le_alpha_tail alpha (distance a s) ha hdist_ge_one
  have hMulTail :
      |q_i * q_j| * ewaldRealSpaceCore (distance a s) alpha ≤
      |q_i * q_j| * ((2 / alpha ^ 4) / (distance a s) ^ 3) := by
    gcongr
  have hPow : R ^ 3 ≤ (distance a s) ^ 3 := by
    have hDiff : 0 ≤ distance a s - R := by linarith [hFar a]
    have hSum : 0 ≤ (distance a s) ^ 2 + distance a s * R + R ^ 2 := by positivity
    have hCubeDiff : 0 ≤ (distance a s) ^ 3 - R ^ 3 := by
      have hFact : (distance a s) ^ 3 - R ^ 3 =
          (distance a s - R) * ((distance a s) ^ 2 + distance a s * R + R ^ 2) := by ring
      rw [hFact]
      exact mul_nonneg hDiff hSum
    linarith
  have hR3pos : 0 < R ^ 3 := by positivity
  have hInv : 1 / (distance a s) ^ 3 ≤ 1 / R ^ 3 := by
    exact one_div_le_one_div_of_le hR3pos hPow
  have hCoeffNonneg : 0 ≤ 2 / alpha ^ 4 := by positivity
  have hFrac : ((2 / alpha ^ 4) / (distance a s) ^ 3) ≤ ((2 / alpha ^ 4) / R ^ 3) := by
    simpa [div_eq_mul_inv] using mul_le_mul_of_nonneg_left hInv hCoeffNonneg
  have hAbsCoeff : 0 ≤ |q_i * q_j| := abs_nonneg _
  have hFinalMul :
      |q_i * q_j| * ((2 / alpha ^ 4) / (distance a s) ^ 3) ≤
      |q_i * q_j| * ((2 / alpha ^ 4) / R ^ 3) := by
    gcongr
  calc
    |exactRealEwaldScore q_i q_j alpha (distance a s)|
        ≤ |q_i * q_j| * ewaldRealSpaceCore (distance a s) alpha := hEnv
    _ ≤ |q_i * q_j| * ((2 / alpha ^ 4) / (distance a s) ^ 3) := hMulTail
    _ ≤ |q_i * q_j| * ((2 / alpha ^ 4) / R ^ 3) := hFinalMul

/--
  Adding the far-field Ewald correction to a base scorer stays within an explicit
  uniform radius of the base scorer on the chosen state.
-/
theorem additive_exactRealEwald_uniform_error_at_state
    {A : Type u} {S : Type v}
    (uBase : A → S → ℝ)
    (distance : A → S → ℝ)
    (q_i q_j alpha R : ℝ)
    (s : S)
    (ha : 0 < alpha)
    (hR : 1 ≤ R)
    (hFar : ∀ a, R ≤ distance a s) :
    ∀ a,
      |((uBase a s + exactRealEwaldScore q_i q_j alpha (distance a s)) - uBase a s)|
        ≤ realEwaldFarFieldErrorBound q_i q_j alpha R := by
  intro a
  simpa [sub_eq_add_neg, add_assoc] using
    exactRealEwaldScore_far_field_bound distance q_i q_j alpha R s ha hR hFar a

/--
  A far-field Ewald correction yields a theorem-backed certified top-1 survivor
  set around any chosen base scorer.
-/
noncomputable def additive_exactRealEwald_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [DecidableEq A] [Nonempty A]
    (uBase : A → S → ℝ)
    (distance : A → S → ℝ)
    (q_i q_j alpha R : ℝ)
    (s : S)
    (ha : 0 < alpha)
    (hR : 1 ≤ R)
    (hFar : ∀ a, R ≤ distance a s) :
    CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => uBase a s + exactRealEwaldScore q_i q_j alpha (distance a s))
    (fun a => uBase a s)
    (realEwaldFarFieldErrorBound q_i q_j alpha R)
    (additive_exactRealEwald_uniform_error_at_state uBase distance q_i q_j alpha R s ha hR hFar)
    (realEwaldFarFieldErrorBound_nonneg q_i q_j alpha R ha hR)

/-- Soundness of the additive far-field Ewald certified top-1 survivor set. -/
theorem additive_exactRealEwald_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [DecidableEq A] [Nonempty A]
    (uBase : A → S → ℝ)
    (distance : A → S → ℝ)
    (q_i q_j alpha R : ℝ)
    (s : S)
    (ha : 0 < alpha)
    (hR : 1 ≤ R)
    (hFar : ∀ a, R ≤ distance a s) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => uBase a s + exactRealEwaldScore q_i q_j alpha (distance a s))
      (fun a => uBase a s)
      (realEwaldFarFieldErrorBound q_i q_j alpha R)
      (additive_exactRealEwald_uniform_error_at_state uBase distance q_i q_j alpha R s ha hR hFar)
      (realEwaldFarFieldErrorBound_nonneg q_i q_j alpha R ha hR)).exactTopK
      ⊆ (additive_exactRealEwald_certified_top1 uBase distance q_i q_j alpha R s ha hR hFar).survivors := by
  simpa [additive_exactRealEwald_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => uBase a s + exactRealEwaldScore q_i q_j alpha (distance a s))
      (fun a => uBase a s)
      (realEwaldFarFieldErrorBound q_i q_j alpha R)
      (additive_exactRealEwald_uniform_error_at_state uBase distance q_i q_j alpha R s ha hR hFar)
      (realEwaldFarFieldErrorBound_nonneg q_i q_j alpha R ha hR)

/--
  The far-field Ewald correction also yields a runtime-facing optimizer witness
  over the base scorer's ambiguity band.
-/
noncomputable def additive_exactRealEwald_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uBase : A → S → ℝ)
    (distance : A → S → ℝ)
    (q_i q_j alpha R : ℝ)
    (s : S)
    (ha : 0 < alpha)
    (hR : 1 ≤ R)
    (hFar : ∀ a, R ≤ distance a s) :
    CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => uBase a s + exactRealEwaldScore q_i q_j alpha (distance a s))
    (fun a => uBase a s)
    (realEwaldFarFieldErrorBound q_i q_j alpha R)
    (additive_exactRealEwald_uniform_error_at_state uBase distance q_i q_j alpha R s ha hR hFar)
    (realEwaldFarFieldErrorBound_nonneg q_i q_j alpha R ha hR)

noncomputable def additive_exactRealEwald_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uBase : A → S → ℝ)
    (distance : A → S → ℝ)
    (q_i q_j alpha R : ℝ)
    (s : S)
    (ha : 0 < alpha)
    (hR : 1 ≤ R)
    (hFar : ∀ a, R ≤ distance a s) :
    OptimizerWitness A :=
  (additive_exactRealEwald_coherent_optimizer_witness uBase distance q_i q_j alpha R s ha hR hFar).toOptimizerWitness

/--
  Because Real-Space Ewald decays exponentially, it is easily dominated by
  the 6-power polynomial tail at large R, allowing us to re-use our lattice sum bounds.
 -/
theorem exactRealEwald_satisfiesBoundedPotential_of_tailBound
    (prob : MolecularSrank.MDBindingProblem)
    [Fintype MolecularSrank.MDAction] [Fintype MolecularSrank.MDState] [Nonempty MolecularSrank.MDState]
    (distance : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    (q_i q_j alpha tail_coefficient : ℝ)
    (w : ∀ s : MolecularSrank.MDState,
      { a : MolecularSrank.MDAction // StrictOpt prob.toDecisionProblem a s })
    (hGapPos : 0 < finiteMinimumGap prob w)
    (hUtility : prob.utility = fun a s => exactRealEwaldScore q_i q_j alpha (distance a s))
    (hTail : ∀ (atomIdx : Nat) (hAtomInProtein : atomIdx < prob.protein.numAtoms)
      (axis : Fin 3) (s s' : MolecularSrank.MDState) (R : ℝ),
      0 < R →
      (∀ j : Fin (MolecularSrank.numMDCoordinates prob),
        j ≠ MolecularSrank.proteinCoordFin prob atomIdx hAtomInProtein axis →
        MolecularSrank.mdProj prob s j = MolecularSrank.mdProj prob s' j) →
      ¬ MolecularSrank.atomWithinCutoff
        (MolecularSrank.proteinAtom prob atomIdx hAtomInProtein) prob.bindingSite R →
      (∀ a : MolecularSrank.MDAction,
        |exactRealEwaldScore q_i q_j alpha (distance a s) - exactRealEwaldScore q_i q_j alpha (distance a s')|
          ≤ tail_coefficient * latticeTailSum 6 R)) :
    SatisfiesBoundedPotential prob tail_coefficient (finiteMinimumGap prob w) := by
  apply satisfiesBoundedPotential_of_tailBound_and_finiteGap prob tail_coefficient w hGapPos
  intro atomIdx hAtomInProtein axis s s' R hR hSame hOutside a
  simpa [hUtility] using hTail atomIdx hAtomInProtein axis s s' R hR hSame hOutside a

end CoulombApproximation
end Tractability
end DecisionQuotient
