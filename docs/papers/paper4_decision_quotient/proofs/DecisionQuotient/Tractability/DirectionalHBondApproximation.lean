/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/DirectionalHBondApproximation.lean

  Directional hydrogen-bond surrogate with radial and angular factors.
  The key theorem shows that if the factor fields are bounded in [0, 1] and each
  factor is Lipschitz under a grid lift, then the multiplicative surrogate is
  itself Lipschitz with the sum of the factor constants.
-/
import DecisionQuotient.Tractability.GridConvergence
import DecisionQuotient.Tractability.SignInvariance
import Mathlib.Data.Real.Basic

namespace DecisionQuotient
namespace Tractability
namespace DirectionalHBondApproximation

open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer
open GridConvergence
open SignInvariance

universe u v

/-- Directional hydrogen-bond surrogate with normalized radial and angular factors. -/
noncomputable def directionalHBondScore (radial donorAngle acceptorAngle : ℝ) : ℝ :=
  radial * donorAngle * acceptorAngle

noncomputable def directionalHBondDecisionProblem {A : Type u} {S : Type v}
    (radial donorAngle acceptorAngle : A → S → ℝ) : DecisionProblem A S where
  utility := fun a s =>
    directionalHBondScore (radial a s) (donorAngle a s) (acceptorAngle a s)

/-- Pointwise unit-interval constraint for normalized hydrogen-bond factors. -/
def UnitIntervalFactor {A : Type u} {S : Type v} (f : A → S → ℝ) : Prop :=
  ∀ a s, 0 ≤ f a s ∧ f a s ≤ 1

/-- A state-only weight in [0,1] absorbed into a UnitIntervalFactor field
    preserves the unit-interval constraint.

    Key application: in pi-stacking, `strengths s = rec_strength s * lig_strength s`
    depends only on the ring-pair state `s`, not the pose `a`. This lemma shows
    that `fun a s => strengths s * radial a s` is still a valid UnitIntervalFactor,
    so the three-factor certificate applies unchanged after absorption. -/
theorem scaledByStateWeight_unitIntervalFactor {A : Type u} {S : Type v}
    (w : S → ℝ) (f : A → S → ℝ)
    (hw : ∀ s, 0 ≤ w s ∧ w s ≤ 1)
    (hf : UnitIntervalFactor f) :
    UnitIntervalFactor (fun a s => w s * f a s) := by
  intro a s
  rcases hw s with ⟨hlo, hhi⟩
  rcases hf a s with ⟨hflo, hfhi⟩
  exact ⟨mul_nonneg hlo hflo, by nlinarith⟩

/-- A state-only weight in [0,1] absorbed into a factor does not worsen
    the Lipschitz constant.

    Proof: `|w(lift sg) * fCont - w(lift sg) * fGrid|`
            `= w(lift sg) * |fCont - fGrid|`   (since w ≥ 0)
            `≤ 1 * |fCont - fGrid|`             (since w ≤ 1)
            `≤ L * stateError sg`               (by hypothesis)

    The Lipschitz constant `L` is unchanged. This holds because `w` is
    state-only: it factors out of the pose-difference absolutely. -/
theorem scaledByStateWeight_lipschitz {A : Type u} {Scont : Type v} {Sgrid : Type v}
    (wCont : Scont → ℝ) (fCont : A → Scont → ℝ) (fGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont) (stateError : Sgrid → ℝ) (L : ℝ)
    (hw : ∀ s, 0 ≤ wCont s ∧ wCont s ≤ 1)
    (hLip : LipschitzUtilityApprox fCont fGrid lift stateError L) :
    LipschitzUtilityApprox
      (fun a s => wCont s * fCont a s)
      (fun a sGrid => wCont (lift sGrid) * fGrid a sGrid)
      lift stateError L := by
  intro a sGrid
  rcases hw (lift sGrid) with ⟨hlo, hhi⟩
  have hKey : |wCont (lift sGrid) * fCont a (lift sGrid) - wCont (lift sGrid) * fGrid a sGrid|
      = wCont (lift sGrid) * |fCont a (lift sGrid) - fGrid a sGrid| := by
    rw [← mul_sub, abs_mul, abs_of_nonneg hlo]
  rw [hKey]
  exact ((mul_le_mul_of_nonneg_right hhi (abs_nonneg _)).trans_eq (one_mul _)).trans
    (hLip a sGrid)

theorem directionalHBondScore_sub_le_component_sum
    {r1 r2 d1 d2 a1 a2 Lr Ld La err : ℝ}
    (hr1 : 0 ≤ r1) (hr1b : r1 ≤ 1)
    (hr2 : 0 ≤ r2) (hr2b : r2 ≤ 1)
    (hd1 : 0 ≤ d1) (hd1b : d1 ≤ 1)
    (hd2 : 0 ≤ d2) (hd2b : d2 ≤ 1)
    (ha1 : 0 ≤ a1) (ha1b : a1 ≤ 1)
    (ha2 : 0 ≤ a2) (ha2b : a2 ≤ 1)
    (hLr : 0 ≤ Lr) (hLd : 0 ≤ Ld) (hLa : 0 ≤ La)
    (herr : 0 ≤ err)
    (hRadial : |r1 - r2| ≤ Lr * err)
    (hDonor : |d1 - d2| ≤ Ld * err)
    (hAcceptor : |a1 - a2| ≤ La * err) :
    |directionalHBondScore r1 d1 a1 - directionalHBondScore r2 d2 a2| ≤
      (Lr + Ld + La) * err := by
  have hProd_d1a1_nonneg : 0 ≤ d1 * a1 := by positivity
  have hProd_d1a1_le_one : d1 * a1 ≤ 1 := by nlinarith
  have hAbs_d1a1_le_one : |d1 * a1| ≤ 1 := by
    rw [abs_of_nonneg hProd_d1a1_nonneg]
    exact hProd_d1a1_le_one

  have hProd_r2a1_nonneg : 0 ≤ r2 * a1 := by positivity
  have hProd_r2a1_le_one : r2 * a1 ≤ 1 := by nlinarith
  have hAbs_r2a1_le_one : |r2 * a1| ≤ 1 := by
    rw [abs_of_nonneg hProd_r2a1_nonneg]
    exact hProd_r2a1_le_one

  have hProd_r2d2_nonneg : 0 ≤ r2 * d2 := by positivity
  have hProd_r2d2_le_one : r2 * d2 ≤ 1 := by nlinarith
  have hAbs_r2d2_le_one : |r2 * d2| ≤ 1 := by
    rw [abs_of_nonneg hProd_r2d2_nonneg]
    exact hProd_r2d2_le_one

  have hTerm1 : |(r1 - r2) * d1 * a1| ≤ Lr * err := by
    have hBase : |(r1 - r2) * d1 * a1| ≤ |r1 - r2| := by
      calc
        |(r1 - r2) * d1 * a1| = |r1 - r2| * |d1 * a1| := by
          rw [show (r1 - r2) * d1 * a1 = (r1 - r2) * (d1 * a1) by ring, abs_mul]
        _ ≤ |r1 - r2| * 1 := by
          exact mul_le_mul_of_nonneg_left hAbs_d1a1_le_one (abs_nonneg _)
        _ = |r1 - r2| := by ring
    exact hBase.trans hRadial

  have hTerm2 : |r2 * (d1 - d2) * a1| ≤ Ld * err := by
    have hBase : |r2 * (d1 - d2) * a1| ≤ |d1 - d2| := by
      calc
        |r2 * (d1 - d2) * a1| = |d1 - d2| * |r2 * a1| := by
          rw [show r2 * (d1 - d2) * a1 = (d1 - d2) * (r2 * a1) by ring, abs_mul]
        _ ≤ |d1 - d2| * 1 := by
          exact mul_le_mul_of_nonneg_left hAbs_r2a1_le_one (abs_nonneg _)
        _ = |d1 - d2| := by ring
    exact hBase.trans hDonor

  have hTerm3 : |r2 * d2 * (a1 - a2)| ≤ La * err := by
    have hBase : |r2 * d2 * (a1 - a2)| ≤ |a1 - a2| := by
      calc
        |r2 * d2 * (a1 - a2)| = |a1 - a2| * |r2 * d2| := by
          rw [show r2 * d2 * (a1 - a2) = (a1 - a2) * (r2 * d2) by ring, abs_mul]
        _ ≤ |a1 - a2| * 1 := by
          exact mul_le_mul_of_nonneg_left hAbs_r2d2_le_one (abs_nonneg _)
        _ = |a1 - a2| := by ring
    exact hBase.trans hAcceptor

  have hSplit :
      directionalHBondScore r1 d1 a1 - directionalHBondScore r2 d2 a2 =
        (r1 - r2) * d1 * a1 + r2 * (d1 - d2) * a1 + r2 * d2 * (a1 - a2) := by
    unfold directionalHBondScore
    ring

  calc
    |directionalHBondScore r1 d1 a1 - directionalHBondScore r2 d2 a2|
        = |((r1 - r2) * d1 * a1 + r2 * (d1 - d2) * a1) + r2 * d2 * (a1 - a2)| := by
            rw [hSplit]
    _ ≤ |(r1 - r2) * d1 * a1 + r2 * (d1 - d2) * a1| + |r2 * d2 * (a1 - a2)| :=
          abs_add_le _ _
    _ ≤ (|(r1 - r2) * d1 * a1| + |r2 * (d1 - d2) * a1|) + |r2 * d2 * (a1 - a2)| := by
          gcongr
          exact abs_add_le _ _
    _ ≤ (Lr * err + Ld * err) + La * err := by
          linarith [hTerm1, hTerm2, hTerm3]
    _ = (Lr + Ld + La) * err := by ring

/-- Any two finite directional H-bond score families admit a canonical finite-domain discrepancy radius. -/
noncomputable def finiteDirectionalHBondErrorRadius
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) : ℝ :=
  finiteUniformErrorRadius
    (directionalHBondDecisionProblem radialExact donorExact acceptorExact)
    (directionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse)

theorem finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) :
    UniformUtilityApprox
      (directionalHBondDecisionProblem radialExact donorExact acceptorExact)
      (directionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse)
      (finiteDirectionalHBondErrorRadius radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse) := by
  unfold finiteDirectionalHBondErrorRadius
  exact finiteUniformErrorRadius_witnesses_uniformApprox
    (directionalHBondDecisionProblem radialExact donorExact acceptorExact)
    (directionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse)

theorem finiteDirectionalHBondErrorRadius_nonneg
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) :
    0 ≤ finiteDirectionalHBondErrorRadius
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse := by
  classical
  rcases ‹Nonempty A› with ⟨a⟩
  rcases ‹Nonempty S› with ⟨s⟩
  exact le_trans (abs_nonneg _)
    (finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)

/-- Generic finite exact-vs-coarse directional H-bond certified top-1 survivor set. -/
noncomputable def exact_vs_coarse_directionalHBond_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) : CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => directionalHBondDecisionProblem radialExact donorExact acceptorExact |>.utility a s)
    (fun a => directionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse |>.utility a s)
    (finiteDirectionalHBondErrorRadius
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
    (fun a => finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)
    (finiteDirectionalHBondErrorRadius_nonneg
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

theorem exact_vs_coarse_directionalHBond_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => directionalHBondDecisionProblem radialExact donorExact acceptorExact |>.utility a s)
      (fun a => directionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse |>.utility a s)
      (finiteDirectionalHBondErrorRadius
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
      (fun a => finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)
      (finiteDirectionalHBondErrorRadius_nonneg
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)).exactTopK
      ⊆ (exact_vs_coarse_directionalHBond_certified_top1
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s).survivors := by
  simpa [exact_vs_coarse_directionalHBond_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => directionalHBondDecisionProblem radialExact donorExact acceptorExact |>.utility a s)
      (fun a => directionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse |>.utility a s)
      (finiteDirectionalHBondErrorRadius
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
      (fun a => finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)
      (finiteDirectionalHBondErrorRadius_nonneg
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

noncomputable def exact_vs_coarse_directionalHBond_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) : CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => directionalHBondDecisionProblem radialExact donorExact acceptorExact |>.utility a s)
    (fun a => directionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse |>.utility a s)
    (finiteDirectionalHBondErrorRadius
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
    (fun a => finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)
    (finiteDirectionalHBondErrorRadius_nonneg
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

noncomputable def exact_vs_coarse_directionalHBond_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) : OptimizerWitness A :=
  (exact_vs_coarse_directionalHBond_coherent_optimizer_witness
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s).toOptimizerWitness

/-- Attractive directional H-bond energy family with fixed precomputed factor fields. -/
noncomputable def attractiveDirectionalHBondDecisionProblem {A : Type u} {S : Type v}
    (radial donorAngle acceptorAngle : A → S → ℝ) : DecisionProblem A S :=
  negDecisionProblem <| directionalHBondDecisionProblem radial donorAngle acceptorAngle

theorem exact_vs_coarse_attractiveDirectionalHBond_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) :
    UniformUtilityApprox
      (attractiveDirectionalHBondDecisionProblem radialExact donorExact acceptorExact)
      (attractiveDirectionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse)
      (finiteDirectionalHBondErrorRadius radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse) := by
  unfold attractiveDirectionalHBondDecisionProblem
  exact neg_uniformApprox
    (directionalHBondDecisionProblem radialExact donorExact acceptorExact)
    (directionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse)
    (finiteDirectionalHBondErrorRadius radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
    (finiteDirectionalHBondErrorRadius_witnesses_uniformApprox radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

noncomputable def exact_vs_coarse_attractiveDirectionalHBond_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) : CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_negated_uniformApprox
    (fun a => directionalHBondDecisionProblem radialExact donorExact acceptorExact |>.utility a s)
    (fun a => directionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse |>.utility a s)
    (finiteDirectionalHBondErrorRadius radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
    (fun a => finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)
    (finiteDirectionalHBondErrorRadius_nonneg radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

theorem exact_vs_coarse_attractiveDirectionalHBond_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (negUtility <| fun a => directionalHBondDecisionProblem radialExact donorExact acceptorExact |>.utility a s)
      (negUtility <| fun a => directionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse |>.utility a s)
      (finiteDirectionalHBondErrorRadius radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
      (neg_utility_uniformApprox
        (fun a => directionalHBondDecisionProblem radialExact donorExact acceptorExact |>.utility a s)
        (fun a => directionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse |>.utility a s)
        (finiteDirectionalHBondErrorRadius radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
        (fun a => finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
          radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s))
      (finiteDirectionalHBondErrorRadius_nonneg radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)).exactTopK
      ⊆ (exact_vs_coarse_attractiveDirectionalHBond_certified_top1 radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s).survivors := by
  simpa [exact_vs_coarse_attractiveDirectionalHBond_certified_top1]
    using certified_top1_survivor_set_of_negated_uniformApprox_sound
      (fun a => directionalHBondDecisionProblem radialExact donorExact acceptorExact |>.utility a s)
      (fun a => directionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse |>.utility a s)
      (finiteDirectionalHBondErrorRadius radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
      (fun a => finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)
      (finiteDirectionalHBondErrorRadius_nonneg radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

noncomputable def exact_vs_coarse_attractiveDirectionalHBond_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) : CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_negated_uniformApprox_top1
    (fun a => directionalHBondDecisionProblem radialExact donorExact acceptorExact |>.utility a s)
    (fun a => directionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse |>.utility a s)
    (finiteDirectionalHBondErrorRadius radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
    (fun a => finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)
    (finiteDirectionalHBondErrorRadius_nonneg radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

noncomputable def exact_vs_coarse_attractiveDirectionalHBond_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) : OptimizerWitness A :=
  (exact_vs_coarse_attractiveDirectionalHBond_coherent_optimizer_witness
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s).toOptimizerWitness

/-- Factor-wise Lipschitz bounds over a grid lift imply a Lipschitz bound for the full directional H-bond surrogate. -/
theorem directionalHBond_lipschitzUtilityApprox
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    (radialCont donorCont acceptorCont : A → Scont → ℝ)
    (radialGrid donorGrid acceptorGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (stateError : Sgrid → ℝ)
    (Lr Ld La : ℝ)
    (hRadial : LipschitzUtilityApprox radialCont radialGrid lift stateError Lr)
    (hDonor : LipschitzUtilityApprox donorCont donorGrid lift stateError Ld)
    (hAcceptor : LipschitzUtilityApprox acceptorCont acceptorGrid lift stateError La)
    (hRadialContUnit : UnitIntervalFactor radialCont)
    (hDonorContUnit : UnitIntervalFactor donorCont)
    (hAcceptorContUnit : UnitIntervalFactor acceptorCont)
    (hRadialGridUnit : UnitIntervalFactor radialGrid)
    (hDonorGridUnit : UnitIntervalFactor donorGrid)
    (hAcceptorGridUnit : UnitIntervalFactor acceptorGrid)
    (hLr : 0 ≤ Lr) (hLd : 0 ≤ Ld) (hLa : 0 ≤ La)
    (hErr : ∀ sGrid, 0 ≤ stateError sGrid) :
    LipschitzUtilityApprox
      (fun a s => directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s))
      (fun a sGrid => directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid))
      lift stateError (Lr + Ld + La) := by
  intro a sGrid
  rcases hRadialContUnit a (lift sGrid) with ⟨hr1, hr1b⟩
  rcases hRadialGridUnit a sGrid with ⟨hr2, hr2b⟩
  rcases hDonorContUnit a (lift sGrid) with ⟨hd1, hd1b⟩
  rcases hDonorGridUnit a sGrid with ⟨hd2, hd2b⟩
  rcases hAcceptorContUnit a (lift sGrid) with ⟨ha1, ha1b⟩
  rcases hAcceptorGridUnit a sGrid with ⟨ha2, ha2b⟩
  exact directionalHBondScore_sub_le_component_sum
    hr1 hr1b hr2 hr2b hd1 hd1b hd2 hd2b ha1 ha1b ha2 ha2b
    hLr hLd hLa (hErr sGrid)
    (hRadial a sGrid) (hDonor a sGrid) (hAcceptor a sGrid)

theorem directionalHBond_resolutionControlledApprox
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    (radialCont donorCont acceptorCont : A → Scont → ℝ)
    (radialGrid donorGrid acceptorGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (stateError : Sgrid → ℝ)
    (Lr Ld La res : ℝ)
    (hRadial : LipschitzUtilityApprox radialCont radialGrid lift stateError Lr)
    (hDonor : LipschitzUtilityApprox donorCont donorGrid lift stateError Ld)
    (hAcceptor : LipschitzUtilityApprox acceptorCont acceptorGrid lift stateError La)
    (hRadialContUnit : UnitIntervalFactor radialCont)
    (hDonorContUnit : UnitIntervalFactor donorCont)
    (hAcceptorContUnit : UnitIntervalFactor acceptorCont)
    (hRadialGridUnit : UnitIntervalFactor radialGrid)
    (hDonorGridUnit : UnitIntervalFactor donorGrid)
    (hAcceptorGridUnit : UnitIntervalFactor acceptorGrid)
    (hState : ∀ sGrid, stateError sGrid ≤ res)
    (hLr : 0 ≤ Lr) (hLd : 0 ≤ Ld) (hLa : 0 ≤ La)
    (hErr : ∀ sGrid, 0 ≤ stateError sGrid) :
    ResolutionControlledApprox
      (fun a s => directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s))
      (fun a sGrid => directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid))
      lift (fun r => (Lr + Ld + La) * r) res := by
  apply lipschitzUtilityApprox_implies_resolutionControlled
  · exact directionalHBond_lipschitzUtilityApprox
      radialCont donorCont acceptorCont
      radialGrid donorGrid acceptorGrid
      lift stateError Lr Ld La
      hRadial hDonor hAcceptor
      hRadialContUnit hDonorContUnit hAcceptorContUnit
      hRadialGridUnit hDonorGridUnit hAcceptorGridUnit
      hLr hLd hLa hErr
  · exact hState
  · linarith

noncomputable def directionalHBond_resolutionControlled_certified_top1
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    [Fintype A] [Fintype Sgrid] [DecidableEq A] [Nonempty A]
    (radialCont donorCont acceptorCont : A → Scont → ℝ)
    (radialGrid donorGrid acceptorGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (stateError : Sgrid → ℝ)
    (Lr Ld La res : ℝ)
    (sGrid : Sgrid)
    (hApprox : ResolutionControlledApprox
      (fun a s => directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s))
      (fun a sGrid => directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid))
      lift (fun r => (Lr + Ld + La) * r) res)
    (hL : 0 ≤ Lr + Ld + La)
    (hRes : 0 ≤ res) :
    CertifiedSurvivorSet A :=
  let hEps : 0 ≤ (Lr + Ld + La) * res := mul_nonneg hL hRes
  resolutionControlledApprox_certified_top1
    (fun a s => directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s))
    (fun a sGrid => directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid))
    lift (fun r => (Lr + Ld + La) * r) res sGrid hApprox hEps

theorem directionalHBond_resolutionControlled_certified_top1_sound
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    [Fintype A] [Fintype Sgrid] [DecidableEq A] [Nonempty A]
    (radialCont donorCont acceptorCont : A → Scont → ℝ)
    (radialGrid donorGrid acceptorGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (stateError : Sgrid → ℝ)
    (Lr Ld La res : ℝ)
    (sGrid : Sgrid)
    (hApprox : ResolutionControlledApprox
      (fun a s => directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s))
      (fun a sGrid => directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid))
      lift (fun r => (Lr + Ld + La) * r) res)
    (hL : 0 ≤ Lr + Ld + La)
    (hRes : 0 ≤ res) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => directionalHBondScore (radialCont a (lift sGrid)) (donorCont a (lift sGrid)) (acceptorCont a (lift sGrid)))
      (fun a => directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid))
      ((Lr + Ld + La) * res)
      (fun a => hApprox a sGrid)
      (mul_nonneg hL hRes)).exactTopK
      ⊆ (directionalHBond_resolutionControlled_certified_top1
        radialCont donorCont acceptorCont radialGrid donorGrid acceptorGrid
        lift stateError Lr Ld La res sGrid hApprox hL hRes).survivors := by
  simpa [directionalHBond_resolutionControlled_certified_top1]
    using resolutionControlledApprox_certified_top1_sound
      (fun a s => directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s))
      (fun a sGrid => directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid))
      lift (fun r => (Lr + Ld + La) * r) res sGrid hApprox (mul_nonneg hL hRes)

noncomputable def directionalHBond_resolutionControlled_coherent_optimizer_witness
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    [Fintype A] [Fintype Sgrid] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (radialCont donorCont acceptorCont : A → Scont → ℝ)
    (radialGrid donorGrid acceptorGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (stateError : Sgrid → ℝ)
    (Lr Ld La res : ℝ)
    (sGrid : Sgrid)
    (hApprox : ResolutionControlledApprox
      (fun a s => directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s))
      (fun a sGrid => directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid))
      lift (fun r => (Lr + Ld + La) * r) res)
    (hL : 0 ≤ Lr + Ld + La)
    (hRes : 0 ≤ res) :
    CoherentOptimizerWitness A :=
  resolutionControlledApprox_coherent_optimizer_witness
    (fun a s => directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s))
    (fun a sGrid => directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid))
    lift (fun r => (Lr + Ld + La) * r) res sGrid hApprox (mul_nonneg hL hRes)

noncomputable def directionalHBond_resolutionControlled_optimizer_witness
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    [Fintype A] [Fintype Sgrid] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (radialCont donorCont acceptorCont : A → Scont → ℝ)
    (radialGrid donorGrid acceptorGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (stateError : Sgrid → ℝ)
    (Lr Ld La res : ℝ)
    (sGrid : Sgrid)
    (hApprox : ResolutionControlledApprox
      (fun a s => directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s))
      (fun a sGrid => directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid))
      lift (fun r => (Lr + Ld + La) * r) res)
    (hL : 0 ≤ Lr + Ld + La)
    (hRes : 0 ≤ res) :
    OptimizerWitness A :=
  (directionalHBond_resolutionControlled_coherent_optimizer_witness
    radialCont donorCont acceptorCont radialGrid donorGrid acceptorGrid
    lift stateError Lr Ld La res sGrid hApprox hL hRes).toOptimizerWitness

theorem attractiveDirectionalHBond_resolutionControlledApprox
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    (radialCont donorCont acceptorCont : A → Scont → ℝ)
    (radialGrid donorGrid acceptorGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (stateError : Sgrid → ℝ)
    (Lr Ld La res : ℝ)
    (hApprox : ResolutionControlledApprox
      (fun a s => directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s))
      (fun a sGrid => directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid))
      lift (fun r => (Lr + Ld + La) * r) res) :
    ResolutionControlledApprox
      (fun a s => -(directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s)))
      (fun a sGrid => -(directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid)))
      lift (fun r => (Lr + Ld + La) * r) res := by
  exact neg_resolutionControlledApprox
    (fun a s => directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s))
    (fun a sGrid => directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid))
    lift (fun r => (Lr + Ld + La) * r) res hApprox

noncomputable def attractiveDirectionalHBond_resolutionControlled_certified_top1
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    [Fintype A] [Fintype Sgrid] [DecidableEq A] [Nonempty A]
    (radialCont donorCont acceptorCont : A → Scont → ℝ)
    (radialGrid donorGrid acceptorGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (stateError : Sgrid → ℝ)
    (Lr Ld La res : ℝ)
    (sGrid : Sgrid)
    (hApprox : ResolutionControlledApprox
      (fun a s => directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s))
      (fun a sGrid => directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid))
      lift (fun r => (Lr + Ld + La) * r) res)
    (hL : 0 ≤ Lr + Ld + La)
    (hRes : 0 ≤ res) :
    CertifiedSurvivorSet A :=
  resolutionControlledApprox_certified_top1
    (fun a s => -(directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s)))
    (fun a sGrid => -(directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid)))
    lift (fun r => (Lr + Ld + La) * r) res sGrid
    (attractiveDirectionalHBond_resolutionControlledApprox radialCont donorCont acceptorCont radialGrid donorGrid acceptorGrid lift stateError Lr Ld La res hApprox)
    (mul_nonneg hL hRes)

noncomputable def attractiveDirectionalHBond_resolutionControlled_coherent_optimizer_witness
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    [Fintype A] [Fintype Sgrid] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (radialCont donorCont acceptorCont : A → Scont → ℝ)
    (radialGrid donorGrid acceptorGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (stateError : Sgrid → ℝ)
    (Lr Ld La res : ℝ)
    (sGrid : Sgrid)
    (hApprox : ResolutionControlledApprox
      (fun a s => directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s))
      (fun a sGrid => directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid))
      lift (fun r => (Lr + Ld + La) * r) res)
    (hL : 0 ≤ Lr + Ld + La)
    (hRes : 0 ≤ res) :
    CoherentOptimizerWitness A :=
  resolutionControlledApprox_coherent_optimizer_witness
    (fun a s => -(directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s)))
    (fun a sGrid => -(directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid)))
    lift (fun r => (Lr + Ld + La) * r) res sGrid
    (attractiveDirectionalHBond_resolutionControlledApprox radialCont donorCont acceptorCont radialGrid donorGrid acceptorGrid lift stateError Lr Ld La res hApprox)
    (mul_nonneg hL hRes)

noncomputable def attractiveDirectionalHBond_resolutionControlled_optimizer_witness
    {A : Type u} {Scont : Type v} {Sgrid : Type v}
    [Fintype A] [Fintype Sgrid] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (radialCont donorCont acceptorCont : A → Scont → ℝ)
    (radialGrid donorGrid acceptorGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (stateError : Sgrid → ℝ)
    (Lr Ld La res : ℝ)
    (sGrid : Sgrid)
    (hApprox : ResolutionControlledApprox
      (fun a s => directionalHBondScore (radialCont a s) (donorCont a s) (acceptorCont a s))
      (fun a sGrid => directionalHBondScore (radialGrid a sGrid) (donorGrid a sGrid) (acceptorGrid a sGrid))
      lift (fun r => (Lr + Ld + La) * r) res)
    (hL : 0 ≤ Lr + Ld + La)
    (hRes : 0 ≤ res) :
    OptimizerWitness A :=
  (attractiveDirectionalHBond_resolutionControlled_coherent_optimizer_witness
    radialCont donorCont acceptorCont radialGrid donorGrid acceptorGrid
    lift stateError Lr Ld La res sGrid hApprox hL hRes).toOptimizerWitness

end DirectionalHBondApproximation
end Tractability
end DecisionQuotient
