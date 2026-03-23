/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/RichChemistryApproximation.lean

  Additive composition theorems for richer chemistry families built from:
  - theorem-backed nonbonded terms (LJ + screened Coulomb)
  - bounded contact/desolvation surrogate
  - directional hydrogen-bond surrogate
-/
import DecisionQuotient.Tractability.NonbondedApproximation
import DecisionQuotient.Tractability.ContactApproximation
import DecisionQuotient.Tractability.DirectionalHBondApproximation
import DecisionQuotient.Tractability.SignInvariance
import DecisionQuotient.Tractability.ConformerSearch

namespace DecisionQuotient
namespace Tractability
namespace RichChemistryApproximation

open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer
open NonbondedApproximation
open ContactApproximation
open DirectionalHBondApproximation
open SignInvariance
open ConformerSearch

universe u v

/-- Exact/coarse additive polar surrogate: bounded contact plus two directional H-bond channels. -/
noncomputable def exactPolarSurrogateDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (w β : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ) :
    DecisionProblem A S :=
  sumDecisionProblems
    (sumDecisionProblems
      (exactContactDecisionProblem w β distance)
      (directionalHBondDecisionProblem
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor))
    (directionalHBondDecisionProblem
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor)

noncomputable def coarsePolarSurrogateDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) :
    DecisionProblem A S :=
  sumDecisionProblems
    (sumDecisionProblems
      (cutoffContactDecisionProblem w β rc distance)
      (directionalHBondDecisionProblem
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor))
    (directionalHBondDecisionProblem
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

noncomputable def polarSurrogateErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) : ℝ :=
  contactCutoffErrorRadius w β rc distance +
    finiteDirectionalHBondErrorRadius
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor +
    finiteDirectionalHBondErrorRadius
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor

theorem exact_vs_coarse_polarSurrogate_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) :
    UniformUtilityApprox
      (exactPolarSurrogateDecisionProblem distance w β
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor)
      (coarsePolarSurrogateDecisionProblem distance w β rc
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
      (polarSurrogateErrorRadius
        distance w β rc
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor) := by
  unfold exactPolarSurrogateDecisionProblem coarsePolarSurrogateDecisionProblem polarSurrogateErrorRadius
  have hContactRecDonor :
      UniformUtilityApprox
        (sumDecisionProblems
          (exactContactDecisionProblem w β distance)
          (directionalHBondDecisionProblem
            radialExactRecDonor donorExactRecDonor acceptorExactRecDonor))
        (sumDecisionProblems
          (cutoffContactDecisionProblem w β rc distance)
          (directionalHBondDecisionProblem
            radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor))
        (contactCutoffErrorRadius w β rc distance +
          finiteDirectionalHBondErrorRadius
            radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
            radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor) := by
    exact sum_uniformApprox
      (exactContactDecisionProblem w β distance)
      (cutoffContactDecisionProblem w β rc distance)
      (directionalHBondDecisionProblem
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor)
      (directionalHBondDecisionProblem
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor)
      (contactCutoffErrorRadius w β rc distance)
      (finiteDirectionalHBondErrorRadius
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor)
      (exact_vs_cutoff_contact_uniformApprox w β rc distance)
      (finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor)
  exact sum_channel_uniformApprox
    (sumDecisionProblems
      (exactContactDecisionProblem w β distance)
      (directionalHBondDecisionProblem
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor))
    (sumDecisionProblems
      (cutoffContactDecisionProblem w β rc distance)
      (directionalHBondDecisionProblem
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor))
    (directionalHBondDecisionProblem
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor)
    (directionalHBondDecisionProblem
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    (contactCutoffErrorRadius w β rc distance +
      finiteDirectionalHBondErrorRadius
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor)
    (finiteDirectionalHBondErrorRadius
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    hContactRecDonor
    (finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

theorem polarSurrogateErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) :
    0 ≤ polarSurrogateErrorRadius
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor := by
  unfold polarSurrogateErrorRadius
  exact add_nonneg
    (add_nonneg
      (contactCutoffErrorRadius_nonneg w β rc distance)
      (finiteDirectionalHBondErrorRadius_nonneg
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor))
    (finiteDirectionalHBondErrorRadius_nonneg
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

noncomputable def exact_vs_coarse_polarSurrogate_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) : CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactPolarSurrogateDecisionProblem distance w β
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
    (fun a => coarsePolarSurrogateDecisionProblem distance w β rc
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
    (polarSurrogateErrorRadius
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    (fun a => exact_vs_coarse_polarSurrogate_uniformApprox
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s)
    (polarSurrogateErrorRadius_nonneg
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

theorem exact_vs_coarse_polarSurrogate_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactPolarSurrogateDecisionProblem distance w β
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
      (fun a => coarsePolarSurrogateDecisionProblem distance w β rc
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
      (polarSurrogateErrorRadius
        distance w β rc
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
      (fun a => exact_vs_coarse_polarSurrogate_uniformApprox
        distance w β rc
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s)
      (polarSurrogateErrorRadius_nonneg
        distance w β rc
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)).exactTopK
      ⊆ (exact_vs_coarse_polarSurrogate_certified_top1
        distance w β rc
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor s).survivors := by
  simpa [exact_vs_coarse_polarSurrogate_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactPolarSurrogateDecisionProblem distance w β
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
      (fun a => coarsePolarSurrogateDecisionProblem distance w β rc
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
      (polarSurrogateErrorRadius
        distance w β rc
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
      (fun a => exact_vs_coarse_polarSurrogate_uniformApprox
        distance w β rc
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s)
      (polarSurrogateErrorRadius_nonneg
        distance w β rc
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

noncomputable def exact_vs_coarse_polarSurrogate_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) : CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactPolarSurrogateDecisionProblem distance w β
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
    (fun a => coarsePolarSurrogateDecisionProblem distance w β rc
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
    (polarSurrogateErrorRadius
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    (fun a => exact_vs_coarse_polarSurrogate_uniformApprox
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s)
    (polarSurrogateErrorRadius_nonneg
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

noncomputable def exact_vs_coarse_polarSurrogate_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) : OptimizerWitness A :=
  (exact_vs_coarse_polarSurrogate_coherent_optimizer_witness
    distance w β rc
    radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
    radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
    radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
    radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor s).toOptimizerWitness

/-- Attractive polar energy family: the negative of the positive contact/H-bond compatibility surrogate. -/
noncomputable def exactAttractivePolarSurrogateDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (w β : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ) : DecisionProblem A S :=
  negDecisionProblem <|
    exactPolarSurrogateDecisionProblem distance w β
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor

noncomputable def coarseAttractivePolarSurrogateDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) : DecisionProblem A S :=
  negDecisionProblem <|
    coarsePolarSurrogateDecisionProblem distance w β rc
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor

noncomputable def attractivePolarSurrogateErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) : ℝ :=
  polarSurrogateErrorRadius
    distance w β rc
    radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
    radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
    radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
    radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor

theorem exact_vs_coarse_attractivePolarSurrogate_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) :
    UniformUtilityApprox
      (exactAttractivePolarSurrogateDecisionProblem distance w β
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor)
      (coarseAttractivePolarSurrogateDecisionProblem distance w β rc
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
      (attractivePolarSurrogateErrorRadius
        distance w β rc
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor) := by
  unfold exactAttractivePolarSurrogateDecisionProblem coarseAttractivePolarSurrogateDecisionProblem attractivePolarSurrogateErrorRadius
  exact neg_uniformApprox
    (exactPolarSurrogateDecisionProblem distance w β
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor)
    (coarsePolarSurrogateDecisionProblem distance w β rc
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    (polarSurrogateErrorRadius
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    (exact_vs_coarse_polarSurrogate_uniformApprox
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

theorem attractivePolarSurrogateErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) :
    0 ≤ attractivePolarSurrogateErrorRadius
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor := by
  exact polarSurrogateErrorRadius_nonneg
    distance w β rc
    radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
    radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
    radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
    radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor

noncomputable def exact_vs_coarse_attractivePolarSurrogate_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) : CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_negated_uniformApprox
    (fun a => exactPolarSurrogateDecisionProblem distance w β
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
    (fun a => coarsePolarSurrogateDecisionProblem distance w β rc
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
    (attractivePolarSurrogateErrorRadius
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    (fun a => exact_vs_coarse_polarSurrogate_uniformApprox
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s)
    (attractivePolarSurrogateErrorRadius_nonneg
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

theorem exact_vs_coarse_attractivePolarSurrogate_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (negUtility <| fun a => exactPolarSurrogateDecisionProblem distance w β
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
      (negUtility <| fun a => coarsePolarSurrogateDecisionProblem distance w β rc
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
      (attractivePolarSurrogateErrorRadius
        distance w β rc
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
      (neg_utility_uniformApprox
        (fun a => exactPolarSurrogateDecisionProblem distance w β
          radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
          radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
        (fun a => coarsePolarSurrogateDecisionProblem distance w β rc
          radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
          radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
        (attractivePolarSurrogateErrorRadius
          distance w β rc
          radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
          radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
          radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
          radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
        (fun a => exact_vs_coarse_polarSurrogate_uniformApprox
          distance w β rc
          radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
          radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
          radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
          radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s))
      (attractivePolarSurrogateErrorRadius_nonneg
        distance w β rc
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)).exactTopK
      ⊆ (exact_vs_coarse_attractivePolarSurrogate_certified_top1
        distance w β rc
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor s).survivors := by
  simpa [exact_vs_coarse_attractivePolarSurrogate_certified_top1]
    using certified_top1_survivor_set_of_negated_uniformApprox_sound
      (fun a => exactPolarSurrogateDecisionProblem distance w β
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
      (fun a => coarsePolarSurrogateDecisionProblem distance w β rc
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
      (attractivePolarSurrogateErrorRadius
        distance w β rc
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
      (fun a => exact_vs_coarse_polarSurrogate_uniformApprox
        distance w β rc
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s)
      (attractivePolarSurrogateErrorRadius_nonneg
        distance w β rc
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

noncomputable def exact_vs_coarse_attractivePolarSurrogate_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) : CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_negated_uniformApprox_top1
    (fun a => exactPolarSurrogateDecisionProblem distance w β
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
    (fun a => coarsePolarSurrogateDecisionProblem distance w β rc
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
    (attractivePolarSurrogateErrorRadius
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    (fun a => exact_vs_coarse_polarSurrogate_uniformApprox
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s)
    (attractivePolarSurrogateErrorRadius_nonneg
      distance w β rc
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

noncomputable def exact_vs_coarse_attractivePolarSurrogate_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) : OptimizerWitness A :=
  (exact_vs_coarse_attractivePolarSurrogate_coherent_optimizer_witness
    distance w β rc
    radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
    radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
    radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
    radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor s).toOptimizerWitness

/-- Full exact/coarse rich chemistry family: LJ + screened Coulomb + contact + directional H-bond. -/
noncomputable def exactRichChemistryDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (ε σ q_i q_j κ w β : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (exactLJScreenedCoulombDecisionProblem distance ε σ q_i q_j κ)
    (exactPolarSurrogateDecisionProblem distance w β
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor)

noncomputable def coarseRichChemistryDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (cutoffLJScreenedCoulombDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC)
    (coarsePolarSurrogateDecisionProblem distance w β rcCT
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

noncomputable def richChemistryErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) : ℝ :=
  ljScreenedCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j κ rcSC +
    polarSurrogateErrorRadius
      distance w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor

theorem exact_vs_coarse_richChemistry_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) :
    UniformUtilityApprox
      (exactRichChemistryDecisionProblem distance ε σ q_i q_j κ w β
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor)
      (coarseRichChemistryDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
      (richChemistryErrorRadius
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor) := by
  unfold exactRichChemistryDecisionProblem coarseRichChemistryDecisionProblem richChemistryErrorRadius
  exact sum_uniformApprox
    (exactLJScreenedCoulombDecisionProblem distance ε σ q_i q_j κ)
    (cutoffLJScreenedCoulombDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC)
    (exactPolarSurrogateDecisionProblem distance w β
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor)
    (coarsePolarSurrogateDecisionProblem distance w β rcCT
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    (ljScreenedCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j κ rcSC)
    (polarSurrogateErrorRadius
      distance w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    (exact_vs_cutoff_lj_screened_coulomb_uniformApprox distance ε σ rcLJ q_i q_j κ rcSC)
    (exact_vs_coarse_polarSurrogate_uniformApprox
      distance w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

theorem richChemistryErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) :
    0 ≤ richChemistryErrorRadius
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor := by
  unfold richChemistryErrorRadius
  exact add_nonneg
    (ljScreenedCoulombCutoffErrorRadius_nonneg distance ε σ rcLJ q_i q_j κ rcSC)
    (polarSurrogateErrorRadius_nonneg
      distance w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

noncomputable def exact_vs_coarse_richChemistry_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) : CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactRichChemistryDecisionProblem distance ε σ q_i q_j κ w β
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
    (fun a => coarseRichChemistryDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
    (richChemistryErrorRadius
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    (fun a => exact_vs_coarse_richChemistry_uniformApprox
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s)
    (richChemistryErrorRadius_nonneg
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

theorem exact_vs_coarse_richChemistry_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactRichChemistryDecisionProblem distance ε σ q_i q_j κ w β
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
      (fun a => coarseRichChemistryDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
      (richChemistryErrorRadius
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
      (fun a => exact_vs_coarse_richChemistry_uniformApprox
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s)
      (richChemistryErrorRadius_nonneg
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)).exactTopK
      ⊆ (exact_vs_coarse_richChemistry_certified_top1
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor s).survivors := by
  simpa [exact_vs_coarse_richChemistry_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactRichChemistryDecisionProblem distance ε σ q_i q_j κ w β
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
      (fun a => coarseRichChemistryDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
      (richChemistryErrorRadius
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
      (fun a => exact_vs_coarse_richChemistry_uniformApprox
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s)
      (richChemistryErrorRadius_nonneg
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

noncomputable def exact_vs_coarse_richChemistry_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) : CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactRichChemistryDecisionProblem distance ε σ q_i q_j κ w β
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
    (fun a => coarseRichChemistryDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
    (richChemistryErrorRadius
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    (fun a => exact_vs_coarse_richChemistry_uniformApprox
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s)
    (richChemistryErrorRadius_nonneg
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

noncomputable def exact_vs_coarse_richChemistry_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) : OptimizerWitness A :=
  (exact_vs_coarse_richChemistry_coherent_optimizer_witness
    distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
    radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
    radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
    radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
    radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor s).toOptimizerWitness

/-- Full exact/coarse attractive rich chemistry family: LJ + screened Coulomb + attractive polar surrogate. -/
noncomputable def exactAttractiveRichChemistryDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (ε σ q_i q_j κ w β : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (exactLJScreenedCoulombDecisionProblem distance ε σ q_i q_j κ)
    (exactAttractivePolarSurrogateDecisionProblem distance w β
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor)

noncomputable def coarseAttractiveRichChemistryDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (cutoffLJScreenedCoulombDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC)
    (coarseAttractivePolarSurrogateDecisionProblem distance w β rcCT
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

noncomputable def attractiveRichChemistryErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) : ℝ :=
  richChemistryErrorRadius
    distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
    radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
    radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
    radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
    radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor

theorem exact_vs_coarse_attractiveRichChemistry_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) :
    UniformUtilityApprox
      (exactAttractiveRichChemistryDecisionProblem distance ε σ q_i q_j κ w β
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor)
      (coarseAttractiveRichChemistryDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
      (attractiveRichChemistryErrorRadius
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor) := by
  unfold exactAttractiveRichChemistryDecisionProblem coarseAttractiveRichChemistryDecisionProblem attractiveRichChemistryErrorRadius
  exact sum_uniformApprox
    (exactLJScreenedCoulombDecisionProblem distance ε σ q_i q_j κ)
    (cutoffLJScreenedCoulombDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC)
    (exactAttractivePolarSurrogateDecisionProblem distance w β
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor)
    (coarseAttractivePolarSurrogateDecisionProblem distance w β rcCT
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    (ljScreenedCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j κ rcSC)
    (attractivePolarSurrogateErrorRadius
      distance w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    (exact_vs_cutoff_lj_screened_coulomb_uniformApprox distance ε σ rcLJ q_i q_j κ rcSC)
    (exact_vs_coarse_attractivePolarSurrogate_uniformApprox
      distance w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

theorem attractiveRichChemistryErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ) :
    0 ≤ attractiveRichChemistryErrorRadius
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor := by
  exact richChemistryErrorRadius_nonneg
    distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
    radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
    radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
    radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
    radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor

noncomputable def exact_vs_coarse_attractiveRichChemistry_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) : CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactAttractiveRichChemistryDecisionProblem distance ε σ q_i q_j κ w β
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
    (fun a => coarseAttractiveRichChemistryDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
    (attractiveRichChemistryErrorRadius
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    (fun a => exact_vs_coarse_attractiveRichChemistry_uniformApprox
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s)
    (attractiveRichChemistryErrorRadius_nonneg
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

theorem exact_vs_coarse_attractiveRichChemistry_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactAttractiveRichChemistryDecisionProblem distance ε σ q_i q_j κ w β
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
      (fun a => coarseAttractiveRichChemistryDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
      (attractiveRichChemistryErrorRadius
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
      (fun a => exact_vs_coarse_attractiveRichChemistry_uniformApprox
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s)
      (attractiveRichChemistryErrorRadius_nonneg
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)).exactTopK
      ⊆ (exact_vs_coarse_attractiveRichChemistry_certified_top1
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor s).survivors := by
  simpa [exact_vs_coarse_attractiveRichChemistry_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactAttractiveRichChemistryDecisionProblem distance ε σ q_i q_j κ w β
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
      (fun a => coarseAttractiveRichChemistryDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
      (attractiveRichChemistryErrorRadius
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
      (fun a => exact_vs_coarse_attractiveRichChemistry_uniformApprox
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s)
      (attractiveRichChemistryErrorRadius_nonneg
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
        radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
        radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
        radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

noncomputable def exact_vs_coarse_attractiveRichChemistry_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) : CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactAttractiveRichChemistryDecisionProblem distance ε σ q_i q_j κ w β
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor |>.utility a s)
    (fun a => coarseAttractiveRichChemistryDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor |>.utility a s)
    (attractiveRichChemistryErrorRadius
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)
    (fun a => exact_vs_coarse_attractiveRichChemistry_uniformApprox
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor a s)
    (attractiveRichChemistryErrorRadius_nonneg
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
      radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
      radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
      radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor)

noncomputable def exact_vs_coarse_attractiveRichChemistry_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor : A → S → ℝ)
    (radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor : A → S → ℝ)
    (radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) : OptimizerWitness A :=
  (exact_vs_coarse_attractiveRichChemistry_coherent_optimizer_witness
    distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
    radialExactRecDonor donorExactRecDonor acceptorExactRecDonor
    radialExactLigDonor donorExactLigDonor acceptorExactLigDonor
    radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor
    radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor s).toOptimizerWitness

end RichChemistryApproximation
end Tractability
end DecisionQuotient
