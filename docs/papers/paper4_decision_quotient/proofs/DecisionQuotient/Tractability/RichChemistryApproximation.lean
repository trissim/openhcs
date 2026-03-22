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

universe u v

/-- Exact/coarse additive polar surrogate: bounded contact plus directional H-bond. -/
noncomputable def exactPolarSurrogateDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (w β : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (exactContactDecisionProblem w β distance)
    (directionalHBondDecisionProblem radialExact donorExact acceptorExact)

noncomputable def coarsePolarSurrogateDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (cutoffContactDecisionProblem w β rc distance)
    (directionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse)

noncomputable def polarSurrogateErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) : ℝ :=
  contactCutoffErrorRadius w β rc distance +
    finiteDirectionalHBondErrorRadius
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse

theorem exact_vs_coarse_polarSurrogate_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) :
    UniformUtilityApprox
      (exactPolarSurrogateDecisionProblem distance w β radialExact donorExact acceptorExact)
      (coarsePolarSurrogateDecisionProblem distance w β rc radialCoarse donorCoarse acceptorCoarse)
      (polarSurrogateErrorRadius
        distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse) := by
  unfold exactPolarSurrogateDecisionProblem coarsePolarSurrogateDecisionProblem polarSurrogateErrorRadius
  exact sum_uniformApprox
    (exactContactDecisionProblem w β distance)
    (cutoffContactDecisionProblem w β rc distance)
    (directionalHBondDecisionProblem radialExact donorExact acceptorExact)
    (directionalHBondDecisionProblem radialCoarse donorCoarse acceptorCoarse)
    (contactCutoffErrorRadius w β rc distance)
    (finiteDirectionalHBondErrorRadius
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
    (exact_vs_cutoff_contact_uniformApprox w β rc distance)
    (finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

theorem polarSurrogateErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) :
    0 ≤ polarSurrogateErrorRadius
      distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse := by
  unfold polarSurrogateErrorRadius
  exact add_nonneg
    (contactCutoffErrorRadius_nonneg w β rc distance)
    (finiteDirectionalHBondErrorRadius_nonneg
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

noncomputable def exact_vs_coarse_polarSurrogate_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) : CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactPolarSurrogateDecisionProblem distance w β radialExact donorExact acceptorExact |>.utility a s)
    (fun a => coarsePolarSurrogateDecisionProblem distance w β rc radialCoarse donorCoarse acceptorCoarse |>.utility a s)
    (polarSurrogateErrorRadius
      distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
    (fun a => exact_vs_coarse_polarSurrogate_uniformApprox
      distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)
    (polarSurrogateErrorRadius_nonneg
      distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

theorem exact_vs_coarse_polarSurrogate_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactPolarSurrogateDecisionProblem distance w β radialExact donorExact acceptorExact |>.utility a s)
      (fun a => coarsePolarSurrogateDecisionProblem distance w β rc radialCoarse donorCoarse acceptorCoarse |>.utility a s)
      (polarSurrogateErrorRadius
        distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
      (fun a => exact_vs_coarse_polarSurrogate_uniformApprox
        distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)
      (polarSurrogateErrorRadius_nonneg
        distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)).exactTopK
      ⊆ (exact_vs_coarse_polarSurrogate_certified_top1
        distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s).survivors := by
  simpa [exact_vs_coarse_polarSurrogate_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactPolarSurrogateDecisionProblem distance w β radialExact donorExact acceptorExact |>.utility a s)
      (fun a => coarsePolarSurrogateDecisionProblem distance w β rc radialCoarse donorCoarse acceptorCoarse |>.utility a s)
      (polarSurrogateErrorRadius
        distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
      (fun a => exact_vs_coarse_polarSurrogate_uniformApprox
        distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)
      (polarSurrogateErrorRadius_nonneg
        distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

noncomputable def exact_vs_coarse_polarSurrogate_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) : CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactPolarSurrogateDecisionProblem distance w β radialExact donorExact acceptorExact |>.utility a s)
    (fun a => coarsePolarSurrogateDecisionProblem distance w β rc radialCoarse donorCoarse acceptorCoarse |>.utility a s)
    (polarSurrogateErrorRadius
      distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
    (fun a => exact_vs_coarse_polarSurrogate_uniformApprox
      distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)
    (polarSurrogateErrorRadius_nonneg
      distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

noncomputable def exact_vs_coarse_polarSurrogate_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) : OptimizerWitness A :=
  (exact_vs_coarse_polarSurrogate_coherent_optimizer_witness
    distance w β rc radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s).toOptimizerWitness

/-- Full exact/coarse rich chemistry family: LJ + screened Coulomb + contact + directional H-bond. -/
noncomputable def exactRichChemistryDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (ε σ q_i q_j κ w β : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (exactLJScreenedCoulombDecisionProblem distance ε σ q_i q_j κ)
    (exactPolarSurrogateDecisionProblem distance w β radialExact donorExact acceptorExact)

noncomputable def coarseRichChemistryDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (cutoffLJScreenedCoulombDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC)
    (coarsePolarSurrogateDecisionProblem distance w β rcCT radialCoarse donorCoarse acceptorCoarse)

noncomputable def richChemistryErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) : ℝ :=
  ljScreenedCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j κ rcSC +
    polarSurrogateErrorRadius
      distance w β rcCT radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse

theorem exact_vs_coarse_richChemistry_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) :
    UniformUtilityApprox
      (exactRichChemistryDecisionProblem distance ε σ q_i q_j κ w β radialExact donorExact acceptorExact)
      (coarseRichChemistryDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC w β rcCT radialCoarse donorCoarse acceptorCoarse)
      (richChemistryErrorRadius
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse) := by
  unfold exactRichChemistryDecisionProblem coarseRichChemistryDecisionProblem richChemistryErrorRadius
  exact sum_uniformApprox
    (exactLJScreenedCoulombDecisionProblem distance ε σ q_i q_j κ)
    (cutoffLJScreenedCoulombDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC)
    (exactPolarSurrogateDecisionProblem distance w β radialExact donorExact acceptorExact)
    (coarsePolarSurrogateDecisionProblem distance w β rcCT radialCoarse donorCoarse acceptorCoarse)
    (ljScreenedCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j κ rcSC)
    (polarSurrogateErrorRadius
      distance w β rcCT radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
    (exact_vs_cutoff_lj_screened_coulomb_uniformApprox distance ε σ rcLJ q_i q_j κ rcSC)
    (exact_vs_coarse_polarSurrogate_uniformApprox
      distance w β rcCT radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

theorem richChemistryErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) :
    0 ≤ richChemistryErrorRadius
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse := by
  unfold richChemistryErrorRadius
  exact add_nonneg
    (ljScreenedCoulombCutoffErrorRadius_nonneg distance ε σ rcLJ q_i q_j κ rcSC)
    (polarSurrogateErrorRadius_nonneg
      distance w β rcCT radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

noncomputable def exact_vs_coarse_richChemistry_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) : CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactRichChemistryDecisionProblem distance ε σ q_i q_j κ w β radialExact donorExact acceptorExact |>.utility a s)
    (fun a => coarseRichChemistryDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC w β rcCT radialCoarse donorCoarse acceptorCoarse |>.utility a s)
    (richChemistryErrorRadius
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
    (fun a => exact_vs_coarse_richChemistry_uniformApprox
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)
    (richChemistryErrorRadius_nonneg
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

theorem exact_vs_coarse_richChemistry_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactRichChemistryDecisionProblem distance ε σ q_i q_j κ w β radialExact donorExact acceptorExact |>.utility a s)
      (fun a => coarseRichChemistryDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC w β rcCT radialCoarse donorCoarse acceptorCoarse |>.utility a s)
      (richChemistryErrorRadius
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
      (fun a => exact_vs_coarse_richChemistry_uniformApprox
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)
      (richChemistryErrorRadius_nonneg
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)).exactTopK
      ⊆ (exact_vs_coarse_richChemistry_certified_top1
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s).survivors := by
  simpa [exact_vs_coarse_richChemistry_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactRichChemistryDecisionProblem distance ε σ q_i q_j κ w β radialExact donorExact acceptorExact |>.utility a s)
      (fun a => coarseRichChemistryDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC w β rcCT radialCoarse donorCoarse acceptorCoarse |>.utility a s)
      (richChemistryErrorRadius
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
      (fun a => exact_vs_coarse_richChemistry_uniformApprox
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)
      (richChemistryErrorRadius_nonneg
        distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
        radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

noncomputable def exact_vs_coarse_richChemistry_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) : CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactRichChemistryDecisionProblem distance ε σ q_i q_j κ w β radialExact donorExact acceptorExact |>.utility a s)
    (fun a => coarseRichChemistryDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC w β rcCT radialCoarse donorCoarse acceptorCoarse |>.utility a s)
    (richChemistryErrorRadius
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)
    (fun a => exact_vs_coarse_richChemistry_uniformApprox
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse a s)
    (richChemistryErrorRadius_nonneg
      distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
      radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse)

noncomputable def exact_vs_coarse_richChemistry_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) : OptimizerWitness A :=
  (exact_vs_coarse_richChemistry_coherent_optimizer_witness
    distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s).toOptimizerWitness

end RichChemistryApproximation
end Tractability
end DecisionQuotient
