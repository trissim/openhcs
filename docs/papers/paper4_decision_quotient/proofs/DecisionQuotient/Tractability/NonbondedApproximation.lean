/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/NonbondedApproximation.lean

  Finite-domain exact/coarse approximation for additive nonbonded score families.
-/
import DecisionQuotient.Tractability.LJApproximation
import DecisionQuotient.Tractability.CoulombApproximation
import DecisionQuotient.Tractability.ScreenedCoulombApproximation

namespace DecisionQuotient
namespace Tractability
namespace NonbondedApproximation

open LJApproximation
open CoulombApproximation
open ScreenedCoulombApproximation
open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer

universe u v

/-- Exact additive nonbonded score: LJ plus Coulomb. -/
noncomputable def exactLJCoulombDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ) (ε σ q_i q_j : ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (exactLJDecisionProblem distance ε σ)
    (exactCoulombDecisionProblem q_i q_j distance)

/-- Cutoff additive nonbonded score: cutoff LJ plus cutoff Coulomb. -/
noncomputable def cutoffLJCoulombDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j rcC : ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (cutoffLJDecisionProblem distance ε σ rcLJ)
    (cutoffCoulombDecisionProblem q_i q_j rcC distance)

noncomputable def ljCoulombCutoffErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j rcC : ℝ) : ℝ :=
  ljCutoffErrorRadius distance ε σ rcLJ +
    coulombCutoffErrorRadius q_i q_j rcC distance

theorem exact_vs_cutoff_lj_coulomb_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j rcC : ℝ) :
    UniformUtilityApprox
      (exactLJCoulombDecisionProblem distance ε σ q_i q_j)
      (cutoffLJCoulombDecisionProblem distance ε σ rcLJ q_i q_j rcC)
      (ljCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j rcC) := by
  unfold exactLJCoulombDecisionProblem cutoffLJCoulombDecisionProblem ljCoulombCutoffErrorRadius
  exact sum_uniformApprox
    (exactLJDecisionProblem distance ε σ)
    (cutoffLJDecisionProblem distance ε σ rcLJ)
    (exactCoulombDecisionProblem q_i q_j distance)
    (cutoffCoulombDecisionProblem q_i q_j rcC distance)
    (ljCutoffErrorRadius distance ε σ rcLJ)
    (coulombCutoffErrorRadius q_i q_j rcC distance)
    (exact_vs_cutoff_lj_uniformApprox distance ε σ rcLJ)
    (exact_vs_cutoff_coulomb_uniformApprox q_i q_j rcC distance)

theorem ljCoulombCutoffErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j rcC : ℝ) :
    0 ≤ ljCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j rcC := by
  unfold ljCoulombCutoffErrorRadius
  exact add_nonneg
    (ljCutoffErrorRadius_nonneg distance ε σ rcLJ)
    (coulombCutoffErrorRadius_nonneg q_i q_j rcC distance)

noncomputable def exact_vs_cutoff_lj_coulomb_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j rcC : ℝ) (s : S) :
    CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactLJCoulombDecisionProblem distance ε σ q_i q_j |>.utility a s)
    (fun a => cutoffLJCoulombDecisionProblem distance ε σ rcLJ q_i q_j rcC |>.utility a s)
    (ljCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j rcC)
    (fun a => exact_vs_cutoff_lj_coulomb_uniformApprox distance ε σ rcLJ q_i q_j rcC a s)
    (ljCoulombCutoffErrorRadius_nonneg distance ε σ rcLJ q_i q_j rcC)

theorem exact_vs_cutoff_lj_coulomb_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j rcC : ℝ) (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactLJCoulombDecisionProblem distance ε σ q_i q_j |>.utility a s)
      (fun a => cutoffLJCoulombDecisionProblem distance ε σ rcLJ q_i q_j rcC |>.utility a s)
      (ljCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j rcC)
      (fun a => exact_vs_cutoff_lj_coulomb_uniformApprox distance ε σ rcLJ q_i q_j rcC a s)
      (ljCoulombCutoffErrorRadius_nonneg distance ε σ rcLJ q_i q_j rcC)).exactTopK
      ⊆ (exact_vs_cutoff_lj_coulomb_certified_top1 distance ε σ rcLJ q_i q_j rcC s).survivors := by
  simpa [exact_vs_cutoff_lj_coulomb_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactLJCoulombDecisionProblem distance ε σ q_i q_j |>.utility a s)
      (fun a => cutoffLJCoulombDecisionProblem distance ε σ rcLJ q_i q_j rcC |>.utility a s)
      (ljCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j rcC)
      (fun a => exact_vs_cutoff_lj_coulomb_uniformApprox distance ε σ rcLJ q_i q_j rcC a s)
      (ljCoulombCutoffErrorRadius_nonneg distance ε σ rcLJ q_i q_j rcC)

noncomputable def exact_vs_cutoff_lj_coulomb_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j rcC : ℝ) (s : S) :
    CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactLJCoulombDecisionProblem distance ε σ q_i q_j |>.utility a s)
    (fun a => cutoffLJCoulombDecisionProblem distance ε σ rcLJ q_i q_j rcC |>.utility a s)
    (ljCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j rcC)
    (fun a => exact_vs_cutoff_lj_coulomb_uniformApprox distance ε σ rcLJ q_i q_j rcC a s)
    (ljCoulombCutoffErrorRadius_nonneg distance ε σ rcLJ q_i q_j rcC)

noncomputable def exact_vs_cutoff_lj_coulomb_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j rcC : ℝ) (s : S) :
    OptimizerWitness A :=
  (exact_vs_cutoff_lj_coulomb_coherent_optimizer_witness distance ε σ rcLJ q_i q_j rcC s).toOptimizerWitness

/-- Exact additive nonbonded score: LJ plus screened Coulomb. -/
noncomputable def exactLJScreenedCoulombDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ) (ε σ q_i q_j κ : ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (exactLJDecisionProblem distance ε σ)
    (exactScreenedCoulombDecisionProblem q_i q_j κ distance)

/-- Cutoff additive nonbonded score: cutoff LJ plus cutoff screened Coulomb. -/
noncomputable def cutoffLJScreenedCoulombDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j κ rcSC : ℝ) : DecisionProblem A S :=
  sumDecisionProblems
    (cutoffLJDecisionProblem distance ε σ rcLJ)
    (cutoffScreenedCoulombDecisionProblem q_i q_j κ rcSC distance)

noncomputable def ljScreenedCoulombCutoffErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j κ rcSC : ℝ) : ℝ :=
  ljCutoffErrorRadius distance ε σ rcLJ +
    screenedCoulombCutoffErrorRadius q_i q_j κ rcSC distance

theorem exact_vs_cutoff_lj_screened_coulomb_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j κ rcSC : ℝ) :
    UniformUtilityApprox
      (exactLJScreenedCoulombDecisionProblem distance ε σ q_i q_j κ)
      (cutoffLJScreenedCoulombDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC)
      (ljScreenedCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j κ rcSC) := by
  unfold exactLJScreenedCoulombDecisionProblem cutoffLJScreenedCoulombDecisionProblem ljScreenedCoulombCutoffErrorRadius
  exact sum_uniformApprox
    (exactLJDecisionProblem distance ε σ)
    (cutoffLJDecisionProblem distance ε σ rcLJ)
    (exactScreenedCoulombDecisionProblem q_i q_j κ distance)
    (cutoffScreenedCoulombDecisionProblem q_i q_j κ rcSC distance)
    (ljCutoffErrorRadius distance ε σ rcLJ)
    (screenedCoulombCutoffErrorRadius q_i q_j κ rcSC distance)
    (exact_vs_cutoff_lj_uniformApprox distance ε σ rcLJ)
    (exact_vs_cutoff_screened_coulomb_uniformApprox q_i q_j κ rcSC distance)

theorem ljScreenedCoulombCutoffErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j κ rcSC : ℝ) :
    0 ≤ ljScreenedCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j κ rcSC := by
  unfold ljScreenedCoulombCutoffErrorRadius
  exact add_nonneg
    (ljCutoffErrorRadius_nonneg distance ε σ rcLJ)
    (screenedCoulombCutoffErrorRadius_nonneg q_i q_j κ rcSC distance)

noncomputable def exact_vs_cutoff_lj_screened_coulomb_certified_top1 {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j κ rcSC : ℝ) (s : S) :
    CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactLJScreenedCoulombDecisionProblem distance ε σ q_i q_j κ |>.utility a s)
    (fun a => cutoffLJScreenedCoulombDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC |>.utility a s)
    (ljScreenedCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j κ rcSC)
    (fun a => exact_vs_cutoff_lj_screened_coulomb_uniformApprox distance ε σ rcLJ q_i q_j κ rcSC a s)
    (ljScreenedCoulombCutoffErrorRadius_nonneg distance ε σ rcLJ q_i q_j κ rcSC)

theorem exact_vs_cutoff_lj_screened_coulomb_certified_top1_sound {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j κ rcSC : ℝ) (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactLJScreenedCoulombDecisionProblem distance ε σ q_i q_j κ |>.utility a s)
      (fun a => cutoffLJScreenedCoulombDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC |>.utility a s)
      (ljScreenedCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j κ rcSC)
      (fun a => exact_vs_cutoff_lj_screened_coulomb_uniformApprox distance ε σ rcLJ q_i q_j κ rcSC a s)
      (ljScreenedCoulombCutoffErrorRadius_nonneg distance ε σ rcLJ q_i q_j κ rcSC)).exactTopK
      ⊆ (exact_vs_cutoff_lj_screened_coulomb_certified_top1 distance ε σ rcLJ q_i q_j κ rcSC s).survivors := by
  simpa [exact_vs_cutoff_lj_screened_coulomb_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactLJScreenedCoulombDecisionProblem distance ε σ q_i q_j κ |>.utility a s)
      (fun a => cutoffLJScreenedCoulombDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC |>.utility a s)
      (ljScreenedCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j κ rcSC)
      (fun a => exact_vs_cutoff_lj_screened_coulomb_uniformApprox distance ε σ rcLJ q_i q_j κ rcSC a s)
      (ljScreenedCoulombCutoffErrorRadius_nonneg distance ε σ rcLJ q_i q_j κ rcSC)

noncomputable def exact_vs_cutoff_lj_screened_coulomb_coherent_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j κ rcSC : ℝ) (s : S) :
    CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactLJScreenedCoulombDecisionProblem distance ε σ q_i q_j κ |>.utility a s)
    (fun a => cutoffLJScreenedCoulombDecisionProblem distance ε σ rcLJ q_i q_j κ rcSC |>.utility a s)
    (ljScreenedCoulombCutoffErrorRadius distance ε σ rcLJ q_i q_j κ rcSC)
    (fun a => exact_vs_cutoff_lj_screened_coulomb_uniformApprox distance ε σ rcLJ q_i q_j κ rcSC a s)
    (ljScreenedCoulombCutoffErrorRadius_nonneg distance ε σ rcLJ q_i q_j κ rcSC)

noncomputable def exact_vs_cutoff_lj_screened_coulomb_optimizer_witness {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j κ rcSC : ℝ) (s : S) :
    OptimizerWitness A :=
  (exact_vs_cutoff_lj_screened_coulomb_coherent_optimizer_witness distance ε σ rcLJ q_i q_j κ rcSC s).toOptimizerWitness

end NonbondedApproximation
end Tractability
end DecisionQuotient
