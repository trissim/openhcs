/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/ConformerSupportCoverage.lean

  Runtime-facing theorems for finite conformer support libraries.

  The missing bridge for unknown-conformer docking is not another heuristic seed
  count, but a proof that a deterministic conformer support family is rich enough
  to contain a near-optimal representative. This file packages exactly that idea.

  Core message:

  * if a finite support family epsilon-covers the ambient conformer space, and
  * if the exact utility is Lipschitz on that ambient space,

  then some supported conformer is provably delta-near-optimal with

      delta = L * epsilon.

  This does not yet choose how the runtime constructs the support family. It only
  states what must be true of that family for the blind-conformer pipeline to be
  theorem-faithful.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Fintype.Pi
import Mathlib.Topology.MetricSpace.Lipschitz
import DecisionQuotient.Computation.ArrayDSL
import DecisionQuotient.Tractability.FiniteTopK
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.ConformerSearch
import DecisionQuotient.Tractability.SampledDocking
import DecisionQuotient.Tractability.EnergyRMSDConvergence
import DecisionQuotient.Tractability.GaussianDecayBounds
import DecisionQuotient.Tractability.LJApproximation
import DecisionQuotient.Tractability.CoulombApproximation

namespace DecisionQuotient
namespace Tractability
namespace ConformerSupportCoverage

open SampledDocking
open FiniteTopK
open CoarseApproximation
open ConformerSearch
open EnergyRMSDConvergence
open Computation.ArrayDSL
open GaussianDecayBounds
open LJApproximation
open CoulombApproximation

universe u v

/-- Exact optimality for a utility family on an ambient action space. -/
def IsOptimal {A : Type u} (u : A → ℝ) (a : A) : Prop :=
  ∀ a', u a' ≤ u a

/-- Delta-near-optimality for a utility family on an ambient action space. -/
def IsApproxOptimal {A : Type u} (u : A → ℝ) (δ : ℝ) (a : A) : Prop :=
  ∀ a', u a' ≤ u a + δ

/-- A finite support family epsilon-covers an ambient metric action space when
    every ambient action lies within epsilon of some supported action. -/
def SupportCovers
    {A : Type u} [PseudoMetricSpace A]
    (support : Finset A) (ε : ℝ) : Prop :=
  ∀ a, ∃ b ∈ support, dist a b ≤ ε

/-- Every action in the feasible set lies within epsilon of some supported action. -/
def SupportCoversOn
    {A : Type u} [PseudoMetricSpace A]
    (support : Finset A) (ε : ℝ) (feasible : Set A) : Prop :=
  ∀ a ∈ feasible, ∃ b ∈ support, dist a b ≤ ε

theorem optimal_isApproxOptimal_zero
    {A : Type u}
    (u : A → ℝ)
    {a : A}
    (hOpt : IsOptimal u a) :
    IsApproxOptimal u 0 a := by
  intro a'
  simpa using hOpt a'

theorem approxOptimal_mono
    {A : Type u}
    (u : A → ℝ)
    {δ₁ δ₂ : ℝ}
    (hδ : δ₁ ≤ δ₂)
    {a : A}
    (hApprox : IsApproxOptimal u δ₁ a) :
    IsApproxOptimal u δ₂ a := by
  intro a'
  have h := hApprox a'
  linarith

theorem supportCovers_mono
    {A : Type u} [PseudoMetricSpace A] [DecidableEq A]
    {support₁ support₂ : Finset A}
    {ε : ℝ}
    (hSubset : support₁ ⊆ support₂)
    (hCover : SupportCovers support₁ ε) :
    SupportCovers support₂ ε := by
  intro a
  rcases hCover a with ⟨b, hb, hdist⟩
  exact ⟨b, hSubset hb, hdist⟩

/-- Covering plus a Lipschitz utility implies the support contains a delta-near-
    optimal representative of every exact optimum, with `delta = L * epsilon`. -/
theorem lipschitz_cover_yields_supported_approxOptimal
    {A : Type u} [PseudoMetricSpace A] [DecidableEq A]
    (u : A → ℝ)
    (L : ℝ)
    (support : Finset A)
    (ε : ℝ)
    (hCover : SupportCovers support ε)
    (hL : 0 ≤ L)
    (hε : 0 ≤ ε)
    {aStar : A}
    (hOpt : IsOptimal u aStar)
    (hLip : ∀ x y, |u x - u y| ≤ L * dist x y) :
    ∃ b, b ∈ support ∧ IsApproxOptimal u (L * ε) b := by
  rcases hCover aStar with ⟨b, hb, hdist⟩
  refine ⟨b, hb, ?_⟩
  intro a
  have hOptA : u a ≤ u aStar := hOpt a
  have hAbs : |u aStar - u b| ≤ L * dist aStar b := hLip aStar b
  have hDistBound : L * dist aStar b ≤ L * ε := by
    exact mul_le_mul_of_nonneg_left hdist hL
  have hAB : u aStar ≤ u b + (L : ℝ) * ε := by
    have hAbs' := abs_le.mp hAbs
    linarith
  linarith

/-- Runtime-facing version for a sampled action family on a fixed state. If the
    sampled support epsilon-covers the ambient action space and the exact utility
    is L-Lipschitz in the action argument, then some supported action is delta-
    near-optimal for that state. -/
theorem sampledActionFamily_cover_yields_supported_approxOptimal
    {A : Type u} {S : Type v}
    [PseudoMetricSpace A] [DecidableEq A]
    (dp : DecisionProblem A S)
    (F : SampledActionFamily A)
    (s : S)
    (L : ℝ)
    (ε : ℝ)
    (hCover : SupportCovers F.support ε)
    (hL : 0 ≤ L)
    (hε : 0 ≤ ε)
    {aStar : A}
    (hOpt : IsOptimal (fun a => dp.utility a s) aStar)
    (hLip : ∀ a a', |dp.utility a s - dp.utility a' s| ≤ L * dist a a') :
    ∃ a : SupportedAction F,
      IsApproxOptimal (fun a' => dp.utility a' s) (L * ε) a.1 := by
  rcases lipschitz_cover_yields_supported_approxOptimal
      (u := fun a => dp.utility a s)
      (L := L)
      (support := F.support)
      (ε := ε)
      hCover hL hε hOpt hLip with ⟨b, hb, hApprox⟩
  exact ⟨⟨b, hb⟩, hApprox⟩

theorem optimal_mem_topKSet_one
    {A : Type u} [Fintype A] [DecidableEq A]
    (u : A → ℝ)
    {a : A}
    (hOpt : IsOptimal u a) :
    a ∈ topKSet u 1 := by
  rw [mem_topKSet_iff]
  have hCountZero : strictBetterCount u a = 0 := by
    unfold strictBetterCount
    apply Finset.card_eq_zero.mpr
    rw [Finset.filter_eq_empty_iff]
    intro b hb
    exact not_lt_of_ge (hOpt b)
  rw [hCountZero]
  simp

theorem restrictedDecisionProblem_exists_opt
    {A : Type u} {S : Type v} [DecidableEq A]
    (dp : DecisionProblem A S)
    (F : SampledActionFamily A)
    (s : S) :
    ∃ a : SupportedAction F, a ∈ (restrictedDecisionProblem dp F).Opt s := by
  rcases F.nonempty with ⟨a0, ha0⟩
  obtain ⟨aBest, haBest, hBest⟩ :=
    Finset.exists_max_image F.support (fun a => dp.utility a s) ⟨a0, ha0⟩
  refine ⟨⟨aBest, haBest⟩, ?_⟩
  intro a'
  exact hBest a'.1 a'.2

theorem sampledActionFamily_cover_yields_restricted_opt_approxAmbient
    {A : Type u} {S : Type v}
    [PseudoMetricSpace A] [DecidableEq A]
    (dp : DecisionProblem A S)
    (F : SampledActionFamily A)
    (s : S)
    (L : ℝ)
    (ε : ℝ)
    (hCover : SupportCovers F.support ε)
    (hL : 0 ≤ L)
    (hε : 0 ≤ ε)
    {aStar : A}
    (hOpt : IsOptimal (fun a => dp.utility a s) aStar)
    (hLip : ∀ a a', |dp.utility a s - dp.utility a' s| ≤ L * dist a a') :
    ∃ a : SupportedAction F,
      a ∈ (restrictedDecisionProblem dp F).Opt s ∧
      IsApproxOptimal (fun a' => dp.utility a' s) (L * ε) a.1 := by
  rcases sampledActionFamily_cover_yields_supported_approxOptimal
      (dp := dp)
      (F := F)
      (s := s)
      (L := L)
      (ε := ε)
      hCover hL hε hOpt hLip with ⟨aCov, hCovApprox⟩
  rcases restrictedDecisionProblem_exists_opt dp F s with ⟨aBest, hBestOpt⟩
  refine ⟨aBest, hBestOpt, ?_⟩
  intro a
  have hCov : dp.utility a s ≤ dp.utility aCov.1 s + L * ε := hCovApprox a
  have hBestGe : dp.utility aCov.1 s ≤ dp.utility aBest.1 s := hBestOpt aCov
  linarith

theorem sampledActionFamily_cover_and_uniformApprox_yields_near_opt_in_runtime_support
    {A : Type u} {S : Type v}
    [PseudoMetricSpace A] [DecidableEq A]
    (exactDP coarseDP : DecisionProblem A S)
    (F : SampledActionFamily A)
    (s : S)
    (L δ ε : ℝ)
    [LinearOrder (SupportedAction F)]
    (hCover : SupportCovers F.support ε)
    (hL : 0 ≤ L)
    (hε : 0 ≤ ε)
    {aStar : A}
    (hOpt : IsOptimal (fun a => exactDP.utility a s) aStar)
    (hLip : ∀ a a', |exactDP.utility a s - exactDP.utility a' s| ≤ L * dist a a')
    (hApprox : ∀ a : SupportedAction F,
      |(restrictedDecisionProblem exactDP F).utility a s -
        (restrictedDecisionProblem coarseDP F).utility a s| ≤ δ)
    (hδ : 0 ≤ δ) :
    ∃ a : SupportedAction F,
      a ∈ (coherent_optimizer_witness_of_uniformApprox_top1
        (fun a => (restrictedDecisionProblem exactDP F).utility a s)
        (fun a => (restrictedDecisionProblem coarseDP F).utility a s)
        δ hApprox hδ).belief.selection.support
      ∧ IsApproxOptimal (fun a' => exactDP.utility a' s) (L * ε) a.1 := by
  rcases sampledActionFamily_cover_yields_restricted_opt_approxAmbient
      (dp := exactDP)
      (F := F)
      (s := s)
      (L := L)
      (ε := ε)
      hCover hL hε hOpt hLip with ⟨aBest, hBestOpt, hApproxAmbient⟩
  refine ⟨aBest, ?_, hApproxAmbient⟩
  have hTop : aBest ∈ topKSet
      (fun a : SupportedAction F => (restrictedDecisionProblem exactDP F).utility a s) 1 := by
    exact optimal_mem_topKSet_one _ hBestOpt
  exact coherent_uniformApprox_exactTop1_subset_support
      (fun a : SupportedAction F => (restrictedDecisionProblem exactDP F).utility a s)
      (fun a : SupportedAction F => (restrictedDecisionProblem coarseDP F).utility a s)
      δ hApprox hδ hTop

/-- RMSD-specific covering predicate for finite conformer libraries. -/
def RMSDSupportCovers
    {n : ℕ} [DecidableEq (CoordSet n)]
    (library : Finset (CoordSet n)) (ε : ℝ) : Prop :=
  ∀ x, ∃ y, y ∈ library ∧ rmsd x y ≤ ε

/-- Energy is L-Lipschitz with respect to RMSD. -/
def RMSDLipschitzEnergy
    {n : ℕ}
    (energy : CoordSet n → ℝ) (L : ℝ) : Prop :=
  ∀ x y, |energy x - energy y| ≤ L * rmsd x y

theorem rmsdSupportCovers_mono
    {n : ℕ} [DecidableEq (CoordSet n)]
    {library₁ library₂ : Finset (CoordSet n)}
    {ε : ℝ}
    (hSubset : library₁ ⊆ library₂)
    (hCover : RMSDSupportCovers library₁ ε) :
    RMSDSupportCovers library₂ ε := by
  intro x
  rcases hCover x with ⟨y, hy, hrmsd⟩
  exact ⟨y, hSubset hy, hrmsd⟩

/-- If a deterministic conformer library epsilon-covers the ambient conformer
    space in RMSD and the exact conformer energy is RMSD-Lipschitz, then the
    library contains a delta-near-optimal conformer with `delta = L * epsilon`. -/
theorem rmsd_cover_yields_library_approxOptimal
    {n : ℕ} [DecidableEq (CoordSet n)]
    (energy : CoordSet n → ℝ)
    (library : Finset (CoordSet n))
    (L ε : ℝ)
    (hL : 0 ≤ L)
    (hCover : RMSDSupportCovers library ε)
    (hε : 0 ≤ ε)
    {xStar : CoordSet n}
    (hOpt : IsOptimal energy xStar)
    (hLip : RMSDLipschitzEnergy energy L) :
    ∃ y, y ∈ library ∧ IsApproxOptimal energy (L * ε) y := by
  rcases hCover xStar with ⟨y, hy, hrmsd⟩
  refine ⟨y, hy, ?_⟩
  intro x
  have hOptX : energy x ≤ energy xStar := hOpt x
  have hAbs : |energy xStar - energy y| ≤ L * rmsd xStar y := hLip xStar y
  have hDistBound : L * rmsd xStar y ≤ L * ε := by
    exact mul_le_mul_of_nonneg_left hrmsd hL
  have hNear : energy xStar ≤ energy y + L * ε := by
    have hAbs' := abs_le.mp hAbs
    linarith
  linarith

/-- Geometry-energy bridge for conformer libraries: if a finite conformer
    library RMSD-covers the ambient conformer space with radius `εCover`, the
    exact conformer energy is RMSD-Lipschitz with constant `L`, and the induced
    energy slack `L * εCover` lies below the quadratic-basin target-energy
    threshold for RMSD tolerance `εTarget`, then the library contains a conformer
    within `εTarget` RMSD of the optimal conformer `xStar`.

    This is the theorem-facing bridge from support coverage plus energy control
    back to geometry. It is exactly the missing connective tissue needed when the
    runtime wants `target_rmsd` to mean more than mere support adequacy. -/
theorem rmsd_cover_yields_library_rmsd_target
    {n : ℕ} [DecidableEq (CoordSet n)]
    (energy : CoordSet n → ℝ)
    (library : Finset (CoordSet n))
    (L εCover εTarget : ℝ)
    (hL : 0 ≤ L)
    (hCover : RMSDSupportCovers library εCover)
    (hεCover : 0 ≤ εCover)
    {xStar : CoordSet n}
    (hOpt : IsOptimal (fun x => -energy x) xStar)
    (hLip : RMSDLipschitzEnergy energy L)
    (basin : CertifiedQuadraticBasin energy xStar)
    (hn : 0 < n)
    (hεTarget : 0 ≤ εTarget)
    (hGapBudget : L * εCover ≤ targetEnergyGap basin.μ n εTarget) :
    ∃ y, y ∈ library ∧ rmsd y xStar ≤ εTarget := by
  have hLipNeg : RMSDLipschitzEnergy (fun x => -energy x) L := by
    intro x y
    calc
      |(-energy x) - (-energy y)| = |energy y - energy x| := by ring_nf
      _ = |energy x - energy y| := abs_sub_comm _ _
      _ ≤ L * rmsd x y := hLip x y
  rcases rmsd_cover_yields_library_approxOptimal
      (energy := fun x => -energy x)
      (library := library)
      (L := L)
      (ε := εCover)
      hL hCover hεCover hOpt hLipNeg with ⟨y, hy, hApprox⟩
  have hNearGap : energy y - energy xStar ≤ L * εCover := by
    have hApproxStar : -energy xStar ≤ -energy y + L * εCover := hApprox xStar
    linarith
  refine ⟨y, hy, ?_⟩
  apply rmsd_le_of_energyGap_le_target energy xStar y basin hn εTarget hεTarget
  exact le_trans hNearGap hGapBudget

/-- Any conformer whose negative energy is `δ`-approx-optimal around a certified
    optimum `xStar` inherits an RMSD target once that same `δ` lies below the
    quadratic-basin target-energy threshold. This is the direct geometry-energy
    bridge for a concrete supported witness, independent of how that witness was
    constructed. -/
theorem approxOptimal_negEnergy_yields_rmsd_target
    {n : ℕ} [DecidableEq (CoordSet n)]
    (energy : CoordSet n → ℝ)
    {xStar y : CoordSet n}
    (δ εTarget : ℝ)
    (hOpt : IsOptimal (fun x => -energy x) xStar)
    (hApprox : IsApproxOptimal (fun x => -energy x) δ y)
    (basin : CertifiedQuadraticBasin energy xStar)
    (hn : 0 < n)
    (hεTarget : 0 ≤ εTarget)
    (hGapBudget : δ ≤ targetEnergyGap basin.μ n εTarget) :
    rmsd y xStar ≤ εTarget := by
  have hApproxStar : -energy xStar ≤ -energy y + δ := hApprox xStar
  have hNearGap : energy y - energy xStar ≤ δ := by
    linarith
  apply rmsd_le_of_energyGap_le_target energy xStar y basin hn εTarget hεTarget
  exact le_trans hNearGap hGapBudget

/-- If a finite conformer library contains any `δ`-approx-optimal witness and that
    same `δ` is small enough for the certified quadratic basin around the optimum,
    then the library already contains a conformer within the requested RMSD target. -/
theorem supported_approxOptimal_yields_library_rmsd_target
    {n : ℕ} [DecidableEq (CoordSet n)]
    (energy : CoordSet n → ℝ)
    (library : Finset (CoordSet n))
    {xStar : CoordSet n}
    (δ εTarget : ℝ)
    (hOpt : IsOptimal (fun x => -energy x) xStar)
    (hWitness : ∃ y, y ∈ library ∧ IsApproxOptimal (fun x => -energy x) δ y)
    (basin : CertifiedQuadraticBasin energy xStar)
    (hn : 0 < n)
    (hεTarget : 0 ≤ εTarget)
    (hGapBudget : δ ≤ targetEnergyGap basin.μ n εTarget) :
    ∃ y, y ∈ library ∧ rmsd y xStar ≤ εTarget := by
  rcases hWitness with ⟨y, hy, hApprox⟩
  refine ⟨y, hy, ?_⟩
  exact approxOptimal_negEnergy_yields_rmsd_target
    energy δ εTarget hOpt hApprox basin hn hεTarget hGapBudget

/-- The exact winner of a finite conformer library is at least as good as any
    supported `δ`-approx-optimal witness, so it inherits the same `δ`-near-optimal
    guarantee. This is the proof-side winner-to-witness connection needed by the
    runtime after exact rescoring of a certified support family. -/
theorem exact_library_winner_inherits_approxOptimal_of_supportedWitness
    {n : ℕ} [DecidableEq (CoordSet n)]
    (energy : CoordSet n → ℝ)
    (library : Finset (CoordSet n))
    {winner witness : CoordSet n}
    (δ : ℝ)
    (hWitnessMem : witness ∈ library)
    (hWitnessApprox : IsApproxOptimal (fun x => -energy x) δ witness)
    (hWinnerMem : winner ∈ library)
    (hWinnerBest : ∀ z, z ∈ library → energy winner ≤ energy z) :
    IsApproxOptimal (fun x => -energy x) δ winner := by
  intro x
  have hApproxWitness : -energy x ≤ -energy witness + δ := hWitnessApprox x
  have hWinnerLeWitness : energy winner ≤ energy witness := hWinnerBest witness hWitnessMem
  linarith

/-- End-to-end geometry-energy theorem on a finite conformer library: if the
    library RMSD-covers conformer space well enough, and the exact rescored winner
    of that library is selected, then that returned winner satisfies the requested
    RMSD target. This is the theorem-side version of the conformer-active returned
    pose contract before runtime witness packaging. -/
theorem exact_library_winner_of_cover_yields_rmsd_target
    {n : ℕ} [DecidableEq (CoordSet n)]
    (energy : CoordSet n → ℝ)
    (library : Finset (CoordSet n))
    (L εCover εTarget : ℝ)
    (hL : 0 ≤ L)
    (hCover : RMSDSupportCovers library εCover)
    (hεCover : 0 ≤ εCover)
    {xStar winner : CoordSet n}
    (hOpt : IsOptimal (fun x => -energy x) xStar)
    (hLip : RMSDLipschitzEnergy energy L)
    (hWinnerMem : winner ∈ library)
    (hWinnerBest : ∀ z, z ∈ library → energy winner ≤ energy z)
    (basin : CertifiedQuadraticBasin energy xStar)
    (hn : 0 < n)
    (hεTarget : 0 ≤ εTarget)
    (hGapBudget : L * εCover ≤ targetEnergyGap basin.μ n εTarget) :
    rmsd winner xStar ≤ εTarget := by
  rcases rmsd_cover_yields_library_approxOptimal
      (energy := fun x => -energy x)
      (library := library)
      (L := L)
      (ε := εCover)
      hL hCover hεCover hOpt
      (by
        intro x y
        calc
          |(-energy x) - (-energy y)| = |energy y - energy x| := by ring_nf
          _ = |energy x - energy y| := abs_sub_comm _ _
          _ ≤ L * rmsd x y := hLip x y)
      with ⟨witness, hWitnessMem, hWitnessApprox⟩
  have hWinnerApprox : IsApproxOptimal (fun x => -energy x) (L * εCover) winner :=
    exact_library_winner_inherits_approxOptimal_of_supportedWitness
      energy library (L * εCover) hWitnessMem hWitnessApprox hWinnerMem hWinnerBest
  exact approxOptimal_negEnergy_yields_rmsd_target
    energy (L * εCover) εTarget hOpt hWinnerApprox basin hn hεTarget hGapBudget

/-- The exact rescored winner of a covered finite conformer library has energy gap
    at most `L * εCover` above the optimal conformer. This is the energy-gap form
    of `exact_library_winner_of_cover_yields_rmsd_target`, useful when composing
    winner guarantees with ambiguity-band budgets. -/
theorem exact_library_winner_energy_gap_of_cover_le
    {n : ℕ} [DecidableEq (CoordSet n)]
    (energy : CoordSet n → ℝ)
    (library : Finset (CoordSet n))
    (L εCover : ℝ)
    (hL : 0 ≤ L)
    (hCover : RMSDSupportCovers library εCover)
    (hεCover : 0 ≤ εCover)
    {xStar winner : CoordSet n}
    (hOpt : IsOptimal (fun x => -energy x) xStar)
    (hLip : RMSDLipschitzEnergy energy L)
    (hWinnerMem : winner ∈ library)
    (hWinnerBest : ∀ z, z ∈ library → energy winner ≤ energy z) :
    energy winner - energy xStar ≤ L * εCover := by
  rcases rmsd_cover_yields_library_approxOptimal
      (energy := fun x => -energy x)
      (library := library)
      (L := L)
      (ε := εCover)
      hL hCover hεCover hOpt
      (by
        intro x y
        calc
          |(-energy x) - (-energy y)| = |energy y - energy x| := by ring_nf
          _ = |energy x - energy y| := abs_sub_comm _ _
          _ ≤ L * rmsd x y := hLip x y)
      with ⟨witness, hWitnessMem, hWitnessApprox⟩
  have hWinnerApprox : IsApproxOptimal (fun x => -energy x) (L * εCover) winner :=
    exact_library_winner_inherits_approxOptimal_of_supportedWitness
      energy library (L * εCover) hWitnessMem hWitnessApprox hWinnerMem hWinnerBest
  have hApproxStar : -energy xStar ≤ -energy winner + L * εCover := hWinnerApprox xStar
  linarith

/-- Runtime-facing bridge: an RMSD target `ε` induces an energy-resolution budget
    `L * ε` whenever the exact energy is `L`-Lipschitz with respect to RMSD. This
    is the direct theorem used to derive certified score-resolution budgets from
    `target_rmsd`. -/
theorem energyBudget_of_rmsdTarget
    {n : ℕ}
    (energy : CoordSet n → ℝ)
    (L ε : ℝ)
    (hL : 0 ≤ L)
    (hLip : RMSDLipschitzEnergy energy L)
    {x y : CoordSet n}
    (hrmsd : rmsd x y ≤ ε) :
    |energy x - energy y| ≤ L * ε := by
  have hAbs : |energy x - energy y| ≤ L * rmsd x y := hLip x y
  exact le_trans hAbs (mul_le_mul_of_nonneg_left hrmsd hL)

/-- A finite support family of cell centers epsilon-covers parameter space when
    every parameter point lies in at least one supported hypercube cell. -/
def HypercubeSupportCovers
    (n : ℕ)
    (support : Finset (Fin n → ℝ))
    (halfWidths : Fin n → ℝ) : Prop :=
  ∀ p : Fin n → ℝ, ∃ center, center ∈ support ∧ ∀ i, |p i - center i| ≤ halfWidths i

/-- Box-restricted hypercube coverage: every point in the declared parameter box is
    covered by some supported hypercube cell. This is the theorem-facing notion for
    bounded torsion spaces such as `[-π, π]^n`. -/
def HypercubeSupportCoversOnBox
    (n : ℕ)
    (support : Finset (Fin n → ℝ))
    (lower upper halfWidths : Fin n → ℝ) : Prop :=
  ∀ p : Fin n → ℝ,
    (∀ i, lower i ≤ p i ∧ p i ≤ upper i) →
    ∃ center, center ∈ support ∧ ∀ i, |p i - center i| ≤ halfWidths i

/-- Deterministic tensor-product support from per-coordinate center sets. -/
noncomputable def coordinateCenterSupport
    (n : ℕ)
    (centers : Fin n → Finset ℝ) : Finset (Fin n → ℝ) :=
  Fintype.piFinset centers

theorem mem_coordinateCenterSupport_iff
    (n : ℕ)
    (centers : Fin n → Finset ℝ)
    (f : Fin n → ℝ) :
    f ∈ coordinateCenterSupport n centers ↔ ∀ i, f i ∈ centers i := by
  unfold coordinateCenterSupport
  simpa using (Fintype.mem_piFinset : f ∈ Fintype.piFinset centers ↔ ∀ i, f i ∈ centers i)

theorem coordinateCenterSupport_card
    (n : ℕ)
    (centers : Fin n → Finset ℝ) :
    (coordinateCenterSupport n centers).card = ∏ i, (centers i).card := by
  unfold coordinateCenterSupport
  simpa using (Fintype.card_piFinset centers)

theorem coordinateCenterSupport_nonempty
    (n : ℕ)
    (centers : Fin n → Finset ℝ)
    (hNonempty : ∀ i, (centers i).Nonempty) :
    (coordinateCenterSupport n centers).Nonempty := by
  unfold coordinateCenterSupport
  exact (Fintype.piFinset_nonempty).2 hNonempty

/-- Coordinatewise interval covering data. -/
def CoordinatewiseIntervalCover
    (n : ℕ)
    (centers : Fin n → Finset ℝ)
    (lower upper halfWidths : Fin n → ℝ) : Prop :=
  ∀ i x, lower i ≤ x → x ≤ upper i → ∃ c, c ∈ centers i ∧ |x - c| ≤ halfWidths i

theorem hypercubeSupportCovers_mono
    (n : ℕ)
    {support₁ support₂ : Finset (Fin n → ℝ)}
    {halfWidths : Fin n → ℝ}
    (hSubset : support₁ ⊆ support₂)
    (hCover : HypercubeSupportCovers n support₁ halfWidths) :
    HypercubeSupportCovers n support₂ halfWidths := by
  intro p
  rcases hCover p with ⟨center, hcenter, hcell⟩
  exact ⟨center, hSubset hcenter, hcell⟩

theorem hypercubeSupportCoversOnBox_mono
    (n : ℕ)
    {support₁ support₂ : Finset (Fin n → ℝ)}
    {lower upper halfWidths : Fin n → ℝ}
    (hSubset : support₁ ⊆ support₂)
    (hCover : HypercubeSupportCoversOnBox n support₁ lower upper halfWidths) :
    HypercubeSupportCoversOnBox n support₂ lower upper halfWidths := by
  intro p hp
  rcases hCover p hp with ⟨center, hcenter, hcell⟩
  exact ⟨center, hSubset hcenter, hcell⟩

theorem coordinatewise_cover_yields_hypercubeSupportOnBox
    (n : ℕ)
    (centers : Fin n → Finset ℝ)
    (lower upper halfWidths : Fin n → ℝ)
    (hCover : CoordinatewiseIntervalCover n centers lower upper halfWidths) :
    HypercubeSupportCoversOnBox n (coordinateCenterSupport n centers) lower upper halfWidths := by
  intro p hp
  choose c hcMem hcDist using
    (fun i => hCover i (p i) (hp i).1 (hp i).2)
  let center : Fin n → ℝ := fun i => c i
  refine ⟨center, ?_, ?_⟩
  · rw [mem_coordinateCenterSupport_iff]
    intro i
    exact hcMem i
  · intro i
    simpa [center] using hcDist i

theorem hypercubeSupportCoversOnBox_of_cover_through_support
    (n : ℕ)
    (support₁ support₂ : Finset (Fin n → ℝ))
    (lower upper halfWidths₁ halfWidths₂ : Fin n → ℝ)
    (hCover₁ : HypercubeSupportCoversOnBox n support₁ lower upper halfWidths₁)
    (hRefine : ∀ center,
      center ∈ support₁ →
      ∃ lifted, lifted ∈ support₂ ∧ ∀ i, |center i - lifted i| ≤ halfWidths₂ i) :
    HypercubeSupportCoversOnBox n support₂ lower upper (fun i => halfWidths₁ i + halfWidths₂ i) := by
  intro p hp
  rcases hCover₁ p hp with ⟨center, hcenter, hcell⟩
  rcases hRefine center hcenter with ⟨lifted, hlifted, hrefine⟩
  refine ⟨lifted, hlifted, ?_⟩
  intro i
  calc
    |p i - lifted i| = |(p i - center i) + (center i - lifted i)| := by ring_nf
    _ ≤ |p i - center i| + |center i - lifted i| := abs_add_le _ _
    _ ≤ halfWidths₁ i + halfWidths₂ i := add_le_add (hcell i) (hrefine i)

/-- Deterministic six-dimensional rigid-pose seed support built as a tensor
    product of per-coordinate center sets. The intended coordinates are three
    translation axes and three rotation parameters. -/
noncomputable def pocketGuidedRigidSeedSupport
    (centers : Fin 6 → Finset ℝ) : Finset (Fin 6 → ℝ) :=
  coordinateCenterSupport 6 centers

/-- Sampler-side coverage theorem for the current rigid pocket-guided seed family:
    if each of the six coordinate center sets covers its interval with certified
    half-width, then the full tensor-product seed family covers the entire rigid
    search box. This is the finite SE(3)-net hypothesis the runtime must satisfy
    before the returned-pose RMSD theorems can fire. -/
theorem pocketGuidedRigidSeedSupport_cover_on_box
    (centers : Fin 6 → Finset ℝ)
    (lower upper halfWidths : Fin 6 → ℝ)
    (hCover : CoordinatewiseIntervalCover 6 centers lower upper halfWidths) :
    HypercubeSupportCoversOnBox 6
      (pocketGuidedRigidSeedSupport centers)
      lower upper halfWidths := by
  simpa [pocketGuidedRigidSeedSupport] using
    coordinatewise_cover_yields_hypercubeSupportOnBox 6 centers lower upper halfWidths hCover

/-- Sampled-action-family wrapper for `pocketGuidedRigidSeedSupport_cover_on_box`.
    If the runtime seed family support is exactly the deterministic rigid seed
    tensor product, the sampled family inherits the same certified box cover. -/
theorem sampledPocketGuidedRigidSeedFamily_cover_on_box
    (F : SampledActionFamily (Fin 6 → ℝ))
    (centers : Fin 6 → Finset ℝ)
    (lower upper halfWidths : Fin 6 → ℝ)
    (hSupport : F.support = pocketGuidedRigidSeedSupport centers)
    (hCover : CoordinatewiseIntervalCover 6 centers lower upper halfWidths) :
    HypercubeSupportCoversOnBox 6 F.support lower upper halfWidths := by
  simpa [hSupport] using
    pocketGuidedRigidSeedSupport_cover_on_box centers lower upper halfWidths hCover

/-- The current runtime rigid-seed family: an arbitrary finite translation subset
    crossed with the fixed `quaternionDictionary8` orientation dictionary. This
    matches the implementation shape used by `formal_sampling.py` after the
    translation lattice is downselected to the requested seed budget. -/
noncomputable def translationSubsetQuaternionDictionary8Support
    (translationSupport : Finset (Fin 3 → ℝ)) : Finset ((Fin 3 → ℝ) × Fin 8) :=
  translationSupport.product (Finset.univ : Finset (Fin 8))

/-- Concrete low-level witness predicate for the current runtime dictionary:
    one of the first four quaternion dictionary entries matches a coordinate of the
    target unit quaternion with absolute value at least `1/2`. -/
def quaternionDictionary8CoordinateWitness
    (q : MDArray 4) (k : Fin 8) : Prop :=
  ∃ h : k.1 < 4, 1 / 2 ≤ |q ⟨k.1, h⟩|

theorem quaternionDictionary8_signed_distance_le_one_of_coordinate_witness
    (q : MDArray 4)
    (hNorm : Computation.ArrayDSL.norm q = 1)
    (k : Fin 8)
    (hWitness : quaternionDictionary8CoordinateWitness q k) :
    min ‖q - quaternionDictionary8 k‖ ‖q + quaternionDictionary8 k‖ ≤ 1 := by
  rcases hWitness with ⟨hk, hCoord⟩
  let i : Fin 4 := ⟨k.1, hk⟩
  have hBasis : quaternionDictionary8 k = EuclideanSpace.single i (1 : ℝ) := by
    fin_cases k
    · simpa [i] using quaternionDictionary8_basis0_eq
    · simpa [i] using quaternionDictionary8_basis1_eq
    · simpa [i] using quaternionDictionary8_basis2_eq
    · simpa [i] using quaternionDictionary8_basis3_eq
    · simp at hk
    · simp at hk
    · simp at hk
    · simp at hk
  have hDistBasis :
      min ‖q - EuclideanSpace.single i (1 : ℝ)‖ ‖q + EuclideanSpace.single i (1 : ℝ)‖ ≤ 1 := by
    exact unit_norm_mdarray4_dist_to_basis_or_negBasis_le_one_of_coordinate_abs_ge_half q i hNorm (by simpa [i] using hCoord)
  simpa [hBasis] using hDistBasis

theorem quaternionDictionary8_contains_coordinate_witness
    (q : MDArray 4)
    (hNorm : Computation.ArrayDSL.norm q = 1) :
    ∃ k : Fin 8,
      k ∈ (Finset.univ : Finset (Fin 8)) ∧ quaternionDictionary8CoordinateWitness q k := by
  rcases unit_norm_mdarray4_has_coordinate_abs_ge_half q hNorm with ⟨i, hi⟩
  refine ⟨⟨i.1, by omega⟩, by simp, ?_⟩
  refine ⟨i.2, ?_⟩
  simpa using hi

theorem quaternionDictionary8_contains_coordinate_and_signed_distance_witness
    (q : MDArray 4)
    (hNorm : Computation.ArrayDSL.norm q = 1) :
    ∃ k : Fin 8,
      k ∈ (Finset.univ : Finset (Fin 8)) ∧
      quaternionDictionary8CoordinateWitness q k ∧
      min ‖q - quaternionDictionary8 k‖ ‖q + quaternionDictionary8 k‖ ≤ 1 := by
  rcases quaternionDictionary8_contains_coordinate_witness q hNorm with ⟨k, hkMem, hkCoord⟩
  refine ⟨k, hkMem, hkCoord, ?_⟩
  exact quaternionDictionary8_signed_distance_le_one_of_coordinate_witness q hNorm k hkCoord

theorem quaternionDictionary8_contains_signed_distance_witness
    (q : MDArray 4)
    (hNorm : Computation.ArrayDSL.norm q = 1) :
    ∃ k : Fin 8,
      k ∈ (Finset.univ : Finset (Fin 8)) ∧
      min ‖q - quaternionDictionary8 k‖ ‖q + quaternionDictionary8 k‖ ≤ 1 := by
  rcases quaternionDictionary8_contains_coordinate_and_signed_distance_witness q hNorm with
      ⟨k, hkMem, _hkCoord, hkDist⟩
  exact ⟨k, hkMem, hkDist⟩

theorem quaternionDictionary8_contains_rotation_witness
    {Ω : Type u}
    (rotationWitness : Ω → Fin 8 → Prop)
    {ω : Ω}
    {k : Fin 8}
    (hk : rotationWitness ω k) :
    ∃ j : Fin 8, j ∈ (Finset.univ : Finset (Fin 8)) ∧ rotationWitness ω j := by
  exact ⟨k, by simp, hk⟩

/-- If the runtime translation subset carries an explicit box-cover certificate
    and the fixed quaternion dictionary carries an explicit orientation witness
    predicate, then the current translation-subset × quaternionDictionary8 rigid
    family contains a pose whose coordinates lie within the requested RMSD cover
    radius of the target rigid pose. This is the theorem-backed bridge for the
    sampler shape the runtime actually uses today. -/
theorem translationSubsetQuaternionDictionary8Support_contains_rmsd_witness_of_component_covers
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (translationSupport : Finset (Fin 3 → ℝ))
    (lower upper halfWidths : Fin 3 → ℝ)
    (targetCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (rotationWitness : Ω → Fin 8 → Prop)
    (εCover : ℝ)
    (hTransCover : HypercubeSupportCoversOnBox 3 translationSupport lower upper halfWidths)
    (hRotCover : ∀ ω, ∃ k : Fin 8, rotationWitness ω k)
    (hCompose : ∀ t ts ω k,
      (∀ i, |t i - ts i| ≤ halfWidths i) →
      rotationWitness ω k →
      rmsd (supportedCoords ts k) (targetCoords t ω) ≤ εCover) :
    ∀ t ω,
      (∀ i, lower i ≤ t i ∧ t i ≤ upper i) →
      ∃ seed ∈ translationSubsetQuaternionDictionary8Support translationSupport,
        rmsd (supportedCoords seed.1 seed.2) (targetCoords t ω) ≤ εCover := by
  intro t ω hBox
  rcases hTransCover t hBox with ⟨ts, hts, hcell⟩
  rcases hRotCover ω with ⟨k, hk⟩
  refine ⟨(ts, k), ?_, hCompose t ts ω k hcell hk⟩
  simp [translationSubsetQuaternionDictionary8Support, hts]

theorem translationSubsetQuaternionDictionary8Support_contains_explicit_rmsd_witness
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (translationSupport : Finset (Fin 3 → ℝ))
    (halfWidths : Fin 3 → ℝ)
    (targetCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (rotationWitness : Ω → Fin 8 → Prop)
    (εCover : ℝ)
    {t ts : Fin 3 → ℝ}
    {ω : Ω}
    {k : Fin 8}
    (hts : ts ∈ translationSupport)
    (hCell : ∀ i, |t i - ts i| ≤ halfWidths i)
    (hRot : rotationWitness ω k)
    (hCompose : ∀ t ts ω k,
      (∀ i, |t i - ts i| ≤ halfWidths i) ->
      rotationWitness ω k ->
      rmsd (supportedCoords ts k) (targetCoords t ω) ≤ εCover) :
    ∃ seed ∈ translationSubsetQuaternionDictionary8Support translationSupport,
      rmsd (supportedCoords seed.1 seed.2) (targetCoords t ω) ≤ εCover := by
  refine ⟨(ts, k), ?_, hCompose t ts ω k hCell hRot⟩
  simp [translationSubsetQuaternionDictionary8Support, hts]

/-- Runtime-shape bridge for rigid seeds when the composition budget is provided
    by explicit pointwise translation and orientation sub-budgets.

    This theorem isolates the remaining deterministic rigid-seed gap: the runtime
    only needs to prove (1) a translation pointwise budget and (2) a fixed
    orientation pointwise budget for the dictionary element `k`, then RMSD support
    follows automatically by `rmsd_le_of_pointwiseSplitDist_le`. -/
theorem translationSubsetQuaternionDictionary8Support_contains_rmsd_witness_of_split_component_budgets
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (translationSupport : Finset (Fin 3 → ℝ))
    (lower upper halfWidths : Fin 3 → ℝ)
    (targetCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (rotationWitness : Ω → Fin 8 → Prop)
    (tBound rBound εCover : ℝ)
    (hTransCover : HypercubeSupportCoversOnBox 3 translationSupport lower upper halfWidths)
    (hRotCover : ∀ ω, ∃ k : Fin 8, rotationWitness ω k)
    (hn : 0 < n)
    (ht : 0 ≤ tBound)
    (hr : 0 ≤ rBound)
    (hBudget : tBound + rBound ≤ εCover)
    (hTrans : ∀ t ts ω,
      (∀ i, |t i - ts i| ≤ halfWidths i) →
      ∀ j, dist (targetCoords t ω j) (midCoords ts ω j) ≤ tBound)
    (hRot : ∀ ts ω k,
      rotationWitness ω k →
      ∀ j, dist (midCoords ts ω j) (supportedCoords ts k j) ≤ rBound) :
    ∀ t ω,
      (∀ i, lower i ≤ t i ∧ t i ≤ upper i) →
      ∃ seed ∈ translationSubsetQuaternionDictionary8Support translationSupport,
        rmsd (supportedCoords seed.1 seed.2) (targetCoords t ω) ≤ εCover := by
  intro t ω hBox
  rcases hTransCover t hBox with ⟨ts, hts, hcell⟩
  rcases hRotCover ω with ⟨k, hk⟩
  have hRMSD : rmsd (supportedCoords ts k) (targetCoords t ω) ≤ εCover := by
    exact rmsd_le_of_pointwiseSplitDist_le
      (supportedCoords ts k) (midCoords ts ω) (targetCoords t ω)
      hn rBound tBound εCover hr ht (by simpa [add_comm] using hBudget)
      (fun j => by simpa [dist_comm] using hRot ts ω k hk j)
      (fun j => by simpa [dist_comm] using hTrans t ts ω hcell j)
  refine ⟨(ts, k), ?_, ?_⟩
  · simp [translationSubsetQuaternionDictionary8Support, hts]
  · exact hRMSD

/-- Concrete runtime-facing support theorem for the current rigid seed family once
    translation budgets and a coordinate-based quaternionDictionary8 witness budget
    are certified separately. -/
theorem translationSubsetQuaternionDictionary8Support_contains_rmsd_witness_of_coordinate_witness_budget
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (translationSupport : Finset (Fin 3 → ℝ))
    (lower upper halfWidths : Fin 3 → ℝ)
    (targetCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (tBound rBound εCover : ℝ)
    (hTransCover : HypercubeSupportCoversOnBox 3 translationSupport lower upper halfWidths)
    (hQuatUnit : ∀ ω, Computation.ArrayDSL.norm (targetQuaternion ω) = 1)
    (hn : 0 < n)
    (ht : 0 ≤ tBound)
    (hr : 0 ≤ rBound)
    (hBudget : tBound + rBound ≤ εCover)
    (hTrans : ∀ t ts ω,
      (∀ i, |t i - ts i| ≤ halfWidths i) →
      ∀ j, dist (targetCoords t ω j) (midCoords ts ω j) ≤ tBound)
    (hRot : ∀ ts ω k,
      quaternionDictionary8CoordinateWitness (targetQuaternion ω) k →
      ∀ j, dist (midCoords ts ω j) (supportedCoords ts k j) ≤ rBound) :
    ∀ t ω,
      (∀ i, lower i ≤ t i ∧ t i ≤ upper i) →
      ∃ seed ∈ translationSubsetQuaternionDictionary8Support translationSupport,
        rmsd (supportedCoords seed.1 seed.2) (targetCoords t ω) ≤ εCover := by
  apply translationSubsetQuaternionDictionary8Support_contains_rmsd_witness_of_split_component_budgets
    translationSupport lower upper halfWidths targetCoords midCoords supportedCoords
    (fun ω k => quaternionDictionary8CoordinateWitness (targetQuaternion ω) k)
    tBound rBound εCover hTransCover
  · intro ω
    rcases quaternionDictionary8_contains_coordinate_witness (targetQuaternion ω) (hQuatUnit ω) with
        ⟨k, _hkMem, hkWitness⟩
    exact ⟨k, hkWitness⟩
  · exact hn
  · exact ht
  · exact hr
  · exact hBudget
  · exact hTrans
  · exact hRot

/-- Pointwise signed-distance orientation budgets can be assembled from two
    unsigned quaternion-distance bounds (relative to `q_k` and `-q_k`). This
    isolates the geometric core needed by runtime proofs: one bound for
    `‖q - q_k‖` and one bound for `‖q + q_k‖`, with a shared Lipschitz factor. -/
theorem pointwise_signed_quaternion_distance_lipschitz_of_two_sided_unsigned_bounds
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (rotLipschitz : ℝ)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (hRotPlus : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖)
    (hRotMinus : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖) :
    ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
  intro ts ω k j
  set a : ℝ := ‖targetQuaternion ω - quaternionDictionary8 k‖
  set b : ℝ := ‖targetQuaternion ω + quaternionDictionary8 k‖
  have hPlus : dist (midCoords ts ω j) (supportedCoords ts k j) ≤ rotLipschitz * a := by
    simpa [a] using hRotPlus ts ω k j
  have hMinus : dist (midCoords ts ω j) (supportedCoords ts k j) ≤ rotLipschitz * b := by
    simpa [b] using hRotMinus ts ω k j
  have hMinScaled :
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        min (rotLipschitz * a) (rotLipschitz * b) :=
    le_min hPlus hMinus
  have hScaleMin : min (rotLipschitz * a) (rotLipschitz * b) = rotLipschitz * min a b := by
    by_cases hab : a ≤ b
    · have hmul : rotLipschitz * a ≤ rotLipschitz * b :=
        mul_le_mul_of_nonneg_left hab hRotLipNonneg
      rw [min_eq_left hmul, min_eq_left hab]
    · have hba : b ≤ a := le_of_not_ge hab
      have hmul : rotLipschitz * b ≤ rotLipschitz * a :=
        mul_le_mul_of_nonneg_left hba hRotLipNonneg
      rw [min_eq_right hmul, min_eq_right hba]
  exact (le_trans hMinScaled (by simpa [a, b, hScaleMin]))

/-- Signed-distance pointwise budget from one unsigned geometric bound plus a
    sign-invariant model witness. The runtime can use this form when it proves
    a geometric bound for `q` and separately proves the same bound for `-q`,
    together with sign invariance of the intermediate coordinates. -/
theorem pointwise_signed_quaternion_distance_lipschitz_of_unsigned_bound_and_neg_invariance
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (midCoordsNeg : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (rotLipschitz : ℝ)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (hMidNeg : ∀ ts ω j, midCoords ts ω j = midCoordsNeg ts ω j)
    (hRotPlus : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖)
    (hRotPlusNeg : ∀ ts ω k j,
      dist (midCoordsNeg ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz * ‖(-targetQuaternion ω) - quaternionDictionary8 k‖) :
    ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
  have hRotMinus : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
    intro ts ω k j
    calc
      dist (midCoords ts ω j) (supportedCoords ts k j)
          = dist (midCoordsNeg ts ω j) (supportedCoords ts k j) := by
              simpa [hMidNeg ts ω j]
      _ ≤ rotLipschitz * ‖(-targetQuaternion ω) - quaternionDictionary8 k‖ :=
            hRotPlusNeg ts ω k j
      _ = rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
            have hnorm :
                ‖(-targetQuaternion ω) - quaternionDictionary8 k‖ =
                  ‖quaternionDictionary8 k + targetQuaternion ω‖ := by
              calc
                ‖(-targetQuaternion ω) - quaternionDictionary8 k‖
                    = ‖-quaternionDictionary8 k + -targetQuaternion ω‖ := by
                        simp [sub_eq_add_neg, add_comm]
                _ = ‖quaternionDictionary8 k + targetQuaternion ω‖ := by
                        simpa [neg_add, add_comm, add_left_comm, add_assoc] using
                          (norm_neg (quaternionDictionary8 k + targetQuaternion ω))
            simpa [hnorm, add_comm]
  exact pointwise_signed_quaternion_distance_lipschitz_of_two_sided_unsigned_bounds
    midCoords supportedCoords targetQuaternion rotLipschitz hRotLipNonneg hRotPlus hRotMinus

/-- Concrete rigid-transform instantiation of the signed-distance pointwise
    budget from a single unsigned geometric inequality parameterized by the
    candidate quaternion `q`. The `-q` branch is obtained automatically using
    rigid-transform sign invariance. -/
theorem pointwise_signed_quaternion_distance_lipschitz_of_two_sided_unsigned_bounds_dep
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Ω → Fin 8 → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (rotLipschitz : ℝ)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (hRotPlus : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts ω k j) ≤
        rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖)
    (hRotMinus : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts ω k j) ≤
        rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖) :
    ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts ω k j) ≤
        rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
  intro ts ω k j
  set a : ℝ := ‖targetQuaternion ω - quaternionDictionary8 k‖
  set b : ℝ := ‖targetQuaternion ω + quaternionDictionary8 k‖
  have hPlus : dist (midCoords ts ω j) (supportedCoords ts ω k j) ≤ rotLipschitz * a := by
    simpa [a] using hRotPlus ts ω k j
  have hMinus : dist (midCoords ts ω j) (supportedCoords ts ω k j) ≤ rotLipschitz * b := by
    simpa [b] using hRotMinus ts ω k j
  have hMinScaled :
      dist (midCoords ts ω j) (supportedCoords ts ω k j) ≤
        min (rotLipschitz * a) (rotLipschitz * b) :=
    le_min hPlus hMinus
  have hScaleMin : min (rotLipschitz * a) (rotLipschitz * b) = rotLipschitz * min a b := by
    by_cases hab : a ≤ b
    · have hmul : rotLipschitz * a ≤ rotLipschitz * b :=
        mul_le_mul_of_nonneg_left hab hRotLipNonneg
      rw [min_eq_left hmul, min_eq_left hab]
    · have hba : b ≤ a := le_of_not_ge hab
      have hmul : rotLipschitz * b ≤ rotLipschitz * a :=
        mul_le_mul_of_nonneg_left hba hRotLipNonneg
      rw [min_eq_right hmul, min_eq_right hba]
  exact (le_trans hMinScaled (by simpa [a, b, hScaleMin]))

theorem pointwise_signed_quaternion_distance_lipschitz_of_unsigned_bound_and_neg_invariance_dep
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (midCoordsNeg : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Ω → Fin 8 → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (rotLipschitz : ℝ)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (hMidNeg : ∀ ts ω j, midCoords ts ω j = midCoordsNeg ts ω j)
    (hRotPlus : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts ω k j) ≤
        rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖)
    (hRotPlusNeg : ∀ ts ω k j,
      dist (midCoordsNeg ts ω j) (supportedCoords ts ω k j) ≤
        rotLipschitz * ‖(-targetQuaternion ω) - quaternionDictionary8 k‖) :
    ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts ω k j) ≤
        rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
  have hRotMinus : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts ω k j) ≤
        rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
    intro ts ω k j
    calc
      dist (midCoords ts ω j) (supportedCoords ts ω k j)
          = dist (midCoordsNeg ts ω j) (supportedCoords ts ω k j) := by
              simpa [hMidNeg ts ω j]
      _ ≤ rotLipschitz * ‖(-targetQuaternion ω) - quaternionDictionary8 k‖ :=
            hRotPlusNeg ts ω k j
      _ = rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
            have hnorm :
                ‖(-targetQuaternion ω) - quaternionDictionary8 k‖ =
                  ‖quaternionDictionary8 k + targetQuaternion ω‖ := by
              calc
                ‖(-targetQuaternion ω) - quaternionDictionary8 k‖
                    = ‖-quaternionDictionary8 k + -targetQuaternion ω‖ := by
                        simp [sub_eq_add_neg, add_comm]
                _ = ‖quaternionDictionary8 k + targetQuaternion ω‖ := by
                        simpa [neg_add, add_comm, add_left_comm, add_assoc] using
                          (norm_neg (quaternionDictionary8 k + targetQuaternion ω))
            simpa [hnorm, add_comm]
  exact pointwise_signed_quaternion_distance_lipschitz_of_two_sided_unsigned_bounds_dep
    midCoords supportedCoords targetQuaternion rotLipschitz hRotLipNonneg hRotPlus hRotMinus

theorem rigidTransform3D_pointwise_signed_quaternion_distance_lipschitz_of_unsigned_parametric_bound
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (baseCoords : Ω → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (rotLipschitz : ℝ)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (hRotParam : ∀ ts ω q k j,
      dist (rigidTransform3D (baseCoords ω) q (mkMDArray ts) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤
        rotLipschitz * ‖q - quaternionDictionary8 k‖) :
    ∀ ts ω k j,
      dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤
        rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
  let midCoords : (Fin 3 → ℝ) → Ω → CoordSet n :=
    fun ts ω => rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts)
  let midCoordsNeg : (Fin 3 → ℝ) → Ω → CoordSet n :=
    fun ts ω => rigidTransform3D (baseCoords ω) (-targetQuaternion ω) (mkMDArray ts)
  let supportedCoords : (Fin 3 → ℝ) → Ω → Fin 8 → CoordSet n :=
    fun ts ω k => rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts)
  have hMidNeg : ∀ ts ω j, midCoords ts ω j = midCoordsNeg ts ω j := by
    intro ts ω j
    have hEq :
        rigidTransform3D (baseCoords ω) (-targetQuaternion ω) (mkMDArray ts) =
          rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) :=
      rigidTransform3D_negQuaternion_eq (baseCoords ω) (targetQuaternion ω) (mkMDArray ts)
    simpa [midCoords, midCoordsNeg] using congrArg (fun c : CoordSet n => c j) hEq.symm
  have hRotPlus : ∀ ts ω k j,
      dist (midCoords ts ω j)
          (supportedCoords ts ω k j) ≤
        rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖ := by
    intro ts ω k j
    simpa [midCoords, supportedCoords] using hRotParam ts ω (targetQuaternion ω) k j
  have hRotPlusNeg : ∀ ts ω k j,
      dist (midCoordsNeg ts ω j)
          (supportedCoords ts ω k j) ≤
        rotLipschitz * ‖(-targetQuaternion ω) - quaternionDictionary8 k‖ := by
    intro ts ω k j
    simpa [midCoordsNeg, supportedCoords] using hRotParam ts ω (-targetQuaternion ω) k j
  intro ts ω k j
  have hMain := pointwise_signed_quaternion_distance_lipschitz_of_unsigned_bound_and_neg_invariance_dep
    midCoords midCoordsNeg supportedCoords
    targetQuaternion rotLipschitz hRotLipNonneg
    hMidNeg hRotPlus hRotPlusNeg
  simpa [midCoords, supportedCoords] using hMain ts ω k j

/-- Rigid-transform unsigned quaternion-distance bounds are translation-invariant:
    it is enough to prove the geometric inequality at zero translation. -/
theorem rigidTransform3D_pointwise_unsigned_parametric_bound_of_zero_translation
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (baseCoords : Ω → CoordSet n)
    (rotLipschitz : ℝ)
    (hRotParamZero : ∀ ω q k j,
      dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖q - quaternionDictionary8 k‖) :
    ∀ ts ω q k j,
      dist (rigidTransform3D (baseCoords ω) q (mkMDArray ts) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤
        rotLipschitz * ‖q - quaternionDictionary8 k‖ := by
  intro ts ω q k j
  calc
    dist (rigidTransform3D (baseCoords ω) q (mkMDArray ts) j)
        (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j)
        = dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
            (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray (fun _ => 0)) j) := by
              exact Computation.ArrayDSL.rigidTransform3D_same_translation_dist_eq_zero_translation
                (baseCoords ω) q (quaternionDictionary8 k) (mkMDArray ts) j
    _ ≤ rotLipschitz * ‖q - quaternionDictionary8 k‖ := hRotParamZero ω q k j

/-- Coordinate-witness rigid-transform signed-distance budget from a zero-translation
    unsigned parametric bound against the first four basis quaternions.
    This theorem isolates the remaining geometric work to proving the basis-indexed
    unsigned inequality at translation `0`. -/
theorem rigidTransform3D_pointwise_signed_quaternion_distance_lipschitz_of_basis_unsigned_parametric_bound_on_coordinate_witness
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (baseCoords : Ω → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (rotLipschitz : ℝ)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (hRotParamZeroBasis : ∀ ω q (i : Fin 4) j,
      dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (EuclideanSpace.single i (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖q - EuclideanSpace.single i (1 : ℝ)‖) :
    ∀ ts ω k,
      quaternionDictionary8CoordinateWitness (targetQuaternion ω) k →
      ∀ j,
        dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
            (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤
          rotLipschitz *
            min ‖targetQuaternion ω - quaternionDictionary8 k‖
                ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
  intro ts ω k hkCoord j
  rcases hkCoord with ⟨hk, _hAbs⟩
  let i : Fin 4 := ⟨k.1, hk⟩
  have hDict : quaternionDictionary8 k = EuclideanSpace.single i (1 : ℝ) := by
    fin_cases k
    · simpa [i] using quaternionDictionary8_basis0_eq
    · simpa [i] using quaternionDictionary8_basis1_eq
    · simpa [i] using quaternionDictionary8_basis2_eq
    · simpa [i] using quaternionDictionary8_basis3_eq
    · simp at hk
    · simp at hk
    · simp at hk
    · simp at hk
  have hPlusZero :
      dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖ := by
    simpa [hDict] using hRotParamZeroBasis ω (targetQuaternion ω) i j
  have hPlus :
      dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤
        rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖ := by
    have hTransEq :=
      Computation.ArrayDSL.rigidTransform3D_same_translation_dist_eq_zero_translation
        (baseCoords ω) (targetQuaternion ω) (quaternionDictionary8 k) (mkMDArray ts) j
    rw [hTransEq]
    exact hPlusZero
  have hMinusZero :
      dist (rigidTransform3D (baseCoords ω) (-targetQuaternion ω) (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖(-targetQuaternion ω) - quaternionDictionary8 k‖ := by
    simpa [hDict] using hRotParamZeroBasis ω (-targetQuaternion ω) i j
  have hMinusWithNegMid :
      dist (rigidTransform3D (baseCoords ω) (-targetQuaternion ω) (mkMDArray ts) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤
        rotLipschitz * ‖(-targetQuaternion ω) - quaternionDictionary8 k‖ := by
    have hTransEq :=
      Computation.ArrayDSL.rigidTransform3D_same_translation_dist_eq_zero_translation
        (baseCoords ω) (-targetQuaternion ω) (quaternionDictionary8 k) (mkMDArray ts) j
    rw [hTransEq]
    exact hMinusZero
  have hMidNegEq :
      rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j =
        rigidTransform3D (baseCoords ω) (-targetQuaternion ω) (mkMDArray ts) j := by
    have hEq :
        rigidTransform3D (baseCoords ω) (-targetQuaternion ω) (mkMDArray ts) =
          rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) :=
      rigidTransform3D_negQuaternion_eq (baseCoords ω) (targetQuaternion ω) (mkMDArray ts)
    simpa using congrArg (fun c : CoordSet n => c j) hEq.symm
  have hMinus :
      dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤
        rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
    have hA :
        dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
            (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j)
          = dist (rigidTransform3D (baseCoords ω) (-targetQuaternion ω) (mkMDArray ts) j)
              (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) := by
      simpa [hMidNegEq]
    have hNorm :
        ‖(-targetQuaternion ω) - quaternionDictionary8 k‖ =
          ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
      calc
        ‖(-targetQuaternion ω) - quaternionDictionary8 k‖
            = ‖-quaternionDictionary8 k + -targetQuaternion ω‖ := by
                simp [sub_eq_add_neg, add_comm]
        _ = ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
                simpa [neg_add, add_comm, add_left_comm, add_assoc] using
                  (norm_neg (quaternionDictionary8 k + targetQuaternion ω))
    rw [hA]
    exact (hMinusWithNegMid.trans (by simpa [hNorm]))
  have hMinScaled :
      dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤
        min (rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖)
            (rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖) :=
    le_min hPlus hMinus
  have hScaleMin :
      min (rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖)
          (rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖)
        = rotLipschitz *
            min ‖targetQuaternion ω - quaternionDictionary8 k‖
                ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
    by_cases hab : ‖targetQuaternion ω - quaternionDictionary8 k‖ ≤ ‖targetQuaternion ω + quaternionDictionary8 k‖
    · have hmul :
          rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖ ≤
            rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖ :=
        mul_le_mul_of_nonneg_left hab hRotLipNonneg
      rw [min_eq_left hmul, min_eq_left hab]
    · have hba : ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ ‖targetQuaternion ω - quaternionDictionary8 k‖ :=
        le_of_not_ge hab
      have hmul :
          rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤
            rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖ :=
        mul_le_mul_of_nonneg_left hba hRotLipNonneg
      rw [min_eq_right hmul, min_eq_right hba]
  exact le_trans hMinScaled (by simpa [hScaleMin])

/-- Coordinate-witness rigid-transform pointwise orientation budget in the runtime
    `rBound` form, derived from a zero-translation unsigned basis-bound plus a
    scale relation `rotLipschitz ≤ rBound`. -/
theorem rigidTransform3D_pointwise_budget_of_coordinate_witness_basis_unsigned_parametric_bound
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (baseCoords : Ω → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (rotLipschitz rBound : ℝ)
    (hQuatUnit : ∀ ω, Computation.ArrayDSL.norm (targetQuaternion ω) = 1)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (hRotScale : rotLipschitz ≤ rBound)
    (hRotParamZeroBasis : ∀ ω q (i : Fin 4) j,
      dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (EuclideanSpace.single i (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖q - EuclideanSpace.single i (1 : ℝ)‖) :
    ∀ ts ω k,
      quaternionDictionary8CoordinateWitness (targetQuaternion ω) k →
      ∀ j,
        dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
            (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤ rBound := by
  intro ts ω k hkCoord j
  have hBase :
      dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤
        rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
    exact rigidTransform3D_pointwise_signed_quaternion_distance_lipschitz_of_basis_unsigned_parametric_bound_on_coordinate_witness
      baseCoords targetQuaternion rotLipschitz hRotLipNonneg hRotParamZeroBasis ts ω k hkCoord j
  have hkSigned :
      min ‖targetQuaternion ω - quaternionDictionary8 k‖
          ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ 1 :=
    quaternionDictionary8_signed_distance_le_one_of_coordinate_witness
      (targetQuaternion ω) (hQuatUnit ω) k hkCoord
  have hUnitScale :
      rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ rotLipschitz :=
    mul_le_of_le_one_right hRotLipNonneg hkSigned
  exact le_trans hBase (le_trans hUnitScale hRotScale)

/-- To establish the zero-translation unsigned basis-bound over all `i : Fin 4`,
    it is enough to prove the four concrete Cartesian basis cases explicitly. -/
theorem rigidTransform3D_pointwise_unsigned_parametric_bound_zero_translation_of_basis_cases
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (baseCoords : Ω → CoordSet n)
    (rotLipschitz : ℝ)
    (h0 : ∀ ω q j,
      dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (EuclideanSpace.single (0 : Fin 4) (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖q - EuclideanSpace.single (0 : Fin 4) (1 : ℝ)‖)
    (h1 : ∀ ω q j,
      dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (EuclideanSpace.single (1 : Fin 4) (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖q - EuclideanSpace.single (1 : Fin 4) (1 : ℝ)‖)
    (h2 : ∀ ω q j,
      dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (EuclideanSpace.single (2 : Fin 4) (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖q - EuclideanSpace.single (2 : Fin 4) (1 : ℝ)‖)
    (h3 : ∀ ω q j,
      dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (EuclideanSpace.single (3 : Fin 4) (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖q - EuclideanSpace.single (3 : Fin 4) (1 : ℝ)‖) :
    ∀ ω q (i : Fin 4) j,
      dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (EuclideanSpace.single i (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖q - EuclideanSpace.single i (1 : ℝ)‖ := by
  intro ω q i j
  fin_cases i
  · simpa using h0 ω q j
  · simpa using h1 ω q j
  · simpa using h2 ω q j
  · simpa using h3 ω q j

/-- Concrete support-adequacy bridge for the runtime shape where both midpoint
    and supported rigid coordinates are generated by `rigidTransform3D` on a
    per-instance base conformer, with quaternionDictionary8 on the support side.
    The orientation side is discharged from a zero-translation unsigned basis
    bound plus a scale relation `rotLipschitz ≤ rBound`. -/
theorem rigidTransform3D_pointwise_budget_of_coordinate_witness_basis_unsigned_parametric_bound_of_basis_cases
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (baseCoords : Ω → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (rotLipschitz rBound : ℝ)
    (hQuatUnit : ∀ ω, Computation.ArrayDSL.norm (targetQuaternion ω) = 1)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (hRotScale : rotLipschitz ≤ rBound)
    (h0 : ∀ ω q j,
      dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (EuclideanSpace.single (0 : Fin 4) (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖q - EuclideanSpace.single (0 : Fin 4) (1 : ℝ)‖)
    (h1 : ∀ ω q j,
      dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (EuclideanSpace.single (1 : Fin 4) (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖q - EuclideanSpace.single (1 : Fin 4) (1 : ℝ)‖)
    (h2 : ∀ ω q j,
      dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (EuclideanSpace.single (2 : Fin 4) (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖q - EuclideanSpace.single (2 : Fin 4) (1 : ℝ)‖)
    (h3 : ∀ ω q j,
      dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (EuclideanSpace.single (3 : Fin 4) (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖q - EuclideanSpace.single (3 : Fin 4) (1 : ℝ)‖) :
    ∀ ts ω k,
      quaternionDictionary8CoordinateWitness (targetQuaternion ω) k →
      ∀ j,
        dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
            (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤ rBound := by
  intro ts ω k hkCoord j
  exact rigidTransform3D_pointwise_budget_of_coordinate_witness_basis_unsigned_parametric_bound
    baseCoords targetQuaternion rotLipschitz rBound hQuatUnit hRotLipNonneg hRotScale
    (rigidTransform3D_pointwise_unsigned_parametric_bound_zero_translation_of_basis_cases
      baseCoords rotLipschitz h0 h1 h2 h3)
    ts ω k hkCoord j

/-- Unit-quaternion variant of the coordinate-witness rigid-transform runtime
    budget bridge. This version matches the geometric basis-case theorems that
    are proved under `norm q = 1`. -/
theorem rigidTransform3D_pointwise_budget_of_coordinate_witness_basis_unsigned_parametric_bound_unit
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (baseCoords : Ω → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (rotLipschitz rBound : ℝ)
    (hQuatUnit : ∀ ω, Computation.ArrayDSL.norm (targetQuaternion ω) = 1)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (hRotScale : rotLipschitz ≤ rBound)
    (hRotParamZeroBasisUnit : ∀ ω q,
      Computation.ArrayDSL.norm q = 1 →
      ∀ i : Fin 4, ∀ j,
        dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
            (rigidTransform3D (baseCoords ω) (EuclideanSpace.single i (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
          rotLipschitz * ‖q - EuclideanSpace.single i (1 : ℝ)‖) :
    ∀ ts ω k,
      quaternionDictionary8CoordinateWitness (targetQuaternion ω) k →
      ∀ j,
        dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
            (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤ rBound := by
  intro ts ω k hkCoord j
  rcases hkCoord with ⟨hk, hAbs⟩
  let i : Fin 4 := ⟨k.1, hk⟩
  have hDict : quaternionDictionary8 k = EuclideanSpace.single i (1 : ℝ) := by
    fin_cases k
    · simpa [i] using quaternionDictionary8_basis0_eq
    · simpa [i] using quaternionDictionary8_basis1_eq
    · simpa [i] using quaternionDictionary8_basis2_eq
    · simpa [i] using quaternionDictionary8_basis3_eq
    · simp at hk
    · simp at hk
    · simp at hk
    · simp at hk
  have hPlusZero :
      dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖ := by
    simpa [hDict] using
      hRotParamZeroBasisUnit ω (targetQuaternion ω) (hQuatUnit ω) i j
  have hPlus :
      dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤
        rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖ := by
    have hTransEq :=
      Computation.ArrayDSL.rigidTransform3D_same_translation_dist_eq_zero_translation
        (baseCoords ω) (targetQuaternion ω) (quaternionDictionary8 k) (mkMDArray ts) j
    rw [hTransEq]
    exact hPlusZero
  have hNegQuatUnit : Computation.ArrayDSL.norm (-targetQuaternion ω) = 1 := by
    simpa [Computation.ArrayDSL.norm] using hQuatUnit ω
  have hMinusZero :
      dist (rigidTransform3D (baseCoords ω) (-targetQuaternion ω) (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray (fun _ => 0)) j) ≤
        rotLipschitz * ‖(-targetQuaternion ω) - quaternionDictionary8 k‖ := by
    simpa [hDict] using
      hRotParamZeroBasisUnit ω (-targetQuaternion ω) hNegQuatUnit i j
  have hMinusWithNegMid :
      dist (rigidTransform3D (baseCoords ω) (-targetQuaternion ω) (mkMDArray ts) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤
        rotLipschitz * ‖(-targetQuaternion ω) - quaternionDictionary8 k‖ := by
    have hTransEq :=
      Computation.ArrayDSL.rigidTransform3D_same_translation_dist_eq_zero_translation
        (baseCoords ω) (-targetQuaternion ω) (quaternionDictionary8 k) (mkMDArray ts) j
    rw [hTransEq]
    exact hMinusZero
  have hMidNegEq :
      rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j =
        rigidTransform3D (baseCoords ω) (-targetQuaternion ω) (mkMDArray ts) j := by
    have hEq :
        rigidTransform3D (baseCoords ω) (-targetQuaternion ω) (mkMDArray ts) =
          rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) :=
      rigidTransform3D_negQuaternion_eq (baseCoords ω) (targetQuaternion ω) (mkMDArray ts)
    simpa using congrArg (fun c : CoordSet n => c j) hEq.symm
  have hMinus :
      dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤
        rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
    have hA :
        dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
            (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j)
          = dist (rigidTransform3D (baseCoords ω) (-targetQuaternion ω) (mkMDArray ts) j)
              (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) := by
      simpa [hMidNegEq]
    have hNorm :
        ‖(-targetQuaternion ω) - quaternionDictionary8 k‖ =
          ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
      calc
        ‖(-targetQuaternion ω) - quaternionDictionary8 k‖
            = ‖-quaternionDictionary8 k + -targetQuaternion ω‖ := by
                simp [sub_eq_add_neg, add_comm]
        _ = ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
                simpa [neg_add, add_comm, add_left_comm, add_assoc] using
                  (norm_neg (quaternionDictionary8 k + targetQuaternion ω))
    rw [hA]
    exact (hMinusWithNegMid.trans (by simpa [hNorm]))
  have hMinScaled :
      dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤
        min (rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖)
            (rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖) :=
    le_min hPlus hMinus
  have hScaleMin :
      min (rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖)
          (rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖)
        = rotLipschitz *
            min ‖targetQuaternion ω - quaternionDictionary8 k‖
                ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
    by_cases hab : ‖targetQuaternion ω - quaternionDictionary8 k‖ ≤ ‖targetQuaternion ω + quaternionDictionary8 k‖
    · have hmul :
          rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖ ≤
            rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖ :=
        mul_le_mul_of_nonneg_left hab hRotLipNonneg
      rw [min_eq_left hmul, min_eq_left hab]
    · have hba : ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ ‖targetQuaternion ω - quaternionDictionary8 k‖ :=
        le_of_not_ge hab
      have hmul :
          rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤
            rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖ :=
        mul_le_mul_of_nonneg_left hba hRotLipNonneg
      rw [min_eq_right hmul, min_eq_right hba]
  have hSignedBase :
      dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
          (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤
        rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖ :=
    le_trans hMinScaled (by simpa [hScaleMin])
  have hkSigned :
      min ‖targetQuaternion ω - quaternionDictionary8 k‖
          ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ 1 :=
    quaternionDictionary8_signed_distance_le_one_of_coordinate_witness
      (targetQuaternion ω) (hQuatUnit ω) k ⟨hk, hAbs⟩
  have hUnitScale :
      rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ rotLipschitz :=
    mul_le_of_le_one_right hRotLipNonneg hkSigned
  exact le_trans hSignedBase (le_trans hUnitScale hRotScale)

/-- Unit-quaternion basis-case assembly for the coordinate-witness runtime budget.
    This matches the concrete basis0/1/2/3 closure theorems in `ArrayDSL`. -/
theorem rigidTransform3D_pointwise_budget_of_coordinate_witness_basis_unsigned_parametric_bound_of_basis_cases_unit
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (baseCoords : Ω → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (rotLipschitz rBound : ℝ)
    (hQuatUnit : ∀ ω, Computation.ArrayDSL.norm (targetQuaternion ω) = 1)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (hRotScale : rotLipschitz ≤ rBound)
    (h0 : ∀ ω q,
      Computation.ArrayDSL.norm q = 1 →
      ∀ j,
        dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
            (rigidTransform3D (baseCoords ω) (EuclideanSpace.single (0 : Fin 4) (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
          rotLipschitz * ‖q - EuclideanSpace.single (0 : Fin 4) (1 : ℝ)‖)
    (h1 : ∀ ω q,
      Computation.ArrayDSL.norm q = 1 →
      ∀ j,
        dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
            (rigidTransform3D (baseCoords ω) (EuclideanSpace.single (1 : Fin 4) (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
          rotLipschitz * ‖q - EuclideanSpace.single (1 : Fin 4) (1 : ℝ)‖)
    (h2 : ∀ ω q,
      Computation.ArrayDSL.norm q = 1 →
      ∀ j,
        dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
            (rigidTransform3D (baseCoords ω) (EuclideanSpace.single (2 : Fin 4) (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
          rotLipschitz * ‖q - EuclideanSpace.single (2 : Fin 4) (1 : ℝ)‖)
    (h3 : ∀ ω q,
      Computation.ArrayDSL.norm q = 1 →
      ∀ j,
        dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
            (rigidTransform3D (baseCoords ω) (EuclideanSpace.single (3 : Fin 4) (1 : ℝ)) (mkMDArray (fun _ => 0)) j) ≤
          rotLipschitz * ‖q - EuclideanSpace.single (3 : Fin 4) (1 : ℝ)‖) :
    ∀ ts ω k,
      quaternionDictionary8CoordinateWitness (targetQuaternion ω) k →
      ∀ j,
        dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
            (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤ rBound := by
  intro ts ω k hkCoord j
  exact rigidTransform3D_pointwise_budget_of_coordinate_witness_basis_unsigned_parametric_bound_unit
    baseCoords targetQuaternion rotLipschitz rBound hQuatUnit hRotLipNonneg hRotScale
    (fun ω q hNorm i j => by
      fin_cases i
      · simpa using h0 ω q hNorm j
      · simpa using h1 ω q hNorm j
      · simpa using h2 ω q hNorm j
      · simpa using h3 ω q hNorm j)
    ts ω k hkCoord j

/-- Concrete runtime pointwise orientation budget from the closed basis0/1/2/3
    zero-translation arm-bound theorems in `ArrayDSL`. -/
theorem rigidTransform3D_pointwise_budget_of_coordinate_witness_basis_unsigned_parametric_bound_of_uniform_pointL1Radius_bound
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (baseCoords : Ω → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (armBound rBound : ℝ)
    (hQuatUnit : ∀ ω, Computation.ArrayDSL.norm (targetQuaternion ω) = 1)
    (hArm : ∀ ω j, pointL1Radius (baseCoords ω j) ≤ armBound)
    (hArmNonneg : 0 ≤ armBound)
    (hRotScale : 48 * armBound ≤ rBound) :
    ∀ ts ω k,
      quaternionDictionary8CoordinateWitness (targetQuaternion ω) k →
      ∀ j,
        dist (rigidTransform3D (baseCoords ω) (targetQuaternion ω) (mkMDArray ts) j)
            (rigidTransform3D (baseCoords ω) (quaternionDictionary8 k) (mkMDArray ts) j) ≤ rBound := by
  have hRotLipNonneg : 0 ≤ 48 * armBound := by nlinarith
  intro ts ω k hkCoord j
  exact rigidTransform3D_pointwise_budget_of_coordinate_witness_basis_unsigned_parametric_bound_of_basis_cases_unit
    baseCoords targetQuaternion (48 * armBound) rBound hQuatUnit hRotLipNonneg hRotScale
    (fun ω q hNorm j => by
      simpa [quaternionDictionary8_basis0_eq] using
        rigidTransform3D_zero_translation_dist_to_basis0_le_of_pointL1Radius_bound
          (baseCoords ω) q hNorm armBound (hArm ω) j)
    (fun ω q hNorm j => by
      simpa [quaternionDictionary8_basis1_eq] using
        rigidTransform3D_zero_translation_dist_to_basis1_le_of_pointL1Radius_bound'
          (baseCoords ω) q hNorm armBound (hArm ω) j)
    (fun ω q hNorm j => by
      simpa [quaternionDictionary8_basis2_eq] using
        rigidTransform3D_zero_translation_dist_to_basis2_le_of_pointL1Radius_bound'
          (baseCoords ω) q hNorm armBound (hArm ω) j)
    (fun ω q hNorm j => by
      simpa [quaternionDictionary8_basis3_eq] using
        rigidTransform3D_zero_translation_dist_to_basis3_le_of_pointL1Radius_bound'
          (baseCoords ω) q hNorm armBound (hArm ω) j)
    ts ω k hkCoord j

/-- Runtime-facing signed-distance bridge using two unsigned geometric
    sub-bounds. This theorem is intended for the next proof step where the
    runtime supplies separate geometric displacement bounds for `q_k` and
    `-q_k`, then composes them into the signed-distance budget shape. -/
theorem translationSubsetQuaternionDictionary8Support_contains_rmsd_witness_of_signed_quaternion_distance_two_sided_lipschitz_budget
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (translationSupport : Finset (Fin 3 → ℝ))
    (lower upper halfWidths : Fin 3 → ℝ)
    (targetCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (rotLipschitz tBound rBound εCover : ℝ)
    (hTransCover : HypercubeSupportCoversOnBox 3 translationSupport lower upper halfWidths)
    (hQuatUnit : ∀ ω, Computation.ArrayDSL.norm (targetQuaternion ω) = 1)
    (hn : 0 < n)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (ht : 0 ≤ tBound)
    (hr : 0 ≤ rBound)
    (hRotScale : rotLipschitz ≤ rBound)
    (hBudget : tBound + rBound ≤ εCover)
    (hTrans : ∀ t ts ω,
      (∀ i, |t i - ts i| ≤ halfWidths i) →
      ∀ j, dist (targetCoords t ω j) (midCoords ts ω j) ≤ tBound)
    (hRotPlus : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖)
    (hRotMinus : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖) :
    ∀ t ω,
      (∀ i, lower i ≤ t i ∧ t i ≤ upper i) →
      ∃ seed ∈ translationSubsetQuaternionDictionary8Support translationSupport,
        rmsd (supportedCoords seed.1 seed.2) (targetCoords t ω) ≤ εCover := by
  apply translationSubsetQuaternionDictionary8Support_contains_rmsd_witness_of_split_component_budgets
    translationSupport lower upper halfWidths targetCoords midCoords supportedCoords
    (fun ω k =>
      min ‖targetQuaternion ω - quaternionDictionary8 k‖
          ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ 1)
    tBound rBound εCover hTransCover
  · intro ω
    rcases quaternionDictionary8_contains_signed_distance_witness (targetQuaternion ω) (hQuatUnit ω) with
        ⟨k, _hkMem, hkDist⟩
    exact ⟨k, hkDist⟩
  · exact hn
  · exact ht
  · exact hr
  · exact hBudget
  · exact hTrans
  · intro ts ω k hkSigned j
    have hBase :
        dist (midCoords ts ω j) (supportedCoords ts k j) ≤
          rotLipschitz *
            min ‖targetQuaternion ω - quaternionDictionary8 k‖
                ‖targetQuaternion ω + quaternionDictionary8 k‖ := by
      exact pointwise_signed_quaternion_distance_lipschitz_of_two_sided_unsigned_bounds
        midCoords supportedCoords targetQuaternion rotLipschitz hRotLipNonneg hRotPlus hRotMinus ts ω k j
    have hUnitScale :
        rotLipschitz *
            min ‖targetQuaternion ω - quaternionDictionary8 k‖
                ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ rotLipschitz := by
      exact mul_le_of_le_one_right hRotLipNonneg hkSigned
    exact le_trans hBase (le_trans hUnitScale hRotScale)

/-- Runtime-facing support theorem for the current rigid seed family when the
    orientation side is provided as a signed quaternion-distance budget against
    the fixed dictionary: a witness dictionary element `k` is required to satisfy
    `min ‖q - q_k‖ ‖q + q_k‖ ≤ 1`, and that witness must imply the pointwise
    orientation budget. -/
theorem translationSubsetQuaternionDictionary8Support_contains_rmsd_witness_of_signed_quaternion_distance_budget
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (translationSupport : Finset (Fin 3 → ℝ))
    (lower upper halfWidths : Fin 3 → ℝ)
    (targetCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (tBound rBound εCover : ℝ)
    (hTransCover : HypercubeSupportCoversOnBox 3 translationSupport lower upper halfWidths)
    (hQuatUnit : ∀ ω, Computation.ArrayDSL.norm (targetQuaternion ω) = 1)
    (hn : 0 < n)
    (ht : 0 ≤ tBound)
    (hr : 0 ≤ rBound)
    (hBudget : tBound + rBound ≤ εCover)
    (hTrans : ∀ t ts ω,
      (∀ i, |t i - ts i| ≤ halfWidths i) →
      ∀ j, dist (targetCoords t ω j) (midCoords ts ω j) ≤ tBound)
    (hRot : ∀ ts ω k,
      min ‖targetQuaternion ω - quaternionDictionary8 k‖
          ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ 1 →
      ∀ j, dist (midCoords ts ω j) (supportedCoords ts k j) ≤ rBound) :
    ∀ t ω,
      (∀ i, lower i ≤ t i ∧ t i ≤ upper i) →
      ∃ seed ∈ translationSubsetQuaternionDictionary8Support translationSupport,
        rmsd (supportedCoords seed.1 seed.2) (targetCoords t ω) ≤ εCover := by
  apply translationSubsetQuaternionDictionary8Support_contains_rmsd_witness_of_split_component_budgets
    translationSupport lower upper halfWidths targetCoords midCoords supportedCoords
    (fun ω k =>
      min ‖targetQuaternion ω - quaternionDictionary8 k‖
          ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ 1)
    tBound rBound εCover hTransCover
  · intro ω
    rcases quaternionDictionary8_contains_signed_distance_witness (targetQuaternion ω) (hQuatUnit ω) with
        ⟨k, _hkMem, hkDist⟩
    exact ⟨k, hkDist⟩
  · exact hn
  · exact ht
  · exact hr
  · exact hBudget
  · exact hTrans
  · exact hRot

/-- Runtime-facing signed-distance bridge with an explicit pointwise quaternion
    displacement Lipschitz constant. If each supported orientation displacement is
    bounded by `rotLipschitz * dSigned` where `dSigned` is the signed distance to
    `quaternionDictionary8`, then the theorem-backed witness budget `dSigned ≤ 1`
    yields a concrete orientation budget `rBound` whenever
    `rotLipschitz ≤ rBound`. -/
theorem translationSubsetQuaternionDictionary8Support_contains_rmsd_witness_of_signed_quaternion_distance_lipschitz_budget
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (translationSupport : Finset (Fin 3 → ℝ))
    (lower upper halfWidths : Fin 3 → ℝ)
    (targetCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (rotLipschitz tBound rBound εCover : ℝ)
    (hTransCover : HypercubeSupportCoversOnBox 3 translationSupport lower upper halfWidths)
    (hQuatUnit : ∀ ω, Computation.ArrayDSL.norm (targetQuaternion ω) = 1)
    (hn : 0 < n)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (ht : 0 ≤ tBound)
    (hr : 0 ≤ rBound)
    (hRotScale : rotLipschitz ≤ rBound)
    (hBudget : tBound + rBound ≤ εCover)
    (hTrans : ∀ t ts ω,
      (∀ i, |t i - ts i| ≤ halfWidths i) →
      ∀ j, dist (targetCoords t ω j) (midCoords ts ω j) ≤ tBound)
    (hRotLip : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖) :
    ∀ t ω,
      (∀ i, lower i ≤ t i ∧ t i ≤ upper i) →
      ∃ seed ∈ translationSubsetQuaternionDictionary8Support translationSupport,
        rmsd (supportedCoords seed.1 seed.2) (targetCoords t ω) ≤ εCover := by
  apply translationSubsetQuaternionDictionary8Support_contains_rmsd_witness_of_signed_quaternion_distance_budget
    translationSupport lower upper halfWidths targetCoords midCoords supportedCoords
    targetQuaternion tBound rBound εCover hTransCover hQuatUnit hn ht hr hBudget hTrans
  intro ts ω k hkSigned j
  have hBase :
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖ :=
    hRotLip ts ω k j
  have hUnitScale :
      rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ rotLipschitz := by
    exact mul_le_of_le_one_right hRotLipNonneg hkSigned
  exact le_trans hBase (le_trans hUnitScale hRotScale)

/-- Coordinate-witness version of the signed-distance Lipschitz bridge.
    This packages the runtime shape that already carries
    `quaternionDictionary8CoordinateWitness`: if orientation displacement is
    bounded by `rotLipschitz * dSigned`, then the coordinate witness plus unit
    quaternion normalization is enough to derive the needed orientation budget. -/
theorem translationSubsetQuaternionDictionary8Support_contains_rmsd_witness_of_coordinate_witness_lipschitz_budget
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (translationSupport : Finset (Fin 3 → ℝ))
    (lower upper halfWidths : Fin 3 → ℝ)
    (targetCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (rotLipschitz tBound rBound εCover : ℝ)
    (hTransCover : HypercubeSupportCoversOnBox 3 translationSupport lower upper halfWidths)
    (hQuatUnit : ∀ ω, Computation.ArrayDSL.norm (targetQuaternion ω) = 1)
    (hn : 0 < n)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (ht : 0 ≤ tBound)
    (hr : 0 ≤ rBound)
    (hRotScale : rotLipschitz ≤ rBound)
    (hBudget : tBound + rBound ≤ εCover)
    (hTrans : ∀ t ts ω,
      (∀ i, |t i - ts i| ≤ halfWidths i) →
      ∀ j, dist (targetCoords t ω j) (midCoords ts ω j) ≤ tBound)
    (hRotLip : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖) :
    ∀ t ω,
      (∀ i, lower i ≤ t i ∧ t i ≤ upper i) →
      ∃ seed ∈ translationSubsetQuaternionDictionary8Support translationSupport,
        rmsd (supportedCoords seed.1 seed.2) (targetCoords t ω) ≤ εCover := by
  apply translationSubsetQuaternionDictionary8Support_contains_rmsd_witness_of_coordinate_witness_budget
    translationSupport lower upper halfWidths targetCoords midCoords supportedCoords
    targetQuaternion tBound rBound εCover hTransCover hQuatUnit hn ht hr hBudget hTrans
  intro ts ω k hkCoord j
  have hkSigned :
      min ‖targetQuaternion ω - quaternionDictionary8 k‖
          ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ 1 :=
    quaternionDictionary8_signed_distance_le_one_of_coordinate_witness
      (targetQuaternion ω) (hQuatUnit ω) k hkCoord
  have hBase :
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖ :=
    hRotLip ts ω k j
  have hUnitScale :
      rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ rotLipschitz := by
    exact mul_le_of_le_one_right hRotLipNonneg hkSigned
  exact le_trans hBase (le_trans hUnitScale hRotScale)

/-- End-to-end rigid-seed adequacy bridge for the runtime's current rigid seed
    family shape: translation subset crossed with `quaternionDictionary8`.

    If the component-cover witness theorem produces a supported pose within
    `εCover` RMSD of the exact rigid optimum, and the exact energy is
    RMSD-Lipschitz with constant `L`, then the exact rescored winner of that
    finite rigid seed library is within the requested RMSD target whenever the
    induced energy budget `L * εCover` fits the certified quadratic basin gap. -/
theorem translationSubsetQuaternionDictionary8Library_winner_of_component_covers_yields_rmsd_target
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (translationSupport : Finset (Fin 3 → ℝ))
    (lower upper halfWidths : Fin 3 → ℝ)
    (targetCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (rotationWitness : Ω → Fin 8 → Prop)
    (energy : CoordSet n → ℝ)
    (L εCover εTarget : ℝ)
    (hTransCover : HypercubeSupportCoversOnBox 3 translationSupport lower upper halfWidths)
    (hRotCover : ∀ ω, ∃ k : Fin 8, rotationWitness ω k)
    (hCompose : ∀ t ts ω k,
      (∀ i, |t i - ts i| ≤ halfWidths i) →
      rotationWitness ω k →
      rmsd (supportedCoords ts k) (targetCoords t ω) ≤ εCover)
    (hL : 0 ≤ L)
    (hεCover : 0 ≤ εCover)
    {t : Fin 3 → ℝ}
    {ω : Ω}
    (hBox : ∀ i, lower i ≤ t i ∧ t i ≤ upper i)
    (hOpt : IsOptimal (fun x => -energy x) (targetCoords t ω))
    (hLip : RMSDLipschitzEnergy energy L)
    {winner : CoordSet n}
    (hWinnerMem : winner ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2))
    (hWinnerBest : ∀ z,
      z ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2) →
      energy winner ≤ energy z)
    (basin : CertifiedQuadraticBasin energy (targetCoords t ω))
    (hn : 0 < n)
    (hεTarget : 0 ≤ εTarget)
    (hGapBudget : L * εCover ≤ targetEnergyGap basin.μ n εTarget) :
    rmsd winner (targetCoords t ω) ≤ εTarget := by
  rcases translationSubsetQuaternionDictionary8Support_contains_rmsd_witness_of_component_covers
      translationSupport lower upper halfWidths targetCoords supportedCoords rotationWitness εCover
      hTransCover hRotCover hCompose t ω hBox with ⟨seed, hSeedMem, hSeedRMSD⟩
  let witness : CoordSet n := supportedCoords seed.1 seed.2
  have hWitnessMem :
      witness ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2) := by
    exact Finset.mem_image.mpr ⟨seed, hSeedMem, rfl⟩
  have hWitnessApprox : IsApproxOptimal (fun x => -energy x) (L * εCover) witness := by
    intro x
    have hOptX : -energy x ≤ -energy (targetCoords t ω) := hOpt x
    have hGap : |energy witness - energy (targetCoords t ω)| ≤ L * εCover := by
      exact energyBudget_of_rmsdTarget energy L εCover hL hLip hSeedRMSD
    have hWitnessNear : -energy (targetCoords t ω) ≤ -energy witness + L * εCover := by
      have hGap' := abs_le.mp hGap
      linarith
    linarith
  have hWinnerApprox : IsApproxOptimal (fun x => -energy x) (L * εCover) winner :=
    exact_library_winner_inherits_approxOptimal_of_supportedWitness
      energy
      ((translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2))
      (L * εCover)
      hWitnessMem hWitnessApprox hWinnerMem hWinnerBest
  exact approxOptimal_negEnergy_yields_rmsd_target
    energy (L * εCover) εTarget hOpt hWinnerApprox basin hn hεTarget hGapBudget

/-- End-to-end rigid-seed adequacy bridge when the runtime certifies separate
    translation and orientation pointwise budgets and composes them through the
    RMSD split-budget lemma. This theorem is the direct runtime-facing theorem for
    a future explicit quaternion-cover implementation. -/
theorem translationSubsetQuaternionDictionary8Library_winner_of_split_component_budgets_yields_rmsd_target
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (translationSupport : Finset (Fin 3 → ℝ))
    (lower upper halfWidths : Fin 3 → ℝ)
    (targetCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (rotationWitness : Ω → Fin 8 → Prop)
    (energy : CoordSet n → ℝ)
    (tBound rBound L εCover εTarget : ℝ)
    (hTransCover : HypercubeSupportCoversOnBox 3 translationSupport lower upper halfWidths)
    (hRotCover : ∀ ω, ∃ k : Fin 8, rotationWitness ω k)
    (hn : 0 < n)
    (ht : 0 ≤ tBound)
    (hr : 0 ≤ rBound)
    (hBudget : tBound + rBound ≤ εCover)
    (hTrans : ∀ t ts ω,
      (∀ i, |t i - ts i| ≤ halfWidths i) →
      ∀ j, dist (targetCoords t ω j) (midCoords ts ω j) ≤ tBound)
    (hRot : ∀ ts ω k,
      rotationWitness ω k →
      ∀ j, dist (midCoords ts ω j) (supportedCoords ts k j) ≤ rBound)
    (hL : 0 ≤ L)
    (hεCover : 0 ≤ εCover)
    {t : Fin 3 → ℝ}
    {ω : Ω}
    (hBox : ∀ i, lower i ≤ t i ∧ t i ≤ upper i)
    (hOpt : IsOptimal (fun x => -energy x) (targetCoords t ω))
    (hLip : RMSDLipschitzEnergy energy L)
    {winner : CoordSet n}
    (hWinnerMem : winner ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2))
    (hWinnerBest : ∀ z,
      z ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2) →
      energy winner ≤ energy z)
    (basin : CertifiedQuadraticBasin energy (targetCoords t ω))
    (hεTarget : 0 ≤ εTarget)
    (hGapBudget : L * εCover ≤ targetEnergyGap basin.μ n εTarget) :
    rmsd winner (targetCoords t ω) ≤ εTarget := by
  rcases translationSubsetQuaternionDictionary8Support_contains_rmsd_witness_of_split_component_budgets
      translationSupport lower upper halfWidths targetCoords midCoords supportedCoords rotationWitness
      tBound rBound εCover hTransCover hRotCover hn ht hr hBudget hTrans hRot t ω hBox with
      ⟨seed, hSeedMem, hSeedRMSD⟩
  let witness : CoordSet n := supportedCoords seed.1 seed.2
  have hWitnessMem :
      witness ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2) := by
    exact Finset.mem_image.mpr ⟨seed, hSeedMem, rfl⟩
  have hWitnessApprox : IsApproxOptimal (fun x => -energy x) (L * εCover) witness := by
    intro x
    have hOptX : -energy x ≤ -energy (targetCoords t ω) := hOpt x
    have hGap : |energy witness - energy (targetCoords t ω)| ≤ L * εCover := by
      exact energyBudget_of_rmsdTarget energy L εCover hL hLip hSeedRMSD
    have hWitnessNear : -energy (targetCoords t ω) ≤ -energy witness + L * εCover := by
      have hGap' := abs_le.mp hGap
      linarith
    linarith
  have hWinnerApprox : IsApproxOptimal (fun x => -energy x) (L * εCover) winner :=
    exact_library_winner_inherits_approxOptimal_of_supportedWitness
      energy
      ((translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2))
      (L * εCover)
      hWitnessMem hWitnessApprox hWinnerMem hWinnerBest
  exact approxOptimal_negEnergy_yields_rmsd_target
    energy (L * εCover) εTarget hOpt hWinnerApprox basin hn hεTarget hGapBudget

/-- Concrete end-to-end rigid-seed adequacy bridge for the current runtime family
    using the coordinate-based quaternionDictionary8 witness predicate. -/
theorem translationSubsetQuaternionDictionary8Library_winner_of_coordinate_witness_budget_yields_rmsd_target
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (translationSupport : Finset (Fin 3 → ℝ))
    (lower upper halfWidths : Fin 3 → ℝ)
    (targetCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (energy : CoordSet n → ℝ)
    (tBound rBound L εCover εTarget : ℝ)
    (hTransCover : HypercubeSupportCoversOnBox 3 translationSupport lower upper halfWidths)
    (hQuatUnit : ∀ ω, Computation.ArrayDSL.norm (targetQuaternion ω) = 1)
    (hn : 0 < n)
    (ht : 0 ≤ tBound)
    (hr : 0 ≤ rBound)
    (hBudget : tBound + rBound ≤ εCover)
    (hTrans : ∀ t ts ω,
      (∀ i, |t i - ts i| ≤ halfWidths i) →
      ∀ j, dist (targetCoords t ω j) (midCoords ts ω j) ≤ tBound)
    (hRot : ∀ ts ω k,
      quaternionDictionary8CoordinateWitness (targetQuaternion ω) k →
      ∀ j, dist (midCoords ts ω j) (supportedCoords ts k j) ≤ rBound)
    (hL : 0 ≤ L)
    (hεCover : 0 ≤ εCover)
    {t : Fin 3 → ℝ}
    {ω : Ω}
    (hBox : ∀ i, lower i ≤ t i ∧ t i ≤ upper i)
    (hOpt : IsOptimal (fun x => -energy x) (targetCoords t ω))
    (hLip : RMSDLipschitzEnergy energy L)
    {winner : CoordSet n}
    (hWinnerMem : winner ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2))
    (hWinnerBest : ∀ z,
      z ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2) →
      energy winner ≤ energy z)
    (basin : CertifiedQuadraticBasin energy (targetCoords t ω))
    (hεTarget : 0 ≤ εTarget)
    (hGapBudget : L * εCover ≤ targetEnergyGap basin.μ n εTarget) :
    rmsd winner (targetCoords t ω) ≤ εTarget := by
  exact translationSubsetQuaternionDictionary8Library_winner_of_split_component_budgets_yields_rmsd_target
    translationSupport lower upper halfWidths targetCoords midCoords supportedCoords
    (fun ω k => quaternionDictionary8CoordinateWitness (targetQuaternion ω) k)
    energy tBound rBound L εCover εTarget hTransCover
    (fun ω => by
      rcases quaternionDictionary8_contains_coordinate_witness (targetQuaternion ω) (hQuatUnit ω) with
          ⟨k, _hkMem, hkWitness⟩
      exact ⟨k, hkWitness⟩)
    hn ht hr hBudget hTrans hRot hL hεCover hBox hOpt hLip hWinnerMem hWinnerBest basin hεTarget hGapBudget

/-- End-to-end rigid-seed adequacy bridge for the current runtime family when
    the orientation side is provided as a signed quaternion-distance budget
    against `quaternionDictionary8`. -/
theorem translationSubsetQuaternionDictionary8Library_winner_of_signed_quaternion_distance_budget_yields_rmsd_target
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (translationSupport : Finset (Fin 3 → ℝ))
    (lower upper halfWidths : Fin 3 → ℝ)
    (targetCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (energy : CoordSet n → ℝ)
    (tBound rBound L εCover εTarget : ℝ)
    (hTransCover : HypercubeSupportCoversOnBox 3 translationSupport lower upper halfWidths)
    (hQuatUnit : ∀ ω, Computation.ArrayDSL.norm (targetQuaternion ω) = 1)
    (hn : 0 < n)
    (ht : 0 ≤ tBound)
    (hr : 0 ≤ rBound)
    (hBudget : tBound + rBound ≤ εCover)
    (hTrans : ∀ t ts ω,
      (∀ i, |t i - ts i| ≤ halfWidths i) →
      ∀ j, dist (targetCoords t ω j) (midCoords ts ω j) ≤ tBound)
    (hRot : ∀ ts ω k,
      min ‖targetQuaternion ω - quaternionDictionary8 k‖
          ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ 1 →
      ∀ j, dist (midCoords ts ω j) (supportedCoords ts k j) ≤ rBound)
    (hL : 0 ≤ L)
    (hεCover : 0 ≤ εCover)
    {t : Fin 3 → ℝ}
    {ω : Ω}
    (hBox : ∀ i, lower i ≤ t i ∧ t i ≤ upper i)
    (hOpt : IsOptimal (fun x => -energy x) (targetCoords t ω))
    (hLip : RMSDLipschitzEnergy energy L)
    {winner : CoordSet n}
    (hWinnerMem : winner ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2))
    (hWinnerBest : ∀ z,
      z ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2) →
      energy winner ≤ energy z)
    (basin : CertifiedQuadraticBasin energy (targetCoords t ω))
    (hεTarget : 0 ≤ εTarget)
    (hGapBudget : L * εCover ≤ targetEnergyGap basin.μ n εTarget) :
    rmsd winner (targetCoords t ω) ≤ εTarget := by
  exact translationSubsetQuaternionDictionary8Library_winner_of_split_component_budgets_yields_rmsd_target
    translationSupport lower upper halfWidths targetCoords midCoords supportedCoords
    (fun ω k =>
      min ‖targetQuaternion ω - quaternionDictionary8 k‖
          ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ 1)
    energy tBound rBound L εCover εTarget hTransCover
    (fun ω => by
      rcases quaternionDictionary8_contains_signed_distance_witness (targetQuaternion ω) (hQuatUnit ω) with
          ⟨k, _hkMem, hkDist⟩
      exact ⟨k, hkDist⟩)
    hn ht hr hBudget hTrans hRot hL hεCover hBox hOpt hLip hWinnerMem hWinnerBest basin hεTarget hGapBudget

/-- End-to-end signed-distance bridge with an explicit pointwise quaternion
    displacement Lipschitz constant. This theorem removes the need to provide a
    separate per-witness orientation budget assumption: it is derived from
    `rotLipschitz * dSigned` and `dSigned ≤ 1`. -/
theorem translationSubsetQuaternionDictionary8Library_winner_of_signed_quaternion_distance_lipschitz_budget_yields_rmsd_target
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (translationSupport : Finset (Fin 3 → ℝ))
    (lower upper halfWidths : Fin 3 → ℝ)
    (targetCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (energy : CoordSet n → ℝ)
    (rotLipschitz tBound rBound L εCover εTarget : ℝ)
    (hTransCover : HypercubeSupportCoversOnBox 3 translationSupport lower upper halfWidths)
    (hQuatUnit : ∀ ω, Computation.ArrayDSL.norm (targetQuaternion ω) = 1)
    (hn : 0 < n)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (ht : 0 ≤ tBound)
    (hr : 0 ≤ rBound)
    (hRotScale : rotLipschitz ≤ rBound)
    (hBudget : tBound + rBound ≤ εCover)
    (hTrans : ∀ t ts ω,
      (∀ i, |t i - ts i| ≤ halfWidths i) →
      ∀ j, dist (targetCoords t ω j) (midCoords ts ω j) ≤ tBound)
    (hRotLip : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖)
    (hL : 0 ≤ L)
    (hεCover : 0 ≤ εCover)
    {t : Fin 3 → ℝ}
    {ω : Ω}
    (hBox : ∀ i, lower i ≤ t i ∧ t i ≤ upper i)
    (hOpt : IsOptimal (fun x => -energy x) (targetCoords t ω))
    (hLip : RMSDLipschitzEnergy energy L)
    {winner : CoordSet n}
    (hWinnerMem : winner ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2))
    (hWinnerBest : ∀ z,
      z ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2) →
      energy winner ≤ energy z)
    (basin : CertifiedQuadraticBasin energy (targetCoords t ω))
    (hεTarget : 0 ≤ εTarget)
    (hGapBudget : L * εCover ≤ targetEnergyGap basin.μ n εTarget) :
    rmsd winner (targetCoords t ω) ≤ εTarget := by
  exact translationSubsetQuaternionDictionary8Library_winner_of_signed_quaternion_distance_budget_yields_rmsd_target
    translationSupport lower upper halfWidths targetCoords midCoords supportedCoords
    targetQuaternion energy tBound rBound L εCover εTarget hTransCover hQuatUnit hn
    ht hr hBudget hTrans
    (fun ts ω k hkSigned j => by
      have hBase :
          dist (midCoords ts ω j) (supportedCoords ts k j) ≤
            rotLipschitz *
              min ‖targetQuaternion ω - quaternionDictionary8 k‖
                  ‖targetQuaternion ω + quaternionDictionary8 k‖ :=
        hRotLip ts ω k j
      have hUnitScale :
          rotLipschitz *
              min ‖targetQuaternion ω - quaternionDictionary8 k‖
                  ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ rotLipschitz := by
        exact mul_le_of_le_one_right hRotLipNonneg hkSigned
      exact le_trans hBase (le_trans hUnitScale hRotScale))
    hL hεCover hBox hOpt hLip hWinnerMem hWinnerBest basin hεTarget hGapBudget

/-- End-to-end signed-distance winner bridge assembled from two unsigned
    quaternion-distance geometric bounds (`‖q-q_k‖` and `‖q+q_k‖`). This is the
    runtime-facing reduction that isolates the remaining geometric work for
    `hRotLip` into two explicit subproofs. -/
theorem translationSubsetQuaternionDictionary8Library_winner_of_signed_quaternion_distance_two_sided_lipschitz_budget_yields_rmsd_target
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (translationSupport : Finset (Fin 3 → ℝ))
    (lower upper halfWidths : Fin 3 → ℝ)
    (targetCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (energy : CoordSet n → ℝ)
    (rotLipschitz tBound rBound L εCover εTarget : ℝ)
    (hTransCover : HypercubeSupportCoversOnBox 3 translationSupport lower upper halfWidths)
    (hQuatUnit : ∀ ω, Computation.ArrayDSL.norm (targetQuaternion ω) = 1)
    (hn : 0 < n)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (ht : 0 ≤ tBound)
    (hr : 0 ≤ rBound)
    (hRotScale : rotLipschitz ≤ rBound)
    (hBudget : tBound + rBound ≤ εCover)
    (hTrans : ∀ t ts ω,
      (∀ i, |t i - ts i| ≤ halfWidths i) →
      ∀ j, dist (targetCoords t ω j) (midCoords ts ω j) ≤ tBound)
    (hRotPlus : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz * ‖targetQuaternion ω - quaternionDictionary8 k‖)
    (hRotMinus : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz * ‖targetQuaternion ω + quaternionDictionary8 k‖)
    (hL : 0 ≤ L)
    (hεCover : 0 ≤ εCover)
    {t : Fin 3 → ℝ}
    {ω : Ω}
    (hBox : ∀ i, lower i ≤ t i ∧ t i ≤ upper i)
    (hOpt : IsOptimal (fun x => -energy x) (targetCoords t ω))
    (hLip : RMSDLipschitzEnergy energy L)
    {winner : CoordSet n}
    (hWinnerMem : winner ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2))
    (hWinnerBest : ∀ z,
      z ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2) →
      energy winner ≤ energy z)
    (basin : CertifiedQuadraticBasin energy (targetCoords t ω))
    (hεTarget : 0 ≤ εTarget)
    (hGapBudget : L * εCover ≤ targetEnergyGap basin.μ n εTarget) :
    rmsd winner (targetCoords t ω) ≤ εTarget := by
  exact translationSubsetQuaternionDictionary8Library_winner_of_signed_quaternion_distance_lipschitz_budget_yields_rmsd_target
    translationSupport lower upper halfWidths targetCoords midCoords supportedCoords
    targetQuaternion energy rotLipschitz tBound rBound L εCover εTarget hTransCover
    hQuatUnit hn hRotLipNonneg ht hr hRotScale hBudget hTrans
    (pointwise_signed_quaternion_distance_lipschitz_of_two_sided_unsigned_bounds
      midCoords supportedCoords targetQuaternion rotLipschitz hRotLipNonneg hRotPlus hRotMinus)
    hL hεCover hBox hOpt hLip hWinnerMem hWinnerBest basin hεTarget hGapBudget

/-- Coordinate-witness version of the signed-distance Lipschitz winner bridge.
    This is the end-to-end theorem for runtimes that expose coordinate witnesses
    directly and prove orientation displacement via a signed-distance Lipschitz
    constant. -/
theorem translationSubsetQuaternionDictionary8Library_winner_of_coordinate_witness_lipschitz_budget_yields_rmsd_target
    {Ω : Type u}
    {n : ℕ} [DecidableEq (CoordSet n)]
    (translationSupport : Finset (Fin 3 → ℝ))
    (lower upper halfWidths : Fin 3 → ℝ)
    (targetCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (midCoords : (Fin 3 → ℝ) → Ω → CoordSet n)
    (supportedCoords : (Fin 3 → ℝ) → Fin 8 → CoordSet n)
    (targetQuaternion : Ω → MDArray 4)
    (energy : CoordSet n → ℝ)
    (rotLipschitz tBound rBound L εCover εTarget : ℝ)
    (hTransCover : HypercubeSupportCoversOnBox 3 translationSupport lower upper halfWidths)
    (hQuatUnit : ∀ ω, Computation.ArrayDSL.norm (targetQuaternion ω) = 1)
    (hn : 0 < n)
    (hRotLipNonneg : 0 ≤ rotLipschitz)
    (ht : 0 ≤ tBound)
    (hr : 0 ≤ rBound)
    (hRotScale : rotLipschitz ≤ rBound)
    (hBudget : tBound + rBound ≤ εCover)
    (hTrans : ∀ t ts ω,
      (∀ i, |t i - ts i| ≤ halfWidths i) →
      ∀ j, dist (targetCoords t ω j) (midCoords ts ω j) ≤ tBound)
    (hRotLip : ∀ ts ω k j,
      dist (midCoords ts ω j) (supportedCoords ts k j) ≤
        rotLipschitz *
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖)
    (hL : 0 ≤ L)
    (hεCover : 0 ≤ εCover)
    {t : Fin 3 → ℝ}
    {ω : Ω}
    (hBox : ∀ i, lower i ≤ t i ∧ t i ≤ upper i)
    (hOpt : IsOptimal (fun x => -energy x) (targetCoords t ω))
    (hLip : RMSDLipschitzEnergy energy L)
    {winner : CoordSet n}
    (hWinnerMem : winner ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2))
    (hWinnerBest : ∀ z,
      z ∈ (translationSubsetQuaternionDictionary8Support translationSupport).image (fun seed => supportedCoords seed.1 seed.2) →
      energy winner ≤ energy z)
    (basin : CertifiedQuadraticBasin energy (targetCoords t ω))
    (hεTarget : 0 ≤ εTarget)
    (hGapBudget : L * εCover ≤ targetEnergyGap basin.μ n εTarget) :
    rmsd winner (targetCoords t ω) ≤ εTarget := by
  exact translationSubsetQuaternionDictionary8Library_winner_of_coordinate_witness_budget_yields_rmsd_target
    translationSupport lower upper halfWidths targetCoords midCoords supportedCoords
    targetQuaternion energy tBound rBound L εCover εTarget hTransCover hQuatUnit hn
    ht hr hBudget hTrans
    (fun ts ω k hkCoord j => by
      have hkSigned :
          min ‖targetQuaternion ω - quaternionDictionary8 k‖
              ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ 1 :=
        quaternionDictionary8_signed_distance_le_one_of_coordinate_witness
          (targetQuaternion ω) (hQuatUnit ω) k hkCoord
      have hBase :
          dist (midCoords ts ω j) (supportedCoords ts k j) ≤
            rotLipschitz *
              min ‖targetQuaternion ω - quaternionDictionary8 k‖
                  ‖targetQuaternion ω + quaternionDictionary8 k‖ :=
        hRotLip ts ω k j
      have hUnitScale :
          rotLipschitz *
              min ‖targetQuaternion ω - quaternionDictionary8 k‖
                  ‖targetQuaternion ω + quaternionDictionary8 k‖ ≤ rotLipschitz := by
        exact mul_le_of_le_one_right hRotLipNonneg hkSigned
      exact le_trans hBase (le_trans hUnitScale hRotScale))
    hL hεCover hBox hOpt hLip hWinnerMem hWinnerBest basin hεTarget hGapBudget

/-- If the exact optimum lies in a certified hypercube cell centered at `center`,
    then the center is delta-near-optimal with delta equal to the weighted-L1
    Lipschitz slack over that cell. -/
theorem cell_center_is_approxOptimal_of_optimum_in_cell
    (n : ℕ)
    (f : (Fin n → ℝ) → ℝ)
    (L : Fin n → ℝ)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |f p - f q| ≤ L i * |p i - q i|)
    (center halfWidths pStar : Fin n → ℝ)
    (hL : ∀ i, 0 ≤ L i)
    (hCell : ∀ i, |pStar i - center i| ≤ halfWidths i)
    (hOpt : IsOptimal f pStar) :
    IsApproxOptimal f (Finset.univ.sum (fun i => L i * halfWidths i)) center := by
  intro p
  let slack : ℝ := Finset.univ.sum (fun i => L i * halfWidths i)
  have hOptP : f p ≤ f pStar := hOpt p
  have hAbsCenter : |f pStar - f center| ≤ slack := by
    have hBase := per_dimension_lipschitz_bound n f L hLip pStar center
    have hCoord : Finset.univ.sum (fun i => L i * |pStar i - center i|) ≤ slack := by
      refine Finset.sum_le_sum ?_
      intro i _
      have hHalfNonneg : 0 ≤ halfWidths i := le_trans (abs_nonneg _) (hCell i)
      exact mul_le_mul_of_nonneg_left (hCell i) (hL i)
    exact le_trans hBase hCoord
  have hNear : f pStar ≤ f center + slack := by
    have hAbs' := abs_le.mp hAbsCenter
    dsimp [slack] at hAbs' ⊢
    linarith
  linarith

/-- A support family of hypercube centers contains a delta-near-optimal point
    whenever it covers the ambient optimum. -/
theorem hypercube_support_yields_supported_approxOptimal
    (n : ℕ)
    (f : (Fin n → ℝ) → ℝ)
    (support : Finset (Fin n → ℝ))
    (L halfWidths : Fin n → ℝ)
    (hCover : HypercubeSupportCovers n support halfWidths)
    (hL : ∀ i, 0 ≤ L i)
    {pStar : Fin n → ℝ}
    (hOpt : IsOptimal f pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |f p - f q| ≤ L i * |p i - q i|) :
    ∃ center, center ∈ support ∧
      IsApproxOptimal f (Finset.univ.sum (fun i => L i * halfWidths i)) center := by
  rcases hCover pStar with ⟨center, hcenter, hcell⟩
  refine ⟨center, hcenter, ?_⟩
  exact cell_center_is_approxOptimal_of_optimum_in_cell n f L hLip center halfWidths pStar hL hcell hOpt

theorem hypercube_supportOnBox_yields_supported_approxOptimal
    (n : ℕ)
    (f : (Fin n → ℝ) → ℝ)
    (support : Finset (Fin n → ℝ))
    (lower upper L halfWidths : Fin n → ℝ)
    (hCover : HypercubeSupportCoversOnBox n support lower upper halfWidths)
    (hL : ∀ i, 0 ≤ L i)
    {pStar : Fin n → ℝ}
    (hBox : ∀ i, lower i ≤ pStar i ∧ pStar i ≤ upper i)
    (hOpt : IsOptimal f pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |f p - f q| ≤ L i * |p i - q i|) :
    ∃ center, center ∈ support ∧
      IsApproxOptimal f (Finset.univ.sum (fun i => L i * halfWidths i)) center := by
  rcases hCover pStar hBox with ⟨center, hcenter, hcell⟩
  refine ⟨center, hcenter, ?_⟩
  exact cell_center_is_approxOptimal_of_optimum_in_cell n f L hLip center halfWidths pStar hL hcell hOpt

theorem hypercube_support_yields_restricted_opt_approxAmbient
    (n : ℕ)
    {S : Type v}
    (exactDP : DecisionProblem (Fin n → ℝ) S)
    (F : SampledActionFamily (Fin n → ℝ))
    (s : S)
    (L halfWidths : Fin n → ℝ)
    (hCover : HypercubeSupportCovers n F.support halfWidths)
    (hL : ∀ i, 0 ≤ L i)
    {pStar : Fin n → ℝ}
    (hOpt : IsOptimal (fun p => exactDP.utility p s) pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |exactDP.utility p s - exactDP.utility q s| ≤ L i * |p i - q i|) :
    ∃ a : SupportedAction F,
      a ∈ (restrictedDecisionProblem exactDP F).Opt s ∧
      IsApproxOptimal
        (fun p => exactDP.utility p s)
        (Finset.univ.sum (fun i => L i * halfWidths i)) a.1 := by
  rcases hypercube_support_yields_supported_approxOptimal
      n (fun p => exactDP.utility p s) F.support L halfWidths hCover hL hOpt hLip
      with ⟨center, hcenter, hApproxCenter⟩
  rcases restrictedDecisionProblem_exists_opt exactDP F s with ⟨aBest, hBestOpt⟩
  refine ⟨aBest, hBestOpt, ?_⟩
  intro p
  have hApproxP : exactDP.utility p s ≤ exactDP.utility center s + Finset.univ.sum (fun i => L i * halfWidths i) :=
    hApproxCenter p
  have hBestGe : exactDP.utility center s ≤ exactDP.utility aBest.1 s := by
    exact hBestOpt ⟨center, hcenter⟩
  linarith

theorem hypercube_supportOnBox_yields_restricted_opt_approxAmbient
    (n : ℕ)
    {S : Type v}
    (exactDP : DecisionProblem (Fin n → ℝ) S)
    (F : SampledActionFamily (Fin n → ℝ))
    (s : S)
    (lower upper L halfWidths : Fin n → ℝ)
    (hCover : HypercubeSupportCoversOnBox n F.support lower upper halfWidths)
    (hL : ∀ i, 0 ≤ L i)
    {pStar : Fin n → ℝ}
    (hBox : ∀ i, lower i ≤ pStar i ∧ pStar i ≤ upper i)
    (hOpt : IsOptimal (fun p => exactDP.utility p s) pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |exactDP.utility p s - exactDP.utility q s| ≤ L i * |p i - q i|) :
    ∃ a : SupportedAction F,
      a ∈ (restrictedDecisionProblem exactDP F).Opt s ∧
      IsApproxOptimal
        (fun p => exactDP.utility p s)
        (Finset.univ.sum (fun i => L i * halfWidths i)) a.1 := by
  rcases hypercube_supportOnBox_yields_supported_approxOptimal
      n (fun p => exactDP.utility p s) F.support lower upper L halfWidths hCover hL hBox hOpt hLip
      with ⟨center, hcenter, hApproxCenter⟩
  rcases restrictedDecisionProblem_exists_opt exactDP F s with ⟨aBest, hBestOpt⟩
  refine ⟨aBest, hBestOpt, ?_⟩
  intro p
  have hApproxP : exactDP.utility p s ≤ exactDP.utility center s + Finset.univ.sum (fun i => L i * halfWidths i) :=
    hApproxCenter p
  have hBestGe : exactDP.utility center s ≤ exactDP.utility aBest.1 s := by
    exact hBestOpt ⟨center, hcenter⟩
  linarith

/-- Uniform arithmetic centers on `[-π, π]` with spacing `2π / segments`. We use
    endpoints rather than midpoints, so the certified cover radius is exactly the
    grid spacing. This is slightly conservative but fully explicit and easy to
    instantiate in runtime code. -/
noncomputable def uniformArithmeticCentersPi
    (segments : ℕ) : Finset ℝ :=
  (Finset.range (segments + 1)).image
    (fun k : ℕ => -Real.pi + (k : ℝ) * (2 * Real.pi / segments))

/-- Grid spacing for the arithmetic-center support on `[-π, π]`. -/
noncomputable def uniformArithmeticStep
    (segments : ℕ) : ℝ :=
  2 * Real.pi / segments

theorem uniformArithmeticCentersPi_card
    (segments : ℕ)
    (hSeg : 0 < segments) :
    (uniformArithmeticCentersPi segments).card = segments + 1 := by
  unfold uniformArithmeticCentersPi
  have hInj : Set.InjOn (fun k : ℕ => -Real.pi + (k : ℝ) * (2 * Real.pi / segments)) ↑(Finset.range (segments + 1)) := by
    intro k hk l hl hEq
    have hStepPos : 0 < 2 * Real.pi / segments := by positivity
    have hkEq : (k : ℝ) = l := by
      nlinarith [hEq, hStepPos]
    exact_mod_cast hkEq
  calc
    (Finset.image (fun k : ℕ => -Real.pi + (k : ℝ) * (2 * Real.pi / segments)) (Finset.range (segments + 1))).card
        = (Finset.range (segments + 1)).card := Finset.card_image_iff.mpr hInj
    _ = segments + 1 := Finset.card_range (segments + 1)

theorem uniformArithmeticCentersPi_intervalCover
    (segments : ℕ)
    (hSeg : 0 < segments)
    (x : ℝ)
    (hxLower : -Real.pi ≤ x)
    (hxUpper : x ≤ Real.pi) :
    ∃ center, center ∈ uniformArithmeticCentersPi segments ∧
      |x - center| ≤ uniformArithmeticStep segments := by
  have hSegR : 0 < (segments : ℝ) := by exact_mod_cast hSeg
  have hStepPos : 0 < uniformArithmeticStep segments := by
    unfold uniformArithmeticStep
    positivity
  let y : ℝ := (x + Real.pi) / uniformArithmeticStep segments
  let k : ℕ := Nat.floor y
  have hyNonneg : 0 ≤ y := by
    dsimp [y]
    exact div_nonneg (by linarith) hStepPos.le
  have hyLe : y ≤ segments := by
    rw [_root_.div_le_iff₀ hStepPos]
    dsimp [y, uniformArithmeticStep]
    field_simp [hSegR.ne']
    nlinarith [hxLower, hxUpper]
  have hkLe : k ≤ segments := by
    have hkLeR : (k : ℝ) ≤ segments := by
      exact le_trans (Nat.floor_le hyNonneg) hyLe
    exact_mod_cast hkLeR
  let center : ℝ := -Real.pi + (k : ℝ) * uniformArithmeticStep segments
  refine ⟨center, ?_, ?_⟩
  · refine Finset.mem_image.mpr ?_
    refine ⟨k, Finset.mem_range.mpr ?_, rfl⟩
    omega
  · have hyMul : y * uniformArithmeticStep segments = x + Real.pi := by
      dsimp [y]
      field_simp [show uniformArithmeticStep segments ≠ 0 by positivity]
    have hkFloor : (k : ℝ) ≤ y := Nat.floor_le hyNonneg
    have hkSucc : y < k + 1 := Nat.lt_floor_add_one y
    have hDiffNonneg : 0 ≤ y - k := by linarith
    have hDiffLe : y - k ≤ 1 := by linarith
    have hCenterEq : x - center = (y - k) * uniformArithmeticStep segments := by
      dsimp [center]
      linarith [hyMul]
    rw [hCenterEq, abs_of_nonneg (mul_nonneg hDiffNonneg hStepPos.le)]
    calc
      (y - k) * uniformArithmeticStep segments ≤ 1 * uniformArithmeticStep segments := by
        gcongr
      _ = uniformArithmeticStep segments := by ring

theorem uniformArithmeticCentersPi_coordinateCover
    (segments : ℕ)
    (hSeg : 0 < segments) :
    CoordinatewiseIntervalCover 1
      (fun _ : Fin 1 => uniformArithmeticCentersPi segments)
      (fun _ => -Real.pi)
      (fun _ => Real.pi)
      (fun _ => uniformArithmeticStep segments) := by
  intro i x hxLower hxUpper
  rcases uniformArithmeticCentersPi_intervalCover segments hSeg x hxLower hxUpper with ⟨center, hcenter, hdist⟩
  refine ⟨center, hcenter, ?_⟩
  simpa using hdist

theorem uniformArithmeticCentersPi_cover_on_box
    (n segments : ℕ)
    (hSeg : 0 < segments) :
    HypercubeSupportCoversOnBox n
      (coordinateCenterSupport n (fun _ : Fin n => uniformArithmeticCentersPi segments))
      (fun _ => -Real.pi)
      (fun _ => Real.pi)
      (fun _ => uniformArithmeticStep segments) := by
  apply coordinatewise_cover_yields_hypercubeSupportOnBox
  intro i x hxLower hxUpper
  rcases uniformArithmeticCentersPi_intervalCover segments hSeg x hxLower hxUpper with ⟨center, hcenter, hdist⟩
  exact ⟨center, hcenter, hdist⟩

theorem uniformArithmeticCentersPi_tensor_card
    (n segments : ℕ)
    (hSeg : 0 < segments) :
    (coordinateCenterSupport n (fun _ : Fin n => uniformArithmeticCentersPi segments)).card =
      (segments + 1) ^ n := by
  rw [coordinateCenterSupport_card]
  simp [uniformArithmeticCentersPi_card, hSeg]

/-- Per-dimension arithmetic-center support obtained by giving each torsion its
    own deterministic segment count. -/
noncomputable def adaptiveArithmeticCenterSupport
    (n : ℕ)
    (segments : Fin n → ℕ) : Finset (Fin n → ℝ) :=
  coordinateCenterSupport n (fun i => uniformArithmeticCentersPi (segments i))

/-- Tensor-product cardinality for the adaptive arithmetic-center support. -/
theorem adaptiveArithmeticCenterSupport_card
    (n : ℕ)
    (segments : Fin n → ℕ)
    (hSeg : ∀ i, 0 < segments i) :
    (adaptiveArithmeticCenterSupport n segments).card = ∏ i, (segments i + 1) := by
  unfold adaptiveArithmeticCenterSupport
  rw [coordinateCenterSupport_card]
  refine Finset.prod_congr rfl ?_
  intro i _
  simpa using uniformArithmeticCentersPi_card (segments i) (hSeg i)

/-- The adaptive arithmetic-center support covers the full torsion box with the
    per-dimension arithmetic step as its certified half-width parameter. -/
theorem adaptiveArithmeticCenterSupport_cover_on_box
    (n : ℕ)
    (segments : Fin n → ℕ)
    (hSeg : ∀ i, 0 < segments i) :
    HypercubeSupportCoversOnBox n
      (adaptiveArithmeticCenterSupport n segments)
      (fun _ => -Real.pi)
      (fun _ => Real.pi)
      (fun i => uniformArithmeticStep (segments i)) := by
  unfold adaptiveArithmeticCenterSupport
  apply coordinatewise_cover_yields_hypercubeSupportOnBox
  intro i x hxLower hxUpper
  exact uniformArithmeticCentersPi_intervalCover (segments i) (hSeg i) x hxLower hxUpper

/-- The adaptive arithmetic support is never larger than a uniform support using
    a segment count that dominates every per-dimension segment count. -/
theorem adaptiveArithmeticCenterSupport_card_le_uniform
    (n s : ℕ)
    (segments : Fin n → ℕ)
    (hSeg : ∀ i, 0 < segments i)
    (hLe : ∀ i, segments i ≤ s) :
    (adaptiveArithmeticCenterSupport n segments).card ≤ (s + 1) ^ n := by
  rw [adaptiveArithmeticCenterSupport_card n segments hSeg]
  calc
    ∏ i, (segments i + 1) ≤ ∏ i : Fin n, (s + 1) := by
      simpa using
        (Finset.prod_le_prod'
          (s := (Finset.univ : Finset (Fin n)))
          (f := fun i => segments i + 1)
          (g := fun _ => s + 1)
          (fun i _ => Nat.succ_le_succ (hLe i)))
    _ = (s + 1) ^ n := by simp

/-- A per-dimension segment budget is admissible for target slack `δ` when the
    weighted arithmetic-step error sums to at most `δ`. -/
def AdaptiveSegmentBudget
    (n : ℕ)
    (L : Fin n → ℝ)
    (segments : Fin n → ℕ)
    (δ : ℝ) : Prop :=
  Finset.univ.sum (fun i => L i * uniformArithmeticStep (segments i)) ≤ δ

/-- Canonical uniform segment count derived directly from total Lipschitz mass and
    target slack. This is the fully explicit one-parameter fallback support size. -/
noncomputable def canonicalUniformSegmentsFromSlack
    (n : ℕ)
    (L : Fin n → ℝ)
    (δ : ℝ) : ℕ :=
  max 1 (Nat.ceil ((2 * Real.pi * Finset.univ.sum (fun i => L i)) / δ))

theorem canonicalUniformSegmentsFromSlack_pos
    (n : ℕ)
    (L : Fin n → ℝ)
    (δ : ℝ) :
    0 < canonicalUniformSegmentsFromSlack n L δ := by
  unfold canonicalUniformSegmentsFromSlack
  omega

theorem adaptiveBudget_of_canonicalUniformSegmentsFromSlack
    (n : ℕ)
    (L : Fin n → ℝ)
    (δ : ℝ)
    (hL : ∀ i, 0 ≤ L i)
    (hδ : 0 < δ) :
    AdaptiveSegmentBudget n L
      (fun _ => canonicalUniformSegmentsFromSlack n L δ) δ := by
  let totalL : ℝ := Finset.univ.sum (fun i => L i)
  have hSegPosNat : 0 < canonicalUniformSegmentsFromSlack n L δ :=
    canonicalUniformSegmentsFromSlack_pos n L δ
  have hTotalNonneg : 0 ≤ totalL := by
    exact Finset.sum_nonneg (fun i _ => hL i)
  have hLower : (2 * Real.pi * totalL) / δ ≤ (canonicalUniformSegmentsFromSlack n L δ : ℝ) := by
    unfold canonicalUniformSegmentsFromSlack
    calc
      (2 * Real.pi * Finset.univ.sum (fun i => L i)) / δ ≤ ↑⌈(2 * Real.pi * Finset.univ.sum (fun i => L i)) / δ⌉₊ := by
        exact Nat.le_ceil _
      _ ≤ ↑(max 1 ⌈(2 * Real.pi * Finset.univ.sum (fun i => L i)) / δ⌉₊) := by
        exact_mod_cast (le_max_right 1 ⌈(2 * Real.pi * Finset.univ.sum (fun i => L i)) / δ⌉₊)
  have hMain : totalL * uniformArithmeticStep (canonicalUniformSegmentsFromSlack n L δ) ≤ δ := by
    let seg := canonicalUniformSegmentsFromSlack n L δ
    have ha : 2 * Real.pi * totalL ≤ (seg : ℝ) * δ := by
      exact (_root_.div_le_iff₀ hδ).mp hLower
    have hDiv : (2 * Real.pi * totalL) / (seg : ℝ) ≤ δ := by
      exact (_root_.div_le_iff₀ (show 0 < (seg : ℝ) by exact_mod_cast hSegPosNat)).2 (by nlinarith)
    have hEq : totalL * uniformArithmeticStep seg = (2 * Real.pi * totalL) / (seg : ℝ) := by
      unfold uniformArithmeticStep
      field_simp [show (seg : ℝ) ≠ 0 by positivity]
    rw [hEq]
    exact hDiv
  unfold AdaptiveSegmentBudget
  dsimp [totalL]
  simpa [totalL, Finset.sum_mul] using hMain

/-- Canonical adaptive segment count from an equal-share per-dimension slack
    allocation. This gives a deterministic theorem-backed adaptive support family. -/
noncomputable def canonicalAdaptiveSegmentsFromSlack
    (n : ℕ)
    (L : Fin n → ℝ)
    (δ : ℝ) : Fin n → ℕ :=
  fun i => max 1 (Nat.ceil ((2 * Real.pi * (n : ℝ) * L i) / δ))

theorem canonicalAdaptiveSegmentsFromSlack_pos
    (n : ℕ)
    (L : Fin n → ℝ)
    (δ : ℝ)
    (i : Fin n) :
    0 < canonicalAdaptiveSegmentsFromSlack n L δ i := by
  unfold canonicalAdaptiveSegmentsFromSlack
  omega

theorem adaptiveBudget_of_canonicalAdaptiveSegmentsFromSlack
    (n : ℕ)
    (L : Fin n → ℝ)
    (δ : ℝ)
    (hn : 0 < n)
    (hL : ∀ i, 0 ≤ L i)
    (hδ : 0 < δ) :
    AdaptiveSegmentBudget n L (canonicalAdaptiveSegmentsFromSlack n L δ) δ := by
  have hnR : 0 < (n : ℝ) := by exact_mod_cast hn
  have hTerm : ∀ i : Fin n,
      L i * uniformArithmeticStep (canonicalAdaptiveSegmentsFromSlack n L δ i) ≤ δ / n := by
    intro i
    let seg := canonicalAdaptiveSegmentsFromSlack n L δ i
    have hSegPosNat : 0 < seg := canonicalAdaptiveSegmentsFromSlack_pos n L δ i
    have hLower : (2 * Real.pi * (n : ℝ) * L i) / δ ≤ (canonicalAdaptiveSegmentsFromSlack n L δ i : ℝ) := by
      unfold canonicalAdaptiveSegmentsFromSlack
      calc
        (2 * Real.pi * (n : ℝ) * L i) / δ ≤ ↑⌈(2 * Real.pi * (n : ℝ) * L i) / δ⌉₊ := by
          exact Nat.le_ceil _
        _ ≤ ↑(max 1 ⌈(2 * Real.pi * (n : ℝ) * L i) / δ⌉₊) := by
          exact_mod_cast (le_max_right 1 ⌈(2 * Real.pi * (n : ℝ) * L i) / δ⌉₊)
    have ha : 2 * Real.pi * (n : ℝ) * L i ≤ (seg : ℝ) * δ := by
      exact (_root_.div_le_iff₀ hδ).mp hLower
    have hDiv : (2 * Real.pi * (n : ℝ) * L i) / (seg : ℝ) ≤ δ := by
      exact (_root_.div_le_iff₀ (show 0 < (seg : ℝ) by exact_mod_cast hSegPosNat)).2 (by nlinarith)
    rw [_root_.le_div_iff₀ hnR]
    have hEq : L i * uniformArithmeticStep seg * n = (2 * Real.pi * (n : ℝ) * L i) / (seg : ℝ) := by
      unfold uniformArithmeticStep
      field_simp [show (seg : ℝ) ≠ 0 by positivity]
    rw [hEq]
    exact hDiv
  unfold AdaptiveSegmentBudget
  have hsum : Finset.univ.sum (fun i => L i * uniformArithmeticStep (canonicalAdaptiveSegmentsFromSlack n L δ i))
      ≤ Finset.univ.sum (fun _ : Fin n => δ / n) := by
    refine Finset.sum_le_sum ?_
    intro i _
    exact hTerm i
  have hconst : Finset.univ.sum (fun _ : Fin n => δ / n) = δ := by
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin]
    have : (n : ℝ) * (δ / n) = δ := by
      field_simp [hnR.ne']
    simpa using this
  exact hsum.trans_eq hconst

/-- If the per-dimension arithmetic support budget is admissible, the support
    contains a delta-near-optimal action. This is the direct implementation-facing
    theorem for deterministic torsion libraries. -/
theorem adaptiveArithmeticSupport_yields_supported_approxOptimal_of_budget
    (n : ℕ)
    (f : (Fin n → ℝ) → ℝ)
    (L : Fin n → ℝ)
    (segments : Fin n → ℕ)
    (δ : ℝ)
    (hSeg : ∀ i, 0 < segments i)
    (hL : ∀ i, 0 ≤ L i)
    (hBudget : AdaptiveSegmentBudget n L segments δ)
    {pStar : Fin n → ℝ}
    (hBox : ∀ i, -Real.pi ≤ pStar i ∧ pStar i ≤ Real.pi)
    (hOpt : IsOptimal f pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |f p - f q| ≤ L i * |p i - q i|) :
    ∃ center, center ∈ adaptiveArithmeticCenterSupport n segments ∧
      IsApproxOptimal f δ center := by
  rcases hypercube_supportOnBox_yields_supported_approxOptimal
      n f (adaptiveArithmeticCenterSupport n segments)
      (fun _ => -Real.pi)
      (fun _ => Real.pi)
      L
      (fun i => uniformArithmeticStep (segments i))
      (adaptiveArithmeticCenterSupport_cover_on_box n segments hSeg)
      hL hBox hOpt hLip with ⟨center, hcenter, hApprox⟩
  refine ⟨center, hcenter, ?_⟩
  exact approxOptimal_mono f hBudget hApprox

/-- Restricted sampled version of the adaptive arithmetic support theorem. -/
theorem adaptiveArithmeticSupport_yields_restricted_opt_approxAmbient_of_budget
    (n : ℕ)
    {S : Type v}
    (exactDP : DecisionProblem (Fin n → ℝ) S)
    (F : SampledActionFamily (Fin n → ℝ))
    (s : S)
    (L : Fin n → ℝ)
    (segments : Fin n → ℕ)
    (δ : ℝ)
    (hSupport : F.support = adaptiveArithmeticCenterSupport n segments)
    (hSeg : ∀ i, 0 < segments i)
    (hL : ∀ i, 0 ≤ L i)
    (hBudget : AdaptiveSegmentBudget n L segments δ)
    {pStar : Fin n → ℝ}
    (hBox : ∀ i, -Real.pi ≤ pStar i ∧ pStar i ≤ Real.pi)
    (hOpt : IsOptimal (fun p => exactDP.utility p s) pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |exactDP.utility p s - exactDP.utility q s| ≤ L i * |p i - q i|) :
    ∃ a : SupportedAction F,
      a ∈ (restrictedDecisionProblem exactDP F).Opt s ∧
      IsApproxOptimal (fun p => exactDP.utility p s) δ a.1 := by
  have hCoverF : HypercubeSupportCoversOnBox n F.support
      (fun _ => -Real.pi)
      (fun _ => Real.pi)
      (fun i => uniformArithmeticStep (segments i)) := by
    simpa [hSupport] using adaptiveArithmeticCenterSupport_cover_on_box n segments hSeg
  rcases hypercube_supportOnBox_yields_restricted_opt_approxAmbient
      n exactDP F s
      (fun _ => -Real.pi)
      (fun _ => Real.pi)
      L
      (fun i => uniformArithmeticStep (segments i))
      hCoverF
      hL hBox hOpt hLip with ⟨a, haOpt, hApprox⟩
  refine ⟨a, haOpt, ?_⟩
  exact approxOptimal_mono (fun p => exactDP.utility p s) hBudget hApprox

/-- Runtime-facing theorem: if the sampled support consists of one representative
    per certified hypercube cell and those cells cover the ambient optimum, then a
    delta-near-optimal supported action lies in the runtime support after the
    support-side uniform approximation step. -/
theorem hypercube_support_and_uniformApprox_yields_near_opt_in_runtime_support
    (n : ℕ)
    {S : Type v}
    (exactDP coarseDP : DecisionProblem (Fin n → ℝ) S)
    (F : SampledActionFamily (Fin n → ℝ))
    (s : S)
    (L halfWidths : Fin n → ℝ)
    [LinearOrder (SupportedAction F)]
    (hCover : HypercubeSupportCovers n F.support halfWidths)
    (hL : ∀ i, 0 ≤ L i)
    {pStar : Fin n → ℝ}
    (hOpt : IsOptimal (fun p => exactDP.utility p s) pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |exactDP.utility p s - exactDP.utility q s| ≤ L i * |p i - q i|)
    (δ : ℝ)
    (hApprox : ∀ a : SupportedAction F,
      |(restrictedDecisionProblem exactDP F).utility a s -
        (restrictedDecisionProblem coarseDP F).utility a s| ≤ δ)
    (hδ : 0 ≤ δ) :
    ∃ a : SupportedAction F,
      a ∈ (coherent_optimizer_witness_of_uniformApprox_top1
        (fun a => (restrictedDecisionProblem exactDP F).utility a s)
        (fun a => (restrictedDecisionProblem coarseDP F).utility a s)
        δ hApprox hδ).belief.selection.support
      ∧ IsApproxOptimal
          (fun p => exactDP.utility p s)
          (Finset.univ.sum (fun i => L i * halfWidths i)) a.1 := by
  rcases hypercube_support_yields_restricted_opt_approxAmbient
      n exactDP F s L halfWidths hCover hL hOpt hLip
      with ⟨aBest, hBestOpt, hApproxBest⟩
  refine ⟨aBest, ?_, hApproxBest⟩
  have hTop : aBest ∈ topKSet
      (fun a : SupportedAction F => (restrictedDecisionProblem exactDP F).utility a s) 1 := by
    exact optimal_mem_topKSet_one _ hBestOpt
  exact coherent_uniformApprox_exactTop1_subset_support
      (fun a : SupportedAction F => (restrictedDecisionProblem exactDP F).utility a s)
      (fun a : SupportedAction F => (restrictedDecisionProblem coarseDP F).utility a s)
      δ hApprox hδ hTop

/-- Runtime-support version of the adaptive arithmetic support theorem after the
    support-side coarse approximation step. -/
theorem adaptiveArithmeticSupport_and_uniformApprox_yields_near_opt_in_runtime_support_of_budget
    (n : ℕ)
    {S : Type v}
    (exactDP coarseDP : DecisionProblem (Fin n → ℝ) S)
    (F : SampledActionFamily (Fin n → ℝ))
    (s : S)
    (L : Fin n → ℝ)
    (segments : Fin n → ℕ)
    (δ_cover δ_coarse : ℝ)
    [LinearOrder (SupportedAction F)]
    (hSupport : F.support = adaptiveArithmeticCenterSupport n segments)
    (hSeg : ∀ i, 0 < segments i)
    (hL : ∀ i, 0 ≤ L i)
    (hBudget : AdaptiveSegmentBudget n L segments δ_cover)
    {pStar : Fin n → ℝ}
    (hBox : ∀ i, -Real.pi ≤ pStar i ∧ pStar i ≤ Real.pi)
    (hOpt : IsOptimal (fun p => exactDP.utility p s) pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |exactDP.utility p s - exactDP.utility q s| ≤ L i * |p i - q i|)
    (hApprox : ∀ a : SupportedAction F,
      |(restrictedDecisionProblem exactDP F).utility a s -
        (restrictedDecisionProblem coarseDP F).utility a s| ≤ δ_coarse)
    (hδ : 0 ≤ δ_coarse) :
    ∃ a : SupportedAction F,
      a ∈ (coherent_optimizer_witness_of_uniformApprox_top1
        (fun a => (restrictedDecisionProblem exactDP F).utility a s)
        (fun a => (restrictedDecisionProblem coarseDP F).utility a s)
        δ_coarse hApprox hδ).belief.selection.support
      ∧ IsApproxOptimal (fun p => exactDP.utility p s) δ_cover a.1 := by
  have hCoverF : HypercubeSupportCoversOnBox n F.support
      (fun _ => -Real.pi)
      (fun _ => Real.pi)
      (fun i => uniformArithmeticStep (segments i)) := by
    simpa [hSupport] using adaptiveArithmeticCenterSupport_cover_on_box n segments hSeg
  rcases hypercube_supportOnBox_yields_restricted_opt_approxAmbient
      n exactDP F s
      (fun _ => -Real.pi)
      (fun _ => Real.pi)
      L
      (fun i => uniformArithmeticStep (segments i))
      hCoverF
      hL hBox hOpt hLip with ⟨aBest, hBestOpt, hApproxBest⟩
  refine ⟨aBest, ?_, ?_⟩
  · have hTop : aBest ∈ topKSet
        (fun a : SupportedAction F => (restrictedDecisionProblem exactDP F).utility a s) 1 := by
      exact optimal_mem_topKSet_one _ hBestOpt
    exact coherent_uniformApprox_exactTop1_subset_support
        (fun a : SupportedAction F => (restrictedDecisionProblem exactDP F).utility a s)
        (fun a : SupportedAction F => (restrictedDecisionProblem coarseDP F).utility a s)
        δ_coarse hApprox hδ hTop
  · exact approxOptimal_mono (fun p => exactDP.utility p s) hBudget hApproxBest

theorem canonicalUniformSupport_yields_supported_approxOptimal
    (n : ℕ)
    (f : (Fin n → ℝ) → ℝ)
    (L : Fin n → ℝ)
    (δ : ℝ)
    (hL : ∀ i, 0 ≤ L i)
    (hδ : 0 < δ)
    {pStar : Fin n → ℝ}
    (hBox : ∀ i, -Real.pi ≤ pStar i ∧ pStar i ≤ Real.pi)
    (hOpt : IsOptimal f pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |f p - f q| ≤ L i * |p i - q i|) :
    ∃ center,
      center ∈ coordinateCenterSupport n (fun _ : Fin n => uniformArithmeticCentersPi (canonicalUniformSegmentsFromSlack n L δ))
      ∧ IsApproxOptimal f δ center := by
  let seg := canonicalUniformSegmentsFromSlack n L δ
  have hSeg : 0 < seg := canonicalUniformSegmentsFromSlack_pos n L δ
  have hBudget : AdaptiveSegmentBudget n L (fun _ : Fin n => seg) δ :=
    adaptiveBudget_of_canonicalUniformSegmentsFromSlack n L δ hL hδ
  simpa [adaptiveArithmeticCenterSupport, seg] using
    adaptiveArithmeticSupport_yields_supported_approxOptimal_of_budget
      n f L (fun _ : Fin n => seg) δ (fun _ => hSeg) hL hBudget hBox hOpt hLip

theorem canonicalAdaptiveSupport_yields_supported_approxOptimal
    (n : ℕ)
    (f : (Fin n → ℝ) → ℝ)
    (L : Fin n → ℝ)
    (δ : ℝ)
    (hn : 0 < n)
    (hL : ∀ i, 0 ≤ L i)
    (hδ : 0 < δ)
    {pStar : Fin n → ℝ}
    (hBox : ∀ i, -Real.pi ≤ pStar i ∧ pStar i ≤ Real.pi)
    (hOpt : IsOptimal f pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |f p - f q| ≤ L i * |p i - q i|) :
    ∃ center,
      center ∈ adaptiveArithmeticCenterSupport n (canonicalAdaptiveSegmentsFromSlack n L δ)
      ∧ IsApproxOptimal f δ center := by
  have hBudget : AdaptiveSegmentBudget n L (canonicalAdaptiveSegmentsFromSlack n L δ) δ :=
    adaptiveBudget_of_canonicalAdaptiveSegmentsFromSlack n L δ hn hL hδ
  exact adaptiveArithmeticSupport_yields_supported_approxOptimal_of_budget
    n f L (canonicalAdaptiveSegmentsFromSlack n L δ) δ
    (canonicalAdaptiveSegmentsFromSlack_pos n L δ) hL hBudget hBox hOpt hLip

/-- Sparse adaptive support that varies only the active torsion coordinates and
    keeps all inactive coordinates fixed to an anchor value. -/
noncomputable def sparseAdaptiveArithmeticCenterSupport
    (n : ℕ)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (segments : Fin n → ℕ) : Finset (Fin n → ℝ) :=
  coordinateCenterSupport n (fun i => if i ∈ active then uniformArithmeticCentersPi (segments i) else {anchor i})

noncomputable def sparseLowerBounds
    (n : ℕ)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ) : Fin n → ℝ :=
  fun i => if i ∈ active then -Real.pi else anchor i

noncomputable def sparseUpperBounds
    (n : ℕ)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ) : Fin n → ℝ :=
  fun i => if i ∈ active then Real.pi else anchor i

noncomputable def sparseHalfWidths
    (n : ℕ)
    (active : Finset (Fin n))
    (segments : Fin n → ℕ) : Fin n → ℝ :=
  fun i => if i ∈ active then uniformArithmeticStep (segments i) else 0

def ActiveSegmentBudget
    (n : ℕ)
    (active : Finset (Fin n))
    (L : Fin n → ℝ)
    (segments : Fin n → ℕ)
    (δ : ℝ) : Prop :=
  Finset.sum active (fun i => L i * uniformArithmeticStep (segments i)) ≤ δ

def maskedAnchorProjection
    (n : ℕ)
    (active : Finset (Fin n))
    (anchor p : Fin n → ℝ) : Fin n → ℝ :=
  fun i => if i ∈ active then p i else anchor i

def InvariantUnderMaskedProjection
    (n : ℕ)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (u : (Fin n → ℝ) → ℝ) : Prop :=
  ∀ p, u (maskedAnchorProjection n active anchor p) = u p

def DependsOnlyOnActiveCoords
    (n : ℕ)
    (active : Finset (Fin n))
    (u : (Fin n → ℝ) → ℝ) : Prop :=
  ∀ p q, (∀ i, i ∈ active → p i = q i) → u p = u q

def UtilityRepresentativeCover
    {A : Type u}
    (u : A → ℝ)
    (support reps : Finset A)
    (η : ℝ) : Prop :=
  ∀ a, a ∈ support → ∃ r, r ∈ reps ∧ |u a - u r| ≤ η

theorem sparseAdaptiveArithmeticCenterSupport_card
    (n : ℕ)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (segments : Fin n → ℕ)
    (hSeg : ∀ i, i ∈ active → 0 < segments i) :
    (sparseAdaptiveArithmeticCenterSupport n active anchor segments).card =
      ∏ i, if i ∈ active then (segments i + 1) else 1 := by
  unfold sparseAdaptiveArithmeticCenterSupport
  rw [coordinateCenterSupport_card]
  refine Finset.prod_congr rfl ?_
  intro i _
  by_cases hi : i ∈ active
  · simp [hi, uniformArithmeticCentersPi_card, hSeg i hi]
  · simp [hi]

theorem sparseAdaptiveArithmeticCenterSupport_card_mono_active
    (n : ℕ)
    {active₁ active₂ : Finset (Fin n)}
    (anchor : Fin n → ℝ)
    (segments : Fin n → ℕ)
    (hSeg : ∀ i, 0 < segments i)
    (hSubset : active₁ ⊆ active₂) :
    (sparseAdaptiveArithmeticCenterSupport n active₁ anchor segments).card ≤
      (sparseAdaptiveArithmeticCenterSupport n active₂ anchor segments).card := by
  rw [sparseAdaptiveArithmeticCenterSupport_card n active₁ anchor segments (fun i _ => hSeg i)]
  rw [sparseAdaptiveArithmeticCenterSupport_card n active₂ anchor segments (fun i _ => hSeg i)]
  calc
    ∏ i : Fin n, (if i ∈ active₁ then (segments i + 1) else 1)
        ≤ ∏ i : Fin n, if i ∈ active₂ then (segments i + 1) else 1 := by
          simpa using
            (Finset.prod_le_prod'
              (s := (Finset.univ : Finset (Fin n)))
              (f := fun i => if i ∈ active₁ then (segments i + 1) else 1)
              (g := fun i => if i ∈ active₂ then (segments i + 1) else 1)
              (fun i _ => by
                by_cases h1 : i ∈ active₁
                · have h2 : i ∈ active₂ := hSubset h1
                  simp [h1, h2]
                · by_cases h2 : i ∈ active₂
                  · simp [h1, h2]
                  · simp [h1, h2]))

theorem sparseAdaptiveArithmeticCenterSupport_cover_on_box
    (n : ℕ)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (segments : Fin n → ℕ)
    (hSeg : ∀ i, i ∈ active → 0 < segments i) :
    HypercubeSupportCoversOnBox n
      (sparseAdaptiveArithmeticCenterSupport n active anchor segments)
      (sparseLowerBounds n active anchor)
      (sparseUpperBounds n active anchor)
      (sparseHalfWidths n active segments) := by
  unfold sparseAdaptiveArithmeticCenterSupport sparseLowerBounds sparseUpperBounds sparseHalfWidths
  apply coordinatewise_cover_yields_hypercubeSupportOnBox
  intro i x hxLower hxUpper
  by_cases hi : i ∈ active
  · have hxLower' : -Real.pi ≤ x := by simpa [hi] using hxLower
    have hxUpper' : x ≤ Real.pi := by simpa [hi] using hxUpper
    rcases uniformArithmeticCentersPi_intervalCover (segments i) (hSeg i hi) x hxLower' hxUpper' with ⟨center, hcenter, hdist⟩
    exact ⟨center, by simp [hi, hcenter], by simpa [hi] using hdist⟩
  · have hxLower' : anchor i ≤ x := by simpa [hi] using hxLower
    have hxUpper' : x ≤ anchor i := by simpa [hi] using hxUpper
    have hEq : x = anchor i := le_antisymm hxUpper' hxLower'
    refine ⟨anchor i, by simp [hi], ?_⟩
    simpa [hi, hEq]

theorem activeSegmentBudget_mono
    (n : ℕ)
    {active₁ active₂ : Finset (Fin n)}
    (L : Fin n → ℝ)
    (segments : Fin n → ℕ)
    (δ : ℝ)
    (hNonneg : ∀ i, 0 ≤ L i * uniformArithmeticStep (segments i))
    (hSubset : active₁ ⊆ active₂)
    (hBudget : ActiveSegmentBudget n active₂ L segments δ) :
    ActiveSegmentBudget n active₁ L segments δ := by
  unfold ActiveSegmentBudget at *
  exact le_trans
    (Finset.sum_le_sum_of_subset_of_nonneg hSubset (fun i _ _ => hNonneg i))
    hBudget

theorem sparseBudget_as_hypercubeSlack
    (n : ℕ)
    (active : Finset (Fin n))
    (L : Fin n → ℝ)
    (segments : Fin n → ℕ)
    (δ : ℝ)
    (hBudget : ActiveSegmentBudget n active L segments δ) :
    Finset.univ.sum (fun i => L i * sparseHalfWidths n active segments i) ≤ δ := by
  unfold ActiveSegmentBudget sparseHalfWidths at *
  simpa using hBudget

theorem maskedAnchorProjection_eq_self_of_inactive_anchor
    (n : ℕ)
    (active : Finset (Fin n))
    (anchor p : Fin n → ℝ)
    (hInactive : ∀ i, i ∉ active → p i = anchor i) :
    maskedAnchorProjection n active anchor p = p := by
  funext i
  unfold maskedAnchorProjection
  by_cases hi : i ∈ active
  · simp [hi]
  · simp [hi, hInactive i hi]

theorem maskedAnchorProjection_preserves_active
    (n : ℕ)
    (active : Finset (Fin n))
    (anchor p : Fin n → ℝ)
    (i : Fin n)
    (hi : i ∈ active) :
    maskedAnchorProjection n active anchor p i = p i := by
  unfold maskedAnchorProjection
  simp [hi]

theorem maskedAnchorProjection_sets_inactive_to_anchor
    (n : ℕ)
    (active : Finset (Fin n))
    (anchor p : Fin n → ℝ)
    (i : Fin n)
    (hi : i ∉ active) :
    maskedAnchorProjection n active anchor p i = anchor i := by
  unfold maskedAnchorProjection
  simp [hi]

theorem maskedAnchorProjection_in_sparse_box
    (n : ℕ)
    (active : Finset (Fin n))
    (anchor p : Fin n → ℝ)
    (hBox : ∀ i, i ∈ active → -Real.pi ≤ p i ∧ p i ≤ Real.pi) :
    ∀ i,
      sparseLowerBounds n active anchor i ≤ maskedAnchorProjection n active anchor p i
      ∧ maskedAnchorProjection n active anchor p i ≤ sparseUpperBounds n active anchor i := by
  intro i
  unfold sparseLowerBounds sparseUpperBounds maskedAnchorProjection
  by_cases hi : i ∈ active
  · simpa [hi] using hBox i hi
  · simp [hi]

theorem maskedAnchorProjection_preserves_optimality
    (n : ℕ)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (u : (Fin n → ℝ) → ℝ)
    (hInv : InvariantUnderMaskedProjection n active anchor u)
    {pStar : Fin n → ℝ}
    (hOpt : IsOptimal u pStar) :
    IsOptimal u (maskedAnchorProjection n active anchor pStar) := by
  intro p
  have hOptP : u p ≤ u pStar := hOpt p
  have hEq : u (maskedAnchorProjection n active anchor pStar) = u pStar := hInv pStar
  linarith

theorem dependsOnlyOnActiveCoords_implies_projection_invariance
    (n : ℕ)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (u : (Fin n → ℝ) → ℝ)
    (hDep : DependsOnlyOnActiveCoords n active u) :
    InvariantUnderMaskedProjection n active anchor u := by
  intro p
  apply hDep
  intro i hi
  exact maskedAnchorProjection_preserves_active n active anchor p i hi

theorem utilityRepresentativeCover_mono
    {A : Type u}
    (u : A → ℝ)
    {support₁ support₂ reps₁ reps₂ : Finset A}
    {η₁ η₂ : ℝ}
    (hSupport : support₁ ⊆ support₂)
    (hReps : reps₁ ⊆ reps₂)
    (hη : η₁ ≤ η₂)
    (hCover : UtilityRepresentativeCover u support₁ reps₁ η₁) :
    UtilityRepresentativeCover u support₁ reps₂ η₂ := by
  intro a ha
  rcases hCover a ha with ⟨r, hr, hdist⟩
  exact ⟨r, hReps hr, le_trans hdist hη⟩

theorem zero_lipschitz_outside_active_implies_dependsOnlyOnActiveCoords
    (n : ℕ)
    (active : Finset (Fin n))
    (f : (Fin n → ℝ) → ℝ)
    (L : Fin n → ℝ)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |f p - f q| ≤ L i * |p i - q i|)
    (hZero : ∀ i, i ∉ active → L i = 0) :
    DependsOnlyOnActiveCoords n active f := by
  intro p q hActive
  have hAbs := per_dimension_lipschitz_bound n f L hLip p q
  have hSumZero : Finset.univ.sum (fun i => L i * |p i - q i|) = 0 := by
    apply Finset.sum_eq_zero
    intro i _
    by_cases hi : i ∈ active
    · have hEq : p i = q i := hActive i hi
      simp [hEq]
    · simp [hZero i hi]
  have hAbsZero : |f p - f q| ≤ 0 := by simpa [hSumZero] using hAbs
  have hAbs' := abs_le.mp hAbsZero
  linarith

theorem zero_lipschitz_outside_active_implies_projection_invariance
    (n : ℕ)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (f : (Fin n → ℝ) → ℝ)
    (L : Fin n → ℝ)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |f p - f q| ≤ L i * |p i - q i|)
    (hZero : ∀ i, i ∉ active → L i = 0) :
    InvariantUnderMaskedProjection n active anchor f := by
  apply dependsOnlyOnActiveCoords_implies_projection_invariance
  exact zero_lipschitz_outside_active_implies_dependsOnlyOnActiveCoords n active f L hLip hZero

theorem dependsOnlyOnActiveCoords_mono
    (n : ℕ)
    {active₁ active₂ : Finset (Fin n)}
    (u : (Fin n → ℝ) → ℝ)
    (hSubset : active₁ ⊆ active₂)
    (hDep : DependsOnlyOnActiveCoords n active₁ u) :
    DependsOnlyOnActiveCoords n active₂ u := by
  intro p q hEq
  apply hDep
  intro i hi
  exact hEq i (hSubset hi)

theorem invariantUnderMaskedProjection_add
    (n : ℕ)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (u v : (Fin n → ℝ) → ℝ)
    (hu : InvariantUnderMaskedProjection n active anchor u)
    (hv : InvariantUnderMaskedProjection n active anchor v) :
    InvariantUnderMaskedProjection n active anchor (fun p => u p + v p) := by
  intro p
  simp [hu p, hv p]

theorem dependsOnlyOnActiveCoords_add
    (n : ℕ)
    (active_u active_v : Finset (Fin n))
    (u v : (Fin n → ℝ) → ℝ)
    (hu : DependsOnlyOnActiveCoords n active_u u)
    (hv : DependsOnlyOnActiveCoords n active_v v) :
    DependsOnlyOnActiveCoords n (active_u ∪ active_v) (fun p => u p + v p) := by
  intro p q hEq
  have hu' : u p = u q := hu p q (fun i hi => hEq i (Finset.mem_union.mpr (Or.inl hi)))
  have hv' : v p = v q := hv p q (fun i hi => hEq i (Finset.mem_union.mpr (Or.inr hi)))
  linarith

theorem projectionInvariance_of_channelwise_dependence
    (n : ℕ)
    (active_u active_v : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (u v : (Fin n → ℝ) → ℝ)
    (hu : DependsOnlyOnActiveCoords n active_u u)
    (hv : DependsOnlyOnActiveCoords n active_v v) :
    InvariantUnderMaskedProjection n (active_u ∪ active_v) anchor (fun p => u p + v p) := by
  apply dependsOnlyOnActiveCoords_implies_projection_invariance
  exact dependsOnlyOnActiveCoords_add n active_u active_v u v hu hv

theorem representativeCover_preserves_supported_approxOptimal
    {A : Type u} [DecidableEq A]
    (u : A → ℝ)
    (support reps : Finset A)
    (δ η : ℝ)
    (hRepSubset : reps ⊆ support)
    (hCover : UtilityRepresentativeCover u support reps η)
    {a : A}
    (ha : a ∈ support)
    (hApprox : IsApproxOptimal u δ a) :
    ∃ r, r ∈ reps ∧ IsApproxOptimal u (δ + η) r := by
  rcases hCover a ha with ⟨r, hr, hClose⟩
  refine ⟨r, hr, ?_⟩
  intro a'
  have hApprox' : u a' ≤ u a + δ := hApprox a'
  have hClose' := abs_le.mp hClose
  linarith

theorem representativeCover_preserves_restricted_opt_approxAmbient
    {A : Type u} {S : Type v}
    [DecidableEq A]
    (dp : DecisionProblem A S)
    (support reps : Finset A)
    (s : S)
    (δ η : ℝ)
    (hRepSubset : reps ⊆ support)
    (hCover : UtilityRepresentativeCover (fun a => dp.utility a s) support reps η)
    {a : A}
    (ha : a ∈ support)
    (hApprox : IsApproxOptimal (fun a => dp.utility a s) δ a) :
    ∃ r, r ∈ reps ∧ IsApproxOptimal (fun a => dp.utility a s) (δ + η) r := by
  exact representativeCover_preserves_supported_approxOptimal
    (u := fun a => dp.utility a s) support reps δ η hRepSubset hCover ha hApprox

theorem sparseAdaptiveSupport_yields_supported_approxOptimal_of_budget
    (n : ℕ)
    (f : (Fin n → ℝ) → ℝ)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (L : Fin n → ℝ)
    (segments : Fin n → ℕ)
    (δ : ℝ)
    (hSeg : ∀ i, i ∈ active → 0 < segments i)
    (hL : ∀ i, 0 ≤ L i)
    (hBudget : ActiveSegmentBudget n active L segments δ)
    {pStar : Fin n → ℝ}
    (hBox : ∀ i, i ∈ active → -Real.pi ≤ pStar i ∧ pStar i ≤ Real.pi)
    (hAnchor : ∀ i, i ∉ active → pStar i = anchor i)
    (hOpt : IsOptimal f pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |f p - f q| ≤ L i * |p i - q i|) :
    ∃ center, center ∈ sparseAdaptiveArithmeticCenterSupport n active anchor segments ∧
      IsApproxOptimal f δ center := by
  rcases hypercube_supportOnBox_yields_supported_approxOptimal
      n f
      (sparseAdaptiveArithmeticCenterSupport n active anchor segments)
      (sparseLowerBounds n active anchor)
      (sparseUpperBounds n active anchor)
      L
      (sparseHalfWidths n active segments)
      (sparseAdaptiveArithmeticCenterSupport_cover_on_box n active anchor segments hSeg)
      hL
      (by
        intro i
        unfold sparseLowerBounds sparseUpperBounds
        by_cases hi : i ∈ active
        · simpa [hi] using hBox i hi
        · simp [hi, hAnchor i hi])
      hOpt hLip with ⟨center, hcenter, hApprox⟩
  refine ⟨center, hcenter, ?_⟩
  exact approxOptimal_mono f (sparseBudget_as_hypercubeSlack n active L segments δ hBudget) hApprox

theorem sparseAdaptiveSupport_yields_supported_approxOptimal_of_projection_invariance
    (n : ℕ)
    (f : (Fin n → ℝ) → ℝ)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (L : Fin n → ℝ)
    (segments : Fin n → ℕ)
    (δ : ℝ)
    (hSeg : ∀ i, i ∈ active → 0 < segments i)
    (hL : ∀ i, 0 ≤ L i)
    (hBudget : ActiveSegmentBudget n active L segments δ)
    {pStar : Fin n → ℝ}
    (hBox : ∀ i, i ∈ active → -Real.pi ≤ pStar i ∧ pStar i ≤ Real.pi)
    (hInv : InvariantUnderMaskedProjection n active anchor f)
    (hOpt : IsOptimal f pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |f p - f q| ≤ L i * |p i - q i|) :
    ∃ center, center ∈ sparseAdaptiveArithmeticCenterSupport n active anchor segments ∧
      IsApproxOptimal f δ center := by
  let pProj := maskedAnchorProjection n active anchor pStar
  have hOptProj : IsOptimal f pProj := maskedAnchorProjection_preserves_optimality n active anchor f hInv hOpt
  have hAnchorProj : ∀ i, i ∉ active → pProj i = anchor i := by
    intro i hi
    exact maskedAnchorProjection_sets_inactive_to_anchor n active anchor pStar i hi
  have hBoxProj : ∀ i, i ∈ active → -Real.pi ≤ pProj i ∧ pProj i ≤ Real.pi := by
    intro i hi
    simpa [pProj, maskedAnchorProjection, hi] using hBox i hi
  exact sparseAdaptiveSupport_yields_supported_approxOptimal_of_budget
    n f active anchor L segments δ hSeg hL hBudget hBoxProj hAnchorProj hOptProj hLip

theorem sparseAdaptiveSupport_yields_restricted_opt_approxAmbient_of_budget
    (n : ℕ)
    {S : Type v}
    (exactDP : DecisionProblem (Fin n → ℝ) S)
    (F : SampledActionFamily (Fin n → ℝ))
    (s : S)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (L : Fin n → ℝ)
    (segments : Fin n → ℕ)
    (δ : ℝ)
    (hSupport : F.support = sparseAdaptiveArithmeticCenterSupport n active anchor segments)
    (hSeg : ∀ i, i ∈ active → 0 < segments i)
    (hL : ∀ i, 0 ≤ L i)
    (hBudget : ActiveSegmentBudget n active L segments δ)
    {pStar : Fin n → ℝ}
    (hBox : ∀ i, i ∈ active → -Real.pi ≤ pStar i ∧ pStar i ≤ Real.pi)
    (hAnchor : ∀ i, i ∉ active → pStar i = anchor i)
    (hOpt : IsOptimal (fun p => exactDP.utility p s) pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |exactDP.utility p s - exactDP.utility q s| ≤ L i * |p i - q i|) :
    ∃ a : SupportedAction F,
      a ∈ (restrictedDecisionProblem exactDP F).Opt s ∧
      IsApproxOptimal (fun p => exactDP.utility p s) δ a.1 := by
  have hCoverF : HypercubeSupportCoversOnBox n F.support
      (sparseLowerBounds n active anchor)
      (sparseUpperBounds n active anchor)
      (sparseHalfWidths n active segments) := by
    simpa [hSupport] using sparseAdaptiveArithmeticCenterSupport_cover_on_box n active anchor segments hSeg
  rcases hypercube_supportOnBox_yields_restricted_opt_approxAmbient
      n exactDP F s
      (sparseLowerBounds n active anchor)
      (sparseUpperBounds n active anchor)
      L
      (sparseHalfWidths n active segments)
      hCoverF
      hL
      (by
        intro i
        unfold sparseLowerBounds sparseUpperBounds
        by_cases hi : i ∈ active
        · simpa [hi] using hBox i hi
        · simp [hi, hAnchor i hi])
      hOpt hLip with ⟨a, haOpt, hApprox⟩
  refine ⟨a, haOpt, ?_⟩
  exact approxOptimal_mono (fun p => exactDP.utility p s)
    (sparseBudget_as_hypercubeSlack n active L segments δ hBudget) hApprox

theorem sparseAdaptiveSupport_and_uniformApprox_yields_near_opt_in_runtime_support_of_budget
    (n : ℕ)
    {S : Type v}
    (exactDP coarseDP : DecisionProblem (Fin n → ℝ) S)
    (F : SampledActionFamily (Fin n → ℝ))
    (s : S)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (L : Fin n → ℝ)
    (segments : Fin n → ℕ)
    (δ_cover δ_coarse : ℝ)
    [LinearOrder (SupportedAction F)]
    (hSupport : F.support = sparseAdaptiveArithmeticCenterSupport n active anchor segments)
    (hSeg : ∀ i, i ∈ active → 0 < segments i)
    (hL : ∀ i, 0 ≤ L i)
    (hBudget : ActiveSegmentBudget n active L segments δ_cover)
    {pStar : Fin n → ℝ}
    (hBox : ∀ i, i ∈ active → -Real.pi ≤ pStar i ∧ pStar i ≤ Real.pi)
    (hAnchor : ∀ i, i ∉ active → pStar i = anchor i)
    (hOpt : IsOptimal (fun p => exactDP.utility p s) pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |exactDP.utility p s - exactDP.utility q s| ≤ L i * |p i - q i|)
    (hApprox : ∀ a : SupportedAction F,
      |(restrictedDecisionProblem exactDP F).utility a s -
        (restrictedDecisionProblem coarseDP F).utility a s| ≤ δ_coarse)
    (hδ : 0 ≤ δ_coarse) :
    ∃ a : SupportedAction F,
      a ∈ (coherent_optimizer_witness_of_uniformApprox_top1
        (fun a => (restrictedDecisionProblem exactDP F).utility a s)
        (fun a => (restrictedDecisionProblem coarseDP F).utility a s)
        δ_coarse hApprox hδ).belief.selection.support
      ∧ IsApproxOptimal (fun p => exactDP.utility p s) δ_cover a.1 := by
  rcases sparseAdaptiveSupport_yields_restricted_opt_approxAmbient_of_budget
      n exactDP F s active anchor L segments δ_cover hSupport hSeg hL hBudget hBox hAnchor hOpt hLip
      with ⟨aBest, hBestOpt, hApproxBest⟩
  refine ⟨aBest, ?_, hApproxBest⟩
  have hTop : aBest ∈ topKSet
      (fun a : SupportedAction F => (restrictedDecisionProblem exactDP F).utility a s) 1 := by
    exact optimal_mem_topKSet_one _ hBestOpt
  exact coherent_uniformApprox_exactTop1_subset_support
      (fun a : SupportedAction F => (restrictedDecisionProblem exactDP F).utility a s)
      (fun a : SupportedAction F => (restrictedDecisionProblem coarseDP F).utility a s)
      δ_coarse hApprox hδ hTop

noncomputable def canonicalSparseAdaptiveSegmentsFromSlack
    (n : ℕ)
    (active : Finset (Fin n))
    (L : Fin n → ℝ)
    (δ : ℝ) : Fin n → ℕ :=
  fun i => if i ∈ active then max 1 (Nat.ceil ((2 * Real.pi * (active.card : ℝ) * L i) / δ)) else 1

theorem canonicalSparseAdaptiveSegmentsFromSlack_pos
    (n : ℕ)
    (active : Finset (Fin n))
    (L : Fin n → ℝ)
    (δ : ℝ)
    (i : Fin n) :
    0 < canonicalSparseAdaptiveSegmentsFromSlack n active L δ i := by
  unfold canonicalSparseAdaptiveSegmentsFromSlack
  by_cases hi : i ∈ active
  · simp [hi]
  · simp [hi]

theorem canonicalSparseAdaptiveSegmentsFromSlack_eq_one_of_inactive
    (n : ℕ)
    (active : Finset (Fin n))
    (L : Fin n → ℝ)
    (δ : ℝ)
    {i : Fin n}
    (hi : i ∉ active) :
    canonicalSparseAdaptiveSegmentsFromSlack n active L δ i = 1 := by
  simp [canonicalSparseAdaptiveSegmentsFromSlack, hi]

theorem activeSegmentBudget_of_canonicalSparseAdaptiveSegmentsFromSlack
    (n : ℕ)
    (active : Finset (Fin n))
    (L : Fin n → ℝ)
    (δ : ℝ)
    (hL : ∀ i, 0 ≤ L i)
    (hδ : 0 < δ) :
    ActiveSegmentBudget n active L (canonicalSparseAdaptiveSegmentsFromSlack n active L δ) δ := by
  by_cases hEmpty : active = ∅
  · subst hEmpty
    unfold ActiveSegmentBudget
    simp [le_of_lt hδ]
  · have hActNonempty : active.Nonempty := Finset.nonempty_iff_ne_empty.mpr hEmpty
    have hActCardPosNat : 0 < active.card := Finset.card_pos.mpr hActNonempty
    have hActCardPos : 0 < (active.card : ℝ) := by exact_mod_cast hActCardPosNat
    have hTerm : ∀ i ∈ active,
        L i * uniformArithmeticStep (canonicalSparseAdaptiveSegmentsFromSlack n active L δ i) ≤ δ / active.card := by
      intro i hi
      let seg := canonicalSparseAdaptiveSegmentsFromSlack n active L δ i
      have hSegPosNat : 0 < seg := canonicalSparseAdaptiveSegmentsFromSlack_pos n active L δ i
      have hSegEq : seg = max 1 (Nat.ceil ((2 * Real.pi * (active.card : ℝ) * L i) / δ)) := by
        simp [canonicalSparseAdaptiveSegmentsFromSlack, seg, hi]
      have hLower : (2 * Real.pi * (active.card : ℝ) * L i) / δ ≤ (seg : ℝ) := by
        rw [hSegEq]
        calc
          (2 * Real.pi * (active.card : ℝ) * L i) / δ ≤ ↑⌈(2 * Real.pi * (active.card : ℝ) * L i) / δ⌉₊ := by
            exact Nat.le_ceil _
          _ ≤ ↑(max 1 ⌈(2 * Real.pi * (active.card : ℝ) * L i) / δ⌉₊) := by
            exact_mod_cast (le_max_right 1 ⌈(2 * Real.pi * (active.card : ℝ) * L i) / δ⌉₊)
      have hDiv : (2 * Real.pi * (active.card : ℝ) * L i) / (seg : ℝ) ≤ δ := by
        apply (_root_.div_le_iff₀ (show 0 < (seg : ℝ) by exact_mod_cast hSegPosNat)).2
        have hMul : 2 * Real.pi * (active.card : ℝ) * L i ≤ δ * (seg : ℝ) := by
          simpa [mul_comm, mul_left_comm, mul_assoc] using (_root_.div_le_iff₀ hδ).mp hLower
        exact hMul
      rw [_root_.le_div_iff₀ hActCardPos]
      have hEq : L i * uniformArithmeticStep seg * active.card = (2 * Real.pi * (active.card : ℝ) * L i) / (seg : ℝ) := by
        unfold uniformArithmeticStep
        field_simp [show (seg : ℝ) ≠ 0 by positivity]
      rw [hEq]
      exact hDiv
    unfold ActiveSegmentBudget
    have hsum : Finset.sum active (fun i => L i * uniformArithmeticStep (canonicalSparseAdaptiveSegmentsFromSlack n active L δ i))
        ≤ Finset.sum active (fun _ => δ / active.card) := by
      refine Finset.sum_le_sum ?_
      intro i hi
      exact hTerm i hi
    have hconst : Finset.sum active (fun _ => δ / active.card) = δ := by
      rw [Finset.sum_const, nsmul_eq_mul, Finset.card_eq_sum_ones]
      have : (active.card : ℝ) * (δ / active.card) = δ := by
        field_simp [show (active.card : ℝ) ≠ 0 by exact_mod_cast hActCardPosNat.ne']
      simpa using this
    exact hsum.trans_eq hconst

theorem canonicalSparseAdaptiveSupport_yields_supported_approxOptimal_of_projection_invariance
    (n : ℕ)
    (f : (Fin n → ℝ) → ℝ)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (L : Fin n → ℝ)
    (δ : ℝ)
    (hL : ∀ i, 0 ≤ L i)
    (hδ : 0 < δ)
    {pStar : Fin n → ℝ}
    (hBox : ∀ i, i ∈ active → -Real.pi ≤ pStar i ∧ pStar i ≤ Real.pi)
    (hInv : InvariantUnderMaskedProjection n active anchor f)
    (hOpt : IsOptimal f pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |f p - f q| ≤ L i * |p i - q i|) :
    ∃ center,
      center ∈ sparseAdaptiveArithmeticCenterSupport n active anchor (canonicalSparseAdaptiveSegmentsFromSlack n active L δ)
      ∧ IsApproxOptimal f δ center := by
  have hBudget : ActiveSegmentBudget n active L (canonicalSparseAdaptiveSegmentsFromSlack n active L δ) δ :=
    activeSegmentBudget_of_canonicalSparseAdaptiveSegmentsFromSlack n active L δ hL hδ
  exact sparseAdaptiveSupport_yields_supported_approxOptimal_of_projection_invariance
    n f active anchor L (canonicalSparseAdaptiveSegmentsFromSlack n active L δ) δ
    (fun i hi => canonicalSparseAdaptiveSegmentsFromSlack_pos n active L δ i)
    hL hBudget hBox hInv hOpt hLip

theorem sparseAdaptiveSupport_yields_supported_approxOptimal_of_zero_lipschitz_outside_active
    (n : ℕ)
    (f : (Fin n → ℝ) → ℝ)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (L : Fin n → ℝ)
    (segments : Fin n → ℕ)
    (δ : ℝ)
    (hSeg : ∀ i, i ∈ active → 0 < segments i)
    (hL : ∀ i, 0 ≤ L i)
    (hBudget : ActiveSegmentBudget n active L segments δ)
    {pStar : Fin n → ℝ}
    (hBox : ∀ i, i ∈ active → -Real.pi ≤ pStar i ∧ pStar i ≤ Real.pi)
    (hOpt : IsOptimal f pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |f p - f q| ≤ L i * |p i - q i|)
    (hZero : ∀ i, i ∉ active → L i = 0) :
    ∃ center, center ∈ sparseAdaptiveArithmeticCenterSupport n active anchor segments ∧
      IsApproxOptimal f δ center := by
  apply sparseAdaptiveSupport_yields_supported_approxOptimal_of_projection_invariance
    n f active anchor L segments δ hSeg hL hBudget hBox
    (zero_lipschitz_outside_active_implies_projection_invariance n active anchor f L hLip hZero)
    hOpt hLip

theorem canonicalSparseAdaptiveSupport_yields_supported_approxOptimal_of_zero_lipschitz_outside_active
    (n : ℕ)
    (f : (Fin n → ℝ) → ℝ)
    (active : Finset (Fin n))
    (anchor : Fin n → ℝ)
    (L : Fin n → ℝ)
    (δ : ℝ)
    (hL : ∀ i, 0 ≤ L i)
    (hδ : 0 < δ)
    {pStar : Fin n → ℝ}
    (hBox : ∀ i, i ∈ active → -Real.pi ≤ pStar i ∧ pStar i ≤ Real.pi)
    (hOpt : IsOptimal f pStar)
    (hLip : ∀ i : Fin n, ∀ p q : Fin n → ℝ,
      (∀ j, j ≠ i → p j = q j) →
      |f p - f q| ≤ L i * |p i - q i|)
    (hZero : ∀ i, i ∉ active → L i = 0) :
    ∃ center,
      center ∈ sparseAdaptiveArithmeticCenterSupport n active anchor (canonicalSparseAdaptiveSegmentsFromSlack n active L δ)
      ∧ IsApproxOptimal f δ center := by
  have hBudget : ActiveSegmentBudget n active L (canonicalSparseAdaptiveSegmentsFromSlack n active L δ) δ :=
    activeSegmentBudget_of_canonicalSparseAdaptiveSegmentsFromSlack n active L δ hL hδ
  exact sparseAdaptiveSupport_yields_supported_approxOptimal_of_zero_lipschitz_outside_active
    n f active anchor L (canonicalSparseAdaptiveSegmentsFromSlack n active L δ) δ
    (fun i hi => canonicalSparseAdaptiveSegmentsFromSlack_pos n active L δ i)
    hL hBudget hBox hOpt hLip hZero

-- ---------------------------------------------------------------------------
-- Theorem: Support quotient preserves coverage (deduplication)
-- ---------------------------------------------------------------------------

/-- Theorem CSC20: Certified Support Deduplication.
    
    If a support family S ε-covers the space, and we form a quotient support S'
    by merging all points within distance τ, then S' (ε+τ)-covers the space.
    The resulting energy optimality slack increases by at most L * τ. -/
theorem dedup_preserves_coverage
    {A : Type u} [PseudoMetricSpace A] [DecidableEq A]
    (S : Finset A) (ε : ℝ)
    (h_cov : SupportCovers S ε)
    (τ : ℝ)
    (S' : Finset A) 
    (h_dedup : ∀ s ∈ S, ∃ s' ∈ S', dist s s' ≤ τ) :
    SupportCovers S' (ε + τ) := by
  intro p
  rcases h_cov p with ⟨s, hs, h_dist⟩
  rcases h_dedup s hs with ⟨s', hs', h_tau⟩
  refine ⟨s', hs', ?_⟩
  calc dist p s' ≤ dist p s + dist s s' := dist_triangle _ _ _
    _ ≤ ε + τ := add_le_add h_dist h_tau
    _ = ε + τ := by ring

-- Note: greedy_quotient_preserves_coverage was removed because it depended on
-- the undefined `greedy_τ_quotient` algorithm.  The parametric version
-- `dedup_preserves_coverage` (above) already certifies the coverage slack of
-- any dedup map; a concrete algorithm can instantiate that theorem directly.

-- ---------------------------------------------------------------------------
-- Theorem: Locality bound for torsion activity
-- ---------------------------------------------------------------------------

/-- Theorem LP1: Score Locality Interaction Radius (Axiom-Free).
    
    If a scoring function evaluates to exactly 0 beyond a cutoff radius R_c, 
    and a kinematic rotation keeps an atom strictly beyond R_c from the receptor 
    for ALL possible rotation angles, then that torsion has a Lipschitz constant
    of exactly 0 with respect to that receptor atom. -/
theorem torsion_locality_radius
    {Param Coord : Type*} [PseudoMetricSpace Param] [PseudoMetricSpace Coord]
    (energy : ℝ → ℝ) 
    (R_c : ℝ)
    -- The Cutoff Physics: Energy is 0 beyond R_c
    (h_cutoff : ∀ r > R_c, energy r = 0)
    -- The Kinematics: A function mapping a torsion angle to a 3D coordinate
    (kinematics : Param → Coord)
    (receptor_atom : Coord)
    -- The Swept Volume Bound: The rotating atom NEVER enters the cutoff radius
    (h_far : ∀ θ : Param, dist receptor_atom (kinematics θ) > R_c) :
    LipschitzWith 0 (fun θ => energy (dist receptor_atom (kinematics θ))) := by
  apply LipschitzWith.of_dist_le_mul
  intro θ₁ θ₂
  -- Evaluate the energy at the two angles. 
  -- Because the distance is always > R_c, the energy is exactly 0.
  have h_e1 : energy (dist receptor_atom (kinematics θ₁)) = 0 := h_cutoff _ (h_far θ₁)
  have h_e2 : energy (dist receptor_atom (kinematics θ₂)) = 0 := h_cutoff _ (h_far θ₂)
  -- Substitute the zeros into the distance calculation
  -- dist(0, 0) = 0, which is ≤ 0 * dist(θ₁, θ₂)
  rw [h_e1, h_e2, dist_self]
  simp

/-- Corollary: Multiple receptor atoms.
    
    If ALL rotating atoms are beyond cutoff from ALL receptor atoms,
    then the total energy contribution of that torsion is exactly 0,
    i.e., the torsion is inactive for that pose. -/
theorem torsion_locality_radius_multi_atom
    {Param Coord : Type*} [PseudoMetricSpace Param] [PseudoMetricSpace Coord]
    (energy : ℝ → ℝ) 
    (R_c : ℝ)
    (h_cutoff : ∀ r > R_c, energy r = 0)
    (kinematics : Param → Finset Coord)  -- Maps torsion angle to set of rotating atom positions
    (receptor_atoms : Finset Coord)
    (h_far : ∀ θ : Param, ∀ a ∈ kinematics θ, ∀ b ∈ receptor_atoms, dist a b > R_c) :
    LipschitzWith 0 (fun θ => ∑ a ∈ kinematics θ, ∑ b ∈ receptor_atoms, energy (dist a b)) := by
  apply LipschitzWith.of_dist_le_mul
  intro θ₁ θ₂
  -- All energy terms are 0 due to h_far and h_cutoff
  have h_zero : ∀ θ : Param, ∑ a ∈ kinematics θ, ∑ b ∈ receptor_atoms, energy (dist a b) = 0 := by
    intro θ
    apply Finset.sum_eq_zero
    intro a ha
    apply Finset.sum_eq_zero
    intro b hb
    exact h_cutoff (dist a b) (h_far θ a ha b hb)
  rw [h_zero θ₁, h_zero θ₂, dist_self]
  simp

/-- Theorem CSC47: Runtime torsion inactivity radius for cutoff score families.

    If every rotating atom of a torsion remains beyond the largest runtime
    cutoff radius from every receptor atom, then the combined cutoff Gaussian,
    cutoff Lennard-Jones, and cutoff Coulomb contribution of that torsion is
    identically zero. Hence the torsion is 0-Lipschitz for that score slice. -/
noncomputable def runtimeTorsionCutoffScore
    {Coord : Type*} [PseudoMetricSpace Coord]
    (w β R_gaussian : ℝ)
    (ε_lj σ R_lj : ℝ)
    (q_i q_j R_coulomb : ℝ)
    (torsion_kinematics : ℝ → Finset Coord)
    (receptor_atoms : Finset Coord)
    (θ : ℝ) : ℝ :=
  (torsion_kinematics θ).sum fun a =>
    receptor_atoms.sum fun b =>
      cutoffGaussianScore w β R_gaussian (dist a b) +
      cutoffLJScore ε_lj σ R_lj (dist a b) +
      cutoffCoulombScore q_i q_j R_coulomb (dist a b)

theorem runtime_torsion_inactivity_radius
    {Coord : Type*} [PseudoMetricSpace Coord]
    (w β R_gaussian : ℝ)
    (ε_lj σ R_lj : ℝ)
    (q_i q_j R_coulomb : ℝ)
    (torsion_kinematics : ℝ → Finset Coord)
    (receptor_atoms : Finset Coord)
    (h_far : ∀ θ : ℝ, ∀ a ∈ torsion_kinematics θ, ∀ b ∈ receptor_atoms,
      @dist Coord _ a b > max R_gaussian (max R_lj R_coulomb)) :
    LipschitzWith 0
      (runtimeTorsionCutoffScore w β R_gaussian ε_lj σ R_lj q_i q_j R_coulomb
        torsion_kinematics receptor_atoms) := by
  apply LipschitzWith.of_dist_le_mul
  intro θ₁ θ₂
  have h_zero : ∀ θ,
      runtimeTorsionCutoffScore w β R_gaussian ε_lj σ R_lj q_i q_j R_coulomb
        torsion_kinematics receptor_atoms θ = 0 := by
    intro θ
    unfold runtimeTorsionCutoffScore
    apply Finset.sum_eq_zero
    intro a ha
    apply Finset.sum_eq_zero
    intro b hb
    have h_dist : @dist Coord _ a b > max R_gaussian (max R_lj R_coulomb) := h_far θ a ha b hb
    have h_gaussian : cutoffGaussianScore w β R_gaussian (@dist Coord _ a b) = 0 := by
      unfold cutoffGaussianScore
      apply if_neg
      exact not_lt.mpr (le_trans (le_max_left _ _) (le_of_lt h_dist))
    have h_lj : cutoffLJScore ε_lj σ R_lj (@dist Coord _ a b) = 0 := by
      unfold cutoffLJScore
      apply if_neg
      exact not_lt.mpr (le_trans (le_trans (le_max_left _ _) (le_max_right _ _)) (le_of_lt h_dist))
    have h_coulomb : cutoffCoulombScore q_i q_j R_coulomb (@dist Coord _ a b) = 0 := by
      unfold cutoffCoulombScore
      apply if_neg
      exact not_lt.mpr (le_trans (le_trans (le_max_right _ _) (le_max_right _ _)) (le_of_lt h_dist))
    simp [h_gaussian, h_lj, h_coulomb]
  rw [h_zero θ₁, h_zero θ₂, dist_self]
  simp

/-- Theorem CS11: B&B stopping radius yields uniform coverage.
    
    If branch-and-bound stops when all active cells have radius < ε,
    then the set of evaluated cell centers forms an ε-net of the
    feasible parameter space (modulo pruning).
    
    This theorem bridges the runtime's `config.min_cell_radius` to the
    abstract ε-coverage requirement. -/
theorem bb_stopping_radius_yields_coverage
    {Param : Type*} [PseudoMetricSpace Param]
    (feasible : Set Param)
    (center : Param → Param)  -- Maps parameters to cell centers
    (radius : Param → ℝ)      -- Cell radius function
    (ε : ℝ)
    (h_stop : ∀ p ∈ feasible, radius p < ε)
    (h_center_in_cell : ∀ p ∈ feasible, dist (center p) p ≤ radius p)
    (support : Finset Param)  -- Set of evaluated cell centers
    (h_support : ∀ p ∈ feasible, center p ∈ support) :
    SupportCoversOn support ε feasible := by
  intro p hp
  exact ⟨center p, h_support p hp, by
    calc dist p (center p)
        = dist (center p) p := dist_comm _ _
      _ ≤ radius p := h_center_in_cell p hp
      _ ≤ ε := le_of_lt (h_stop p hp)⟩

-- ---------------------------------------------------------------------------
-- Theorem CSC50: Parameter-space resolution to coordinate-space resolution
-- ---------------------------------------------------------------------------

/-- Theorem CSC50: Parameter-space resolution to coordinate-space resolution.
    
    If the kinematic map from parameter space (torsions) to coordinate space
    is K-Lipschitz, then any set of parameter points that (ε/K)-covers the
    parameter space will induce a coordinate set that ε-covers the
    reachable coordinate space.
    
    This justifies the Python implementation's rule:
    `min_cell_radius <= target_rmsd / kinematics.lipschitz_constant`. -/
theorem min_cell_radius_derivation
    {Param Coord : Type*} [PseudoMetricSpace Param] [PseudoMetricSpace Coord]
    (kinematics : Param → Coord)
    (K : NNReal) (hK : LipschitzWith K kinematics)
    (ε : ℝ) (hε : 0 < ε)
    (support : Finset Param)
    (feasible : Set Param)
    (hK_pos : 0 < K)
    (hCover : SupportCoversOn support (ε / K) feasible) :
    let coord_feasible := kinematics '' feasible
    ∀ c ∈ coord_feasible, ∃ p ∈ support, dist c (kinematics p) ≤ ε := by
  show ∀ c ∈ kinematics '' feasible, ∃ p ∈ support, dist c (kinematics p) ≤ ε
  intro c hc
  simp_rw [Set.mem_image] at hc
  rcases hc with ⟨q, hq_feas, rfl⟩
  rcases hCover q hq_feas with ⟨p, hp_supp, h_dist⟩
  have hK_pos_r : (0 : ℝ) < ↑K := by positivity
  refine ⟨p, hp_supp, ?_⟩
  calc dist (kinematics q) (kinematics p)
      ≤ ↑K * dist q p := hK.dist_le_mul q p
    _ ≤ ↑K * (ε / ↑K) := mul_le_mul_of_nonneg_left h_dist hK_pos_r.le
    _ = ε := mul_div_cancel₀ ε hK_pos_r.ne'

-- ---------------------------------------------------------------------------
-- Theorem CSC60: Worst-case conformer support budget
-- ---------------------------------------------------------------------------

/-- Theorem CSC60: Worst-Case Conformer Support Budget.

    For an N-dimensional torsion space [-π, π]^N, a uniform grid
    requires at most (⌈2π / ε⌉₊ + 1)^N grid points to ε-cover the space
    in every coordinate.

    Note: the segment count is ⌈2π/ε⌉₊ (not ⌈π/ε⌉₊) because the
    existing `uniformArithmeticCentersPi` coverage theorem certifies
    radius equal to the full grid spacing 2π/S, so achieving radius ≤ ε
    requires S ≥ 2π/ε.

    This establishes the absolute mathematical ceiling for `max_cells`. -/
theorem max_uniform_cells_bound
    (N : ℕ)
    (ε : ℝ)
    (hε : 0 < ε) :
    ∃ (support : Finset (Fin N → ℝ)),
      -- The support covers the bounded torsion space
      (∀ p : Fin N → ℝ, (∀ i, -Real.pi ≤ p i ∧ p i ≤ Real.pi) →
        ∃ c ∈ support, ∀ i, |p i - c i| ≤ ε) ∧
      -- The support cardinality is strictly bounded
      support.card ≤ (⌈2 * Real.pi / ε⌉₊ + 1) ^ N := by
  -- Choose S = ⌈2π/ε⌉₊ segments so that grid spacing = 2π/S ≤ ε
  let S := ⌈2 * Real.pi / ε⌉₊
  have hS_pos : 0 < S := (Nat.ceil_pos).2 (by positivity)
  refine ⟨coordinateCenterSupport N (fun _ : Fin N => uniformArithmeticCentersPi S), ?_, ?_⟩
  -- Part A: Coverage — every point in [-π,π]^N has a nearby grid center
  · intro p hp
    rcases uniformArithmeticCentersPi_cover_on_box N S hS_pos p hp with ⟨c, hcMem, hcDist⟩
    -- The grid spacing 2π/S ≤ ε because S = ⌈2π/ε⌉₊ ≥ 2π/ε
    have hStep : uniformArithmeticStep S ≤ ε := by
      unfold uniformArithmeticStep
      have hSR : (0 : ℝ) < (↑S : ℝ) := Nat.cast_pos.mpr hS_pos
      rw [_root_.div_le_iff₀ hSR]
      have := (_root_.div_le_iff₀ hε).mp (Nat.le_ceil (2 * Real.pi / ε))
      linarith [mul_comm (↑S : ℝ) ε]
    exact ⟨c, hcMem, fun i => le_trans (hcDist i) hStep⟩
  -- Part B: Cardinality — tensor product of (S+1)-element 1D supports
  · exact le_of_eq (uniformArithmeticCentersPi_tensor_card N S hS_pos)

/-- The binary branch-and-bound cell budget induced by a certified finite support
    of size `B`. A full binary refinement tree with `B` leaf cells evaluates at
    most `2 * B - 1` cell centers. Combined with `max_uniform_cells_bound`, this
    yields a theorem-backed absolute ceiling for the runtime `max_cells` budget. -/
theorem max_binary_cells_bound
    (N : ℕ)
    (ε : ℝ)
    (hε : 0 < ε) :
    ∃ (support : Finset (Fin N → ℝ)),
      (∀ p : Fin N → ℝ, (∀ i, -Real.pi ≤ p i ∧ p i ≤ Real.pi) →
        ∃ c ∈ support, ∀ i, |p i - c i| ≤ ε) ∧
      (2 * support.card - 1 ≤ 2 * ((⌈2 * Real.pi / ε⌉₊ + 1) ^ N) - 1) := by
  rcases max_uniform_cells_bound N ε hε with ⟨support, hCover, hCard⟩
  refine ⟨support, hCover, ?_⟩
  omega

end ConformerSupportCoverage
end Tractability
end DecisionQuotient
