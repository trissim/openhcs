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

namespace DecisionQuotient
namespace Tractability
namespace ConformerSupportCoverage

open SampledDocking
open FiniteTopK
open CoarseApproximation
open ConformerSearch
open EnergyRMSDConvergence
open Computation.ArrayDSL

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
    {A : Type u} [PseudoMetricSpace A] [DecidableEq A]
    (support : Finset A) (ε : ℝ) : Prop :=
  ∀ a, ∃ b, b ∈ support ∧ dist a b ≤ ε

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

end ConformerSupportCoverage
end Tractability
end DecisionQuotient
