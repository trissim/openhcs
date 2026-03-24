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

end ConformerSupportCoverage
end Tractability
end DecisionQuotient
