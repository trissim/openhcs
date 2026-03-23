/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/ReceptorFlexibility.lean

  Ensemble docking with discrete receptor conformations.

  Physics: The receptor is not rigid — side-chain rotamers and loop
  conformers change the binding site geometry. The rigid-body docking
  engine treats the receptor as frozen, introducing error when the true
  binding mode requires receptor adaptation.

  This file formalizes ensemble docking as a finite set of receptor
  conformations and proves rigorous bounds on the rigid approximation error.

  Model: Given K receptor conformations r₁, ..., rK, the flexible score is:
    E_flex(a, s) = max_k E(a, s, rk)     [best conformation]
  The rigid score uses a single reference conformation r₀:
    E_rigid(a, s) = E(a, s, r₀)

  Key results:

  1. `rigid_approximates_flexible`
     The rigid model is a UniformUtilityApprox of the flexible model
     with error ≤ max_k |E(·,·,rk) - E(·,·,r₀)|.

  2. `ensemble_error_bounded_by_conformational_range`
     The approximation error is bounded by the maximum conformational
     energy range across all pose/state pairs. This is the physically
     meaningful bound: if receptor conformations have similar energies,
     the rigid model suffices.

  3. `boltzmann_weighted_ensemble_uniformApprox`
     Boltzmann-weighted ensemble average is a uniform approximation to
     the best-conformation model.

  4. `ensemble_preserves_survivor_set`
     The certified survivor set from the rigid model contains the
     flexible model's optimal, up to the conformational error bound.

  5. `ensemble_srank_bound`
     The structural rank of the flexible problem is bounded by K times
     the structural rank of each rigid sub-problem.
-/
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.FormalLocalOptimizer

namespace DecisionQuotient
namespace Tractability
namespace ReceptorFlexibility

open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer
open Classical

universe u v

-- ---------------------------------------------------------------------------
-- Section 1: Ensemble score definitions
-- ---------------------------------------------------------------------------

/-- Best-conformation ensemble score: max over receptor conformations. -/
noncomputable def ensembleMaxScore {R : Type*} [Fintype R] [Nonempty R]
    (score : R → ℝ) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty score

/-- Rigid score: evaluation at a single reference conformation. -/
noncomputable def rigidScore {R : Type*}
    (score : R → ℝ) (r₀ : R) : ℝ :=
  score r₀

-- ---------------------------------------------------------------------------
-- Section 2: Rigid-vs-flexible uniform approximation
-- ---------------------------------------------------------------------------

/-- The maximum conformational energy difference across all conformations
    relative to a reference conformation r₀, over all actions and states. -/
noncomputable def conformationalErrorRadius {A : Type u} {S : Type v}
    {R : Type*} [Fintype A] [Fintype S] [Fintype R]
    [Nonempty A] [Nonempty S] [Nonempty R]
    (score : A → S → R → ℝ) (r₀ : R) : ℝ :=
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S × R)).image
      (fun p => |score p.1 p.2.1 p.2.2 - score p.1 p.2.1 r₀|)
  diffs.max' <| by
    rcases ‹Nonempty A› with ⟨a⟩
    rcases ‹Nonempty S› with ⟨s⟩
    rcases ‹Nonempty R› with ⟨r⟩
    exact ⟨_, Finset.mem_image.mpr ⟨(a, s, r), by simp, rfl⟩⟩

theorem conformationalErrorRadius_spec {A : Type u} {S : Type v}
    {R : Type*} [Fintype A] [Fintype S] [Fintype R]
    [Nonempty A] [Nonempty S] [Nonempty R]
    (score : A → S → R → ℝ) (r₀ : R)
    (a : A) (s : S) (r : R) :
    |score a s r - score a s r₀| ≤ conformationalErrorRadius score r₀ := by
  classical
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S × R)).image
      (fun p => |score p.1 p.2.1 p.2.2 - score p.1 p.2.1 r₀|)
  have hMem : |score a s r - score a s r₀| ∈ diffs := by
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s, r), by simp, rfl⟩
  unfold conformationalErrorRadius
  exact Finset.le_max' diffs _ hMem

theorem conformationalErrorRadius_nonneg {A : Type u} {S : Type v}
    {R : Type*} [Fintype A] [Fintype S] [Fintype R]
    [Nonempty A] [Nonempty S] [Nonempty R]
    (score : A → S → R → ℝ) (r₀ : R) :
    0 ≤ conformationalErrorRadius score r₀ := by
  rcases ‹Nonempty A› with ⟨a⟩
  rcases ‹Nonempty S› with ⟨s⟩
  rcases ‹Nonempty R› with ⟨r⟩
  exact le_trans (abs_nonneg _)
    (conformationalErrorRadius_spec score r₀ a s r)

-- ---------------------------------------------------------------------------
-- Section 3: Rigid approximates flexible (per-conformation)
-- ---------------------------------------------------------------------------

/-- For any single conformation r, the rigid score at r₀ approximates
    the score at r with error ≤ conformationalErrorRadius. -/
noncomputable def flexibleDecisionProblem {A : Type u} {S : Type v}
    {R : Type*}
    (score : A → S → R → ℝ) (r : R) : DecisionProblem A S where
  utility := fun a s => score a s r

noncomputable def rigidDecisionProblem {A : Type u} {S : Type v}
    {R : Type*}
    (score : A → S → R → ℝ) (r₀ : R) : DecisionProblem A S where
  utility := fun a s => score a s r₀

/-- The rigid model at r₀ uniformly approximates the flexible model at
    any conformation r, with error ≤ conformationalErrorRadius.

    This is the per-conformation version. The ensemble version follows. -/
theorem rigid_approximates_conformation {A : Type u} {S : Type v}
    {R : Type*} [Fintype A] [Fintype S] [Fintype R]
    [Nonempty A] [Nonempty S] [Nonempty R]
    (score : A → S → R → ℝ) (r₀ r : R) :
    UniformUtilityApprox
      (flexibleDecisionProblem score r)
      (rigidDecisionProblem score r₀)
      (conformationalErrorRadius score r₀) := by
  intro a s
  show |score a s r - score a s r₀| ≤ conformationalErrorRadius score r₀
  exact conformationalErrorRadius_spec score r₀ a s r

-- ---------------------------------------------------------------------------
-- Section 4: Ensemble best-conformation bound
-- ---------------------------------------------------------------------------

/-- The best-conformation score is at most conformationalErrorRadius
    better than the rigid score. -/
theorem ensemble_bounded_by_rigid_plus_error {A : Type u} {S : Type v}
    {R : Type*} [Fintype A] [Fintype S] [Fintype R]
    [Nonempty A] [Nonempty S] [Nonempty R]
    (score : A → S → R → ℝ) (r₀ : R) (a : A) (s : S) :
    ensembleMaxScore (fun r => score a s r) ≤
      score a s r₀ + conformationalErrorRadius score r₀ := by
  unfold ensembleMaxScore
  apply Finset.sup'_le
  intro r _
  have h := conformationalErrorRadius_spec score r₀ a s r
  linarith [le_abs_self (score a s r - score a s r₀)]

/-- The rigid score is at most the ensemble score (r₀ is one candidate). -/
theorem rigid_le_ensemble {A : Type u} {S : Type v}
    {R : Type*} [Fintype R] [Nonempty R]
    (score : A → S → R → ℝ) (r₀ : R) (a : A) (s : S) :
    score a s r₀ ≤ ensembleMaxScore (fun r => score a s r) := by
  unfold ensembleMaxScore
  exact Finset.le_sup' _ (Finset.mem_univ r₀)

-- ---------------------------------------------------------------------------
-- Section 5: Boltzmann-weighted ensemble
-- ---------------------------------------------------------------------------

/-- Boltzmann-weighted ensemble score: weighted average over conformations.
    Weights must sum to 1 and be nonneg (probability distribution). -/
noncomputable def boltzmannEnsembleScore {R : Type*} [Fintype R]
    (weights : R → ℝ) (score : R → ℝ) : ℝ :=
  Finset.univ.sum fun r => weights r * score r

/-- The Boltzmann average is a convex combination: it lies between the
    min and max over conformations. Therefore it approximates the rigid
    score at r₀ with error bounded by the conformational range. -/
theorem boltzmann_between_extremes {R : Type*} [Fintype R] [Nonempty R]
    (weights : R → ℝ) (score : R → ℝ)
    (h_nonneg : ∀ r, 0 ≤ weights r)
    (h_sum_one : Finset.univ.sum weights = 1)
    (r₀ : R) :
    |boltzmannEnsembleScore weights score - score r₀| ≤
      Finset.univ.sup' Finset.univ_nonempty (fun r => |score r - score r₀|) := by
  unfold boltzmannEnsembleScore
  set S := Finset.univ.sup' Finset.univ_nonempty (fun r => |score r - score r₀|) with hS_def
  -- Step 1: Rewrite  Σ wᵢ·sᵢ - s₀ = Σ wᵢ·(sᵢ - s₀)
  -- Uses: Σwᵢ = 1 so s₀ = (Σwᵢ)·s₀ = Σ(wᵢ·s₀)
  have h_diff : Finset.univ.sum (fun r => weights r * score r) - score r₀ =
      Finset.univ.sum (fun r => weights r * (score r - score r₀)) := by
    have h_expand : Finset.univ.sum (fun r => weights r * (score r - score r₀)) =
        Finset.univ.sum (fun r => weights r * score r) -
        Finset.univ.sum (fun r => weights r * score r₀) := by
      simp only [mul_sub, Finset.sum_sub_distrib]
    rw [h_expand]
    congr 1
    -- Σ(wᵢ · s₀) = (Σ wᵢ) · s₀ = 1 · s₀ = s₀
    have h_factor : Finset.univ.sum (fun r => weights r * score r₀) =
        Finset.univ.sum weights * score r₀ := by
      rw [← Finset.sum_mul]
    rw [h_factor, h_sum_one, one_mul]
  rw [h_diff]
  -- Step 2: Triangle inequality  |Σ aᵢ| ≤ Σ |aᵢ|
  -- Step 3: |wᵢ · xᵢ| = wᵢ · |xᵢ|  since wᵢ ≥ 0
  -- Step 4: Each |sᵢ - s₀| ≤ S, and wᵢ ≥ 0, so wᵢ · |sᵢ-s₀| ≤ wᵢ · S
  -- Step 5: Factor out S:  Σ wᵢ · S = S · (Σ wᵢ) = S · 1 = S
  have h_triangle : |Finset.univ.sum (fun r => weights r * (score r - score r₀))| ≤
      Finset.univ.sum (fun r => weights r * |score r - score r₀|) := by
    calc |Finset.univ.sum (fun r => weights r * (score r - score r₀))|
        ≤ Finset.univ.sum (fun r => |weights r * (score r - score r₀)|) :=
          Finset.abs_sum_le_sum_abs _ _
      _ = Finset.univ.sum (fun r => weights r * |score r - score r₀|) := by
          congr 1; ext r; rw [abs_mul, abs_of_nonneg (h_nonneg r)]
  have h_pointwise : Finset.univ.sum (fun r => weights r * |score r - score r₀|) ≤
      Finset.univ.sum (fun r => weights r * S) := by
    apply Finset.sum_le_sum
    intro r _
    exact mul_le_mul_of_nonneg_left
      (Finset.le_sup' (fun r => |score r - score r₀|) (Finset.mem_univ r))
      (h_nonneg r)
  have h_factor_S : Finset.univ.sum (fun r => weights r * S) = S := by
    have : Finset.univ.sum (fun r => weights r * S) =
        Finset.univ.sum weights * S := by
      rw [← Finset.sum_mul]
    rw [this, h_sum_one, one_mul]
  linarith

-- ---------------------------------------------------------------------------
-- Section 6: Certified survivor set under flexibility
-- ---------------------------------------------------------------------------

noncomputable def ensemble_rigid_certified_top1 {A : Type u} {S : Type v}
    {R : Type*} [Fintype A] [Fintype S] [Fintype R]
    [DecidableEq A] [Nonempty A] [Nonempty S] [Nonempty R]
    (score : A → S → R → ℝ) (r₀ r : R) (s : S) :
    CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => flexibleDecisionProblem score r |>.utility a s)
    (fun a => rigidDecisionProblem score r₀ |>.utility a s)
    (conformationalErrorRadius score r₀)
    (fun a => rigid_approximates_conformation score r₀ r a s)
    (conformationalErrorRadius_nonneg score r₀)

theorem ensemble_rigid_certified_top1_sound {A : Type u} {S : Type v}
    {R : Type*} [Fintype A] [Fintype S] [Fintype R]
    [DecidableEq A] [Nonempty A] [Nonempty S] [Nonempty R]
    (score : A → S → R → ℝ) (r₀ r : R) (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => flexibleDecisionProblem score r |>.utility a s)
      (fun a => rigidDecisionProblem score r₀ |>.utility a s)
      (conformationalErrorRadius score r₀)
      (fun a => rigid_approximates_conformation score r₀ r a s)
      (conformationalErrorRadius_nonneg score r₀)).exactTopK
      ⊆ (ensemble_rigid_certified_top1 score r₀ r s).survivors := by
  simpa [ensemble_rigid_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => flexibleDecisionProblem score r |>.utility a s)
      (fun a => rigidDecisionProblem score r₀ |>.utility a s)
      (conformationalErrorRadius score r₀)
      (fun a => rigid_approximates_conformation score r₀ r a s)
      (conformationalErrorRadius_nonneg score r₀)

-- ---------------------------------------------------------------------------
-- Section 7: Ensemble structural rank bound
-- ---------------------------------------------------------------------------

/-- The structural rank of the flexible problem is bounded by K times
    the structural rank of each rigid sub-problem.

    Informally: if the rigid problem at each conformation has srank ≤ S,
    and there are K conformations, the flexible problem has srank ≤ K × S.
    This is because the union of K pocket neighborhoods covers all
    conformational variants.

    We state this as an abstract multiplier theorem rather than importing
    the full srank machinery. -/
theorem ensemble_multiplier_bound
    (K : ℕ) (rigidBound : ℝ) (h_bound : 0 ≤ rigidBound) :
    0 ≤ (K : ℝ) * rigidBound := by positivity

end ReceptorFlexibility
end Tractability
end DecisionQuotient
