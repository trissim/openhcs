/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/ConformerSearch.lean

  Abstract certificates for conformer-space search.

  Generic certificates, each at maximum generality:

  1. `sum_channel_uniformApprox`
     Uniform approximations compose under utility addition. Directly certifies
     any two-channel scoring decomposition (e.g. rec-as-donor + lig-as-donor
     H-bond channels, or any future additive term split).

  2. `conformer_dominated`
     If every pose score for conformer `a` is bounded above by `ub`, and some
     other conformer achieves `ub` strictly, then `a` is globally dominated.
     This is the core pruning certificate: geometry need only supply `ub`.

  3. `energy_conformer_dominated`
     Minimization-space version of conformer pruning: if one conformer's energy
     is bounded below by `lb` and some other conformer achieves energy strictly
     below `lb`, the first conformer is globally dominated.

  4. `lipschitz_score_composition`
     If a score is Lipschitz in coordinates and a kinematic map is Lipschitz
     in parameters, the composed score is Lipschitz in parameters. Directly
     certifies torsional flexibility: rotating a bond is an isometry
     (K = 1), so the score Lipschitz constant is unchanged.

  5. `lipschitz_energy_lower_bound_on_ball`
     A Lipschitz energy evaluated at the center of a parameter cell induces a
     certified lower bound throughout the cell. This is the direct bridge from
     torsional Lipschitz continuity to safe branch-and-bound pruning.

  No molecular physics appears in this file. The theorems apply to any
  additive scoring decomposition, any coordinate parameterization, and any
  conformer space.
-/
import DecisionQuotient.Tractability.CoarseApproximation
import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Data.NNReal.Basic

namespace DecisionQuotient
namespace Tractability
namespace ConformerSearch

open CoarseApproximation

universe u v

-- ---------------------------------------------------------------------------
-- Theorem 1: Two-channel additivity
-- ---------------------------------------------------------------------------

/-- Uniform approximations compose under pointwise utility addition.
    The combined error radius is the sum of the individual radii.

    Direct corollary of `sum_uniformApprox`. Stated here as a named
    standalone theorem so it can be cited as a handle for any two-channel
    decomposition without importing the full `CoarseApproximation` namespace
    at the call site. -/
theorem sum_channel_uniformApprox
    {A : Type u} {S : Type v}
    (exact1 coarse1 exact2 coarse2 : DecisionProblem A S)
    (δ1 δ2 : ℝ)
    (h1 : UniformUtilityApprox exact1 coarse1 δ1)
    (h2 : UniformUtilityApprox exact2 coarse2 δ2) :
    UniformUtilityApprox
      (sumDecisionProblems exact1 exact2)
      (sumDecisionProblems coarse1 coarse2)
      (δ1 + δ2) :=
  sum_uniformApprox exact1 coarse1 exact2 coarse2 δ1 δ2 h1 h2

-- ---------------------------------------------------------------------------
-- Theorem 2: Conformer domination
-- ---------------------------------------------------------------------------

/-- If every pose score for action `a` is bounded above by `ub`, and some
    reference action `a'` at state `s'` strictly exceeds `ub`, then `a`
    is strictly dominated at every state.

    This is the abstract pruning certificate. The caller supplies:
    - `h_ub` : an upper bound on `a`'s score across all states
    - `h_dom` : evidence that the bound is exceeded by some achievable score

    Geometry, energy, and physics are entirely outside this theorem.
    Any mechanism that produces a valid `h_ub` — pocket volume, diameter
    bound, clash count, or learned surrogate — plugs in directly. -/
theorem conformer_dominated
    {A S : Type*}
    (score : A → S → ℝ)
    (a a' : A) (s' : S)
    (ub : ℝ)
    (h_ub  : ∀ s, score a s ≤ ub)
    (h_dom : ub < score a' s') :
    ∀ s, score a s < score a' s' := fun s =>
  lt_of_le_of_lt (h_ub s) h_dom

/-- Contrapositive: if `a` is optimal (achieves the maximum), then its
    score upper bound is at least as large as any achievable score. -/
theorem optimal_conformer_bound_tight
    {A S : Type*}
    (score : A → S → ℝ)
    (a_opt a' : A) (s_opt s' : S)
    (ub_opt : ℝ)
    (h_ub  : ∀ s, score a_opt s ≤ ub_opt)
    (h_opt : ∀ b t, score b t ≤ score a_opt s_opt) :
    score a' s' ≤ ub_opt :=
  le_trans (h_opt a' s') (h_ub s_opt)

/-- Pointwise lower bounds compose under utility addition. This is the energy
    lower-bound analogue of additive score decomposition. -/
theorem sum_channel_lowerBound
    {A : Type u} {S : Type v}
    (score1 score2 : A → S → ℝ)
    (lb1 lb2 : ℝ)
    (h1 : ∀ a s, lb1 ≤ score1 a s)
    (h2 : ∀ a s, lb2 ≤ score2 a s) :
    ∀ a s, lb1 + lb2 ≤ score1 a s + score2 a s := by
  intro a s
  linarith [h1 a s, h2 a s]

/-- Minimization-space pruning certificate. If every pose energy for conformer
    `a` is bounded below by `lb`, and some reference conformer already achieves
    energy strictly below `lb`, then `a` is globally dominated. -/
theorem energy_conformer_dominated
    {A S : Type*}
    (energy : A → S → ℝ)
    (a a' : A) (s' : S)
    (lb : ℝ)
    (h_lb  : ∀ s, lb ≤ energy a s)
    (h_dom : energy a' s' < lb) :
    ∀ s, energy a' s' < energy a s := fun s =>
  lt_of_lt_of_le h_dom (h_lb s)

-- ---------------------------------------------------------------------------
-- Theorem 4: Lipschitz composition for torsional kinematics
-- ---------------------------------------------------------------------------

/-- If a score function is `M`-Lipschitz in coordinates, and a kinematic
    map is `K`-Lipschitz in parameters, then the composed score
    `fun p => score (kinematics p) s` is `(K * M)`-Lipschitz in parameters.

    Physical interpretation:
    - `Param`  : torsion-angle space (or any continuous parameterization)
    - `Coord`  : atom-coordinate space (ℝ^{3N})
    - `State`  : pose / receptor state
    - `kinematics` : forward kinematics map (torsions → coordinates)
    - `score`  : docking score function (coordinates × pose → ℝ)

    For rigid-body rotation, `kinematics` is an isometry so K = 1 and the
    score Lipschitz constant is preserved exactly. For bond-torsion
    kinematics the constant depends on the chain geometry but is always
    finite and computable. -/
theorem lipschitz_score_composition
    {Param Coord State : Type*}
    [PseudoMetricSpace Param] [PseudoMetricSpace Coord]
    (kinematics : Param → Coord)
    (score : Coord → State → ℝ)
    (K M : NNReal)
    (h_kine  : LipschitzWith K kinematics)
    (h_score : ∀ s, LipschitzWith M (fun c => score c s)) :
    ∀ s, LipschitzWith (M * K) (fun p => score (kinematics p) s) := fun s =>
  (h_score s).comp h_kine

/-- Isometric kinematics (K = 1) preserve the score Lipschitz constant
    exactly. Rotation matrices and rigid-body transforms are isometries. -/
theorem isometric_kinematics_preserves_lipschitz
    {Param Coord State : Type*}
    [PseudoMetricSpace Param] [PseudoMetricSpace Coord]
    (kinematics : Param → Coord)
    (score : Coord → State → ℝ)
    (M : NNReal)
    (h_kine  : LipschitzWith (1 : NNReal) kinematics)
    (h_score : ∀ s, LipschitzWith M (fun c => score c s)) :
    ∀ s, LipschitzWith M (fun p => score (kinematics p) s) := by
  intro s
  have h := (h_score s).comp h_kine
  rwa [mul_one] at h

-- ---------------------------------------------------------------------------
-- Theorem 5: Cellwise Lipschitz energy bounds
-- ---------------------------------------------------------------------------

/-- A Lipschitz energy evaluated at `p₀` gives a certified lower bound at any
    point `p`. This is the basic minimization-side branch-and-bound inequality. -/
theorem lipschitz_energy_lower_bound
    {Param : Type*}
    [PseudoMetricSpace Param]
    (energy : Param → ℝ)
    (L : NNReal)
    (h_lip : LipschitzWith L energy)
    (p p₀ : Param) :
    energy p₀ - L * dist p p₀ ≤ energy p := by
  have h := h_lip.le_add_mul p₀ p
  simpa [dist_comm, sub_le_iff_le_add'] using h

/-- If `p` lies within radius `r` of `p₀`, then the center energy minus `L * r`
    is a certified lower bound throughout the whole cell. -/
theorem lipschitz_energy_lower_bound_on_ball
    {Param : Type*}
    [PseudoMetricSpace Param]
    (energy : Param → ℝ)
    (L : NNReal)
    (h_lip : LipschitzWith L energy)
    (p p₀ : Param)
    (r : ℝ)
    (h_ball : dist p p₀ ≤ r) :
    energy p₀ - L * r ≤ energy p := by
  have h_point := lipschitz_energy_lower_bound energy L h_lip p p₀
  have h_mul : L * dist p p₀ ≤ L * r := by
    gcongr
  linarith

end ConformerSearch
end Tractability
end DecisionQuotient
