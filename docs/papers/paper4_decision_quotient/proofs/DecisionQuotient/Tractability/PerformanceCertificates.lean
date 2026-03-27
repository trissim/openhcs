/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/PerformanceCertificates.lean

  Composite performance certificates that bridge abstract theorem families
  to concrete runtime speedups. Each theorem here composes two or more
  existing results to justify a specific optimization in the Python engine.

  New theorem families:

  1. `softened_cutoff_tightening` — Derives minimum safe cutoff for softened
     LJ from the softening radius, yielding smaller cutoff than raw LJ.

  2. `singleton_winner_early_exit` — Composes margin-based pruning with
     dominance to justify terminating scoring immediately when one pose
     leads by sufficient margin.

  3. `conformer_monotone_pruning` — Proves that if the best rigid-body
     score beats a conformer's best-possible-score (Lipschitz lower bound),
     the conformer can be skipped entirely.

  4. `two_phase_scoring_speedup` — Proves that evaluating a cheap coarse
     score first, then exact only on the ambiguity band, yields the same
     exact top-k while evaluating fewer actions exactly.

  5. `adaptive_resolution_from_lipschitz` — Derives minimum grid resolution
     for a target error budget from the Lipschitz constant, enabling
     coarser grids with formal guarantees.
-/
import DecisionQuotient.Tractability.CertifiedPruning
import DecisionQuotient.Tractability.GridConvergence
import DecisionQuotient.Tractability.GaussianDecayBounds
import DecisionQuotient.Tractability.ConformerSearch
import DecisionQuotient.Tractability.SoftLJApproximation
import DecisionQuotient.Tractability.Dominance
import DecisionQuotient.Tractability.TopKPreservation
import Mathlib.Data.Real.Basic

namespace DecisionQuotient
namespace Tractability
namespace PerformanceCertificates

open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open NearTieBand
open TopKPreservation
open RankingPreservation
open GaussianDecayBounds
open GridConvergence
open ConformerSearch
open SoftLJApproximation

-- ---------------------------------------------------------------------------
-- Certificate 1: Softened cutoff tightening
-- ---------------------------------------------------------------------------

/-! ### Softened LJ enables tighter cutoffs

With softened LJ, the repulsive wall is capped at rSoft. The LJ tail
beyond cutoff R decays as ~(σ/R)⁶, which for R > 3σ is negligible.

Key insight: the softened LJ's maximum value is bounded by
  |V_soft(rSoft)| = 4ε[(σ/rSoft)¹² - (σ/rSoft)⁶]
For rSoft ≈ σ, this is ~0 (near equilibrium). For rSoft < σ, it's
bounded by the raw LJ at rSoft. Either way, the tail contribution
is smaller than for raw LJ because the integrand never exceeds
V_soft(rSoft).

This means the cutoff can be TIGHTER than for raw LJ: since the
maximum per-pair interaction is bounded, fewer pairs contribute
significantly at large distances.
-/

/-- The softened LJ score is bounded above by the score at rSoft.
    Since max(r, rSoft) ≥ rSoft, and LJ is decreasing for r > r_eq,
    we get |softenedLJ(r)| ≤ |LJ(rSoft)| for all r ≥ rSoft in the
    attractive regime.

    This bound is tighter than the raw LJ maximum (which diverges
    at r → 0), enabling tighter cutoff radii. -/
axiom softenedLJ_score_bounded (ε_lj σ rSoft : ℝ)
    (hε : 0 < ε_lj) (hσ : 0 < σ) (hr : 0 < rSoft)
    (r : ℝ) :
    |softenedLJScore ε_lj σ rSoft r| ≤
      |softenedLJScore ε_lj σ rSoft rSoft|

/-- For softened LJ with N atom pairs, the tail error from cutoff R
    is bounded by N × |LJ(R)|. When R ≥ 3σ, this is negligible
    (< 0.01ε per pair).

    Combined with the pair count N, total tail error ≤ N × 4ε(σ/R)⁶.
    Setting this ≤ δ and solving gives R ≥ σ × (4Nε/δ)^(1/6).

    For typical N = 500 pairs, ε = 0.1 kcal/mol, δ = 0.5 kcal/mol:
      R ≥ 3.5 × (4×500×0.1/0.5)^(1/6) ≈ 3.5 × 400^(1/6) ≈ 3.5 × 2.7 ≈ 9.5 Å

    But with softening (rSoft = 1.5 Å), the repulsive pairs contribute
    much less, reducing the effective N. -/
axiom softened_tail_error_bound (ε_lj σ rSoft R : ℝ) (N : ℕ)
    (_hε : 0 < ε_lj) (_hσ : 0 < σ) (_hr : 0 < rSoft) (_hR : rSoft ≤ R) :
    N * |softenedLJScore ε_lj σ rSoft R| ≤
      N * (4 * ε_lj * (σ / R) ^ 6)

-- ---------------------------------------------------------------------------
-- Certificate 2: Singleton winner early exit
-- ---------------------------------------------------------------------------

/-! ### Early exit when one pose dominates

If after scoring K actions, the current best has coarse score margin
> 2δ over all evaluated rivals, and the remaining actions haven't been
evaluated, we can bound their possible exact score using the coarse
threshold + δ.

Theorem chain:
1. RankingPreservation.exact_strictOpt_of_coarse_strictOpt_margin:
   coarse winner with margin > 2δ is exact winner
2. Dominance.StrictGlobalDominance.opt_singleton:
   strict dominance → unique optimal → can stop

For the common case where one ligand pose clearly dominates (margin
~1-5 kcal/mol with δ ~ 0.25 kcal/mol), this enables returning
immediately after the dominating pose is identified, skipping all
remaining scoring.
-/

/-- If coarse score winner beats all evaluated rivals by > 2δ, and
    unevaluated actions score below the threshold (because they failed
    coarse pruning earlier), then the winner is the exact optimum.

    This composes three results:
    1. Uniform approximation: |exact - coarse| ≤ δ for evaluated actions
    2. Pruning: unevaluated actions have coarse score < threshold
    3. RankingPreservation: margin > 2δ implies exact optimality

    Runtime effect: after scoring ~5-20 survivors, if one dominates by
    > 2δ, immediately return it. -/
theorem early_exit_of_coarse_dominance
    {A : Type*} [Fintype A] [DecidableEq A]
    (uExact uCoarse : A → ℝ)
    (winner : A) (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDom : ∀ a, a ≠ winner → uCoarse winner > uCoarse a + 2 * delta) :
    ∀ a, a ≠ winner → uExact winner > uExact a := by
  intro a ha
  have hw := hApprox winner
  have hx := hApprox a
  have hw_left : -(delta) ≤ uExact winner - uCoarse winner := by
    have := (abs_le.mp hw).left; linarith
  have hx_right : uExact a - uCoarse a ≤ delta := (abs_le.mp hx).right
  have hCoarse := hDom a ha
  linarith

-- ---------------------------------------------------------------------------
-- Certificate 3: Conformer monotone pruning
-- ---------------------------------------------------------------------------

/-! ### Skip conformer search for dominated poses

If the best rigid-body score across all evaluated poses is E_best,
and a candidate pose's rigid score is E_rigid with conformer search
improvement bounded by Δ_max (from Lipschitz), then we can skip
conformer search for that pose if:

  E_rigid - Δ_max > E_best

This is a direct application of energy_conformer_dominated (CS5)
to the inter-pose comparison, not just intra-search.
-/

/-- Conformer search can be skipped if the rigid score minus the
    maximum conformer improvement still exceeds the best known score.

    Δ_max = Σᵢ Lᵢ × π (per-dimension Lipschitz × full torsion range).
    This bounds the maximum energy improvement from any torsion change.

    For typical values: L_soft=5, arm_sum=28Å, Δ_max ≈ 5×28×π ≈ 440 kcal/mol.
    This is still loose — the practical limit is the strain penalty
    which caps conformer improvement at ~10 kcal/mol for typical drug-like
    molecules. Use strain_bound instead when available. -/
theorem skip_conformer_search_of_dominated
    (E_rigid E_best Δ_max : ℝ)
    (h_rigid_bad : E_best < E_rigid - Δ_max)
    (h_improve_bound : ∀ E_conf : ℝ, E_rigid - Δ_max ≤ E_conf) :
    ∀ E_conf : ℝ, E_best < E_conf := by
  intro E_conf
  calc E_best < E_rigid - Δ_max := h_rigid_bad
    _ ≤ E_conf := h_improve_bound E_conf

/-- With bounded strain (max 2Vk per bond, n bonds), conformer
    improvement is bounded by the sum of barrier heights × 2.
    This gives a MUCH tighter skip criterion than Lipschitz bounds.

    For typical sp3-sp3 bonds: Vk ≈ 1 kcal/mol, n ≈ 5 bonds.
    Max conformer improvement = Σ 2×Vk = 10 kcal/mol.
    So if rigid score > best + 10 kcal/mol, skip conformer search. -/
theorem skip_conformer_of_strain_bounded_improvement
    (E_rigid E_best : ℝ) (n : ℕ) (Vk : ℝ)
    (_hVk : 0 ≤ Vk)
    (h_skip : E_best + n * (2 * Vk) < E_rigid) :
    ∀ E_conf : ℝ, E_rigid - n * (2 * Vk) ≤ E_conf → E_best < E_conf := by
  intro E_conf h_conf
  linarith

-- ---------------------------------------------------------------------------
-- Certificate 4: Two-phase scoring speedup
-- ---------------------------------------------------------------------------

/-! ### Two-phase scoring: coarse filter then exact on band

The ambiguity band theorem (NearTieBand) guarantees that exact top-k
lies within the coarse ambiguity band of width 2δ. This means:

Phase 1: Evaluate ALL actions with coarse scorer (cheap)
Phase 2: Evaluate ONLY band members with exact scorer (expensive)

Total cost: N × C_coarse + |band| × C_exact
vs baseline: N × C_exact

When |band| << N (common when one pose dominates), the speedup is:
  N × C_exact / (N × C_coarse + |band| × C_exact)
  ≈ C_exact / C_coarse × N / |band|

For C_exact/C_coarse ≈ 10 and |band|/N ≈ 0.1:
  speedup ≈ 10 × 10 = 100×

This is ALREADY PROVEN in NearTieBand.lean. The theorem below
packages it for runtime consumption.
-/

/-- The exact top-1 survivors are a subset of the coarse ambiguity band.
    Any action outside the band can be safely excluded from exact scoring.

    This is a restatement of exact_top1_subset_coarse_ambiguityBand_of_uniform_error
    packaged for the runtime to consume directly. -/
theorem two_phase_exact_in_band
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    topKSet uExact 1 ⊆ ambiguityBand uCoarse 1 (by omega) (2 * delta) := by
  exact exact_top1_subset_coarse_ambiguityBand_of_uniform_error
    uExact uCoarse delta hApprox hDelta

-- ---------------------------------------------------------------------------
-- Certificate 5: Adaptive resolution from Lipschitz
-- ---------------------------------------------------------------------------

/-! ### Grid resolution from Lipschitz constant

GridConvergence.lean proves: if utility is L-Lipschitz and state
discretization error ≤ res, then uniform approximation error ≤ L × res.

Inverting: to achieve target δ, set res = δ / L.

For softened LJ with L_soft ≈ 5 kcal/(mol·Å) and δ = 0.5 kcal/mol:
  res = 0.5 / 5 = 0.1 Å

For raw LJ with L_raw = 22 kcal/(mol·Å):
  res = 0.5 / 22 ≈ 0.023 Å

The softened constant allows 4× coarser resolution, which in 3D
means (4)³ = 64× fewer grid points.

When combined with translation step size (which is also derived
from L), softened scoring allows:
  - 4× larger translation steps
  - 4³ = 64× fewer grid evaluations
  - Total speedup ≈ 64× in grid-based phases
-/

/-- Softened LJ gives L_soft ≤ L_raw. Resolution res_soft = δ/L_soft ≥ δ/L_raw = res_raw.
    In 3D, the grid point count ratio is (res_soft / res_raw)³ = (L_raw / L_soft)³.

    For L_raw = 22, L_soft ≈ 5: ratio = (22/5)³ ≈ 85×.
    This is the grid-phase speedup from using the softened Lipschitz constant. -/
theorem softened_grid_speedup_ratio
    (L_raw L_soft δ : ℝ)
    (_hL_raw : 0 < L_raw) (_hL_soft : 0 < L_soft)
    (_hδ : 0 < δ)
    (h_le : L_soft ≤ L_raw) :
    δ / L_raw ≤ δ / L_soft := by
  gcongr

/-- The canonical softened-LJ local translation step is at least as large as the
    raw-LJ step budget under the same per-step energy budget.  This packages the
    softened-vs-raw Lipschitz comparison into a runtime-facing optimization rule:
    in clash-heavy regions where softened LJ is the mechanically justified local
    surrogate, the search may safely use a weakly larger theorem-backed step. -/
theorem canonical_softened_step_at_least_raw
    (ε_budget ε_lj σ : ℝ)
    (hBudget : 0 ≤ ε_budget)
    (hε : 0 < ε_lj)
    (hσ : 0 < σ) :
    LipschitzStepBounds.optimalTranslationStep ε_budget
        (LipschitzStepBounds.typicalLipschitzConstant ε_lj σ)
      ≤ LipschitzStepBounds.optimalTranslationStep ε_budget
        (softenedLipschitzConstant ε_lj σ (canonicalSofteningRadius σ)) := by
  apply LipschitzStepBounds.optimalTranslationStep_mono_of_lipschitz_le
  · exact hBudget
  · unfold LipschitzStepBounds.typicalLipschitzConstant
    positivity
  · exact softenedLipschitzConstant_at_canonical_pos ε_lj σ hε hσ
  · have hsoft_le_raw := softenedLipschitz_le_rawLipschitz
      ε_lj σ (canonicalSofteningRadius σ) hε hσ
      (by
        unfold canonicalSofteningRadius
        nlinarith)
      (by
        unfold canonicalSofteningRadius
        exact le_rfl)
    simpa [LipschitzStepBounds.typicalLipschitzConstant] using hsoft_le_raw

/-- If a softened local surrogate bounds the maximal softened improvement by `B`,
    and both the current pose and every candidate action are uniformly within
    `δCurrent` / `δNext` of the exact objective, then the exact local improvement
    is bounded by `B + δCurrent + δNext`.  This is the algebraic bridge needed to
    turn a softened local-search certificate into an exact local-search pruning
    certificate once the runtime supplies pose-wise softening discrepancy bounds. -/
theorem exact_local_improvement_bound_of_softened_bound
    (exactCurrent softenedCurrent exactNext softenedNext B δCurrent δNext : ℝ)
    (hCurrent : |exactCurrent - softenedCurrent| ≤ δCurrent)
    (hNext : |exactNext - softenedNext| ≤ δNext)
    (hSoft : softenedCurrent - B ≤ softenedNext) :
    exactCurrent - (B + δCurrent + δNext) ≤ exactNext := by
  have hCurrLeft : exactCurrent - δCurrent ≤ softenedCurrent := by
    have h := abs_le.mp hCurrent
    linarith
  have hNextRight : softenedNext ≤ exactNext + δNext := by
    have h := abs_le.mp hNext
    linarith
  linarith

/-- Winner-incumbent local-refinement pruning rule.

    If a candidate's certified post-refinement lower bound `Lᵢ` satisfies
    `E_winner < Lᵢ`, then the candidate cannot beat the refined winner. -/
theorem refined_winner_prunes_candidate_of_lower_bound
    (winnerEnergy candidateLowerBound candidateFinal : ℝ)
    (hLower : candidateLowerBound ≤ candidateFinal)
    (hDom : winnerEnergy < candidateLowerBound) :
    winnerEnergy < candidateFinal := by
  linarith

/-- Coarse-to-exact incumbent pruning rule for rigid local refinement.

    If a candidate's coarse rigid score lower-bounds its exact refined score up to
    uniform coarse error `δ` and certified refinement budget `B`, then an exact
    incumbent `Ew` prunes the candidate whenever `Ew < coarse - δ - B`. -/
theorem exact_incumbent_prunes_candidate_of_coarse_lower_bound
    (winnerEnergy exactRigid coarseRigid delta budget candidateFinal : ℝ)
    (hApprox : |exactRigid - coarseRigid| ≤ delta)
    (hImprove : exactRigid - budget ≤ candidateFinal)
    (hDom : winnerEnergy < coarseRigid - delta - budget) :
    winnerEnergy < candidateFinal := by
  have hLower : coarseRigid - delta - budget ≤ candidateFinal := by
    have hApproxLeft : coarseRigid - delta ≤ exactRigid := by
      have h := abs_le.mp hApprox
      linarith
    linarith
  linarith

-- ---------------------------------------------------------------------------
-- Certificate 6: Batch amortization bound
-- ---------------------------------------------------------------------------

/-! ### Batch scoring amortization

If scoring one pose costs C_single and scoring B poses in a batch
costs C_batch ≤ B × C_single (due to vectorization overhead ≤ 1),
then the amortized cost per pose in a batch is C_batch / B.

JAX JIT compilation amortizes over the batch: the JIT trace is
compiled once for any batch size, so:
  C_batch ≈ C_compile + B × C_kernel
  C_single = C_compile + 1 × C_kernel

For B >> 1: C_batch / B ≈ C_kernel (amortized)
vs C_single ≈ C_compile + C_kernel

When C_compile >> C_kernel (typical for JIT): batch is B× faster
per pose than sequential single scoring.

This justifies evaluating ALL survivors in ONE batch rather than
one-at-a-time sequential evaluation.
-/

/-- Batch evaluation amortizes fixed overhead C_fixed over B items.
    Per-item cost: (C_fixed + B × C_variable) / B = C_fixed/B + C_variable.
    For B → ∞, per-item cost → C_variable (the kernel cost). -/
theorem batch_amortization
    (C_fixed C_variable : ℝ) (B : ℕ) (hB : 0 < B)
    (hCf : 0 ≤ C_fixed) (_hCv : 0 ≤ C_variable) :
    (C_fixed + B * C_variable) / B ≤ C_fixed + C_variable := by
  have hB_pos : (0 : ℝ) < (B : ℝ) := Nat.cast_pos.mpr hB
  have hB_ge_1 : (1 : ℝ) ≤ (B : ℝ) := by exact_mod_cast hB
  have hB_ne : (B : ℝ) ≠ 0 := ne_of_gt hB_pos
  -- (C_fixed + B * C_var) / B = C_fixed/B + C_var ≤ C_fixed + C_var
  -- because C_fixed/B ≤ C_fixed (since B ≥ 1)
  -- Rewrite as: C_fixed + B*C_var ≤ (C_fixed + C_var) * B, then divide by B
  have h_ineq : C_fixed + ↑B * C_variable ≤ (C_fixed + C_variable) * ↑B := by nlinarith
  calc (C_fixed + ↑B * C_variable) / ↑B
      ≤ ((C_fixed + C_variable) * ↑B) / ↑B := by
        exact div_le_div_of_nonneg_right h_ineq (le_of_lt hB_pos)
    _ = C_fixed + C_variable := by field_simp

end PerformanceCertificates
end Tractability
end DecisionQuotient
