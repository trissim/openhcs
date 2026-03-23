/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/LigandStrainApproximation.lean

  Ligand internal strain energy for conformer-aware docking.

  Physics: Each rotatable bond contributes a torsion strain potential
  V(φ) = Vk · (1 - cos(nφ - φ₀)), bounded in [0, 2Vk].

  Key results:

  1. `cosineTorsionStrain_nonneg` / `cosineTorsionStrain_le_twoVk`
     Cosine torsion strain is bounded in [0, 2Vk]. This constrains the
     conformer search space and enables certified pruning.

  2. `strain_preserves_uniformApprox`
     Adding an exact (non-approximated) strain penalty to both sides of a
     uniform approximation preserves the error bound. This is the core
     integration theorem: strain doesn't degrade certified docking accuracy.

  3. `strain_augmented_lipschitz`
     The combined (score - strain) function is Lipschitz with constant
     L_score + L_strain. Certifies branch-and-bound cell bounds under
     strain-aware scoring.

  4. `additive_strain_bounded`
     Sum of bounded strain terms is bounded by the sum of bounds. Certifies
     multi-bond torsion strain accumulation.

  5. `strain_energy_lower_bound_on_ball`
     Cellwise energy lower bound including strain, for branch-and-bound.

  No molecular physics beyond the cosine potential appears here. The abstract
  theorems (2-5) apply to any bounded strain model.
-/
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.ConformerSearch
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Topology.MetricSpace.Lipschitz

namespace DecisionQuotient
namespace Tractability
namespace LigandStrainApproximation

open CoarseApproximation
open ConformerSearch

universe u v

-- ---------------------------------------------------------------------------
-- Section 1: Cosine torsion strain potential
-- ---------------------------------------------------------------------------

/-- Cosine torsion strain potential per rotatable bond.
    V(φ) = Vk · (1 - cos(nφ - φ₀))
    Minimum 0 at equilibrium (nφ = φ₀), maximum 2Vk at anti-equilibrium. -/
noncomputable def cosineTorsionStrain (Vk n φ φ₀ : ℝ) : ℝ :=
  Vk * (1 - Real.cos (n * φ - φ₀))

theorem cosineTorsionStrain_nonneg (Vk n φ φ₀ : ℝ) (hVk : 0 ≤ Vk) :
    0 ≤ cosineTorsionStrain Vk n φ φ₀ := by
  unfold cosineTorsionStrain
  apply mul_nonneg hVk
  linarith [Real.cos_le_one (n * φ - φ₀)]

theorem cosineTorsionStrain_le_twoVk (Vk n φ φ₀ : ℝ) (hVk : 0 ≤ Vk) :
    cosineTorsionStrain Vk n φ φ₀ ≤ 2 * Vk := by
  unfold cosineTorsionStrain
  have h_cos : -1 ≤ Real.cos (n * φ - φ₀) := Real.neg_one_le_cos _
  have h_sub : 1 - Real.cos (n * φ - φ₀) ≤ 2 := by linarith
  calc Vk * (1 - Real.cos (n * φ - φ₀)) ≤ Vk * 2 := by
        apply mul_le_mul_of_nonneg_left h_sub hVk
    _ = 2 * Vk := by ring

/-- At equilibrium (nφ = φ₀), torsion strain vanishes. -/
theorem cosineTorsionStrain_at_equilibrium (Vk φ₀ : ℝ) :
    cosineTorsionStrain Vk 1 φ₀ φ₀ = 0 := by
  unfold cosineTorsionStrain
  simp [Real.cos_zero]

-- ---------------------------------------------------------------------------
-- Section 2: Strain-preserving uniform approximation
-- ---------------------------------------------------------------------------

/-- Strain-augmented decision problem: docking score minus internal strain.
    The strain depends only on the action (conformer), not the state (pose). -/
noncomputable def strainAugmentedDecisionProblem {A : Type u} {S : Type v}
    (base : DecisionProblem A S) (strain : A → ℝ) : DecisionProblem A S where
  utility := fun a s => base.utility a s - strain a

/-- Core integration theorem: adding an exact strain penalty to both sides
    of a uniform approximation preserves the error bound.

    Proof: |(exact(a,s) - strain(a)) - (coarse(a,s) - strain(a))|
         = |exact(a,s) - coarse(a,s)| ≤ δ

    Physical meaning: the strain is computed exactly from known torsion angles,
    so it contributes zero approximation error. Only the docking score
    approximation (exact vs cutoff) matters. -/
theorem strain_preserves_uniformApprox {A : Type u} {S : Type v}
    (exact coarse : DecisionProblem A S) (δ : ℝ)
    (strain : A → ℝ)
    (h : UniformUtilityApprox exact coarse δ) :
    UniformUtilityApprox
      (strainAugmentedDecisionProblem exact strain)
      (strainAugmentedDecisionProblem coarse strain)
      δ := by
  intro a s
  show |(exact.utility a s - strain a) - (coarse.utility a s - strain a)| ≤ δ
  have heq : (exact.utility a s - strain a) - (coarse.utility a s - strain a) =
      exact.utility a s - coarse.utility a s := by ring
  rw [heq]
  exact h a s

-- ---------------------------------------------------------------------------
-- Section 3: Lipschitz composition for strain-aware branch-and-bound
-- ---------------------------------------------------------------------------

/-- Combined (score - strain) Lipschitz constant.

    If the external score is L_s-Lipschitz in parameters and the strain
    penalty is L_st-Lipschitz in parameters, the strain-augmented score
    is (L_s + L_st)-Lipschitz. This certifies branch-and-bound cell
    bounds for the conformer search. -/
theorem strain_augmented_lipschitz
    {P : Type*} [PseudoMetricSpace P]
    (score strain : P → ℝ)
    (Ls Lst : NNReal)
    (h_score : LipschitzWith Ls score)
    (h_strain : LipschitzWith Lst strain) :
    LipschitzWith (Ls + Lst) (fun p => score p - strain p) :=
  h_score.sub h_strain

/-- Cellwise energy lower bound including strain, for branch-and-bound.
    If the combined (score - strain) is L-Lipschitz, then evaluating at
    the cell center and subtracting L × radius gives a valid lower bound
    throughout the cell. -/
theorem strain_energy_lower_bound_on_ball
    {P : Type*} [PseudoMetricSpace P]
    (score strain : P → ℝ)
    (L : NNReal)
    (h_lip : LipschitzWith L (fun p => score p - strain p))
    (p p₀ : P) (r : ℝ) (h_ball : dist p p₀ ≤ r) :
    (score p₀ - strain p₀) - L * r ≤ score p - strain p :=
  lipschitz_energy_lower_bound_on_ball (fun p => score p - strain p) L h_lip p p₀ r h_ball

-- ---------------------------------------------------------------------------
-- Section 4: Additive strain composition over multiple bonds
-- ---------------------------------------------------------------------------

/-- Abstract bounded strain: strain values lie in [0, B].
    Any bounded potential (cosine, harmonic with clamp, MMFF, UFF) satisfies
    this with an appropriate bound B. -/
def BoundedStrain (strain : α → ℝ) (B : ℝ) : Prop :=
  ∀ a, 0 ≤ strain a ∧ strain a ≤ B

/-- Cosine torsion strain is a BoundedStrain with bound 2Vk. -/
theorem cosineTorsionStrain_bounded (Vk n φ₀ : ℝ) (hVk : 0 ≤ Vk) :
    BoundedStrain (fun φ => cosineTorsionStrain Vk n φ φ₀) (2 * Vk) :=
  fun φ => ⟨cosineTorsionStrain_nonneg Vk n φ φ₀ hVk,
            cosineTorsionStrain_le_twoVk Vk n φ φ₀ hVk⟩

/-- Sum of two bounded strains is bounded by the sum of bounds. -/
theorem additive_strain_bounded
    (f g : α → ℝ) (Bf Bg : ℝ)
    (hf : BoundedStrain f Bf) (hg : BoundedStrain g Bg) :
    BoundedStrain (fun a => f a + g a) (Bf + Bg) := by
  intro a
  rcases hf a with ⟨hf_lo, hf_hi⟩
  rcases hg a with ⟨hg_lo, hg_hi⟩
  exact ⟨by linarith, by linarith⟩

/-- Strain-aware conformer pruning: if a conformer's docking score minus
    its strain is bounded below by lb, and another conformer achieves a
    combined score strictly below lb, the first conformer is dominated.

    This is energy_conformer_dominated applied to the strain-augmented
    energy, stated explicitly for clarity and handle mapping. -/
theorem strain_aware_conformer_dominated
    {A S : Type*}
    (energy : A → S → ℝ) (strain : A → ℝ)
    (a a' : A) (s' : S) (lb : ℝ)
    (h_lb : ∀ s, lb ≤ energy a s + strain a)
    (h_dom : energy a' s' + strain a' < lb) :
    ∀ s, energy a' s' + strain a' < energy a s + strain a := fun s =>
  lt_of_lt_of_le h_dom (h_lb s)

end LigandStrainApproximation
end Tractability
end DecisionQuotient
