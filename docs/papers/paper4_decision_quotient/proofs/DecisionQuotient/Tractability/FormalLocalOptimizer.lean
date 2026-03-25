/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/FormalLocalOptimizer.lean

  Small wrapper theorems connecting the local certified optimizer runtime design
  to existing admissibility, pruning, and ambiguity-band machinery.
-/
import DecisionQuotient.IntegrityCompetence
import DecisionQuotient.Computation.ArrayDSL
import DecisionQuotient.StochasticSequential.TemporalLearning
import DecisionQuotient.Tractability.CertifiedPruning
import DecisionQuotient.Tractability.FiniteTopK
import DecisionQuotient.Tractability.NearTieBand

namespace DecisionQuotient
namespace Tractability
namespace FormalLocalOptimizer

open IntegrityCompetence
open Computation.ArrayDSL
open CertifiedPruning
open FiniteTopK
open NearTieBand
open Classical

variable {A : Type*} [DecidableEq A]
variable {X : Type*} {Y : Type*} {W : Type*}

/-- Declared finite prior used by the local optimizer over a finite action family. -/
structure DeclaredFinitePrior (A : Type*) [Fintype A] where
  prob : A → ℝ
  nonneg : ∀ a, 0 ≤ prob a
  sum_one : Finset.univ.sum prob = 1

/-- Explicit action-selection branches used by the runtime. -/
inductive SelectionBranch where
  | ambiguityBand
  | supportFallback

/-- Explicit posterior-update branch used by the runtime. -/
inductive PosteriorUpdateBranch where
  | survivorConditioning

/-- Object-level witness for deterministic selection provenance. -/
structure SelectionWitness (A : Type*) [DecidableEq A] where
  branch : SelectionBranch
  support : Finset A
  choice : A
  sound : choice ∈ support

/-- Object-level witness for posterior-update provenance. -/
structure PosteriorUpdateWitness where
  branch : PosteriorUpdateBranch

/-- Combined object-level witness for the active belief update. -/
structure BeliefWitness (A : Type*) [DecidableEq A] where
  posteriorUpdate : PosteriorUpdateWitness
  selection : SelectionWitness A

/-- Combined object-level witness for one optimizer step. -/
structure OptimizerWitness (A : Type*) [DecidableEq A] where
  survivorSet : CertifiedPruning.CertifiedSurvivorSet A
  belief : BeliefWitness A

/-- Property-level output contract for a finite returned support. When this holds,
    every action in the declared output support satisfies the advertised property. -/
def CertifiedOutputSet (support : Finset A) (P : A → Prop) : Prop :=
  ∀ a, a ∈ support → P a

/--
  Strengthened optimizer witness ensuring that the selected support used by the
  runtime is exactly the certified survivor set produced by the pruning layer.
  This rules out category errors where a witness mixes unrelated supports.
-/
structure CoherentOptimizerWitness (A : Type*) [DecidableEq A] where
  survivorSet : CertifiedPruning.CertifiedSurvivorSet A
  belief : BeliefWitness A
  support_eq : belief.selection.support = survivorSet.survivors

/-- Forget the coherence proof and recover the original optimizer witness. -/
def CoherentOptimizerWitness.toOptimizerWitness
    {A : Type*} [DecidableEq A]
    (w : CoherentOptimizerWitness A) :
    OptimizerWitness A :=
  { survivorSet := w.survivorSet, belief := w.belief }

/-- In a coherent witness, every certified exact top-k action lies in the runtime support. -/
theorem CoherentOptimizerWitness.exactTopK_subset_support
    {A : Type*} [DecidableEq A]
    (w : CoherentOptimizerWitness A) :
    w.survivorSet.certificate.exactTopK ⊆ w.belief.selection.support := by
  intro a ha
  rw [w.support_eq]
  rw [w.survivorSet.survivors_eq]
  exact w.survivorSet.certificate.sound ha

/-- In a coherent witness, the chosen action lies in the certified survivor set. -/
theorem CoherentOptimizerWitness.choice_mem_survivors
    {A : Type*} [DecidableEq A]
    (w : CoherentOptimizerWitness A) :
    w.belief.selection.choice ∈ w.survivorSet.survivors := by
  rw [← w.support_eq]
  exact w.belief.selection.sound

/-- Any property proved for every member of a selection support holds in
    particular for the deterministically chosen action. -/
theorem SelectionWitness.choice_inherits_of_support
    {A : Type*} [DecidableEq A]
    (w : SelectionWitness A)
    {P : A → Prop}
    (hSupport : CertifiedOutputSet w.support P) :
    P w.choice :=
  hSupport w.choice w.sound

/-- Belief-level specialization of `SelectionWitness.choice_inherits_of_support`. -/
theorem BeliefWitness.choice_inherits_of_support
    {A : Type*} [DecidableEq A]
    (w : BeliefWitness A)
    {P : A → Prop}
    (hSupport : CertifiedOutputSet w.selection.support P) :
    P w.selection.choice :=
  w.selection.choice_inherits_of_support hSupport

/-- Coherent optimizer witnesses transport support-level property guarantees to
    the returned action immediately. This is the proof-side hook for both unique
    winner contracts and ambiguity-set contracts. -/
theorem CoherentOptimizerWitness.choice_inherits_of_support
    {A : Type*} [DecidableEq A]
    (w : CoherentOptimizerWitness A)
    {P : A → Prop}
    (hSupport : CertifiedOutputSet w.belief.selection.support P) :
    P w.belief.selection.choice :=
  w.belief.choice_inherits_of_support hSupport

/-- If a coherent witness advertises singleton support `{aStar}`, then its chosen
    action is exactly `aStar`. -/
theorem CoherentOptimizerWitness.choice_eq_of_singleton_support
    {A : Type*} [DecidableEq A]
    (w : CoherentOptimizerWitness A)
    (aStar : A)
    (hSingleton : w.belief.selection.support = ({aStar} : Finset A)) :
    w.belief.selection.choice = aStar := by
  have hMem : w.belief.selection.choice ∈ ({aStar} : Finset A) := by
    simpa [hSingleton] using w.belief.selection.sound
  simpa using hMem

/-- Singleton-support specialization of `choice_inherits_of_support`. -/
theorem CoherentOptimizerWitness.choice_inherits_of_singleton_support
    {A : Type*} [DecidableEq A]
    (w : CoherentOptimizerWitness A)
    (aStar : A)
    {P : A → Prop}
    (hSingleton : w.belief.selection.support = ({aStar} : Finset A))
    (hP : P aStar) :
    P w.belief.selection.choice := by
  rw [w.choice_eq_of_singleton_support aStar hSingleton]
  exact hP

/-- Support restriction of a finite prior along a certified survivor set. -/
def restrictedPrior {A : Type*} [Fintype A] [DecidableEq A]
    (prior : DeclaredFinitePrior A)
    (survivors : Finset A) : A → ℝ :=
  fun a => if a ∈ survivors then prior.prob a else 0

/-- A certified pruning certificate directly yields the survivor containment fact
    used by the runtime support-restriction update. -/
theorem survivor_certificate_support
    (cert : PruningCertificate A) :
    cert.exactTopK ⊆ cert.survivors :=
  cert.sound

/-- Any typed evidence object for a declared report still yields an admissible
    claim in the local optimizer setting; this is the paper-side wrapper used by
    the runtime posterior update. -/
theorem conditioned_claim_admissible_of_evidence
    (R : Set (X × Y)) (Rε : EpsilonRelation X Y)
    (Γ : Regime X) (Q : CertifyingSolver X Y W)
    (r : ClaimReport) :
    EvidenceForReport R Rε Γ Q r → ClaimAdmissible R Rε Γ Q r :=
  claim_admissible_of_evidence (R := R) (Rε := Rε) (Γ := Γ) (Q := Q) (r := r)

/-- Deterministic tie-breaking by selecting the minimum element of the certified
    ambiguity band stays inside the ambiguity band whenever the band is nonempty.
    This is the finite ordered analogue of the runtime's stable action selection. -/
theorem deterministic_pick_mem_ambiguityBand
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (u : A → ℝ) (k : Nat) (hk : 0 < k) (eps : ℝ)
    (hBand : (ambiguityBand u k hk eps).Nonempty) :
    (ambiguityBand u k hk eps).min' hBand ∈ ambiguityBand u k hk eps := by
  exact Finset.min'_mem _ _

/-- Deterministic tie-breaking by selecting the minimum element of a nonempty
    posterior support set stays inside that support set. This matches the
    runtime fallback branch when the ambiguity band is empty. -/
theorem deterministic_pick_mem_supportSet
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (support : Finset A)
    (hSupport : support.Nonempty) :
    support.min' hSupport ∈ support := by
  exact Finset.min'_mem _ _

/-- Restricted prior places zero mass outside the declared survivor support. -/
theorem restrictedPrior_zero_outside
    {A : Type*} [Fintype A] [DecidableEq A]
    (prior : DeclaredFinitePrior A)
    (survivors : Finset A)
    (a : A)
    (hOut : a ∉ survivors) :
    restrictedPrior prior survivors a = 0 := by
  simp [restrictedPrior, hOut]

/-- If the survivor set is the full support, support restriction leaves the prior unchanged. -/
theorem restrictedPrior_eq_prior_of_full_support
    {A : Type*} [Fintype A] [DecidableEq A]
    (prior : DeclaredFinitePrior A)
    (survivors : Finset A)
    (hFull : survivors = (Finset.univ : Finset A)) :
    restrictedPrior prior survivors = prior.prob := by
  funext a
  simp [restrictedPrior, hFull]

/-- Normalized support restriction still assigns zero mass outside the declared
    survivor mask. This is the runtime shape used by posterior updates. -/
theorem normalize_supportConditioning_zero_of_mask_false
    {n : ℕ}
    (probs : MDArray n)
    (mask : Fin n → Bool)
    (i : Fin n)
    (hFalse : mask i = false) :
    normalizeProbabilityVector (supportConditioning probs mask) i = 0 := by
  unfold normalizeProbabilityVector
  change supportConditioning probs mask i / reduce_sum (supportConditioning probs mask) = 0
  rw [supportConditioning_zero_of_mask_false probs mask i hFalse]
  simp

/-- Support restriction followed by normalization yields a unit-sum posterior
    whenever the restricted support has positive total mass. -/
theorem normalize_supportConditioning_sum_one
    {n : ℕ}
    (probs : MDArray n)
    (mask : Fin n → Bool)
    (hPos : 0 < reduce_sum (supportConditioning probs mask)) :
    reduce_sum (normalizeProbabilityVector (supportConditioning probs mask)) = 1 := by
  exact normalizeProbabilityVector_sum_one _ hPos

/-- The runtime survivor-conditioning update is pointwise identical to a Bayesian
    posterior with indicator likelihood and evidence equal to the restricted mass. -/
theorem normalize_supportConditioning_eq_bayesian_posterior
    {n : ℕ}
    (probs : MDArray n)
    (mask : Fin n → Bool)
    (i : Fin n) :
    normalizeProbabilityVector (supportConditioning probs mask) i =
      DecisionQuotient.StochasticSequential.posterior
        (fun j => probs j)
        (fun j => if mask j then (1 : ℝ) else 0)
        (reduce_sum (supportConditioning probs mask))
        i := by
  by_cases h : mask i
  · have hNorm : normalizeProbabilityVector (supportConditioning probs mask) i =
        probs i / reduce_sum (supportConditioning probs mask) := by
      change (if mask i = true then probs i else 0) / reduce_sum (supportConditioning probs mask) =
          probs i / reduce_sum (supportConditioning probs mask)
      simp [h]
    rw [hNorm]
    simp [DecisionQuotient.StochasticSequential.posterior, h]
  · have hFalse : mask i = false := by simp [h]
    rw [normalize_supportConditioning_zero_of_mask_false probs mask i hFalse]
    simp [DecisionQuotient.StochasticSequential.posterior, supportConditioning, h]

/-- Branch-indexed selection membership theorem. -/
theorem selection_branch_member
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (branch : SelectionBranch) :
    match branch with
    | .ambiguityBand =>
        ∀ (u : A → ℝ) (k : Nat) (hk : 0 < k) (eps : ℝ)
          (hBand : (ambiguityBand u k hk eps).Nonempty),
          (ambiguityBand u k hk eps).min' hBand ∈ ambiguityBand u k hk eps
    | .supportFallback =>
        ∀ (support : Finset A) (hSupport : support.Nonempty),
          support.min' hSupport ∈ support
  := by
  cases branch
  · intro u k hk eps hBand
    exact deterministic_pick_mem_ambiguityBand u k hk eps hBand
  · intro support hSupport
    exact deterministic_pick_mem_supportSet support hSupport

/-- Branch-indexed posterior-update theorem. -/
theorem posterior_update_branch_eq_bayesian_posterior
    (branch : PosteriorUpdateBranch)
    {n : ℕ}
    (probs : MDArray n)
    (mask : Fin n → Bool)
    (i : Fin n) :
    match branch with
    | .survivorConditioning =>
        normalizeProbabilityVector (supportConditioning probs mask) i =
          DecisionQuotient.StochasticSequential.posterior
            (fun j => probs j)
            (fun j => if mask j then (1 : ℝ) else 0)
            (reduce_sum (supportConditioning probs mask))
            i := by
  cases branch
  exact normalize_supportConditioning_eq_bayesian_posterior probs mask i

noncomputable def selectionWitness_of_ambiguityBand
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (u : A → ℝ) (k : Nat) (hk : 0 < k) (eps : ℝ)
    (hBand : (ambiguityBand u k hk eps).Nonempty) :
    SelectionWitness A :=
  { branch := SelectionBranch.ambiguityBand
    support := ambiguityBand u k hk eps
    choice := (ambiguityBand u k hk eps).min' hBand
    sound := deterministic_pick_mem_ambiguityBand u k hk eps hBand }

noncomputable def selectionWitness_of_supportSet
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (support : Finset A)
    (hSupport : support.Nonempty) :
    SelectionWitness A :=
  { branch := SelectionBranch.supportFallback
    support := support
    choice := support.min' hSupport
    sound := deterministic_pick_mem_supportSet support hSupport }

def posteriorUpdateWitness_of_survivorConditioning : PosteriorUpdateWitness :=
  { branch := PosteriorUpdateBranch.survivorConditioning }

noncomputable def beliefWitness_of_survivorConditioning_ambiguityBand
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (u : A → ℝ) (k : Nat) (hk : 0 < k) (eps : ℝ)
    (hBand : (ambiguityBand u k hk eps).Nonempty) :
    BeliefWitness A :=
  { posteriorUpdate := posteriorUpdateWitness_of_survivorConditioning
    selection := selectionWitness_of_ambiguityBand u k hk eps hBand }

noncomputable def beliefWitness_of_survivorConditioning_supportSet
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (support : Finset A)
    (hSupport : support.Nonempty) :
    BeliefWitness A :=
  { posteriorUpdate := posteriorUpdateWitness_of_survivorConditioning
    selection := selectionWitness_of_supportSet support hSupport }

noncomputable def optimizerWitness_of_exact_top1
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact : A → ℝ)
    (hTop : (topKSet uExact 1).Nonempty) :
    OptimizerWitness A :=
  let hBand : (ambiguityBand uExact 1 (by omega) 0).Nonempty := by
    rw [ambiguityBand_zero_eq_top1]
    exact hTop
  { survivorSet := CertifiedPruning.certifiedSurvivorSet_of_exact_top1 uExact
    belief := beliefWitness_of_survivorConditioning_ambiguityBand uExact 1 (by omega) 0 hBand }

noncomputable def coherentOptimizerWitness_of_exact_top1
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact : A → ℝ)
    (hTop : (topKSet uExact 1).Nonempty) :
    CoherentOptimizerWitness A :=
  let hBand : (ambiguityBand uExact 1 (by omega) 0).Nonempty := by
    rw [ambiguityBand_zero_eq_top1]
    exact hTop
  { survivorSet := CertifiedPruning.certifiedSurvivorSet_of_exact_top1 uExact
    belief := beliefWitness_of_survivorConditioning_ambiguityBand uExact 1 (by omega) 0 hBand
    support_eq := by
      change ambiguityBand uExact 1 (by omega) 0 =
        (CertifiedPruning.certificate_of_exact_top1 uExact).survivors
      simp [CertifiedPruning.certificate_of_exact_top1, NearTieBand.ambiguityBand_zero_eq_top1] }

noncomputable def optimizerWitness_of_top1_coarse_ambiguityBand
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta)
    (hBand : (ambiguityBand uCoarse 1 (by omega) (2 * delta)).Nonempty) :
    OptimizerWitness A :=
  { survivorSet :=
      CertifiedPruning.certifiedSurvivorSet_of_top1_coarse_ambiguityBand
        uExact uCoarse delta hApprox hDelta
    belief := beliefWitness_of_survivorConditioning_ambiguityBand uCoarse 1 (by omega) (2 * delta) hBand }

noncomputable def coherentOptimizerWitness_of_top1_coarse_ambiguityBand
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta)
    (hBand : (ambiguityBand uCoarse 1 (by omega) (2 * delta)).Nonempty) :
    CoherentOptimizerWitness A :=
  { survivorSet :=
      CertifiedPruning.certifiedSurvivorSet_of_top1_coarse_ambiguityBand
        uExact uCoarse delta hApprox hDelta
    belief := beliefWitness_of_survivorConditioning_ambiguityBand uCoarse 1 (by omega) (2 * delta) hBand
    support_eq := by
      change ambiguityBand uCoarse 1 (by omega) (2 * delta) =
        (CertifiedPruning.certificate_of_top1_coarse_ambiguityBand uExact uCoarse delta hApprox hDelta).survivors
      rfl }

noncomputable def optimizerWitness_of_exact_singleton_winner
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact uCoarse : A → ℝ)
    (aStar : A)
    (delta : ℝ)
    (hApprox : ∀ x, |uExact x - uCoarse x| ≤ delta)
    (hStrict : ∀ b, b ≠ aStar → RankingPreservation.PairwiseGap uCoarse aStar b > 2 * delta) :
    OptimizerWitness A :=
  { survivorSet :=
      CertifiedPruning.certifiedSurvivorSet_of_exact_singleton_winner
        uExact uCoarse aStar delta hApprox hStrict
    belief := beliefWitness_of_survivorConditioning_supportSet ({aStar} : Finset A) (by simp) }

noncomputable def coherentOptimizerWitness_of_exact_singleton_winner
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact uCoarse : A → ℝ)
    (aStar : A)
    (delta : ℝ)
    (hApprox : ∀ x, |uExact x - uCoarse x| ≤ delta)
    (hStrict : ∀ b, b ≠ aStar → RankingPreservation.PairwiseGap uCoarse aStar b > 2 * delta) :
    CoherentOptimizerWitness A :=
  { survivorSet :=
      CertifiedPruning.certifiedSurvivorSet_of_exact_singleton_winner
        uExact uCoarse aStar delta hApprox hStrict
    belief := beliefWitness_of_survivorConditioning_supportSet ({aStar} : Finset A) (by simp)
    support_eq := by
      change ({aStar} : Finset A) =
        (CertifiedPruning.certificate_of_exact_singleton_winner uExact uCoarse aStar delta hApprox hStrict).survivors
      rfl }

/-- If the exact top-1 set is already known to be the singleton `{aStar}`, the
    runtime can package the support-fallback branch with singleton support as a
    coherent optimizer witness directly, without going through an additional
    coarse-margin theorem. This matches runtime paths that certify singleton
    winners from exact score gaps inside a finite retained family. -/
noncomputable def coherentOptimizerWitness_of_exact_singleton_top1
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact : A → ℝ)
    (aStar : A)
    (hSingleton : topKSet uExact 1 = ({aStar} : Finset A)) :
    CoherentOptimizerWitness A :=
  { survivorSet :=
      { survivors := ({aStar} : Finset A)
        certificate :=
          { survivors := ({aStar} : Finset A)
            exactTopK := topKSet uExact 1
            sound := by
              intro a ha
              rw [hSingleton] at ha
              exact ha }
        survivors_eq := rfl }
    belief := beliefWitness_of_survivorConditioning_supportSet ({aStar} : Finset A) (by simp)
    support_eq := rfl }

theorem coherentOptimizerWitness_of_exact_singleton_top1_choice
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact : A → ℝ)
    (aStar : A)
    (hSingleton : topKSet uExact 1 = ({aStar} : Finset A)) :
    let w := coherentOptimizerWitness_of_exact_singleton_top1 uExact aStar hSingleton
    w.belief.selection.choice = aStar ∧
      w.belief.selection.support = ({aStar} : Finset A) := by
  simp [coherentOptimizerWitness_of_exact_singleton_top1,
    beliefWitness_of_survivorConditioning_supportSet,
    selectionWitness_of_supportSet]

/-- If two exact score families both certify the same singleton top-1 action,
    then the coherent optimizer witness induced by either family has the same
    support and the same selected action. This is the ranking-consistency bridge
    used when a richer score is only trusted after it agrees with a simpler
    theorem-backed disambiguation score on a certified singleton winner. -/
theorem exact_singleton_top1_choice_agreement
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (u₁ u₂ : A → ℝ)
    (aStar : A)
    (h₁ : topKSet u₁ 1 = ({aStar} : Finset A))
    (h₂ : topKSet u₂ 1 = ({aStar} : Finset A)) :
    let w₁ := coherentOptimizerWitness_of_exact_singleton_top1 u₁ aStar h₁
    let w₂ := coherentOptimizerWitness_of_exact_singleton_top1 u₂ aStar h₂
    w₁.belief.selection.choice = w₂.belief.selection.choice ∧
      w₁.belief.selection.support = w₂.belief.selection.support := by
  have hw₁ := coherentOptimizerWitness_of_exact_singleton_top1_choice u₁ aStar h₁
  have hw₂ := coherentOptimizerWitness_of_exact_singleton_top1_choice u₂ aStar h₂
  simp at hw₁ hw₂
  aesop

/-- Energy-form runtime bridge: if two coarse energy families each certify the
    same singleton top-1 action `aStar` under their respective uniform error
    bounds, then the induced coherent optimizer witnesses agree on the selected
    action and support. This is the theorem-facing version of the runtime policy
    that only trusts a richer score after shared singleton certification with a
    simpler disambiguation score. -/
theorem exact_energy_gap_certified_choice_agreement
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (e₁Exact e₁Coarse e₂Exact e₂Coarse : A → ℝ)
    (aStar : A)
    (δ₁ δ₂ : ℝ)
    (hApprox₁ : ∀ x, |e₁Exact x - e₁Coarse x| ≤ δ₁)
    (hApprox₂ : ∀ x, |e₂Exact x - e₂Coarse x| ≤ δ₂)
    (hStrict₁ : ∀ b, b ≠ aStar → e₁Coarse aStar + 2 * δ₁ < e₁Coarse b)
    (hStrict₂ : ∀ b, b ≠ aStar → e₂Coarse aStar + 2 * δ₂ < e₂Coarse b) :
    let w₁ := coherentOptimizerWitness_of_exact_singleton_top1
      (fun a => -e₁Exact a) aStar
      (RankingPreservation.exact_top1_eq_singleton_of_coarse_energy_gap_margin
        e₁Exact e₁Coarse aStar δ₁ hApprox₁ hStrict₁)
    let w₂ := coherentOptimizerWitness_of_exact_singleton_top1
      (fun a => -e₂Exact a) aStar
      (RankingPreservation.exact_top1_eq_singleton_of_coarse_energy_gap_margin
        e₂Exact e₂Coarse aStar δ₂ hApprox₂ hStrict₂)
    w₁.belief.selection.choice = w₂.belief.selection.choice ∧
      w₁.belief.selection.support = w₂.belief.selection.support := by
  simpa using exact_singleton_top1_choice_agreement
    (u₁ := fun a => -e₁Exact a)
    (u₂ := fun a => -e₂Exact a)
    (aStar := aStar)
    (h₁ := RankingPreservation.exact_top1_eq_singleton_of_coarse_energy_gap_margin
      e₁Exact e₁Coarse aStar δ₁ hApprox₁ hStrict₁)
    (h₂ := RankingPreservation.exact_top1_eq_singleton_of_coarse_energy_gap_margin
      e₂Exact e₂Coarse aStar δ₂ hApprox₂ hStrict₂)

/-- Property inheritance for the exact-energy singleton-winner branch: once a
    coarse energy gap margin certifies that the exact top-1 support is exactly
    `{aStar}`, every property already proved for `aStar` transfers to the
    runtime's returned choice. -/
theorem exact_energy_gap_certified_choice_inherits_property
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (eExact eCoarse : A → ℝ)
    (aStar : A)
    (δ : ℝ)
    (hApprox : ∀ x, |eExact x - eCoarse x| ≤ δ)
    (hStrict : ∀ b, b ≠ aStar → eCoarse aStar + 2 * δ < eCoarse b)
    {P : A → Prop}
    (hP : P aStar) :
    let w := coherentOptimizerWitness_of_exact_singleton_top1
      (fun a => -eExact a) aStar
      (RankingPreservation.exact_top1_eq_singleton_of_coarse_energy_gap_margin
        eExact eCoarse aStar δ hApprox hStrict)
    P w.belief.selection.choice := by
  let hSingleton :=
    RankingPreservation.exact_top1_eq_singleton_of_coarse_energy_gap_margin
      eExact eCoarse aStar δ hApprox hStrict
  let w : CoherentOptimizerWitness A :=
    coherentOptimizerWitness_of_exact_singleton_top1 (fun a => -eExact a) aStar hSingleton
  have hw := coherentOptimizerWitness_of_exact_singleton_top1_choice
    (uExact := fun a => -eExact a) aStar hSingleton
  have hChoiceEq : w.belief.selection.choice = aStar := by
    simpa [w] using hw.1
  simpa [w, hChoiceEq] using hP

noncomputable def optimizerWitness_of_top1_branch
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (branch : CertifiedPruning.Top1PruningBranch)
    (uExact uCoarse : A → ℝ)
    (aStar : A)
    (delta : ℝ)
    (hApprox : ∀ x, |uExact x - uCoarse x| ≤ delta)
    (hDelta : 0 ≤ delta)
    (hTop : (topKSet uExact 1).Nonempty)
    (hBand : (ambiguityBand uCoarse 1 (by omega) (2 * delta)).Nonempty)
    (hStrict : ∀ b, b ≠ aStar → RankingPreservation.PairwiseGap uCoarse aStar b > 2 * delta) :
    OptimizerWitness A :=
  match branch with
  | .exactTop1 => optimizerWitness_of_exact_top1 uExact hTop
  | .exactSingletonWinner =>
      optimizerWitness_of_exact_singleton_winner uExact uCoarse aStar delta hApprox hStrict
  | .top1CoarseAmbiguityBand =>
      optimizerWitness_of_top1_coarse_ambiguityBand uExact uCoarse delta hApprox hDelta hBand

end FormalLocalOptimizer
end Tractability
end DecisionQuotient
