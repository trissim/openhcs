/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/CertifiedPruning.lean

  Certificate objects for theorem-backed pruning.
-/
import DecisionQuotient.Tractability.TopKPreservation
import DecisionQuotient.Tractability.NearTieBand
import DecisionQuotient.Tractability.RankingPreservation

namespace DecisionQuotient
namespace Tractability
namespace CertifiedPruning

open FiniteTopK
open TopKPreservation
open NearTieBand

variable {A : Type*} [Fintype A] [DecidableEq A]

/-- A theorem-backed pruning certificate: every exact top-k action survives in
    the retained survivor set. -/
structure PruningCertificate (A : Type*) [DecidableEq A] where
  survivors : Finset A
  exactTopK : Finset A
  sound : exactTopK ⊆ survivors

/-- A survivor set packaged together with a pruning certificate. -/
structure CertifiedSurvivorSet (A : Type*) [DecidableEq A] where
  survivors : Finset A
  certificate : PruningCertificate A
  survivors_eq : survivors = certificate.survivors

/-- Explicit top-1 pruning branches used by the runtime and proof packaging. -/
inductive Top1PruningBranch where
  | exactTop1
  | exactSingletonWinner
  | top1CoarseAmbiguityBand

/-- Package the basic top-k survivor containment theorem into a certificate. -/
noncomputable def certificate_of_topK_margin
    (uExact uCoarse : A → ℝ)
    (k : Nat)
    (tau delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hMargin : ∀ a, a ∈ topKWithTies uExact k → tau + delta ≤ uExact a) :
    PruningCertificate A :=
  { survivors := survivorSet uCoarse tau
    exactTopK := topKWithTies uExact k
    sound := exact_topK_subset_survivorSet_of_margin uExact uCoarse k tau delta hApprox hMargin }

theorem certificate_sound
    (uExact uCoarse : A → ℝ)
    (k : Nat)
    (tau delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hMargin : ∀ a, a ∈ topKWithTies uExact k → tau + delta ≤ uExact a) :
    (certificate_of_topK_margin uExact uCoarse k tau delta hApprox hMargin).exactTopK
      ⊆ (certificate_of_topK_margin uExact uCoarse k tau delta hApprox hMargin).survivors :=
  (certificate_of_topK_margin uExact uCoarse k tau delta hApprox hMargin).sound

noncomputable def certifiedSurvivorSet_of_topK_margin
    (uExact uCoarse : A → ℝ)
    (k : Nat)
    (tau delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hMargin : ∀ a, a ∈ topKWithTies uExact k → tau + delta ≤ uExact a) :
    CertifiedSurvivorSet A :=
  let cert := certificate_of_topK_margin uExact uCoarse k tau delta hApprox hMargin
  { survivors := cert.survivors
    certificate := cert
    survivors_eq := rfl }

/-- Exact-path top-1 pruning certificate: when no coarse relaxation is used, the
    survivor set is exactly the top-1-with-ties set. -/
noncomputable def certificate_of_exact_top1
    {A : Type*} [Fintype A] [DecidableEq A] :
    (uExact : A → ℝ) → PruningCertificate A
  | uExact =>
    { survivors := topKSet uExact 1
      exactTopK := topKSet uExact 1
      sound := by intro a ha; exact ha }

theorem certificate_exact_top1_sound
    {A : Type*} [Fintype A] [DecidableEq A]
    (uExact : A → ℝ) :
    (certificate_of_exact_top1 uExact).exactTopK ⊆
      (certificate_of_exact_top1 uExact).survivors :=
  (certificate_of_exact_top1 uExact).sound

noncomputable def certifiedSurvivorSet_of_exact_top1
    {A : Type*} [Fintype A] [DecidableEq A]
    (uExact : A → ℝ) :
    CertifiedSurvivorSet A :=
  let cert := certificate_of_exact_top1 uExact
  { survivors := cert.survivors
    certificate := cert
    survivors_eq := rfl }

/-- Package the coarse top-1 ambiguity band into a pruning certificate. Under a
    uniform approximation radius `delta`, every exact top-1 action survives in
    the coarse ambiguity band of width `2 * delta`. -/
noncomputable def certificate_of_top1_coarse_ambiguityBand
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    PruningCertificate A :=
  { survivors := ambiguityBand uCoarse 1 (by omega) (2 * delta)
    exactTopK := topKSet uExact 1
    sound := exact_top1_subset_coarse_ambiguityBand_of_uniform_error uExact uCoarse delta hApprox hDelta }

theorem certificate_top1_coarse_ambiguityBand_sound
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    (certificate_of_top1_coarse_ambiguityBand uExact uCoarse delta hApprox hDelta).exactTopK
      ⊆ (certificate_of_top1_coarse_ambiguityBand uExact uCoarse delta hApprox hDelta).survivors :=
  (certificate_of_top1_coarse_ambiguityBand uExact uCoarse delta hApprox hDelta).sound

noncomputable def certifiedSurvivorSet_of_top1_coarse_ambiguityBand
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    CertifiedSurvivorSet A :=
  let cert := certificate_of_top1_coarse_ambiguityBand uExact uCoarse delta hApprox hDelta
  { survivors := cert.survivors
    certificate := cert
    survivors_eq := rfl }

/-- If a coarse winner has pairwise margin `> 2 * delta` against every rival,
    the exact top-1 set collapses to that singleton winner. -/
noncomputable def certificate_of_exact_singleton_winner
    {A : Type*} [Fintype A] [DecidableEq A]
    (uExact uCoarse : A → ℝ)
    (aStar : A)
    (delta : ℝ)
    (hApprox : ∀ x, |uExact x - uCoarse x| ≤ delta)
    (hStrict : ∀ b, b ≠ aStar → RankingPreservation.PairwiseGap uCoarse aStar b > 2 * delta) :
    PruningCertificate A :=
  { survivors := ({aStar} : Finset A)
    exactTopK := topKSet uExact 1
    sound := by
      intro a ha
      rw [Finset.mem_singleton]
      by_contra hne
      have hlt : uExact a < uExact aStar :=
        RankingPreservation.exact_strictOpt_of_coarse_strictOpt_margin uExact uCoarse aStar delta hApprox hStrict a hne
      have hmem : aStar ∈ (Finset.univ : Finset A).filter (fun x => uExact a < uExact x) := by
        simp [hlt]
      rw [mem_topKSet_iff] at ha
      unfold strictBetterCount at ha
      have hCardPos : 0 < ((Finset.univ : Finset A).filter (fun x => uExact a < uExact x)).card :=
        Finset.card_pos.mpr ⟨aStar, hmem⟩
      omega }

theorem certificate_exact_singleton_winner_sound
    {A : Type*} [Fintype A] [DecidableEq A]
    (uExact uCoarse : A → ℝ)
    (aStar : A)
    (delta : ℝ)
    (hApprox : ∀ x, |uExact x - uCoarse x| ≤ delta)
    (hStrict : ∀ b, b ≠ aStar → RankingPreservation.PairwiseGap uCoarse aStar b > 2 * delta) :
    (certificate_of_exact_singleton_winner uExact uCoarse aStar delta hApprox hStrict).exactTopK
      ⊆ (certificate_of_exact_singleton_winner uExact uCoarse aStar delta hApprox hStrict).survivors :=
  (certificate_of_exact_singleton_winner uExact uCoarse aStar delta hApprox hStrict).sound

noncomputable def certifiedSurvivorSet_of_exact_singleton_winner
    {A : Type*} [Fintype A] [DecidableEq A]
    (uExact uCoarse : A → ℝ)
    (aStar : A)
    (delta : ℝ)
    (hApprox : ∀ x, |uExact x - uCoarse x| ≤ delta)
    (hStrict : ∀ b, b ≠ aStar → RankingPreservation.PairwiseGap uCoarse aStar b > 2 * delta) :
    CertifiedSurvivorSet A :=
  let cert := certificate_of_exact_singleton_winner uExact uCoarse aStar delta hApprox hStrict
  { survivors := cert.survivors
    certificate := cert
    survivors_eq := rfl }

/-- Branch-indexed certificate constructor for top-1 pruning. -/
noncomputable def certificate_of_top1_branch
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (branch : Top1PruningBranch)
    (uExact uCoarse : A → ℝ)
    (aStar : A)
    (delta : ℝ)
    (hApprox : ∀ x, |uExact x - uCoarse x| ≤ delta)
    (hDelta : 0 ≤ delta)
    (hStrict : ∀ b, b ≠ aStar → RankingPreservation.PairwiseGap uCoarse aStar b > 2 * delta) :
    PruningCertificate A :=
  match branch with
  | .exactTop1 => certificate_of_exact_top1 uExact
  | .exactSingletonWinner => certificate_of_exact_singleton_winner uExact uCoarse aStar delta hApprox hStrict
  | .top1CoarseAmbiguityBand => certificate_of_top1_coarse_ambiguityBand uExact uCoarse delta hApprox hDelta

noncomputable def certifiedSurvivorSet_of_top1_branch
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (branch : Top1PruningBranch)
    (uExact uCoarse : A → ℝ)
    (aStar : A)
    (delta : ℝ)
    (hApprox : ∀ x, |uExact x - uCoarse x| ≤ delta)
    (hDelta : 0 ≤ delta)
    (hStrict : ∀ b, b ≠ aStar → RankingPreservation.PairwiseGap uCoarse aStar b > 2 * delta) :
    CertifiedSurvivorSet A :=
  let cert := certificate_of_top1_branch branch uExact uCoarse aStar delta hApprox hDelta hStrict
  { survivors := cert.survivors
    certificate := cert
    survivors_eq := rfl }

end CertifiedPruning
end Tractability
end DecisionQuotient
