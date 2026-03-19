/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/CertifiedPruning.lean

  Certificate objects for theorem-backed pruning.
-/
import DecisionQuotient.Tractability.TopKPreservation

namespace DecisionQuotient
namespace Tractability
namespace CertifiedPruning

open FiniteTopK
open TopKPreservation

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

end CertifiedPruning
end Tractability
end DecisionQuotient
