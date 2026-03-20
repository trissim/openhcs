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
import DecisionQuotient.Tractability.NearTieBand

namespace DecisionQuotient
namespace Tractability
namespace FormalLocalOptimizer

open IntegrityCompetence
open Computation.ArrayDSL
open CertifiedPruning
open NearTieBand
open Classical

variable {A : Type*} [DecidableEq A]
variable {X : Type*} {Y : Type*} {W : Type*}

/-- Declared finite prior used by the local optimizer over a finite action family. -/
structure DeclaredFinitePrior (A : Type*) [Fintype A] where
  prob : A → ℝ
  nonneg : ∀ a, 0 ≤ prob a
  sum_one : Finset.univ.sum prob = 1

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

end FormalLocalOptimizer
end Tractability
end DecisionQuotient
