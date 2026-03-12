import DecisionQuotient.IntegrityCompetence
import DecisionQuotient.Physics.PhysicalHardness
import DecisionQuotient.Physics.ClaimTransport
import Mathlib.Tactic

/-!
# Physical Core Theorems (Conditional, Mechanized)

This module isolates the strongest physically grounded statements that are fully
mechanized from existing paper machinery:

1. Finite physical budget blocks exact certification beyond a threshold
   (conditional on required-operation lower bound).
2. Substrate transport from physical encodings to core claims is sound.
3. Under evidence-gated RLFF, abstain is a strict global maximizer when all
   non-abstain reports are uncertified and abstain beats penalty.
4. Certified-bit accounting induces a physical work lower bound under positive
   per-bit cost.
-/

namespace DecisionQuotient
namespace Physics
namespace PhysicalCore

open ClaimTransport
open DecisionQuotient.IntegrityCompetence
open PhysicalComplexity

universe u v w

/-! ## 1) Finite-budget impossibility (conditional) -/

/-- Finite physical budget blocks exact certification beyond a size threshold,
provided the required operation count has an exponential lower bound. -/
theorem finite_budget_blocks_exact_certification_beyond_threshold
    (c : PhysicalComputer)
    (req : ComputationalRequirement) :
    ∃ n₀ : ℕ, ∀ n : ℕ, n ≥ n₀ →
      c.E_max < req.required_ops n * bit_energy_cost c :=
  coNP_physically_impossible c req

/-- Canonical specialization to the module's exponential requirement model. -/
theorem finite_budget_blocks_canonical_exponential_certification
    (c : PhysicalComputer) :
    ∃ n₀ : ℕ, ∀ n : ℕ, n ≥ n₀ →
      c.E_max < coNP_requirement.required_ops n * bit_energy_cost c :=
  sufficiency_physically_impossible c

/-! ## 2) Substrate transport soundness -/

/-- Sound conditional transport: if a core theorem holds under transferred
assumptions, it holds for all encoded physical instances in scope. -/
theorem substrate_transport_sound_conditional
    {PInst : Type u} {A : Type v} {S : Type w}
    (E : PhysicalEncoding PInst A S)
    (CoreAssm : DecisionProblem A S → Prop)
    (CoreClaim : DecisionProblem A S → Prop)
    (PhysAssm : PInst → Prop)
    (hCore : ∀ d : DecisionProblem A S, CoreAssm d → CoreClaim d)
    (hAssmTransfer : ∀ p : PInst, PhysAssm p → CoreAssm (E.encode p)) :
    ∀ p : PInst, PhysAssm p → CoreClaim (E.encode p) :=
  physical_claim_lifts_from_core_conditional
    (E := E) (CoreAssm := CoreAssm) (CoreClaim := CoreClaim)
    (PhysAssm := PhysAssm) hCore hAssmTransfer

/-- If the transported core theorem holds on encoded instances,
no encoded physical counterexample exists in-scope. -/
theorem substrate_transport_blocks_encoded_counterexamples
    {PInst : Type u} {A : Type v} {S : Type w}
    (E : PhysicalEncoding PInst A S)
    (PhysAssm : PInst → Prop)
    (CoreClaim : DecisionProblem A S → Prop)
    (hLifted : ∀ p : PInst, PhysAssm p → CoreClaim (E.encode p)) :
    ¬ ∃ p : PInst, PhysAssm p ∧ ¬ CoreClaim (E.encode p) :=
  no_physical_counterexample_of_core_theorem
    (E := E) (PhysAssm := PhysAssm) (CoreClaim := CoreClaim) hLifted

/-! ## 3) Evidence-gated abstain optimality -/

/-- Under typed evidence gating, if exact and every ε report are uncertified
and abstain reward strictly exceeds inadmissibility penalty, abstain is a
strict global maximizer over all non-abstain report types. -/
theorem rlff_abstain_strict_global_maximizer_of_no_certificates
    {X : Type u} {Y : Type v} {W : Type w}
    (R : Set (X × Y)) (Rε : EpsilonRelation X Y)
    (Γ : Regime X) (Q : CertifyingSolver X Y W)
    (w : RLFFWeights)
    (hNoExact : ¬ Nonempty (EvidenceForReport R Rε Γ Q ClaimReport.exact))
    (hNoEpsilon : ∀ ε : ℝ, ¬ Nonempty (EvidenceForReport R Rε Γ Q (ClaimReport.epsilon ε)))
    (hAbstainBeatsPenalty : -w.inadmissiblePenalty < w.abstainReward) :
    ∀ r : ClaimReport, r ≠ ClaimReport.abstain →
      rlffReward R Rε Γ Q w ClaimReport.abstain > rlffReward R Rε Γ Q w r := by
  intro r hr
  cases r with
  | abstain =>
      cases (hr rfl)
  | exact =>
      have hA : rlffReward R Rε Γ Q w ClaimReport.abstain = w.abstainReward :=
        rlffReward_abstain (R := R) (Rε := Rε) (Γ := Γ) (Q := Q) (w := w)
      have hE : rlffReward R Rε Γ Q w ClaimReport.exact = -w.inadmissiblePenalty :=
        rlffReward_of_no_evidence
          (R := R) (Rε := Rε) (Γ := Γ) (Q := Q) (w := w)
          (r := ClaimReport.exact) hNoExact
      calc
        rlffReward R Rε Γ Q w ClaimReport.abstain
            = w.abstainReward := hA
        _ > -w.inadmissiblePenalty := hAbstainBeatsPenalty
        _ = rlffReward R Rε Γ Q w ClaimReport.exact := by
              symm
              exact hE
  | epsilon ε =>
      have hA : rlffReward R Rε Γ Q w ClaimReport.abstain = w.abstainReward :=
        rlffReward_abstain (R := R) (Rε := Rε) (Γ := Γ) (Q := Q) (w := w)
      have hEps :
          rlffReward R Rε Γ Q w (ClaimReport.epsilon ε) = -w.inadmissiblePenalty :=
        rlffReward_of_no_evidence
          (R := R) (Rε := Rε) (Γ := Γ) (Q := Q) (w := w)
          (r := ClaimReport.epsilon ε) (hNoEpsilon ε)
      calc
        rlffReward R Rε Γ Q w ClaimReport.abstain
            = w.abstainReward := hA
        _ > -w.inadmissiblePenalty := hAbstainBeatsPenalty
        _ = rlffReward R Rε Γ Q w (ClaimReport.epsilon ε) := by
              symm
              exact hEps

/-! ## 4) Certified bits induce physical work lower bounds -/

/-- Physical work lower bound from report-bit accounting:
certified total bits dominate raw bits, therefore corresponding work (at
positive per-bit cost) also dominates raw work. -/
theorem certified_work_lower_bound_raw
    {X : Type u} {Y : Type v} {W : Type w}
    (R : Set (X × Y)) (Rε : EpsilonRelation X Y)
    (Γ : Regime X) (Q : CertifyingSolver X Y W)
    (M : ReportBitModel) (r : ClaimReport)
    (c : PhysicalComputer) :
    M.rawBits r * bit_energy_cost c ≤
      certifiedTotalBits R Rε Γ Q M r * bit_energy_cost c := by
  have hBits :
      M.rawBits r ≤ certifiedTotalBits R Rε Γ Q M r :=
    certifiedTotalBits_ge_raw
      (R := R) (Rε := Rε) (Γ := Γ) (Q := Q) (M := M) (r := r)
  have hMul :=
    Nat.mul_le_mul_right (bit_energy_cost c) hBits
  simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using hMul

/-- If a report is admissible and its certificate-bit allocation is positive,
its certified physical work is strictly greater than raw-only work. -/
theorem admissible_report_requires_strict_certified_work
    {X : Type u} {Y : Type v} {W : Type w}
    (R : Set (X × Y)) (Rε : EpsilonRelation X Y)
    (Γ : Regime X) (Q : CertifyingSolver X Y W)
    (M : ReportBitModel) (r : ClaimReport)
    (hPos : 0 < M.certBits r)
    (hAdm : ClaimAdmissible R Rε Γ Q r)
    (c : PhysicalComputer) :
    M.rawBits r * bit_energy_cost c <
      certifiedTotalBits R Rε Γ Q M r * bit_energy_cost c := by
  have hBits :
      M.rawBits r < certifiedTotalBits R Rε Γ Q M r :=
    report_admissible_implies_raw_lt_certifiedTotal
      (R := R) (Rε := Rε) (Γ := Γ) (Q := Q) (M := M) (r := r) hPos hAdm
  exact Nat.mul_lt_mul_of_pos_right hBits c.hbit_pos

/-! ## 5) Composition: finite budget + evidence-gated exact claims -/

/-- Physical feasibility predicate for a declared bit footprint. -/
def feasibleWork (c : PhysicalComputer) (bits : ℕ) : Prop :=
  bits * bit_energy_cost c ≤ c.E_max

/-- If required operation-energy exceeds budget at size `n`, any bit footprint
that upper-bounds the required operations is physically infeasible. -/
theorem infeasibleWork_of_required_ops_exceeds_budget
    (c : PhysicalComputer)
    (req : ComputationalRequirement)
    (n bits : ℕ)
    (hExceed : c.E_max < req.required_ops n * bit_energy_cost c)
    (hReqLeBits : req.required_ops n ≤ bits) :
    ¬ feasibleWork c bits := by
  intro hFeas
  have hReqLeMul : req.required_ops n * bit_energy_cost c ≤ bits * bit_energy_cost c := by
    exact Nat.mul_le_mul_right (bit_energy_cost c) hReqLeBits
  have hReqLeBudget : req.required_ops n * bit_energy_cost c ≤ c.E_max :=
    le_trans hReqLeMul hFeas
  exact (Nat.not_lt_of_ge hReqLeBudget) hExceed

/-- End-to-end composition theorem:
if exact admissibility requires at least `req.required_ops n` certified bits
and admissible exact reports must be physically feasible, then finite budget and
an exponential lower bound force eventual exact inadmissibility. -/
theorem eventual_no_exact_admissibility_under_budget
    {X : Type u} {Y : Type v} {W : Type w}
    (R : Set (X × Y)) (Rε : EpsilonRelation X Y)
    (Γ : Regime X) (Q : CertifyingSolver X Y W)
    (M : ReportBitModel)
    (c : PhysicalComputer)
    (req : ComputationalRequirement)
    (hExactReqBits :
      ∀ n : ℕ, ClaimAdmissible R Rε Γ Q ClaimReport.exact →
        req.required_ops n ≤ certifiedTotalBits R Rε Γ Q M ClaimReport.exact)
    (hExactFeasible :
      ClaimAdmissible R Rε Γ Q ClaimReport.exact →
        feasibleWork c (certifiedTotalBits R Rε Γ Q M ClaimReport.exact)) :
    ∃ n₀ : ℕ, ∀ n : ℕ, n ≥ n₀ →
      ¬ ClaimAdmissible R Rε Γ Q ClaimReport.exact := by
  obtain ⟨n₀, hThreshold⟩ :=
    finite_budget_blocks_exact_certification_beyond_threshold c req
  refine ⟨n₀, ?_⟩
  intro n hn hAdm
  have hExceed : c.E_max < req.required_ops n * bit_energy_cost c :=
    hThreshold n hn
  have hReqLeBits :
      req.required_ops n ≤ certifiedTotalBits R Rε Γ Q M ClaimReport.exact :=
    hExactReqBits n hAdm
  have hFeas :
      feasibleWork c (certifiedTotalBits R Rε Γ Q M ClaimReport.exact) :=
    hExactFeasible hAdm
  exact (infeasibleWork_of_required_ops_exceeds_budget
    (c := c) (req := req) (n := n)
    (bits := certifiedTotalBits R Rε Γ Q M ClaimReport.exact)
    hExceed hReqLeBits) hFeas

/-- Canonical specialization of the previous theorem to the module's
exponential requirement model. -/
theorem eventual_no_exact_admissibility_under_canonical_requirement
    {X : Type u} {Y : Type v} {W : Type w}
    (R : Set (X × Y)) (Rε : EpsilonRelation X Y)
    (Γ : Regime X) (Q : CertifyingSolver X Y W)
    (M : ReportBitModel)
    (c : PhysicalComputer)
    (hExactReqBits :
      ∀ n : ℕ, ClaimAdmissible R Rε Γ Q ClaimReport.exact →
        coNP_requirement.required_ops n ≤
          certifiedTotalBits R Rε Γ Q M ClaimReport.exact)
    (hExactFeasible :
      ClaimAdmissible R Rε Γ Q ClaimReport.exact →
        feasibleWork c (certifiedTotalBits R Rε Γ Q M ClaimReport.exact)) :
    ∃ n₀ : ℕ, ∀ n : ℕ, n ≥ n₀ →
      ¬ ClaimAdmissible R Rε Γ Q ClaimReport.exact := by
  exact eventual_no_exact_admissibility_under_budget
    (R := R) (Rε := Rε) (Γ := Γ) (Q := Q) (M := M)
    (c := c) (req := coNP_requirement) hExactReqBits hExactFeasible

end PhysicalCore
end Physics
end DecisionQuotient
