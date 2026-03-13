import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Tactic
import Paper1IT.GrowthTagBudget

namespace Ssot
namespace Paper1IT

/-- Finite empirical fiber summary extracted from a deployed representation audit. -/
def worstFiberCardFromList (fiberSizes : List Nat) : Nat :=
  fiberSizes.foldl Nat.max 0

/-- Total audited item count. -/
def totalItemsFromList (fiberSizes : List Nat) : Nat :=
  fiberSizes.sum

/-- Exact recoverable count under a uniform per-fiber tag alphabet of size `2^L`
for a uniform source over the audited items. -/
def exactRecoverableCount (fiberSizes : List Nat) (bits : Nat) : Nat :=
  (fiberSizes.map (fun a => min a (2 ^ bits))).sum

/-- Exact empirical error count under the same uniform per-fiber tag budget. -/
def exactUniformErrorCount (fiberSizes : List Nat) (bits : Nat) : Nat :=
  totalItemsFromList fiberSizes - exactRecoverableCount fiberSizes bits

/-- Total scaled mass of all audited items. The exported weighted lists are assumed to be sorted in
descending order by the external audit pipeline. -/
def totalScaledMassFromLists (fiberWeightsDescending : List (List Nat)) : Nat :=
  (fiberWeightsDescending.map List.sum).sum

/-- Exact recoverable scaled mass under a uniform per-fiber tag alphabet of size `2^L`, using the
exported descending fiber weights. -/
def exactWeightedRecoverableMass (fiberWeightsDescending : List (List Nat)) (bits : Nat) : Nat :=
  (fiberWeightsDescending.map (fun ws => (ws.take (2 ^ bits)).sum)).sum

/-- Exact weighted error mass in the scaled integer representation exported by the Python audit. -/
def exactWeightedErrorMass (fiberWeightsDescending : List (List Nat)) (bits : Nat) : Nat :=
  totalScaledMassFromLists fiberWeightsDescending - exactWeightedRecoverableMass fiberWeightsDescending bits

/-- Worst-fiber floor numerator for the deterministic bound `max(0, 1 - 2^L / A_pi)`.
Using naturals, this is the exact residual count `A_pi - min(A_pi, 2^L)`. -/
def worstFiberFloorNumerator (fiberSizes : List Nat) (bits : Nat) : Nat :=
  let a := worstFiberCardFromList fiberSizes
  a - min a (2 ^ bits)

structure BudgetCertificateRow where
  bits : Nat
  reportedTagAlphabet : Nat
  reportedWorstFiberFloorNumerator : Nat
  reportedWorstFiberFloorDenominator : Nat
  reportedExactUniformErrorNumerator : Nat
  reportedExactUniformErrorDenominator : Nat
  reportedExactWeightedErrorNumeratorScaled : Nat
  reportedExactWeightedErrorDenominatorScaled : Nat
  reportedZeroErrorFeasible : Bool
deriving Repr, DecidableEq

structure EmpiricalCertificate where
  fiberSizes : List Nat
  fiberWeightsDescending : List (List Nat)
  weightScale : Nat
  reportedMaxFiberCard : Nat
  reportedTotalItems : Nat
  reportedThresholdBits : Nat
  budgetRows : List BudgetCertificateRow
deriving Repr, DecidableEq

def ValidBudgetCertificateRow
    (fiberSizes : List Nat) (fiberWeightsDescending : List (List Nat))
    (row : BudgetCertificateRow) : Prop :=
  row.reportedTagAlphabet = 2 ^ row.bits ∧
  row.reportedWorstFiberFloorNumerator = worstFiberFloorNumerator fiberSizes row.bits ∧
  row.reportedWorstFiberFloorDenominator = worstFiberCardFromList fiberSizes ∧
  row.reportedExactUniformErrorNumerator = exactUniformErrorCount fiberSizes row.bits ∧
  row.reportedExactUniformErrorDenominator = totalItemsFromList fiberSizes ∧
  row.reportedExactWeightedErrorNumeratorScaled = exactWeightedErrorMass fiberWeightsDescending row.bits ∧
  row.reportedExactWeightedErrorDenominatorScaled = totalScaledMassFromLists fiberWeightsDescending ∧
  row.reportedZeroErrorFeasible = decide (worstFiberCardFromList fiberSizes ≤ 2 ^ row.bits)

instance instDecidableValidBudgetCertificateRow
    (fiberSizes : List Nat) (fiberWeightsDescending : List (List Nat))
    (row : BudgetCertificateRow) : Decidable (ValidBudgetCertificateRow fiberSizes fiberWeightsDescending row) := by
  unfold ValidBudgetCertificateRow
  infer_instance

def ValidEmpiricalCertificate (cert : EmpiricalCertificate) : Prop :=
  cert.fiberSizes = cert.fiberWeightsDescending.map List.length ∧
  cert.reportedMaxFiberCard = worstFiberCardFromList cert.fiberSizes ∧
  cert.reportedTotalItems = totalItemsFromList cert.fiberSizes ∧
  cert.reportedThresholdBits = requiredTagBits (worstFiberCardFromList cert.fiberSizes) ∧
  ∀ row ∈ cert.budgetRows, ValidBudgetCertificateRow cert.fiberSizes cert.fiberWeightsDescending row

instance instDecidableValidEmpiricalCertificate (cert : EmpiricalCertificate) :
    Decidable (ValidEmpiricalCertificate cert) := by
  unfold ValidEmpiricalCertificate
  infer_instance

theorem validCertificate_threshold_bits (cert : EmpiricalCertificate)
    (hvalid : ValidEmpiricalCertificate cert) :
    cert.reportedThresholdBits = requiredTagBits cert.reportedMaxFiberCard := by
  rcases hvalid with ⟨_hweights, hmax, _htotal, hbits, _rows⟩
  calc
    cert.reportedThresholdBits = requiredTagBits (worstFiberCardFromList cert.fiberSizes) := hbits
    _ = requiredTagBits cert.reportedMaxFiberCard := by rw [hmax]

theorem validCertificate_reportedMax_matches_worstFiber (cert : EmpiricalCertificate)
    (hvalid : ValidEmpiricalCertificate cert) :
    cert.reportedMaxFiberCard = worstFiberCardFromList cert.fiberSizes :=
  hvalid.2.1

theorem validBudgetRow_zeroErrorFeasible_iff
    (fiberSizes : List Nat) (fiberWeightsDescending : List (List Nat)) (row : BudgetCertificateRow)
    (hrow : ValidBudgetCertificateRow fiberSizes fiberWeightsDescending row) :
    row.reportedZeroErrorFeasible = true ↔ worstFiberCardFromList fiberSizes ≤ 2 ^ row.bits := by
  rcases hrow with ⟨_htag, _hnum, _hden, _hexactnum, _hexactden, _hwerr, _hwden, hfeas⟩
  rw [hfeas]
  by_cases h : worstFiberCardFromList fiberSizes ≤ 2 ^ row.bits
  · simp [h]
  · simp [h]

theorem validBudgetRow_exactUniformError_formula
    (fiberSizes : List Nat) (fiberWeightsDescending : List (List Nat)) (row : BudgetCertificateRow)
    (hrow : ValidBudgetCertificateRow fiberSizes fiberWeightsDescending row) :
    row.reportedExactUniformErrorNumerator =
      totalItemsFromList fiberSizes - exactRecoverableCount fiberSizes row.bits := by
  rcases hrow with ⟨_htag, _hnum, _hden, hexactnum, _hexactden, _hwerr, _hwden, _hfeas⟩
  exact hexactnum

theorem validBudgetRow_exactWeightedError_formula
    (fiberSizes : List Nat) (fiberWeightsDescending : List (List Nat)) (row : BudgetCertificateRow)
    (hrow : ValidBudgetCertificateRow fiberSizes fiberWeightsDescending row) :
    row.reportedExactWeightedErrorNumeratorScaled =
      totalScaledMassFromLists fiberWeightsDescending - exactWeightedRecoverableMass fiberWeightsDescending row.bits := by
  rcases hrow with ⟨_htag, _hnum, _hden, _hexactnum, _hexactden, hwerr, _hwden, _hfeas⟩
  exact hwerr

end Paper1IT
end Ssot
