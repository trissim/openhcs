/- 
  Paper 4: Decision-Relevant Uncertainty

  FunctionalInformation.lean

  Two derivation paths for functional information:
  1) First-principles counting path (finite counting measure)
  2) Thermodynamic path (Landauer conversion)

  The bridge theorem shows both paths coincide on the same quantity.
-/

import DecisionQuotient.BayesFoundations
import DecisionQuotient.ThermodynamicLift
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Finset.Card
import Mathlib.Tactic

namespace DecisionQuotient
namespace FunctionalInformation

open DecisionQuotient
open DecisionQuotient.Foundations

/-- Functional success fraction under finite counting measure. -/
noncomputable def functionalFraction
    {Ω : Type*} [Fintype Ω] [DecidableEq Ω]
    (F : Finset Ω) : ℝ :=
  countingProb F

/-- Functional information in bits: `-log2(fraction)`.
    At zero fraction the value is pinned to `0` to keep a total definition. -/
noncomputable def functionalInformationBits
    {Ω : Type*} [Fintype Ω] [DecidableEq Ω]
    (F : Finset Ω) : ℝ :=
  if functionalFraction F = 0 then 0
  else -Real.log (functionalFraction F) / Real.log 2

/-- First-principles counting probability is nonnegative. -/
theorem functionalFraction_nonneg
    {Ω : Type*} [Fintype Ω] [DecidableEq Ω] [Nonempty Ω]
    (F : Finset Ω) :
    0 ≤ functionalFraction F := by
  simpa [functionalFraction] using counting_nonneg F

/-- First-principles counting probability is at most one. -/
theorem functionalFraction_le_one
    {Ω : Type*} [Fintype Ω] [DecidableEq Ω] [Nonempty Ω]
    (F : Finset Ω) :
    functionalFraction F ≤ 1 := by
  unfold functionalFraction countingProb
  have hCard : (F.card : ℝ) ≤ (Fintype.card Ω : ℝ) := by
    exact_mod_cast Finset.card_le_univ F
  have hDenomPos : 0 < (Fintype.card Ω : ℝ) := by
    exact Nat.cast_pos.mpr (Nat.pos_of_ne_zero Fintype.card_ne_zero)
  have hDivLe : (F.card : ℝ) / (Fintype.card Ω : ℝ) ≤ 1 := by
    rw [div_le_one hDenomPos]
    exact hCard
  simpa using hDivLe

/-- Under positive fraction, functional information is exactly the `-log2` form. -/
theorem functionalInformationBits_eq_log_inverse
    {Ω : Type*} [Fintype Ω] [DecidableEq Ω]
    (F : Finset Ω)
    (hPos : 0 < functionalFraction F) :
    functionalInformationBits F = -Real.log (functionalFraction F) / Real.log 2 := by
  unfold functionalInformationBits
  simp [hPos.ne']

/-- Functional information is nonnegative for fractions in `(0,1]`. -/
theorem functionalInformationBits_nonneg
    {Ω : Type*} [Fintype Ω] [DecidableEq Ω]
    (F : Finset Ω)
    (hPos : 0 < functionalFraction F)
    (hLeOne : functionalFraction F ≤ 1) :
    0 ≤ functionalInformationBits F := by
  rw [functionalInformationBits_eq_log_inverse F hPos]
  have hLogNonpos : Real.log (functionalFraction F) ≤ 0 := by
    exact Real.log_nonpos hPos.le hLeOne
  have hNumNonneg : 0 ≤ -Real.log (functionalFraction F) := by linarith
  have hLog2Pos : 0 < Real.log 2 := Real.log_pos (by norm_num : (1 : ℝ) < 2)
  exact div_nonneg hNumNonneg hLog2Pos.le

/-- Functional information is strictly positive for fractions in `(0,1)`. -/
theorem functionalInformationBits_pos_of_subunit
    {Ω : Type*} [Fintype Ω] [DecidableEq Ω]
    (F : Finset Ω)
    (hPos : 0 < functionalFraction F)
    (hLtOne : functionalFraction F < 1) :
    0 < functionalInformationBits F := by
  rw [functionalInformationBits_eq_log_inverse F hPos]
  have hLogNeg : Real.log (functionalFraction F) < 0 := Real.log_neg hPos hLtOne
  have hNumPos : 0 < -Real.log (functionalFraction F) := by linarith
  have hLog2Pos : 0 < Real.log 2 := Real.log_pos (by norm_num : (1 : ℝ) < 2)
  exact div_pos hNumPos hLog2Pos

/-- First-principles derivation in cardinal form. -/
theorem functional_information_from_counting
    {Ω : Type*} [Fintype Ω] [DecidableEq Ω] [Nonempty Ω]
    (F : Finset Ω)
    (hCardPos : 0 < F.card) :
    functionalInformationBits F =
      -Real.log ((F.card : ℝ) / (Fintype.card Ω : ℝ)) / Real.log 2 := by
  have hNumPos : 0 < (F.card : ℝ) := by exact_mod_cast hCardPos
  have hDenPos : 0 < (Fintype.card Ω : ℝ) := by
    exact Nat.cast_pos.mpr (Nat.pos_of_ne_zero Fintype.card_ne_zero)
  have hFracPos : 0 < functionalFraction F := by
    unfold functionalFraction countingProb
    exact div_pos hNumPos hDenPos
  rw [functionalInformationBits_eq_log_inverse F hFracPos]
  rfl

/-- Full-support function set has zero functional information. -/
theorem functionalInformationBits_univ_zero
    {Ω : Type*} [Fintype Ω] [DecidableEq Ω] [Nonempty Ω] :
    functionalInformationBits (Finset.univ : Finset Ω) = 0 := by
  have hTotal : functionalFraction (Finset.univ : Finset Ω) = 1 := by
    simpa [functionalFraction] using (counting_total (Ω := Ω))
  have hNonzero : functionalFraction (Finset.univ : Finset Ω) ≠ 0 := by
    simp [hTotal]
  unfold functionalInformationBits
  simp [hTotal]

/-- Thermodynamic path: minimum energy to realize the functional-information gain. -/
noncomputable def minimumFunctionalEnergy
    (kB T : ℝ)
    {Ω : Type*} [Fintype Ω] [DecidableEq Ω]
    (F : Finset Ω) : ℝ :=
  ThermodynamicLift.landauerJoulesPerBit kB T * functionalInformationBits F

/-- Recover functional information from energy using Landauer conversion. -/
noncomputable def functionalInformationBitsFromEnergy
    (kB T energy : ℝ) : ℝ :=
  energy / ThermodynamicLift.landauerJoulesPerBit kB T

/-- Nonnegativity of the thermodynamic lower bound from nonnegative FI. -/
theorem minimumFunctionalEnergy_nonneg
    (kB T : ℝ)
    {Ω : Type*} [Fintype Ω] [DecidableEq Ω]
    (F : Finset Ω)
    (hkB : 0 < kB) (hT : 0 < T)
    (hFI : 0 ≤ functionalInformationBits F) :
    0 ≤ minimumFunctionalEnergy kB T F := by
  unfold minimumFunctionalEnergy
  have hLandauerPos : 0 < ThermodynamicLift.landauerJoulesPerBit kB T :=
    ThermodynamicLift.landauerJoulesPerBit_pos hkB hT
  exact mul_nonneg hLandauerPos.le hFI

/-- Strict positivity of the thermodynamic lower bound from strictly positive FI. -/
theorem minimumFunctionalEnergy_pos
    (kB T : ℝ)
    {Ω : Type*} [Fintype Ω] [DecidableEq Ω]
    (F : Finset Ω)
    (hkB : 0 < kB) (hT : 0 < T)
    (hFI : 0 < functionalInformationBits F) :
    0 < minimumFunctionalEnergy kB T F := by
  unfold minimumFunctionalEnergy
  exact mul_pos (ThermodynamicLift.landauerJoulesPerBit_pos hkB hT) hFI

/-- Thermodynamic derivation path recovers the same FI exactly. -/
theorem functional_information_from_thermodynamics
    (kB T : ℝ)
    {Ω : Type*} [Fintype Ω] [DecidableEq Ω]
    (F : Finset Ω)
    (hkB : 0 < kB) (hT : 0 < T) :
    functionalInformationBitsFromEnergy kB T (minimumFunctionalEnergy kB T F) =
      functionalInformationBits F := by
  unfold functionalInformationBitsFromEnergy minimumFunctionalEnergy
  have hLandauerPos : 0 < ThermodynamicLift.landauerJoulesPerBit kB T :=
    ThermodynamicLift.landauerJoulesPerBit_pos hkB hT
  field_simp [hLandauerPos.ne']

/-- Bridge theorem: thermodynamic and first-principles paths coincide. -/
theorem first_principles_thermo_coincide
    (kB T : ℝ)
    {Ω : Type*} [Fintype Ω] [DecidableEq Ω] [Nonempty Ω]
    (F : Finset Ω)
    (hkB : 0 < kB) (hT : 0 < T)
    (hCardPos : 0 < F.card) :
    functionalInformationBitsFromEnergy kB T (minimumFunctionalEnergy kB T F) =
      -Real.log ((F.card : ℝ) / (Fintype.card Ω : ℝ)) / Real.log 2 := by
  rw [functional_information_from_thermodynamics kB T F hkB hT]
  exact functional_information_from_counting F hCardPos

end FunctionalInformation
end DecisionQuotient
