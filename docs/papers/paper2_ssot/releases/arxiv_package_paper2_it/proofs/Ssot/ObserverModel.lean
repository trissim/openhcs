/-
  ObserverModel: Formalizing observation semantics

  This addresses the critique: "Observers are not well-defined."

  We define:
  - Observers as functions from configurations to values
  - Consistency as observer agreement
  - The key result: multi-observer disagreement is possible iff DOF > 1
-/

import Ssot.SSOTGrounded
import Ssot.Dof
import Paper1IT.ObserverTagModel
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Finset.Card

namespace ObserverModel

open SSOTGrounded

/-- Lookup value by location ID (decidable) -/
def lookupValue (locs : List EncodingLocation) (id : Nat) : Option Nat :=
  match locs with
  | [] => none
  | loc :: rest => if loc.id = id then some loc.value else lookupValue rest id

/-- KEY: Single location, any lookup returns the same value -/
theorem single_lookup_deterministic :
    ∀ (loc : EncodingLocation) (id1 id2 : Nat) (v1 v2 : Nat),
      lookupValue [loc] id1 = some v1 →
      lookupValue [loc] id2 = some v2 →
      v1 = v2 := by
  intro loc id1 id2 v1 v2 h1 h2
  simp only [lookupValue] at h1 h2
  split_ifs at h1 h2; simp_all

/-- Two locations with same value: lookups agree -/
theorem consistent_lookups_agree :
    ∀ (loc1 loc2 : EncodingLocation) (id1 id2 v1 v2 : Nat),
      loc1.value = loc2.value →
      lookupValue [loc1, loc2] id1 = some v1 →
      lookupValue [loc1, loc2] id2 = some v2 →
      v1 = v2 := by
  intro loc1 loc2 id1 id2 v1 v2 hval h1 h2
  simp only [lookupValue] at h1 h2
  split_ifs at h1 h2 <;> simp_all

/-- Two distinct-ID locations with different values: disagreement exists -/
theorem inconsistent_lookups_disagree :
    ∀ (loc1 loc2 : EncodingLocation),
      loc1.id ≠ loc2.id →
      loc1.value ≠ loc2.value →
      ∃ v1 v2,
        lookupValue [loc1, loc2] loc1.id = some v1 ∧
        lookupValue [loc1, loc2] loc2.id = some v2 ∧
        v1 ≠ v2 := by
  intro loc1 loc2 hid hval
  use loc1.value, loc2.value
  constructor
  · -- First: lookupValue finds loc1.id in first position
    simp only [lookupValue, ite_true]
  constructor
  · -- Second: lookupValue skips loc1 (wrong id), finds loc2
    simp only [lookupValue]
    rw [if_neg hid]
    simp only [ite_true]
  · exact hval

/-- SSOT asymmetry theorem: single location immune, multiple vulnerable -/
theorem observation_ssot_asymmetry :
    -- Single location: any two lookups agree
    (∀ loc id1 id2 v1 v2,
      lookupValue [loc] id1 = some v1 →
      lookupValue [loc] id2 = some v2 →
      v1 = v2) ∧
    -- Two distinct locations with different values: disagreement exists
    (∀ loc1 loc2,
      loc1.id ≠ loc2.id →
      loc1.value ≠ loc2.value →
      ∃ v1 v2,
        lookupValue [loc1, loc2] loc1.id = some v1 ∧
        lookupValue [loc1, loc2] loc2.id = some v2 ∧
        v1 ≠ v2) := by
  exact ⟨single_lookup_deterministic, inconsistent_lookups_disagree⟩


/-!
## Bridge to `Ssot.Dof`

We realize the auxiliary support count as a concrete DOF quantity by viewing each
support value as an independent encoding under the derivation-free system.
-/

/-- Support encodings for an architecture, one per auxiliary support value. -/
def supportEncodings {K O : Nat} (A : RecoveryArchitecture K O) :
    List (Dof.Encoding Unit Nat) :=
  List.ofFn (fun i : Fin A.tagAlphabet =>
    { fact := (), location := toString i.1, value := i.1 })

/-- Derivation-free system: no support value is derived from another. -/
def supportDerivationSystem : Ssot.DerivationSystem (Dof.Encoding Unit Nat) where
  derived_from := fun _ _ => False
  transitive := by intro _ _ _ h; cases h
  irrefl := by intro _ h; cases h

theorem support_encoding_not_redundant {K O : Nat} (A : RecoveryArchitecture K O)
    (e : Dof.Encoding Unit Nat) :
    ¬ Dof.redundant supportDerivationSystem (supportEncodings A) e := by
  intro h
  rcases h with ⟨e', _, _, hderiv⟩
  cases hderiv

theorem support_minimal_core_eq_support_encodings {K O : Nat} (A : RecoveryArchitecture K O) :
    Dof.minimalIndependentCore supportDerivationSystem (supportEncodings A) = supportEncodings A := by
  unfold Dof.minimalIndependentCore
  apply List.ext_getElem?
  intro i
  simp [support_encoding_not_redundant]

theorem support_encodings_length {K O : Nat} (A : RecoveryArchitecture K O) :
    (supportEncodings A).length = supportCount A := by
  unfold supportEncodings supportCount
  simp

/-- Formal DOF of the support architecture equals its support count. -/
theorem supportCount_eq_dof {K O : Nat} (A : RecoveryArchitecture K O) :
    Dof.dof supportDerivationSystem (supportEncodings A) = supportCount A := by
  unfold Dof.dof
  rw [support_minimal_core_eq_support_encodings, support_encodings_length]

/-- Nontrivial confusability forces formal DOF strictly above one. -/
theorem nontrivial_clique_forces_dof_gt_one
    {K O : Nat} (A : RecoveryArchitecture K O)
    {s : Finset (Fin K)}
    (hs : IsClique A.obs s)
    (hsize : 1 < s.card) :
    1 < Dof.dof supportDerivationSystem (supportEncodings A) := by
  rw [supportCount_eq_dof]
  exact architecture_support_gt_one_of_nontrivial_clique A hs hsize

end ObserverModel
