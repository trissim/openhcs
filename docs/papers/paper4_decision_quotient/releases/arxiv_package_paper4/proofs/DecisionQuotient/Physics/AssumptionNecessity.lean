import Mathlib.Data.Set.Basic
import Mathlib.Tactic

/-!
  Paper 4: Decision-Relevant Uncertainty

  AssumptionNecessity.lean

  Meta-level theorem schema:
  - physical consequences are assumption-sensitive;
  - refutable physical claims are not provable assumption-free;
  - when derived from physical assumptions, the support set is non-empty;
  - if those assumptions are empirically justified, the support is empirically justified.

  No new axioms are introduced.
-/

namespace DecisionQuotient
namespace Physics
namespace AssumptionNecessity

universe u v w

/-- Abstract formal system with explicit assumption semantics and soundness. -/
structure FormalSystem where
  Axiom : Type u
  Claim : Type v
  Model : Type w
  Provable : Set Axiom → Claim → Prop
  SatisfiesAxiom : Model → Axiom → Prop
  Holds : Model → Claim → Prop
  PhysicalAxiom : Axiom → Prop
  PhysicalClaim : Claim → Prop
  EmpiricallyJustified : Axiom → Prop
  sound :
    ∀ {Γ : Set Axiom} {φ : Claim},
      Provable Γ φ →
      ∀ m : Model, (∀ a : Axiom, a ∈ Γ → SatisfiesAxiom m a) → Holds m φ

/-- All assumptions in `Γ` are physical assumptions. -/
def PhysicalAssumptionSet (F : FormalSystem) (Γ : Set F.Axiom) : Prop :=
  ∀ a : F.Axiom, a ∈ Γ → F.PhysicalAxiom a

/-- All assumptions in `Γ` are empirically justified. -/
def EmpiricalDiscipline (F : FormalSystem) (Γ : Set F.Axiom) : Prop :=
  ∀ a : F.Axiom, a ∈ Γ → F.EmpiricallyJustified a

/-- Soundness + a countermodel imply no assumption-free derivation
for a refutable claim. -/
theorem no_assumption_free_proof_of_refutable_claim
    (F : FormalSystem)
    {φ : F.Claim}
    (hCounter : ∃ m : F.Model, ¬ F.Holds m φ) :
    ¬ F.Provable (∅ : Set F.Axiom) φ := by
  intro hProv
  rcases hCounter with ⟨m, hNot⟩
  have hHolds : F.Holds m φ :=
    F.sound hProv m (by
      intro a ha
      cases ha)
  exact hNot hHolds

/-- If `φ` is proved from `Γ` but falsified by some model, then that model
must violate at least one assumption from `Γ`. -/
theorem countermodel_violates_some_assumption
    (F : FormalSystem)
    {Γ : Set F.Axiom} {φ : F.Claim}
    (hProv : F.Provable Γ φ)
    (hCounter : ∃ m : F.Model, ¬ F.Holds m φ) :
    ∃ m : F.Model, ¬ F.Holds m φ ∧
      ∃ a : F.Axiom, a ∈ Γ ∧ ¬ F.SatisfiesAxiom m a := by
  rcases hCounter with ⟨m, hNot⟩
  have hNotAll : ¬ (∀ a : F.Axiom, a ∈ Γ → F.SatisfiesAxiom m a) := by
    intro hAll
    have hHolds : F.Holds m φ := F.sound hProv m hAll
    exact hNot hHolds
  have hWitness : ∃ a : F.Axiom, a ∈ Γ ∧ ¬ F.SatisfiesAxiom m a := by
    classical
    by_contra hNo
    apply hNotAll
    intro a ha
    by_contra hSat
    exact hNo ⟨a, ha, hSat⟩
  exact ⟨m, hNot, hWitness⟩

/-- Strong necessity form:
if a physical claim is provable from `Γ`, and a countermodel exists to that
claim, then `Γ` is non-empty and includes at least one physical axiom. -/
theorem physical_claim_requires_physical_assumption
    (F : FormalSystem)
    {Γ : Set F.Axiom} {φ : F.Claim}
    (_hPhysClaim : F.PhysicalClaim φ)
    (hPhysΓ : PhysicalAssumptionSet F Γ)
    (hProv : F.Provable Γ φ)
    (hCounter : ∃ m : F.Model, ¬ F.Holds m φ) :
    ∃ a : F.Axiom, a ∈ Γ ∧ F.PhysicalAxiom a := by
  rcases countermodel_violates_some_assumption F hProv hCounter with
    ⟨_m, _hNot, ⟨a, haΓ, _hNotSat⟩⟩
  exact ⟨a, haΓ, hPhysΓ a haΓ⟩

/-- Empirical-grounding extension:
if the supporting physical assumption set is empirically disciplined, then
at least one required supporting axiom is both physical and empirically justified. -/
theorem physical_claim_requires_empirically_justified_physical_assumption
    (F : FormalSystem)
    {Γ : Set F.Axiom} {φ : F.Claim}
    (_hPhysClaim : F.PhysicalClaim φ)
    (hPhysΓ : PhysicalAssumptionSet F Γ)
    (hEmpΓ : EmpiricalDiscipline F Γ)
    (hProv : F.Provable Γ φ)
    (hCounter : ∃ m : F.Model, ¬ F.Holds m φ) :
    ∃ a : F.Axiom, a ∈ Γ ∧ F.PhysicalAxiom a ∧ F.EmpiricallyJustified a := by
  rcases countermodel_violates_some_assumption F hProv hCounter with
    ⟨_m, _hNot, ⟨a, haΓ, _hNotSat⟩⟩
  exact ⟨a, haΓ, hPhysΓ a haΓ, hEmpΓ a haΓ⟩

/-- Bundled strong claim:
for any sound formal system, a refutable physical claim cannot be proved
assumption-free; if proved from a physical, empirically disciplined set `Γ`,
that set must provide physical and empirically justified support. -/
theorem strong_physical_no_go_meta
    (F : FormalSystem)
    {Γ : Set F.Axiom} {φ : F.Claim}
    (hPhysClaim : F.PhysicalClaim φ)
    (hPhysΓ : PhysicalAssumptionSet F Γ)
    (hEmpΓ : EmpiricalDiscipline F Γ)
    (hProv : F.Provable Γ φ)
    (hCounter : ∃ m : F.Model, ¬ F.Holds m φ) :
    (∃ a : F.Axiom, a ∈ Γ ∧ F.PhysicalAxiom a ∧ F.EmpiricallyJustified a) ∧
    ¬ F.Provable (∅ : Set F.Axiom) φ := by
  refine ⟨?_, ?_⟩
  · exact physical_claim_requires_empirically_justified_physical_assumption
      F hPhysClaim hPhysΓ hEmpΓ hProv hCounter
  · exact no_assumption_free_proof_of_refutable_claim F hCounter

end AssumptionNecessity
end Physics
end DecisionQuotient
