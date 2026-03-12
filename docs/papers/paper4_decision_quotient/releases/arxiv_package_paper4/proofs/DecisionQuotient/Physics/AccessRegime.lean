import Mathlib.Data.Nat.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Tactic
import DecisionQuotient.HardnessDistribution

/-!
# Access Regimes for Physical Devices

## Dependency Graph
- **Nontrivial source:** HardnessDistribution (hardness conservation, DOF)
- **This module:** Typeclass definitions for access regimes. All "theorems" here
  are definitional trivialities. The five-way meet is just bundling definitions.

## Role
This is a type/interface definition module. The nontrivial hardness results
are in HardnessDistribution and ClaimClosure.

## Triviality Level
TRIVIAL — typeclass and elementary lemmas about regimes. Closest nontrivial
results: `HardnessDistribution.hardness_conservation` (DecisionQuotient.HardnessDistribution).
-/

namespace PhysicalComplexity.AccessRegime

set_option linter.dupNamespace false

/-! ## Part I: Physical Devices -/

/-- Physical decision device abstraction with an optional coordinate model. -/
structure PhysicalDevice where
  param_size : ℕ
  explicit_size : ℕ
  input_dim : ℕ := 0
  coord_card : Fin input_dim → ℕ := fun _ => 1

/-- Derived explicit table size from coordinate cardinalities. -/
def PhysicalDevice.derived_explicit_size (d : PhysicalDevice) : ℕ :=
  ∏ i : Fin d.input_dim, d.coord_card i

/-- Succinctness predicate used by regime-side statements. -/
def PhysicalDevice.is_succinct (d : PhysicalDevice) : Prop :=
  d.param_size < d.explicit_size

/-- If the declared size gap holds, the device is succinct. -/
theorem PhysicalDevice.is_succinct_of_gap (d : PhysicalDevice)
    (h : d.param_size < d.explicit_size) :
    d.is_succinct := h

/-! ## Part II: Access Regimes -/

/-- Access regime with query/answer interface and optional cost/round metadata. -/
structure AccessRegime where
  Query : Type
  Answer : Type
  cost_bound : ℕ → Prop := fun _ => True
  rounds : ℕ := 1

/-- Compatibility aliases for prose that uses query/answer terminology. -/
def AccessRegime.query_space (R : AccessRegime) : Type := R.Query
def AccessRegime.answer_space (R : AccessRegime) : Type := R.Answer

/-- Canonical regimes used by paper prose. -/
def RegimeEval : AccessRegime :=
  { Query := Unit, Answer := Bool }

def RegimeSample (α : Type) : AccessRegime :=
  { Query := Unit, Answer := α }

def RegimeProof (P : Type) : AccessRegime :=
  { Query := Unit, Answer := P }

def RegimeWithCertificate (E : Type) : AccessRegime :=
  { Query := Unit, Answer := E }

/-- Size-parameterized variants used when an explicit input dimension is declared. -/
def RegimeEvalOn (n : ℕ) : AccessRegime :=
  { Query := Fin n → Bool, Answer := Bool }

def RegimeSampleOn (n : ℕ) (α : ℕ → Type) : AccessRegime :=
  { Query := Unit, Answer := α n }

def RegimeProofOn (n : ℕ) (P : ℕ → Type) : AccessRegime :=
  { Query := Fin n → Bool, Answer := P n }

def RegimeWithCertificateOn (n : ℕ) (E : ℕ → Type) : AccessRegime :=
  { Query := Unit, Answer := E n }

/-! ## Part III: Regime Preorder -/

/-- Regime preorder by simulation of queries. -/
def AccessRegime.le (R1 R2 : AccessRegime) : Prop :=
  Nonempty (R1.Query → R2.Query)

theorem AccessRegime.le_refl (R : AccessRegime) :
    AccessRegime.le R R := by
  exact ⟨id⟩

theorem AccessRegime.le_trans (R1 R2 R3 : AccessRegime)
    (h12 : AccessRegime.le R1 R2)
    (h23 : AccessRegime.le R2 R3) :
    AccessRegime.le R1 R3 := by
  rcases h12 with ⟨e12⟩
  rcases h23 with ⟨e23⟩
  exact ⟨fun q => e23 (e12 q)⟩

/-! ## Part IV: Paper-Facing Predicates -/

/-- Local helper proposition for regime-hardness statements. -/
def HardUnderEval (d : PhysicalDevice) : Prop := d.is_succinct

/-- Auditable means the eval regime is simulable by certificate access. -/
def AuditableWithCertificate (E : Type) : Prop :=
  AccessRegime.le RegimeEval (RegimeWithCertificate E)

/-! ## Part V: Core Theorems Used by Paper Metadata -/

/-- Certificate access is an interface upgrade. -/
theorem certificate_upgrades_regime (R : AccessRegime) :
    AccessRegime.le R (RegimeWithCertificate Nat) := by
  exact ⟨fun _ => ()⟩

/-- Dimension-aware certificate upgrade theorem. -/
theorem certificate_upgrades_regime_on (n : ℕ) :
    AccessRegime.le (RegimeEvalOn n) (RegimeWithCertificateOn n (fun _ => Nat)) := by
  exact ⟨fun _ => ()⟩

/-- Succinct devices satisfy the regime-hardness premise by definition. -/
theorem physical_succinct_certification_hard
    (d : PhysicalDevice)
    (h_succinct : d.is_succinct)
    (_R : AccessRegime) :
    HardUnderEval d := by
  simpa [HardUnderEval] using h_succinct

/-- Certificate-augmented interface is auditable. -/
theorem certificate_amortizes_hardness
    (_d : PhysicalDevice)
    (E : Type) :
    AuditableWithCertificate E := by
  exact ⟨fun _ => ()⟩

/-- Named upgrade theorem used by prose/handle mapping. -/
theorem regime_upgrade_with_certificate
    (d : PhysicalDevice) :
    AccessRegime.le RegimeEval (RegimeWithCertificate Nat) := by
  exact certificate_amortizes_hardness d Nat

/-- Dimension-aware regime upgrade used by access-slice prose. -/
theorem regime_upgrade_with_certificate_on (n : ℕ) :
    AccessRegime.le (RegimeEvalOn n) (RegimeWithCertificateOn n (fun _ => Nat)) := by
  exact certificate_upgrades_regime_on n

/-! ## Part VI: Access-Channel Law and Five-Way Meet -/

/-- Access-channel law as an explicit assumption-composition theorem. -/
theorem AccessChannelLaw
    (D : PhysicalDevice)
    (R_eval : AccessRegime)
    (_hR : R_eval = RegimeEval)
    (E : Type)
    (hHard : HardUnderEval D)
    (hAudit : AuditableWithCertificate E) :
    HardUnderEval D ∧ AuditableWithCertificate E := by
  subst _hR
  exact ⟨hHard, hAudit⟩

/-- Five-way meet summary theorem in assumption-explicit form. -/
theorem FiveWayMeet
    (D : PhysicalDevice)
    (R : AccessRegime)
    (hPhys : D.is_succinct)
    (hHard : HardUnderEval D)
    (hAudit : AuditableWithCertificate Nat)
    (hUpgrade : AccessRegime.le R (RegimeWithCertificate Nat)) :
    D.is_succinct ∧ HardUnderEval D ∧ AuditableWithCertificate Nat ∧
      AccessRegime.le R (RegimeWithCertificate Nat) := by
  exact ⟨hPhys, hHard, hAudit, hUpgrade⟩

end PhysicalComplexity.AccessRegime
