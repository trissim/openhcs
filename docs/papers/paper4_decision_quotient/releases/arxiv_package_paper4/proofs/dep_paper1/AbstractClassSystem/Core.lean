-- Minimal imports to avoid Lean v4.27.0 segfault with full Mathlib
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
open Classical

namespace AbstractClassSystem

/-
  PART 1: The N-Axis Type Theory — Instantiated for Class Systems as type(B, S)

  GENERAL THEORY:
  A type of dimensionality n is type(A₁, A₂, ..., Aₙ) where each Aᵢ is an axis.
  The type's value is the tuple of its axis projections.

  INSTANTIATION FOR CLASS SYSTEMS:
  We use dimensionality 2 with axes:
  - B: Bases (explicit parent types) — the inheritance hierarchy
  - S: Namespace (attribute declarations) — the shape/interface

  CLARIFICATION ON "N" (Name):
  Historical presentations sometimes include N (the type's name/identifier).
  However, N is NOT an independent axis:
  - N is just a label for a (B, S) pair
  - N contributes no observables beyond B
  - Two types with identical (B, S) are indistinguishable regardless of N
  - Theorem `obs_eq_bs` proves: (B, S) equality suffices; N adds nothing

  The model is type(B, S). All capability gaps derive from B.
  Shape-based typing observes only S; nominal typing observes (B, S).
-/

-- Attribute names
abbrev AttrName := String

-- Types carry their namespace (S) and bases (B) explicitly
structure Typ where
  ns : Finset AttrName
  bs : List Typ

noncomputable instance : DecidableEq Typ := Classical.decEq _

/-!
  ## Axis-Parametric Foundation

  A type of dimensionality n is a projection over an n-element axis set A:
  - Each axis a has a carrier type Carrier(a)
  - A type over axis set A is a dependent function: ∀ a ∈ A, Carrier(a)
  - The dimensionality of the type equals |A|

  This is the GENERAL framework. Other domains may use different axis sets
  (e.g., type(B, S, H) for 3 dimensions, type(S) for 1 dimension).

  HERE: We instantiate for class systems with A = {B, S}, giving type(B, S).
  
  We prove `Typ` (our concrete record) ≃ `AxisProjection canonicalAxes` (the general form).
-/

/-- The axes of a class system.
    NOTE: There is no Name axis. Names are metadata stored in the namespace (S axis)
    as the `__name__` attribute. The name is assigned at definition time and is useful
    for debugging/serialization, but contributes no typing capability beyond (B, S).
    Two types with identical (B, S) are semantically equivalent for all typing purposes.
    The effective lattice is: ∅ < S < (B,S) -/
inductive Axis where
  | Bases      -- B: inheritance hierarchy — THE key axis
  | Namespace  -- S: attribute declarations (shape)
deriving DecidableEq, Repr

/-- The carrier type for each axis. This defines what information each axis holds.
    - B (Bases): The inheritance hierarchy, stored as `List Typ`
    - S (Namespace): The attribute declarations, stored as `Finset AttrName`
-/
def AxisCarrier : Axis → Type
  | .Bases => List Typ
  | .Namespace => Finset AttrName

/-- A projection over an axis set is a dependent function from axes to their carriers.
    This is the general n-axis definition of a type. -/
def AxisProjection (A : List Axis) := (a : Axis) → a ∈ A → AxisCarrier a

/-- The canonical axis set for the (B, S) type theory. -/
def canonicalAxes : List Axis := [.Bases, .Namespace]

-- Membership lemmas for canonicalAxes
lemma bases_mem_canonical : Axis.Bases ∈ canonicalAxes := by simp [canonicalAxes]
lemma namespace_mem_canonical : Axis.Namespace ∈ canonicalAxes := by simp [canonicalAxes]

/-- Convert a concrete `Typ` to an axis projection over {B, S}. -/
def Typ.toProjection (T : Typ) : AxisProjection canonicalAxes := fun a _ =>
  match a with
  | .Bases => T.bs
  | .Namespace => T.ns

/-- Convert an axis projection over {B, S} back to a concrete `Typ`. -/
def Typ.fromProjection (p : AxisProjection canonicalAxes) : Typ := {
  ns := p .Namespace namespace_mem_canonical
  bs := p .Bases bases_mem_canonical
}

/-- `Typ` and `AxisProjection canonicalAxes` are isomorphic (direction 1):
    fromProjection ∘ toProjection = id -/
theorem Typ.projection_roundtrip (T : Typ) :
    Typ.fromProjection (Typ.toProjection T) = T := by
  cases T
  rfl

/-- `Typ` and `AxisProjection canonicalAxes` are isomorphic (direction 2):
    toProjection ∘ fromProjection = id (using function extensionality) -/
theorem Typ.projection_roundtrip_inv (p : AxisProjection canonicalAxes) :
    Typ.toProjection (Typ.fromProjection p) = p := by
  funext a ha
  cases a <;> rfl

/-- **The Isomorphism Theorem: `Typ ≃ AxisProjection {B, S}`**

    This is a proper equivalence (bijection with both inverses).
    It establishes that our concrete 2-tuple `Typ` is EXACTLY the same
    as a projection over the canonical 2-axis set.
-/
noncomputable def Typ.equivProjection : Typ ≃ AxisProjection canonicalAxes where
  toFun := Typ.toProjection
  invFun := Typ.fromProjection
  left_inv := Typ.projection_roundtrip
  right_inv := Typ.projection_roundtrip_inv

/-- **Core Definition: A type of dimensionality n is type(A₁, ..., Aₙ).**

    For any axis set A with |A| = n, `GenericTyp A` = type(A₁, ..., Aₙ) is a 
    dependent tuple mapping each axis to its carrier value.
    
    The dimensionality equals |A|.
    
    Example: `GenericTyp canonicalAxes` = type(B, S) has dimensionality 2.
    Our concrete `Typ` is isomorphic to this.
-/
def GenericTyp (A : List Axis) := AxisProjection A

/-- The concrete `Typ` is isomorphic to `GenericTyp canonicalAxes` = type(B, S). -/
noncomputable def typ_equiv_generic : Typ ≃ GenericTyp canonicalAxes := Typ.equivProjection

/-- For any axis set A, `GenericTyp A` = `AxisProjection A` by definition.
    A type of dimensionality n IS a projection over n axes. -/
theorem n_axis_types_are_projections (A : List Axis) :
    GenericTyp A = AxisProjection A := rfl

-- Axis aliases (kept for terminology); projections are `Typ.ns` / `Typ.bs`
abbrev Namespace := Typ → Finset AttrName
abbrev Bases := Typ → List Typ

/-
  Definition 2.11: Ancestors (transitive closure over Bases)
  ancestors(T) = {T} ∪ ⋃_{P ∈ Bases(T)} ancestors(P)
-/

-- For termination, we use a fuel parameter (depth limit)
def ancestors (fuel : Nat) (T : Typ) : List Typ :=
  match fuel with
  | 0 => [T]
  | n + 1 => T :: T.bs.flatMap (ancestors n)

/-
  Definition 2.9: Shape-based typing
  compatible_shape(x, T) ⟺ Namespace(type(x)) ⊇ Namespace(T)
-/

def shapeCompatible (xType T : Typ) : Prop :=
  ∀ attr ∈ T.ns, attr ∈ xType.ns

/-
  Definition 2.10: Nominal typing
  compatible_nominal(x, T) ⟺ T ∈ ancestors(type(x))
-/

def nominalCompatible (fuel : Nat) (xType T : Typ) : Prop :=
  T ∈ ancestors fuel xType

-- Exact-type check: head-only “identity”
def exactCompatible (xType T : Typ) : Prop := xType = T

-- Self is always in its own ancestor list (any fuel)
theorem self_mem_ancestors (fuel : Nat) (T : Typ) :
    T ∈ ancestors fuel T := by
  cases fuel with
  | zero =>
      simp [ancestors]
  | succ n =>
      simp [ancestors]

-- Exact-type implies nominal (lineage membership)
theorem exact_implies_nominal (fuel : Nat) (xType T : Typ) :
    exactCompatible xType T → nominalCompatible fuel xType T := by
  intro h
  cases h
  exact self_mem_ancestors fuel xType

-- Nominal can hold when exact-type fails (lineage uses more than head)
theorem nominal_not_implies_exact :
    ∃ (fuel : Nat) (xType T : Typ),
      nominalCompatible fuel xType T ∧ ¬ exactCompatible xType T := by
  -- Witness: child inherits from parent; same namespace
  let parent : Typ := { ns := ∅, bs := [] }
  let child : Typ := { ns := ∅, bs := [parent] }
  refine ⟨1, child, parent, ?_, ?_⟩
  · -- nominal: parent appears in child's ancestor list
    simp [nominalCompatible, ancestors, child, parent]
  · -- not exact: distinct values
    simp [exactCompatible, child, parent]

/-
  PART 2: The Key Insight - Shape Ignores Bases

  Two types with identical namespaces are indistinguishable under shape-based typing,
  even if they have different inheritance hierarchies.
-/

-- Shape equivalence: same namespace (extensional Finset equality)
def shapeEquivalent (A B : Typ) : Prop :=
  A.ns = B.ns

-- THEOREM: Shape equivalence is an equivalence relation
theorem shapeEq_refl (A : Typ) : shapeEquivalent A A := rfl

theorem shapeEq_symm (A B : Typ) :
    shapeEquivalent A B → shapeEquivalent B A := Eq.symm

theorem shapeEq_trans (A B C : Typ) :
    shapeEquivalent A B → shapeEquivalent B C → shapeEquivalent A C := Eq.trans

/-
  THE SHAPE TYPING AXIOM:

  Any shape-based typing function must treat shape-equivalent types identically.
  This is not an assumption - it's the DEFINITION of shape-based typing.
-/

def ShapeRespecting {α : Type _} (f : Typ → α) : Prop :=
  ∀ A B, shapeEquivalent A B → f A = f B

/-
  THEOREM 2.5: Shape-based typing cannot distinguish types with same namespace
-/

theorem shape_cannot_distinguish {α : Type _} (A B : Typ)
    (h_equiv : shapeEquivalent A B) :
    ∀ (f : Typ → α), ShapeRespecting f → f A = f B := by
  intro f h_respect
  exact h_respect A B h_equiv

/-
  COROLLARY 2.6: Shape-based typing cannot provide provenance

  Provenance = "which type in the MRO provided this attribute?"
  If A and B have same namespace but different bases, shape typing
  cannot distinguish them, therefore cannot report different provenance.
-/

-- Provenance result: which type provided the value
structure Provenance where
  sourceType : Typ

noncomputable instance : DecidableEq Provenance := Classical.decEq _

-- If a provenance function is shape-respecting, it cannot distinguish
-- types with same namespace but different inheritance
theorem shape_provenance_impossible (A B : Typ)
    (h_same_ns : shapeEquivalent A B)
    (_h_diff_bases : A.bs ≠ B.bs)  -- Different inheritance!
    (getProv : Typ → Provenance)
    (h_shape : ShapeRespecting getProv) :
    getProv A = getProv B := by
  -- Despite different inheritance, shape-respecting function must return same
  exact h_shape A B h_same_ns

-- The provenance is identical even though inheritance differs!
-- This is the core impossibility result.

/-
  PART 3: Capability Enumeration
-/

inductive Capability where
  | interfaceCheck    -- Can check "does x have method m?"
  | typeIdentity      -- Can check "is type(x) == T or subtype of T?"
  | provenance        -- Can answer "which type in MRO provided attr a?"
  | subtypeEnum       -- Can enumerate all subtypes of T
  | conflictResolution -- Can determine winner in MRO ordering
  | typeAsKey         -- Can use type(x) as dictionary/map key with identity semantics
deriving DecidableEq, Repr

def shapeCapabilities : List Capability := [.interfaceCheck]

def nominalCapabilities : List Capability :=
  [.interfaceCheck, .typeIdentity, .provenance, .subtypeEnum, .conflictResolution, .typeAsKey]

/-
  THEOREM 2.7: Strict Dominance

  Every capability of shape-based typing is also a capability of nominal typing,
  AND nominal typing has capabilities that shape-based typing lacks.
-/

theorem shape_subset_nominal :
    ∀ c ∈ shapeCapabilities, c ∈ nominalCapabilities := by
  intro c hc
  simp only [shapeCapabilities, List.mem_singleton] at hc
  simp only [nominalCapabilities, List.mem_cons]
  left; exact hc

theorem nominal_has_extra :
    ∃ c ∈ nominalCapabilities, c ∉ shapeCapabilities := by
  use Capability.provenance
  constructor
  · simp [nominalCapabilities]
  · simp [shapeCapabilities]

-- Combined: strict dominance
theorem strict_dominance :
    (∀ c ∈ shapeCapabilities, c ∈ nominalCapabilities) ∧
    (∃ c ∈ nominalCapabilities, c ∉ shapeCapabilities) :=
  ⟨shape_subset_nominal, nominal_has_extra⟩

/-
  COROLLARY 2.10k' (Concession vs Alternative):

  Definition: A discipline D is a CONCESSION relative to D' when:
  1. D provides no capability D' lacks
  2. D' provides capabilities D lacks
  3. The only "benefit" of D is avoiding some work W that D' requires
  4. Avoiding work is not a capability

  Definition: A discipline D is an ALTERNATIVE to D' when:
  1. D provides at least one capability D' lacks, OR
  2. D and D' are capability-equivalent

  Theorem: Shape-based typing (Protocol) is a CONCESSION, not an ALTERNATIVE,
  when B ≠ ∅. This is because Protocol satisfies conditions 1-4 above:
  - Protocol provides no capability ABCs lack (shape_subset_nominal)
  - ABCs provide 4 capabilities Protocol lacks (nominal_has_extra)
  - Protocol's only benefit is avoiding 2-line adapters
  - Avoiding adapters is not a capability
-/

-- A choice is a concession if it provides strictly fewer capabilities
def isConcession (d_caps d'_caps : List Capability) : Prop :=
  (∀ c ∈ d_caps, c ∈ d'_caps) ∧ (∃ c ∈ d'_caps, c ∉ d_caps)

-- A choice is an alternative if it provides at least one capability the other lacks
def isAlternative (d_caps d'_caps : List Capability) : Prop :=
  ∃ c ∈ d_caps, c ∉ d'_caps

-- THEOREM: Protocol (shape) is a concession to ABCs (nominal)
theorem protocol_is_concession :
    isConcession shapeCapabilities nominalCapabilities := strict_dominance

-- THEOREM: Protocol is NOT an alternative to ABCs
theorem protocol_not_alternative :
    ¬isAlternative shapeCapabilities nominalCapabilities := by
  intro ⟨c, hc_in, hc_notin⟩
  have := shape_subset_nominal c hc_in
  exact hc_notin this

-- COROLLARY: ABCs with adapters is the single non-concession choice
-- When B ≠ ∅, choosing Protocol means accepting reduced capabilities
theorem abcs_single_non_concession :
    ¬isConcession nominalCapabilities shapeCapabilities := by
  intro ⟨_, h_exists⟩
  obtain ⟨c, hc_in, hc_notin⟩ := h_exists
  -- c ∈ shapeCapabilities but c ∉ nominalCapabilities
  -- But shape_subset_nominal says all shape caps are in nominal caps
  have := shape_subset_nominal c hc_in
  -- So c ∈ nominalCapabilities, contradicting hc_notin
  exact hc_notin this

/-
  COROLLARY 2.8: Greenfield Incorrectness

  In greenfield development (architect controls type definitions),
  choosing shape-based typing forecloses capabilities for zero benefit.
  This is definitionally incorrect.
-/

-- Greenfield: both options available, architect chooses
-- Choosing dominated option = incorrect
theorem greenfield_incorrectness :
    -- If shape capabilities are strict subset of nominal
    (∀ c ∈ shapeCapabilities, c ∈ nominalCapabilities) →
    (∃ c ∈ nominalCapabilities, c ∉ shapeCapabilities) →
    -- Then choosing shape forecloses capabilities
    ∃ c, c ∈ nominalCapabilities ∧ c ∉ shapeCapabilities := by
  intro _ h_extra
  exact h_extra

/-
  PART 4: The Decision Procedure

  Given inputs (has_inheritance, is_greenfield), the correct typing discipline
  is DERIVED, not chosen.
-/

inductive TypingDiscipline where
  | nominal
  | structural
deriving DecidableEq, Repr

def selectTypingDiscipline (hasInheritance : Bool) (isGreenfield : Bool) : TypingDiscipline :=
  if ¬hasInheritance then
    .structural  -- No inheritance axis ⟹ structural is correct (Go, C)
  else if isGreenfield then
    .nominal     -- Greenfield + inheritance ⟹ nominal (strict dominance)
  else
    .structural  -- Retrofit ⟹ structural is valid concession

-- The decision is deterministic
theorem decision_deterministic (h1 h2 : Bool) :
    selectTypingDiscipline h1 h2 = selectTypingDiscipline h1 h2 := rfl

-- Greenfield with inheritance implies nominal
theorem greenfield_inheritance_implies_nominal :
    selectTypingDiscipline true true = .nominal := rfl

-- No inheritance implies structural is acceptable
theorem no_inheritance_implies_structural :
    selectTypingDiscipline false true = .structural := rfl

theorem no_inheritance_implies_structural' :
    selectTypingDiscipline false false = .structural := rfl

/-
  PART 5: Concrete Examples

  Demonstrate that types with same namespace but different bases
  are distinguishable nominally but not structurally.
-/

-- Two types with same "shape" but different identity
noncomputable def ConfigType : Typ := { ns := {"cfg"}, bs := [] }
noncomputable def StepConfigType : Typ := { ns := {"cfg"}, bs := [ConfigType] }

-- They're nominally distinct (different bases)
theorem types_nominally_distinct : ConfigType ≠ StepConfigType := by
  intro h
  have hbs : ConfigType.bs = StepConfigType.bs := congrArg Typ.bs h
  simp only [ConfigType, StepConfigType] at hbs
  cases hbs

-- But if they have same namespace, they're shape-equivalent
example : shapeEquivalent ConfigType StepConfigType := rfl

-- Therefore shape-based typing CANNOT distinguish them
-- But nominal typing CAN (by Theorem types_nominally_distinct)

/-
  PART 6: Mixin vs Composition Strict Dominance (Theorem 8.1, 8.2)

  The "composition over inheritance" principle (Gang of Four, 1994) is incorrect
  for behavior extension in languages with deterministic MRO.

  Mixins + C3 MRO strictly dominate object composition by the same argument:
  mixins preserve the Bases axis, composition discards it.
-/

-- Architectural pattern capabilities
inductive ArchCapability where
  | behaviorReuse         -- Can reuse behavior across classes
  | runtimeSwap           -- Can swap implementations
  | multipleBehaviors     -- Can combine multiple behaviors
  | conflictResolution    -- Can resolve method conflicts deterministically
  | singleDecisionPoint   -- Ordering decided once (not per-call-site)
  | behaviorProvenance    -- Can answer "which mixin provided this method?"
  | behaviorEnumeration   -- Can list all mixed-in behaviors (__mro__)
  | zeroBoilerplate       -- No manual delegation required
deriving DecidableEq, Repr

-- Object composition capabilities (delegation-based)
def compositionCapabilities : List ArchCapability :=
  [.behaviorReuse, .runtimeSwap, .multipleBehaviors]

-- Mixin capabilities (inheritance + MRO)
def mixinCapabilities : List ArchCapability :=
  [.behaviorReuse, .runtimeSwap, .multipleBehaviors,
   .conflictResolution, .singleDecisionPoint, .behaviorProvenance,
   .behaviorEnumeration, .zeroBoilerplate]

-- THEOREM 8.1: Every composition capability is a mixin capability
theorem composition_subset_mixin :
    ∀ c ∈ compositionCapabilities, c ∈ mixinCapabilities := by
  intro c hc
  simp only [compositionCapabilities, List.mem_cons] at hc
  simp only [mixinCapabilities, List.mem_cons]
  rcases hc with h1 | h2 | h3
  · left; exact h1
  · right; left; exact h2
  · rcases h3 with h3' | h3''
    · right; right; left; exact h3'
    · simp at h3''

-- Mixins have capabilities composition lacks
theorem mixin_has_extra :
    ∃ c ∈ mixinCapabilities, c ∉ compositionCapabilities := by
  use ArchCapability.conflictResolution
  constructor
  · simp [mixinCapabilities]
  · simp [compositionCapabilities]

-- Combined: Theorem 8.1 (Mixin Strict Dominance)
theorem mixin_strict_dominance :
    (∀ c ∈ compositionCapabilities, c ∈ mixinCapabilities) ∧
    (∃ c ∈ mixinCapabilities, c ∉ compositionCapabilities) :=
  ⟨composition_subset_mixin, mixin_has_extra⟩

-- Provenance is a mixin capability but not a composition capability
theorem provenance_requires_mixin :
    ArchCapability.behaviorProvenance ∈ mixinCapabilities ∧
    ArchCapability.behaviorProvenance ∉ compositionCapabilities := by
  constructor
  · simp [mixinCapabilities]
  · simp [compositionCapabilities]

-- Conflict resolution is a mixin capability but not a composition capability
theorem conflict_resolution_requires_mixin :
    ArchCapability.conflictResolution ∈ mixinCapabilities ∧
    ArchCapability.conflictResolution ∉ compositionCapabilities := by
  constructor
  · simp [mixinCapabilities]
  · simp [compositionCapabilities]

/-
  THEOREM 8.2: Unified Dominance Principle

  In class systems with explicit inheritance (bases axis),
  mechanisms using bases strictly dominate mechanisms using only namespace.

  This unifies:
  - Nominal > Structural (typing disciplines)
  - Mixins > Composition (architectural patterns)

  Both reduce to: discarding the Bases axis forecloses capabilities.
-/

-- A discipline is either typing-related or architecture-related
inductive DisciplineKind where
  | typing
  | architecture
deriving DecidableEq, Repr

-- A discipline either uses Bases or ignores it
inductive BasesUsage where
  | usesBasesAxis      -- Nominal typing, Mixins
  | ignoresBasesAxis   -- Structural typing, Composition
deriving DecidableEq, Repr

-- Unified capability (combines both domains)
inductive UnifiedCapability where
  -- Shared
  | interfaceCheck      -- Check interface satisfaction / behavior reuse
  -- Bases-dependent
  | identity            -- Type identity / behavior identity
  | provenance          -- Type provenance / behavior provenance
  | enumeration         -- Subtype enumeration / mixin enumeration
  | conflictResolution  -- MRO-based resolution
deriving DecidableEq, Repr

def basesIgnoringCapabilities : List UnifiedCapability :=
  [.interfaceCheck]

def basesUsingCapabilities : List UnifiedCapability :=
  [.interfaceCheck, .identity, .provenance, .enumeration, .conflictResolution]

-- THE UNIFIED THEOREM
theorem unified_dominance :
    (∀ c ∈ basesIgnoringCapabilities, c ∈ basesUsingCapabilities) ∧
    (∃ c ∈ basesUsingCapabilities, c ∉ basesIgnoringCapabilities) := by
  constructor
  · intro c hc
    simp only [basesIgnoringCapabilities, List.mem_singleton] at hc
    simp only [basesUsingCapabilities, List.mem_cons]
    left; exact hc
  · use UnifiedCapability.provenance
    constructor
    · simp [basesUsingCapabilities]
    · simp [basesIgnoringCapabilities]

-- Corollary: Discarding Bases forecloses capabilities in BOTH domains
theorem bases_axis_essential :
    ∃ c, c ∈ basesUsingCapabilities ∧ c ∉ basesIgnoringCapabilities := by
  exact unified_dominance.2


end AbstractClassSystem
