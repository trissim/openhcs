/-
  Discipline vs Migration Formalization

  Proves the distinction between:
  - Discipline optimality: abstract capability comparison (universal)
  - Migration optimality: practical cost-benefit decision (context-dependent)

  This distinction clarifies that capability dominance (Section 3) is separate
  from migration cost analysis (practical decision depends on codebase context).
-/

-- Capabilities that a typing discipline can provide
inductive Capability where
  | provenance : Capability        -- "which type provided this value?"
  | identity : Capability          -- "is this the same type?"
  | enumeration : Capability       -- "what are all subtypes?"
  | conflictResolution : Capability -- "which type wins in MRO?"
  | attributeCheck : Capability    -- "does this have attribute X?"
deriving DecidableEq, Repr

-- Typing disciplines
inductive Discipline where
  | Nominal : Discipline    -- Uses (B, S) axes
  | Shape : Discipline      -- Uses S axis only (structural/duck)
deriving DecidableEq, Repr

-- Capability sets for each discipline
def nominalCapabilities : List Capability :=
  [.provenance, .identity, .enumeration, .conflictResolution, .attributeCheck]

def shapeCapabilities : List Capability :=
  [.attributeCheck]

def capabilities (d : Discipline) : List Capability :=
  match d with
  | .Nominal => nominalCapabilities
  | .Shape => shapeCapabilities

-- Declaration cost (both require one class definition)
def declarationCost (d : Discipline) : Nat :=
  match d with
  | .Nominal => 1  -- One ABC definition
  | .Shape => 1    -- One Protocol definition

-- THEOREM: Shape capabilities are subset of nominal capabilities
theorem shape_subset_nominal :
    ∀ c ∈ shapeCapabilities, c ∈ nominalCapabilities := by
  intro c hc
  simp only [shapeCapabilities, List.mem_singleton] at hc
  subst hc
  decide

-- THEOREM: Nominal has capabilities that shape lacks
theorem nominal_has_provenance : Capability.provenance ∈ nominalCapabilities := by
  decide

theorem shape_lacks_provenance : Capability.provenance ∉ shapeCapabilities := by
  decide

-- THEOREM: Capability gap is exactly 4
theorem capability_gap :
    nominalCapabilities.length - shapeCapabilities.length = 4 := by
  native_decide

-- THEOREM: Equal declaration cost
theorem equal_declaration_cost :
    declarationCost .Nominal = declarationCost .Shape := by
  rfl

-- Pareto dominance: A dominates B if A has ≥ capabilities at ≤ cost
def paretoDominates (a b : Discipline) : Prop :=
  (∀ c ∈ capabilities b, c ∈ capabilities a) ∧
  declarationCost a ≤ declarationCost b

-- THEOREM: Nominal Pareto dominates Shape
theorem nominal_pareto_dominates_shape :
    paretoDominates .Nominal .Shape := by
  constructor
  · intro c hc
    simp [capabilities] at hc ⊢
    exact shape_subset_nominal c hc
  · simp [declarationCost]

-- ═══════════════════════════════════════════════════════════════════
-- MIGRATION CONTEXT (Practical, not universal)
-- ═══════════════════════════════════════════════════════════════════

-- Context for migration decisions
structure MigrationContext where
  codebaseSize : Nat           -- Lines of code
  externalDeps : Nat           -- Number of external dependencies
  teamFamiliarity : Nat        -- 0-10 scale
  deadlinePressure : Nat       -- 0-10 scale

-- Migration cost grows with codebase size and external deps
def migrationCost (ctx : MigrationContext) : Nat :=
  ctx.codebaseSize / 100 + ctx.externalDeps * 10

-- Capability benefit (number of additional capabilities)
def capabilityBenefit (from_d to_d : Discipline) : Nat :=
  (capabilities to_d).length - (capabilities from_d).length

-- Migration is beneficial when benefit exceeds cost
def migrationBeneficial (from_d to_d : Discipline) (ctx : MigrationContext) : Bool :=
  capabilityBenefit from_d to_d > migrationCost ctx

-- ═══════════════════════════════════════════════════════════════════
-- KEY INSIGHT: These are INDEPENDENT questions
-- ═══════════════════════════════════════════════════════════════════

-- The discipline comparison is universal (always true)
theorem discipline_comparison_universal :
    paretoDominates .Nominal .Shape :=
  nominal_pareto_dominates_shape

-- The migration decision is context-dependent
-- We show existence of both beneficial and non-beneficial contexts
def smallContext : MigrationContext := ⟨50, 0, 5, 5⟩
def largeContext : MigrationContext := ⟨50000, 20, 5, 5⟩

-- Small codebase: migration IS beneficial
theorem small_context_beneficial : migrationBeneficial .Shape .Nominal smallContext = true := by
  native_decide

-- Large codebase: migration is NOT beneficial
theorem large_context_not_beneficial : migrationBeneficial .Shape .Nominal largeContext = false := by
  native_decide

-- THEOREM: Pareto dominance does NOT imply migration is beneficial
-- We demonstrate with a concrete large codebase example
theorem dominance_not_implies_migration_example :
    paretoDominates .Nominal .Shape ∧
    migrationBeneficial .Shape .Nominal largeContext = false :=
  ⟨nominal_pareto_dominates_shape, large_context_not_beneficial⟩

-- Therefore: migration decision is context-dependent (not universal)
theorem migration_decision_contextual :
    ∃ ctx₁ ctx₂ : MigrationContext,
      migrationBeneficial .Shape .Nominal ctx₁ = true ∧
      migrationBeneficial .Shape .Nominal ctx₂ = false :=
  ⟨smallContext, largeContext, small_context_beneficial, large_context_not_beneficial⟩

