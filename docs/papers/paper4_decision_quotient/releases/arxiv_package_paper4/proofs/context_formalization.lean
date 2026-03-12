/-
  Context Formalization: Greenfield, Retrofit, and Requirement Detection

  Proves:
  1. Greenfield/Retrofit classification is decidable
  2. Provenance requirements are detectable from system queries
  3. Decision procedure is sound

  Addresses potential concerns:
  - Provides precise definitions for greenfield/retrofit distinction
  - Eliminates circularity by deriving requirements from observable queries
-/

-- ═══════════════════════════════════════════════════════════════════
-- PART 1: Context Classification (Greenfield/Retrofit)
-- ═══════════════════════════════════════════════════════════════════

-- A module in the codebase
structure Module where
  name : String
  isExternal : Bool  -- Can architect modify type hierarchies?
deriving DecidableEq, Repr

-- Constraints on the system
inductive Constraint where
  | requiresStructural : String → Constraint  -- e.g., JSON API
  | requiresNominal : String → Constraint     -- e.g., ABC contract
  | noConstraint : Constraint
deriving DecidableEq, Repr

-- Development context
structure Context where
  modules : List Module
  constraints : List Constraint
deriving Repr

-- Greenfield: architect controls all type hierarchies
def isGreenfield (ctx : Context) : Bool :=
  ctx.modules.all (fun m => !m.isExternal) &&
  ctx.constraints.all (fun c => match c with
    | .requiresStructural _ => false
    | _ => true)

-- Retrofit: external dependencies constrain choices
def isRetrofit (ctx : Context) : Bool :=
  ctx.modules.any (fun m => m.isExternal) ||
  ctx.constraints.any (fun c => match c with
    | .requiresStructural _ => true
    | _ => false)

-- Hybrid: some modules greenfield, some retrofit
def isHybrid (ctx : Context) : Bool :=
  ctx.modules.any (fun m => m.isExternal) &&
  ctx.modules.any (fun m => !m.isExternal)

-- THEOREM: Greenfield implies not Retrofit (they are exclusive)
theorem greenfield_not_retrofit (ctx : Context)
    (hg : isGreenfield ctx = true) : isRetrofit ctx = false := by
  simp only [isGreenfield, isRetrofit, Bool.and_eq_true, Bool.or_eq_false_iff] at hg ⊢
  obtain ⟨h_mods, h_cons⟩ := hg
  constructor
  · simp only [List.any_eq_false]
    intro m hm
    have := List.all_eq_true.mp h_mods m hm
    simp only [Bool.not_eq_true', Bool.eq_false_iff] at this
    exact this
  · simp only [List.any_eq_false]
    intro c hc
    have := List.all_eq_true.mp h_cons c hc
    cases c <;> simp_all

-- ═══════════════════════════════════════════════════════════════════
-- PART 2: Requirement Detection
-- ═══════════════════════════════════════════════════════════════════

-- Queries a system needs to answer at runtime
inductive Query where
  | whichTypeProvided : String → Query      -- Provenance query
  | isInstanceOf : Nat → Query              -- Identity query
  | hasAttribute : String → Query           -- Shape query (no provenance)
  | enumerateSubtypes : Nat → Query         -- Enumeration query
  | resolveConflict : String → Query        -- MRO conflict query
deriving DecidableEq, Repr

-- Which queries require the Bases axis (provenance)?
def requiresBasesAxis : Query → Bool
  | .whichTypeProvided _ => true
  | .isInstanceOf _ => true
  | .enumerateSubtypes _ => true
  | .resolveConflict _ => true
  | .hasAttribute _ => false

-- A system specification
structure System where
  queries : List Query
deriving Repr

-- System requires provenance if ANY query needs Bases axis
def requiresProvenance (sys : System) : Bool :=
  sys.queries.any requiresBasesAxis

-- THEOREM: Provenance requirement is decidable (it's a Bool)
-- This is trivially true since requiresProvenance returns Bool
example (sys : System) : Decidable (requiresProvenance sys = true) := inferInstance

-- THEOREM: Shape-only queries don't require provenance
theorem shape_queries_no_provenance :
    ∀ attr : String, requiresBasesAxis (.hasAttribute attr) = false := by
  intro; rfl

-- THEOREM: All other queries require provenance
theorem other_queries_require_provenance :
    ∀ q : Query, ¬(∃ s, q = .hasAttribute s) → requiresBasesAxis q = true := by
  intro q hq
  cases q with
  | whichTypeProvided _ => rfl
  | isInstanceOf _ => rfl
  | hasAttribute s => exact absurd ⟨s, rfl⟩ hq
  | enumerateSubtypes _ => rfl
  | resolveConflict _ => rfl

-- ═══════════════════════════════════════════════════════════════════
-- PART 3: Decision Procedure
-- ═══════════════════════════════════════════════════════════════════

-- Typing disciplines (imported concept)
inductive Discipline where
  | Nominal : Discipline
  | Shape : Discipline
deriving DecidableEq, Repr

-- The decision procedure
def selectDiscipline (ctx : Context) (sys : System) : Discipline :=
  if requiresProvenance sys then
    .Nominal  -- Theorem 3.13: mandatory
  else if isRetrofit ctx then
    .Shape    -- Concession to constraints
  else
    .Nominal  -- Theorem 3.5: strictly dominates

-- THEOREM: If provenance required, nominal is selected
theorem provenance_implies_nominal
    (ctx : Context) (sys : System)
    (h_prov : requiresProvenance sys = true) :
    selectDiscipline ctx sys = .Nominal := by
  simp [selectDiscipline, h_prov]

-- THEOREM: Shape selected ONLY when no provenance AND retrofit
theorem shape_only_when_necessary
    (ctx : Context) (sys : System)
    (h_shape : selectDiscipline ctx sys = .Shape) :
    requiresProvenance sys = false ∧ isRetrofit ctx = true := by
  simp only [selectDiscipline] at h_shape
  split at h_shape
  · contradiction
  · rename_i h_no_prov
    split at h_shape
    · rename_i h_retro
      exact ⟨Bool.eq_false_iff.mpr h_no_prov, h_retro⟩
    · contradiction

-- THEOREM: No false negatives (provenance needed → nominal selected)
theorem no_false_negatives
    (ctx : Context) (sys : System)
    (h_prov : requiresProvenance sys = true) :
    selectDiscipline ctx sys = .Nominal :=
  provenance_implies_nominal ctx sys h_prov

-- THEOREM: Greenfield without provenance still gets nominal (dominance)
theorem greenfield_defaults_nominal
    (ctx : Context) (sys : System)
    (h_green : isGreenfield ctx = true)
    (h_no_prov : requiresProvenance sys = false) :
    selectDiscipline ctx sys = .Nominal := by
  have h_not_retro := greenfield_not_retrofit ctx h_green
  simp only [selectDiscipline, h_no_prov, h_not_retro, Bool.false_eq_true, ↓reduceIte]

-- ═══════════════════════════════════════════════════════════════════
-- PART 4: Example Contexts
-- ═══════════════════════════════════════════════════════════════════

-- Example: Pure greenfield project
def greenfieldExample : Context := {
  modules := [⟨"core", false⟩, ⟨"utils", false⟩],
  constraints := [.noConstraint]
}

-- Example: Retrofit with external dependency
def retrofitExample : Context := {
  modules := [⟨"core", false⟩, ⟨"external_lib", true⟩],
  constraints := [.requiresStructural "JSON API"]
}

-- Example: System requiring provenance
def provenanceSystem : System := {
  queries := [.whichTypeProvided "value", .hasAttribute "name"]
}

-- Example: System not requiring provenance
def shapeOnlySystem : System := {
  queries := [.hasAttribute "x", .hasAttribute "y"]
}

-- THEOREM: Greenfield + provenance → Nominal
example : selectDiscipline greenfieldExample provenanceSystem = .Nominal := by
  native_decide

-- THEOREM: Retrofit + no provenance → Shape
example : selectDiscipline retrofitExample shapeOnlySystem = .Shape := by
  native_decide

-- THEOREM: Greenfield + no provenance → Nominal (dominance)
example : selectDiscipline greenfieldExample shapeOnlySystem = .Nominal := by
  native_decide

