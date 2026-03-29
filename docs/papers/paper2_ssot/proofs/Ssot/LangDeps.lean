/-
  Dependency Manager Models - Formal Classification

  Each dependency manager is modeled as a LanguageFeatures record
  reflecting whether the host natively provides definition-time hooks,
  introspection, structural modification, and hierarchy queries.

  Pattern A (Blind Update): missing causal propagation.
  All four dependency managers share the same classification:
  - No definition hooks: editing package.json / pyproject.toml does not
    automatically update the lockfile through the host language runtime.
  - Introspection present: lockfiles and query commands expose resolved
    dependency provenance.
  - No structural modification: the resolved graph is not modifiable at
    definition time through host-native hooks.
  - No hierarchy queries: no host-native registry of all dependents.
-/

import Ssot.Completeness

namespace Deps

open Ssot

def Npm : Ssot.LanguageFeatures := {
  has_definition_hooks := false
  has_introspection := true
  has_structural_modification := false
  has_hierarchy_queries := false
}

def Cargo : Ssot.LanguageFeatures := {
  has_definition_hooks := false
  has_introspection := true
  has_structural_modification := false
  has_hierarchy_queries := false
}

def Poetry : Ssot.LanguageFeatures := {
  has_definition_hooks := false
  has_introspection := true
  has_structural_modification := false
  has_hierarchy_queries := false
}

def Pnpm : Ssot.LanguageFeatures := {
  has_definition_hooks := false
  has_introspection := true
  has_structural_modification := false
  has_hierarchy_queries := false
}

theorem npm_ssot_incomplete : ¬Ssot.ssot_complete Npm := by
  unfold Ssot.ssot_complete Npm
  simp

theorem cargo_ssot_incomplete : ¬Ssot.ssot_complete Cargo := by
  unfold Ssot.ssot_complete Cargo
  simp

theorem poetry_ssot_incomplete : ¬Ssot.ssot_complete Poetry := by
  unfold Ssot.ssot_complete Poetry
  simp

theorem pnpm_ssot_incomplete : ¬Ssot.ssot_complete Pnpm := by
  unfold Ssot.ssot_complete Pnpm
  simp

theorem npm_has_introspection : Npm.has_introspection = true := rfl

theorem cargo_has_introspection : Cargo.has_introspection = true := rfl

theorem poetry_has_introspection : Poetry.has_introspection = true := rfl

theorem pnpm_has_introspection : Pnpm.has_introspection = true := rfl

theorem npm_no_hooks : Npm.has_definition_hooks = false := rfl

theorem cargo_no_hooks : Cargo.has_definition_hooks = false := rfl

theorem poetry_no_hooks : Poetry.has_definition_hooks = false := rfl

theorem pnpm_no_hooks : Pnpm.has_definition_hooks = false := rfl

end Deps
