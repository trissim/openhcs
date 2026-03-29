/-
  Database Models - Formal Classification

  Two database architecture patterns:

  Pattern C (Coherent Kernel): Engine-maintained materialized views.
  The database engine natively maintains the derived relation and
  exposes catalog metadata for it. Edits to the base table propagate
  automatically through the engine's refresh mechanism, and provenance
  is queryable through system catalogs.

  Pattern A/B (Blind Update / Amnesiac Derivation): External ETL copy.
  Synchronization is externalized to a separate process. Provenance is
  no longer intrinsic to the host database, and base-table edits do not
  automatically propagate to the copy.
-/

import Ssot.Completeness

namespace Database

open Ssot

def EngineMaintainedView : Ssot.LanguageFeatures := {
  has_definition_hooks := true
  has_introspection := true
  has_structural_modification := true
  has_hierarchy_queries := true
}

def ExternalETLCopy : Ssot.LanguageFeatures := {
  has_definition_hooks := false
  has_introspection := false
  has_structural_modification := false
  has_hierarchy_queries := false
}

theorem engine_view_ssot_complete : Ssot.ssot_complete EngineMaintainedView := by
  unfold Ssot.ssot_complete EngineMaintainedView
  simp

theorem engine_view_has_hooks : EngineMaintainedView.has_definition_hooks = true := rfl

theorem engine_view_has_introspection : EngineMaintainedView.has_introspection = true := rfl

theorem external_etl_ssot_incomplete : ¬Ssot.ssot_complete ExternalETLCopy := by
  unfold Ssot.ssot_complete ExternalETLCopy
  simp

theorem external_etl_no_hooks : ExternalETLCopy.has_definition_hooks = false := rfl

theorem external_etl_no_introspection : ExternalETLCopy.has_introspection = false := rfl

end Database
