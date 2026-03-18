import Lake
open Lake DSL

def moreServerArgs := #[
  "-Dpp.unicode.fun=true",
  "-Dpp.proofs.withType=false"
]

def moreLeanArgs := moreServerArgs

def weakLeanArgs : Array String :=
  if get_config? CI |>.isSome then
    #["-DwarningAsError=true"]
  else
    #[]

package «pentalogy_proofs» where
  -- Use shared packages directory
  packagesDir := "/home/ts/code/projects/papers-archive/docs/papers/.lake-shared/packages"
  moreServerArgs := moreServerArgs

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

-- Paper 1: Typing Discipline
@[default_target]
lean_lib «paper1_nominal_resolution» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

lean_lib «paper1_abstract_class_system» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

lean_lib «paper1_discipline_migration» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

lean_lib «paper1_context_formalization» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

lean_lib «paper1_python_instantiation» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

lean_lib «paper1_typescript_instantiation» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

lean_lib «paper1_java_instantiation» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

lean_lib «paper1_rust_instantiation» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

lean_lib «paper1_axis_framework» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

lean_lib «paper1_Paper1» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

lean_lib «paper1_PrintAxioms» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

-- Paper 2: SSOT
lean_lib «paper2_Basic» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

lean_lib «paper2_Foundations» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

lean_lib «paper2_SSOT» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

-- Ssot submodules (includes Inconsistency.lean formal model)
lean_lib «Ssot» where
  globs := #[.submodules `Ssot]
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

-- Paper 3: Leverage
lean_lib «paper3_Foundations» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

lean_lib «paper3_Leverage» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

-- Paper 4: Decision Quotient
lean_lib «paper4_Basic» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

-- DecisionQuotient submodules
lean_lib «DecisionQuotient» where
  globs := #[.submodules `DecisionQuotient]
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

lean_lib «paper4_DecisionQuotient» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

-- Paper 5: Credibility
-- Main target imports Credibility.* submodules (synced from paper5_credibility/proofs/)
lean_lib «Credibility» where
  globs := #[.submodules `Credibility]
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

lean_lib «paper5_Credibility» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs

