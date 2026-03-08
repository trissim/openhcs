import Lake
open Lake DSL

def moreServerArgs := #[
  "-Dpp.unicode.fun=true", -- pretty-prints `fun a ↦ b`
  "-Dpp.proofs.withType=false"
]

-- These settings only apply during `lake build`, but not in VSCode editor.
def moreLeanArgs := moreServerArgs

-- These are additional settings which do not affect the lake hash,
-- so they can be enabled in CI and disabled locally or vice versa.
-- Warning: Do not put any options here that actually change the olean files,
-- or inconsistent behavior may result
def weakLeanArgs : Array String :=
  if get_config? CI |>.isSome then
    #["-DwarningAsError=true"]
  else
    #[]

package «nominal_resolution» where
  moreServerArgs := moreServerArgs
  -- add any package configuration options here

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "a8227f463392ef51e5bd9f68975fe46f5d9057f3"

@[default_target]
lean_lib «nominal_resolution» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- add any library configuration options here

@[default_target]
lean_lib «abstract_class_system» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Language-independent abstract class system formalization

@[default_target]
lean_lib «discipline_migration» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Discipline vs migration optimality separation

@[default_target]
lean_lib «context_formalization» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Greenfield/retrofit classification and requirement detection

@[default_target]
lean_lib «python_instantiation» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Python-specific instantiation: type(), C3 MRO, complexity costs

@[default_target]
lean_lib «typescript_instantiation» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- TypeScript-specific instantiation: structural + branded nominal axes

@[default_target]
lean_lib «java_instantiation» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Java-specific instantiation: nominal class tags, reflection observers

@[default_target]
lean_lib «rust_instantiation» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Rust-specific instantiation: trait sets, type_id-based downcast

@[default_target]
lean_lib «axis_framework» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Axis-parametric framework: generic minimality and uniqueness proofs

@[default_target]
lean_lib «AbstractClassSystem» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Modular abstract class system components

@[default_target]
lean_lib «lwd_converse» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Finite counting converse used by downstream papers

@[default_target]
lean_lib «PrintAxioms» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Axiom verification file

@[default_target]
lean_lib «CrossPaperDependencies» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Cross-paper dependency bridge

@[default_target]
lean_lib «HandleAliases» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Semantic handle aliases for LaTeX citations

@[default_target]
lean_lib «DependencyGraph» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Build-time graph export commands shared by graph bridge declarations

@[default_target]
lean_lib «Paper1IT» where
  globs := #[.submodules `Paper1IT]
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Shared information-theoretic modules consumed by downstream papers
