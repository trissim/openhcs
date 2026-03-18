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
  -- Use shared packages directory
  packagesDir := "/home/ts/code/projects/papers-archive/docs/papers/.lake-shared/packages"
  moreServerArgs := moreServerArgs
  -- add any package configuration options here

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «nominal_resolution» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- add any library configuration options here

lean_lib «abstract_class_system» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Language-independent abstract class system formalization

lean_lib «discipline_migration» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Discipline vs migration optimality separation

lean_lib «context_formalization» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Greenfield/retrofit classification and requirement detection

lean_lib «python_instantiation» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Python-specific instantiation: type(), C3 MRO, complexity costs

lean_lib «typescript_instantiation» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- TypeScript-specific instantiation: structural + branded nominal axes

lean_lib «java_instantiation» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Java-specific instantiation: nominal class tags, reflection observers

lean_lib «rust_instantiation» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Rust-specific instantiation: trait sets, type_id-based downcast

lean_lib «axis_framework» where
  moreLeanArgs := moreLeanArgs
  weakLeanArgs := weakLeanArgs
  -- Axis-parametric framework: generic minimality and uniqueness proofs
