import Lake
open Lake DSL

def weakLeanArgs : Array String :=
  #[
    "-Dlinter.unusedSimpArgs=false",
    "-Dlinter.unnecessarySimpa=false",
    "-Dlinter.unusedSectionVars=false"
  ]

package «ssot» where
  -- Use shared packages directory
  packagesDir := "/home/ts/code/projects/papers-archive/docs/papers/.lake-shared/packages"
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "a8227f463392ef51e5bd9f68975fe46f5d9057f3"

require abstract_class_system from "./dep_paper1"
require axis_framework from "./dep_paper1"
require lwd_converse from "./dep_paper1"
require Paper1IT from "./dep_paper1"

@[default_target]
lean_lib «Ssot» where
  weakLeanArgs := weakLeanArgs
  globs := #[.submodules `Ssot]
  srcDir := "."
