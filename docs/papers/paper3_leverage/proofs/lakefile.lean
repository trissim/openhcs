import Lake
open Lake DSL

package «leverage» where
  -- Use shared packages directory
  packagesDir := "/home/ts/code/projects/papers-archive/docs/papers/.lake-shared/packages"
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "a8227f463392ef51e5bd9f68975fe46f5d9057f3"

require nominal_resolution from "dep_paper1"
require ssot from "dep_paper2"
require decision_quotient from "dep_paper4"

@[default_target]
lean_lib «Leverage» where
  globs := #[.submodules `Leverage]
  srcDir := "."
