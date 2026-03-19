import Lake
open Lake DSL

package «decision_quotient» where
  -- No custom Lean options needed for build

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "a8227f463392ef51e5bd9f68975fe46f5d9057f3"

@[default_target]
lean_lib «DecisionQuotient» where
  globs := #[.submodules `DecisionQuotient]
  srcDir := "."

lean_exe "arraydsl-export" where
  root := `ArrayDSLExportMain
