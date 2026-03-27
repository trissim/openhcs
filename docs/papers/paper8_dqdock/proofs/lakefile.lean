import Lake
open Lake DSL

package «paper8_dqdock» where
  -- Auto-generated scaffold package.

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "a8227f463392ef51e5bd9f68975fe46f5d9057f3"

@[default_target]
lean_lib «Paper8Dqdock» where
  globs := #[.submodules `Paper8Dqdock]
  srcDir := "."
