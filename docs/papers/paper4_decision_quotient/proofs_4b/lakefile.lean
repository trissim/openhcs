import Lake
open Lake DSL

package «paper4_decision_quotient_4b» where
  -- Use shared packages directory
  packagesDir := "/home/ts/code/projects/papers-archive/docs/papers/.lake-shared/packages"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "a8227f463392ef51e5bd9f68975fe46f5d9057f3"

lean_lib «Paper4bStochasticSequential» where
  globs := #[.submodules `Paper4bStochasticSequential]
  srcDir := "."

lean_lib «DecisionQuotient» where
  globs := #[.submodules `DecisionQuotient]
  srcDir := "../proofs"
