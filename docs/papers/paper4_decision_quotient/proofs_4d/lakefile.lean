import Lake
open Lake DSL

package «paper4_decision_quotient_4d» where
  -- Paper-specific frontier layer over the main DecisionQuotient library.

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "a8227f463392ef51e5bd9f68975fe46f5d9057f3"

@[default_target]
lean_lib «Paper4dFrontier» where
  globs := #[.submodules `Paper4dFrontier]
  srcDir := "."

lean_lib «DecisionQuotient» where
  globs := #[.submodules `DecisionQuotient]
  srcDir := "../proofs"
