import Lake
open Lake DSL

package «TheoreticalMinimality» where
  -- Use shared packages directory
  packagesDir := "/home/ts/code/projects/papers-archive/docs/papers/.lake-shared/packages"
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "a8227f463392ef51e5bd9f68975fe46f5d9057f3"

@[default_target]
lean_lib «TheoreticalMinimality» where
  globs := #[.submodules `TheoreticalMinimality]

-- Papers 1-3 proofs copied for self-contained archive
lean_lib «Paper1Proofs» where
  globs := #[.submodules `Paper1Proofs]

lean_lib «Paper2Proofs» where
  globs := #[.submodules `Paper2Proofs]

lean_lib «Paper3Proofs» where
  globs := #[.submodules `Paper3Proofs]
