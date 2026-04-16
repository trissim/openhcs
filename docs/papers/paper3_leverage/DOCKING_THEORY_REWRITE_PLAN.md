# Paper3 Docking Theory Rewrite Plan

## Goal

Recast `paper3` as a single, self-contained, theory-only paper with this identity:

`A complete machine-checked theory of exact molecular docking, stated abstractly for bounded decision systems and instantiated concretely for constrained molecular systems.`

The paper should no longer read as a general thermodynamics paper with a molecular application. It should read as a rigorous docking theory paper whose abstract decision-theoretic and thermodynamic machinery is proved once and then instantiated throughout.

## Non-Negotiables

- One paper, not a split program.
- No empirical validation in this paper.
- Both layers remain:
  - abstract theory for bounded decision systems
  - molecular instantiation for docking and constrained molecular systems
- Nothing important is dropped for elegance.
- Every headline claim is either:
  - Lean-mechanized, or
  - explicitly placed in the axiom / scope ledger.
- The existing universal consequences stay:
  - coherence / SSOT reading
  - finite replication entropy gap
  - finite-budget no-collapse
  - substrate lifetime / entropy-throughput ceiling
- Manuscript tone stays declarative.
- Verification uses `python3 scripts/build_papers.py release paper3`.

## Recommended Title

Primary title:

`Exact Molecular Docking: A Machine-Checked Theory of Configuration Resolution, Complexity, and Thermodynamic Cost`

Shorter alternate:

`A Machine-Checked Theory of Exact Molecular Docking`

Files affected:

- `docs/papers/paper3_leverage/latex/leverage_arxiv.tex`
- `docs/papers/paper3_leverage/latex/paper_title_auto.tex` (regenerated)
- `docs/papers/paper3_leverage/markdown/paper3.md` (regenerated)

## Safe Headline Claims

These are already supported strongly enough by the Lean artifact to foreground.

- Exact resolution is defined by sufficient coordinate sets and the decision quotient.
- Degree of freedom equals structural rank in the canonical exact-resolution encoding.
- Exact-resolution cost is bounded below by the same quotient / rank structure that fixes correctness.
- Constrained molecular systems with `N` atoms and `k` independent holonomic constraints have effective dimension `3N-k` and therefore a Landauer-linear exact-resolution floor.
- Cutoff locality bounds molecular docking structural rank by active-site coordinates plus ligand coordinates.
- The bounded-pocket regime is a theorem-backed low-rank regime.
- Sampled docking admits theorem-backed exact/coarse winner agreement under explicit gap assumptions.
- Nonideal implementations lie strictly above the Landauer floor.
- Rank-1 is simultaneously the minimum-cost, tractable, and coherent regime.
- Replication, no-collapse, and finite-lifetime consequences belong to the same model class and should remain in the paper.

## Claims That Need Local Paper3 Bridging Before They Become Headline Theorems

These claims are mostly present in the larger artifact, but they need local paper3 theorem exposure in molecular language.

- General hardness of exact sufficiency certification, stated in exact docking language.
- Explicit docking-language tractability boundary theorem tying bounded pockets / low rank to exact sufficiency.
- A local sampled-docking certification theorem stated as a paper3 result rather than only an imported paper4 theorem.

## Existing Lean Assets To Promote

### Abstract core

- `docs/papers/paper3_leverage/proofs/Leverage/BridgeToDQ.lean`
- `docs/papers/paper3_leverage/proofs/Leverage/ColumnComplexityBridge.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Information.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/ThermodynamicLift.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Physics/BoundedAcquisition.lean`

### Molecular docking core

- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/MolecularSrank.lean`
  - `outside_cutoff_is_irrelevant`
  - `md_relevant_only_if_within_cutoff`
  - `md_srank_bound`
  - `docking_small_pocket_bound`
  - `md_thermodynamic_lower_bound`

### Sampled docking core

- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SampledDockingGap.lean`
  - `SampledDockingProblem.exact_coarse_opt_agree_of_gap`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SampledDockingCutoff.lean`
  - `sampled_md_srank_bound`
  - `sampled_insideCutoff_sufficient`

### Molecular / RATTLE transport

- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Computation/GeometricConstraints.lean`
  - `constraintObservations_card`
  - `effectiveDOF_eq_cartesian_minus_constraints`
- `docs/papers/paper3_leverage/proofs/Leverage/BridgeToDQ.lean`
  - `rattle_constraintObservations_card`
  - `rattle_srank_eq_effectiveDOF`
  - `rattle_energy_lower_bound`

### General hardness and tractability boundary

- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Reduction.lean`
  - `tautology_iff_sufficient`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Summary.lean`
  - `Summary.conp_completeness`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/StructuralRank.lean`
  - `hard_family_srank_eq_n`
  - `srank_le_sufficient_card`

## Preferred Manuscript Structure

### Recommended top-level flow

1. Introduction
2. Foundations of exact molecular docking
3. Exact resolution, quotient structure, and compression
4. Complexity boundary of exact molecular docking
5. Thermodynamic cost of exact molecular docking
6. Nonideal exact resolution and substrate limits
7. Convergence and universal consequences
8. Related work
9. Conclusion

### Mapping from the current paper

| Current file | Current role | New role |
|---|---|---|
| `abstract.tex` | general thermodynamic abstract | docking-theory abstract |
| `01_introduction.tex` | general introduction | docking-first introduction |
| `02_foundations.tex` | abstract foundations | abstract foundations plus molecular semantics |
| `03_probability_model.tex` | rank and entropy | exact resolution / quotient / compression semantics |
| `04_main_theorems.tex` | thermodynamic consequences | split into complexity + thermodynamics, or rewritten so complexity is promoted before thermodynamics |
| `05_five_way_equivalence.tex` | convergence section | convergence plus universal molecular readings |
| `09_conclusion.tex` | summary of generic theory with molecular target | conclusion of a completed abstract-plus-molecular docking theory |

### File-structure recommendation

Preferred route:

- Add a new file for complexity rather than burying it in the convergence section.

Recommended new content file:

- `docs/papers/paper3_leverage/latex/content/04_exact_docking_complexity.tex`

Then renumber:

- current `04_main_theorems.tex` -> thermodynamic cost section
- current `05_five_way_equivalence.tex` -> convergence and universal consequences

If file churn is undesirable, keep the same filenames but still reorder the manuscript content so the complexity boundary appears before the thermodynamic section.

## File-by-File Change Plan

## 1. `latex/leverage_arxiv.tex`

### Change

- Update the title to the docking-theory framing.

### Draft edit

```tex
\title{Exact Molecular Docking: A Machine-Checked Theory of Configuration Resolution, Complexity, and Thermodynamic Cost}
```

### Notes

- Confirm how `paper_title_auto.tex` is regenerated so the canonical title stays aligned.

## 2. `latex/content/abstract.tex`

### Change

- Rewrite completely.
- Open with the missing-theory problem in molecular docking.
- State the abstract layer and the molecular layer.
- State complexity, thermodynamic floor, nonideal strictness, and universal consequences.
- Keep explicit caveat language: under explicit physical and modeling premises.

### Draft replacement text

```text
Exact molecular docking lacks a rigorous theory of exact correctness, exact complexity, and exact thermodynamic cost. This paper gives a machine-checked theory of exact molecular docking in two layers. Abstractly, exact resolution of a bounded decision system is governed by sufficient coordinate sets, structural rank, and decision-quotient entropy. Concretely, constrained molecular systems instantiate the same framework: holonomic constraint topology yields an effective dimension 3N-k, cutoff locality bounds docking structural rank by active-site and ligand coordinates, and exact resolution of the induced binding decision problem carries a Landauer-linear energy floor.

The theory separates the hard and tractable regimes. Exact sufficiency certification is hard in general, while the bounded-pocket regime admits theorem-backed low-rank control and sampled exact/coarse winner preservation under explicit gap hypotheses. In the canonical exact-resolution encoding, degree of freedom equals structural rank, the decision quotient entropy is bounded by the same count, and under Landauer calibration one exact-resolution cycle satisfies E >= DOF(A) k_B T ln 2 and E >= k_B T H_nats(D). For constrained molecular systems this yields a direct holonomic-constraint floor scaling with 3N-k.

The same framework proves more than the ideal floor. Theorem-level mismatch and residual witnesses place nonideal implementations strictly above Landauer. Finite-capacity substrates therefore have finite lifetime and finite entropy-throughput ceiling. The rank-1 regime is simultaneously the minimum-cost, tractable, and coherent single-source regime. The same model yields a finite replication entropy gap and a finite-budget no-collapse theorem. The result is a complete, theory-only, machine-checked foundation for exact molecular docking under explicit physical and modeling axioms.
```

### Follow-up edits

- Update keywords toward docking, exact resolution, structural rank, Landauer cost, certification.

## 3. `latex/content/01_introduction.tex`

### Change

- Rewrite opening paragraphs so the first sentence is about molecular docking, not generic finite information-processing systems.
- Keep the abstract layer visible from paragraph one.
- Reorder the contributions so they tell the docking-theory story:
  1. semantics
  2. complexity boundary
  3. thermodynamic floor
  4. nonideal overhead and lifetime
  5. convergence and universal consequences

### Draft opening paragraph

```text
Molecular docking is used as if its correctness were understood, but the field lacks a rigorous theory of what it means to solve a docking problem exactly, what exact resolution costs, and where the tractable regime ends. This paper gives a machine-checked answer. The abstract theory is stated for bounded decision systems and their canonical exact-resolution encoding; the molecular theory instantiates the same framework for constrained molecular systems, holonomic constraint topology, cutoff-local interactions, and binding decision problems. The central claim is that correctness, complexity, and thermodynamic cost are controlled by the same object: the exact decision quotient and its structural rank.
```

### Draft central-result paragraph

```text
The paper proves a complete chain for exact molecular docking. Exact resolution requires a sufficient coordinate set. The canonical exact-resolution problem identifies the required interaction dimension with structural rank. The same rank controls quotient entropy, separates hard and tractable regimes, and determines the Landauer-linear floor of exact resolution. For constrained molecular systems with N atoms and k independent holonomic constraints, the effective dimension is 3N-k, so exact molecular resolution carries a corresponding thermodynamic floor. Nonideal implementations lie strictly above that floor.
```

### Draft contributions list

```text
1. Exact docking semantics: exact molecular docking is defined as exact configuration resolution under constraints through sufficient coordinate sets and the binding decision quotient.
2. Complexity boundary: exact sufficiency certification is hard in general, while the bounded-pocket regime admits theorem-backed low-rank control and exact/coarse winner preservation under explicit gap assumptions.
3. Thermodynamic floor: exact resolution cost is bounded below by structural rank, and constrained molecular systems inherit a direct 3N-k Landauer floor from holonomic constraint topology.
4. Nonideal exact resolution: mismatch and residual witnesses force strict overhead above the Landauer floor, and finite-capacity substrates therefore have finite lifetime and finite entropy throughput.
5. Convergence and universal consequences: the rank-1 regime is simultaneously minimum-cost, tractable, and coherent; the same model yields a finite replication entropy gap and a finite-budget no-collapse theorem.
```

### Keep and reposition

- The existing `Informally:` lines stay, but they should be woven into the docking narrative rather than left as isolated motifs.

## 4. `latex/content/02_foundations.tex`

### Change

- Keep the abstract definitions.
- After each key abstract object, add a molecular interpretation sentence or short paragraph.
- Add one explicit definition or explanatory paragraph for exact molecular docking in terms of exact configuration resolution.

### Draft additions

After `Bounded Decision System`:

```text
Molecular instantiation. In the docking setting, the bounded decision system is a constrained molecular configuration space together with the binding decision problem induced by the chosen interaction model. The abstract degree-of-freedom count is later instantiated by holonomic constraint topology and local interaction structure.
```

After `DOF`:

```text
Molecular instantiation. For a constrained molecular system with N atoms and k independent holonomic constraints, the transported degree-of-freedom count is 3N-k. Later sections combine this finite topological count with cutoff-local docking structure.
```

After `Canonical Decision Problem`:

```text
Docking reading. The canonical exact-resolution problem records the distinctions that any exact docking resolver must preserve. The quotient of this problem is therefore the exact abstraction of docking correctness, not an auxiliary coding artifact.
```

### New local proposition to add in prose

- A short paper-level proposition or remark titled `Exact Molecular Docking as Exact Configuration Resolution`.
- This can be prose if no new Lean theorem is needed.

## 5. `latex/content/03_probability_model.tex`

### Change

- Reframe the section as exact resolution, quotient structure, and compression semantics.
- Keep the finite compression bridge as a central correctness-cost bridge, not as a side proposition.

### Section-title draft

```text
Exact Resolution, Quotient Structure, and Compression
```

### Draft framing paragraph

```text
The exact object in docking is not a score table alone, but the quotient of configurations by exact optimal-action agreement. Structural rank counts the distinctions that must be preserved for correctness. The compression bridge makes the same point in finite combinatorial language: avoiding collisions in exact resolution is the same distinction structure that later appears as thermodynamic cost.
```

### Keep and strengthen

- Proposition 3.7 stays.
- The post-proof bridge sentence already added should remain.
- Add one docking-specific sentence after Proposition 3.7:

```text
In docking language, the same finite fiber structure records when distinct molecular configurations remain exactly distinguishable under the binding decision relation.
```

## 6. New complexity section

### Preferred new file

- `docs/papers/paper3_leverage/latex/content/04_exact_docking_complexity.tex`

### Purpose

- Make complexity a first-class section rather than a late imported theorem.

### Section contents

1. General hardness of exact sufficiency certification.
2. Structural-rank boundary.
3. Molecular docking low-rank / bounded-pocket regime.
4. Sampled docking exact/coarse preservation and inside-cutoff sufficiency.

### Paper-level claims to include

- `Theorem [General Hardness of Exact Sufficiency Certification]`
- `Theorem [Structural-Rank Boundary for the Hard Family]`
- `Theorem [Cutoff-Local Structural-Rank Bound for Exact Docking]`
- `Corollary [Bounded-Pocket Tractable Regime]`
- `Theorem [Sampled Exact-Coarse Winner Preservation]`
- `Corollary [Inside-Cutoff Sufficiency for Sampled Docking]`

### Draft section opener

```text
Exact docking is not merely expensive; it has a genuine tractability boundary. The general exact-sufficiency problem is hard, while molecular locality and bounded active-site structure induce a low-rank regime in which exact resolution becomes structurally controlled. The docking theorems in this section are the first half of the theory: they state when exact docking is hard and when the molecular geometry forces a tractable exact regime.
```

### Lean sources to cite in `\leanmeta`

- `DecisionQuotient.Summary.conp_completeness`
- `DecisionQuotient.hard_family_srank_eq_n`
- `DecisionQuotient.Tractability.MolecularSrank.md_srank_bound`
- `DecisionQuotient.Tractability.MolecularSrank.docking_small_pocket_bound`
- `DecisionQuotient.Tractability.SampledDockingGap.SampledDockingProblem.exact_coarse_opt_agree_of_gap`
- `DecisionQuotient.Tractability.SampledDockingCutoff.sampled_insideCutoff_sufficient`

## 7. `latex/content/04_main_theorems.tex`

### Change

- Recast as the thermodynamic-cost section of exact molecular docking.
- Keep the abstract theorem statements, but every subsection must carry an explicit molecular instantiation sentence.

### Section-title draft

```text
Thermodynamic Cost of Exact Molecular Docking
```

### Draft opening paragraph

```text
The complexity boundary identifies when exact docking is structurally hard or low-rank. The present section identifies what exact docking costs physically once exact resolution is demanded. The abstract cost statements are proved for bounded decision systems, and the molecular corollaries transport them to constrained molecular systems, holonomic constraint topology, and cutoff-local binding problems.
```

### Keep prominent

- `thm:energy-rank`
- `thm:energy-entropy`
- `cor:holonomic-landauer-floor`
- strict mismatch / residual overhead
- finite lifetime and throughput ceiling

### Draft molecular sentence after `thm:energy-entropy`

```text
For docking, the same theorem says that the exact binding quotient is not only the correctness object but also the thermodynamic cost object: any exact resolver must pay for the quotient distinctions it preserves.
```

## 8. `latex/content/05_five_way_equivalence.tex`

### Change

- Keep the convergence theorem and the universal consequences.
- Add molecular reading sentences after each major result.
- Make clear that these are not detached abstract add-ons; they are universal consequences of the exact docking framework once molecular systems instantiate it.

### Draft molecular readings

For coherence / SSOT:

```text
Molecular reading. In a molecular simulation or docking pipeline, the coherent single-source regime is the regime in which one coordinate description is authoritative and all others remain derived, preventing exact-resolution drift across redundant encodings.
```

For replication gap:

```text
Molecular reading. In the exact-resolution model, molecular copying carries an irreducible entropy premium above the single-copy baseline.
```

For no-collapse:

```text
Molecular reading. In the hard regime, no bounded-energy physical architecture can realize a universal exact polynomial collapse of docking certification demand.
```

For lifetime / throughput if moved here in summary form:

```text
Molecular reading. Repeated exact molecular resolution on a finite-capacity substrate has finite lifetime and finite cumulative entropy throughput.
```

## 9. `latex/content/09_conclusion.tex`

### Change

- The conclusion must stop describing molecular docking as a transport target.
- It must say that the paper already gives the abstract theory and the molecular instantiation together.

### Draft summary paragraph

```text
The paper gives a complete machine-checked theory of exact molecular docking in an abstract-plus-molecular form. Abstractly, exact resolution is governed by sufficient coordinate sets, structural rank, quotient entropy, and thermodynamic floor. Concretely, constrained molecular systems instantiate the same framework through holonomic constraint topology, cutoff-local interaction structure, sampled exact/coarse stability, and direct Landauer-linear cost bounds. The result is a theory of exact docking correctness, complexity, and thermodynamic cost stated in one formal system.
```

### Draft final paragraph

```text
The abstract theory is not decorative generality, and the molecular layer is not an afterthought. The abstract theorems state what exact resolution costs for any bounded decision system. The molecular instantiation shows that the same theorems govern exact docking, constrained molecular computation, and repeated exact molecular resolution in matter.
```

## Lean Work Required

## A. New local paper3 bridge module

Recommended new file:

- `docs/papers/paper3_leverage/proofs/Leverage/DockingTheoryBridge.lean`

Purpose:

- expose docking-specific theorems locally in the paper3 namespace
- avoid overloading `BridgeToDQ.lean`

Recommended contents:

- `docking_general_sufficiency_hardness`
  - wraps `DecisionQuotient.Summary.conp_completeness`
- `docking_hard_family_srank_maximal`
  - wraps `hard_family_srank_eq_n`
- `docking_cutoff_locality`
  - wraps `outside_cutoff_is_irrelevant` and `md_relevant_only_if_within_cutoff`
- `docking_srank_bound`
  - wraps `md_srank_bound`
- `docking_bounded_pocket_regime`
  - wraps `docking_small_pocket_bound`
- `sampled_docking_exact_coarse_agreement`
  - wraps `SampledDockingProblem.exact_coarse_opt_agree_of_gap`
- `sampled_docking_inside_cutoff_sufficient`
  - wraps `sampled_insideCutoff_sufficient`

This module should then be imported by `Leverage.lean`.

## B. Local paper3 theorem exposure

Add paper-level claims in the new complexity section with `\leanmeta{...}` pointing to the local paper3 theorems from `DockingTheoryBridge.lean`.

## C. Keep existing local molecular transport

Do not move or delete the current `rattle_*` and constrained-molecular theorems in `Leverage/BridgeToDQ.lean`. They are already the right bridge for the thermodynamic section.

## D. Optional strengthening if needed

If the unrestricted docking-hardness statement needs a cleaner molecular wrapper than a local alias, add a short abstract-to-molecular theorem explaining that the unrestricted docking family contains the general exact-sufficiency problem as a subfamily or inherits the same certification hardness boundary. Only do this if the alias-level presentation feels too abrupt.

## Claim and Handle Work

- Add the new complexity-section claims to the manuscript with local `\leanmeta` references.
- Regenerate:
  - `claim_mapping_auto.tex`
  - `lean_handle_ids_auto.tex`
  - `12_complete_theorem_index.tex`
  - `markdown/paper3.md`
- Update the theorem index categories so complexity is a main section rather than only a convergence-side imported fact.

## Scope and Axiom Ledger Changes

The docking framing is strongest when the premises are explicit.

Add or strengthen scope language for:

- Landauer calibration
- bounded acquisition interface
- independent holonomic constraints
- cutoff approximation bound
- strict utility-gap assumptions
- sampled-optimum capture assumptions
- any remaining axiomatic MD facts if cited in the main paper

The headline should remain:

- `machine-checked theory of exact molecular docking under explicit physical and modeling axioms`

Avoid the stronger phrase:

- `fully first-principles docking physics`

unless the axiomatic layers are substantially reduced.

## Draft Informal Spine

Keep the current aphoristic rhythm, but make it summarize the docking paper rather than only the generic theory.

Current three:

- `Informally: exact resolution must be paid for.`
- `Informally: to avoid collisions is to pay for distinctions.`
- `Informally: the quotient fixing correctness also fixes cost.`

Recommended additions later in the paper:

- `Informally: every independent coordinate raises the floor.`
- `Informally: rank one is the ground state.`

These five lines are enough. Do not add more unless they replace weaker explanatory prose.

## Implementation Order

### Phase 1: identity shift

- Rewrite title, abstract, introduction, and conclusion.
- Add the molecular interpretation lines to the foundations section.

### Phase 2: complexity promotion

- Add `Leverage/DockingTheoryBridge.lean`.
- Add the local complexity claims to paper3.
- Create the dedicated complexity section or equivalent reordered content.

### Phase 3: thermodynamic molecularization

- Reframe Section 4 around exact molecular docking cost.
- Keep the RATTLE / `3N-k` corollary prominent.

### Phase 4: universal consequences

- Add molecular readings to coherence, replication, no-collapse, and lifetime results.

### Phase 5: cleanup

- Update theorem index, scope statements, and conclusion.
- Run release build and claim checks.

## Verification Checklist

Lean:

- `lake build Leverage`
- if a new bridge file is added: `lake build Leverage.DockingTheoryBridge`

Paper:

- `python3 scripts/build_papers.py release paper3`

Check:

- `true-path` valid
- claim mapping complete
- no missing local handles
- theorem index reflects the new complexity section

## Acceptance Criteria

The rewrite is complete when all of the following are true.

- A reader would identify the paper as a molecular docking theory paper after reading only the title, abstract, and first two introduction paragraphs.
- The abstract and molecular layers are both explicit from Section 1 onward.
- Complexity is a main section, not a side theorem.
- The molecular theorems are promoted by name, not left implicit in imported infrastructure.
- The paper states a complete theory rather than a curated subset.
- Every major abstract theorem has a molecular interpretation sentence.
- The manuscript still builds cleanly with `release` and full claim mapping.
