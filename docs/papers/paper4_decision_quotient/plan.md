# Paper 4 Proof-First Plan

## Goal

Exhaust the proof-side work needed to make the stochastic and cross-regime story as defensible as possible before returning to prose cleanup.

## Current proved pieces

- Static core is already strong.
- Stochastic preservation now has a Lean-level explicit-state predicate and explicit-state `InP` package.
- Stochastic decisiveness now has:
  - explicit-state decidability,
  - succinct PP-hardness,
  - scoped complement upper-bound machinery in `OracleUpperBounds.lean`.
- Stochastic anchor/minimum already have paper-level existential upper bounds backed by scoped witness machinery.

## Immediate proof targets

### 1. Succinct stochastic decisiveness upper bounds

Target: push the current scoped upper-bound package as far as possible.

Questions to settle:
- Can the new `FitsCoNPOverPPStyle` decisiveness result be strengthened to a cleaner reusable theorem package?
- Can decisiveness be connected to a standard paper-level `coNP^PP` claim more sharply, even if full oracle-TM machinery is absent?
- Can the complement characterization be reused to simplify anchor/minimum upper-bound statements?

Concrete files:
- `proofs/DecisionQuotient/StochasticSequential/OracleUpperBounds.lean`
- `proofs/DecisionQuotient/StochasticSequential/Computation.lean`

Current investigation result:
- already proved: scoped complement upper bound and complement-as-existential packaging
  - `stochastic_decisiveness_query_fits_conp_over_ppstyle`
  - `stochastic_decisiveness_complement_fits_np_over_ppstyle`
- strongest likely next theorem package:
  1. dualized `ExistsMajority` reduction for the complement of decisiveness,
  2. reduction into a decisiveness-query-family input package,
  3. honest `NP-over-PP-style` hardness/completeness package for the complement decisiveness family.

### 2. Stochastic sufficiency beyond explicit state

Target: determine whether the preservation predicate admits any stronger theorem package than the current explicit-state `InP` result.

Questions to settle:
- Is there a clean bridge from preservation to existing `fiberDecisionProblem` or `SetValued` machinery?
- Is there a nontrivial transfer/hardness theorem for preservation under succinct encoding?
- If not, can we prove impossibility of a simple collapse/identification with decisiveness?

Concrete files:
- `proofs/DecisionQuotient/StochasticSequential/Basic.lean`
- `proofs/DecisionQuotient/StochasticSequential/SetValued.lean`
- `proofs/DecisionQuotient/StochasticSequential/Quotient.lean`
- `proofs/DecisionQuotient/StochasticSequential/Information.lean`

Current investigation result:
- already proved: explicit-state predicate, counted search, and `InP`
- strongest likely next local theorem package:
  1. `stochastic_preservation_implies_static_sufficiency`
  2. `static_sufficiency_iff_stochastic_preservation_of_full_support`
  3. `stochasticDecisionEquiv_iff_decisionEquiv_of_preservation`
- likely blocked without new infrastructure:
  - unconditional converse,
  - direct hardness reductions to preservation,
  - stronger oracle-class claims.

### 3. Sequential upper bounds

Target: sharpen the proof-side support for sequential membership, if feasible, after the stochastic middle stabilizes.

Questions to settle:
- Can current counted-search or verifier machinery be lifted into a cleaner paper-facing upper-bound theorem?
- Are there reusable witness/no-witness schemas analogous to the stochastic side?

Concrete files:
- `proofs/DecisionQuotient/StochasticSequential/Computation.lean`
- `proofs/DecisionQuotient/StochasticSequential/OracleUpperBounds.lean`
- `proofs/DecisionQuotient/StochasticSequential/PolynomialReduction.lean`

## Investigation workflow

1. Inspect current Lean definitions and theorem dependencies.
2. Prove the strongest theorem that the current machinery naturally supports.
3. Rebuild the specific Lean modules after each step.
4. Only after proofs stabilize, update paper prose and handle mappings.

## Stop condition

Stop only when no further nontrivial proof strengthening seems reachable without building a substantially new complexity-class infrastructure from scratch.

## Current recommended order

1. Prove the preservation bridge package (forward implication, then full-support converse, then quotient corollaries).
2. Package the decisiveness complement side with a dual existential-majority reduction if feasible.
3. Reassess whether any stronger hierarchy-style claims become defensible.

## Reachable theorem backlog

This is the implementation queue. We should work through these items in order and
only stop when all items marked "reachable" are either proved or shown blocked
by a genuinely new infrastructure requirement.

### A. Stochastic sufficiency (preservation) package

Status: central and likely reachable with local work.

1. `stochastic_preservation_implies_static_sufficiency`
   - File: `proofs/DecisionQuotient/StochasticSequential/Basic.lean`
   - Goal: if `fiberOpt P I s = P.toDecisionProblem.Opt s` for all `s`, then the
     underlying static decision problem is sufficient on `I`.
   - Priority: highest.
   - Status: proved.

2. `static_sufficiency_of_full_support_implies_stochastic_preservation`
   - File: `proofs/DecisionQuotient/StochasticSequential/Basic.lean`
   - Goal: under full-support distribution and the right local hypotheses,
     static sufficiency lifts to stochastic preservation.
   - Priority: high.
   - Status: proved as `static_sufficiency_implies_stochastic_preservation_of_full_support`.

3. `static_sufficiency_iff_stochastic_preservation_of_full_support`
   - File: `proofs/DecisionQuotient/StochasticSequential/Basic.lean`
   - Goal: combine A1 and A2 into the strongest defensible equivalence.
   - Priority: high.
   - Status: proved.

4. `stochasticDecisionEquiv_iff_decisionEquiv_of_preservation`
   - File: `proofs/DecisionQuotient/StochasticSequential/Quotient.lean`
   - Goal: under preservation, stochastic fiber-based equivalence matches the
     static optimizer quotient relation.
   - Priority: medium.
   - Status: proved.

5. `stochastic_preservation_inP_explicit_summary`
   - File: `proofs/DecisionQuotient/StochasticSequential/OracleUpperBounds.lean`
   - Goal: package the explicit-state preservation theorem more cleanly for
     paper-facing use.
   - Priority: medium.
   - Status: proved as `stochastic_preservation_explicit_summary`.

### B. Stochastic decisiveness upper-bound / hardness package

Status: partially proved; strongest near-term route is via complement.

6. `stochastic_decisiveness_counterexample_input`
   - File: `proofs/DecisionQuotient/StochasticSequential/OracleUpperBounds.lean`
   - Goal: define a reusable packaged input/output wrapper for the complement
     side, so the no-witness and existential-witness statements compose better.
   - Priority: medium.
   - Status: proved via `StochasticDecisivenessQueryInput` and the decisiveness complement query packaging.

7. `reduceExistsMajorityPureDecisivenessComplement_correct`
   - File: `proofs/DecisionQuotient/StochasticSequential/NPPPHardness.lean`
   - Goal: dual gadget theorem reducing `ExistsMajority` to failure of
     decisiveness.
   - Priority: high.
   - Status: proved as `reduceExistsNonMajorityPureDecisiveness_correct`.

8. `reduceExistsMajority_to_stochastic_decisiveness_complement_query_family_reduction`
   - File: `proofs/DecisionQuotient/StochasticSequential/NPPPHardness.lean`
   - Goal: packaged reduction into decisiveness-complement query inputs.
   - Priority: high.
   - Status: proved via `reduceExistsMajority_to_stochastic_decisiveness_query_family_reduction`.

9. `existsMajority_decisiveness_complement_query_family_honest_np_over_ppstyle_hard`
   - File: `proofs/DecisionQuotient/StochasticSequential/NPPPHardness.lean`
   - Goal: honest hardness packaging for complement decisiveness.
   - Priority: high.
   - Status: proved.

10. `existsMajority_decisiveness_complement_query_family_honest_np_over_ppstyle_complete`
    - File: `proofs/DecisionQuotient/StochasticSequential/NPPPHardness.lean`
    - Goal: combine the reduction with `stochastic_decisiveness_complement_fits_np_over_ppstyle`.
    - Priority: high.
    - Status: proved.

11. `stochastic_decisiveness_upper_bound_summary`
    - File: `proofs/DecisionQuotient/StochasticSequential/ExistentialHardness.lean`
    - Goal: one clean theorem bundling PP-hardness + scoped complement upper bound.
    - Priority: medium.
    - Status: proved as `stochastic_decisiveness_scoped_oracle_bounds`.

### C. Stochastic existential anchor/minimum packaging cleanup

Status: largely present, but worth tightening if local.

12. `stochastic_anchor_and_minimum_upper_bound_summary`
    - File: `proofs/DecisionQuotient/StochasticSequential/OracleUpperBounds.lean`
    - Goal: package the two existential upper bounds plus the new decisiveness
      upper-bound machinery in one summary theorem.
    - Priority: medium.
    - Status: already present as `stochastic_existential_queries_fit_np_over_ppstyle`.

13. `stochastic_anchor_query_family_honest_complete_summary`
    - File: `proofs/DecisionQuotient/StochasticSequential/ExistentialHardness.lean`
    - Goal: paper-facing wrapper theorem for the anchor family after the new
      decisiveness complement work lands.
    - Priority: medium.
    - Status: proved, and complemented by decisiveness-complement query-family wrappers.

### D. Sequential proof-side strengthening

Status: secondary, but still within reach if stochastic work stabilizes.

14. `sequential_sufficiency_upper_bound_summary`
    - File: `proofs/DecisionQuotient/StochasticSequential/Computation.lean`
    - Goal: a cleaner summary theorem from the counted-search package for
      explicit-state sequential sufficiency.
    - Priority: medium.
    - Status: proved.

15. `sequential_anchor_upper_bound_summary`
    - File: `proofs/DecisionQuotient/StochasticSequential/Computation.lean`
    - Goal: same for sequential anchor.
    - Priority: low.
    - Status: proved.

16. `sequential_minimum_upper_bound_summary`
    - File: `proofs/DecisionQuotient/StochasticSequential/Computation.lean`
    - Goal: same for sequential minimum.
    - Priority: low.
    - Status: proved as a counted-search summary rather than an `InP` theorem.

### E. Only-if-reachable hierarchy support

Status: do not touch until A-D stabilize.

17. `benchmark_escalation_summary`
    - File: likely `proofs/DecisionQuotient/StochasticSequential/ExistentialHardness.lean`
      or `Hierarchy.lean`
    - Goal: a careful benchmark escalation statement from static -> decisiveness -> sequential,
      using only the theorem packages actually proved.
    - Priority: lowest.
    - Status: proved in `StochasticSequential/Hierarchy.lean` as a careful benchmark summary rather than a stronger unsupported hierarchy claim.

## Implementation rule

- Work top to bottom.
- After each theorem, rebuild the smallest relevant Lean target.
- If a theorem fails because of a real conceptual blocker, mark it blocked in
  this file and move on.
- Do not return to prose until the reachable subset of A-D is exhausted.

## Current status

- All reachable items in A-D are now proved or packaged in Lean.
- Item E is also now satisfied in cautious form via `benchmark_escalation_summary`.
- Remaining work before prose is mainly validation and theorem-to-paper remapping,
  not obvious missing local proofs.
