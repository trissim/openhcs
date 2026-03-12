# Paper 1 Rate-Distortion Extension and Mechanization Plan

## Purpose

This document records:

1. what theoremic extensions have been proposed for Paper 1,
2. what mechanized machinery is already available,
3. what is missing,
4. a realistic mechanization strategy for each proposed component,
5. and whether existing Fano-style machinery from Paper 2 can and should be moved into Paper 1 or a shared dependency layer.

This plan is based on direct inspection of the current Lean sources in:

- `docs/papers/paper1_typing_discipline/proofs/`
- `docs/papers/paper2_ssot/proofs/`
- the vendored mathlib under `paper1_typing_discipline/proofs/.lake/packages/mathlib/`

---

## Executive Summary

The extension splits into two tracks.

### Audit update after migration

The generic finite-source / probabilistic / Fano / observer-tag machinery has now been migrated into Paper 1 and both Paper 1 and Paper 2 compile against the new structure. This removes the main dependency uncertainty in the original plan.

As a result, the practical roadmap is now sharper:

- **Phase A is genuinely ready to start** for deterministic / finite / fiberwise extensions.
- **Phase B remains a separate longer-term program** because the old Paper 1 entropy layer is still partly axiomatic and there is still no full Paper-1-native Shannon rate-distortion framework.

### Track A: exact finite / deterministic / fiberwise theory

This track is strongly supported by the current Paper 1 mechanization base. It can plausibly deliver:

- exact multi-fiber distortion decomposition,
- deterministic rate-distortion laws beyond the uniform single-block theorem,
- fiberwise optimal allocation theorems,
- collision/growth bounds,
- and abstract budget-split optimization results.

This is the best near-term mechanization target.

### Track B: full Shannon-style rate-distortion / Fano / random-coding theory

This track is only partially supported.

- Paper 2 already contains substantial Fano-style finite-source machinery.
- Mathlib now contains PMFs, KL divergence, binary entropy, Bernoulli and Poisson distributions.
- But Paper 1 still relies on an axiomatized entropy layer in its current internal `Entropy.lean`.
- There is no already-integrated mutual information / conditional entropy / rate-distortion framework in Paper 1.

So full `R(D)` mechanization is possible in principle, but it is a larger program and should not be confused with the finite deterministic extension, which is much more tractable now.

---

## Proposed New Components

The proposed additions are:

1. full `R(D)` characterization with fiber decomposition,
2. optimal allocation across fibers,
3. collision probability under growth,
4. zero-error corner `R(0) = H(C|U)` or the correct mechanized zero-error entropy sandwich,
5. expected tag budget under growth,
6. budget split optimization between representation bits and tag bits,
7. and a stronger methodology claim based on mechanization.

These components do **not** all have the same mechanization readiness.

---

## What Is Already Available In Paper 1

### Audit update: newly available migrated modules

The following modules are now present in Paper 1 and compile successfully:

- `proofs/Paper1IT/EntropyGeneral.lean`
- `proofs/Paper1IT/FiniteSource.lean`
- `proofs/Paper1IT/ProbabilisticFinite.lean`
- `proofs/Paper1IT/FanoFinite.lean`
- `proofs/Paper1IT/ObserverTagModel.lean`

Paper 2 now imports these generic modules back from Paper 1, so the migration phase is complete for current purposes.

## 1. Fiber geometry and zero-error machinery

From `proofs/Paper1IT/GraphEntropy.lean`:

- `confusabilityGraph`
- `ZeroErrorTagging`
- `maxFiberCard` at line ~222
- `maxFiberCard_le_one_of_injective`
- `tagFeasible_iff_maxFiberCard_le`
- `TaskRecoverable`
- task/helper-view theorems
- exact block growth laws in `GraphEntropyAsymptotic.lean`

This is the strongest existing base.

### Key existing deterministic coding results already mechanized

- worst-fiber zero-error threshold,
- exact feasibility by maximal fiber size,
- block scaling,
- helper-view/task sufficiency,
- monotonicity under coarsening/refinement.

## 2. Adaptive fiberwise coding machinery

From `proofs/Paper1IT/GraphEntropy.lean`:

- `AdaptiveBitBudgetFeasible` (line ~716)
- `optimalFiberBitLength` (line ~720)
- `expectedAdaptiveBitLength` (line ~724)
- `optimalExpectedAdaptiveBitLength` (line ~728)
- `conditionalOnFiber` (line ~772)
- `conditionalEntropyGiven` (line ~795)
- lower bound:
  - `conditionalEntropyGiven_le_log2_mul_expectedAdaptiveBitLength` (line ~816)
- upper bound:
  - `exists_conditionalCodes_expectedLength_le_entropy_bits_plus_one` (line ~888)

This is enough to support a serious zero-error conditional-entropy section and some more advanced adaptive finite-source results.

## 3. Paper 1 entropy layer

From `proofs/Paper1IT/Entropy.lean`:

- `FiniteDist`
- `entropy`
- `expectedLength`

But crucially:

- `ClassicalEntropyAssumptions` (line ~94)
- `ClassicalCodingAssumptions` (line ~104)

So the current Paper 1 entropy facts are not fully derived. They are assumed through a theoremic interface.

This is the biggest limitation on any strong “fully mechanized rate-distortion” claim.

## 4. Exact counting converse core

From `proofs/lwd_converse.lean`:

- direct finite injectivity / counting converse for colliding classes,
- exact `a ≤ 2^L` style lemmas.

These remain useful for deterministic rate-distortion lower bounds and growth-based corollaries.

---

## What Is Still Useful In Paper 2

Paper 2 still contains SSOT-specific wrappers and applications, but the generic finite-source / Fano machinery needed by Paper 1 has already been extracted. For the purposes of the Phase A extension, there is no longer any dependency reason to keep mining Paper 2 before starting new Paper 1 modules.

### Important conclusion

The migration target that originally lived in Paper 2 has been achieved. The relevant generic machinery is now in Paper 1. Paper 2 is no longer the place to look for missing foundations before beginning Phase A.

---

## Fano Migration Status

## Status

Done for the currently needed generic layer.

## Dependency reality

Current package relation:

- Paper 2 depends on Paper 1
- Paper 1 cannot import Paper 2
- therefore any generic finite-source / Fano machinery needed by Paper 1 had to be moved downward into Paper 1

That migration has now been carried out for the core generic layer.

## Resulting direction

The next mechanization phase should now proceed entirely inside Paper 1.

## Imported targets now present

- `FiniteSource`
- source entropy / mass bounds
- success/failure sets and probabilities
- probabilistic finite-source pushforward machinery
- Fano-style finite coding theorems
- generic observer/tag exact-recovery and clique counting machinery

---

## What Is Missing Even If Fano Moves Over

Moving Paper 2 Fano machinery helps, but it does **not** automatically give full rate-distortion mechanization.

Still missing after migration:

- a paper-1-native distortion theorem beyond the current uniform single-block case,
- a global fiberwise distortion decomposition formalism,
- optimization / allocation theorems,
- growth/collision laws,
- and possibly a cleaner entropy API integrated with Paper 1's existing `VertexDist` rather than Paper 2's `FiniteSource`.

So the migration is valuable, but it is only one part of the extension program.

---

## Mechanization Strategy By Proposed Component

### Audit summary of readiness

| Component | Readiness now | Recommendation |
|---|---|---|
| Exact deterministic multi-fiber distortion | Ready | Start first |
| Zero-error entropy sandwich | Ready | Start second |
| Fiberwise allocation | Ready in discrete form | Do after distortion core |
| Growth collisions | Ready if scoped to simple bounds | Do after allocation |
| Growth tag budget | Ready as bounds | Do after collisions |
| Budget split optimization | Ready only abstractly | Do late in Phase A |
| Full Shannon `R(D)` | Not ready as near-term target | Treat as Phase B |

## Component A. Exact deterministic multi-fiber rate-distortion

### Goal
Extend the current single-collision-block distortion theorem to a global deterministic theorem over multiple fibers.

### Best formulation

Avoid full Shannon `R(D)` first.

Instead define:

- a source distribution on classes,
- an observation map `observe`,
- a per-fiber tag budget,
- and the optimal success mass captured in each fiber under a finite budget.

This yields an exact finite theorem that is much easier to mechanize.

### Available machinery

- `conditionalOnFiber`
- `fiberMass`
- `optimalFiberBitLength`
- `expectedAdaptiveBitLength`
- the exact deterministic single-block theorem already in the paper prose

### Missing machinery

- formal definition of best top-`m` mass on a finite fiber distribution
- a decomposition theorem expressing global distortion as the weighted sum of fiberwise distortions

### Mechanization path

1. define `topMass` for finite distributions on a fiber,
2. define fiberwise success under budget `m = 2^L`,
3. define global distortion as weighted failure across fibers,
4. prove exact decomposition,
5. recover the current uniform theorem as a corollary.

### New module

- `proofs/Paper1IT/FiberRateDistortion.lean`

### Difficulty

- medium

### Recommendation

This should be the first new theorem module.

### Audit note

The path here is now clear. No further dependency extraction is needed before starting this module.

---

## Component B. Zero-error corner and conditional-entropy sandwich

### Goal
Make the paper's `R(0)` corner precise in a mechanized way.

### Available machinery

- `conditionalEntropyGiven`
- lower bound to adaptive bit length
- conditional code upper bound up to `+1`

### Missing machinery

- clean packaging theorem in Paper 1 language
- decision on exact equality vs. lower/upper bounds

### Mechanization path

1. define the paper-level zero-error expected-length quantity,
2. prove the lower bound from existing `conditionalEntropyGiven` theorem,
3. prove the coding upper bound from existing conditional coding theorem,
4. present equality only if moved to asymptotic or idealized real-length setting.

### New module

- `proofs/Paper1IT/ZeroErrorConditionalEntropy.lean`

### Difficulty

- low to medium

### Recommendation

Mechanize the sandwich theorem first, not exact `R(0)=H(C|U)` unless the precise operational meaning is nailed down.

### Audit note

This module should be low-risk because the key lower and upper ingredients already exist in `GraphEntropy.lean`.

---

## Component C. Fiberwise optimal allocation / water-filling-style theorem

### Goal
Show how distortion should be allocated across fibers under a total budget.

### Best first target

Do not start with continuous KKT.

Start with the discrete finite optimization problem:

- integer budgets by fiber,
- exact fiberwise distortion functions,
- prove a greedy or threshold optimality theorem.

### Available machinery

- finite sums
- argmin in mathlib
- convexity tools if later needed

### Missing machinery

- the budgeted distortion objective
- discrete marginal-gain lemmas

### Mechanization path

1. define fiberwise distortion functions,
2. define total expected distortion under allocation `ℓ`,
3. prove monotonicity and diminishing returns where possible,
4. prove greedy optimality for uniform-fiber case first,
5. generalize later.

### New module

- `proofs/Paper1IT/FiberAllocation.lean`

### Difficulty

- medium

### Recommendation

Prove the uniform-fiber allocation theorem first. It is much more likely to go through smoothly in Lean.

### Audit note

Treat this as a discrete optimization module, not a KKT/water-filling module. That keeps the path clear.

---

## Component D. Collision probability under growth

### Goal
Formalize statements of the form: as new items arrive, collisions appear with calculable probability.

### Available machinery

- mathlib Poisson PMF
- mathlib Binomial PMF

### Missing machinery

- occupancy-process layer for representation cells
- global collision aggregation lemmas

### Best first model

Poissonized independent cell counts.

This gives simple, mechanizable formulas like:

- `P[cell collision] = 1 - e^{-μ}(1+μ)`

and then union bounds for any collision.

### Mechanization path

1. define representation-cell counts,
2. prove per-cell collision probability,
3. prove union bound over cells,
4. optionally later derive exact no-collision products under independence.

### New module

- `proofs/Paper1IT/GrowthCollisions.lean`

### Difficulty

- low to medium if using per-cell formulas + union bounds
- medium-hard for exact occupancy combinatorics

### Recommendation

Do per-cell collision and union bounds first.

### Audit note

This component is clear only if the model is fixed early and kept modest. Do not start with exact global occupancy formulas.

---

## Component E. Expected tag budget under growth

### Goal
Turn growth into expected future auxiliary-bit demand.

### Available machinery

- Poisson/Binomial PMFs
- `Nat.clog`

### Missing machinery

- expectation lemmas for `clog`
- clean occupancy-to-bit expectation bounds

### Mechanization path

1. define required tag budget per cell as a function of occupancy,
2. prove monotonicity and simple bounds,
3. derive expectation bounds from occupancy distributions,
4. postpone exact formulas unless really needed.

### New module

- `proofs/Paper1IT/GrowthTagBudget.lean`

### Difficulty

- medium

### Recommendation

Do lower/upper bounds first, not exact expectation formulas.

### Audit note

This should be defined as an expectation-bound module, not an exact closed-form module.

---

## Component F. Budget split optimization

### Goal
Formalize the tradeoff between spending bits on richer representation vs. spending bits on auxiliary identity tags.

### Available machinery

- monotonicity of collision multiplicity under refinement
- finite argmin support in mathlib

### Missing machinery

- an abstract family of representations indexed by representation cost
- a formal notion of total budget and residual ambiguity

### Mechanization path

1. define an abstract refinement family `pi_r`,
2. assume monotonicity of residual ambiguity under refinement,
3. define total budget and objective,
4. prove existence/comparison theorems for optimal split,
5. keep it abstract unless a concrete learned-model law is introduced.

### New module

- `proofs/Paper1IT/BudgetSplit.lean`

### Difficulty

- medium as an abstract theorem
- hard if concrete representation models are demanded

### Recommendation

Keep this theorem abstract.

### Audit note

Do not wait for a concrete learned-representation model. The abstract theorem is enough for Phase A.

---

## Component G. Full Shannon/Fano/random-coding `R(D)`

### Goal
Derive a fully Shannon-style rate-distortion characterization with genuine converse and achievability.

### Available machinery

- PMFs in mathlib
- KL divergence in mathlib
- binary entropy in mathlib
- Paper 2 Fano-style theorems

### Missing machinery

- unified PMF entropy layer in Paper 1
- mutual information / conditional mutual information in the Paper 1 formal environment
- paper-1-native Fano API
- random coding / achievability framework

### Mechanization path

1. extract generic Paper 2 finite-source machinery into Paper 1/shared modules,
2. build PMF entropy + conditional entropy bridge,
3. expose Paper 2 Fano results in a generic Paper 1 namespace,
4. define rate-distortion objects,
5. prove converse,
6. prove achievability.

### New modules

- `proofs/Paper1IT/FiniteSource.lean`
- `proofs/Paper1IT/PMFEntropy.lean`
- `proofs/Paper1IT/FanoFinite.lean`
- `proofs/Paper1IT/RateDistortion.lean`

### Difficulty

- hard to very hard

### Recommendation

Do **not** make this the first extension target.

### Audit note

This is now clearly a Phase B task. The current path is not clear enough to safely start here without reopening foundational entropy work.

---

## Migration Plan For Paper 2 Fano Machinery

## Why migrate it

Paper 2 currently contains generic finite-source and Fano material that Paper 1 now wants. Since Paper 2 already depends on Paper 1, the correct dependency direction is:

- move generic probabilistic IT machinery downward into Paper 1 or a shared base,
- keep Paper 2-specific wrappers in Paper 2.

## Suggested migration phases

### Phase 0. Inventory and isolate generic pieces

From `paper2_ssot/proofs/Ssot/Probabilistic.lean` isolate:

- `FiniteSource`
- entropy on finite sources
- success/failure sets
- observation/tag/decode model
- generic Fano inequalities

### Phase 1. Create generic Paper 1 modules

Add under `docs/papers/paper1_typing_discipline/proofs/Paper1IT/`:

- `FiniteSource.lean`
- `ProbabilisticFinite.lean`
- `FanoFinite.lean`

### Phase 2. Make Paper 2 import these

Replace Paper 2's local generic definitions with imports from Paper 1 where possible, leaving only SSOT-specific wrappers, names, and handle aliases in Paper 2.

### Phase 3. Bridge to current Paper 1 `VertexDist`

Either:

- prove conversion lemmas between `FiniteSource` and `VertexDist`, or
- replace one with the other if the abstractions are close enough.

### Phase 4. Build Paper 1 extension modules on top

Use the migrated finite-source/Fano layer for the stronger probabilistic extension theorems.

---

## Methodology Claim: What Is Safe To Say

### Safe now

- exact finite/deterministic coding laws are mechanized,
- formal instantiations are machine-checked,
- finite-source Fano-style converses can likely be machine-checked if migrated cleanly.

### Not safe yet

Do **not** currently claim:

> first machine-verified rate-distortion characterization for a non-trivial coding problem

Reason:

- Paper 1 entropy/coding layer still contains axiomatic assumptions in `Paper1IT/Entropy.lean`
- full Shannon-style `R(D)` machinery is not yet present in Paper 1

### Safer intermediate formulation

Something like:

> the exact finite and deterministic coding laws, together with their formal instantiations, are machine-checked; the proposed probabilistic and rate-distortion extensions can build on existing finite-source Fano machinery already developed in the broader proof ecosystem.

---

## Recommended Development Order

## Near-term order

1. `FiberRateDistortion.lean`
2. `ZeroErrorConditionalEntropy.lean`
3. migrate generic finite-source / Fano layer from Paper 2 into Paper 1
4. `FiberAllocation.lean`
5. `GrowthCollisions.lean`
6. `GrowthTagBudget.lean`
7. `BudgetSplit.lean`

This order remains correct after the migration audit.

## Long-term order

8. `PMFEntropy.lean`
9. `FanoFinite.lean` cleanup and full Paper 1 integration
10. `RateDistortion.lean`

---

## Concrete Next Actions

1. Audit `paper2_ssot/proofs/Ssot/Probabilistic.lean` into:
   - generic parts to migrate,
   - SSOT-specific wrappers to leave behind.
2. Create a new `Paper1IT/FiniteSource.lean` module and port `FiniteSource`, source entropy, and success/error definitions.
3. Create a new `Paper1IT/FanoFinite.lean` module and port the genuinely generic Fano theorems.
4. Add conversion lemmas between `FiniteSource` and `VertexDist` if both are kept.
5. Only after that start the new deterministic multi-fiber rate-distortion module.

---

## Final Recommendation

If the goal is to add significant new IT content on a realistic timeline, the best path is:

- make Paper 1 the generic IT foundation,
- move Paper 2's generic finite-source / Fano machinery into that foundation,
- extend Paper 1 first with exact deterministic multi-fiber theorems,
- and defer the full Shannon/random-coding `R(D)` program until the entropy layer is no longer axiomatic.

That path gives the best ratio of theoremic payoff to mechanization risk.

## Final Audit Conclusion

The path was clear for Phase A, and Phase A has now been completed.

- The migration phase is complete.
- Both Paper 1 and Paper 2 compile.
- The necessary generic finite-source / probabilistic / Fano / observer-tag foundations now live in Paper 1.
- No additional cross-paper extraction is required before beginning new Paper 1 mechanization.

## Status Checkpoint

### Migration phase

Completed.

Generic finite-source, probabilistic, Fano, and observer-tag machinery has been moved into Paper 1. Paper 2 imports that machinery back from Paper 1 and both proof trees compile successfully.

### Phase A

Completed and checkpointed.

Implemented modules:

- `Paper1IT/FiberRateDistortion.lean`
- `Paper1IT/ZeroErrorConditionalEntropy.lean`
- `Paper1IT/FiberAllocation.lean`
- `Paper1IT/GrowthCollisions.lean`
- `Paper1IT/GrowthTagBudget.lean`
- `Paper1IT/BudgetSplit.lean`

Paper-facing theorem handles now include:

- `FRD1`, `FRD2`
- `ZEC1`
- `FAL1`, `FAL2`
- `GRC1`, `GRC2`, `GRC3`
- `GTB1`, `GTB2`
- `BST1`, `BST2`, `BST3`

These modules and handles have already been wired into the Paper 1 prose where appropriate.

### Phase B

Completed for the current finite entropy-sensitive scope.

Implemented Phase B modules:

- `Paper1IT/FiniteRateDistortionConverse.lean`
- `Paper1IT/FiniteRateDistortionBounds.lean`
- `Paper1IT/PMFEntropy.lean`
- `Paper1IT/RateDistortion.lean`

These modules package the entropy-sensitive finite converse layer already available from the migrated `FanoFinite` machinery into paper-usable theorems, add a PMF/finite-source entropy bridge, and expose a first `RateDistortion` interface in the finite deterministic-budgeted setting. Current handles added:

- `RDC1` finite budgeted converse
- `RDC2` conditional-entropy converse
- `RDC3` observation-only converse
- `RDC4` min-entropy budget converse
- `RDC5` uniform finite converse
- `RDC6` absorbed budget converse
- `RDC7` logarithmic budget lower bound from error
- `RDC8` observation-only entropy-sensitive converse
- `RDC9` observation-only min-entropy bound
- `PMF1`, `PMF2`, `PMF3` for the PMF entropy bridge

### Immediate next Phase B target

The original next-step list has now been partly completed. What remains, if Phase B is to be pushed further, is no longer the finite converse packaging layer. What remains is the genuinely harder work beyond the current finite deterministic-budgeted interface:

1. deeper de-axiomatization of the older `Entropy.lean` interface,
2. a fuller Shannon-style `R(D)` statement beyond the current finite converse packaging,
3. random-coding or stronger achievability infrastructure.

The practical reading is therefore:

1. the finite entropy-sensitive converse layer is now in place,
2. the PMF bridge is now in place,
3. a first `RateDistortion.lean` interface now exists,
4. anything beyond this is already entering the harder Shannon/random-coding frontier.

The correct next move, if more is desired, would now be a genuine Shannon-style expansion rather than more finite-converse packaging.

### Phase B completion condition

For the current plan, Phase B is satisfied by the following completed modules:

- `Paper1IT/FiniteRateDistortionConverse.lean`
- `Paper1IT/FiniteRateDistortionBounds.lean`
- `Paper1IT/PMFEntropy.lean`
- `Paper1IT/RateDistortion.lean`

These provide:

- finite entropy-sensitive converses,
- conditional-entropy and PMF bridges,
- packaged logarithmic budget lower bounds,
- and a first Paper-1-native rate-distortion interface over the finite deterministic-budgeted setting.

What remains after Phase B is a different, strictly harder program:

- de-axiomatizing the older entropy layer,
- full Shannon-style `R(D)` definitions and chain rules,
- and random-coding / achievability beyond the existing finite deterministic constructions.
