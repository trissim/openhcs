# Paper 1 Fano Migration Checklist

## Goal

Move the generic finite-source / Fano-style machinery currently living in Paper 2 into Paper 1's proof foundation, so that:

- Paper 1 can use it directly for the planned rate-distortion extension,
- Paper 2 can continue to depend on Paper 1 rather than the other way around,
- and the generic information-theoretic layer is no longer duplicated across papers.

---

## Why This Migration Is Needed

Current dependency direction:

- Paper 2 depends on Paper 1
- therefore Paper 1 cannot import Paper 2

But the relevant generic machinery already exists in Paper 2 under:

- `docs/papers/paper2_ssot/proofs/Ssot/Probabilistic.lean`
- `docs/papers/paper2_ssot/proofs/Ssot/EntropyGeneral.lean`

So the clean architecture is:

1. extract generic finite-source and Fano material from Paper 2,
2. re-home it under `docs/papers/paper1_typing_discipline/proofs/Paper1IT/`,
3. let Paper 2 import that migrated machinery back through its existing dependency on Paper 1.

---

## Source Inventory: What To Migrate

## From `Ssot/EntropyGeneral.lean`

Generic items:

- PMF entropy wrapper `pmfEntropy`
- Bernoulli / uniform entropy lemmas
- KL-import sanity theorems

These are generic and should move into a Paper 1 entropy-support module.

## From `Ssot/Probabilistic.lean`

Likely generic and migratable:

- `FiniteSource`
- `maxMass`, `minEntropy`, `sourceEntropy`
- `successSet`, `failureSet`, `successProb`, `errorProb`
- `pairFiber`, `pairMass`
- `conditionalEntropyGivenPair`
- `mutualInfoDeterministic` and its upper-bound lemmas
- generic Fano theorems:
  - `fano_arbitrary_budgeted`
  - `fano_arbitrary_observation_only`
  - `conditionalEntropyGivenPair_le_fano_arbitrary`
  - `weak_fano_uniform_budget_lower_bound`
  - `fano_uniform_budgeted`

Likely Paper-2-specific wrappers to leave behind:

- SSOT-oriented aliases / handle names
- any theorem statements phrased in Paper 2’s narrative language rather than generic coding language

---

## Target Module Layout In Paper 1

## Phase 1: foundations

- `proofs/Paper1IT/FiniteSource.lean`
  - `FiniteSource`
  - `maxMass`, `minEntropy`, `sourceEntropy`
  - `successSet`, `failureSet`, `successProb`, `errorProb`
  - `pairFiber`, `pairMass`
  - PMF conversions

- `proofs/Paper1IT/EntropyGeneral.lean`
  - `pmfEntropy`
  - Bernoulli and uniform entropy lemmas
  - KL utility imports / wrappers

## Phase 2: finite probabilistic coding layer

- `proofs/Paper1IT/ProbabilisticFinite.lean`
  - `conditionalEntropyGivenPair`
  - `mutualInfoDeterministic`
  - budget-cardinality lemmas
  - entropy partition lemmas

## Phase 3: Fano layer

- `proofs/Paper1IT/FanoFinite.lean`
  - weak Fano uniform results
  - arbitrary-source Fano results
  - observation-only specializations
  - conditional-entropy-via-Fano corollaries

---

## Migration Order

## Step 1. Port generic entropy support

Port from Paper 2:

- `pmfEntropy`
- Bernoulli / uniform entropy lemmas

This should be nearly mechanical.

## Step 2. Port `FiniteSource`

Port from Paper 2:

- `FiniteSource`
- source entropy
- mass bounds
- PMF conversions

This is the correct first implementation step because later Fano theorems depend on it.

## Step 3. Port generic decoding / success-error infrastructure

Port:

- `successSet`
- `failureSet`
- `successProb`
- `errorProb`
- pair fibers / pair masses

## Step 4. Port entropy decomposition lemmas

Port:

- `conditionalEntropyGivenPair`
- entropy partition lemmas
- decoded-output entropy lemmas

## Step 5. Port Fano theorems

Port only after the earlier layers compile unchanged.

---

## Rename / Namespace Policy

The migrated modules should not keep the Paper 2 namespace `ObserverModel` unless there is a compelling reason.

Recommended target namespaces:

- `Ssot.Paper1IT.FiniteSource`
- `Ssot.Paper1IT.FanoFinite`

or a flatter style such as:

- `namespace Paper1IT`

The key requirement is to make clear that these are generic coding modules, not SSOT-specific observer semantics.

---

## API Compatibility Goal

When possible, keep theorem statements close enough that Paper 2 can switch imports with only:

- namespace updates,
- minor import changes,
- and handle alias rewiring.

That means:

- avoid gratuitous renaming during the first migration,
- prefer semantic cleanup after the initial port compiles.

---

## Paper 2 Follow-Up Checklist

After the Paper 1 port compiles:

1. replace local generic definitions in Paper 2 with imports from Paper 1,
2. leave only SSOT-specific wrappers / aliases in Paper 2,
3. rerun Paper 2 builds,
4. verify handle aliases still point to the right declarations.

---

## Immediate First Implementation Task

Start with:

- `proofs/Paper1IT/FiniteSource.lean`

Port into it:

- `FiniteSource`
- `maxMass`, `minEntropy`, `sourceEntropy`
- `successSet`, `failureSet`, `successProb`, `errorProb`
- PMF conversion lemmas

This gives Paper 1 a proper finite probabilistic source layer and unlocks later porting of Fano theorems.

---

## Success Condition

The migration is successful when:

1. Paper 1 has a self-contained generic finite-source probabilistic layer,
2. Paper 2 imports that layer instead of re-owning it,
3. new Paper 1 rate-distortion extension modules can depend only on Paper 1 internals,
4. no dependency edge from Paper 1 to Paper 2 is introduced.
