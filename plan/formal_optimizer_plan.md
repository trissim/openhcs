# Formal Optimizer Plan

## Objective

Replace the current heuristic certified-mode local refinement path with an
implementation-grade optimizer whose runtime objects and update steps match the
Paper 4 decision-theoretic story:

- finite admissible action family
- explicit belief/posterior state
- Bayes/DQ update over that action family
- deterministic action selection backed by existing sampled-docking and top-k
  theorems
- JAX execution through the generated Lean ArrayDSL bridge

This plan is intentionally narrow: it targets the **certified-mode optimizer in
the docking pipeline**, not the entire stochastic search stack.

## Current gap

### What is already formal enough

- Certified LJ score and cutoff/error bound:
  - `dq_dock_engine/docking/scoring.py`
  - `LatticeSum.lean::lj6_tail_bound`
- Gap certification and winner-preservation reasoning:
  - `DecisionQuotient/Tractability/SampledDockingGap.lean`
  - `DecisionQuotient/Tractability/CertifiedPruning.lean`
  - `DecisionQuotient/Tractability/NearTieBand.lean`
  - `DecisionQuotient/Tractability/FiniteTopK.lean`
- Finite sampled action support as a first-class theorem object:
  - `DecisionQuotient/Tractability/SampledDocking.lean`

### What is still heuristic in certified mode today

- `dq_dock_engine/docking/optimization.py`
  - fixed learning rates
  - norm thresholding
  - clipped step sizes
  - direct gradient descent update rule
- `dq_dock_engine/docking/pipeline.py`
  - top-k-to-optimize then local gradient refinement

The plan below removes those heuristic update mechanics from certified mode.

## Theorem inventory to reuse directly

### Paper 4 / Bayes layer

- `DecisionQuotient/BayesFromDQ.lean`
  - `Admissible`
  - `Efficient`
  - `BayesOptimality`
  - `bayes_uniquely_admissible`
  - `admissible_unique`
- `DecisionQuotient/StochasticSequential/TemporalLearning.lean`
  - `posterior`
  - `bayesian_update`
- `DecisionQuotient/IntegrityCompetence.lean`
  - `ClaimAdmissible`
  - `EvidenceForReport`
  - `signal_consistent_of_claim_admissible`

### Sampled docking / tractability layer

- `DecisionQuotient/Tractability/SampledDocking.lean`
  - `SampledActionFamily`
  - `restrictedDecisionProblem`
  - `restricted_opt_eq_ambient_slice_of_exists_global_sampled_opt`
- `DecisionQuotient/Tractability/SampledDockingCutoff.lean`
  - restricted relevance transport under sampled-support capture
- `DecisionQuotient/Tractability/SampledDockingGap.lean`
  - exact/coarse winner preservation under bounded perturbation
- `DecisionQuotient/Tractability/FiniteTopK.lean`
  - `topKWithTies`
  - `survivorSet`
  - deterministic set semantics for tie-safe ranking
- `DecisionQuotient/Tractability/NearTieBand.lean`
  - ambiguity-band retention semantics
- `DecisionQuotient/Tractability/CertifiedPruning.lean`
  - theorem-backed survivor certificates

## Design decision: discrete optimizer, not continuous gradient descent

The formal optimizer will be implemented as a **finite action-family Bayes
selector**.

### Why this is the right boundary

- It matches the existing Lean formalization exactly.
- It removes heuristic step clipping and learning-rate tuning.
- It lets the optimizer remain local by making the action family a small
  deterministic perturbation stencil around the current pose.
- It makes the update rule an explicit `UpdateRule` over a finite support,
  which is the object the Paper 4 theorems already talk about.

### What we are not doing in this phase

- no continuous-time gradient-flow proof
- no stochastic policy in certified mode
- no heuristic annealing / random restarts / basin hopping
- no reciprocal-space spectral optimizer logic

## Formal runtime objects

### 1. Action representation

Certified local refinement acts on a finite action family of rigid-body moves.

```python
@dataclass(frozen=True)
class CertifiedLocalAction:
    action_id: int
    translation_delta: jax.Array  # shape (3,)
    quaternion_delta: jax.Array   # shape (4,), already normalized
    is_noop: bool
```

Requirements:

- deterministic order
- no duplicate `action_id`
- explicit no-op action present
- all deltas generated from a fixed stencil, not random noise

### 2. Action family

```python
@dataclass(frozen=True)
class CertifiedActionFamily:
    actions: tuple[CertifiedLocalAction, ...]
    translation_step: float
    rotation_step_rad: float
    stencil_level: int
```

This is the runtime analogue of `SampledActionFamily`.

### 3. Belief state

```python
@dataclass(frozen=True)
class CertifiedBeliefState:
    prior: jax.Array              # shape (A,)
    posterior: jax.Array          # shape (A,)
    coarse_scores: jax.Array      # shape (A,)
    exact_scores: jax.Array       # shape (A,)
    exact_error_bound: float
    survivor_mask: jax.Array      # bool, shape (A,)
    ambiguity_mask: jax.Array     # bool, shape (A,)
    selected_action: int
    step_index: int
```

Invariants:

- `prior.sum() == 1`
- `posterior.sum() == 1`
- `posterior[a] == 0` whenever `survivor_mask[a] == False`
- `selected_action` must index an action with nonzero posterior mass

### 4. Optimizer state

```python
@dataclass(frozen=True)
class CertifiedOptimizerState:
    translation: jax.Array        # shape (3,)
    quaternion: jax.Array         # shape (4,)
    action_family: CertifiedActionFamily
    belief: CertifiedBeliefState
```

The optimizer state is therefore not “current point + gradient”. It is “current
point + admissible local action family + posterior over next actions”.

## Non-heuristic admissible action set

### Deterministic local stencil

For a current rigid pose `(t, q)`, define a deterministic local action family:

- no-op action
- 6 axis-aligned translations:
  - `(+dx,0,0)`, `(-dx,0,0)`, ...
- optional 12 diagonal translations at the next stencil level
- 6 signed axis-angle rotations around canonical axes
- optional product actions `(translation, rotation)` only if we explicitly cap
  the family size and can keep the set deterministic

Recommended first certified family:

- 1 no-op
- 6 translations
- 6 rotations
- total = 13 actions

This keeps the Lean/runtime interface small and auditable.

### Quaternion perturbation rule

- Build a unit quaternion from `(axis, +/- theta)`
- Compose with current quaternion using a deterministic multiplication order
- normalize once through DSL-backed rigid transform machinery

### Why this is non-heuristic enough

The family still depends on step sizes, but those step sizes are now:

- explicit parts of the declared action family
- serialized in the optimizer state
- visible in theorem assumptions
- not hidden inside a gradient-clipping rule

The action family is then a **declared finite search support**, not an implicit
heuristic optimizer dynamic.

## Bayes/DQ update rule in runtime terms

### Key idea

The update rule will not be “take a gradient step”.

It will be:

1. generate finite local actions
2. evaluate their coarse and exact/certified utilities
3. use theorem-backed survivor/ambiguity logic to derive evidence
4. update posterior over actions via Bayes
5. choose the admissible maximizer deterministically

### Evidence channel

We need an evidence object `e_t` that is not heuristic. The cleanest first
choice is a **certified survival event** induced by exact/coarse agreement.

For each action `a`, define:

- `u_exact(a)` = certified LJ utility
- `u_coarse(a)` = certified coarse proxy at the same action family level
- `delta` = exact/coarse perturbation bound
- `survivor(a)` = whether `a` lies in the theorem-backed survivor set

Then the evidence channel is:

- observed evidence at step `t`: the set of actions surviving certified pruning

This is theorem-aligned because:

- `CertifiedPruning.lean` packages survivor-set soundness
- `NearTieBand.lean` gives a conservative ambiguity band
- `SampledDockingGap.lean` gives exact/coarse winner preservation under gap
  conditions

### Likelihood model

To avoid temperature/softmax heuristics, use a support-restriction likelihood:

- `likelihood(a, e_t) = 1` if `a` survives the certified evidence filter
- `likelihood(a, e_t) = 0` otherwise

Then the posterior is exactly the Bayes restriction:

`posterior(a) ∝ prior(a) * 1_{survivor(a)}`

This has several benefits:

- no arbitrary temperature
- no hidden Gibbs assumption
- posterior update is literally finite-support Bayes conditioning
- evidence semantics are theorem-backed by pruning/survivor certificates

### Prior choice

This is the one place where the plan must be explicit about assumptions.

#### Step 0 prior

Recommended first implementation:

- uniform prior over the finite action family

Why this is acceptable for now:

- deterministic
- symmetry-respecting on a declared stencil
- easy to audit

But note carefully:

- uniform prior is still a modeling assumption, not currently a proved theorem in
  the repository

So the runtime should label this assumption explicitly in docs/metadata.

#### Step k>0 prior

Use posterior carry-forward between refinement rounds:

- `prior_{t+1} = posterior_t` reindexed onto the next local family by pushing the
  retained winner to the new center and resetting all local perturbation mass to
  a uniform refinement prior around that new center

For first implementation, simpler is better:

- each refinement round starts from a uniform prior on its newly declared local
  family

This is still deterministic and easy to reason about.

### Action selection rule

After computing the posterior:

1. compute the posterior-maximizing set
2. intersect with the certified ambiguity band if ties remain
3. choose the deterministic first element of the ordered action list

This gives:

- theorem-backed admissible action filtering
- deterministic tie-breaking
- no stochastic optimizer behavior in certified mode

## Exact/coarse score pairing

We need two score families because the tractability theorems speak in exact vs
coarse terms.

### Runtime interpretation

- `exact score`: certified LJ score with exact certified cutoff/error metadata
- `coarse score`: a cheaper certified coarse surrogate over the same action family

### First concrete coarse surrogate

Use the same certified LJ form with one or more restrictions:

- fewer receptor atoms retained by theorem-backed cutoff locality
- reduced action family or filtered contact set
- possibly lower-fidelity but still DSL-backed typed-pair aggregation

Important:

- in certified mode, the coarse scorer must not be the old heuristic LJ scorer
- it must be a structurally simpler certified surrogate with an explicit error
  relation to the exact scorer

### Required implementation work

This likely needs a new module:

- `dq_dock_engine/docking/formal_surrogates.py`

with:

- `score_certified_exact_local_family(...)`
- `score_certified_coarse_local_family(...)`
- `bound_exact_minus_coarse(...)`

## Concrete module plan

### New modules

- `dq_dock_engine/docking/formal_actions.py`
  - deterministic local action family generation
  - quaternion perturbation helpers
  - ordered action list serialization

- `dq_dock_engine/docking/formal_belief.py`
  - prior/posterior dataclasses
  - support-restriction Bayes update
  - posterior normalization

- `dq_dock_engine/docking/formal_pruning.py`
  - survivor masks
  - ambiguity-band masks
  - top-k-with-ties helper wrappers

- `dq_dock_engine/docking/formal_optimizer.py`
  - optimizer state dataclasses
  - one-step admissible update
  - iterative refinement loop

- `dq_dock_engine/docking/formal_surrogates.py`
  - exact/coarse certified family scoring

### Existing modules to modify

- `dq_dock_engine/docking/pipeline.py`
  - certified mode calls the formal optimizer path
  - heuristic mode may keep the current optimizer

- `dq_dock_engine/docking/optimization.py`
  - either remove certified path entirely
  - or retain only heuristic mode here and move certified mode out

- `dq_dock_engine/docking/core.py`
  - add typed dataclasses for belief/action state if needed centrally

## Detailed runtime algorithm

### Certified local refinement round

Input:

- current pose `(t, q)`
- base ligand coordinates
- receptor coordinates/radii
- certified scoring config
- local stencil parameters

Procedure:

1. Build deterministic local action family `A_t`
2. Materialize transformed poses for every action using DSL-backed rigid
   transforms
3. Evaluate certified exact scores `u_exact`
4. Evaluate certified coarse scores `u_coarse`
5. Compute exact/coarse perturbation certificate `delta`
6. Compute survivor set and ambiguity band
7. Form evidence `e_t = survivor_mask`
8. Update posterior:
   - `posterior ∝ prior * likelihood(_, e_t)`
   - with likelihood = survivor indicator
9. Select admissible action:
   - posterior-maximizer
   - tie-break through ambiguity band
   - final deterministic order by `action_id`
10. Apply chosen action to get next pose

Output:

- next pose
- full optimizer state (including masks, posterior, chosen action)

### Certified refinement loop

Run a fixed finite number of rounds:

- `n_rounds` replaces `n_opt_steps`

At each round:

- rebuild the local family around the new current pose
- repeat the certified admissible update rule

Termination:

- fixed rounds
- or early stop only when the no-op action is selected and remains the unique
  admissible maximizer for one full round

That stop condition is deterministic and auditable.

## Mapping to Lean theorem names in metadata/docs

### Theorem hooks to cite directly

- Bayes update:
  - `DecisionQuotient.StochasticSequential.bayesian_update`
- Unique admissible update rule:
  - `DecisionQuotient.bayes_uniquely_admissible`
- Finite sampled action support:
  - `DecisionQuotient.Tractability.SampledDocking.restrictedDecisionProblem`
- Restricted-optimum preservation:
  - `DecisionQuotient.Tractability.SampledDocking.restricted_opt_eq_ambient_slice_of_exists_global_sampled_opt`
- Exact/coarse winner agreement:
  - `DecisionQuotient.Tractability.SampledDockingGap.sampled_epsilon_margin_invariance`
- Survivor certificate:
  - `DecisionQuotient.Tractability.CertifiedPruning.certificate_of_topK_margin`
- Ambiguity band:
  - `DecisionQuotient.Tractability.NearTieBand.exact_topK_subset_ambiguityBand`

### New theorem wrappers likely needed

The current Lean library is close, but two runtime-facing theorem wrappers would
remove handwaving:

1. a theorem stating that support-restriction Bayes conditioning over a certified
   survivor set is admissible under the Paper 4 update discipline
2. a theorem that deterministic tie-breaking over an ambiguity band preserves
   admissibility when the ambiguity band is certified

These are small wrapper theorems around existing objects, not new foundations.

## JAX / DSL implementation details

### DSL primitives already needed by this plan

- `rigidTransform3D`
- `pairwiseDistances3D`
- `typedLennardJonesMatrix`
- `typedLennardJonesCutoff`
- `coulombCutoff`
- `minimumImagePairwiseDistances`
- `upperTriangleMaskedSum`

### Additional simple primitives we may still want

- `normalizeProbabilityVector`
- `indicatorConditioning`
- `argmaxStable`
- `lexicographicTieBreak`

These are not mathematically deep, but adding them to the DSL keeps the
certified-mode runtime free from hidden ad hoc Python logic.

## Validation matrix

### Unit tests

- posterior sums to one
- zero-mass eliminated actions remain zero after normalization
- no-op action is present in every certified action family
- certified mode never calls heuristic optimizer functions
- deterministic tie-breaking is stable across runs

### Equivalence tests

- one-round formal optimizer vs exhaustive exact score over the local family
- survivor-set logic matches Lean-exported theorem assumptions
- ambiguity-band membership matches the threshold semantics

### Integration tests

- `run_docking_pipeline(..., config=CERTIFIED_DOCKING)` no longer imports or
  calls the heuristic optimization step path
- pose refinement still improves or preserves the best certified score across the
  declared action family

## Migration sequence

### Step A - planning/code boundary

- keep `dq_dock_engine/docking/optimization.py` heuristic-only
- create `dq_dock_engine/docking/formal_optimizer.py`

### Step B - deterministic action family

- implement 13-action stencil
- wire DSL rigid transform application

### Step C - exact/coarse certified family scoring

- implement exact/coarse score pair for the same action family
- expose perturbation bound/certificate

### Step D - Bayes support-restriction update

- implement prior/posterior state
- implement survivor-indicator likelihood
- implement deterministic admissible action selection

### Step E - pipeline cutover

- in certified mode, replace local gradient descent with formal optimizer rounds

### Step F - theorem wrapper cleanup

- if runtime/documentation still has handwaving around admissibility of the
  support-restriction update, add the small Lean wrappers named above

## Remaining honest caveat

Even after this work, certified mode will only be as theorem-complete as:

- the declared finite action family
- the exact/coarse bound relating certified surrogate to exact certified score
- the stated prior assumption for the local family

This is still far better than the current heuristic optimizer, because every
remaining assumption becomes:

- explicit
- typed
- serialized
- testable
- and theorem-addressable

## Success condition

Certified mode no longer relies on:

- learning rates
- gradient clipping
- gradient direction normalization
- ad hoc local step caps

and instead performs local refinement as:

- finite declared action family
- theorem-backed certified survivor/ambiguity reasoning
- explicit Bayes conditioning
- deterministic admissible action selection
