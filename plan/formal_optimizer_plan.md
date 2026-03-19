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

Status: implemented

### Step B - deterministic action family

- implement 13-action stencil
- wire DSL rigid transform application

Status: implemented in `dq_dock_engine/docking/formal_actions.py`

### Step C - exact/coarse certified family scoring

- implement exact/coarse score pair for the same action family
- expose perturbation bound/certificate

Status: partially implemented

- exact family scoring is implemented via `score_certified_batch`
- coarse family scoring currently aliases exact certified scoring, so `delta = 0`
- this removes heuristic scoring from certified refinement, but does not yet give
  a cheaper certified surrogate/pruning pass

### Step D - Bayes support-restriction update

- implement prior/posterior state
- implement survivor-indicator likelihood
- implement deterministic admissible action selection

Status: implemented in `dq_dock_engine/docking/formal_belief.py` and
`dq_dock_engine/docking/formal_pruning.py`

### Step E - pipeline cutover

- in certified mode, replace local gradient descent with formal optimizer rounds

Status: implemented in `dq_dock_engine/docking/pipeline.py`

### Step F - theorem wrapper cleanup

- if runtime/documentation still has handwaving around admissibility of the
  support-restriction update, add the small Lean wrappers named above

Status: not yet implemented

## Implemented runtime modules

- `dq_dock_engine/docking/formal_actions.py`
  - deterministic 13-action local stencil
  - DSL-backed rigid transform application
- `dq_dock_engine/docking/formal_belief.py`
  - uniform prior
  - survivor-set Bayes conditioning
  - deterministic posterior-based action selection
- `dq_dock_engine/docking/formal_pruning.py`
  - top-k-with-ties mask
  - ambiguity band mask
  - certified survivor mask
- `dq_dock_engine/docking/formal_optimizer.py`
  - round-based certified local refinement over a finite action family
- `dq_dock_engine/docking/pipeline.py`
  - certified mode now routes local refinement through the formal optimizer

## Validation completed

- `dq_dock_engine/tests/test_formal_optimizer.py`
  - action family determinism
  - posterior normalization
  - certified local action selection
  - certified pipeline avoids heuristic optimizer path
- benchmark smoke test
  - `1hvr`, `500` poses, `3` certified refinement rounds
  - RMSD `0.40A`
  - benchmark completed successfully end-to-end

## DSL lowering follow-up

The optimizer implementation should keep shrinking the amount of handwritten
Python policy math. To that end, the ArrayDSL bridge now needs to own the core
belief-update algebra as well, not just geometry and pairwise physics.

Required bridge primitives:

- `supportConditioning`
- `normalizeProbabilityVector`

These make the posterior update mechanically Lean-exported/JAX-callable even if
the higher-level action-family orchestration remains in Python.

## Remaining proof/completeness gaps

1. **Coarse certified surrogate is still degenerate**
   - exact and coarse scores are currently identical in certified refinement
   - this is rigorous but does not yet exploit the sampled-docking gap/pruning
     theorems for cheaper selection

2. **Prior choice is explicit, not yet theorem-wrapped**
   - current prior is uniform over the declared finite action family
   - acceptable as an explicit assumption, but not yet discharged by a Lean
     wrapper theorem

3. **Admissibility wrapper theorem still missing**
   - we still need a small Lean theorem that support-restriction conditioning over
     a certified survivor set is an admissible update in the Paper 4 sense

4. **Certified sampling outside local refinement is still heuristic**
   - certified mode still enters the local optimizer from a heuristic sampled pose
     family upstream in the pipeline

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

## Iteration 2 - Closing the four remaining gaps

This section replaces the previous vague follow-up list with explicit work
packages, theorem hooks, runtime objects, and acceptance criteria.

### Gap 1 - Coarse certified surrogate is still degenerate

#### Problem

The current formal optimizer sets:

- `u_exact = certified LJ score`
- `u_coarse = u_exact`
- `delta = 0`

This is rigorous but useless for tractability. It does not exercise:

- `CoarseApproximation.UniformUtilityApprox`
- `SampledDockingGap.sampled_epsilon_margin_invariance`
- `CertifiedPruning.certificate_of_topK_margin`
- `TopKPreservation.exact_topK_subset_survivorSet_of_margin`

#### Implementation target

Define a genuine **certified coarse local surrogate** over the same local action
family. The first version should be cheaper than exact scoring while preserving a
uniform certified discrepancy bound.

#### Recommended surrogate

Use a receptor-trimmed certified score:

- `u_exact(a)`
  - certified LJ against the full retained receptor atom set
- `u_coarse(a)`
  - certified LJ against a theorem-backed trimmed receptor subset
  - e.g. atoms inside a stricter local relevance shell, or a capped local
    top-contact subset determined deterministically from the current pose center

#### Why this is the right next step

- stays within certified LJ scoring
- avoids reintroducing heuristic scorers
- creates a real exact/coarse pair
- naturally supports uniform discrepancy bounds via explicit omitted-interaction
  tail accounting

#### Lean hooks

- `CoarseApproximation.finiteUniformErrorRadius_witnesses_uniformApprox`
- `CoarseApproximation.uniform_approx_implies_opt_invariance`
- `CertifiedPruning.certificate_of_topK_margin`
- `TopKPreservation.exact_topK_subset_survivorSet_of_margin`
- `RankingPreservation.pairwise_order_preserved_of_uniform_error`
- `NearTieBand.exact_topK_subset_ambiguityBand`

#### Runtime design

Add:

```python
@dataclass(frozen=True)
class CertifiedCoarseScoreBundle:
    exact_scores: jax.Array
    coarse_scores: jax.Array
    delta: float
    survivor_mask: jax.Array
    ambiguity_mask: jax.Array
```

New module responsibilities:

- `dq_dock_engine/docking/formal_surrogates.py`
  - `select_trimmed_receptor_subset(...)`
  - `score_exact_local_family(...)`
  - `score_coarse_local_family(...)`
  - `certified_uniform_delta(...)`

#### Acceptance criteria

- `u_coarse` differs from `u_exact` on real examples
- `delta` is explicit and nonnegative
- survivor set is derived from the theorem-backed margin conditions, not from ad
  hoc masking
- unit tests cover exact/coarse agreement on easy cases and nontrivial pruning on
  realistic cases

### Gap 2 - Prior choice is explicit, not yet theorem-wrapped

#### Problem

The current local optimizer uses a uniform prior over the finite local action
family. That is explicit and deterministic, but it is still just an assumption.

#### Implementation target

Promote the local prior to a typed, declared object with an explicit semantic
contract and theorem wrapper.

#### Recommended first semantics

Use a **symmetry-respecting local prior**:

- no-op action has declared prior mass `p_noop`
- all non-noop actions share the remaining mass equally
- default first pass may still set all masses uniformly, but the runtime object
  must make the assumption inspectable

#### Runtime design

Add:

```python
@dataclass(frozen=True)
class CertifiedPriorSpec:
    kind: Literal["uniform", "noop_biased"]
    noop_mass: float
```

Rules:

- every certified optimizer run serializes the chosen prior spec
- prior construction is a pure function of `CertifiedPriorSpec` and the action
  family size
- prior validity is checked fail-loud (`sum=1`, all masses nonnegative)

#### Lean theorem wrapper needed

Add a small wrapper theorem stating that, for a declared finite support family,
Bayesian conditioning from any valid prior over that family remains inside the
admissible update discipline as long as the likelihood/evidence channel is the
certified survivor event.

This does **not** need to prove uniform is uniquely correct; it needs to prove
the update is admissible once the prior is declared.

#### Acceptance criteria

- prior object appears in optimizer state / report artifacts
- no implicit prior construction remains in code
- Lean wrapper exists, or at minimum the runtime is organized so the theorem can
  refer to the exact Python-side object semantics

### Gap 3 - Admissibility wrapper theorem still missing

#### Problem

The runtime already does support-restriction Bayes conditioning, but the exact
paper-to-runtime theorem bridge is not yet stated.

#### Required theorem wrappers

We need two small, high-value wrappers:

1. **Certified survivor conditioning is admissible**
   - If a finite action family carries a declared prior and the evidence event is
     a certified survivor set produced by the exact/coarse margin theorem, then
     the posterior update is an admissible update rule in the Paper 4 sense.

2. **Deterministic tie-break inside certified ambiguity band is admissible**
   - If the exact top-k lies inside the ambiguity band, then selecting the first
     action under a fixed deterministic order remains a valid admissible report /
     action-selection refinement.

#### Likely Lean home

Recommended file:

- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/FormalLocalOptimizer.lean`

Imports:

- `BayesFromDQ`
- `TemporalLearning`
- `SampledDocking`
- `SampledDockingGap`
- `CertifiedPruning`
- `NearTieBand`
- `IntegrityCompetence`

#### Runtime impact

Once these wrappers exist, the Python code should attach theorem handles to:

- posterior update
- survivor certificate construction
- deterministic action selection

#### Acceptance criteria

- theorem names exist and are exported through handle aliases
- optimizer module docstrings cite the exact theorem handles
- no remaining comment says “admissible in spirit” or similar handwaving

### Gap 4 - Certified sampling outside local refinement is still heuristic

#### Problem

The formal optimizer currently refines poses that were obtained from upstream
heuristic pocket-guided sampling.

#### Scope split

There are two distinct layers:

- **global proposal generation**
- **local certified refinement**

The current work fully addresses the second layer only.

#### Recommended next certified replacement

Introduce a deterministic **coarse certified global action family** over the
docking box:

- finite translation lattice over the box center/extent
- finite quaternion dictionary
- deterministic enumeration order

This can reuse the same sampled-docking finite support semantics as the local
optimizer.

#### Runtime design

Add:

- `dq_dock_engine/docking/formal_sampling.py`
  - box lattice generation
  - deterministic quaternion dictionary
  - global action family packaging

Then certified pipeline becomes:

1. certified global finite support
2. theorem-backed certified pruning / top-k retention
3. certified local optimizer over deterministic local action families

#### Why not random Sobol yet

Low-discrepancy sequences are promising, but the existing Lean library already
has a more immediate bridge via finite sampled supports. Deterministic finite
enumeration is easier to make theorem-faithful right now.

#### Acceptance criteria

- certified mode no longer calls `sample_intelligent_poses`
- certified mode no longer uses `SamplingStrategy.HYBRID`
- certified global proposals are serialized as finite action-family metadata

## Ordered next implementation steps

### Workstream A - non-degenerate certified coarse surrogate

1. create `formal_surrogates.py`
2. implement trimmed-receptor exact/coarse family scoring
3. derive survivor mask from theorem-style `delta`
4. add tests showing `u_exact != u_coarse` on realistic inputs

### Workstream B - explicit prior object

1. add `CertifiedPriorSpec`
2. replace hardcoded `uniform_prior(...)` calls with typed prior construction
3. emit prior metadata in optimizer state / benchmark artifacts

### Workstream C - theorem wrapper prep

1. create `FormalLocalOptimizer.lean` scaffold
2. state survivor-conditioning admissibility theorem
3. state deterministic tie-break admissibility theorem
4. export handles through `HandleAliases.lean`

### Workstream D - certified global sampler

1. create `formal_sampling.py`
2. deterministic translation lattice + quaternion dictionary
3. certified mode pipeline bypasses heuristic pocket-guided sampling

## Exit condition for the next pass

The next iteration should be considered complete only when:

- exact/coarse certified local refinement is nontrivial
- prior is explicit and serialized
- theorem wrapper file exists for local optimizer admissibility
- certified mode no longer depends on heuristic global sampling
