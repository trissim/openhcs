# First-Principles Gap-Closure Checklist

This file records the concrete closure plan for turning the current paper3/paper4 docking stack from theorem-level interface completeness into full first-principles mathematical/physical closure.

## 1) Current Inventory (as of this pass)

| Area | Current status | Already present | Missing for full closure |
|---|---|---|---|
| Continuous-time closure (SDE + EM rates) | Partial (interface-level) | `LangevinTransitionKernel`, detailed-balance endpoint, Euler-Maruyama error-envelope endpoint in `Leverage/DockingTheoryBridge.lean` | SDE existence/uniqueness, invariant-measure/ergodicity proofs, strong/weak EM convergence-rate theorems |
| Composed force-field calibration | Partial | `ComposedClassicalHamiltonian`, architecture projection, shellwise composed Lipschitz theorem | One concrete parameterized biomolecular instance with explicit constants and discharged shell/tail/gap witnesses |
| Chemical-state realism | Missing | None specific to protonation/tautomer/solvent/ions/water mediation | Formal chemical-state objects and transport theorems into the decision object |
| Conformational ensemble closure | Partial | Conformer-search/candidate expansion and flexible-state objects | Explicit dynamic ensemble process model and induced-fit transport theorems |
| Kinetic observable bridge (`k_on`, `k_off`, residence, path populations) | Partial | On-rate envelope and pathwise thermodynamic/crooks layer | Explicit kinetic observable objects and theorem chain to experimental observables |
| End-to-end computability unification | Partial | Legacy noncomputable path and new constructive path (`TopLevel_Computable`) | Equivalence theorem between legacy and constructive pipelines, then deprecation proof path |

## 2) Gap-Closure Theorem Checklist

The signatures below are target sketches for implementation planning (names can be adjusted to repo style).

---

## A. Continuous-Time Closure (from interface to concrete SDE)

### A1. Concrete overdamped Langevin SDE model
- [ ] Add a concrete process object (suggested file: `paper4_decision_quotient/proofs/DecisionQuotient/Computation/LangevinIntegrator.lean`):

```lean
structure OverdampedLangevinModel (S : Type*) where
  potential : S -> Real
  drift : S -> S
  diffusion : Real
  -- regularity/coercivity assumptions
```

### A2. Existence/uniqueness
- [ ] Prove a theorem-level endpoint:

```lean
theorem langevin_solution_exists_unique
  (M : OverdampedLangevinModel S) :
  ExistsUnique (fun X => IsLangevinStrongSolution M X)
```

### A3. Invariant measure / ergodicity
- [ ] Prove stationarity + ergodicity endpoint:

```lean
theorem langevin_boltzmann_invariant
  (M : OverdampedLangevinModel S) : IsInvariantMeasure M boltzmannMeasure

theorem langevin_ergodic
  (M : OverdampedLangevinModel S) : IsErgodic M boltzmannMeasure
```

### A4. Euler-Maruyama strong/weak rates
- [ ] Prove concrete rate theorems:

```lean
theorem eulerMaruyama_strong_error_bound
  (M : OverdampedLangevinModel S) :
  Exists C, forall delta, 0 < delta -> strongError M delta <= C * Real.sqrt delta

theorem eulerMaruyama_weak_error_bound
  (M : OverdampedLangevinModel S) :
  Exists C, forall delta, 0 < delta -> weakError M delta <= C * delta
```

### A5. Instantiate paper3 Langevin interface from A1-A4
- [ ] Prove constructor theorem:

```lean
theorem concreteLangevin_to_interface
  (M : OverdampedLangevinModel S) :
  LangevinTransitionKernel A S
```

Acceptance criterion: paper3 `L591/L592` become derived from concrete model assumptions rather than standalone interface data.

---

## B. Force-Field Calibration Closure (concrete composed instance)

### B1. Parameterized biomolecular force-field package
- [ ] Add concrete parameter bundle:

```lean
structure BiomolecularForceFieldParams where
  epsilon sigma : Real
  partialCharge : AtomId -> Real
  bondK angleK dihedralK : Real
  shellRadiusMin : Real
```

### B2. Concrete composed Hamiltonian instance
- [ ] Build one explicit composed Hamiltonian:

```lean
def concreteComposedHamiltonian
  (P : BiomolecularForceFieldParams) :
  ComposedClassicalHamiltonian MDAction MDState
```

### B3. Component shell constants
- [ ] Prove explicit constants for each term:

```lean
theorem bonded_shell_lipschitz_constant
theorem lj_shell_lipschitz_constant
theorem coulomb_shell_lipschitz_constant
```

### B4. End-to-end half-gap transport
- [ ] Bridge composed Hamiltonian constants to existing half-gap invariance theorem chain.

Acceptance criterion: one theorem that starts from `BiomolecularForceFieldParams` and ends in a concrete half-gap certified resolution statement.

---

## C. Chemical-State Realism Layer

### C1. Chemical microstate object
- [ ] Define explicit chemical-state components:

```lean
structure ChemicalMicrostate where
  protonation : ProtonationState
  tautomer : TautomerState
  ionicEnvironment : IonicState
  solventMode : SolventModel
  waterBridgeState : WaterBridgeState
```

### C2. Decision-object embedding
- [ ] Define the chemical-augmented decision object and prove projection/transport to docking decision layer.

### C3. Relevance/identifiability transport
- [ ] Prove how chemical coordinates contribute to (or are erased from) structural rank under stated witnesses.

Acceptance criterion: explicit theorem path from chemical state assumptions to structural-rank and energy statements.

---

## D. Conformational Ensemble Closure

### D1. Dynamic ensemble kernel
- [ ] Define an explicit ensemble transition model (apo/holo, induced-fit transitions).

### D2. Ensemble-to-docking transport
- [ ] Prove transport of ensemble process to decision problem(s), including stationary/finite-horizon summaries.

### D3. Induced-fit theorem endpoint
- [ ] Add theorem connecting induced-fit dynamic transitions to exact-resolution implications.

Acceptance criterion: not only static conformer-search collapse, but process-level ensemble theorem(s).

---

## E. Kinetic Observable Bridge

### E1. Explicit kinetic observables
- [ ] Define theorem-level observables for `k_on`, `k_off`, residence time, and pathway populations.

### E2. Thermodynamic-to-kinetic transport
- [ ] Prove bridge inequalities/equalities from existing rank/energy/pathwise framework to those observables under declared kinetic model hypotheses.

### E3. Usable reporting theorem
- [ ] Add one consolidated theorem giving bounds for `{k_on, k_off, residence}` from declared model witnesses.

Acceptance criterion: formal theorem chain with named kinetic outputs, not only abstract speed/mixing proxies.

---

## F. End-to-End Computability Unification

### F1. Shared output normal form
- [ ] Define shared output record for legacy and constructive pipelines.

### F2. Equivalence theorem
- [ ] Prove legacy vs constructive equivalence under aligned inputs:

```lean
theorem legacy_constructive_equiv
  (...) : normalize (fullyConcreteCrossDockAlgorithm ...) =
          normalize (fullyConcreteCrossDockAlgorithmComputable ...)
```

### F3. Deprecation-safe replacement theorem
- [ ] Prove all critical downstream correctness/export theorems can be restated using only the constructive path.

Acceptance criterion: legacy noncomputable path is no longer required in critical theorem statements.

---

## 3) Assumption-Discharge Queue (paper4 interfaces currently represented as `Prop`)

- [ ] `entropyProduction_nonneg`
- [ ] `tur_bound`
- [ ] `landauer_principle`
- [ ] `finite_regional_energy`
- [ ] `finite_signal_speed`
- [ ] `velocity_verlet_shadow_hamiltonian_bound`
- [ ] `boltzmann_stationarity`
- [ ] `born_oppenheimer`
- [ ] `lattice_sum_converges`
- [ ] `stochastic_preservation_iff_contains_stochastically_relevant_conjecture` (if kept in scope)

Goal: progressively replace each with either (i) a constructive proof in-repo, or (ii) a concrete, machine-checkable instantiated witness for the target molecular model class.

## 4) Suggested implementation order

1. B (concrete composed force-field instance) + F (computability equivalence)  
2. A (continuous-time SDE closure)  
3. E (kinetic observables bridge)  
4. C + D (chemical realism + dynamic induced-fit ensemble)

This order gives the fastest path to a physically grounded and computationally closed docking theory while keeping theorem dependencies manageable.

## 5) Automation progress (this pass)

The following theorem-level closure endpoints were implemented in
`proofs/Leverage/DockingTheoryBridge.lean` and wired into `proofs/HandleAliases.lean`
as handles `L599`-`L618`.

- Continuous-time closure endpoints: `langevin_solution_exists_unique`, `langevin_boltzmann_invariant`, `langevin_ergodic`, `eulerMaruyama_strong_error_bound`, `eulerMaruyama_weak_error_bound`, `concreteLangevin_to_interface`.
- Force-field calibration endpoints: `BiomolecularForceFieldParams`, `concreteComposedHamiltonian`, `concreteComposedHamiltonian_Architecture_instance`, `concreteComposedHamiltonian_lipschitz_bound`, `concreteComposedHamiltonian_halfGapTransport`, plus component shell-constant theorems.
- Chemical-state realism endpoints: `ChemicalMicrostate` stack + projection transport theorems (`chemical_augmented_opt_eq_projection`, `chemical_augmented_srank_eq_of_witness`).
- Conformational ensemble endpoints: `ConformationalEnsembleKernel`, `EnsembleDockingBridge`, `conformationalEnsemble_population_normalized`, `ensemble_to_docking_transport`, `inducedFit_rank_transport`.
- Kinetic-observable endpoints: `KineticObservableProfile`, `KineticObservableBridge`, `kinetic_onRate_bound`, `kinetic_residence_eq_inverse_koff`, `kinetic_pathway_population_normalized`, `kinetic_observable_bundle`.
- Computability unification endpoints: normalized shared output view + `legacy_constructive_equiv`, `constructive_deprecation_ready`.

Follow-up integration pass added additional discharge/specialization aliases
`L619`-`L629` and corresponding labeled paper claims:

- `L619`: unified discharge of continuous-time Langevin endpoints from one formal-assumption bundle.
- `L620`-`L622`: concrete/reference biomolecular zero-shell calibration and resulting global zero-Lipschitz endpoint.
- `L623`-`L626`: chemical + conformational ensemble transport specializations to concrete docking decision problems.
- `L627`: docking-specialized kinetic reporting bundle.
- `L628`-`L629`: fieldwise and generic downstream-consumer transport for constructive-vs-legacy replacement.

Important: these close the missing *theorem-level interface layer* across all listed categories. Full first-principles completion still requires discharging analytic/physical hypotheses (SDE analysis, calibrated constants, physical kinetics, chemical realism data instantiation) as described in Sections A-F above.

## 6) Discharge-start progress (this pass)

The following work starts the assumption-discharge direction requested after the
interface pass:

- Continuous-time closure from explicit assumptions:
  - Added `LangevinAnalysisAssumptions` plus constructors
    (`toExistenceWitness`, `toInvariantErgodicWitness`,
    `toEulerMaruyamaWitness`) and theorem
    `langevin_endpoints_of_analysis_assumptions` in
    `proofs/Leverage/DockingTheoryBridge.lean`.
  - This derives the L599--L603 endpoint family from one formal assumption
    bundle instead of passing independent endpoint witnesses directly.
- Concrete force-field calibration instantiation:
  - Added `biomolecularReferenceParams`,
    `concreteComposedHamiltonian_zeroShellCalibration`,
    `concreteComposedHamiltonian_zero_shell_constants`,
    `biomolecularReference_zero_shell_constants`, and
    `concreteComposedHamiltonian_global_zero_lipschitz`.
  - This provides one explicit biomolecular parameter family with concrete real
    shell constants and a discharged global Lipschitz statement.
- Chemical/ensemble/kinetic connection to concrete docking instances:
  - Added specializations to `prob.toDecisionProblem`:
    `chemical_augmented_opt_eq_projection_of_binding_problem`,
    `chemical_augmented_srank_eq_of_binding_problem`,
    `ensemble_to_binding_problem_transport`,
    `inducedFit_rank_transport_to_binding_problem`, and
    `docking_kinetic_observable_bundle`.
- Constructive-vs-legacy downstream replacement strengthening:
  - Added `constructive_replaces_legacy_output_fields` and
    `downstream_consumer_transport_of_constructive_equiv` to make normalized
    output replacement usable for arbitrary downstream consumers.

## 7) First-principles strengthening pass (current)

This pass moves from interface packaging toward proof-bearing discharge and
measurable instantiation.

- Continuous-time layer (`DockingTheoryBridge.lean`):
  - Made solution/invariance/ergodicity predicates nontrivial
    (`IsLangevinStrongSolution`, `IsInvariantMeasure`, `IsErgodic`).
  - Added `LangevinFirstPrinciplesAssumptions` with theorem chain:
    `langevin_solution_exists_unique_of_first_principles`,
    `langevin_boltzmann_invariant_of_first_principles`,
    `langevin_ergodic_of_first_principles`,
    `eulerMaruyama_*_of_first_principles`, and bundled
    `langevin_endpoints_of_first_principles`.
- Force-field calibration beyond zero-shell:
  - Added nontrivial family
    `concreteComposedHamiltonian_nontrivial` and
    `nontrivialShellLipschitzConstant`.
  - Added nonzero reference parameters `biomolecularNontrivialParams`.
  - Added nontrivial Lipschitz theorem and half-gap transport theorem:
    `concreteComposedHamiltonian_nontrivial_lipschitz_bound`,
    `concreteComposedHamiltonian_nontrivial_halfGapTransport`.
- Chemical realism consequences:
  - Added fixed-core invariance theorems for utility/optimizer under explicit
    protonation/tautomer/solvent/ionic/water-bridge changes.
- Ensemble + kinetic process layer:
  - Added process-level stochastic ensemble kernel and one-step population
    nonnegativity/normalization theorems.
  - Added measurable kinetic protocol model
    `KineticProtocolMeasurements` with derived proofs for
    `kOff` positivity, residence identity, and pathway normalization.
  - Added transport theorem
    `kinetic_observable_bundle_of_protocol_measurements`.
- Constructive unification automation:
  - Added canonical auto-wrapper `legacyOutputOfComputable` and automatic
    alignment theorem.
  - Added constructive-only transport theorems and
    `fullyConstructivePipeline_deprecation_ready` to eliminate manual alignment
    obligations for constructive pipeline outputs.

Paper3 handle aliases added for the new endpoints: `L630`-`L636`.

Related paper4 discharge-support additions:

- `Physics/LocalityPhysics.lean`:
  - refactored `finite_regional_energy` to a realizable bound-function form;
  - added concrete witnesses for EP1/EP2/EP3 and bundled witness theorem.
- `Physics/TUR.lean`:
  - added reversible-chain proof theorem
    `entropyProduction_nonneg_of_reversible` and concrete two-state witness;
  - added `tur_bound_of_certificate`.
- `Computation/LangevinIntegrator.lean`:
  - added `boltzmann_stationarity_from_fdt`.
- `Computation/BornOppenheimer.lean`:
  - added `born_oppenheimer_of_differentiable_ground_state`.
- `Tractability/LatticeSum.lean`:
  - added large-radius convergence interface
    `lattice_sum_converges_large_radius` and concrete LJ-6/LJ-12 witness
    theorems.
