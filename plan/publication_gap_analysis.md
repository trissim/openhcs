# Publication Gap Analysis: Current State → Publication-Ready

## Summary

92 Lean proof files, 18 JAX source files. The engine runs and passes basic tests (Verlet energy conservation |ΔH|=3.82e-08). Below is every gap between current state and "most complete and efficient molecular dynamics simulator on Earth" backed by the publication plan.

---

## Layer 1: Lean → JAX Translation Status (Computation)

These Lean files define the core MD algorithms and DSL. Every one must have a JAX counterpart.

| Lean File | JAX File | Status | Notes |
|-----------|----------|--------|-------|
| [Computation/ArrayDSL.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Computation/ArrayDSL.lean) | [physics/kernels.py](file:///home/ts/code/projects/dq-dock-engine/physics/kernels.py) | ✅ Done | DSL primitives: pairwise distances, LJ, cutoff, reduce_sum |
| [Computation/SymplecticIntegrator.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Computation/SymplecticIntegrator.lean) | [physics/engine.py](file:///home/ts/code/projects/dq-dock-engine/physics/engine.py) | ✅ Done | VelocityVerlet, hamiltonian |
| [Computation/LangevinIntegrator.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Computation/LangevinIntegrator.lean) | [physics/engine.py](file:///home/ts/code/projects/dq-dock-engine/physics/engine.py) | ✅ Done | Langevin thermostat with fluctuation-dissipation |
| [Computation/GeometricConstraints.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Computation/GeometricConstraints.lean) | [physics/constraints.py](file:///home/ts/code/projects/dq-dock-engine/physics/constraints.py) | ✅ Done | SHAKE/RATTLE |
| [Computation/LennardJonesDeriv.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Computation/LennardJonesDeriv.lean) | [physics/potentials.py](file:///home/ts/code/projects/dq-dock-engine/physics/potentials.py) | ✅ Done | LJ potential + gradient |
| [Computation/BackwardErrorAnalysis.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Computation/BackwardErrorAnalysis.lean) | — | ❌ **Missing** | Error bounds for symplectic integrators |
| [Computation/BornOppenheimer.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Computation/BornOppenheimer.lean) | — | ❌ **Missing** | Born-Oppenheimer approximation / adiabatic separation |
| [Computation/PhaseSpaceGeometry.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Computation/PhaseSpaceGeometry.lean) | — | ❌ **Missing** | Phase space manifold structure, symplectic form |

---

## Layer 2: Lean → JAX Translation Status (Tractability)

These Lean files define tractability classification — the core innovation behind srank routing.

| Lean File | JAX File | Status | Notes |
|-----------|----------|--------|-------|
| [Tractability/StructuralRank.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/StructuralRank.lean) | [physics/srank.py](file:///home/ts/code/projects/dq-dock-engine/physics/srank.py) | ⚠️ **Approximate** | Uses gradient norm, not decision-relevance (see Approximation 1) |
| [Tractability/EwaldSummation.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/EwaldSummation.lean) | [physics/ewald.py](file:///home/ts/code/projects/dq-dock-engine/physics/ewald.py) | ✅ Done | Real + reciprocal + self correction |
| [Tractability/LatticeSum.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/LatticeSum.lean) | — | ❌ **Missing** | Lattice sum convergence bounds |
| [Tractability/CutoffEpsilon.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/CutoffEpsilon.lean) | [physics/kernels.py](file:///home/ts/code/projects/dq-dock-engine/physics/kernels.py) | ✅ Done | apply_cutoff function |
| [Tractability/MolecularSrank.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/MolecularSrank.lean) | — | ❌ **Missing** | Molecular-specific srank theorems |
| [Tractability/SeparableUtility.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SeparableUtility.lean) | — | ❌ **Missing** | Separable utility detection (router needs this) |
| `Tractability/TreeStructure.lean` | — | ❌ **Missing** | Tree structure detection (router needs this) |
| `Tractability/BoundedActions.lean` | — | ❌ **Missing** | Bounded action enumeration |
| `Tractability/BoundedStateSpace.lean` | — | ❌ **Missing** | State space bounds |
| `Tractability/MultiplicativeSeparable.lean` | — | ❌ **Missing** | Multiplicative separability check |
| `Tractability/FPT.lean` | — | ❌ **Missing** | Fixed-parameter tractability algorithms |
| `Tractability/Dimensional.lean` | — | ❌ **Missing** | Dimensional reduction |
| [Tractability/DiscretizedState.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/DiscretizedState.lean) | — | ❌ **Missing** | State discretization |
| `Tractability/Dominance.lean` | — | ❌ **Missing** | Action dominance detection |
| [Tractability/EpsilonUtilityGap.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/EpsilonUtilityGap.lean) | — | ❌ **Missing** | Utility gap thresholds |
| `Tractability/SingleAction.lean` | — | ❌ **Missing** | Single-action trivial case |
| `Tractability/Tightness.lean` | — | ❌ **Missing** | Tightness bounds for approximations |

---

## Layer 3: Lean → JAX Translation Status (Physics / Thermodynamics)

| Lean File | JAX File | Status | Notes |
|-----------|----------|--------|-------|
| [ThermodynamicLift.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/ThermodynamicLift.lean) | [physics/thermodynamics.py](file:///home/ts/code/projects/dq-dock-engine/physics/thermodynamics.py) | ✅ Done | Landauer constant, thermodynamic cost |
| `Physics/PhysicalCore.lean` | — | ❌ **Missing** | Physical constants, counting gap theorem |
| `Physics/BoundedAcquisition.lean` | — | ❌ **Missing** | Acquisition rate bounds (c/d) |
| `Physics/DecisionTime.lean` | — | ❌ **Missing** | srank × d/c minimum decision time |
| `Physics/DiscreteSpacetime.lean` | — | ❌ **Missing** | Discrete spacetime structure |
| `Physics/HeisenbergStrong.lean` | — | ❌ **Missing** | Heisenberg uncertainty bound |
| `Physics/TransportCost.lean` | — | ❌ **Missing** | Transport cost bounds |
| [Physics/TUR.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Physics/TUR.lean) | — | ❌ **Missing** | Thermodynamic uncertainty relations |
| `Physics/MolecularIntegrity.lean` | — | ❌ **Missing** | Molecular integrity verification |
| `Physics/ConstraintForcing.lean` | — | ❌ **Missing** | Constraint forcing formalization |
| `Physics/TemporalCountingGap.lean` | — | ❌ **Missing** | Temporal counting gap |
| [Physics/LocalityPhysics.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Physics/LocalityPhysics.lean) | — | ❌ **Missing** | Locality constraints |
| `Physics/Uncertainty.lean` | — | ❌ **Missing** | Uncertainty propagation |

---

## Layer 4: Lean → JAX Translation Status (Decision Theory / Stochastic-Sequential)

| Lean File | JAX File | Status | Notes |
|-----------|----------|--------|-------|
| `StochasticSequential/CrossRegime.lean` | — | ❌ **Missing** | Regime transfer conditions |
| `StochasticSequential/SubstrateCost.lean` | — | ❌ **Missing** | Substrate cost κ |
| `StochasticSequential/TemporalIntegrity.lean` | — | ❌ **Missing** | Evidence-monotone retractions |
| `StochasticSequential/TemporalLearning.lean` | — | ❌ **Missing** | Bayesian structure detection |
| `StochasticSequential/ThermodynamicLift.lean` | — | ❌ **Missing** | Stochastic thermodynamic cost |
| `StochasticSequential/Tractability.lean` | — | ❌ **Missing** | Stochastic tractability |

---

## Layer 5: Benchmarking Infrastructure (from Publication Plan)

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| PDBbind loader | `benchmark/loader.py` | ❌ **Missing** | Parse PDBbind refined set (5,316 complexes) |
| Utility matrix | `benchmark/utility.py` | ❌ **Missing** | Build U[a,s] matrix for srank computation |
| AutoDock Vina runner | `benchmark/baselines.py` | ❌ **Missing** | Run Vina, parse output, record timing |
| Results aggregation | `benchmark/results.py` | ❌ **Missing** | Speedup, RMSE, Kendall τ, success rate |
| Figure generation | `benchmark/figures.py` | ❌ **Missing** | srank distribution, speedup, accuracy plots |

---

## Documented Approximations

> [!WARNING]
> These are the only two places where engineering pragmatism diverges from mathematical exactness. Both must be documented in the paper.

### Approximation 1: srank via gradient norm (srank.py)

**Lean defines**: relevance as ∃ states differing only at coordinate i with different optimal action sets.

**JAX implements**: `‖∇_i U‖ > threshold` — gradient being nonzero.

**Gap**: A coordinate could have nonzero gradient everywhere yet never change which action is optimal. Gradient nonzero is necessary but not sufficient for decision-relevance.

**Mitigation**: For molecular docking, gradient and decision-relevance align closely. Document as approximation in paper.

### Approximation 2: Router thresholds (router.py)

**Lean proves**: structural conditions (separable utility, bounded treewidth, tree structure) determine tractability.

**JAX implements**: float ratio thresholds (`ratio < 0.1 → TENSOR_RANK`, `ratio < 0.5 → TREEWIDTH`).

**Gap**: These numbers aren't derived from Lean. The router should check structural conditions, not approximate with float thresholds.

**Fix**: Translate [SeparableUtility.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SeparableUtility.lean), `TreeStructure.lean`, `BoundedActions.lean` to JAX and use structural condition checks in the router.

### Performance Note: neighbors.py

Dense N×N boolean matrix is O(N²) memory. Fine for PDBbind (small complexes), would blow up for large systems. Not a correctness issue — performance issue for future work.

---

## Mechanical Implementation Checklist

### Priority 1: Core Translation Gaps (blocks publication)

- [ ] **1.1** `physics/backward_error.py` ← [BackwardErrorAnalysis.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Computation/BackwardErrorAnalysis.lean) — error bounds for integrators
- [ ] **1.2** `physics/born_oppenheimer.py` ← [BornOppenheimer.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Computation/BornOppenheimer.lean) — adiabatic separation
- [ ] **1.3** `physics/phase_space.py` ← [PhaseSpaceGeometry.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Computation/PhaseSpaceGeometry.lean) — symplectic form, Liouville theorem
- [ ] **1.4** `physics/physical_core.py` ← `Physics/PhysicalCore.lean` — counting gap, constants
- [ ] **1.5** `physics/bounded_acquisition.py` ← `Physics/BoundedAcquisition.lean` — c/d acquisition rate
- [ ] **1.6** `physics/decision_time.py` ← `Physics/DecisionTime.lean` — srank × d/c lower bound

### Priority 2: Tractability Router (replaces magic thresholds)

- [ ] **2.1** `physics/tractability/separable.py` ← [SeparableUtility.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SeparableUtility.lean) — detect separable utility
- [ ] **2.2** `physics/tractability/tree_structure.py` ← `TreeStructure.lean` — detect tree structure
- [ ] **2.3** `physics/tractability/bounded_actions.py` ← `BoundedActions.lean` — bounded action case
- [ ] **2.4** `physics/tractability/molecular_srank.py` ← [MolecularSrank.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/MolecularSrank.lean) — molecular-specific srank
- [ ] **2.5** `physics/tractability/fpt.py` ← `FPT.lean` — fixed-parameter tractability
- [ ] **2.6** Update [pipeline/router.py](file:///home/ts/code/projects/dq-dock-engine/pipeline/router.py) to use structural condition checks instead of float thresholds

### Priority 3: Benchmarking Pipeline (needed for paper results)

- [ ] **3.1** `benchmark/loader.py` — PDBbind refined set loader
- [ ] **3.2** `benchmark/utility.py` — utility matrix construction (BindingState/BindingAction)
- [ ] **3.3** `benchmark/baselines.py` — AutoDock Vina runner + parser
- [ ] **3.4** `benchmark/results.py` — aggregation (speedup, RMSE, Kendall τ)
- [ ] **3.5** `benchmark/figures.py` — matplotlib figure generation

### Priority 4: Stochastic/Sequential Extension (paper claims these)

- [ ] **4.1** `pipeline/cross_regime.py` ← `CrossRegime.lean` — regime transfer
- [ ] **4.2** `pipeline/substrate_cost.py` ← `SubstrateCost.lean` — κ function
- [ ] **4.3** `pipeline/temporal_integrity.py` ← `TemporalIntegrity.lean` — evidence-monotone retractions
- [ ] **4.4** `pipeline/temporal_learning.py` ← `TemporalLearning.lean` — Bayesian structure detection

### Priority 5: Additional Physics (completeness)

- [ ] **5.1** `physics/lattice_sum.py` ← [LatticeSum.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/LatticeSum.lean) — convergence bounds
- [ ] **5.2** `physics/tur.py` ← [TUR.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Physics/TUR.lean) — thermodynamic uncertainty relations
- [ ] **5.3** `physics/heisenberg.py` ← `HeisenbergStrong.lean` — uncertainty bound
- [ ] **5.4** `physics/transport_cost.py` ← `TransportCost.lean` — optimal transport bounds
- [ ] **5.5** `physics/dominance.py` ← `Dominance.lean` — action dominance detection

### Priority 6: Paper Writing

- [ ] **6.1** Fill in benchmark results template in [PUBLICATION_PLAN.md](file:///home/ts/code/projects/dq-dock-engine/plan/PUBLICATION_PLAN.md)
- [ ] **6.2** Generate figures 1-4 (srank distribution, speedup, accuracy, tractability breakdown)
- [ ] **6.3** Generate tables 1-3 (PDBbind stats, speedup by tractability, Vina comparison)
- [ ] **6.4** Document Approximations 1 & 2 in Methods section

---

## Not Required for Publication (from plan)

These Lean files formalize theory that doesn't need JAX translation:

| Category | Count | Examples | Reason |
|----------|-------|---------|--------|
| Hardness proofs | 11 | [SAT.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Hardness/SAT.lean), [QBF.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Hardness/QBF.lean), [ETH.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Hardness/ETH.lean) | Lower bounds — no code to write |
| Reduction theory | 5 | [Reduction.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Reduction.lean), [PolynomialReduction.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/PolynomialReduction.lean) | Theoretical — cited, not implemented |
| Bayes theory | 4 | [BayesFoundations.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/BayesFoundations.lean), [BayesOptimalityProof.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/BayesOptimalityProof.lean) | Theory — cited |
| Statistics | 2 | `FisherInformation.lean`, `ProbabilisticBridge.lean` | Theory — cited |
| Examples | 2 | [ConcreteExamples.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Examples/ConcreteExamples.lean), [PreservationExamples.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Examples/PreservationExamples.lean) | Illustrations |
| Infrastructure | 5 | [CheckAxioms.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/CheckAxioms.lean), [HandleAliases.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/HandleAliases.lean), [Summary.lean](file:///home/ts/code/projects/dq-dock-engine/docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Hardness/Sigma2PExhaustive/Summary.lean) | Lean scaffolding |

---

## Score

| Category | Done | Remaining | % |
|----------|------|-----------|---|
| Computation (7 Lean) | 5 | 3 | 71% |
| Tractability (17 Lean) | 3 | 14 | 18% |
| Physics/Thermo (13 Lean) | 1 | 12 | 8% |
| Stochastic-Sequential (6 Lean for JAX) | 0 | 6 | 0% |
| Benchmarking | 0 | 5 | 0% |
| **Total actionable items** | **9** | **40** | **18%** |
