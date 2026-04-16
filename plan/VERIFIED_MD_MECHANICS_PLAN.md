# Verified MD Mechanics: Formalization & JAX Implementation Plan

## Executive Summary

This plan outlines the mechanical roadmap for formalizing Molecular Dynamics (MD) within the **Decision Quotient (DQ)** framework. It bridges the gap between the high-level theoretical proofs in Lean (`StructuralRank.lean`, `ThermodynamicLift.lean`) and the high-performance implementation in JAX. 

The goal is to produce a **machine-checked specification** of MD that proves the existence of a "Decision-Theoretic Backdoor"—where the efficiency of docking simulations is not just an empirical observation, but a mathematical necessity derived from the **Structural Rank (srank)** of molecular interactions.

---

## 1. The Formalization Bridge: MD as a Decision Problem

To use Lean to guide JAX, we must first map the physical primitives of MD to the abstract primitives of the Decision Quotient.

| MD Primitive | DQ Formalization (Lean) | JAX Representation |
|--------------|-------------------------|--------------------|
| **Phase Space** | `CoordinateSpace S n` (Tangent Bundle) | `jnp.ndarray` (q, p) |
| **Potential Energy** | `UtilityFunction U : S → ℝ` | `@jit energy_fn` |
| **Integrator Map** | `DiscreteStep Φ : S → S` | `@jit step_fn` |
| **Constraint** | `Submanifold M ⊂ S` | `Projection Operator` |
| **Srank** | `card {relevant_atoms}` | `Sparsity Pattern` |

### 1.1 The Central Hypothesis to Prove
**"For any potential $U$ with finite cutoff $r_c$, the Structural Rank of the binding decision $D$ is bounded by the local density of the interaction graph, placing the docking problem in the Tractable Regime."**

---

## 2. Mechanized Proof Architecture (Lean 4)

We divide the formalization into four mechanical layers. Each layer must be machine-checked to provide the "proof-carrying" foundation for the JAX engine.

### Layer 1: Geometric Foundations (`Physics/SymplecticGeometry.lean`)
*Current Gap: Mathlib4 lacks explicit Poisson/Symplectic algebra for MD.*
- **Mechanics:** Define the Symplectic Form $\omega$ on $\mathbb{R}^{2n}$.
- **Theorem:** Prove that the Hamiltonian vector field $X_H$ satisfies $i_{X_H}\omega = dH$.
- **Theorem:** Prove Liouville's Theorem (Volume preservation of the flow).

### Layer 2: The MD DSL & Executable Spec (`Computation/ArrayDSL.lean`)
*Mechanical Strategy: Create a restricted DSL that Lean can reason about and JAX can execute.*
- **Mechanics:** Define a set of "Verified Primitives": `Map`, `Reduce`, `Scatter`, `Gather`, `Norm`.
- **Reasoning:** Prove that these primitives preserve the properties of the underlying real-valued functions (e.g., the derivative of a `Sum` of `Squares` is a `Scaled Identity`).
- **Bridge:** This DSL serves as the "Common Language" between Lean and JAX.

### Layer 3: Integrator Symplecticity (`Physics/IntegratorRigor.lean`)
- **Theorem:** Define the **Störmer-Verlet** map $\Phi_h$ in the DSL.
- **Mechanical Proof:** Prove that $\Phi_h$ is a symplectic map (preserves $\omega$).
- **Mechanical Proof:** Prove the **Backward Error Analysis (BEA)**: $\Phi_h$ is the exact flow of a "Shadow Hamiltonian" $\tilde{H}$.

### Layer 4: srank-Sparsity Proof (`Tractability/MolecularSrank.lean`)
- **Mechanics:** Formulate the Binding Decision as finding $\text{argmax}_a U(a, s)$.
- **Theorem:** Prove that if $U$ is a sum of pair potentials with cutoff $r_c$, then `isRelevant(atom_i)` is false if the distance to the binding site exceeds $r_c$.
- **Result:** This yields the **Machine-Checked srank bound** for docking.

---

## 4. JAX Implementation Strategy (The "Mechanical Translation")

The JAX implementation follows the "Executable Spec" from Lean Layer 2.

### 4.1 DSL Transcription
Instead of writing raw Python, we implement the MD kernels using the "Verified Primitives":

```python
# JAX Implementation following ArrayDSL.lean spec
def verified_force_kernel(q, params):
    # This structure must match the Lean DSL exactly
    distances = dsl.compute_distances(q) 
    relevant_mask = dsl.apply_cutoff(distances, params.rc)
    energy = dsl.sum_interactions(relevant_mask, params.epsilon, params.sigma)
    return jax.grad(energy)(q)
```

### 4.2 Structural Rank Routing
The JAX engine uses the results from Layer 4 to skip computations:

```python
def route_by_srank(protein, ligand):
    srank_bound = lean_exported_srank_theorem(protein, ligand)
    if srank_bound < TENSOR_RANK_THRESHOLD:
        return execute_low_rank_docking(protein, ligand)
    else:
        return execute_standard_md(protein, ligand)
```

---

## 5. Detailed Formalization Roadmap

### Phase 1: Mechanical Primitives
- [ ] **File:** `Physics/Basic.lean` - Define masses, velocities, and potentials as typed objects.
- [ ] **File:** `Physics/Hamiltonian.lean` - Define the Hamiltonian and prove conservation of energy in the continuous limit.
- [ ] **File:** `Tractability/MolecularSrank.lean` - Define the `isRelevant` predicate for atomic coordinates.

### Phase 2: Integrator Verification
- [ ] **File:** `Physics/Verlet.lean` - Formally define the Velocity Verlet algorithm.
- [ ] **Proof:** Prove second-order accuracy $O(dt^2)$ relative to the `Hamiltonian.lean` flow.
- [ ] **Proof:** Prove energy stability (boundedness of the Shadow Hamiltonian).

### Phase 3: The JAX Bridge
- [ ] **File:** `Computation/JAXBridge.lean` - Define the mapping from Lean `List ℝ` to JAX `f32[N]`.
- [ ] **Artifact:** Export a JSON specification of the verified integrator for the JAX test harness.

### Phase 4: Thermodynamic Validation
- [ ] **File:** `Physics/ThermodynamicMD.lean` - Link `ThermodynamicLift.lean` to the bit-count of an MD step.
- [ ] **Theorem:** Prove that the MD step respects the Landauer Floor calculated in Paper 4.

---

## 6. Verification & Validation Protocol

### 6.1 Mechanical Verification (Lean)
Every theorem in the roadmap must be proven without `admit`.
- Run `lake build` to verify the entire chain from `Mathlib` -> `DQ` -> `MD Mechanics`.

### 6.2 Implementation Validation (JAX)
- **Invariant Checking:** Use `jax.checkify` to assert that the Symplectic form is preserved within $\epsilon$ at every step.
- **Shadow Energy Test:** Monitor the Shadow Hamiltonian $\tilde{H}$ during long runs to ensure it remains bounded, as proven in Layer 3.
- **srank Consistency:** Cross-check the JAX-computed sparsity pattern against the Lean-predicted `relevantSet`.

---

## 7. Future Work: Mechanical Generalization
- **Langevin Dynamics:** Formalize the SDE-to-ODE mapping for stochastic thermostats.
- **Quantum-to-Classical:** Formalize the Born-Oppenheimer approximation as a "Lossy Quotient" from Paper 4.
- **Learned Potentials:** Prove that neural potentials (GNNs) preserve the necessary srank properties for tractability.
