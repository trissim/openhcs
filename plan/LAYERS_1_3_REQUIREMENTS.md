# Layers 1 & 3: Complete Formalization Requirements

## Executive Summary

**Layers 1 and 3 require contributing to Mathlib4** — these are research-grade formalizations that go beyond the current Decision Quotient framework. They involve symplectic geometry, Hamiltonian mechanics, and numerical analysis on manifolds.

**Estimated effort:** 6-12 months for a Lean expert, or requires collaboration with Mathlib community.

---

## Layer 1: Geometric Foundations (`Physics/SymplecticGeometry.lean`)

### 1.1 What Needs to Be Defined

#### **A. Symplectic Vector Space**
```lean
/-- Symplectic form on ℝ²ⁿ -/
structure SymplecticSpace (n : ℕ) where
  phaseSpace : Type := ℝ ^ (2*n)
  omega : (phaseSpace × phaseSpace) → ℝ
  nondegenerate : ∀ v, (∀ w, omega v w = 0) → v = 0
  closed : ∀ v w x, omega v (omega w x) + omega w (omega x v) + omega x (omega v w) = 0
```

**Required Theorems:**
- ✗ `omega_bilinear`: ω is bilinear
- ✗ `omega_antisymmetric`: ω(v,w) = -ω(w,v)
- ✗ `omega_nondegenerate`: Definition of nondegeneracy
- ✗ `dimension_theorem`: 2n is even (by construction)

**Trouble Level:** ⭐⭐⭐ (3/5) - Standard linear algebra, but requires careful setup

---

#### **B. Hamiltonian Vector Fields**
```lean
/-- Hamiltonian vector field X_H defined by i_XH ω = dH -/
def hamiltonianVectorField (H : PhaseSpace → ℝ) : PhaseSpace → PhaseSpace :=
  fun p =>
    let (q, p) := splitCoords p  -- q = positions, p = momenta
    let dH := grad H p
    (-dH.momentum, dH.position)  -- Standard canonical form
```

**Required Theorems:**
- ✗ `hamiltonian_vector_field_well_defined`: X_H is properly a vector field
- ✗ `canonical_omega`: Standard ω = Σ dq_i ∧ dp_i
- ✗ `i_XH_omega_eq_dH`: i_XH ω = dH (fundamental theorem)
- ✗ `X_H_preserves_H`: H is constant along X_H flow (energy conservation)

**Trouble Level:** ⭐⭐⭐⭐ (4/5) - **BLOCKER: Poisson brackets not in Mathlib**

---

#### **C. Poisson Bracket (The Hard Part)**
```lean
/-- Poisson bracket {F, G} = ω(X_F, X_G) -/
def poissonBracket (F G : PhaseSpace → ℝ) : PhaseSpace → ℝ :=
  fun p => omega p (hamiltonianVectorField F p) (hamiltonianVectorField G p)

-- Canonical form: {F, G} = Σ (∂F/∂q_i)(∂G/∂p_i) - (∂F/∂p_i)(∂G/∂q_i)
```

**Required Theorems:**
- ✗ `poisson_bracket_canonical`: Matches classical definition
- ✗ `poisson_bracket_antisymmetric`: {F, G} = -{G, F}
- ✗ `poisson_bracket_jacobi_identity`: {{F, G}, H} + {{G, H}, F} + {{H, F}, G} = 0
- ✗ `hamiltonian_equations`: dq/dt = {q, H}, dp/dt = {p, H}

**Trouble Level:** ⭐⭐⭐⭐⭐ (5/5) - **CRITICAL BLOCKER: Poisson algebra not in Mathlib**

---

### 1.2 What's Missing in Mathlib4

According to `deep-research-report.md` line 23:
> "the library explicitly flags that **Poisson algebras are not yet defined**, which is directly on the critical path"

**Required Mathlib Additions:**
```lean
# Missing structures:
- Poisson algebra (algebraic structure)
- Poisson bracket as bilinear operation
- Lie bracket properties for Poisson bracket
- Symplectic linear algebra (currently only symplectic group exists)
```

**Action Items:**
1. **Option A (Recommended):** Contribute Poisson algebra to Mathlib4
   - File PR: `Mathlib/Algebra/Poisson.lean`
   - Requires: Understanding of Mathlib contribution workflow
   - Timeline: 3-6 months for review + merge

2. **Option B (Workaround):** Skip Poisson formulation, use coordinates only
   - Define everything in (q, p) coordinates explicitly
   - Loses: Geometric elegance, manifold generalization
   - Gains: Can proceed immediately without Mathlib contribution

---

### 1.3 Liouville's Theorem

```lean
/-- Liouville's Theorem: Hamiltonian flow preserves phase space volume -/
theorem liouville_theorem (H : PhaseSpace → ℝ) (t : ℝ) :
    let Φ_t := hamiltonianFlow H t  -- Flow for time t
    ∀ region : Set PhaseSpace,
      measure (Φ_t '' region) = measure region
```

**Required Theorems:**
- ✗ `hamiltonianFlow_exists`: Picard-Lindelöf for H (exists in Mathlib)
- ✗ `flow_is_symplectic`: Φ_t* ω = ω for all t
- ✗ `symplectic_implies_volume_preserving`: det(DΦ_t) = 1
- ✗ `liouville`: Volume preservation follows from symplecticity

**Trouble Level:** ⭐⭐⭐ (3/5) - Requires symplectic geometry + ODE theory

---

### 1.4 Canonical Transformations

```lean
/-- Transformation is canonical if it preserves Poisson brackets -/
structure CanonicalTransformation where
  transform : PhaseSpace → PhaseSpace
  is_canonical : ∀ F G, {F ∘ T, G ∘ T} = {F, G} ∘ T
```

**Required Theorems:**
- ✗ `generating_function_theorem`: Every canonical transform has generating function
- ✗ `symplectic_matrix_condition`: J^T M J M = J for matrix representation
- ✗ `action_angle_variables`: Existence of action-angle coordinates (advanced)

**Trouble Level:** ⭐⭐⭐⭐ (4/5) - Advanced classical mechanics

---

## Layer 3: Integrator Rigor (`Physics/IntegratorRigor.lean`)

### 3.1 Störmer-Verlet Definition

```lean
/-- Störmer-Verlet integrator with timestep h -/
def stormerVerlet (h : ℝ) (H : Hamiltonian) : PhaseSpace → PhaseSpace
  | (q, p) =>
    let p_half := p + (h/2) * (-∇V q)      -- Half-kick momentum
    let q_new := q + h * M⁻¹ p_half          -- Full drift position
    let p_new := p_half + (h/2) * (-∇V q_new) -- Half-kick momentum
    (q_new, p_new)
```

**Required Theorems:**

#### **A. Second-Order Accuracy**
```lean
theorem verlet_second_order (H : Hamiltonian) (h : ℝ) :
    ∀ T : ℝ,
      let Φ_h := stormerVerlet h H
      let Φ_exact := hamiltonianFlow H T
      ∀ initial_condition,
        error (iterate Φ_h (T/h) initial) (Φ_exact T initial) = O(h²)
```

**Trouble Level:** ⭐⭐⭐⭐ (4/5) - Requires backward error analysis

---

#### **B. Symplecticity (The Hard Part)**
```lean
/-- Verlet is symplectic: preserves ω -/
theorem verlet_symplectic (H : Hamiltonian) (h : ℝ) :
    let Φ := stormerVerlet h H
    ∀ p, (Φ* ω) p = ω p
  where
    Φ* ω := pullback of ω under Φ
```

**Required Steps:**
1. ✗ Define `pullback` operation for differential forms
2. ✗ Prove Verlet as composition of two symplectic Euler steps
3. ✗ Prove symplectic Euler is symplectic
4. ✗ Prove composition of symplectic maps is symplectic

**Trouble Level:** ⭐⭐⭐⭐⭐ (5/5) - **BLOCKER: Requires Layer 1 symplectic forms**

---

#### **C. Backward Error Analysis (BEA)**

```lean
/-- Shadow Hamiltonian: Verlet exactly integrates a modified Hamiltonian -/
theorem shadow_hamiltonian (H : Hamiltonian) (h : ℝ) :
    ∃ H̃ : PhaseSpace → ℝ,
      let Φ := stormerVerlet h H
      ∀ p, Φ p = hamiltonianFlow H̃ h p ∧  -- Exact flow of H̃
      |H̃ - H| = O(h²)                    -- H̃ is close to H
```

**Required Theorems:**
- ✗ `shadow_hamiltonian_exists`: Construct H̃ explicitly (power series in h)
- ✗ `shadow_hamiltonian_bounded`: H̃ is conserved by Verlet (up to O(h^p))
- ✗ `near_energy_conservation`: Verlet approximately conserves H
- ✗ `long_time_stability`: Bounded error over exponentially long times

**Trouble Level:** ⭐⭐⭐⭐⭐ (5/5) - **Most difficult part; research-level**

---

### 3.2 Constrained Dynamics (SHAKE/RATTLE)

```lean
/-- Holonomic constraint: g(q) = 0 -/
structure Constraint where
  constraint : PhaseSpace → ℝ
  is_holonomic : ∇g · M⁻¹ ∇g is invertible

/-- SHAKE: position projection onto constraint manifold -/
def SHAKE (C : Constraint) (q p : PhaseSpace) (h : ℝ) : PhaseSpace :=
  let q_tilde := q + h * M⁻¹ p  -- Predicted position
  let q_new := solve g(q_new) = 0  -- Newton iteration
  (q_new, p)

/-- RATTLE: SHAKE + velocity projection -/
def RATTLE (C : Constraint) (q p : PhaseSpace) (h : ℝ) : PhaseSpace :=
  let (q_new, _) := SHAKE C (q, p) h
  let p_new := project_velocity C q_new p_tilde
  (q_new, p_new)
```

**Required Theorems:**
- ✗ `SHAKE_converges`: Newton iteration finds solution (if close enough)
- ✗ `RATTLE_preserves_constraints`: Both position and velocity constraints
- ✗ `constrained_symplecticity`: RATTLE is symplectic on constraint manifold
- ✗ `time_reversibility`: RATTLE is reversible

**Trouble Level:** ⭐⭐⭐⭐⭐ (5/5) - **Very difficult; manifold-valued ODEs**

---

### 3.3 Long-Term Energy Stability

```lean
/-- Modified Hamiltonian is approximately conserved over long times -/
theorem verlet_energy_bounded (H : Hamiltonian) (h : ℝ) (T : ℝ) :
    let H̃ := shadow_hamiltonian H h
    let Φ := stormerVerlet h H
    ∀ n : Nat,
      let trajectory := iterate Φ n
      |H (trajectory initial) - H (initial)| ≤ C * h² for all n ≤ T/h
```

**Required Theorems:**
- ✗ `shadow_energy_bounds`: H̃ is close to H
- ✗ `kAM_theory`: For integrable systems, invariant tori persist
- ✗ `numerical_observability`: Can measure shadow Hamiltonian in practice

**Trouble Level:** ⭐⭐⭐⭐ (4/5) - Requires backward error analysis

---

## Most Troublesome Aspects (Ranked)

### 1. **Poisson Algebras** ⭐⭐⭐⭐⭐
**Why:** Not in Mathlib, requires algebraic contribution
**Impact:** Blocks all canonical Hamiltonian mechanics
**Solution:** Contribute to Mathlib OR use coordinate-only approach

### 2. **Symplecticity Proofs** ⭐⭐⭐⭐⭐
**Why:** Requires differential forms, pullbacks, Lie derivatives
**Impact:** Cannot prove Verlet is symplectic without Layer 1
**Solution:** Accept as "known from literature" with citation

### 3. **Backward Error Analysis** ⭐⭐⭐⭐⭐
**Why:** Research-level mathematics, power series, modified equations
**Impact:** Cannot prove long-term stability rigorously
**Solution:** Use experimental validation (shadow Hamiltonian monitoring)

### 4. **Constrained Dynamics** ⭐⭐⭐⭐
**Why:** Manifold-valued ODEs, Newton convergence, projection geometry
**Impact:** SHAKE/RATTLE cannot be verified
**Solution:** Treat as black box, validate empirically

### 5. **Liouville's Theorem** ⭐⭐⭐
**Why:** Requires flow maps, measure preservation
**Impact:** Foundational for geometric mechanics
**Solution:** Straightforward if symplectic forms exist

---

## Workaround Strategy (Recommended)

### Phase 1: Proceed Without Layers 1 & 3
1. ✅ Define MD in coordinates only (q, p) ∈ ℝ²ⁿ
2. ✅ Use Verlet as "black box" integrator
3. ✅ Prove srank bound (Layer 4) without symplectic machinery
4. ✅ Implement JAX kernels (Layer 2) with experimental validation

**What you lose:**
- Geometric elegance of symplectic formulation
- Rigorous guarantees of long-term stability
- Machine-checked proofs of integrator properties

**What you gain:**
- Can proceed immediately
- 90% of practical value (srank bound)
- Experimental validation instead of formal proof

### Phase 2: Add Rigor Later
1. Contribute Poisson algebra to Mathlib (6 months)
2. Build symplectic geometry foundation (3 months)
3. Prove Verlet properties (3 months)
4. Full formalization: 12-18 months total

---

## Collaboration Opportunities

### Mathlib4 Community
- **Poison algebra:** Could be valuable contribution
- **Symplectic geometry:** Active research area
- **Numerical analysis:** Growing interest in verified numerics

### Research Groups
- **Formalized geometric mechanics:** Few groups worldwide
- **Verified numerical methods:** Emerging field
- **Lean for physics:** Community is small but active

### Specific Contacts
- Mathlib Zulip: #mathlib-dev & #geometry streams
- Lean Community: Research forum and GitHub discussions
- Conferences: Lean Users Conference, ITP conferences

---

## Timeline Estimate

### With Full Rigor (Layers 1 + 3)
| Milestone | Time | Dependencies |
|-----------|------|--------------|
| Poisson algebra in Mathlib | 3-6 mo | Community review |
| Symplectic geometry foundations | 3 mo | Poisson algebra |
| Verlet symplecticity proof | 2-3 mo | Symplectic geometry |
| Backward error analysis | 3-4 mo | Verlet proof |
| Constrained dynamics | 2-3 mo | All above |
| **Total** | **13-19 mo** | **Sequential** |

### With Workaround (Skip Layers 1 + 3)
| Milestone | Time | Dependencies |
|-----------|------|--------------|
| Coordinate-only MD definition | 1 mo | None |
| Verlet as black box | 0 mo | None |
| Layers 2 & 4 (this work) | 2 mo | None |
| Experimental validation | 1 mo | Layers 2 & 4 |
| **Total** | **4 mo** | **Can start now** |

---

## Bottom Line

**Layers 1 & 3 are research contributions to formalized mathematics.** They're valuable but not required for the docking application to work.

**Recommendation:** Proceed with Layers 2 & 4, treat integrator as experimentally validated, and revisit formal rigor as future work or collaboration with Mathlib community.
