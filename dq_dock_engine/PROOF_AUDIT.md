# Proof Status Audit Report
## DQ-Dock Engine - Certified vs Heuristic Components

Generated: 2026-03-18

---

## Summary

| Category | Count | Certified | Conditionally Certified | Empirical Constant | Heuristic |
|----------|-------|-----------|------------------------|-------------------|-----------|
| **Physics Core** | | | | | |
| Lattice Sums | 3 | 3 | 0 | 0 | 0 |
| Molecular srank | 3 | 2 | 1 | 0 | 0 |
| Ewald Summation | 4 | 1 | 3 | 0 | 0 |
| Integrators | 2 | 2 | 0 | 0 | 0 |
| **Potentials** | | | | | |
| Lennard-Jones | 1 | 0 | 1 | 0 | 0 |
| Electrostatic | 1 | 0 | 1 | 0 | 0 |
| Hydrophobic | 1 | 0 | 0 | 0 | 1 |
| **Scoring** | | | | | |
| Internal LJ | 1 | 0 | 0 | 0 | 1 |
| SMINA External | 1 | 0 | 0 | 0 | 1 |
| **Parameters** | | | | | |
| VdW Radii | 1 | 0 | 0 | 1 | 0 |
| Boltzmann constant | 1 | 0 | 0 | 1 | 0 |

---

## CERTIFIED Components ✓

These have formal Lean 4 proofs guaranteeing correctness.

### Lattice Sum Bounds
| Function | Theorem | Bound |
|----------|----------|-------|
| `lattice_tail_bound` | `LatticeSum.lean::latticeTailSum6_le_M_div_R3` | M/R^(s-3) |
| `lj6_cutoff_error` | `LatticeSum.lean::lj6_tail_bound` | O(1/R³) |
| `lj12_cutoff_error` | `LatticeSum.lean::lj12_tail_bound` | O(1/R⁹) |

**Key Constants:**
- `M = 4π × 2 = 8π` (dyadic shell decomposition)
- Explicit: 512×(8/7) for LJ-6, 512×(512/511) for LJ-12

### Molecular Structural Rank
| Function | Theorem | Bound |
|----------|---------|-------|
| `molecular_srank_bound` | `MolecularSrank.lean::md_srank_bound` | 3K + 3L |
| `compute_srank` | `StructuralRank.lean::srank_eq_relevant_card` | Gradient-based |
| `thermodynamic_lower_bound` | `MolecularSrank.lean::md_thermodynamic_lower_bound` | srank × kBT × ln(2) |

### Integrators
| Function | Theorem |
|----------|---------|
| `VelocityVerlet.step` | `SymplecticIntegrator.lean::velocityVerletStep` |
| `hamiltonian` | `SymplecticIntegrator.lean::hamiltonian` |

### Other Certified
| Function | Theorem |
|----------|---------|
| `ewald_self_energy` | `EwaldSummation.lean` (exact formula) |

---

## CONDITIONALLY CERTIFIED Components ⚠️

These are proven in Lean 4, **subject to stated assumptions**.

### Lennard-Jones with Cutoff
**Assumption:** Cutoff radius R is chosen to achieve desired error tolerance.

```
Error(R) ≤ M/R³  (from lattice_tail_bound)
```

### Ewald Summation
**Assumptions:**
1. `Real.erfc` correctly implements complementary error function
2. Minimum image convention is physically appropriate for system
3. Ewald splitting parameter α is chosen appropriately
4. k-space truncation (k_max) is sufficient for convergence

### Thermodynamic Lower Bound
**Assumptions:**
1. Landauer principle is applicable to molecular system
2. System is at thermodynamic equilibrium

---

## EMPIRICAL CONSTANTS ⚗️

**Physical constants measured by experiment. These are GROUND TRUTH, not heuristics.**

### Van der Waals Radii
- **Source:** Bondi (1964) J. Phys. Chem. 68, 441-451
- **Values:** Experimentally measured from crystal structures
- **Status:** NIST-quality data, no formal proof needed

### Boltzmann Constant
- **Source:** NIST CODATA 2018
- **Value:** kB = 1.380649 × 10⁻²³ J/K (exact by definition)
- **Status:** Defined constant, not measured

---

## HEURISTIC Components ⚡

**Ad-hoc algorithm design or empirical parameters with NO formal justification.**

### Internal LJ Scoring Weights
- **Issue:** Ad-hoc weights (4.0 repulsion, 0.4 attraction)
- **Origin:** Design choice based on empirical observation
- **Use:** Fast approximate scoring for screening

### SMINA/Vina External
- **Issue:** Closed-source external binary
- **Origin:** Not peer-verified algorithm
- **Use:** Ground-truth comparison only

### Hydrophobic Potential
- **Issue:** Contact-based model is design choice
- **Origin:** Statistical analysis, not fundamental physics
- **Use:** Virtual screening enrichment

---

## Runtime Usage Guidelines

### For Certified Docking

```python
from dq_dock_engine.physics.lattice_sum import lj6_cutoff_error

# CERTIFIED: Use proven bounds
R = 10.0  # Angstroms
error_bound = lj6_cutoff_error(R)  # Proven O(1/R³)

# Decision: if gap > 2 * error_bound → CERTIFIED result
if energy_gap > 2 * error_bound:
    return "CERTIFIED_PRUNED", winner
```

### For Heuristic Screening

```python
from dq_dock_engine.docking.scoring import route_scoring, ScoringEngine

# HEURISTIC: Ad-hoc scoring for initial screening
scores = route_scoring(ScoringEngine.INTERNAL_LJ, ...)
# NOTE: These scores are NOT certified
```

---

## Adding New Components

When adding new physics or scoring functions:

1. **Determine proof status** using the decorators:
   ```python
   from dq_dock_engine.proof_status import certified, conditionally_certified, heuristic
   
   @certified("Theorem.lean::theorem_name")
   def new_proven_function(...):
       ...
   
   @conditionally_certified("Theorem.lean::theorem", assumptions=["assumption1"])
   def new_conditional_function(...):
       ...
   
   @heuristic()
   def new_heuristic_function(...):
       ...
   ```

2. **Document assumptions** for conditionally certified functions

3. **Update this audit report** with new components

---

## References

- Lean Theorems: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/`
- VdW Radii Source: Bondi, A. (1964). "Van der Waals Volumes and Radii." J. Phys. Chem. 68, 441-451.
- Boltzmann Constant: NIST CODATA 2018: kB = 1.380649 × 10⁻²³ J/K
