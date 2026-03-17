# DQ-Dock Physics Engine Plan

## Executive Summary

This document outlines the architecture for a JAX-based molecular docking physics engine built on the OpenHCS platform. The engine will implement the decision-theoretic physics from Paper 4's Lean proofs, providing provably correct feature selection for molecular binding prediction.

## 1. Understanding OpenHCS

### 1.1 Core Philosophy
OpenHCS is a generic pipeline system that:
- Takes **arbitrary 3D tensors** as input
- Applies **registered functions** to them
- Returns **3D tensors** as output
- Supports sidechannels for metadata (currently implicit, needs refactoring)

### 1.2 Key Architectural Components

| Component | Description | Adaptation for DQ-Dock |
|-----------|-------------|------------------------|
| **Component System** | Generic dimensional framework (Well, Site, Channel, Z, Time) | Remap to Ligand, Protein, InteractionType, Conformation |
| **Function Registry** | 574+ GPU functions with memory type decorators | Add docking-specific functions |
| **Memory Types** | NumPy, CuPy, PyTorch, TensorFlow, JAX, pyclesperanto | Primarily JAX for physics |
| **Pipeline Compiler** | 5-phase: Path→Zarr→Materialization→Memory→GPU | Add thermodynamic validation phase |
| **ProcessingContext** | Frozen immutable execution state | Add binding energy sidechannel |

### 1.3 Data Class Mapping

```
OpenHCS (Bioimaging)          →  DQ-Dock (Molecular Docking)
─────────────────────────────────────────────────────────────
Well                          →  Protein (or BindingSite)
Plate                         →  Ligand Library  
Image (Z,Y,X)                 →  Interaction Tensor (C,Y,X)
Channel                       →  Interaction Type (electrostatic, hydrophobic, H-bond)
Z-Stack                       →  Conformation Ensemble
Site                          →  Pose/Conformation
Multiprocessing Axis          →  Ligand-parallel or Protein-parallel
```

## 1.4 Architecture: Engine + Fork

---

## 2. Benchmark Datasets

### 2.1 Standard Datasets

| Dataset | Size | Purpose | Access |
|---------|------|---------|--------|
| **PDBbind Refined** | 5,316 complexes | Main benchmark | pdbbind-plus.org.cn |
| **PDBbind Core** | 195 complexes | Stringent evaluation | Same |
| **CASF** | 290 complexes | Scoring function competition | Same |
| **Leak Proof PDBBind** | ~4,000 | No data leakage | arXiv:2308.09639 |

### 2.2 Data Format

Each complex provides:
- `protein.pdb` - Protein structure
- `ligand.mol2` / `ligand.sdf` - Ligand structure  
- `Kd/Ki/IC50` - Binding affinity

### 2.3 Usage in srank Framework

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class BenchmarkCase:
    """Single benchmark case from PDBbind"""
    pdb_id: str
    protein_path: Path
    ligand_path: Path
    affinity: float  # Kd in nM
    binding_criterion: BindingCriterion

def load_pdbbind_subset(version: str = "v2020") -> list[BenchmarkCase]:
    """Load PDBbind refined set"""
    # Available via:
    # - pdbbind-plus.org.cn (registration)
    # - HuggingFace: photonmz/pdbbindpp-2020
    # - DeepChem: dc.molnet.load_function.pdbbind_datasets
    pass

def benchmark_srouting(cases: list[BenchmarkCase]) -> BenchmarkResults:
    """Run srank routing on benchmark"""
    results = []
    for case in cases:
        analysis = analyze(case.protein, case.binding_criterion)
        results.append({
            'pdb_id': case.pdb_id,
            'srank': analysis.srank,
            'tractability': analysis.tractability,
            'speedup': analysis.speedup,
        })
    return BenchmarkResults(results)
```

### 2.4 Comparison Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Speedup vs Vina | >10x | Per-ligand time ratio |
| RMSE | <2.0 kcal/mol | Binding affinity prediction |
| Kendall τ | >0.4 | Ranking correlation |
| Success rate | >80% | Top predictions within 2Å RMSD |

### 2.5 References

- PDBbind: Liu et al., Acc. Chem. Res. 2017, 50, 302-309
- CASF: Su et al., J. Chem. Inf. Model. 2019, 59, 895-913
- Leak Proof: Li et al., arXiv:2308.09639

Following OpenHCS best practices: separate research engine from platform fork.

```
/home/ts/code/projects/
├── papers-archive/
│   └── dq_dock_engine/      # Standalone JAX physics engine (THIS REPO)
│       ├── physics/          # srank, thermodynamics, binding
│       ├── srank/            # Structural rank computation
│       ├── thermodynamics/   # Energy bounds
│       └── functions/        # Registered physics functions
│
└── dq-dock/                 # OpenHCS fork (separate repo)
    └── (submodule → papers-archive/dq_dock_engine)
```

**Rationale:**

| Concern | Solution |
|---------|----------|
| Research code lives in papers-archive | Maintains provenance |
| OpenHCS fork remains clean | Platform fork is separate |
| Engine can be used standalone | Import from any project |
| Version sync via git submodule | Reproducible builds |
| Clean dependency graph | dq-dock depends on dq_dock_engine |

**Import pattern:**

```python
# In dq-dock (OpenHCS fork)
from dq_dock_engine.physics import compute_srank, route_batch
from dq_dock_engine.functions import register_physics_functions
```

**Module structure in papers-archive:**

```
dq_dock_engine/
├── __init__.py
├── pyproject.toml           # Standalone package
├── physics/
│   ├── __init__.py
│   ├── srank.py            # Structural rank (Section 2.2)
│   ├── thermodynamics.py    # Energy bounds
│   ├── binding.py           # Binding affinity
│   └── geometry.py          # Decision geometry
├── functions/
│   ├── __init__.py
│   ├── registry.py          # Function registration
│   └── examples/            # Physics function examples
├── data/
│   ├── __init__.py
│   ├── tensors.py           # Molecular tensor representations
│   └── conversion.py        # PDB/SDF → Tensor
└── lean/
    ├── __init__.py
    ├── proof_gen.py        # Generate Lean proofs
    └── export.py           # Export for verification
```

## 1.5 Benchmark Datasets

### 1.5.1 Standard Datasets

| Dataset | Size | Purpose | Access |
|---------|------|---------|--------|
| **PDBbind Refined** | 5,316 complexes | Main benchmark | pdbbind-plus.org.cn |
| **PDBbind Core** | 195 complexes | Stringent evaluation | Same |
| **CASF** | 290 complexes | Scoring function competition | Same |
| **Leak Proof PDBBind** | ~4,000 | No data leakage | arXiv:2308.09639 |

### 1.5.2 Data Format

Each complex provides:
- `protein.pdb` - Protein structure
- `ligand.mol2` / `ligand.sdf` - Ligand structure  
- `Kd/Ki/IC50` - Binding affinity

### 1.5.3 Comparison Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Speedup vs Vina | >10x | Per-ligand time ratio |
| RMSE | <2.0 kcal/mol | Binding affinity prediction |
| Kendall τ | >0.4 | Ranking correlation |

---

## 2. Physics to Implement

### 2.1 Structural Rank (srank)

From `StructuralRank.lean`:
```lean
def DecisionProblem.srank [CoordinateSpace S n] (dp : DecisionProblem A S) : ℕ :=
  Finset.card (Finset.univ.filter (fun i => dp.isRelevant i))
```

**Interpretation for Docking**:
- `srank` = minimal number of molecular features needed to determine binding
- Features: atomic positions, partial charges, hydrophobicity, electrostatics, etc.
- Goal: Find provably minimal feature set that predicts binding affinity

**Implementation**:
```python
# srank computation for molecular binding
def compute_srank(binding_data: InteractionTensor, utility: BindingUtility) -> int:
    """
    Compute structural rank - minimal sufficient features for binding decision.
    
    Args:
        binding_data: 5D tensor (ligand, protein, conformation, feature, value)
        utility: Binding utility function (affinity predictor)
    
    Returns:
        Minimal number of features required for binding decision
    """
    # 1. Build decision problem from binding data
    dp = build_decision_problem(binding_data, utility)
    
    # 2. Find relevant coordinates (features that affect optimal action)
    relevant = find_relevant_coordinates(dp)
    
    # 3. srank = cardinality of minimal sufficient set
    return len(minimal_sufficient_set(relevant))
```

### 2.2 The Efficiency Backdoor: Exact Algorithmic Framework

This section provides a complete algorithmic specification for leveraging the structural rank (srank) theorem from Paper 4 to achieve provably efficient molecular docking. All claims are derived directly from the Lean proofs in `StructuralRank.lean`.

---

#### 2.2.1 Mathematical Foundation

The central theorem from `StructuralRank.lean` (lines 40-41):

```lean
-- Under P ≠ coNP and ETH:
--   SUFFICIENCY-CHECK is polynomial ↔ srank(dp) is polynomially bounded

-- Definition (lines 96-98):
noncomputable def DecisionProblem.srank [CoordinateSpace S n] (dp : DecisionProblem A S) : ℕ :=
  Finset.card (Finset.univ.filter (fun i => dp.isRelevant i))
```

**What this mathematically guarantees:**

1. **IF** `srank(dp) ≤ n^k` for some constant k, **THEN** a polynomial-time algorithm EXISTS
2. **ONLY IF** `srank(dp)` is polynomially bounded can SUFFICIENCY-CHECK be polynomial

**For molecular docking, this translates to:**

| Condition | Mathematical Status | Algorithmic Guarantee |
|-----------|---------------------|---------------------|
| `srank = 0` | Trivial | Immediate decision without computation |
| `srank ≤ n` | Polynomial | Algorithm runs in O(n^k) where k is fixed |
| `srank = n` | coNP-hard (under ETH) | No polynomial algorithm exists |

---

#### 2.2.2 Decision Problem Encoding (OpenHCS-Style)

Following OpenHCS architectural principles: ABC contracts, dataclass patterns, enum-driven configuration.

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Callable, Protocol
from enum import Enum, auto
import jax.numpy as jnp

A = TypeVar('A')  # Action space
S = TypeVar('S')  # State space

# ===== ABC Contract (Principle 1: ABC Contract Enforcement) =====

class DecisionProblem(ABC, Generic[A, S]):
    """ABC enforcing contract: utility + allowed + isRelevant + isSufficient"""
    
    @abstractmethod
    def utility(self, action: A, state: S) -> float: ...
    
    @abstractmethod
    def allowed(self, state: S) -> list[A]: ...
    
    @abstractmethod
    def isRelevant(self, coord: int) -> bool: ...
    
    @abstractmethod
    def isSufficient(self, coords: set[int]) -> bool: ...

# ===== Dataclass Patterns (Principle 3: Dataclass Patterns) =====

@dataclass(frozen=True)
class BindingState:
    """S = ℝ^(3×n_atoms) × ℝ^(n_atoms) × ℝ^(n_residues)"""
    positions: jnp.ndarray      # (n_atoms, 3)
    charges: jnp.ndarray       # (n_atoms,)
    accessibility: jnp.ndarray # (n_residues,)

@dataclass(frozen=True)
class BindingAction:
    """A = valid ligand conformations"""
    positions: jnp.ndarray  # (n_ligand_atoms, 3)
    rotation: jnp.ndarray  # (3, 3)
    translation: jnp.ndarray  # (3,)

@dataclass(frozen=True)
class BindingProblem(DecisionProblem[BindingAction, BindingState]):
    """Molecular binding as decision problem"""
    utility_fn: Callable[[BindingAction, BindingState], float]
    ligand_atoms: int
    protein_residues: int
    
    @property
    def coordinate_dim(self) -> int:
        return self.ligand_atoms * self.protein_residues
    
    def utility(self, action: BindingAction, state: BindingState) -> float:
        return self.utility_fn(action, state)
    
    def allowed(self, state: BindingState) -> list[BindingAction]:
        return []  # Placeholder
    
    def isRelevant(self, coord: int) -> bool:
        return True  # Computed via srank algorithm
    
    def isSufficient(self, coords: set[int]) -> bool:
        return len(coords) >= self.coordinate_dim  # Simplified

# ===== Enum-Driven Configuration (Principle 5: Enum-Driven) =====

class Tractability(Enum):
    """Six tractable cases + hard case"""
    SEPARABLE = auto()      # srank = 0
    TENSOR_RANK = auto()    # Low tensor rank
    TREE = auto()           # Tree structure
    TREEWIDTH = auto()      # Bounded treewidth
    BOUNDED_ACTIONS = auto()  # Explicit enumeration
    SYMMETRIC = auto()      # Orbit reduction
    HARD = auto()           # coNP-hard

@dataclass(frozen=True)
class SrankAnalysis:
    tractability: Tractability
    srank: int
    n_dimensions: int
    algorithm: str
    action: str

# ===== Fail-Loud Error Handling (Principle 4: Fail-Loud) =====

class SrankComputationError(Exception):
    """Specific exception - no silent failures"""
    pass

# ===== Pure Functions (Principle 6: Stateless Architecture) =====

def compute_srank(problem: DecisionProblem) -> int:
    """srank = |{i | isRelevant(i)}| - pure function"""
    return sum(1 for i in range(problem.coordinate_dim) if problem.isRelevant(i))

def classify_tractability(srank: int, n: int) -> Tractability:
    """Enum-driven classification - no magic strings"""
    if srank == 0:
        return Tractability.SEPARABLE
    elif srank < n * 0.1:
        return Tractability.TENSOR_RANK
    elif srank < n:
        return Tractability.TREEWIDTH
    return Tractability.HARD

def analyze(protein: Protein, criterion: BindingCriterion) -> SrankAnalysis:
    """Main analysis - declarative, terse"""
    problem = BindingProblem(...)
    n = problem.coordinate_dim
    srank = compute_srank(problem)
    tractability = classify_tractability(srank, n)
    
    return SrankAnalysis(
        tractability=tractability,
        srank=srank,
        n_dimensions=n,
        algorithm=_algorithm_for(tractability),
        action=_action_for(tractability)
    )

# Lookup table (Principle: Mathematical Simplification)
_ALGORITHM_MAP = {
    Tractability.SEPARABLE: "immediate_decision",
    Tractability.TENSOR_RANK: "factorized_computation", 
    Tractability.TREE: "tree_dp",
    Tractability.TREEWIDTH: "csp_algorithm",
    Tractability.BOUNDED_ACTIONS: "explicit_enumeration",
    Tractability.SYMMETRIC: "orbit_reduction",
    Tractability.HARD: "approximation_required",
}

_ACTION_MAP = {
    Tractability.SEPARABLE: "return_constant",
    Tractability.TENSOR_RANK: "compute_with_factors",
    Tractability.TREE: "run_tree_dp",
    Tractability.TREEWIDTH: "run_csp",
    Tractability.BOUNDED_ACTIONS: "enumerate_explicitly",
    Tractability.SYMMETRIC: "reduce_by_orbits",
    Tractability.HARD: "skip_or_approximate",
}

def _algorithm_for(t: Tractability) -> str:
    return _ALGORITHM_MAP[t]

def _action_for(t: Tractability) -> str:
    return _ACTION_MAP[t]

---

#### 2.2.3 JAX-Accelerated Computation

Following OpenHCS: pure functions, JIT compilation, vectorized operations.

```python
import jax
from jax import jit, vmap

@jit
def _isRelevant(utility_matrix: jnp.ndarray, coord: int) -> bool:
    """Vectorized isRelevant check"""
    original_opt = jnp.argmax(utility_matrix, axis=1)
    perturbed = jnp.roll(utility_matrix, shift=1, axis=1)  # Simplified perturbation
    perturbed_opt = jnp.argmax(perturbed, axis=1)
    return jnp.any(original_opt != perturbed_opt)

@jit
def compute_srank(utility_matrix: jnp.ndarray) -> int:
    """srank = Σ isRelevant(i) - fully vectorized"""
    relevance_checks = vmap(_isRelevant, in_axes=(None, 0))(utility_matrix, jnp.arange(utility_matrix.shape[1]))
    return int(jnp.sum(relevance_checks))
```

---

#### 2.2.4 Six Tractable Cases (Lookup Table Pattern)

Each case maps to algorithm via lookup table (Mathematical Simplification).

```python
# ===== Protocol for Algorithm Selection (Principle 1: ABC) =====

class TractableCase(Protocol):
    """Protocol enforcing check + algorithm contract"""
    def check(self, problem: BindingProblem) -> bool: ...
    def algorithm(self) -> str: ...

# ===== Six Cases as Simple Dataclasses =====

@dataclass(frozen=True)
class SeparableCase:
    """U(a,s) = f(a) + g(s) → srank = 0"""
    def check(self, p: BindingProblem) -> bool:
        return _check_separability(p.utility_fn)  # Implementation elsewhere
    def algorithm(self) -> str: return "immediate_decision"

@dataclass(frozen=True)
class TensorRankCase:
    """Low tensor rank → bounded factor interactions"""
    threshold: int = 10
    def check(self, p: BindingProblem) -> bool:
        return _compute_rank(p.utility_matrix) < self.threshold
    def algorithm(self) -> str: return "factorized_computation"

# ... (TREE, TREEWIDTH, BOUNDED_ACTIONS, SYMMETRIC follow same pattern)

# ===== Case Registry (Enum-Driven) =====

TRACTABLE_CASES: tuple[TractableCase, ...] = (
    SeparableCase(),
    TensorRankCase(),
    TreeCase(),
    TreewidthCase(bound=4),
    BoundedActionsCase(bound=1000),
    SymmetricCase(),
)

def select_case(problem: BindingProblem) -> TractableCase:
    """First matching case - fail-loud if none found"""
    for case in TRACTABLE_CASES:
        if case.check(problem):
            return case
    raise SrankComputationError(f"No tractable case for problem")
```

---

#### 2.2.5 Complete Router (Declarative Pipeline)

```python
@dataclass(frozen=True)
class RouterResult:
    case: TractableCase
    srank: int
    speedup: float

def route(protein: Protein, ligands: list[Ligand], criterion: BindingCriterion) -> RouterResult:
    """Declarative routing - no imperative conditionals"""
    problem = BindingProblem(utility_fn=criterion.utility, ...)
    case = select_case(problem)
    srank = compute_srank(problem.utility_matrix)
    
    return RouterResult(
        case=case,
        srank=srank,
        speedup=(problem.n_dimensions / max(srank, 1)) ** 2
    )

def execute_batch(result: RouterResult, protein: Protein, ligands: list[Ligand]) -> list[BindingResult]:
    """Execute based on selected case - dispatch table pattern"""
    return _EXECUTION_DISPATCH[result.case.__class__](result, protein, ligands)

_EXECUTION_DISPATCH: dict[type, Callable] = {
    SeparableCase: _execute_trivial,
    TensorRankCase: _execute_factorized,
    TreeCase: _execute_tree_dp,
    TreewidthCase: _execute_csp,
    BoundedActionsCase: _execute_explicit,
    SymmetricCase: _execute_orbit_reduced,
    # HARD falls through to approximation
}
```

---

#### 2.2.6 Complexity Analysis

| Operation | Cost | Frequency | Total |
|-----------|------|-----------|-------|
| srank analysis | O(n × m) | Once | O(n) |
| Per-ligand (reduced) | O(k²) | Per ligand | O(L × k²) |
| Traditional | O(n²) | Per ligand | O(L × n²) |

**Speedup**: (n/k)² when srank = k << n

---

#### 2.2.7 Integration with OpenHCS Pipeline

```python
# Register as OpenHCS function following 3D→3D contract
@register_function("srank_analysis")
def analyze_binding_efficiency(
    protein_tensor: Array3D,  # (C, Y, X) = features
    ligand_library: Array3D,
    binding_criterion: BindingCriterion
) -> tuple[Array3D, dict]:  # Returns tensor + sidechannel
    """OpenHCS-compatible srank analysis"""
    result = route(protein_tensor, ligand_library, binding_criterion)
    return result.tensor, {'srank': result.srank, 'speedup': result.speedup}
```

---

#### 2.2.8 Key References

- `StructuralRank.lean` lines 40-50: Central theorem
- `StructuralRank.lean` lines 96-98: srank definition  
- `StructuralRank.lean` lines 131-149: Separable utility case
- `StructuralRank.lean` lines 252-290: Tautology boundary theorem
- `StructuralRank.lean` lines 291-326: Six tractable cases
- `PhysicalCore.lean` lines 33-48: Finite budget impossibility

### 2.3 Thermodynamic Bounds

From `ThermodynamicLift.lean`:
```lean
-- Landauer constant at room temperature
noncomputable def landauerConstant : ℝ := 1.38e-23 * 300 * Real.log 2

-- Thermodynamic cost proportional to state space
noncomputable def thermodynamicCost {A S : Type*} [Fintype A] [Fintype S]
    (_P : StochasticDecisionProblem A S) : ℝ :=
  landauerConstant * (Fintype.card S : ℝ)
```

**Interpretation for Docking**:
- Landauer principle: erasing information requires energy
- Binding decision requires energy proportional to state complexity
- Provides fundamental lower bound on computation needed

**Implementation**:
```python
# Physical constants
KBOLTZMANN = 1.380649e-23  # J/K
ROOM_TEMPERATURE = 300  # K
LANDAUER_CONSTANT = KBOLTZMANN * ROOM_TEMPERATURE * np.log(2)

def thermodynamic_cost(n_features: int, n_states: int) -> float:
    """
    Compute thermodynamic lower bound on binding computation.
    
    Args:
        n_features: Number of molecular features (srank)
        n_states: Size of state space (conformations × ligand poses)
    
    Returns:
        Minimum energy required (Joules)
    """
    bits = n_features * np.log2(n_states)
    return LANDAUER_CONSTANT * bits

def energy_time_tradeoff(energy_budget: float, n_features: int) -> float:
    """
    Compute time bound given energy budget.
    
    From Lean: energy_time_tradeoff theorem
    """
    return energy_budget / (LANDAUER_CONSTANT * n_features)
```

### 2.3 Decision Geometry

From `DecisionGeometry.lean` and related files:
- Decision boundary lives in high-dimensional feature space
- Structural rank determines dimensionality of decision boundary
- Tractable when srank << full dimensionality

**Implementation**:
```python
class DecisionGeometry:
    """Geometric representation of binding decision boundary."""
    
    def __init__(self, binding_problem):
        self.problem = binding_problem
        self.feature_dim = binding_problem.n_features
        self.srank = compute_srank(binding_problem)
    
    def decision_boundary_dimension(self) -> int:
        """Dimensionality of binding decision boundary."""
        return self.srank
    
    def is_tractable(self, poly_bound: int = 4) -> bool:
        """Check if binding prediction is tractable."""
        # Tractable if srank is polynomially bounded
        return self.srank <= self.feature_dim ** poly_bound
    
    def complexity_class(self) -> str:
        """Return complexity class: P, coNP, PSPACE, etc."""
        if self.srank == 0:
            return "P (trivial)"
        elif self.srank <= np.log2(self.problem.n_states):
            return "P (bounded state)"
        elif self.is_tractable():
            return "P (bounded srank)"
        else:
            return "coNP-hard"  # From Paper 4 hardness proofs
```

## 3. JAX Physics Engine Architecture

### 3.1 Module Structure

```
dq_dock/
├── physics/
│   ├── __init__.py
│   ├── srank.py           # Structural rank computation
│   ├── thermodynamics.py   # Energy bounds
│   ├── geometry.py        # Decision geometry
│   ├── binding.py         # Binding affinity models
│   └── forces.py          # Molecular force calculations
├── functions/
│   ├── __init__.py
│   ├── registry.py        # Function registration
│   ├── decorators.py      # @physics_func decorators
│   └── examples/
│       ├── electrostatic.py
│       ├── vdw.py
│       └── hydrophobic.py
├── data/
│   ├── __init__.py
│   ├── tensors.py         # Molecular tensor representations
│   ├── conversion.py      # PDB/SDF → Tensor converters
│   └── batch.py           # Batch processing utilities
├── pipeline/
│   ├── __init__.py
│   ├── steps.py           # Physics-specific pipeline steps
│   ├── compilation.py     # Extended compiler phases
│   └── validation.py      # Thermodynamic validation
└── lean/
    ├── __init__.py
    ├── proof_gen.py       # Generate Lean proof snippets
    └── export.py          # Export proofs for verification
```

### 3.2 Core Tensor Representations

```python
# Molecular data as 3D+ tensors (following OpenHCS 3D contract)

class MolecularTensor:
    """
    5D tensor representation: (batch, ligand, protein, conformation, feature)
    
    OpenHCS mapping:
    - batch: parallel processing unit
    - ligand: ligand ID (replaces Well)
    - protein: protein/target ID (replaces Plate)  
    - conformation: pose/conformation (replaces Site)
    - feature: interaction type (replaces Channel)
    
    Final shape for OpenHCS: (C, Y, X) where:
    - C = feature dimension (electrostatic, vdw, etc.)
    - Y = conformation dimension
    - X = atom/residue dimension
    """
    
    @staticmethod
    def from_pdb_ligand(pdb_file: str, ligand_resname: str) -> 'jnp.ndarray':
        """Parse PDB and extract ligand atoms as tensor."""
        # Extract atomic coordinates, charges, types
        # Return shape: (n_atoms, 3) → mapped to (1, n_atoms, 3)
        pass
    
    @staticmethod  
    def from_protein(pdb_file: str) -> 'jnp.ndarray':
        """Parse protein structure as tensor."""
        # Extract CA positions, residue types, accessibility
        # Return shape: (n_residues, feature_dim)
        pass
    
    @staticmethod
    def interaction_tensor(
        ligand: 'jnp.ndarray',
        protein: 'jnp.ndarray',
        interactions: List[str] = ['electrostatic', 'vdw', 'hbond']
    ) -> 'jnp.ndarray':
        """
        Compute interaction tensor between ligand and protein.
        
        Shape: (n_interactions, n_ligand_atoms, n_protein_residues)
        OpenHCS: (C, Y, X) where C=n_interactions
        """
        pass
```

### 3.3 Physics Functions with JAX

```python
from dq_dock.functions.decorators import physics_func

@physics_func(
    input_memory_type='jax',
    output_memory_type='jax',
    sidechannels=['binding_energy', 'srank']
)
def compute_electrostatic(
    ligand_coords: 'jnp.ndarray',    # (n_atoms, 3)
    ligand_charges: 'jnp.ndarray',   # (n_atoms,)
    protein_coords: 'jnp.ndarray',   # (n_residues, 3)
    protein_charges: 'jnp.ndarray',  # (n_residues,)
    distance_cutoff: float = 4.0    # Angstroms
) -> tuple['jnp.ndarray', dict]:
    """
    Compute electrostatic interaction energy.
    
    Returns:
        energy: scalar binding energy
        sidechannel: {'srank_contribution': int, 'energy': float}
    """
    # Coulomb energy computation
    # E = k * q1 * q2 / r
    
    distances = compute_distances(ligand_coords, protein_coords)
    within_cutoff = distances < distance_cutoff
    
    energy = jnp.sum(
        within_cutoff * ligand_charges[:, None] * protein_charges[None, :] / distances
    )
    
    # srank contribution: which features matter for electrostatics
    srank_contribution = jnp.sum(within_cutoff.any(axis=0))
    
    return energy, {
        'srank_contribution': srank_contribution,
        'energy': energy,
        'interaction_type': 'electrostatic'
    }

@physics_func(input_memory_type='jax', output_memory_type='jax')
def compute_vdw(
    ligand_coords: 'jnp.ndarray',
    protein_coords: 'jnp.ndarray',
    ligand_radii: 'jnp.ndarray',
    protein_radii: 'jnp.ndarray'
) -> tuple['jnp.ndarray', dict]:
    """Lennard-Jones van der Waals interaction."""
    # 12-6 Lennard-Jones potential
    pass

@physics_func(input_memory_type='jax', output_memory_type='jax')
def compute_binding_energy(
    interaction_tensor: 'jnp.ndarray',  # (C, Y, X)
    weights: dict = None
) -> tuple['jnp.ndarray', dict]:
    """
    Compute total binding energy from interaction components.
    
    Sidechannel returns srank for feature selection.
    """
    if weights is None:
        weights = {'electrostatic': 1.0, 'vdw': 1.0, 'hydrophobic': 0.5}
    
    total_energy = sum(
        weights[k] * v for k, v in interaction_tensor.items()
    )
    
    # Compute srank: which features are relevant?
    relevant_features = find_relevant_coordinates(interaction_tensor, total_energy)
    srank = len(relevant_features)
    
    return total_energy, {
        'srank': srank,
        'relevant_features': relevant_features,
        'energy': total_energy
    }
```

### 3.4 Function Registry Extension

```python
# Extend OpenHCS function registry with physics functions

from openhcs.processing.func_registry import FunctionRegistry
from dq_dock.physics import srank, thermodynamics, geometry

class PhysicsFunctionRegistry(FunctionRegistry):
    """Extended registry for physics functions."""
    
    def register_physics_functions(self):
        """Register all physics functions with proper decorators."""
        
        # Structural rank functions
        self.register(srank.compute_srank)
        self.register(srank.find_relevant_coordinates)
        self.register(srank.minimal_sufficient_set)
        
        # Thermodynamic functions
        self.register(thermodynamics.thermodynamic_cost)
        self.register(thermodynamics.energy_time_tradeoff)
        self.register(thermodynamics.landauer_limit)
        
        # Geometry functions
        self.register(geometry.decision_boundary_dimension)
        self.register(geometry.complexity_class)
        
        # Binding energy functions
        self.register(binding.compute_electrostatic)
        self.register(binding.compute_vdw)
        self.register(binding.compute_hydrophobic)
        self.register(binding.compute_binding_energy)
        
        # Force calculations
        self.register(forces.gradient_descent)
        self.register(forces.molecular_dynamics)
        self.register(forces.energy_minimization)
```

## 4. Integration with OpenHCS Pipeline

### 4.1 Extended Compilation Phases

```python
# New Phase 6: Thermodynamic Validation

class ThermodynamicValidator:
    """Validates that binding computation is physically feasible."""
    
    def validate(self, context: ProcessingContext, steps: List[Step]):
        """
        Check thermodynamic constraints:
        1. Energy budget sufficient for srank computation
        2. Computation time within bounds
        3. Physical feasibility of binding prediction
        """
        for step in steps:
            if step.is_physics_step:
                energy_required = step.thermodynamic_cost
                if energy_required > context.energy_budget:
                    raise ThermodynamicViolation(
                        f"Step {step.name} requires {energy_required}J "
                        f"but budget is {context.energy_budget}J"
                    )
```

### 4.2 Sidechannel Refactoring

```python
# Current: Implicit sidechannels
# Target: Explicit sidechannel system

@dataclass
class PhysicsSidechannel:
    """Explicit sidechannel for physics computations."""
    
    # Thermodynamic metrics
    energy: float = 0.0
    thermodynamic_cost: float = 0.0
    computation_bits: int = 0
    
    # Decision metrics
    srank: int = 0
    relevant_features: List[str] = field(default_factory=list)
    complexity_class: str = "unknown"
    
    # Binding metrics  
    binding_energy: float = 0.0
    confidence: float = 0.0

class PhysicsContext(ProcessingContext):
    """Extended context with physics sidechannels."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.physics_sidechannel = PhysicsSidechannel()
    
    def update_physics(self, updates: PhysicsSidechannel):
        """Update physics sidechannel during computation."""
        for key, value in updates.__dict__.items():
            if value is not None:
                setattr(self.physics_sidechannel, key, value)
```

### 4.3 Pipeline Definition Example

```python
from openhcs.core.steps import FunctionStep
from dq_dock.physics import compute_binding_energy, compute_srank

# Define binding prediction pipeline
pipeline = Pipeline([
    # Step 1: Compute interaction tensor
    FunctionStep(
        func=compute_interaction_tensor,
        variable_components=[VariableComponents.LIGAND],
        name="interaction_tensor"
    ),
    
    # Step 2: Compute binding energy
    FunctionStep(
        func=compute_binding_energy,
        variable_components=[VariableComponents.LIGAND],
        sidechannels=['binding_energy', 'srank'],
        name="binding_energy"
    ),
    
    # Step 3: Compute structural rank (feature selection)
    FunctionStep(
        func=compute_srank,
        variable_components=[VariableComponents.LIGAND],
        sidechannels=['srank', 'relevant_features'],
        name="feature_selection"
    ),
    
    # Step 4: Validate thermodynamic bounds
    FunctionStep(
        func=validate_thermodynamics,
        variable_components=[VariableComponents.LIGAND],
        name="thermodynamic_validation"
    ),
    
    # Step 5: Rank ligands by binding affinity
    FunctionStep(
        func=rank_ligands,
        variable_components=[VariableComponents.LIGAND],
        group_by=GroupBy.NONE,
        name="ranking"
    )
])
```

## 5. Lean Proof Integration

### 5.1 Proof Generation

```python
# Generate Lean proof snippets from computed values

def generate_srank_proof(srank_value: int, features: List[str]) -> str:
    """Generate Lean proof that srank computation is correct."""
    return f"""
theorem srank_computation_correct
    (features : Finset (Fin n))
    (hRelevant : ∀ i ∈ features, bindingProblem.isRelevant i)
    (hSufficient : bindingProblem.isSufficient features) :
    bindingProblem.srank = {srank_value} := by
  have hLe : bindingProblem.srank ≤ features.card := 
    srank_le_sufficient_card bindingProblem features hSufficient
  have hGe : features.card ≤ bindingProblem.srank :=
    sufficient_contains_relevant bindingProblem features hRelevant
  exact le_antisymm hLe hGe
"""

def generate_thermodynamic_proof(
    energy_budget: float,
    computation_bits: int,
    required_energy: float
) -> str:
    """Generate Lean proof of thermodynamic feasibility."""
    return f"""
theorem thermodynamically_feasible
    (E_budget : ℝ) (hE : E_budget = {energy_budget})
    (bits : ℕ) (hBits : bits = {computation_bits}) :
    landauerEnergyFloor bits ROOM_TEMPERATURE ≤ E_budget := by
  have hCost := landauerEnergyFloor_mono_bits 
    (show bits ≤ bits from le_refl bits) 
    (show 0 ≤ ROOM_TEMPERATURE from by positivity)
  rw [hBits, hE] at hCost
  exact hCost
"""
```

### 5.2 Verification Workflow

```python
class LeanProofVerifier:
    """Verify physics computations against Lean proofs."""
    
    def __init__(self, lean_path: str):
        self.lean_path = lean_path
    
    def verify_srank(self, srank_value: int, features: List[str]) -> bool:
        """Verify srank computation matches Lean proof."""
        proof = generate_srank_proof(srank_value, features)
        return self._run_lean(proof)
    
    def verify_thermodynamics(
        self, 
        energy_budget: float, 
        computation_bits: int
    ) -> bool:
        """Verify thermodynamic bounds from Lean."""
        proof = generate_thermodynamic_proof(
            energy_budget, computation_bits
        )
        return self._run_lean(proof)
```

## 6. Implementation Phases

### Phase 1: Foundation (Weeks 1-4)
- [ ] Set up module structure
- [ ] Implement basic tensor representations
- [ ] Add JAX physics functions (electrostatic, VdW)
- [ ] Register functions with OpenHCS

### Phase 2: Core Physics (Weeks 5-10)
- [ ] Implement srank computation
- [ ] Implement thermodynamic bounds
- [ ] Implement decision geometry
- [ ] Add sidechannel system

### Phase 3: Pipeline Integration (Weeks 11-16)
- [ ] Extend pipeline compiler with Phase 6
- [ ] Implement thermodynamic validation
- [ ] Add binding affinity ranking
- [ ] Test end-to-end pipeline

### Phase 4: Lean Verification (Weeks 17-22)
- [ ] Implement proof generation
- [ ] Build Lean export utilities
- [ ] Verify computations against proofs
- [ ] Document verification workflow

### Phase 5: Optimization (Weeks 23-26)
- [ ] GPU optimization of physics functions
- [ ] Batch processing improvements
- [ ] Memory optimization
- [ ] Performance benchmarking

## 7. Key Design Decisions

### 7.1 Why JAX?
- **Automatic differentiation**: Essential for gradient-based optimization
- **Just-in-time compilation**: Fast execution of physics kernels
- **Hardware acceleration**: GPU/TPU support built-in
- **Immutable functions**: Aligns with OpenHCS stateless philosophy

### 7.2 Why Extend OpenHCS Rather Than Build Fresh?
- **Proven infrastructure**: 574+ functions, pipeline compiler, memory management
- **Familiar interface**: Researchers already know OpenHCS
- **GPU coordination**: Already handles multi-GPU scheduling
- **Function registry**: Automatic discovery and validation

### 7.3 Sidechannel Design
Current OpenHCS has implicit sidechannels. We're making them explicit:
- **Type-safe**: Strongly typed sidechannel classes
- **Validated**: Thermodynamic constraints checked at compile time
- **Documented**: Clear contract between steps

## 8. References

### OpenHCS Architecture
- `docs/architecture/COMPLETE_SYSTEM_ARCHITECTURE.md`
- `docs/source/architecture/function_pattern_system.rst`
- `docs/source/architecture/memory_type_system.rst`
- `docs/source/architecture/compilation_system_detailed.rst`

### Lean Proofs (Paper 4)
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/StructuralRank.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/StochasticSequential/ThermodynamicLift.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Physics/PhysicalCore.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Physics/Instantiation.lean`

## 9. PyMOL Streaming Integration

### 9.1 Architecture Overview

Following the OpenHCS streaming pattern (napari, Fiji), we'll add PyMOL streaming support. This enables real-time visualization of molecular structures during docking calculations.

**Key References**:
- `openhcs/runtime/fiji_stream_visualizer.py` - Fiji implementation pattern
- `external/PolyStore/src/polystore/fiji_stream.py` - Streaming backend
- `external/zmqruntime/docs/source/architecture/viewer_streaming_architecture.rst` - Generic ZMQ streaming

### 9.2 Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DQ-Dock Pipeline                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Physics Step │→ │ Binding Step │→ │ Ranking Step │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│         └─────────────────┴─────────────────┘                  │
│                           │                                    │
│           ┌───────────────▼───────────────┐                   │
│           │  PyMOLStreamingBackend        │                   │
│           │  (polystore-style backend)    │                   │
│           └───────────────┬───────────────┘                   │
│                           │ ZMQ + Shared Memory                │
│           ┌───────────────▼───────────────┐                   │
│           │  PyMOLStreamVisualizer        │                   │
│           │  (Process manager)            │                   │
│           └───────────────┬───────────────┘                   │
│                           │                                    │
│           ┌───────────────▼───────────────┐                   │
│           │  PyMOLViewerServer            │                   │
│           │  (PyMOL Python API)          │                   │
│           └───────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 Module Structure

```
dq_dock/visualization/
├── __init__.py
├── pymol_stream/
│   ├── __init__.py
│   ├── backend.py          # PyMOLStreamingBackend (PolyStore style)
│   ├── visualizer.py       # PyMOLStreamVisualizer (OpenHCS style)
│   ├── server.py           # PyMOLViewerServer (ZMQ server)
│   └── config.py           # PyMOLStreamingConfig
└── converters/
    ├── __init__.py
    ├── pdb_converter.py    # PDB → PyMOL object
    ├── sdf_converter.py    # SDF → PyMOL object  
    └── tensor_converter.py # JAX tensor → PyMOL object
```

### 9.4 PyMOL Streaming Backend

Following the FijiStreamingBackend pattern from `polystore/fiji_stream.py`:

```python
# In polystore/src/polystore/pymol_stream.py

from polystore.streaming import StreamingBackend
from polystore.constants import Backend

class PyMOLStreamingBackend(StreamingBackend):
    """PyMOL streaming backend with ZMQ publisher pattern."""
    
    _backend_type = Backend.PYMOL_STREAM.value
    
    VIEWER_TYPE = 'pymol'
    SHM_PREFIX = 'pymol_'
    
    def _prepare_molecule_data(self, data, file_path):
        """
        Prepare molecular data for transmission to PyMOL.
        
        Supports:
        - PDB format strings
        - SDF format strings
        - MOL2 format
        - XYZ coordinates
        """
        # Convert to PDB format (PyMOL native)
        if isinstance(data, str):
            # Already formatted
            return data
        elif hasattr(data, 'to_pdb'):
            # RDKit Mol object
            return data.to_pdb()
        else:
            # Assume coordinates array
            return self._coords_to_pdb(data)
    
    def _prepare_batch_item(self, data, file_path, data_type):
        """Prepare molecular data for batch streaming."""
        # Create shared memory for coordinates/structure
        mol_data = self._prepare_molecule_data(data, file_path)
        return self._create_shared_memory(mol_data, file_path)
    
    def save_batch(self, data_list, file_paths, **kwargs):
        """
        Stream batch of molecular structures to PyMOL via ZMQ.
        
        Key difference from Fiji: sends molecular structures (PDB/SDF)
        rather than image arrays.
        """
        host = kwargs.get('host', 'localhost')
        port = kwargs['port']
        transport_mode = kwargs['transport_mode']
        
        # Build message with molecular data
        batch_molecules = []
        for data, path in zip(data_list, file_paths):
            mol_data = self._prepare_molecule_data(data, path)
            batch_molecules.append({
                'path': str(path),
                'data': mol_data,
                'format': self._detect_format(data)
            })
        
        message = {
            'molecules': batch_molecules,
            'display_config': kwargs.get('display_config', {}),
        }
        
        # Send via ZMQ (PUB/SUB or REQ/REP)
        self._send_message(message, port, host, transport_mode)
```

### 9.5 PyMOL Viewer Server

Following the FijiViewerServer pattern:

```python
# In openhcs/runtime/pymol_viewer_server.py

from zmqruntime.streaming import StreamingVisualizerServer

class PyMOLViewerServer(StreamingVisualizerServer):
    """ZMQ server for PyMOL viewer with Python API integration."""
    
    VIEWER_TYPE = 'pymol'
    
    def start(self):
        """Initialize PyMOL in headless mode."""
        import pymol
        pymol.finish_launching(['-q'])  # Quiet mode
        
        self.cmd = pymol.cmd
        self.viewer = pymol.viewing
    
    def display_molecule(self, data: dict):
        """Display molecular structure in PyMOL."""
        mol_data = data['data']
        mol_format = data.get('format', 'pdb')
        object_name = data.get('path', 'molecule')
        
        # Load structure into PyMOL
        self.cmd.read_molstr(mol_data, object_name, format=mol_format)
        
        # Apply display settings
        self.cmd.show('cartoon', object_name)
        self.cmd.color('cyan', object_name)
        self.cmd.bg_color('white')
        
        # Zoom to fit
        self.cmd.zoom(object_name)
    
    def process_message(self, message):
        """Process incoming molecular data message."""
        for mol_data in message.get('molecules', []):
            self.display_molecule(mol_data)
```

### 9.6 PyMOL Stream Visualizer

Following the Napari/Fiji pattern in `openhcs/runtime/`:

```python
# In openhcs/runtime/pymol_stream_visualizer.py

class PyMOLStreamVisualizer(VisualizerProcessManager):
    """Manages PyMOL viewer instance for real-time visualization."""
    
    def __init__(
        self,
        filemanager,
        visualizer_config,
        viewer_title: str = "DQ-Dock PyMOL Visualization",
        persistent: bool = True,
        port: int = None,
        transport_mode = OpenHCSTransportMode.IPC,
    ):
        # Similar to FijiStreamVisualizer...
        pass
    
    def start_viewer(self, async_mode: bool = False):
        """Start PyMOL viewer process."""
        # Spawn PyMOL process with ZMQ listener
        # Uses pymolQt (Qt-based PyMOL) or pymol -q (headless)
        pass
    
    def send_molecule(self, pdb_data: str, object_name: str):
        """Send molecule to running PyMOL viewer."""
        # ZMQ message to PyMOL
        pass
```

### 9.7 Molecular Data Converters

```python
# In dq_dock/visualization/converters/tensor_converter.py

class PyMOLConverter:
    """Convert various molecular formats to PyMOL-compatible data."""
    
    @staticmethod
    def jax_to_pdb(coords: 'jnp.ndarray', atom_names: List[str] = None) -> str:
        """
        Convert JAX coordinate tensor to PDB format.
        
        Args:
            coords: (n_atoms, 3) coordinate tensor
            atom_names: List of atom names
        
        Returns:
            PDB format string
        """
        n_atoms = coords.shape[0]
        if atom_names is None:
            atom_names = ['ATOM'] * n_atoms
        
        pdb_lines = []
        for i in range(n_atoms):
            x, y, z = coords[i]
            line = f"ATOM  {i+1:5d} {atom_names[i]:4s} MOL A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
            pdb_lines.append(line)
        
        return '\n'.join(pdb_lines)
    
    @staticmethod
    def rdkit_mol_to_pdb(mol) -> str:
        """Convert RDKit Mol object to PDB format."""
        return Chem.MolToPDBBlock(mol)
    
    @staticmethod
    def binding_pose_to_pdb(pose_tensor: 'jnp.ndarray', ligand_id: int) -> str:
        """
        Convert binding pose tensor to PDB.
        
        Args:
            pose_tensor: (n_atoms, 4) - [x, y, z, element]
            ligand_id: Ligand identifier
        """
        elements = ['C', 'N', 'O', 'S', 'H']  # Map integers to elements
        # Convert and format as PDB
        pass
```

### 9.8 Configuration System

Following OpenHCS lazy config pattern:

```python
# In openhcs/core/config.py

@dataclass(frozen=True)
class PyMOLStreamingConfig(StreamingConfig):
    """Configuration for PyMOL streaming."""
    
    pymol_port: int = 5558  # Default port (distinct from napari/fiji)
    pymol_host: str = 'localhost'
    
    # Display settings
    representation: PyMOLRepresentation = PyMOLRepresentation.CARTOON
    color_scheme: PyMOLColorScheme = PyMOLColorScheme.CYAN
    bg_color: str = 'white'
    
    # Camera settings
    auto_zoom: bool = True
    orthoscopic: bool = False

@dataclass(frozen=True)
class LazyPyMOLStreamingConfig(LazyStreamingConfig):
    """Lazy version with inheritance support."""
    
    def __new__(cls):
        # Generate lazy config with inheritance from pipeline config
        pass
```

### 9.9 Pipeline Integration

```python
# Using PyMOL streaming in a DQ-Dock pipeline

pipeline = Pipeline([
    # Step 1: Compute interaction tensor
    FunctionStep(
        func=compute_interaction_tensor,
        variable_components=[VariableComponents.LIGAND],
        name="interaction_tensor"
    ),
    
    # Step 2: Stream intermediate structures to PyMOL
    FunctionStep(
        func=compute_poses,
        variable_components=[VariableComponents.LIGAND],
        pymol_streaming_config=LazyPyMOLStreamingConfig(
            representation=PyMOLRepresentation.STICK,
            auto_zoom=True
        ),
        name="pose_generation"
    ),
    
    # Step 3: Binding energy computation
    FunctionStep(
        func=compute_binding_energy,
        variable_components=[VariableComponents.LIGAND],
        name="binding_energy"
    ),
    
    # Step 4: Stream final results with ranking
    FunctionStep(
        func=rank_ligands,
        variable_components=[VariableComponents.LIGAND],
        pymol_streaming_config=LazyPyMOLStreamingConfig(
            representation=PyMOLRepresentation.SURFACE,
            color_by_score=True  # Color by binding energy
        ),
        name="final_results"
    ),
])
```

### 9.10 Integration with External Libraries

**PyMOL Options**:

1. **PyMOL Python API** (`pymol.cmd`): Direct control via Python
   - Most feature-rich
   - Supports all PyMOL features
   - Requires PyMOL installation

2. **RDKit MolViewer**: RDKit's built-in PyMOL viewer
   - Uses socket connection to PyMOL
   - Good for RDKit workflows

3. **PyMOL2 (open-source)**: Command-line focused
   - Can run headless
   - Good for batch processing

**Alternative: Molstar** (for web-based):
- **molstar**: Modern web-based molecular viewer
- **dash-molstar**: Dash component for web integration
- Could use WebSocket streaming instead of ZMQ

### 9.11 Implementation Phases

**Phase 1: Basic Streaming (Weeks 1-4)**
- [ ] Create PyMOLStreamingBackend in polystore style
- [ ] Implement PyMOLViewerServer with ZMQ
- [ ] Add basic PDB conversion utilities
- [ ] Test end-to-end streaming

**Phase 2: Enhanced Features (Weeks 5-8)**
- [ ] Add support for multiple representations (cartoon, stick, surface)
- [ ] Implement color schemes (by element, by energy, by component)
- [ ] Add camera control (zoom, rotate, pan)
- [ ] Support for trajectories/movie mode

**Phase 3: DQ-Dock Integration (Weeks 9-12)**
- [ ] Integrate with binding pose generation
- [ ] Add energy-based visualization (coloring by binding score)
- [ ] Support for interaction annotations (HBonds, pi-pi, salt bridges)
- [ ] Materialization-aware streaming

### 9.12 Key Design Decisions

1. **Why PyMOL?**
   - Industry standard for molecular visualization
   - Rich Python API
   - Supports all common molecular formats
   - Extensive visualization options

2. **Why not Molstar?**
   - Web-based (requires browser)
   - Less mature Python integration
   - PyMOL has better batch processing support

3. **Shared Memory vs Direct Transfer**
   - Use shared memory for large structures (protein-ligand complexes)
   - Direct ZMQ for small structures

4. **Port Allocation**
   - Napari: 5555
   - Fiji: 5556  
   - PyMOL: 5558 (leave room for other viewers)

### JAX Documentation
- https://jax.readthedocs.io/
- Key modules: `jax.numpy`, `jax.lax`, `jax.grad`, `jax.jit`
