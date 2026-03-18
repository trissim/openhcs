# OpenHCS Compliance Checklist

**Purpose**: Reference for implementing DQ-Dock Engine plans
**Date**: 2026-03-18
**Status**: Complete

---

## Six Fundamental Principles

### 1. ABC Contract Enforcement ✅
- [ ] Use ABCs for similar functionality
- [ ] `@abstractmethod` for required methods
- [ ] No defensive `isinstance()` checks

```python
class ScoringBackend(ABC):
    @abstractmethod
    def score_single(self, ...): pass
    
    @abstractmethod
    def score_batch(self, ...): pass
```

### 2. Explicit Dependency Injection ✅
- [ ] Dependencies as constructor parameters
- [ ] Factory functions for object creation
- [ ] No hidden `get_global_*()` calls

```python
def create_scoring_backend(
    family: ScoringFamily,
    config: ScoringConfig | None = None,
) -> ScoringBackend:
    # Explicit factory, no hidden creation
```

### 3. Indirection Minimization ✅
- [ ] Direct enum `match` statements
- [ ] No dispatch tables
- [ ] No `getattr(self, method_name)` routing

```python
match engine:
    case ScoringEngine.VINARDO:
        return VinardoBackend()
    case ScoringEngine.SOFT_LJ:
        return SoftLJBackend()
```

### 4. Genericism Enforcement ✅
- [ ] Use `Generic[T]` for type-safe generics
- [ ] Union types for alternatives
- [ ] No hardcoded assumptions

```python
@dataclass(frozen=True)
class ScoringConfig:
    config: VinardoConfig | SoftLJConfig
```

### 5. Fail-Loud Error Handling ✅
- [ ] No `getattr()` with fallbacks
- [ ] No `hasattr()` for guaranteed attributes
- [ ] No `try/except` catching natural errors
- [ ] `.validate()` raises on invalid

```python
def validate(self) -> None:
    if self.dielectric <= 0:
        raise ValueError(f"Dielectric must be positive, got {self.dielectric}")
```

### 6. Consistent Interface Design ✅
- [ ] ABC + Factory pattern
- [ ] Consistent method naming
- [ ] Uniform patterns across subsystems

---

## Code Review Checklist

### Defensive Programming (ELIMINATE)
- [ ] No `getattr(obj, 'field', default)` for guaranteed fields
- [ ] No `hasattr(obj, 'method')` for constructor-set attributes
- [ ] No `try/except AttributeError` just to provide defaults
- [ ] No `if hasattr(...)` for abstract method existence

### Dataclasses
- [ ] `@dataclass(frozen=True)` for all configuration
- [ ] `@dataclass(frozen=True)` for all results
- [ ] `@dataclass(frozen=True)` for immutable data containers
- [ ] Explicit types for all fields

```python
@dataclass(frozen=True)
class VinardoConfig:
    gaussians: tuple[tuple[float, float], ...] = ((-0.0356, 0.73), (-0.005, 1.25))
    repulsion: float = 0.840
    cutoff: float = 8.0
    
    def validate(self) -> None:
        if not (6.0 <= self.cutoff <= 12.0):
            raise ValueError(f"Cutoff {self.cutoff} outside range")
```

### Enums
- [ ] Enum-driven behavior selection
- [ ] No magic strings
- [ ] `Enum.auto()` for values

```python
class ScoringFamily(Enum):
    VINARDO = auto()
    SOFT_LJ = auto()
    STANDARD_LJ = auto()
```

### Imports
- [ ] All imports at module level
- [ ] No inline imports in functions
- [ ] `from __future__ import annotations` for forward references

### Type Hints
- [ ] All functions have explicit return types
- [ ] All parameters have explicit types
- [ ] Use `tuple[X, ...]` for immutable sequences
- [ ] Use `| ` for union types (Python 3.10+)

```python
def create_scoring_backend(
    family: ScoringFamily,
    config: VinardoConfig | SoftLJConfig | None = None,
) -> ScoringBackend:
    ...
```

### JAX Functions
- [ ] Pure functions (no side effects)
- [ ] `@jax.jit` for GPU acceleration
- [ ] Explicit array shapes in docstrings

```python
@jax.jit
def score_vinardo_single(
    receptor_coords: jnp.ndarray,  # (N_rec, 3)
    pose_coords: jnp.ndarray,        # (N_lig, 3)
    config: VinardoConfig,
) -> float:
    ...
```

### Error Handling
- [ ] Fail-loud validation
- [ ] Specific exception types
- [ ] No silent fallbacks
- [ ] Error context preserved with `from e`

```python
def validate(self) -> None:
    if self.dielectric <= 0:
        raise ValueError(f"Dielectric must be positive, got {self.dielectric}")
```

---

## Pattern Library

### Pattern: ABC Contract

```python
class ScoringBackend(ABC):
    @property
    @abstractmethod
    def config(self) -> ScoringConfig:
        """Direct access to config."""
    
    @abstractmethod
    def score_single(self, ...) -> float:
        """Score single pose."""
    
    @abstractmethod
    def score_batch(self, ...) -> jnp.ndarray:
        """Score batch of poses."""
```

### Pattern: Frozen Dataclass Config

```python
@dataclass(frozen=True)
class ScoringConfig:
    param1: float = 1.0
    param2: int = 2
    
    def validate(self) -> None:
        if self.param1 <= 0:
            raise ValueError(f"param1 must be positive")
```

### Pattern: Factory Function

```python
def create_scoring_backend(
    family: ScoringFamily,
    config: ScoringConfig | None = None,
) -> ScoringBackend:
    cfg = config if config is not None else ScoringConfig()
    cfg.validate()
    
    match family:
        case ScoringFamily.VINARDO:
            return VinardoBackend(cfg)
        case _:
            raise ValueError(f"Unknown family: {family}")
```

### Pattern: Enum Dispatch

```python
match engine:
    case ScoringEngine.VINARDO:
        return np.array(score_vinardo_batch(...))
    case ScoringEngine.SOFT_LJ:
        return np.array(score_soft_lj_batch(...))
    case _:
        raise ValueError(f"Unknown engine: {engine}")
```

### Pattern: Immutable Results

```python
@dataclass(frozen=True)
class ScoringResult:
    scores: jnp.ndarray
    n_poses: int
    runtime_ms: float
```

---

## Literature + OpenHCS Integration

### Per-File Checklist

| File | ABC | Frozen | Enum | Factory | Fail-Loud | Types |
|------|-----|--------|------|---------|-----------|-------|
| `scoring_vinardo.py` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `hbonds_ad4.py` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `sasa.py` | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ |
| `desolvation_ad4.py` | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| `charges.py` | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| `electrostatics.py` | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| `scoring_composite.py` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `scoring_stages.py` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `pocket_analysis.py` | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |

Legend:
- ABC: ABC contract class defined
- Frozen: `@dataclass(frozen=True)` used
- Enum: Enum for behavior selection
- Factory: Factory function defined
- Fail-Loud: `.validate()` method raises on invalid
- Types: Explicit type hints on all functions

---

## Validation Commands

```bash
# Check for defensive programming patterns
grep -r "getattr.*default" --include="*.py"
grep -r "hasattr.*" --include="*.py"
grep -r "try:.*except.*AttributeError" --include="*.py"

# Check for frozen dataclasses
grep -r "@dataclass" --include="*.py" | grep -v "frozen=True"

# Check for imports at top level
grep -r "def .*:$" --include="*.py" -A 5 | grep "import "

# Run type checker
mypy dq_dock_engine/ --strict

# Run linter
ruff check dq_dock_engine/

# Run tests
pytest tests/ -v
```

---

## Summary

All DQ-Dock Engine implementations must follow OpenHCS principles:

1. **ABC Contract Enforcement** - Explicit interfaces
2. **Explicit Dependency Injection** - No hidden dependencies
3. **Indirection Minimization** - Direct method calls
4. **Genericism Enforcement** - Type-safe generics
5. **Fail-Loud Error Handling** - No silent failures
6. **Consistent Interface Design** - Uniform patterns

Every code review should verify:
- ✅ No defensive programming patterns
- ✅ All dataclasses frozen
- ✅ All enums for behavior selection
- ✅ Factory functions for object creation
- ✅ Explicit type hints
- ✅ Fail-loud validation
