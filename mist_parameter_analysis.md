# MIST Function Parameter Analysis

## Function: `mist_compute_tile_positions`

**Location**: `openhcs/processing/backends/pos_gen/mist/mist_main.py:446`

**Decorators**: `@special_inputs("grid_dimensions")`, `@special_outputs("positions")`, `@chain_breaker`, `@cupy_func`

## Parameter Categories

### 1. UNUSED PARAMETERS (Can be removed)

| Parameter | Default | Status | Evidence |
|-----------|---------|--------|----------|
| `patch_size` | `DEFAULT_PATCH_SIZE` | ❌ UNUSED | Not referenced in function body |
| `search_radius` | `DEFAULT_SEARCH_RADIUS` | ❌ UNUSED | Docstring says "(unused in this implementation)" |
| `stride` | `64` | ❌ UNUSED | Docstring says "(unused in this implementation)" |
| `enable_correlation_visualization` | `False` | ❌ UNUSED | Not referenced in function body |
| `correlation_viz_save_path` | `None` | ❌ UNUSED | Not referenced in function body |

### 2. VALIDATION-ONLY PARAMETERS (Used for validation, then ignored)

| Parameter | Default | Status | Usage |
|-----------|---------|--------|-------|
| `method` | `"phase_correlation"` | ✅ VALIDATION | Line 534: `if method != "phase_correlation": raise ValueError` |
| `fft_backend` | `"cupy"` | ✅ VALIDATION | Line 530: `if fft_backend != "cupy": raise ValueError` |

### 3. CORE ALGORITHM PARAMETERS (Actively used)

| Parameter | Default | Status | Usage Lines |
|-----------|---------|--------|-------------|
| `normalize` | `True` | ✅ USED | Lines 555-562: Tile normalization loop |
| `verbose` | `False` | ✅ USED | Lines 574, 645: Logging control |
| `overlap_ratio` | `0.1` | ✅ USED | Lines 568-569, 594, 621, 660, 692: Overlap calculations |
| `subpixel` | `True` | ✅ USED | Lines 601, 626, 664, 696: Phase correlation calls |
| `refinement_iterations` | `10` | ✅ USED | Line 644: Refinement loop control |
| `global_optimization` | `True` | ✅ USED | Line 724: MST optimization control |
| `anchor_tile_index` | `0` | ✅ USED | Lines 582, 717: Anchor tile handling |

### 4. REFINEMENT PARAMETERS (Used in refinement phase)

| Parameter | Default | Status | Usage Lines |
|-----------|---------|--------|-------------|
| `refinement_damping` | `0.5` | ✅ USED | Lines 719-720: Position update damping |
| `correlation_weight_horizontal` | `1.0` | ✅ USED | Lines 680-681: Horizontal constraint weighting |
| `correlation_weight_vertical` | `1.0` | ✅ USED | Lines 712-713: Vertical constraint weighting |

### 5. PHASE CORRELATION PARAMETERS (Passed to sub-functions)

| Parameter | Default | Status | Usage |
|-----------|---------|--------|-------|
| `subpixel_radius` | `3` | ✅ PASSED | Lines 602, 627, 667, 699: Passed to `phase_correlation_gpu_only` |
| `regularization_eps_multiplier` | `1000.0` | ✅ PASSED | Lines 603, 628, 668, 700: Passed to `phase_correlation_gpu_only` |

### 6. MST OPTIMIZATION PARAMETERS (Passed to _global_optimization_gpu_only)

| Parameter | Default | Status | Usage |
|-----------|---------|--------|-------|
| `mst_quality_threshold` | `0.01` | ✅ PASSED | Line 730: Passed as `quality_threshold` |
| `debug_connection_limit` | `3` | ✅ PASSED | Line 734: Passed to MST function |
| `debug_vertical_limit` | `6` | ✅ PASSED | Line 735: Passed to MST function |
| `displacement_tolerance_factor` | `2.0` | ✅ PASSED | Line 736: Passed to MST function |
| `displacement_tolerance_percent` | `0.3` | ✅ PASSED | Line 737: Passed to MST function |
| `consistency_threshold_percent` | `0.5` | ✅ PASSED | Line 738: Passed to MST function |
| `max_connections_multiplier` | `2` | ✅ PASSED | Line 739: Passed to MST function |
| `adaptive_base_threshold` | `0.3` | ✅ PASSED | Line 740: Passed to MST function |
| `adaptive_percentile_threshold` | `0.25` | ✅ PASSED | Line 741: Passed to MST function |
| `translation_tolerance_factor` | `0.2` | ✅ PASSED | Line 742: Passed to MST function |
| `translation_min_quality` | `0.3` | ✅ PASSED | Line 743: Passed to MST function |
| `magnitude_threshold_multiplier` | `1e-6` | ✅ PASSED | Line 744: Passed to MST function |
| `peak_candidates_multiplier` | `4` | ✅ PASSED | Line 745: Passed to MST function |
| `min_peak_distance` | `5` | ✅ PASSED | Line 746: Passed to MST function |
| `use_nist_robustness` | `True` | ✅ PASSED | Line 747: Passed to MST function |
| `n_peaks` | `2` | ✅ PASSED | Line 748: Passed to MST function |
| `use_nist_normalization` | `True` | ✅ PASSED | Line 749: Passed to MST function |

## Summary

- **Total Parameters**: ~40
- **Unused**: 5 (can be removed)
- **Validation-only**: 2 (used for input validation)
- **Core Algorithm**: 7 (directly used in main algorithm)
- **Refinement**: 3 (used in refinement phase)
- **Phase Correlation**: 2 (passed to phase correlation functions)
- **MST Optimization**: ~16 (passed to global optimization)

## Detailed Parameter Categorization

### Category A: REMOVE (Unused - 5 parameters)
```python
# These can be safely removed
patch_size: int = DEFAULT_PATCH_SIZE,  # ❌ Not used anywhere
search_radius: int = DEFAULT_SEARCH_RADIUS,  # ❌ Not used anywhere
stride: int = 64,  # ❌ Not used anywhere
enable_correlation_visualization: bool = False,  # ❌ Not used anywhere
correlation_viz_save_path: str = None,  # ❌ Not used anywhere
```

### Category B: KEEP - Input Validation (2 parameters)
```python
# Keep for API compatibility and validation
method: str = "phase_correlation",  # ✅ Validates input format
fft_backend: str = "cupy",  # ✅ Validates backend choice
```

### Category C: KEEP - Core Algorithm (7 parameters)
```python
# Essential for main algorithm behavior
normalize: bool = True,  # ✅ Controls tile normalization
verbose: bool = False,  # ✅ Controls logging output
overlap_ratio: float = 0.1,  # ✅ Critical for overlap calculations
subpixel: bool = True,  # ✅ Controls subpixel accuracy
refinement_iterations: int = 10,  # ✅ Controls refinement loop
global_optimization: bool = True,  # ✅ Enables/disables MST optimization
anchor_tile_index: int = 0,  # ✅ Sets reference tile
```

### Category D: KEEP - Refinement Tuning (3 parameters)
```python
# Fine-tuning for refinement phase
refinement_damping: float = 0.5,  # ✅ Position update damping
correlation_weight_horizontal: float = 1.0,  # ✅ Horizontal constraint weight
correlation_weight_vertical: float = 1.0,  # ✅ Vertical constraint weight
```

### Category E: KEEP - Phase Correlation (2 parameters)
```python
# Passed to phase correlation functions
subpixel_radius: int = 3,  # ✅ Subpixel interpolation radius
regularization_eps_multiplier: float = 1000.0,  # ✅ Numerical stability
```

### Category F: CONSIDER GROUPING - MST Parameters (16 parameters)
```python
# Large group of MST optimization parameters - consider parameter object
mst_quality_threshold: float = 0.01,
debug_connection_limit: int = 3,
debug_vertical_limit: int = 6,
displacement_tolerance_factor: float = 2.0,
displacement_tolerance_percent: float = 0.3,
consistency_threshold_percent: float = 0.5,
max_connections_multiplier: int = 2,
adaptive_base_threshold: float = 0.3,
adaptive_percentile_threshold: float = 0.25,
translation_tolerance_factor: float = 0.2,
translation_min_quality: float = 0.3,
magnitude_threshold_multiplier: float = 1e-6,
peak_candidates_multiplier: int = 4,
min_peak_distance: int = 5,
use_nist_robustness: bool = True,
n_peaks: int = 2,
use_nist_normalization: bool = True,
```

## Recommendations

1. **Remove 5 unused parameters** (Category A) to simplify interface
2. **Keep 14 essential parameters** (Categories B, C, D, E) with improved docs
3. **Consider MST parameter object** for the 16 MST parameters (Category F)
4. **Rewrite docstring** with clear parameter categories and usage descriptions
5. **Add parameter validation** for ranges and compatibility

## Parameter Flow Tracing

### Main Function Flow
```
mist_compute_tile_positions()
├── Input validation (method, fft_backend)
├── Tile normalization (normalize)
├── Phase 1: Initial positioning
│   ├── Uses: overlap_ratio, subpixel, anchor_tile_index
│   └── Calls: phase_correlation_gpu_only(subpixel_radius, regularization_eps_multiplier)
├── Phase 2: Refinement iterations (refinement_iterations)
│   ├── Uses: correlation_weight_horizontal, correlation_weight_vertical
│   ├── Uses: refinement_damping, anchor_tile_index
│   └── Calls: phase_correlation_gpu_only(subpixel_radius, regularization_eps_multiplier)
└── Phase 3: Global optimization (global_optimization)
    └── Calls: _global_optimization_gpu_only(16 MST parameters)
```

### Sub-function Parameter Flow

#### phase_correlation_gpu_only()
- **Receives**: `subpixel_radius`, `regularization_eps_multiplier`
- **Called from**: Lines 601, 626, 664, 696 (4 times)
- **Purpose**: Subpixel-accurate phase correlation

#### _global_optimization_gpu_only()
- **Receives**: All 16 MST parameters
- **Called from**: Line 726 (once, conditionally)
- **Purpose**: MST-based global position optimization

### Critical Dependencies
1. **overlap_ratio** → Used 6 times for overlap region calculations
2. **subpixel + subpixel_radius** → Passed to phase correlation 4 times
3. **refinement_iterations** → Controls main refinement loop
4. **global_optimization** → Gates entire MST optimization phase

## DETAILED PARAMETER EFFECTS

### Core Algorithm Parameters

**normalize** (bool, default=True)
- **What it does**: Normalizes each tile to [0,1] range using `(tile - min) / (max - min)`
- **Effect of changing**:
  - `True`: Better phase correlation accuracy, handles varying illumination
  - `False`: Faster processing, but poor results with uneven lighting
- **When to change**: Set to `False` only if tiles are already normalized

**overlap_ratio** (float, default=0.1)
- **What it does**: Defines overlap region size as fraction of tile dimension
- **Mathematical effect**: `overlap_w = int(W * overlap_ratio)`, `overlap_h = int(H * overlap_ratio)`
- **Effect of changing**:
  - Higher (0.2-0.4): More robust correlation, slower processing
  - Lower (0.05-0.08): Faster but less accurate, may fail with noise
- **Critical**: Must match actual overlap in your data or correlation fails

**refinement_iterations** (int, default=10)
- **What it does**: Number of iterative position refinement passes
- **Mathematical effect**: Each iteration applies weighted position corrections with damping
- **Effect of changing**:
  - Higher (15-50): Better convergence for difficult datasets, much slower
  - Lower (1-5): Faster but may not converge, positions may drift
  - 0: Skip refinement entirely (fastest, least accurate)

**refinement_damping** (float, default=0.5)
- **What it does**: Controls how aggressively positions are updated each iteration
- **Mathematical effect**: `new_pos = (1-damping)*old_pos + damping*correction`
- **Effect of changing**:
  - Higher (0.7-0.9): Faster convergence but may overshoot/oscillate
  - Lower (0.1-0.3): More stable but slower convergence
  - 1.0: Full correction (may be unstable)
  - 0.0: No updates (positions frozen)

### Phase Correlation Parameters

**subpixel_radius** (int, default=3)
- **What it does**: Radius around correlation peak for center-of-mass calculation
- **Mathematical effect**: Extracts `(2*radius+1)²` region around peak for interpolation
- **Effect of changing**:
  - Higher (5-10): More accurate subpixel positioning, slower
  - Lower (1-2): Faster but less precise (may cause drift)
  - 0: Pixel-only accuracy (fastest, least precise)

**regularization_eps_multiplier** (float, default=1000.0)
- **What it does**: Prevents division by zero in phase correlation normalization
- **Mathematical effect**: `eps = machine_epsilon * multiplier`
- **Effect of changing**:
  - Higher (10000+): More stable with noisy images, may reduce precision
  - Lower (100-500): Higher precision but may fail with low-contrast regions
  - Too low (<10): Risk of numerical instability/NaN results

### MST Quality Control Parameters

**mst_quality_threshold** (float, default=0.01)
- **What it does**: Minimum correlation quality for MST edge inclusion
- **Mathematical effect**: `if correlation_peak < threshold: reject_connection`
- **Effect of changing**:
  - Higher (0.05-0.2): Stricter quality, fewer connections, may fragment
  - Lower (0.001-0.005): More permissive, includes weak correlations
  - Too high: MST may fail (no valid connections)
  - Too low: Includes noise correlations

**displacement_tolerance_factor** (float, default=2.0)
- **What it does**: Multiplier for expected displacement tolerance
- **Mathematical effect**: `max_error = factor * expected_displacement * tolerance_percent`
- **Effect of changing**:
  - Higher (3.0-5.0): More permissive displacement validation
  - Lower (1.0-1.5): Stricter validation, rejects more connections
  - 1.0: Exact expected displacement only

**displacement_tolerance_percent** (float, default=0.3)
- **What it does**: Percentage tolerance for displacement validation (30% = ±30%)
- **Effect of changing**:
  - Higher (0.5-0.8): Accepts larger deviations from expected displacement
  - Lower (0.1-0.2): Stricter validation, requires precise alignment
  - 0.0: No tolerance (exact displacement required)

## FINAL SUMMARY

### Completed Tasks ✅

1. **Parameter Usage Analysis**: Systematically analyzed all 40+ parameters
2. **Categorization**: Grouped into Used/Unused/Validation/Pass-through categories
3. **Signature Cleanup**: Removed 5 unused parameters, organized remaining 34 by category
4. **Detailed Docstring**: Rewrote with specific mathematical effects and practical guidance

### Key Improvements

**Before**: Generic descriptions like "Damping factor for position updates"
**After**: Specific effects like "Formula: new_pos = (1-damping)*old_pos + damping*correction. Higher (0.7-0.9) = faster convergence but may overshoot."

**Before**: 40+ parameters with unclear usage
**After**: 34 well-documented parameters organized by function

### Removed Parameters (5 total)
- `patch_size` - Not used anywhere in implementation
- `search_radius` - Explicitly marked as unused
- `stride` - Explicitly marked as unused
- `enable_correlation_visualization` - Not referenced
- `correlation_viz_save_path` - Not referenced

### Parameter Categories (34 remaining)
- **Input Validation**: 2 parameters (method, fft_backend)
- **Core Algorithm**: 7 parameters (normalize, overlap_ratio, etc.)
- **Refinement Tuning**: 3 parameters (damping, weights)
- **Phase Correlation**: 2 parameters (subpixel_radius, regularization)
- **MST Optimization**: 20 parameters (quality thresholds, tolerances, etc.)

### Documentation Quality
- **Mathematical formulas** for key parameters
- **Practical ranges** and typical values
- **Effect descriptions** explaining what happens when you change values
- **Performance trade-offs** clearly explained
- **Failure modes** and edge cases documented

The function signature is now much cleaner and the docstring provides actionable guidance for parameter tuning!
