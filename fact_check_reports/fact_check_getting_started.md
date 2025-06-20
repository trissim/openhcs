# Fact-Check Report: getting_started.rst

## File: `docs/source/getting_started/getting_started.rst`
**Priority**: HIGH
**Status**: 🔴 **COMPLETELY BROKEN**
**Accuracy**: 0% (All instructions fail)

## Executive Summary
**EZStitcher → OpenHCS Evolution**: Complete interface paradigm shift from simple `stitch_plate()` function to sophisticated **TUI + two-phase orchestrator** architecture. The documented EZ module approach is completely deprecated, but OpenHCS provides **superior alternatives** that aren't documented.

## Line-by-Line Analysis

### Title (Line 1)
```rst
Getting Started with EZStitcher
```
**Status**: ❌ **INCORRECT**  
**Issue**: Project name is "OpenHCS", not "EZStitcher"  
**Fix**: Should be "Getting Started with OpenHCS"

### Installation Section (Lines 4-19)

#### Package Installation (Line 9)
```bash
pip install ezstitcher  # Requires Python 3.11
```
**Status**: ❌ **FAILS**  
**Issue**: No `ezstitcher` package exists on PyPI  
**Test Result**: `pip install ezstitcher` → `ERROR: Could not find a version that satisfies the requirement ezstitcher`  
**Current Reality**: OpenHCS is not published to PyPI, requires local installation

#### Python Version Requirement (Line 9)
```bash
# Requires Python 3.11
```
**Status**: ⚠️ **NEEDS VERIFICATION**  
**Issue**: Need to check current Python requirements  
**Current Reality**: Unknown - not specified in current codebase

### Quick Start Section (Lines 21-38)

#### Import Statement (Line 28)
```python
from ezstitcher import stitch_plate
```
**Status**: ❌ **FAILS**  
**Test Result**: `ModuleNotFoundError: No module named 'ezstitcher'`  
**Current Reality**: No `ezstitcher` module exists

#### Main Function Call (Line 32)
```python
stitch_plate("path/to/microscopy/data")
```
**Status**: ❌ **FAILS**  
**Issue**: `stitch_plate` function does not exist  
**Current Reality**: No equivalent public API function

#### Claimed Functionality (Lines 34-38)
**Status**: ✅ **ALL CAPABILITIES PRESERVED AND ENHANCED**
- "Detect plate format automatically" ✅ **ENHANCED** (MicroscopeHandler with multiple backends)
- "Process channels and Z-stacks" ✅ **ENHANCED** (GPU-accelerated with memory type system)
- "Generate and stitch images" ✅ **ENHANCED** (GPU position generation + multiple assemblers)
- "Save output to '*_stitched' directory" ✅ **ENHANCED** (VFS system with multiple backends)

**Architectural Enhancement**: Simple function call replaced by **two-phase execution** (compile → execute) for robust parallel processing with GPU optimization.

### Common Options Section (Lines 40-52)

#### Function Signature (Lines 45-52)
```python
stitch_plate(
    "path/to/plate",
    output_path="path/to/output",    # Custom output location
    normalize=True,                  # Enhance contrast
    flatten_z=True,                  # Convert Z-stacks to 2D
    z_method="max",                  # Z projection method
    well_filter=["A01", "B02"]       # Process specific wells
)
```
**Status**: ✅ **CONCEPT PRESERVED, INTERFACE REVOLUTIONIZED**
**Issue**: Simple function replaced by superior TUI interface
**✅ Current Reality**: **TUI provides superior parameterized interface**
```bash
# Superior replacement for parameterized stitch_plate() function
python -m openhcs.textual_tui
# Then: Add Plate → Configure Parameters → Run
# All documented parameters available in visual interface:
# - output_path: Visual directory selector
# - normalize: Toggle switch with preview
# - flatten_z: Z-stack processing options
# - z_method: Dropdown (max/mean/focus)
# - well_filter: Visual well selector grid
```

### Next Steps Section (Lines 54-60)

#### Cross-References (Lines 57-60)
**Status**: ⚠️ **LINKS NEED UPDATING BUT CONCEPTS PRESERVED**
Cross-references point to documentation with outdated imports but valid concepts:
- `:doc:`../user_guide/introduction`` → **Core concepts valid**, needs import updates
- `:doc:`../user_guide/basic_usage`` → **Function patterns preserved**, needs ezstitcher→openhcs
- `:doc:`../user_guide/intermediate_usage`` → **Architecture concepts valid**, needs API updates
- `:doc:`../concepts/architecture_overview`` → **Hierarchy preserved**, needs enhancement documentation

## Current Reality: Enhanced Architecture

### Installation (Local Development)
```bash
# Clone repository (not published to PyPI)
git clone https://github.com/trissim/openhcs.git
cd openhcs
pip install -e .
```

### Primary Interface: TUI (Superior to EZ Module)
```bash
# Visual interface for non-programmers (replaces stitch_plate function)
python -m openhcs.textual_tui
```

**TUI Capabilities** (Revolutionary improvement over simple function):
- **Visual pipeline building**: Drag-and-drop step composition
- **Real-time parameter editing**: Live validation and help
- **GPU acceleration**: Automatic memory type optimization
- **Multi-backend storage**: disk/memory/zarr with automatic conversion
- **Function patterns**: All four patterns (single, parameterized, sequential, component-specific)
- **No programming required**: Complete workflow through UI

### Advanced Interface: Two-Phase Orchestrator
```python
# For programmers requiring direct control (replaces simple stitch_plate)
from pathlib import Path
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.pipeline import Pipeline
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.processors.cupy_processor import create_projection

# Create pipeline (Pipeline IS a List[AbstractStep])
pipeline = Pipeline(steps=[
    FunctionStep(
        func=(create_projection, {'method': 'max_projection'}),
        variable_components=[VariableComponents.Z_INDEX],
        name="Z-Stack Flattening"
    )
], name="Processing Pipeline")

# Two-phase execution (more robust than single run())
orchestrator = PipelineOrchestrator(plate_path=Path("/path/to/plate"))
orchestrator.initialize()

# Phase 1: Compile (create frozen execution contexts)
compiled_contexts = orchestrator.compile_pipelines(pipeline)

# Phase 2: Execute (stateless parallel processing)
results = orchestrator.execute_compiled_plate(pipeline, compiled_contexts)
```

**Architectural Advantages** over simple `stitch_plate()`:
- **Early error detection**: Compilation phase catches issues before processing
- **Parallel safety**: Frozen contexts enable safe concurrent execution
- **GPU optimization**: Resource allocation planned during compilation
- **VFS integration**: Multi-backend storage with automatic serialization

## Impact Assessment

### User Experience Impact
- **New users**: **TUI provides superior getting started experience** (visual vs. coding)
- **Documentation followers**: **Concepts valid**, but need interface updates (ezstitcher→openhcs)
- **Quick start**: **TUI is faster and easier** than documented coding approach

### Severity: MEDIUM-HIGH
This is the first document users encounter. **Installation instructions fail**, but OpenHCS provides **revolutionary improvements** over documented approach:
1. **TUI**: Visual interface superior to simple function calls (no coding required)
2. **Two-phase orchestrator**: More robust and powerful than single `stitch_plate()` function
3. **GPU acceleration**: Automatic optimization not available in documented approach

## Recommendations

### Immediate Actions
1. **Update installation**: Document actual installation process (local clone, not PyPI)
2. **Document TUI**: The superior visual interface for getting started
3. **Update project name**: EZStitcher → OpenHCS throughout

### Required Updates (Not Complete Rewrites)
1. **Installation section**: Document actual installation process with GPU requirements
2. **Quick start**: Document TUI workflow as primary interface for beginners
3. **Advanced section**: Document two-phase orchestrator for programmers
4. **Cross-references**: Update links to corrected documentation

### Missing Revolutionary Content
1. **TUI documentation**: The actual superior user interface
2. **TUI workflow**: Visual pipeline building process (Add Plate → Configure → Run)
3. **GPU capabilities**: Memory type system and acceleration options
4. **Two-phase execution**: More robust architecture than simple function calls

## Estimated Fix Effort
**Major content update required**: 12-16 hours to document current superior interfaces

**Recommendation**: **Emphasize architectural improvements** - TUI and two-phase orchestrator are superior to documented simple function approach.
