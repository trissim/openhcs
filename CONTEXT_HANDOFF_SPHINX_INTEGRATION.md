# CONTEXT HANDOFF: SPHINX DOCUMENTATION INTEGRATION

## MISSION CRITICAL UNDERSTANDING

**CORRECTION**: OpenHCS IS the literal evolution of EZStitcher within the same repository. Current working directory `/home/ts/code/projects/openhcs` is the `openhcs` branch of the `ezstitcher` repository. The main branch contains the original CPU stitching tool; the openhcs branch evolved it into a GPU-native bioimage analysis platform.

**Repository Structure**:
- `ezstitcher` repo main branch: Original CPU-based stitching tool
- `ezstitcher` repo openhcs branch: Evolved GPU-native bioimage analysis platform (current working directory)

**Core Gap**: Sphinx docs describe EZStitcher's original capabilities, but OpenHCS preserved and extended the architectural patterns while adding major new systems.

## ARCHITECTURAL EVOLUTION ANALYSIS

### What EZStitcher Already Had (Preserved in OpenHCS)
- **Function Patterns**: Single, tuple `(func, kwargs)`, list `[func1, func2]`, dict `{"1": func1, "2": func2}`
- **Variable Components**: `variable_components=['channel']` for intelligent grouping
- **Group By**: `group_by='channel'` for component-specific processing
- **Pipeline Architecture**: PipelineOrchestrator → Pipeline → Step hierarchy
- **Specialized Steps**: ZFlatStep, CompositeStep, PositionGenerationStep, ImageStitchingStep

### What OpenHCS Added (New Documentation Needed)
- **GPU-Native Processing**: CuPy/PyTorch/JAX with automatic memory type conversion
- **574+ Function Registry**: Auto-discovered from pyclesperanto, scikit-image, CuCIM
- **4-Phase Compilation**: Path planning → Materialization → Contract validation → GPU assignment
- **Production TUI**: SSH-compatible terminal interface with real-time editing
- **Special I/O System**: Cross-step communication for analysis results (cell counts, measurements)
- **ZARR Backend**: 100GB+ dataset handling with memory overlay VFS
- **Bioimage Analysis**: Cell counting, neurite tracing, morphological analysis (expanded from stitching)

## DOCUMENTATION AUDIT STATUS

### ✅ ARCHITECTURE DOCUMENTATION (READY FOR INTEGRATION)
- **Location**: `docs/architecture/` (22 files, 95%+ accuracy verified)
- **Crown Jewels**: `function-pattern-system-unified.md`, `memory-type-system.md`, `pipeline-compilation-system.md`
- **Integration Ready**: All files fact-checked against actual implementation

## SPHINX DOCUMENTATION STATUS

### 🔴 CRITICAL: MIXED EZSTITCHER/OPENHCS STATE
**Location**: `docs/source/` (Sphinx documentation structure)

**UPDATED TO OPENHCS**:
- ✅ `index.rst` - Main landing page
- ✅ `getting_started/getting_started.rst` - Basic examples

**STILL EZSTITCHER (NEEDS MIGRATION)**:
- ❌ `conf.py` - Project configuration still says "EZStitcher"
- ❌ `development/architecture.rst` - Complete EZStitcher architecture (141 lines)
- ❌ `user_guide/basic_usage.rst` - 9+ EZStitcher references throughout
- ❌ `api/ez.rst` - Complete EZStitcher API documentation
- ❌ `concepts/architecture_overview.rst` - "EZStitcher is built around..."

## ACTIONABLE INTEGRATION STRATEGY

### PHASE 1: TERMINOLOGY AND EXAMPLES UPDATE (HIGH PRIORITY)
**Approach**: Update existing docs rather than complete rewrite - architectural patterns are solid

**IMMEDIATE TASKS**:
1. **conf.py**: `project = 'EZStitcher'` → `'OpenHCS'` (line 54)
2. **Import Updates**: Replace `from ezstitcher import` with `from openhcs import`
3. **Example Updates**: Replace `stitch_plate()` calls with `FunctionStep` pipeline examples
4. **API References**: Update module paths from `ezstitcher.core.*` to `openhcs.core.*`
5. **Terminology**: Update "stitching" focus to "bioimage analysis" scope

**Key Files to Update**:
- `docs/source/conf.py` (lines 54, 165)
- `docs/source/user_guide/basic_usage.rst` (lines 43-48, examples throughout)
- `docs/source/development/architecture.rst` (EZStitcher references)
- `docs/source/concepts/architecture_overview.rst` (line 8: "EZStitcher is built around...")

### PHASE 2: NEW SYSTEMS DOCUMENTATION (MEDIUM PRIORITY)
**Approach**: Add new sections for systems that didn't exist in EZStitcher

**NEW DOCUMENTATION NEEDED**:
1. **Memory Type System**: Cross-library conversion (NumPy↔CuPy↔PyTorch)
2. **Function Registry**: 574+ auto-discovered GPU functions
3. **TUI System**: SSH-compatible terminal interface
4. **Special I/O**: Cross-step communication for analysis results
5. **ZARR Backend**: Large dataset storage and memory overlay
6. **GPU Management**: Device allocation and compilation-time assignment

**Integration Options**:
- Option A: New `docs/source/architecture/` section with converted .md→.rst files
- Option B: Expand existing sections with new subsections
- **RECOMMENDED**: Option A for clean separation

### PHASE 3: PRODUCTION EXAMPLES INTEGRATION
**Approach**: Show real OpenHCS workflows, not just stitching

**EXAMPLE SOURCES**:
- `openhcs/debug/example_export.py` - Real production pipeline script
- `openhcs/debug/scenarios/special_io_test.py` - Dict patterns with analysis
- TUI-generated scripts - Self-contained executable examples

**WORKFLOW EXAMPLES TO ADD**:
- Cell counting + neurite tracing pipeline (channel-specific dict patterns)
- GPU memory type conversion workflows
- Large dataset processing with ZARR backend
- TUI-based pipeline creation and execution

## KEY FILES AND LOCATIONS

### ARCHITECTURE DOCS (SOURCE MATERIAL)
```
docs/architecture/
├── README.md - Documentation overview and confidence assessment
├── function-pattern-system-unified.md - Merged function patterns (CONSOLIDATED)
├── memory-type-system.md - Core memory type architecture
├── pipeline-compilation-system.md - High-level compilation overview
├── gpu-resource-management.md - GPU coordination system
├── vfs-backend-system.md - Virtual file system
└── [16 other verified architecture docs]
```

### SPHINX STRUCTURE (TARGET)
```
docs/source/
├── conf.py - Sphinx configuration (NEEDS EZSTITCHER→OPENHCS UPDATE)
├── index.rst - Main page (UPDATED)
├── getting_started/ - User onboarding (UPDATED)
├── user_guide/ - Usage examples (NEEDS EZSTITCHER→OPENHCS UPDATE)
├── concepts/ - Conceptual overview (NEEDS EZSTITCHER→OPENHCS UPDATE)
├── development/ - Developer docs (NEEDS COMPLETE REWRITE)
├── api/ - API reference (NEEDS EZSTITCHER→OPENHCS UPDATE)
└── [PROPOSED] architecture/ - Deep architecture reference (NEW)
```

## PRESERVED AI COLLABORATION CONTENT

### CORE FRAMEWORK (HIGH VALUE)
```
docs/ai_collaboration/
├── README.md - Framework overview
├── ai-collaboration-framework.md - Core breakthrough collaboration techniques
├── transparency-protocol.md - Implementation guide
├── collaborative-dynamics.md - Four-way collaboration model
├── internalization.md - Real-time framework documentation
├── dry_run_methodology.md - Systematic testing methodology (RESTORED)
└── source_inspection_methodology.md - Code analysis methodology (RESTORED)
```

## SEMANTIC COMPRESSION: CORE SYSTEMS OVERVIEW

### FUNCTION PATTERN SYSTEM (PRESERVED FROM EZSTITCHER)
**EZStitcher Foundation**: Already documented function patterns (single, tuple, list, dict) with `variable_components` and `group_by`
**OpenHCS Extension**: Same patterns now work with 574+ auto-discovered GPU functions
**Integration**: Update examples to show GPU functions, preserve existing pattern documentation

### MEMORY TYPE SYSTEM (NEW IN OPENHCS)
**Innovation**: Automatic conversion between NumPy↔CuPy↔PyTorch↔TensorFlow↔JAX with zero-copy GPU operations
**Implementation**: Decorator-based contracts (`@cupy`, `@torch`) with compile-time validation
**Integration**: New section needed - no EZStitcher equivalent

### 4-PHASE COMPILATION SYSTEM (NEW IN OPENHCS)
**Phases**: Path planning → Materialization → Contract validation → GPU assignment
**Innovation**: Immutable execution contexts compiled once, then stateless execution
**Integration**: Expand developer docs - EZStitcher had basic pipeline execution

### TUI SYSTEM (NEW IN OPENHCS)
**Innovation**: Production-grade terminal interface for scientific computing (unprecedented)
**Features**: Real-time pipeline editing, SSH compatibility, professional monitoring
**Integration**: New user guide section - no EZStitcher equivalent

### SPECIAL I/O SYSTEM (NEW IN OPENHCS)
**Purpose**: Cross-step communication for analysis results (cell counts, measurements)
**Implementation**: Decorator-based contracts (`@special_inputs`, `@special_outputs`)
**Integration**: New section - EZStitcher only handled image data flow

## CONVERSION TOOLS AND WORKFLOW

### RECOMMENDED TOOLS
```bash
# MD→RST conversion
pandoc -f markdown -t rst input.md -o output.rst

# Sphinx build testing
cd docs/source && make html

# Link validation
sphinx-build -b linkcheck source build/linkcheck
```

### QUALITY ASSURANCE
- Verify all internal links work after conversion
- Test Sphinx build without errors
- Ensure cross-references between sections function
- Validate code examples execute correctly

## CRITICAL IMPLEMENTATION DETAILS

### REAL PRODUCTION EXAMPLES (VALIDATION SOURCES)
**File**: `openhcs/debug/example_export.py` - TUI-generated production script
**Shows**:
- List patterns: `[(stack_percentile_normalize, {...}), (tophat, {...})]`
- Dict patterns: `{'1': [(count_cells_single_channel, {...})], '2': [(skan_axon_skeletonize_and_analyze, {...})]}`
- GPU functions: CuPy processors, PyTorch processors, GPU position finding
- ZARR backend: 100GB+ dataset handling with memory overlay
- Real workflow: Preprocessing → Composite → Position finding → Assembly → Analysis

### FUNCTION REGISTRY IMPLEMENTATION
**Auto-Discovery**: 574+ functions from pyclesperanto, scikit-image, CuCIM
**Contracts**: `@cupy`, `@torch`, `@numpy` decorators for memory type safety
**Integration**: Import hook system decorates external libraries on import

### COMPILATION PHASES (ACTUAL IMPLEMENTATION)
```python
# From orchestrator.compile_pipelines()
PipelineCompiler.initialize_step_plans_for_context(context, pipeline_definition)
PipelineCompiler.declare_zarr_stores_for_context(context, pipeline_definition, self)
PipelineCompiler.plan_materialization_flags_for_context(context, pipeline_definition, self)
PipelineCompiler.validate_memory_contracts_for_context(context, pipeline_definition, self)
PipelineCompiler.assign_gpu_resources_for_context(context)
context.freeze()  # Immutable execution context
```

## SUCCESS CRITERIA

### PHASE 1 COMPLETE WHEN:
- [ ] `conf.py` project name updated to OpenHCS
- [ ] Import statements updated (`ezstitcher` → `openhcs`)
- [ ] Examples show `FunctionStep` patterns instead of `stitch_plate()`
- [ ] Sphinx builds without EZStitcher-related errors

### PHASE 2 COMPLETE WHEN:
- [ ] New architecture sections added for OpenHCS-specific systems
- [ ] Production examples integrated (cell counting, neurite tracing workflows)
- [ ] Memory type system documented with conversion examples
- [ ] TUI system documented with SSH workflow examples

### FINAL SUCCESS:
- [ ] Documentation reflects OpenHCS as bioimage analysis platform (not just stitching)
- [ ] Clear progression from simple patterns to complex GPU workflows
- [ ] Production-grade examples that users can actually run
- [ ] Preserved EZStitcher architectural foundations while showing OpenHCS evolution

## CONTEXT HANDOFF NOTES

**Repository Reality**: `/home/ts/code/projects/openhcs` is the `openhcs` branch of `ezstitcher` repo
**Architectural Continuity**: Function patterns, variable components, group_by all preserved from EZStitcher
**Documentation Approach**: Update existing docs rather than rewrite - core concepts are solid
**New Systems**: Memory types, function registry, TUI, special I/O, ZARR backend need new documentation
**Validation**: Use `openhcs/debug/example_export.py` as reference for real production workflows
