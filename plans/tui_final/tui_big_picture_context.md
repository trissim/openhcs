# OpenHCS TUI Big Picture Context

## 🌍 THE BIG PICTURE: What We're Building and Why

### What OpenHCS Is:
OpenHCS is a **declarative, GPU-native microscopy image processing pipeline system** for neuroscience research. It processes hundreds of microscope images with sophisticated algorithms like MIST stitching, N2V2 denoising, and custom analysis - all running on GPU with zero-copy memory management.

### The Core Innovation:
**Function Pattern Matching** - Scientists can visually compose image processing pipelines by selecting functions from a registry and configuring parameters through auto-generated UI forms. No coding required.

## 🏗️ OpenHCS Architecture (7 Layers):

1. **Storage Layer (VFS)** - FileManager with backend abstraction (Memory/Disk/Zarr)
2. **Memory Management** - GPU registry, memory type decorators (@torch, @cupy, @numpy)
3. **Processing Function Layer** - FUNC_REGISTRY with auto-discovery of decorated functions
4. **Pipeline Definition** - Pipeline as list of FunctionSteps with variable components
5. **Compilation Layer** - 4-phase compiler (path planning, materialization, memory validation, GPU assignment)
6. **Orchestration Layer** - PipelineOrchestrator with two-phase execution
7. **Microscope Integration** - Auto-detection and metadata handling

### Core Philosophy:
- **Declarative Primacy**: Everything declared upfront, no runtime inference
- **Immutability After Construction**: Once compiled, contexts are frozen
- **Fail Loudly**: No silent fallbacks or degradation
- **GPU-First**: Native GPU processing with explicit memory type declarations
- **Two-Phase Execution**: Compile-all, then execute-all

## 🎯 Why This TUI Architecture Makes Sense:

### 1. The Visual Programming Crown Jewel
The TUI's main purpose is to provide a **sophisticated visual programming interface** where:
- Functions are **auto-discovered** from the FUNC_REGISTRY (decorated with @torch, @cupy, @numpy)
- Parameter forms are **auto-generated** from function signatures using `inspect.signature()`
- Scientists build pipelines by **visual composition**, not coding

**Key Components:**
- **FunctionPatternEditor**: Auto-discovers functions, generates UI from signatures, supports 4 pattern types
- **DualStepFuncEditorPane**: Complete dual Step/Func editor with tabs for AbstractStep parameters and function patterns
- **ParameterEditor**: Dynamic parameter field generation with reset capabilities
- **GroupedDropdown**: Function selection with memory type grouping

### 2. The Two-Phase Execution Model
The TUI manages the OpenHCS orchestration pattern:
```
PHASE 1: Compile-All → Create frozen ProcessingContexts for all wells
PHASE 2: Execute-All → Run stateless pipelines against compiled contexts
```

**Status Progression:**
- `?` (gray): Created but not initialized
- `-` (yellow): Initialized but not compiled  
- `o` (green): Compiled and ready to run
- `!` (red): Running or error state

### 3. Direct Orchestrator Integration
We eliminated over-architected MVC layers because:
- **Simple is better** - Direct `orchestrator.initialize()`, `orchestrator.compile_pipelines()`, `orchestrator.execute_compiled_plate()`
- **Matches the domain** - Each plate = one PipelineOrchestrator instance
- **Fail loudly** - No complex error handling layers that hide problems

## 🚨 The Spaghetti Problem I Created:

In previous threads, I was in performance mode and created:
- **Duplicate implementations** of the same components
- **Mixed organizational patterns** (monolithic vs modular)
- **Import confusion** where different parts of the system got different versions of the same class
- **Architectural inconsistency** that violated SOLID principles

### The Crown Jewel Risk:
The **visual programming components** (FunctionPatternEditor, DualStepFuncEditorPane) are the most sophisticated and valuable parts of the system. They need:
- **Robust Container implementations** for proper prompt_toolkit integration
- **Consistent interfaces** for parameter editing and function selection
- **Clean modular architecture** that matches OpenHCS's sophisticated design

## 🎯 Why This Cleanup is Architecturally Critical:

1. **Consistency with OpenHCS Philosophy** - The core system is beautifully modular and declarative. The TUI should match that elegance.

2. **SOLID Principles** - Each component should have a single responsibility, be open for extension, and follow dependency inversion.

3. **Future-Proofing** - Scientists will want to add new function types, parameter editors, and UI components. The modular structure makes this easy.

4. **Maintainability** - When you publish in Nature Methods, other researchers will want to extend this. Clean architecture is essential.

## 🔍 FUNC_REGISTRY Auto-Discovery System:

The visual programming interface leverages OpenHCS's sophisticated function discovery:

```python
# Functions are auto-discovered by memory type decorators
@torch
def n2v2_denoise_torch(image_stack, **kwargs):
    """GPU-accelerated N2V2 denoising using PyTorch."""
    pass

@cupy  
def mist_compute_tile_positions(image_list, **kwargs):
    """MIST algorithm for tile position computation using CuPy."""
    pass
```

**Auto-Discovery Process:**
1. **Registry Initialization**: Scans processing directory for decorated functions
2. **Memory Type Grouping**: Functions organized by @torch, @cupy, @numpy decorators
3. **UI Generation**: Dropdowns populated from FUNC_REGISTRY by memory type
4. **Parameter Introspection**: `inspect.signature()` generates parameter forms
5. **Contract Validation**: FuncStepContractValidator ensures memory type compatibility

## 🎨 The Vision:
A scientist opens the TUI, adds their microscope data directories, visually composes a pipeline by selecting functions and configuring parameters through auto-generated forms, then runs the two-phase execution to process hundreds of images on GPU - all without writing a single line of code.

**The consolidation work ensures the visual programming interface has the robust, clean architecture it deserves to make this vision reality.**

## 📋 Current Consolidation Task:

**Problem**: Duplicate implementations of core components (InteractiveListItem, ParameterEditor, GroupedDropdown) exist in both:
- Monolithic files (`components.py`, `utils.py`) - Simple widget approach
- Modular directories (`components/`, `utils/`) - Proper Container inheritance with full prompt_toolkit integration

**Solution**: Keep the modular versions (they're more robust and what the crown jewels actually use), delete the monolithic duplicates, update all imports to use canonical versions.

**Why**: The visual programming components need the robust Container implementations for proper prompt_toolkit integration. The modular structure also matches OpenHCS's sophisticated, SOLID-compliant architecture.
