# OpenHCS: Comprehensive System Overview

## 🔬 What is OpenHCS?

**OpenHCS (Open High-Content Screening)** is a GPU-accelerated microscopy image processing system designed for cell biology research, specifically axon regeneration studies. It solves critical format compatibility issues between different microscope systems while providing high-performance image stitching and analysis capabilities.

### 🎯 Core Mission
- **Scientific Problem**: Different microscopes produce incompatible file formats, creating barriers for researchers
- **Solution**: Universal format compatibility with GPU-accelerated processing
- **Target**: Nature Methods publication - solving real scientific problems at scale
- **User**: Cell biology PhD students and researchers processing large microscopy datasets

## 🏗️ Architectural Philosophy

### **Two-Phase Execution Model**
OpenHCS follows a strict **compile-all-then-run-all** pattern:
1. **PHASE 1: Compilation** → Create frozen, immutable ProcessingContexts for all wells
2. **PHASE 2: Execution** → Run stateless pipelines against compiled contexts in parallel

This enables:
- **Batch processing** of entire plates
- **Parallel execution** across wells
- **Reproducible results** through frozen contexts
- **Error detection** before expensive processing begins

### **Backend Independence**
- **VFS Abstraction**: All I/O through FileManager, no direct filesystem access
- **Multi-backend Support**: Local storage, cloud storage, network filesystems
- **GPU-Native Processing**: cupy-based with explicit CPU fallback (never silent)
- **Zero-Copy Interop**: Multi-framework compatibility without data copying

### **1:1 Disk-to-Storage Abstraction**
Every file operation respects the backend abstraction:
```python
# WRONG: Direct filesystem access
with open(file_path, 'rb') as f:
    data = f.read()

# CORRECT: VFS abstraction
data = filemanager.load(file_path, backend=config.backend)
```

## 🧬 Core Architecture Layers

### **1. Processing Layer (`openhcs/processing/`)**
**Purpose**: Function discovery and execution engine

**Key Components**:
- **FUNC_REGISTRY**: Global singleton containing 16+ functions across 5 backends
- **Function Discovery**: Auto-scans processing directory, registers by backend
- **Memory Type Tagging**: Functions tagged with `.backend` attribute (cpu, gpu, hybrid)
- **Signature Inspection**: Automatic parameter discovery for visual programming

**Example**:
```python
# Functions automatically discovered and registered
FUNC_REGISTRY = {
    'cpu': {'gaussian_blur': <function>, 'threshold': <function>},
    'gpu': {'mist_stitch': <function>, 'cuda_filter': <function>},
    'hybrid': {'adaptive_process': <function>}
}
```

### **2. Core Layer (`openhcs/core/`)**
**Purpose**: Pipeline orchestration and execution engine

**Key Components**:
- **PipelineOrchestrator**: Manages two-phase execution model
- **ProcessingContext**: Frozen, immutable execution context per well
- **PipelineCompiler**: Creates step plans and validates dependencies
- **AbstractStep/FunctionStep**: Pipeline building blocks
- **FileManager**: VFS abstraction for all I/O operations

**Workflow**:
```python
# 1. Create orchestrator for plate
orchestrator = PipelineOrchestrator(plate_path, global_config)

# 2. Initialize (discover wells, setup workspace)
orchestrator.initialize()

# 3. Compile pipeline for all wells
contexts = orchestrator.compile_pipelines(pipeline_definition)

# 4. Execute against frozen contexts
orchestrator.execute_compiled_plate()
```

### **3. I/O Layer (`openhcs/io/`)**
**Purpose**: Backend-agnostic file operations

**Key Components**:
- **FileManager**: Central VFS abstraction
- **Backend Handlers**: Local, cloud, network storage implementations
- **Format Converters**: Microscope format compatibility
- **Metadata Extractors**: Image metadata and grid dimension discovery

**Critical Rule**: ALL file operations must use FileManager abstraction

### **4. TUI Layer (`openhcs/tui/`)**
**Purpose**: Visual programming interface for pipeline building

**Architecture**: Direct integration with minimal layers
- **No MVC complexity**: Direct handler → orchestrator method calls
- **Visual Programming**: Function discovery through FUNC_REGISTRY
- **Real-time Editing**: Live parameter adjustment with signature inspection
- **Error Handling**: Comprehensive user feedback through dialog system

## 🎨 TUI Vision: Visual Programming for Scientists

### **Core Concept**
Transform pipeline building from code writing to visual programming:
- **Function Discovery**: Auto-discover available functions by backend
- **Parameter Generation**: Automatically create UI from function signatures
- **Live Preview**: Real-time parameter adjustment
- **Drag-and-Drop**: Visual pipeline construction
- **Scientific Workflow**: Designed for researchers, not programmers

### **Crown Jewel Components**

#### **DualStepFuncEditorPane** (600+ lines)
- **Dual-tab interface**: AbstractStep parameters + Function patterns
- **Live editing**: Real-time parameter adjustment
- **Signature inspection**: Automatic UI generation from function signatures
- **Save/Cancel workflow**: Non-destructive editing with rollback

#### **FunctionPatternEditor** (800+ lines)
- **Function discovery**: Integrates with FUNC_REGISTRY
- **Memory type grouping**: CPU/GPU/Hybrid function organization
- **Parameter editors**: Dynamic field generation based on function signatures
- **Pattern validation**: Real-time validation of function parameters

#### **Visual Programming Components**
- **ParameterEditor**: Dynamic parameter field generation with reset capabilities
- **GroupedDropdown**: Function selection with memory type grouping
- **InteractiveListItem**: Clickable pipeline step management
- **FramedButton**: Custom styled buttons with mouse handlers

### **TUI Workflow**
```
1. Add Plate → PlateManagerPane → PipelineOrchestrator created
2. Edit Step → DualStepFuncEditorPane → Visual function selection
3. Configure → ParameterEditor → Live parameter adjustment
4. Compile → PipelineCompiler → Frozen contexts created
5. Run → PipelineExecutor → Parallel execution across wells
```

## 🔧 Current TUI Implementation Status

### **✅ COMPLETED (Phases 1-5)**
- **Radical Simplification**: Eliminated 4,085+ lines of over-architecture
- **Crown Jewel Preservation**: Kept 1,400+ lines of sophisticated visual programming
- **Direct Handler System**: Replaced complex command system with direct orchestrator calls
- **Visual Programming Integration**: DualStepFuncEditorPane seamlessly integrated
- **Architectural Consistency**: Complete system coherence verified

### **🚧 IN PROGRESS (Phase 6)**
- **Backend Compliance**: Replace direct file I/O with FileManager abstraction
- **Button Implementation**: Complete remaining menu handlers (Global Settings, Help, Exit)
- **VFS Integration**: Ensure all I/O operations use proper backend abstraction

### **🎯 ARCHITECTURAL ACHIEVEMENT**
The TUI now provides:
- **Functional visual programming interface** for scientific pipeline building
- **Direct orchestrator integration** without architectural violations
- **Comprehensive error handling** with user-friendly dialogs
- **Real-time function discovery** through FUNC_REGISTRY integration
- **Live parameter editing** with signature-based UI generation

## 🧪 Scientific Context

### **Research Domain**: Cell Biology - Axon Regeneration
- **Data Type**: High-resolution microscopy images
- **Scale**: Entire plates with multiple wells
- **Processing**: GPU-accelerated stitching and analysis
- **Output**: Quantitative measurements for regeneration studies

### **Technical Challenges Solved**
1. **Format Compatibility**: Universal microscope format support
2. **Performance**: GPU acceleration with cupy integration
3. **Scalability**: Parallel processing across wells
4. **Usability**: Visual programming for non-programmers
5. **Reproducibility**: Frozen contexts ensure consistent results

### **Image Processing Flow**
```
Raw Images → load() → stack() to 3D (ZYX format) → process() → unstack() → save()
```
All operations respect the VFS abstraction and backend independence.

## 🎯 Design Principles

### **1. Architectural Immunity**
- **No backwards compatibility bloat**: Clean, purposeful code
- **Root cause fixes**: Address foundational issues, not symptoms
- **SOLID principles**: Clean architecture with modular design
- **Low entropy**: Declarative code with pure functions over OOP complexity

### **2. Explicit Failure**
- **No silent fallbacks**: GPU processing failures are explicit and visible
- **Loud failures**: Direct dictionary access without fallbacks - let it fail loudly
- **Comprehensive error handling**: User-friendly dialogs for all error conditions

### **3. Compile-Time Resolution**
- **Everything through compiler**: All operations go through compilation phase
- **Frozen contexts**: Runtime context is immutable post-compilation
- **Predeclared paths**: All information passing uses VFS abstraction
- **Parameter resolution**: Compile-time parameter resolution over runtime injection

### **4. Visual Programming First**
- **Function discovery**: Auto-discovery through FUNC_REGISTRY
- **Signature-based UI**: Automatic parameter field generation
- **Live editing**: Real-time parameter adjustment
- **Scientific workflow**: Designed for researchers, not developers

## 🚀 Future Vision

### **Immediate Goals**
- **Complete TUI implementation**: Finish Phase 6 backend compliance
- **Nature Methods publication**: Demonstrate scientific impact
- **Community adoption**: Enable other research groups to use OpenHCS

### **Long-term Vision**
- **Universal microscopy platform**: Support for all major microscope formats
- **Cloud-native processing**: Seamless cloud backend integration
- **Collaborative research**: Multi-user pipeline sharing and collaboration
- **AI integration**: Machine learning pipeline components for automated analysis

## 📚 Key Takeaways for New Contributors

### **Understanding OpenHCS**
1. **It's a scientific tool first**: Designed for cell biology researchers
2. **Architecture matters**: Clean, principled design enables scientific reproducibility
3. **Visual programming is key**: Scientists shouldn't need to write code
4. **Backend independence**: VFS abstraction enables flexible deployment
5. **Two-phase execution**: Compile-all-then-run-all enables parallel processing

### **Working with OpenHCS**
1. **Respect the VFS**: All I/O must use FileManager abstraction
2. **Use the compiler**: Everything goes through the compilation phase
3. **Leverage FUNC_REGISTRY**: Function discovery enables visual programming
4. **Follow the workflow**: Add plate → edit step → compile → run
5. **Maintain architectural immunity**: Clean, purposeful code without bloat

### **TUI Development**
1. **Direct integration**: No complex MVC layers, direct orchestrator calls
2. **Visual programming first**: Leverage existing crown jewel components
3. **Comprehensive error handling**: User-friendly dialogs for all conditions
4. **State-driven architecture**: Observer pattern for clean component communication
5. **Backend compliance**: All I/O through FileManager, no direct filesystem access

## 🧠 Deep Learning & Advanced Processing Algorithms

### **🔬 Deep Learning Algorithms**

#### **N2V2 (Noise2Void 2) - PyTorch Implementation**
- **Purpose**: Self-supervised denoising without clean training data
- **Backend**: PyTorch (`n2v2_denoise_torch`)
- **Innovation**: Eliminates need for paired noisy/clean training datasets
- **Application**: Microscopy image denoising for low-light conditions

#### **Self-Supervised 3D Deconvolution**
- **Purpose**: Point spread function deconvolution without known PSF
- **Backend**: PyTorch (`self_supervised_3d_deconvolution`)
- **Architecture**: 3D CNN with 48 initial features optimized for volumetric data
- **Innovation**: Self-supervised learning eliminates need for PSF calibration
- **Loss Functions**: Reconstruction + invariance + boundary constraints

#### **Self-Supervised 2D Deconvolution**
- **Purpose**: Optimized deconvolution for 2D imaging data
- **Backend**: PyTorch (`self_supervised_2d_deconvolution`)
- **Architecture**: 96 initial features, 128x128 patches, batch size 16
- **Innovation**: 2D-optimized configuration based on latest research

#### **Self-Supervised 3D Segmentation**
- **Purpose**: Volumetric segmentation without manual annotations
- **Backend**: PyTorch (`self_supervised_segmentation_3d`)
- **Architecture**: Encoder-decoder with contrastive learning
- **Innovation**: Combines reconstruction and contrastive losses for unsupervised segmentation

#### **BaSiC (Background and Shading Correction)**
- **Purpose**: Illumination correction and background subtraction
- **Backends**: NumPy (`basic_flatfield_correction_numpy`), CuPy (`basic_flatfield_correction_cupy`)
- **Innovation**: Batch processing capabilities for high-throughput correction

### **🎯 Focus Analysis Algorithms**

#### **Multi-Metric Focus Analyzer**
- **Purpose**: Automated focus plane selection from Z-stacks
- **Backend**: NumPy (`focus_analyzer.py`)
- **Metrics**:
  - **Normalized Variance**: Statistical focus measure
  - **Laplacian Energy**: Edge-based focus detection
  - **Tenengrad Variance**: Gradient-based focus measure
  - **Adaptive FFT Focus**: Frequency domain analysis
  - **Combined Metric**: Weighted combination of all methods
- **Innovation**: Multi-metric approach provides robust focus detection across image types

### **🔄 Novel MIST Implementation - State-of-the-Art**

#### **Why OpenHCS MIST is Novel and SOTA:**

**1. GPU-Native Architecture**
- **Pure GPU Pipeline**: No CPU fallbacks, complete CuPy implementation
- **Parallel Processing**: Multiple image pairs processed simultaneously
- **Memory Efficiency**: Overlap-region optimization reduces bandwidth requirements

**2. NIST Robustness Integration**
- **Multi-Peak Testing**: GPU-accelerated n=2 peak detection by default
- **Directional Constraints**: Horizontal/vertical alignment validation
- **Adaptive Quality Metrics**: Dynamic threshold adjustment based on correlation statistics
- **FFT Interpretation Testing**: Multiple translation interpretations with periodicity handling

**3. Advanced Phase Correlation**
- **Subpixel Accuracy**: GPU-native subpixel refinement
- **NIST Normalization**: fc/abs(fc) normalization for improved robustness
- **Quality-Weighted MST**: Correlation quality influences graph construction

**4. Modern Graph Algorithms**
- **Borůvka MST with Union-Find**: Parallel-friendly minimum spanning tree
- **Iterative Refinement**: Multiple passes with configurable damping
- **Global Optimization**: MST-based position reconstruction

**5. Performance Innovations**
- **Overlap-Region Focus**: More efficient than full-tile correlation
- **Configurable Parameters**: No magic numbers, all parameters exposed
- **Batch Processing**: Entire plates processed in parallel

**Key Advantages over Original NIST MIST:**
- **10-100x faster** through GPU acceleration
- **Better accuracy** through iterative refinement
- **More robust** through multi-peak testing and adaptive thresholds
- **Scalable** to larger datasets through memory optimization

### **⚙️ Basic Image Processing Functions by Backend**

#### **NumPy Backend Functions**
- **Morphological Operations**: White top-hat, disk structuring elements
- **Geometric Transforms**: Resize, anti-aliasing, preserve range
- **Statistical Analysis**: Background estimation, foreground extraction
- **Focus Metrics**: Variance, Laplacian, Tenengrad, FFT analysis

#### **CuPy Backend Functions**
- **MIST Position Generation**: `mist_compute_tile_positions`
- **Ashlar Alignment**: `gpu_ashlar_align_cupy`
- **Stack Assembly**: `assemble_stack_cupy`
- **BaSiC Correction**: `basic_flatfield_correction_cupy`
- **Phase Correlation**: GPU-native with NIST robustness
- **Quality Metrics**: GPU-accelerated correlation quality assessment

#### **PyTorch Backend Functions**
- **N2V2 Denoising**: `n2v2_denoise_torch`
- **3D Deconvolution**: `self_supervised_3d_deconvolution`
- **2D Deconvolution**: `self_supervised_2d_deconvolution`
- **3D Segmentation**: `self_supervised_segmentation_3d`
- **Neural Network Architectures**: 3D CNNs, encoder-decoder models

#### **JAX Backend Functions**
- **Weight Mask Generation**: `create_weight_mask_jax`
- **Margin-based Blending**: Linear ramp generation for seamless stitching
- **Functional Programming**: Pure functions with JAX transformations

#### **TensorFlow Backend Functions**
- **Deep Learning Integration**: TensorFlow model compatibility
- **GPU Acceleration**: CUDA-optimized operations
- **Distributed Processing**: Multi-GPU support for large datasets

### **🔬 Specialized Analysis Functions**

#### **DXF Mask Pipeline**
- **Purpose**: CAD-based region of interest analysis
- **Backend**: Analysis (`dxf_mask_pipeline`)
- **Innovation**: Integration of CAD drawings with microscopy analysis

#### **Multi-Backend Function Registry**
- **Auto-Discovery**: Automatic function registration by backend
- **Memory Type Tagging**: CPU/GPU/Hybrid classification
- **Signature Inspection**: Automatic parameter discovery for visual programming
- **16+ Functions**: Across 5 backends (NumPy, CuPy, PyTorch, TensorFlow, JAX)

---

**OpenHCS represents a new paradigm in scientific software**: combining rigorous architecture with intuitive visual programming to enable breakthrough research in cell biology. The comprehensive algorithm suite provides researchers with state-of-the-art image processing capabilities, from basic operations to cutting-edge deep learning, all accessible through an intuitive visual interface without requiring programming expertise.**
