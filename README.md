# OpenHCS: Open High-Content Screening

<div align="center">
  <img src="https://raw.githubusercontent.com/trissim/openhcs/main/docs/source/_static/ezstitcher_logo.png" alt="OpenHCS Logo" width="400">
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green.svg)](https://github.com/trissim/openhcs)
[![Documentation Status](https://readthedocs.org/projects/openhcs/badge/?version=latest)](https://openhcs.readthedocs.io/en/latest/?badge=latest)
[![Coverage](https://trissim.github.io/openhcs/.github/badges/coverage.svg)](https://trissim.github.io/openhcs/coverage/)

## High-Performance Bioimage Analysis Platform

OpenHCS is a production-grade bioimage analysis platform designed for high-content screening datasets that break traditional tools. Built from the ground up for **100GB+ datasets**, **GPU acceleration**, and **remote computing environments**.

**Evolution from EZStitcher**: OpenHCS represents the next generation of the EZStitcher microscopy stitching library, evolved into a comprehensive GPU-native bioimage analysis platform with a **5-phase compilation system** that validates entire processing chains before execution, preventing the runtime failures that plagued traditional image analysis tools.

## 🚀 Key Features

### **5-Phase Pipeline Compiler**
- **Compile-Time Validation**: Pre-execution validation of entire processing chains
- **Immutable Execution Contexts**: Frozen contexts prevent runtime configuration drift
- **Memory Type Safety**: Compile-time validation of GPU memory operations
- **Resource Planning**: GPU assignment and memory contract validation before execution

### **GPU-Native Processing**
- **574+ GPU Functions**: Auto-discovered from pyclesperanto, CuPy, PyTorch, JAX, TensorFlow
- **Multi-Library Support**: Seamless conversion between NumPy↔CuPy↔PyTorch↔JAX↔TensorFlow
- **Zero-Copy Operations**: GPU memory management with compile-time validation
- **Advanced Stitching**: GPU-accelerated MIST and Ashlar algorithms

### **Production-Scale Data Handling**
- **100GB+ Dataset Support**: ZARR compression with adaptive chunking
- **Virtual File System**: Memory, Disk, and ZARR backends with automatic switching
- **Memory Overlay System**: Efficient intermediate data management
- **Parallel Processing**: Multi-worker execution with GPU scheduling

### **Microscope-Agnostic Design**
- **ImageXpress**: Native support for Molecular Devices systems
- **Opera Phenix**: PerkinElmer high-content screening systems
- **OpenHCS Format**: Optimized internal format for maximum performance
- **Extensible Architecture**: Easy addition of new microscope types

### **Professional Interface**
- **Interactive TUI**: Terminal-based interface that works over SSH
- **PyQt6 GUI**: Native desktop application with full feature parity
- **Real-time Monitoring**: Live pipeline execution with resource tracking
- **Production Logging**: Professional log streaming and analysis

## ⚡ Quick Start

### Installation

OpenHCS offers flexible installation options to match your specific needs:

```bash
# Basic installation (core image processing only)
pip install openhcs

# With Terminal User Interface (recommended for SSH/remote work)
pip install "openhcs[tui]"

# With Desktop GUI (recommended for local development)
pip install "openhcs[gui]"

# With GPU acceleration (high-performance processing)
pip install "openhcs[gpu]"

# Common combinations
pip install "openhcs[tui,gpu]"     # TUI + GPU acceleration
pip install "openhcs[gui,gpu]"     # GUI + GPU acceleration
pip install "openhcs[tui,gui]"     # Both interfaces

# Everything (all features)
pip install "openhcs[all]"

# Optional: Real-time visualization during processing
pip install "openhcs[viz]"         # Adds napari support
```

#### Development Installation

```bash
# Install with pyenv (recommended)
pyenv install 3.11
pyenv global 3.11

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Note: GPU dependencies require CUDA 12.x (see GPU Compatibility section below)

# Clone and install in development mode
git clone https://github.com/trissim/openhcs.git
cd openhcs
pip install -e ".[all]"  # Install with all features for development
```

### Launch Interactive Interface

OpenHCS provides convenient command-line entry points:

```bash
# Terminal interface (works over SSH, requires 'tui' extra)
openhcs-tui
# or: openhcs  (default interface)

# Desktop GUI application (requires 'gui' extra)
openhcs-gui

# Web-based TUI (accessible via browser)
openhcs-tui --web

# Utility: Recache function registry (useful after updates)
openhcs-recache
```

**Legacy commands** (still supported):
```bash
python -m openhcs.textual_tui
python -m openhcs.pyqt_gui
```

### 🎯 Choosing the Right Installation

| Use Case | Installation Command | What You Get |
|----------|---------------------|--------------|
| **Remote server work** | `pip install "openhcs[tui]"` | Terminal interface + monitoring |
| **Local development** | `pip install "openhcs[gui]"` | Desktop GUI + monitoring |
| **High-performance processing** | `pip install "openhcs[gpu]"` | GPU acceleration (574+ functions) |
| **Complete remote setup** | `pip install "openhcs[tui,gpu]"` | Terminal interface + GPU power |
| **Complete local setup** | `pip install "openhcs[gui,gpu]"` | Desktop GUI + GPU power |
| **Research/visualization** | `pip install "openhcs[all]"` | Everything including napari |
| **Minimal/testing** | `pip install openhcs` | Core processing only |

**💡 Pro Tips:**
- Start with `openhcs[tui]` for SSH/remote work
- Use `openhcs[gui,gpu]` for local high-performance analysis
- Add `viz` extra for real-time napari visualization during processing
- The `all` extra includes everything but is ~2GB+ with GPU libraries

### 🔧 GPU Compatibility

OpenHCS GPU dependencies require **CUDA 12.x**. 

**CPU-Only Mode:**
```bash
# Skip GPU dependencies entirely
export OPENHCS_CPU_ONLY=true
pip install "openhcs[tui]"  # or [gui] without [gpu]
```

## 📊 Basic Usage

### Simplified Interface (Recommended for Beginners)

```python
from openhcs.core.orchestrator.pipeline_orchestrator import PipelineOrchestrator
from openhcs.core.config import GlobalPipelineConfig

# Initialize OpenHCS
orchestrator = PipelineOrchestrator(
    input_dir="path/to/microscopy/data",
    global_config=GlobalPipelineConfig(num_workers=4)
)

# Initialize the orchestrator
orchestrator.initialize()

# Run complete analysis pipeline (requires pipeline definition)
# Use the TUI to create pipelines interactively
```

### Advanced Pipeline Creation

```python
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.processors.cupy_processor import (
    stack_percentile_normalize, tophat, create_composite
)
from openhcs.processing.backends.analysis.cell_counting_cupy import count_cells_single_channel
from openhcs.processing.backends.pos_gen.mist.mist_main import mist_compute_tile_positions
from openhcs.processing.backends.pos_gen.ashlar_main_gpu import ashlar_compute_tile_positions_gpu
from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy
from openhcs.constants.constants import VariableComponents

# Create GPU-accelerated processing pipeline
steps = [
    # Image preprocessing
    FunctionStep(
        func=[stack_percentile_normalize],
        name="normalize",
        variable_components=[VariableComponents.SITE]
    ),
    FunctionStep(
        func=[(tophat, {'selem_radius': 25})],
        name="enhance",
        variable_components=[VariableComponents.SITE]
    ),

    # Position generation for stitching
    FunctionStep(
        func=[ashlar_compute_tile_positions_gpu],
        name="positions",
        variable_components=[VariableComponents.SITE]
    ),

    # Image assembly using calculated positions
    FunctionStep(
        func=[assemble_stack_cupy],
        name="assemble",
        variable_components=[VariableComponents.SITE]
    ),

    # Cell analysis
    FunctionStep(
        func=[count_cells_single_channel],
        name="count_cells",
        variable_components=[VariableComponents.SITE]
    )
]

# See openhcs/debug/example_export.py for complete working example
```

## 🔧 Supported Processing Functions

### **Image Processing (120+ Functions)**
- **Preprocessing**: Normalization, denoising, enhancement
- **Filtering**: Gaussian, median, tophat, edge detection
- **Morphology**: Opening, closing, erosion, dilation
- **Projections**: Max, mean, standard deviation

### **Cell Analysis (50+ Functions)**
- **Detection**: Blob detection (LOG, DOG, DOH), watershed, threshold
- **Segmentation**: GPU-accelerated watershed, region growing
- **Measurement**: Intensity, morphology, texture features
- **Tracking**: Cell lineage and migration analysis

### **Stitching Algorithms**
- **MIST**: GPU-accelerated phase correlation with robust optimization
- **Ashlar**: GPU-accelerated position generation with edge-based alignment
- **Assembly**: Subpixel positioning and blending for final image reconstruction
- **Custom**: Extensible framework for new stitching methods

### **Neurite Analysis**
- **Skeletonization**: GPU-accelerated morphological thinning
- **Tracing**: SKAN-based neurite analysis with HMM models
- **Quantification**: Length, branching, connectivity metrics

## 📚 Documentation

### **Complete Documentation**
- **📖 [Read the Docs](https://openhcs.readthedocs.io/)** - Complete API documentation, tutorials, and guides
- **📊 [Coverage Reports](https://trissim.github.io/openhcs/coverage/)** - Live test coverage analysis
- **🔧 [API Reference](https://openhcs.readthedocs.io/en/latest/api/)** - Detailed function and class documentation
- **🎯 [User Guide](https://openhcs.readthedocs.io/en/latest/user_guide/)** - Step-by-step tutorials and examples

### **Quick Links**
- **Architecture**: [Pipeline System](https://openhcs.readthedocs.io/en/latest/architecture/pipeline-compilation-system.html) | [GPU Processing](https://openhcs.readthedocs.io/en/latest/architecture/gpu-resource-management.html) | [VFS](https://openhcs.readthedocs.io/en/latest/architecture/vfs-system.html)
- **Getting Started**: [Installation](https://openhcs.readthedocs.io/en/latest/getting_started/installation.html) | [First Pipeline](https://openhcs.readthedocs.io/en/latest/getting_started/first_pipeline.html)
- **Advanced**: [GPU Optimization](https://openhcs.readthedocs.io/en/latest/guides/gpu_optimization.html) | [Large Datasets](https://openhcs.readthedocs.io/en/latest/guides/large_datasets.html)

## 🏗️ Architecture Overview

### **5-Phase Pipeline Compilation System**
OpenHCS uses a sophisticated compiler that transforms pipeline definitions into optimized execution plans:

1. **Step Plan Initialization**: Creates execution plans and resolves input/output paths within VFS
2. **ZARR Store Declaration**: Declares necessary ZARR stores for large dataset compression
3. **Materialization Planning**: Determines which steps require persistent storage output
4. **Memory Contract Validation**: Validates GPU memory requirements and function compatibility
5. **GPU Resource Assignment**: Assigns specific GPU devices ensuring balanced utilization

```python
# Compilation produces immutable execution contexts
for well_id in wells_to_process:
    context = self.create_context(well_id)

    # 5-Phase Compilation
    PipelineCompiler.initialize_step_plans_for_context(context, pipeline_definition)
    PipelineCompiler.declare_zarr_stores_for_context(context, pipeline_definition, self)
    PipelineCompiler.plan_materialization_flags_for_context(context, pipeline_definition, self)
    PipelineCompiler.validate_memory_contracts_for_context(context, pipeline_definition, self)
    PipelineCompiler.assign_gpu_resources_for_context(context)

    context.freeze()  # Immutable execution context
    compiled_contexts[well_id] = context
```

### **Virtual File System (VFS)**
- **Memory Backend**: Fast intermediate storage for active processing
- **ZARR Backend**: Compressed storage for large datasets (100GB+)
- **Disk Backend**: Traditional file system operations
- **Automatic Switching**: Based on data size and configuration

## 🎯 Supported Microscope Formats

- **ImageXpress**: Molecular Devices high-content screening systems
- **Opera Phenix**: PerkinElmer automated microscopy platforms
- **OpenHCS**: Optimized internal format for maximum performance
- **Extensible**: Framework for adding new microscope types

## 📚 Documentation

Our comprehensive documentation covers all aspects of OpenHCS:

- **[Getting Started Guide](https://openhcs.readthedocs.io/en/latest/getting_started/)**: Quick setup and first analysis
- **[User Guide](https://openhcs.readthedocs.io/en/latest/user_guide/)**: Complete interface tutorials
- **[API Reference](https://openhcs.readthedocs.io/en/latest/api/)**: Detailed function documentation
- **[Architecture Guide](https://openhcs.readthedocs.io/en/latest/architecture/)**: System design and concepts
- **[Production Examples](https://openhcs.readthedocs.io/en/latest/examples/)**: Real-world analysis workflows

## 🔬 Production Example

**Complete neurite analysis pipeline** demonstrating all major OpenHCS features:

```bash
# View the complete production example
git clone https://github.com/trissim/openhcs.git
cat openhcs/debug/example_export.py

# Note: This is a real production script generated by the TUI
```

This example includes:
- ✅ Complete preprocessing → stitching → analysis workflow
- ✅ GPU acceleration with CuPy, PyTorch, and pyclesperanto
- ✅ 100GB+ dataset handling with ZARR compression
- ✅ Parallel processing with resource monitoring
- ✅ Professional configuration and logging

## 🤝 Contributing

We welcome contributions! OpenHCS is actively developed for production neuroscience research.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/trissim/openhcs.git
cd openhcs

# Install in development mode with all features
pip install -e ".[all,dev]"

# Run tests
pytest tests/
```

### Areas for Contribution
- **New Microscope Formats**: Add support for additional imaging systems
- **Processing Functions**: Contribute specialized analysis algorithms
- **GPU Backends**: Extend support for new GPU computing libraries
- **Documentation**: Improve guides and examples

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

OpenHCS builds upon the foundation of EZStitcher and incorporates algorithms and concepts from:
- **Ashlar**: Advanced image stitching algorithms
- **MIST**: Robust phase correlation methods
- **pyclesperanto**: GPU-accelerated image processing
- **scikit-image**: Comprehensive image analysis tools

---

**OpenHCS**: Transforming high-content screening analysis with GPU acceleration and production-grade tooling.
