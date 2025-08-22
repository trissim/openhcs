from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    """Get version from openhcs/__init__.py"""
    init_file = os.path.join(os.path.dirname(__file__), "openhcs", "__init__.py")
    with open(init_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"\'')
    raise RuntimeError("Unable to find version string.")

# GPU Dependencies Version Notes:
# - JAX 0.4.38: Last version compatible with CuDNN 9.5.x (JAX 0.5.0+ requires CuDNN 9.8+)
# - JAX CUDA plugins must exactly match JAX version for PJRT API compatibility
# - PyTorch 2.7.x: Compatible with CUDA 12.6 and CuDNN 9.5.x
# - TensorFlow 2.15-2.19: Stable DLPack support (2.12+ required for DLPack)
# - CuPy: Use cupy-cuda12x for broad CUDA 12.x compatibility (not cupy-cuda120)
# - CuCIM: PyPI cucim-cu12 package is incomplete (missing filters), use conda for full version
# - torbi: Using patched fork that fixes PyTorch 2.6+ CUDA compilation issues (py_limited_api)
#
# Installation Examples:
# - Base package only: pip install openhcs
# - With TUI: pip install "openhcs[tui]"
# - With GUI: pip install "openhcs[gui]"
# - With GPU support: pip install "openhcs[gpu]"
# - Full installation: pip install "openhcs[all]"

# Define extras_require with programmatic "all" extra
extras_require = {
    # Development dependencies (CPU-only testing)
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "coverage>=7.3.2",
        "genbadge[coverage]",
        "pytest-asyncio>=0.21.0",
    ],

    # GUI testing dependencies
    "dev-gui": [
        "pytest-qt>=4.2.0",  # PyQt6 testing framework
    ],

    # Terminal User Interface (TUI) dependencies
    "tui": [
        "textual>=3.7.1",
        "textual-serve>=1.1.2",
        "textual-terminal",
        "textual-universal-directorytree",
        "textual-window @ git+https://github.com/trissim/textual-window.git",
        "toolong @ git+https://github.com/trissim/toolong.git",
        "plotext>=5.3.2",
        "textual-plotext",
        "pygments>=2.19.2",  # For syntax highlighting in TUI error dialogs
        "psutil>=7.0.0",  # For system monitoring (required for TUI)
        "GPUtil>=1.4.0",  # For GPU monitoring (required for TUI)
    ],

    # PyQt6 Graphical User Interface (GUI) dependencies
    "gui": [
        "PyQt6>=6.9.1",
        "PyQt6-QScintilla>=2.14.1",
        "pyqtgraph>=0.13.7",
        "psutil>=7.0.0",  # For system monitoring (required for GUI)
        "GPUtil>=1.4.0",  # For GPU monitoring (required for GUI)
    ],

    # Optional visualization tools
    "viz": [
        "napari",  # Optional real-time visualization during pipeline execution
    ],

    # GPU acceleration dependencies
    "gpu": [
        # PyTorch - let pip resolve compatible versions automatically
        "torch>=2.6.0,<2.8.0",
        "torchvision>=0.21.0,<0.23.0",

        # JAX - pinned to current working versions
        "jax>=0.5.3,<0.6.0",
        "jaxlib>=0.5.3,<0.6.0",

        # JAX CUDA plugins - pinned to current working versions
        "jax-cuda12-pjrt>=0.5.3,<0.6.0",
        "jax-cuda12-plugin>=0.5.3,<0.6.0",

        # CuPy - pinned to current working version
        "cupy-cuda12x>=13.3.0,<14.0.0",

        # CuCIM - pinned to current working version
        "cucim-cu12>=25.6.0,<26.0.0",

        # TensorFlow - pinned to current working version
        "tensorflow>=2.19.0,<2.20.0",

        # TensorFlow Probability - pinned to current working version
        "tensorflow-probability[tf]>=0.25.0,<0.26.0",

        # pyclesperanto - pinned to current working version
        "pyclesperanto>=0.17.1",

        # torbi - GPU-accelerated Viterbi decoding (patched fork for PyTorch 2.6+ compatibility)
        "torbi @ git+https://github.com/trissim/torbi.git",

        # torch_nlm - PyTorch-based non-local means denoising with GPU support
        # "nlm-torch>=0.1.0"  # Disabled: requires numpy==1.23.5 which conflicts with modern stack
    ],
}

# Programmatically create "all" extra by combining all non-dev extras
# This avoids duplication and is automatically maintained
all_deps = []
for extra_name, deps in extras_require.items():
    if extra_name not in ["dev", "dev-gui"]:  # Exclude dev dependencies from "all"
        all_deps.extend(deps)

# Remove duplicates while preserving order (psutil/GPUtil appear in both tui and gui)
seen = set()
unique_deps = []
for dep in all_deps:
    if dep not in seen:
        seen.add(dep)
        unique_deps.append(dep)

extras_require["all"] = unique_deps

setup(
    name="openhcs",
    version=get_version(),
    description="High-Content Screening image processing engine with native GPU support",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Tristan Simas",
    author_email="tristan.simas@mail.mcgill.ca",
    url="https://github.com/trissim/openhcs",
    project_urls={
        "Bug Reports": "https://github.com/trissim/openhcs/issues",
        "Source": "https://github.com/trissim/openhcs",
        "Documentation": "https://github.com/trissim/openhcs/blob/main/README.md",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    keywords="microscopy, image-processing, high-content-screening, gpu, computer-vision",
    packages=find_packages(include=["openhcs", "openhcs.*"]),
    install_requires=[
        # Core image processing and scientific computing
        "numpy>=1.26.4",  # Compatible with modern PyTorch/JAX (2023+)
        "scikit-image>=0.25.2",  # Compatible with modern GPU stack (2024+)
        "scikit-learn>=1.7.1",  # Compatible with modern numpy/scipy
        "scipy>=1.12.0",  # Compatible with modern numpy (2023+)
        "pandas>=2.3.1",  # Modern pandas with better performance

        # Image I/O and formats
        "imageio>=2.37.0",  # Modern imageio with better format support
        "tifffile>=2025.6.11",  # Modern tifffile with zarr support
        "imagecodecs>=2025.3.30",  # Modern imagecodecs with better compression
        "opencv-python>=4.11.0.86",  # Modern OpenCV compatible with Python 3.12
        "Multi-Template-Matching>=2.0.1",

        # Storage and serialization
        "PyYAML>=6.0.2",
        "zarr>=2.18.7,<3.0",
        "ome-zarr>=0.11.1",
        "dill>=0.4.0",

        # Core utilities
        "setuptools",
        "watchdog>=6.0.0",  # For file system monitoring

        # Custom packages
        "basicpy @ git+https://github.com/trissim/BaSiCPy.git",
    ],
    extras_require=extras_require,

    # Console script entry points
    entry_points={
        "console_scripts": [
            # Main interfaces
            "openhcs-tui=openhcs.textual_tui.__main__:main",
            "openhcs-gui=openhcs.pyqt_gui.__main__:main",

            # Convenience aliases
            "openhcs=openhcs.textual_tui.__main__:main",  # Default to TUI

            # Utility scripts
            "openhcs-recache=openhcs.utils.recache_function_registry:main",
        ],
    },
)
