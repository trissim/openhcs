from setuptools import setup, find_packages

# GPU Dependencies Version Notes:
# - JAX 0.4.38: Last version compatible with CuDNN 9.5.x (JAX 0.5.0+ requires CuDNN 9.8+)
# - JAX CUDA plugins must exactly match JAX version for PJRT API compatibility
# - PyTorch 2.7.x: Compatible with CUDA 12.6 and CuDNN 9.5.x
# - TensorFlow 2.15-2.19: Stable DLPack support (2.12+ required for DLPack)
# - CuPy: Use cupy-cuda12x for broad CUDA 12.x compatibility (not cupy-cuda120)
# - CuCIM: PyPI cucim-cu12 package is incomplete (missing filters), use conda for full version
# - torbi: Using patched fork that fixes PyTorch 2.6+ CUDA compilation issues (py_limited_api)
#
# Installation: pip install -e ".[gpu]" for GPU support

setup(
    name="openhcs",
    packages=find_packages(include=["openhcs", "openhcs.*"]),
    install_requires=[
        "numpy>=1.26.4",  # Compatible with modern PyTorch/JAX (2023+)
        "scikit-image>=0.25.2",  # Compatible with modern GPU stack (2024+)
        "scikit-learn>=1.7.1",  # Compatible with modern numpy/scipy
        "scipy>=1.12.0",  # Compatible with modern numpy (2023+)
        "pandas>=2.3.1",  # Modern pandas with better performance
        "imageio>=2.37.0",  # Modern imageio with better format support
        "tifffile>=2025.6.11",  # Modern tifffile with zarr support
        "imagecodecs>=2025.3.30",  # Modern imagecodecs with better compression
        #"ashlar>=1.14.0",
        "opencv-python>=4.11.0.86",  # Modern OpenCV compatible with Python 3.12
        "Multi-Template-Matching>=2.0.1",
        "PyYAML>=6.0.2",
        "zarr>=2.18.7",
        "ome-zarr>=0.11.1",
        "pygments>=2.19.2",
        "textual>=3.7.1",
        "textual-serve>=1.1.2",
        "textual-terminal",
        "textual-universal-directorytree",
        "textual-window @ git+https://github.com/trissim/textual-window.git",
        "toolong @ git+https://github.com/trissim/toolong.git",
        "basicpy @ git+https://github.com/trissim/BaSiCPy.git",
        "plotext>=5.3.2",
        "psutil>=7.0.0",
        "GPUtil>=1.4.0",
        "textual-plotext",
        "watchdog>=6.0.0",
        "napari",
        "setuptools",
        "dill>=0.4.0",
        "PyQt6>=6.9.1",
        "PyQt6-QScintilla>=2.14.1",
        "pyqtgraph>=0.13.7"
    ],
    extras_require={
        "dev": [
            # Testing dependencies
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "coverage>=7.3.2",
            "genbadge[coverage]",
            "pytest-asyncio>=0.21.0",
        ],
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
        ]
    }
)
