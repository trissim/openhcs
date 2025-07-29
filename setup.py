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
        "numpy>=1.20.0",
        "scikit-image>=0.18.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.6.0",
        "pandas>=1.2.0",
        "imageio>=2.9.0",
        "tifffile>=2021.1.1",
        "imagecodecs>=2021.1.1",
        #"ashlar>=1.14.0",
        "opencv-python>=4.5.0",
        "Multi-Template-Matching>=2.0.0",
        "PyYAML>=6.0",
        "zarr>=2.10.0",
        "ome-zarr>=0.8.0",
        "pygments>=2.10.0",
        "textual>=3.0.0",
        "textual-serve>=1.0.0",
        "textual-terminal",
        "textual-universal-directorytree",
        "textual-window @ git+https://github.com/trissim/textual-window.git",
        "toolong @ git+https://github.com/trissim/toolong.git",
        "basicpy @ git+https://github.com/trissim/BaSiCPy.git",
        "plotext>=5.2.0",
        "psutil>=5.9.0",
        "GPUtil>=1.4.0",
        "textual-plotext",
        "watchdog>=2.0.0",
        "napari",
        "setuptools",
        "dill>=0.3.0"
    ],
    extras_require={
        "gpu": [
            # PyTorch - compatible with CUDA 12.6 and CuDNN 9.5.x
            "torch>=2.5.0,<2.8.0",
            "torchvision>=0.20.0,<0.22.0",

            # JAX - compatible with CUDA 12.6 and modern dependencies
            "jax>=0.4.38,<0.6.0",
            "jaxlib>=0.4.38,<0.6.0",

            # JAX CUDA plugins - compatible range for CUDA 12.x
            "jax-cuda12-pjrt>=0.5.0,<0.6.0",
            "jax-cuda12-plugin>=0.5.0,<0.6.0",

            # CuPy - CUDA 12.x optimized version
            "cupy-cuda12x>=13.0.0,<14.0.0",

            # CuCIM - CUDA 12 optimized GPU scikit-image (170 functions, missing filters)
            "cucim-cu12>=25.0.0,<26.0.0",

            # TensorFlow - stable version with DLPack support (2.12+ required)
            "tensorflow>=2.15.0,<2.20.0",

            # TensorFlow Probability - for memory-efficient percentile calculations
            "tensorflow-probability[tf]>=0.25.0",

            # pyclesperanto - OpenCL-based GPU image processing
            "pyclesperanto",

            # torbi - GPU-accelerated Viterbi decoding (patched fork for PyTorch 2.6+ compatibility)
            "torbi @ git+https://github.com/trissim/torbi.git"
        ]
    }
)
