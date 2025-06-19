from setuptools import setup, find_packages

# GPU Dependencies Version Notes:
# - JAX 0.4.38: Last version compatible with CuDNN 9.5.x (JAX 0.5.0+ requires CuDNN 9.8+)
# - JAX CUDA plugins must exactly match JAX version for PJRT API compatibility
# - PyTorch 2.7.x: Compatible with CUDA 12.6 and CuDNN 9.5.x
# - TensorFlow 2.15-2.19: Stable DLPack support (2.12+ required for DLPack)
#
# Installation: pip install -e ".[gpu]" for GPU support

setup(
    name="openhcs",
    packages=find_packages(include=["openhcs", "openhcs.*"]),
    install_requires=[
        "numpy>=1.20.0",
        "scikit-image>=0.18.0",
        "scipy>=1.6.0",
        "pandas>=1.2.0",
        "imageio>=2.9.0",
        "tifffile>=2021.1.1",
        "imagecodecs>=2021.1.1",
        #"ashlar>=1.14.0",
        "opencv-python>=4.5.0",
        "PyYAML>=6.0",
        "zarr>=2.10.0",
        "pygments>=2.10.0",
        "textual>=3.0.0",
        "textual-universal-directorytree",
        "watchdog>=2.0.0",
        "napari",
        "setuptools"
    ],
    extras_require={
        "gpu": [
            # PyTorch - compatible with CUDA 12.6 and CuDNN 9.5.x
            "torch>=2.7.0,<2.8.0",
            "torchvision>=0.20.0,<0.21.0",

            # JAX - pinned to 0.4.38 for CuDNN 9.5.x compatibility
            "jax==0.4.38",
            "jaxlib==0.4.38",

            # JAX CUDA plugins - must match JAX version
            "jax-cuda12-pjrt==0.4.38",
            "jax-cuda12-plugin==0.4.38",

            # CuPy - compatible with CUDA 12.x
            "cupy-cuda12x>=13.0.0,<14.0.0",

            # TensorFlow - stable version with DLPack support (2.12+ required)
            "tensorflow>=2.15.0,<2.20.0",

            # TensorFlow Probability - for memory-efficient percentile calculations
            "tensorflow-probability[tf]>=0.25.0"
        ]
    }
)
