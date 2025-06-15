from setuptools import setup

setup(
    name="openhcs",
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
        "napari",
        "setuptools"
    ],
    extras_require={
        "gpu": [
            "torch",
            "torch-vision",
            "jaxlib",
            "cupy",
            "tf-nightly"
        ]
    }
)
