#!/usr/bin/env python3
"""
OpenHCS Pipeline - BBBC022 Nuclei Segmentation (CellProfiler-equivalent)

This pipeline replicates CellProfiler's IdentifyPrimaryObjects for BBBC022 dataset.
CellProfiler parameters from ExampleFly.cppipe:
- Typical diameter: 10-40 pixels
- Threshold strategy: Global
- Thresholding method: Minimum Cross-Entropy (three-class)
- Declumping: Shape
- Fill holes: After both thresholding and declumping

BBBC022: U2OS cells - Cell Painting (6 channels: DAPI, ER, RNA, AGP, Mito, Syto14)
384-well plate, 9 sites per well, 3456 images per plate

Backend options (uncomment one):
- CPU: preprocess_cpu + identify_primary_objects (numpy/scipy)
- GPU (OpenCL): preprocess_gpu + identify_primary_objects_gpu (pyclesperanto)
- GPU (CUDA): preprocess_cupy + identify_primary_objects_cupy (cupy/cucim)
"""

# Core imports
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.config import (
    LazyProcessingConfig,
    LazyStepMaterializationConfig,
    LazyNapariStreamingConfig,
)
from openhcs.constants.constants import VariableComponents

# ============================================================================
# SELECT BACKEND (uncomment one set)
# ============================================================================

# --- CPU Backend (numpy/scipy) ---
from benchmark.pipelines.cellprofiler_preprocess import preprocess_cpu as preprocess
from benchmark.pipelines.cellprofiler_nuclei import identify_primary_objects as segment

# --- GPU Backend (pyclesperanto - OpenCL, works on AMD/NVIDIA/Intel) ---
# from benchmark.pipelines.cellprofiler_preprocess import preprocess_gpu as preprocess
# from benchmark.pipelines.cellprofiler_nuclei_gpu import identify_primary_objects_gpu as segment

# --- GPU Backend (cupy/cucim - CUDA, NVIDIA only, fastest) ---
# from benchmark.pipelines.cellprofiler_preprocess import preprocess_cupy as preprocess
# from benchmark.pipelines.cellprofiler_nuclei_cupy import identify_primary_objects_cupy as segment

# ============================================================================
# PIPELINE
# ============================================================================

pipeline_steps = []

# CellProfiler preprocessing parameters (from ExampleFly.cppipe)
# Gaussian sigma = diameter / 3.5 = 10 / 3.5 ≈ 2.9
GAUSSIAN_SIGMA = 2.9

# Step 1: Preprocessing (Gaussian smoothing)
step_1 = FunctionStep(
    func=(preprocess, {
        'gaussian_sigma': GAUSSIAN_SIGMA,
        'median_size': 0,
    }),
    name="CellProfiler Preprocessing",
    processing_config=LazyProcessingConfig(
        variable_components=[VariableComponents.CHANNEL]
    ),
)
pipeline_steps.append(step_1)

# Step 2: Nuclei Segmentation (IdentifyPrimaryObjects)
step_2 = FunctionStep(
    func=(segment, {
        'min_diameter': 10,                         # CellProfiler: 10 pixels min
        'max_diameter': 40,                         # CellProfiler: 40 pixels max
        'threshold_method': 'minimum_cross_entropy', # CellProfiler: Min Cross-Entropy
        'threshold_correction': 1.0,
        'declump_method': 'shape',                  # CellProfiler: Shape declumping
        'fill_holes': True,
        'discard_border_objects': True,
        'discard_outside_diameter': True,
    }),
    name="IdentifyPrimaryObjects (Nuclei)",
    processing_config=LazyProcessingConfig(
        variable_components=[VariableComponents.CHANNEL]
    ),
    napari_streaming_config=LazyNapariStreamingConfig(),
    step_materialization_config=LazyStepMaterializationConfig(),
)
pipeline_steps.append(step_2)

