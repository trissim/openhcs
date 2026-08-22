# OpenHCS Pipeline Presets

This directory contains production-tested pipeline examples that demonstrate common OpenHCS workflows. These pipelines serve as:

1. **Templates** for users building similar workflows
2. **Documentation** of real-world usage patterns
3. **Foundation** for the future preset system

## Directory Structure

```
presets/
├── pipelines/          # User-generated pipeline examples
│   ├── *_gpu.py       # GPU-accelerated variants
│   ├── *_cpu.py       # CPU-only variants
│   └── *.py           # Backend-agnostic pipelines
└── README.md          # This file
```

## Pipeline Categories

### 🔬 Stitching Pipelines (Multi-site Assembly)

High-throughput workflows for assembling multi-site acquisitions into seamless images.

| Pipeline | Backend | Sites | Channels | Description |
|----------|---------|-------|----------|-------------|
| `10x_mfd_stitch_gpu.py` | GPU | Multi | 4 | Multi-channel preprocessing + GPU stitching |
| `10x_mfd_stitch_ashlar_cpu.py` | CPU | Multi | 4 | Same as GPU but CPU-based stitching |
| `imx_96_well_neurite_outgrowth_pipeline_gpu.py` | GPU | Multi | Multi | Full workflow with Z-projection + stitching |
| `imx_96_well_neurite_outgrowth_pipeline_cpu.py` | CPU | Multi | Multi | CPU variant of ImageXpress workflow |

**Common Pattern**:
```
preprocess → composite → find_positions → preprocess_again → assemble
```

**Key Features**:
- Per-channel preprocessing (normalize + filter)
- Composite image creation for alignment
- Ashlar-based position computation
- Re-preprocessing from original data
- Final assembly with blending

---

### 🧫 Microfluidic Device Analysis

Specialized workflows for microfluidic device imaging with template-based cropping.

| Pipeline | Compartments | Analysis | Description |
|----------|--------------|----------|-------------|
| `10x_mfd_crop_analyze.py` | 4 | Cell count + Neurite | Basic 2-channel analysis |
| `10x_mfd_crop_analyze_dapi-fitc-cy5.py` | 4 | Cell count + Neurite | Extended 3-channel variant |
| `cy5_axon_cell_body_crop_analysis.py` | 2 (dual) | Cell count + Neurite | Separate cell body & axon analysis |

**Common Pattern**:
```
template_crop → compartment_crop → analysis
```

**Key Features**:
- Template matching for device detection
- Per-compartment spatial cropping
- Channel-specific analysis (cell counting, neurite tracing)
- Optional dual-compartment processing

The maintained crop presets author
`templates/mfd_96_sobel_10x_whole_device.tif` relative to the plate directory.
Place the matching template there or edit that relative path for a different
plate layout.

---

### 🔍 Simple Processing Pipelines

Lightweight workflows for basic image processing tasks.

| Pipeline | Steps | Backend | Description |
|----------|-------|---------|-------------|
| `cy5_ctb_cell_count.py` | 2 | Pyclesperanto | Crop + tophat + normalize |
| `test.py` | 1 | Pyclesperanto | Minimal blur + normalize test |

**Common Pattern**:
```
crop → filter → normalize
```

### 🧠 Loose Opera Phenix Neurite Outgrowth

`loose_operaphenix_neurite_outgrowth.py` is the current CellProfiler-backed
example for a selected set of Opera Phenix TIFFs copied without `Index.xml`.
Its editable `example_inputs` boundary declares exact source files plus
well/site/Z/time identities. With a MAP2 source, MAP2 bodies seed SMI312
propagation. With `map2=None`, the source universe contains only Hoechst and
SMI312, and SMI312 supplies both the neuronal-body and neurite signal. The
compact MetaXpress preset additionally uses Hoechst for nuclear seeds; the
modular preset does not consume that source. The two-channel compact form can
include non-neuronal nuclei; use MAP2 when neuronal-body specificity is
required. Both forms stream a final
`UnifiedNeurons` label layer where each seed and its assigned neurites share one
identity; the diagnostic skeleton is not presented as the final result.

`loose_operaphenix_neurite_outgrowth_metaxpress.py` uses that same source
identity boundary for a compact one-`FunctionStep` alternative. Its public
settings select Hoechst nuclei, the configured body source (MAP2 when present,
otherwise SMI312), and SMI312 neurites from the assembled CHANNEL stack. The
callable returns typed summary/cell measurements, diagnostic masks, and a final
CellProfiler-propagated `neurons` label artifact where every body and its owned
outgrowth have the same label identity.

Use the native Opera Phenix microscope handler for complete plate exports that
still contain `Index.xml`.

---

## GPU vs CPU Variants

### Differences

GPU and CPU variants differ **only** in the stitching function:

**GPU**:
```python
from openhcs.processing.backends.pos_gen.ashlar_main_gpu import ashlar_compute_tile_positions_gpu
func=(ashlar_compute_tile_positions_gpu, {'stitch_alpha': 0.2})
```

**CPU**:
```python
from openhcs.processing.backends.pos_gen.ashlar_main_cpu import ashlar_compute_tile_positions_cpu
func=(ashlar_compute_tile_positions_cpu, {'stitch_alpha': 0.2})
```

All other processing (CuPy, PyTorch, analysis) remains identical.

### When to Use Each

- **GPU**: Faster stitching (recommended when GPU available)
- **CPU**: Fallback for systems without GPU or when GPU memory is constrained

---

## Common Parameters

### Percentile Normalization

```python
low_percentile: 0.1 or 1.0
high_percentile: 99.0 or 99.9
target_max: 65535.0  # Optional
```

### Morphological Filtering

**Tophat** (background removal):
```python
selem_radius: 50  # Default
downsample_factor: 4  # Optional speedup
```

**Sobel** (edge detection):
```python
slice_by_slice: True
```

### Cell Counting

**Small cells** (nuclei):
```python
min_cell_area: 40
max_cell_area: 200-300
detection_method: DetectionMethod.WATERSHED
enable_preprocessing: False
```

**Large cells** (whole cells):
```python
min_cell_area: 100
max_cell_area: 1000
```

### Neurite/Axon Analysis

```python
analysis_dimension: AnalysisDimension.TWO_D
return_skeleton_visualizations: True
skeleton_visualization_mode: OutputMode.SKELETON
min_branch_length: 20.0
```

### Stitching

```python
stitch_alpha: 0.2  # Universal constant
overlap_ratio: 0.1
max_shift: 15.0
```

---

## Usage Examples

### Loading a Pipeline in the GUI

1. Open OpenHCS PyQt or TUI interface
2. Navigate to Pipeline Editor
3. Click "Load Pipeline"
4. Select a preset from `openhcs/processing/presets/pipelines/`
5. Customize parameters as needed
6. Save and execute

### Using as a Template

```python
# Copy a preset and modify for your needs
cp openhcs/processing/presets/pipelines/10x_mfd_stitch_gpu.py my_custom_pipeline.py

# Edit the file to customize:
# - Channel-specific processing
# - Analysis parameters
# - Spatial cropping dimensions
# - Output configurations
```

### Programmatic Usage

```python
# Import the pipeline steps
from openhcs.processing.presets.pipelines.imx_96_well_neurite_outgrowth_pipeline_gpu import pipeline_steps

# Use pipeline_steps with a PipelineConfig in your workflow
from openhcs.core.config import PipelineConfig
pipeline_config = PipelineConfig()
```

---

## Advanced Patterns

### Input Source Branching

Process data multiple ways from the same starting point:

```python
# First analysis branch
step_1 = FunctionStep(func=analyze_compartment_1, name="branch_1")

# Second branch (restarts from original data)
step_2 = FunctionStep(
    func=analyze_compartment_2,
    name="branch_2",
    input_source=InputSource.PIPELINE_START
)
```

**Use Cases**:
- Dual-compartment analysis (cell body + axon)
- Multiple stitching strategies
- Parallel analysis paths

### Variable Components

Control which dimensions are collapsed:

```python
variable_components=[VariableComponents.CHANNEL]  # Collapse channels → composite
variable_components=[VariableComponents.SITE]     # Collapse sites → stitched
variable_components=[VariableComponents.Z_INDEX]  # Collapse Z → projection
```

### Per-Channel Processing

Apply different operations to different channels:

```python
func={
    '1': [sobel, normalize],      # DAPI: edge detection
    '2': [tophat, normalize],     # GFP: background removal
    '3': [tophat, normalize],     # Cy5: background removal
    '4': []                       # Skip channel 4
}
```

---

## File Naming Conventions

- `<microscope>_<assay>_<workflow>_<backend>.py`
  - `microscope`: `10x_mfd`, `imx`, `cy5`, etc.
  - `assay`: `crop_analyze`, `stitch`, `cell_count`, etc.
  - `workflow`: Descriptive name
  - `backend`: `_gpu`, `_cpu`, or omitted if agnostic

**Examples**:
- `10x_mfd_stitch_gpu.py` - 10x microfluidic device, stitching, GPU
- `imx_96_well_neurite_outgrowth_pipeline_cpu.py` - ImageXpress, 96-well neurite assay, CPU
- `cy5_ctb_cell_count.py` - Cy5 channel, CTB staining, cell counting (backend-agnostic)

---

## Contributing New Presets

When adding new pipeline presets:

1. **Test thoroughly** on real data
2. **Document parameters** with comments
3. **Follow naming conventions**
4. **Create both GPU and CPU variants** if using stitching
5. **Update this README** with the new pipeline
6. **Add to pattern analysis** in `docs/pipeline_preset_patterns.md`

---

## Related Documentation

- **Pattern Analysis**: `docs/pipeline_preset_patterns.md` - Detailed analysis of pipeline patterns
- **Complete Examples**: `docs/source/guides/complete_examples.rst` - Full programmatic examples
- **Production Examples**: `docs/source/user_guide/production_examples.rst` - Real-world workflows
- **Pipeline Architecture**: `docs/source/architecture/` - System design documentation

---

## Future: Preset System

These pipelines will inform the design of a formal preset system that will:

1. **Template-based creation** - Select archetype, fill in parameters
2. **Smart defaults** - Channel-specific processing based on naming
3. **Parameter inheritance** - Global settings with step-level overrides
4. **GPU auto-detection** - Automatic backend selection
5. **Visual preset editor** - GUI for customizing templates

See `docs/pipeline_preset_patterns.md` for detailed design recommendations.
