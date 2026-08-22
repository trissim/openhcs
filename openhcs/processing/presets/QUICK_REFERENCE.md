# Pipeline Presets Quick Reference

**Fast lookup guide for choosing and using OpenHCS pipeline presets**

---

## 🎯 Which Pipeline Should I Use?

### I need to stitch multi-site images

**Use**: `10x_mfd_stitch_gpu.py` or `imx_96_well_neurite_outgrowth_pipeline_gpu.py`

**When**:
- Multi-site acquisition (tiled imaging)
- Need seamless assembled image
- Have GPU available

**CPU Alternative**: `*_cpu.py` variants (slower but works without GPU)

---

### I'm analyzing microfluidic devices

**Use**: `10x_mfd_crop_analyze.py` or `cy5_axon_cell_body_crop_analysis.py`

**When**:
- Microfluidic device imaging
- Need template-based device detection
- Want compartment-specific analysis

**Variants**:
- `10x_mfd_crop_analyze.py` - Basic 2-channel analysis
- `10x_mfd_crop_analyze_dapi-fitc-cy5.py` - 3-channel analysis
- `cy5_axon_cell_body_crop_analysis.py` - Separate cell body + axon

---

### I just need basic preprocessing

**Use**: `cy5_ctb_cell_count.py` or `test.py`

**When**:
- Simple crop + filter + normalize
- Testing processing parameters
- Quick image cleanup

---

## 📊 Pipeline Comparison Table

| Pipeline | Sites | Channels | Analysis | GPU | Lines |
|----------|-------|----------|----------|-----|-------|
| `10x_mfd_stitch_gpu.py` | Multi | 4 | None | Yes | 133 |
| `10x_mfd_stitch_ashlar_cpu.py` | Multi | 4 | None | No | 133 |
| `imx_96_well_neurite_outgrowth_pipeline_gpu.py` | Multi | Multi | None | Yes | 74 |
| `imx_96_well_neurite_outgrowth_pipeline_cpu.py` | Multi | Multi | None | No | 74 |
| `10x_mfd_crop_analyze.py` | Single | 4 | Cell + Neurite | Either | 79 |
| `10x_mfd_crop_analyze_dapi-fitc-cy5.py` | Single | 4 | Cell + Neurite | Either | 87 |
| `cy5_axon_cell_body_crop_analysis.py` | Single | 4 | Cell + Neurite | Either | 119 |
| `cy5_ctb_cell_count.py` | Single | 1 | None | Yes | 47 |
| `test.py` | Single | 1 | None | Yes | 27 |

---

## 🔧 Common Customizations

### Change normalization percentiles

**Find**:
```python
(stack_percentile_normalize, {
    'low_percentile': 0.1,
    'high_percentile': 99.9
})
```

**Modify**: Adjust percentiles (0.1-5.0 for low, 95.0-99.9 for high)

---

### Adjust cell size detection

**Find**:
```python
(count_cells_single_channel, {
    'min_cell_area': 40,
    'max_cell_area': 200,
    ...
})
```

**Modify**:
- Small cells (nuclei): 40-200
- Medium cells: 100-500
- Large cells: 200-1000

---

### Change stitching parameters

**Find**:
```python
(ashlar_compute_tile_positions_gpu, {
    'stitch_alpha': 0.2
})
```

**Modify**: `stitch_alpha` (0.0-1.0, higher = more blending)

---

### Modify crop dimensions

**Find**:
```python
(crop, {
    'width': 5046,
    'height': 3694,
    'start_x': 0,
    'start_y': 0
})
```

**Modify**: Adjust dimensions and offsets for your device

---

## 🚀 Quick Start

### 1. Copy a preset

```bash
cp openhcs/processing/presets/pipelines/10x_mfd_stitch_gpu.py my_pipeline.py
```

### 2. Edit parameters

Open `my_pipeline.py` and customize:
- Normalization percentiles
- Cell size ranges
- Crop dimensions
- Channel-specific processing

### 3. Load in GUI

1. Open OpenHCS
2. Pipeline Editor → Load Pipeline
3. Select `my_pipeline.py`
4. Execute

---

## 💡 Tips

### GPU vs CPU

- **GPU pipelines** are 5-10x faster for stitching
- **CPU pipelines** work everywhere but slower
- **All other processing** (CuPy, PyTorch) uses GPU in both variants

### Channel Processing

- **Channel 1 (DAPI)**: Usually gets edge detection (sobel)
- **Other channels**: Usually get background removal (tophat)
- **Customize** by editing the `func={}` dict

### Empty Channel Slots

```python
'1': [],  # Skip processing for channel 1
```

Use empty list `[]` to skip a channel

---

## 📖 Full Documentation

- **Detailed Analysis**: `docs/pipeline_preset_patterns.md`
- **User Guide**: `openhcs/processing/presets/README.md`
- **Summary**: `docs/PIPELINE_PRESETS_SUMMARY.md`

---

## 🆘 Troubleshooting

### "GPU out of memory"

→ Use CPU variant or reduce `num_workers` in global config

### "Template not found"

→ Put the matching image at
`<plate>/templates/mfd_96_sobel_10x_whole_device.tif`, or update the authored
plate-relative `template_path`

### "Wrong cell count"

→ Adjust `min_cell_area` and `max_cell_area` parameters

### "Stitching misaligned"

→ Adjust `stitch_alpha` (try 0.1-0.5 range)

---

## 🔗 Quick Links

**Location**: `openhcs/processing/presets/pipelines/`

**Import Example**:
```python
from openhcs.processing.presets.pipelines.imx_96_well_neurite_outgrowth_pipeline_gpu import pipeline_steps
```

**Load in Code**:
```python
from openhcs.core.config import PipelineConfig
pipeline_config = PipelineConfig()
```
