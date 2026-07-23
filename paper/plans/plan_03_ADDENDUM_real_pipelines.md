# Plan 03 ADDENDUM: Real CellProfiler Pipeline Parameters

## Actual BBBC021 Analysis Pipeline

From https://data.broadinstitute.org/bbbc/BBBC021/analysis.cppipe

### Complete Module Sequence

```python
class BBBC021AnalysisPipeline:
    """
    Real CellProfiler pipeline from BBBC021 dataset.

    Modules extracted from actual .cppipe file.
    """

    modules = [
        # 1-3: LoadData (image loading with metadata)
        {
            "type": "LoadData",
            "module_num": 1,
            "images_per_row": 3,  # DAPI, Actin, Tubulin
            "metadata_columns": ["TableNumber", "ImageNumber", "Image_Metadata_SPOT"],
        },

        # 4: Metadata extraction
        {
            "type": "Metadata",
            "module_num": 2,
            "extract_from": "File name",
            "pattern": r"(?P<Plate>.*)_(?P<Well>[A-P][0-9]{2})_s(?P<Site>[0-9]+)_w(?P<Channel>[0-9]+)",
        },

        # 5-7: Apply illumination correction (per channel)
        {
            "type": "CorrectIlluminationApply",
            "module_num": 5,
            "input_image": "OrigDAPI",
            "illumination_function": "IllumDAPI",
            "output_image": "CorrDAPI",
        },
        {
            "type": "CorrectIlluminationApply",
            "module_num": 6,
            "input_image": "OrigActin",
            "illumination_function": "IllumActin",
            "output_image": "CorrActin",
        },
        {
            "type": "CorrectIlluminationApply",
            "module_num": 7,
            "input_image": "OrigTubulin",
            "illumination_function": "IllumTubulin",
            "output_image": "CorrTubulin",
        },

        # 11: Preprocessing - morphological opening on DAPI
        {
            "type": "Opening",
            "module_num": 11,
            "input_image": "CorrDAPI",
            "output_image": "OpenedDAPI",
            "structuring_element": "disk",
            "radius": 5,
        },

        # 12: Nuclei segmentation
        {
            "type": "IdentifyPrimaryObjects",
            "module_num": 12,
            "input_image": "OpenedDAPI",
            "output_objects": "Nuclei",
            "typical_diameter": (15, 115),  # pixels
            "threshold_method": "Otsu",
            "threshold_scope": "Global",
            "threshold_smoothing_scale": 1.3488,
            "automatic_smoothing": False,
            "declump_method": "Shape",
            "fill_holes": True,
            "size_range": (15, 115),  # Filter by size
        },

        # 13: Cell segmentation (secondary objects)
        {
            "type": "IdentifySecondaryObjects",
            "module_num": 13,
            "input_objects": "Nuclei",
            "input_image": "CorrActin",  # Use Actin to find cell boundaries
            "output_objects": "Cells",
            "method": "Watershed - Image",
            "distance_to_dilate": 10,  # pixels
        },

        # 14: Cytoplasm (tertiary objects)
        {
            "type": "IdentifyTertiaryObjects",
            "module_num": 14,
            "primary_objects": "Nuclei",
            "secondary_objects": "Cells",
            "output_objects": "Cytoplasm",
        },

        # 15-17: Intensity measurements (per compartment)
        {
            "type": "MeasureObjectIntensity",
            "module_num": 15,
            "objects": "Nuclei",
            "images": ["CorrDAPI", "CorrActin", "CorrTubulin"],
        },
        {
            "type": "MeasureObjectIntensity",
            "module_num": 16,
            "objects": "Cells",
            "images": ["CorrDAPI", "CorrActin", "CorrTubulin"],
        },
        {
            "type": "MeasureObjectIntensity",
            "module_num": 17,
            "objects": "Cytoplasm",
            "images": ["CorrDAPI", "CorrActin", "CorrTubulin"],
        },

        # 18-20: Size and shape measurements
        {
            "type": "MeasureObjectSizeShape",
            "module_num": 18,
            "objects": "Nuclei",
            "zernike_degree": 9,  # Zernike shape moments
        },
        {
            "type": "MeasureObjectSizeShape",
            "module_num": 19,
            "objects": "Cells",
            "zernike_degree": 9,
        },
        {
            "type": "MeasureObjectSizeShape",
            "module_num": 20,
            "objects": "Cytoplasm",
            "zernike_degree": 9,
        },

        # 21-23: Texture measurements (Haralick features)
        {
            "type": "MeasureTexture",
            "module_num": 21,
            "objects": "Nuclei",
            "images": ["CorrDAPI", "CorrActin", "CorrTubulin"],
            "scales": [5, 10, 20],  # pixels
        },
        {
            "type": "MeasureTexture",
            "module_num": 22,
            "objects": "Cells",
            "images": ["CorrDAPI", "CorrActin", "CorrTubulin"],
            "scales": [5, 10, 20],
        },
        {
            "type": "MeasureTexture",
            "module_num": 23,
            "objects": "Cytoplasm",
            "images": ["CorrDAPI", "CorrActin", "CorrTubulin"],
            "scales": [5, 10, 20],
        },

        # 24: Granularity (multi-scale morphology)
        {
            "type": "MeasureGranularity",
            "module_num": 24,
            "images": ["CorrDAPI", "CorrActin", "CorrTubulin"],
            "granularity_range": (2, 16),  # pixels
        },

        # 25: Object neighbors (spatial features)
        {
            "type": "MeasureObjectNeighbors",
            "module_num": 25,
            "objects": "Cells",
            "neighbor_objects": "Cells",
            "distance_method": "Adjacent",
        },
        {
            "type": "MeasureObjectNeighbors",
            "module_num": 26,
            "objects": "Nuclei",
            "neighbor_objects": "Nuclei",
            "distance_method": "Expand until adjacent",
            "distance": 2,  # pixels
        },

        # 27: Export to database/CSV
        {
            "type": "ExportToDatabase",
            "module_num": 27,
            "database_type": "SQLite",
            "output_per_object_tables": True,
            "metadata_fields": ["Plate", "Well", "Site"],
        },
    ]
```

## CellProfiler Pipeline Generator

```python
class CellProfilerPipelineGenerator:
    """
    Generate .cppipe XML files programmatically.

    Based on actual BBBC021/022 pipelines.
    """

    def __init__(self):
        self.modules = []
        self.module_counter = 1

    def add_load_data(
        self,
        csv_path: str,
        image_columns: dict[str, str]
    ) -> "CellProfilerPipelineGenerator":
        """
        Add LoadData module.

        Args:
            csv_path: Path to CSV file listing images
            image_columns: Dict mapping channel_name → CSV column name
        """

        self.modules.append({
            'type': 'LoadData',
            'module_num': self.module_counter,
            'csv_location': csv_path,
            'image_columns': image_columns,
        })
        self.module_counter += 1
        return self

    def add_illumination_correction(
        self,
        image_name: str,
        icf_name: str
    ) -> "CellProfilerPipelineGenerator":
        """Add CorrectIlluminationApply module."""

        self.modules.append({
            'type': 'CorrectIlluminationApply',
            'module_num': self.module_counter,
            'input_image': image_name,
            'illumination_function': icf_name,
            'output_image': f"Corr{image_name}",
        })
        self.module_counter += 1
        return self

    def add_nuclei_segmentation(
        self,
        input_image: str,
        diameter_range: tuple[int, int] = (15, 115),
        threshold_method: str = "Otsu Global",
        declump_method: str = "Shape"
    ) -> "CellProfilerPipelineGenerator":
        """Add IdentifyPrimaryObjects for nuclei."""

        # Optional: add Opening preprocessing
        self.modules.append({
            'type': 'Opening',
            'module_num': self.module_counter,
            'input_image': input_image,
            'output_image': f"Opened{input_image}",
            'structuring_element': 'disk',
            'radius': 5,
        })
        self.module_counter += 1

        # Primary object identification
        self.modules.append({
            'type': 'IdentifyPrimaryObjects',
            'module_num': self.module_counter,
            'input_image': f"Opened{input_image}",
            'output_objects': 'Nuclei',
            'typical_diameter': diameter_range,
            'threshold_method': threshold_method,
            'declump_method': declump_method,
            'fill_holes': True,
        })
        self.module_counter += 1
        return self

    def add_cell_segmentation(
        self,
        cell_boundary_image: str,
        distance_to_dilate: int = 10
    ) -> "CellProfilerPipelineGenerator":
        """Add IdentifySecondaryObjects for cells."""

        self.modules.append({
            'type': 'IdentifySecondaryObjects',
            'module_num': self.module_counter,
            'input_objects': 'Nuclei',
            'input_image': cell_boundary_image,
            'output_objects': 'Cells',
            'method': 'Watershed - Image',
            'distance_to_dilate': distance_to_dilate,
        })
        self.module_counter += 1

        # Add cytoplasm (tertiary)
        self.modules.append({
            'type': 'IdentifyTertiaryObjects',
            'module_num': self.module_counter,
            'primary_objects': 'Nuclei',
            'secondary_objects': 'Cells',
            'output_objects': 'Cytoplasm',
        })
        self.module_counter += 1
        return self

    def add_measurements(
        self,
        images: list[str],
        compartments: list[str] = ["Nuclei", "Cells", "Cytoplasm"]
    ) -> "CellProfilerPipelineGenerator":
        """Add standard measurement modules."""

        # Intensity
        for compartment in compartments:
            self.modules.append({
                'type': 'MeasureObjectIntensity',
                'module_num': self.module_counter,
                'objects': compartment,
                'images': images,
            })
            self.module_counter += 1

        # Size/Shape
        for compartment in compartments:
            self.modules.append({
                'type': 'MeasureObjectSizeShape',
                'module_num': self.module_counter,
                'objects': compartment,
                'zernike_degree': 9,
            })
            self.module_counter += 1

        # Texture
        for compartment in compartments:
            self.modules.append({
                'type': 'MeasureTexture',
                'module_num': self.module_counter,
                'objects': compartment,
                'images': images,
                'scales': [5, 10, 20],
            })
            self.module_counter += 1

        # Granularity (image-level, not per object)
        self.modules.append({
            'type': 'MeasureGranularity',
            'module_num': self.module_counter,
            'images': images,
            'granularity_range': (2, 16),
        })
        self.module_counter += 1

        # Neighbors
        self.modules.append({
            'type': 'MeasureObjectNeighbors',
            'module_num': self.module_counter,
            'objects': 'Cells',
            'neighbor_objects': 'Cells',
            'distance_method': 'Adjacent',
        })
        self.module_counter += 1

        return self

    def add_export(
        self,
        output_path: Path,
        metadata_fields: list[str]
    ) -> "CellProfilerPipelineGenerator":
        """Add export module."""

        self.modules.append({
            'type': 'ExportToDatabase',
            'module_num': self.module_counter,
            'database_type': 'SQLite',
            'output_file': str(output_path),
            'metadata_fields': metadata_fields,
        })
        self.module_counter += 1
        return self

    def generate_cppipe(self, output_path: Path):
        """
        Generate CellProfiler .cppipe XML file.

        This is a simplified template - real .cppipe files are verbose XML.
        """

        # CellProfiler pipelines are XML with specific structure
        # For brevity, showing JSON representation that would be converted to XML

        pipeline = {
            'CellProfiler Pipeline': {
                'DateRevision': 20240101,
                'GitHash': 'unknown',
                'ModuleCount': len(self.modules),
                'HasImagePlaneDetails': False,
            },
            'Modules': self.modules
        }

        # In reality, need to convert to XML format
        # See: https://github.com/CellProfiler/CellProfiler/wiki/CellProfiler-pipeline-file-format

        import json
        with open(output_path, 'w') as f:
            json.dump(pipeline, f, indent=2)

        # TODO: Convert JSON to actual .cppipe XML format
        # For now, save as JSON template that CellProfiler can't read
        # Need XML conversion library or manual template

        return output_path
```

## Usage Example

```python
# Generate BBBC021-equivalent pipeline
generator = CellProfilerPipelineGenerator()

pipeline = (
    generator
    .add_load_data(
        csv_path="BBBC021_v1_image.csv",
        image_columns={
            'DAPI': 'PathName_DAPI',
            'Actin': 'PathName_Actin',
            'Tubulin': 'PathName_Tubulin',
        }
    )
    .add_illumination_correction('DAPI', 'IllumDAPI')
    .add_illumination_correction('Actin', 'IllumActin')
    .add_illumination_correction('Tubulin', 'IllumTubulin')
    .add_nuclei_segmentation(
        input_image='CorrDAPI',
        diameter_range=(15, 115),
    )
    .add_cell_segmentation(
        cell_boundary_image='CorrActin',
        distance_to_dilate=10,
    )
    .add_measurements(
        images=['CorrDAPI', 'CorrActin', 'CorrTubulin'],
        compartments=['Nuclei', 'Cells', 'Cytoplasm']
    )
    .add_export(
        output_path=Path("results.db"),
        metadata_fields=['Plate', 'Well', 'Site']
    )
    .generate_cppipe(Path("benchmark_nuclei_segmentation.cppipe"))
)
```

## ImageJ Macro Equivalent

No published ImageJ macros exist for BBBC datasets. Here's a manual translation:

```java
// ImageJ Macro: Nuclei Segmentation (BBBC021-equivalent)
// Translated from CellProfiler analysis.cppipe

// 1. Open DAPI image
open(dapi_path);
dapi = getTitle();

// 2. Apply illumination correction (if ICF available)
imageCalculator("Divide create 32-bit", dapi, "IllumDAPI");
rename("CorrDAPI");

// 3. Morphological opening (disk, radius=5)
run("Morphological Filters", "operation=Opening element=Disk radius=5");
rename("OpenedDAPI");

// 4. Threshold (Otsu)
setAutoThreshold("Otsu dark");
run("Convert to Mask");

// 5. Watershed (declumping)
run("Watershed");

// 6. Analyze particles (size filter: 15-115 px diameter)
// Area = π * (d/2)^2, so d=15 → area=177, d=115 → area=10387
run("Analyze Particles...",
    "size=177-10387 " +
    "circularity=0.00-1.00 " +
    "show=Outlines " +
    "display exclude clear add");

// 7. Measure intensity in corrected channels
selectWindow("CorrDAPI");
roiManager("Measure");

selectWindow("CorrActin");
roiManager("Measure");

selectWindow("CorrTubulin");
roiManager("Measure");

// 8. Save results
saveAs("Results", "nuclei_measurements.csv");

// 9. Save ROIs
roiManager("Save", "nuclei_rois.zip");
```

### ImageJ Macro Generator

```python
class ImageJMacroGenerator:
    """Generate ImageJ macros from pipeline definitions."""

    def __init__(self):
        self.commands = []

    def add_opening(self, image: str, radius: int):
        self.commands.append(
            f'run("Morphological Filters", '
            f'"operation=Opening element=Disk radius={radius}");'
        )
        return self

    def add_threshold(self, method: str = "Otsu"):
        self.commands.append(f'setAutoThreshold("{method} dark");')
        self.commands.append('run("Convert to Mask");')
        return self

    def add_watershed(self):
        self.commands.append('run("Watershed");')
        return self

    def add_analyze_particles(
        self,
        size_min: float,
        size_max: float,
        output: str = "Outlines"
    ):
        self.commands.append(
            f'run("Analyze Particles...", '
            f'"size={size_min}-{size_max} '
            f'circularity=0.00-1.00 '
            f'show={output} '
            f'display exclude clear add");'
        )
        return self

    def generate_macro(self, output_path: Path):
        """Write ImageJ macro file."""

        macro = "// Auto-generated ImageJ macro\n\n"
        macro += "\n".join(self.commands)

        with open(output_path, 'w') as f:
            f.write(macro)

        return output_path
```

## Gap: XML Generation

**BLOCKED**: Neither I nor publications provide actual .cppipe XML generation.

**Workaround**:
1. Use CellProfiler GUI to create template
2. Modify template programmatically (search/replace)
3. Or: use CellProfiler Python API directly instead of .cppipe files

```python
# Alternative: CellProfiler Python API (if available)
import cellprofiler_core.pipeline as cpp
import cellprofiler_core.module as cpm

pipeline = cpp.Pipeline()

# Add modules
load_data = pipeline.create_module("LoadData")
load_data.csv_file_name.value = "BBBC021_v1_image.csv"

identify_primary = pipeline.create_module("IdentifyPrimaryObjects")
identify_primary.image_name.value = "DNA"
identify_primary.object_name.value = "Nuclei"
identify_primary.size_range.min = 15
identify_primary.size_range.max = 115

# Save pipeline
pipeline.save("benchmark_nuclei.cppipe")
```

This requires CellProfiler Python package to be installed in benchmark environment.
