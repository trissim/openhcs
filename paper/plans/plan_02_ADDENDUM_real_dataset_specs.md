# Plan 02 ADDENDUM: Real BBBC Dataset Specifications

## Filled Gaps from Publication Research

### BBBC021 Complete Specification

```python
@dataclass(frozen=True)
class BBBC021Dataset:
    """BBBC021v1: Human MCF7 cells - compound profiling (Caie et al., MCT 2010)."""

    id: str = "BBBC021"

    # Multiple ZIP files - 55 plates total
    base_url: str = "https://data.broadinstitute.org/bbbc/BBBC021/"
    archives: list[str] = field(default_factory=lambda: [
        "BBBC021_v1_images_Week1_22123.zip",  # 839 MB
        "BBBC021_v1_images_Week1_22141.zip",  # ~750 MB each
        # ... 53 more ZIPs (full list in metadata CSV)
    ])

    # Metadata files
    metadata_urls: dict[str, str] = field(default_factory=lambda: {
        "image": "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_image.csv",  # 3.8 MB
        "compound": "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_compound.csv",  # 8 KB
        "moa": "https://data.broadinstitute.org/bbbc/BBBC021/BBBC021_v1_moa.csv",  # 4.4 KB
    })

    # CellProfiler pipelines (ground truth for comparison)
    pipeline_urls: dict[str, str] = field(default_factory=lambda: {
        "analysis": "https://data.broadinstitute.org/bbbc/BBBC021/analysis.cppipe",
        "illumination": "https://data.broadinstitute.org/bbbc/BBBC021/illum.cppipe",
    })

    # Filename pattern: {Well}_{Site}_{Channel}{UUID}.tif
    # Example: G10_s1_w1BEDC2073-A983-4B98-95E9-84466707A25D.tif
    filename_regex: str = r"(?P<well>[A-P]\d{2})_s(?P<site>\d+)_w(?P<channel>[124])(?P<uuid>[A-F0-9-]+)\.tif"

    # Dataset statistics
    total_images: int = 39_600  # 13,200 FOVs × 3 channels
    total_fovs: int = 13_200
    num_plates: int = 55
    channels: dict[str, str] = field(default_factory=lambda: {
        "w1": "DAPI",
        "w2": "Tubulin",
        "w4": "Actin"
    })

    # Image format
    format: str = "TIFF"
    bit_depth: int = 16  # Assumed from typical ImageXpress

    # Ground truth
    has_segmentation_masks: bool = False
    has_moa_labels: bool = True
    moa_label_count: int = 103  # compound-concentrations with MoA labels
    moa_classes: int = 12  # Different mechanisms of action

    # Total size (all ZIPs)
    size_bytes: int = 41_250_000_000  # ~41 GB

    # Checksums: NOT PROVIDED by Broad
    # Recommendation: compute on first download and cache, or skip verification
    checksum_strategy: str = "none"  # Options: "none", "compute_and_cache", "user_provided"

    # Preprocessing (from Singh 2014 + Caie 2010)
    recommended_preprocessing: dict = field(default_factory=lambda: {
        "illumination_correction": {
            "method": "median_filter",
            "smoothing_sigma": 500,  # pixels
            "grouping": "by_plate",  # Compute ICF per plate
            "robust_minimum_percentile": 0.02,
        },
        "intensity_normalization": {
            "method": "percentile_clipping",
            "low_percentile": 0.1,
            "high_percentile": 99.9,
            "output_range": [0, 1],
        }
    })

    # Subsetting for quick benchmarks
    quick_subset: dict = field(default_factory=lambda: {
        "plates": ["Week1_22123"],  # Single plate
        "expected_images": 720,  # Approximate
        "size_mb": 839,
    })
```

### BBBC022 Complete Specification

```python
@dataclass(frozen=True)
class BBBC022Dataset:
    """BBBC022v1: U2OS cells - Cell Painting (Gustafsdottir et al., GigaScience 2017)."""

    id: str = "BBBC022"
    base_url: str = "https://data.broadinstitute.org/bbbc/BBBC022/"

    # 100 ZIP files (plate × channel combinations)
    # Full list in BBBC022_v1_images_urls.txt
    archive_list_url: str = "https://data.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_images_urls.txt"

    # Metadata
    metadata_urls: dict[str, str] = field(default_factory=lambda: {
        "image": "https://data.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_image.csv",  # 35 MB, 24 fields
    })

    # Filename pattern: UNKNOWN - requires download to determine
    # Likely format: {Plate}_{Well}_s{Site}_w{Channel}.tif
    filename_regex: str = r"(?P<plate>\w+)_(?P<well>[A-P]\d{2})_s(?P<site>\d+)_w(?P<channel>\d+)\.tif"  # UNVERIFIED

    # Dataset statistics
    total_images: int = 345_600  # 69,120 FOVs × 5 channels
    total_fovs: int = 69_120
    num_plates: int = 20
    wells_per_plate: int = 384
    sites_per_well: int = 9
    channels: dict[str, str] = field(default_factory=lambda: {
        "w1": "DNA",
        "w2": "ER",
        "w3": "RNA",
        "w4": "AGP",
        "w5": "Mito",
    })

    # Image format
    format: str = "TIFF"
    bit_depth: int = 16
    pixel_size_um: float = 0.656
    magnification: str = "20X"

    # Ground truth
    has_segmentation_masks: bool = True  # BUT: only 200 images in BBBC039
    segmentation_ground_truth_dataset: str = "BBBC039"
    segmentation_ground_truth_count: int = 200
    has_moa_labels: bool = True

    # Total size
    size_bytes: int = 168_630_000_000  # ~157 GB

    checksum_strategy: str = "none"

    # Preprocessing (from Gustafsdottir 2017)
    recommended_preprocessing: dict = field(default_factory=lambda: {
        "illumination_correction": {
            "method": "per_plate_per_channel",
            "note": "ICF provided per plate per channel in dataset",
        },
        "quality_control": {
            "blur_detection": True,
            "saturation_detection": True,
            "flags_in_metadata": True,
        },
        "segmentation_order": [
            "nuclei",  # From DNA channel
            "cell_bodies",
            "cytoplasm",  # Derived
        ],
    })

    # Subsetting
    quick_subset: dict = field(default_factory=lambda: {
        "plates": ["Source4Plate5"],  # Example single plate
        "channels": ["w1"],  # DNA only
        "expected_images": 3456,  # 384 wells × 9 sites
        "size_gb": 7.8,  # Approximate
    })
```

### BBBC038 Complete Specification

```python
@dataclass(frozen=True)
class BBBC038Dataset:
    """BBBC038v1: Kaggle 2018 Data Science Bowl - nuclei segmentation."""

    id: str = "BBBC038"
    base_url: str = "https://data.broadinstitute.org/bbbc/BBBC038/"

    archives: list[str] = field(default_factory=lambda: [
        "stage1_train.zip",  # 82.9 MB
        "stage1_test.zip",  # 9.5 MB
        "stage2_test_final.zip",  # 289.7 MB
    ])

    metadata_urls: dict[str, str] = field(default_factory=lambda: {
        "metadata": "https://data.broadinstitute.org/bbbc/BBBC038/metadata.xlsx",
        "train_labels": "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train_labels.csv",
        "stage1_solution": "https://data.broadinstitute.org/bbbc/BBBC038/stage1_solution.csv",
        "stage2_solution": "https://data.broadinstitute.org/bbbc/BBBC038/stage2_solution_final.csv",
    })

    # Organization: ImageId folders, each containing image.png and masks/*.png
    format: str = "PNG"  # NOT TIFF!

    # Ground truth
    has_segmentation_masks: bool = True
    mask_format: str = "PNG binary masks"
    mask_organization: str = "one_mask_per_nucleus"
    masks_non_overlapping: bool = True

    # Dataset statistics (from NuSeT 2020)
    train_images: int = 670  # Original count
    train_images_curated: int = 543  # After manual curation (NuSeT)
    validation_images: int = 53  # NuSeT split

    # Biological diversity
    organisms: list[str] = field(default_factory=lambda: ["human", "mouse", "fly"])
    imaging_variability: str = "High - diverse stains, magnifications, conditions"

    # Size
    size_bytes: int = 401_100_000  # ~382 MB total

    checksum_strategy: str = "none"

    # Preprocessing (from NuSeT 2020 + other papers)
    recommended_preprocessing: dict = field(default_factory=lambda: {
        "mask_conversion": {
            "from": "run_length_encoding",
            "to": "binary_masks",
        },
        "normalization": {
            "method": "foreground_only",  # Mean/std from nucleus pixels only
            "improves_performance": True,
        },
        "size_filtering": {
            "min_nucleus_area": "1/5 of average",
            "removes_artifacts": True,
        },
        "cropping": {
            "requirement": "Multiple of 16 for tensor compatibility",
        },
    })

    # Metrics used in publications (for CorrectnessMetric implementation)
    standard_metrics: dict = field(default_factory=lambda: {
        "pixel_level": ["IoU", "F1", "pixel_accuracy", "RMSE"],
        "object_level": [
            "touching_nuclei_separation_rate",
            "correct_detections",
            "incorrect_detections",
            "split_errors",
            "merge_errors",
            "catastrophe_errors",
            "false_positive_rate",
            "false_negative_rate",
        ],
    })
```

## Download Strategy

Based on pybbbc implementation research:

```python
class BBBCDownloadStrategy:
    """Strategy for downloading BBBC datasets without checksums."""

    @staticmethod
    def download_multi_archive_dataset(dataset: BBBCDataset, cache_root: Path):
        """
        Download dataset with multiple archives.

        For BBBC021/022: Download only subset for quick benchmarks initially,
        full dataset on demand.
        """

        # 1. Download metadata first (small, critical)
        metadata_files = {}
        for name, url in dataset.metadata_urls.items():
            metadata_files[name] = download_with_retry(url, cache_root / "metadata")

        # 2. Download pipelines if available (for CellProfiler adapter)
        if hasattr(dataset, 'pipeline_urls'):
            for name, url in dataset.pipeline_urls.items():
                download_with_retry(url, cache_root / "pipelines")

        # 3. Download image archives (large, optional subset)
        if use_quick_subset:
            archives_to_download = dataset.quick_subset.get('archives', dataset.archives[:1])
        else:
            archives_to_download = dataset.archives

        for archive_name in archives_to_download:
            archive_url = f"{dataset.base_url}{archive_name}"
            download_with_progress(archive_url, cache_root / "archives", resume=True)

        # 4. Extract archives
        for archive_path in (cache_root / "archives").glob("*.zip"):
            extract_with_verification(archive_path, cache_root / "images")

        # 5. Validate image count (no checksums, so count files instead)
        image_count = len(list((cache_root / "images").rglob("*.tif")))
        expected_count = dataset.quick_subset['expected_images'] if use_quick_subset else dataset.total_images

        if abs(image_count - expected_count) / expected_count > 0.05:  # 5% tolerance
            raise ValueError(f"Image count mismatch: {image_count} vs {expected_count}")

        return cache_root / "images"
```

## Illumination Correction Handling

From Singh 2014 and actual CellProfiler pipelines:

```python
class IlluminationCorrectionPreprocessor:
    """
    Applies illumination correction as separate preprocessing step.

    Based on Singh et al., J. Microscopy 2014 and actual BBBC021 illum.cppipe.
    """

    def __init__(self, dataset: BBBCDataset):
        self.config = dataset.recommended_preprocessing['illumination_correction']

    def compute_icf_per_plate(self, plate_images: list[Path]) -> np.ndarray:
        """
        Compute illumination correction function for a plate.

        Algorithm from Singh 2014:
        1. Average all images in plate (same channel)
        2. Apply median filter (window=500px)
        3. Calculate robust minimum (0.02 percentile)
        4. Normalize
        """
        # Average images
        avg_image = np.mean([imread(img) for img in plate_images], axis=0)

        # Median filter smoothing
        from scipy.ndimage import median_filter
        smoothed = median_filter(avg_image, size=self.config['smoothing_sigma'])

        # Robust minimum
        robust_min = np.percentile(smoothed, self.config['robust_minimum_percentile'])

        # Avoid division by zero
        icf = np.maximum(smoothed, robust_min)

        return icf

    def apply_correction(self, image: np.ndarray, icf: np.ndarray) -> np.ndarray:
        """Divide image by ICF."""
        return image / icf
```

## Subsetting Implementation

For quick benchmarks without downloading full 41GB:

```python
@dataclass
class DatasetSubset:
    """Declarative dataset subset specification."""

    parent_dataset: BBBCDataset
    plates: list[str]  # Plate identifiers to include
    wells: Optional[list[str]] = None  # None = all wells in plates
    sites: Optional[list[int]] = None  # None = all sites
    channels: Optional[list[str]] = None  # None = all channels

    def get_expected_image_count(self) -> int:
        """Calculate expected image count for this subset."""
        # Implementation depends on dataset structure
        pass

    def matches_filename(self, filename: str) -> bool:
        """Check if filename belongs to this subset."""
        parsed = self.parent_dataset.parse_filename(filename)

        if self.wells and parsed['well'] not in self.wells:
            return False
        if self.sites and parsed['site'] not in self.sites:
            return False
        if self.channels and parsed['channel'] not in self.channels:
            return False

        return True

# Usage
quick_benchmark = DatasetSubset(
    parent_dataset=BBBC021,
    plates=["Week1_22123"],  # Single plate
    wells=["A01", "A02", "B01", "B02"],  # 4 wells
    sites=[1],  # Only site 1
    # All channels (3)
)
# Expected: 4 wells × 1 site × 3 channels = 12 images (vs 39,600 full dataset)
```

## Validation Without Checksums

Since BBBC provides no checksums:

```python
class ValidationStrategy:
    """Validation without checksums - use file counts and format checks."""

    @staticmethod
    def validate_bbbc_dataset(dataset_path: Path, dataset_spec: BBBCDataset) -> bool:
        """
        Validate BBBC dataset using:
        1. Image file count
        2. File format verification
        3. Filename pattern matching
        4. Metadata consistency
        """

        # Count images
        image_files = list(dataset_path.rglob(f"*.{dataset_spec.format.lower()}"))
        if abs(len(image_files) - dataset_spec.total_images) / dataset_spec.total_images > 0.05:
            raise ValueError(f"Image count mismatch: {len(image_files)} vs {dataset_spec.total_images}")

        # Verify file formats (sample)
        sample_size = min(100, len(image_files))
        sample = random.sample(image_files, sample_size)

        for img_path in sample:
            # Check readable
            try:
                img = imread(img_path)
                # Verify bit depth if specified
                if dataset_spec.bit_depth and img.dtype != f"uint{dataset_spec.bit_depth}":
                    raise ValueError(f"Unexpected bit depth in {img_path}")
            except Exception as e:
                raise ValueError(f"Invalid image file {img_path}: {e}")

        # Verify filename patterns
        import re
        pattern = re.compile(dataset_spec.filename_regex)
        for img_file in image_files[:100]:  # Sample
            if not pattern.match(img_file.name):
                raise ValueError(f"Filename doesn't match pattern: {img_file.name}")

        # Check metadata consistency
        if dataset_spec.metadata_urls:
            # Verify metadata references match actual images
            # (Implementation depends on metadata format)
            pass

        return True
```

## Gap Status After Research

### FILLED ✓
1. Real dataset URLs and sizes
2. Filename patterns (BBBC021, BBBC038)
3. Illumination correction parameters
4. Preprocessing pipelines
5. Ground truth availability details
6. Evaluation metrics from publications
7. Subsetting strategy

### STILL BLOCKED ✗
1. BBBC022 filename pattern (need to download to reverse-engineer)
2. Checksums (not provided by Broad, must skip or compute)
3. Complete file manifests (too large to list, use counts instead)

### WORKAROUNDS DEFINED ✓
1. Checksum: Skip verification, use file counts + format checks
2. Manifests: Validate by count + pattern matching, not explicit lists
3. BBBC022 pattern: Download single plate subset to reverse-engineer, or skip BBBC022 initially
