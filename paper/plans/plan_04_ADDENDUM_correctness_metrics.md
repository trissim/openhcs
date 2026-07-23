# Plan 04 ADDENDUM: Correctness Metrics from Publications

## Real Evaluation Metrics Used in BBBC Benchmarks

### From NuSeT 2020 (BBBC038 Benchmark)

```python
class CorrectnessMetricBBBC038:
    """
    Correctness evaluation for nuclei segmentation.

    Based on NuSeT (Samacoits et al., PLoS Comput Biol 2020) and
    Mask R-CNN vs U-Net comparisons.
    """

    def __init__(self, ground_truth_masks_path: Path):
        self.gt_path = ground_truth_masks_path

    def evaluate(self, predicted_masks_path: Path) -> dict[str, float]:
        """
        Comprehensive evaluation with pixel-level and object-level metrics.

        Returns dict with all metrics for publication-quality comparison.
        """

        results = {}

        # Pixel-level metrics
        results.update(self._compute_pixel_metrics(predicted_masks_path))

        # Object-level metrics
        results.update(self._compute_object_metrics(predicted_masks_path))

        return results

    def _compute_pixel_metrics(self, pred_path: Path) -> dict:
        """
        Pixel-level metrics from NuSeT 2020.

        Metrics:
        - Mean IoU (Intersection over Union)
        - F1 score
        - Pixel accuracy
        - RMSE (Root Mean Square Error)
        """

        gt_masks = self._load_masks(self.gt_path)
        pred_masks = self._load_masks(pred_path)

        # Flatten to binary pixel classifications
        gt_binary = (gt_masks > 0).astype(int)
        pred_binary = (pred_masks > 0).astype(int)

        # IoU
        intersection = np.logical_and(gt_binary, pred_binary).sum()
        union = np.logical_or(gt_binary, pred_binary).sum()
        iou = intersection / union if union > 0 else 0.0

        # F1 score
        tp = intersection
        fp = (pred_binary & ~gt_binary).sum()
        fn = (gt_binary & ~pred_binary).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Pixel accuracy
        correct_pixels = (gt_binary == pred_binary).sum()
        total_pixels = gt_binary.size
        pixel_accuracy = correct_pixels / total_pixels

        # RMSE
        rmse = np.sqrt(np.mean((gt_binary - pred_binary) ** 2))

        return {
            'pixel_iou': iou,
            'pixel_f1': f1,
            'pixel_accuracy': pixel_accuracy,
            'pixel_rmse': rmse,
            'precision': precision,
            'recall': recall,
        }

    def _compute_object_metrics(self, pred_path: Path) -> dict:
        """
        Object-level metrics from NuSeT 2020.

        Metrics:
        - Touching nuclei separation rate
        - Correct detections
        - Incorrect detections
        - Split errors (1 GT → N predicted)
        - Merge errors (N GT → 1 predicted)
        - Catastrophe errors (major failures)
        - False positive rate
        - False negative rate
        """

        gt_labels = self._load_labeled_masks(self.gt_path)
        pred_labels = self._load_labeled_masks(pred_path)

        # Match predicted objects to ground truth (IoU > 0.5 threshold)
        matches, splits, merges, fps, fns = self._match_objects(
            gt_labels, pred_labels, iou_threshold=0.5
        )

        num_gt = len(np.unique(gt_labels)) - 1  # Exclude background
        num_pred = len(np.unique(pred_labels)) - 1

        # Compute rates
        correct_detections = len(matches)
        split_errors = len(splits)
        merge_errors = len(merges)
        false_positives = len(fps)
        false_negatives = len(fns)

        # Touching nuclei separation (if touching pairs metadata available)
        # This requires additional annotation - skip if not available
        separation_rate = self._compute_separation_rate(gt_labels, pred_labels)

        return {
            'object_correct_detections': correct_detections,
            'object_split_errors': split_errors,
            'object_merge_errors': merge_errors,
            'object_false_positives': false_positives,
            'object_false_negatives': false_negatives,
            'object_fp_rate': false_positives / num_pred if num_pred > 0 else 0.0,
            'object_fn_rate': false_negatives / num_gt if num_gt > 0 else 0.0,
            'object_touching_separation_rate': separation_rate,
        }

    def _match_objects(self, gt_labels, pred_labels, iou_threshold=0.5):
        """
        Match predicted objects to ground truth objects using IoU.

        Returns:
        - matches: List of (gt_id, pred_id) pairs
        - splits: List of gt_ids that split into multiple predictions
        - merges: List of pred_ids that merged multiple GTs
        - false_positives: List of pred_ids with no GT match
        - false_negatives: List of gt_ids with no pred match
        """

        gt_ids = np.unique(gt_labels)[1:]  # Exclude background
        pred_ids = np.unique(pred_labels)[1:]

        # Build IoU matrix
        iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))

        for i, gt_id in enumerate(gt_ids):
            gt_mask = (gt_labels == gt_id)
            for j, pred_id in enumerate(pred_ids):
                pred_mask = (pred_labels == pred_id)
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()
                iou_matrix[i, j] = intersection / union if union > 0 else 0.0

        # Find matches (IoU > threshold)
        matches = []
        splits = []
        merges = []

        gt_matched = set()
        pred_matched = set()

        # First pass: 1-to-1 matches
        for i, gt_id in enumerate(gt_ids):
            for j, pred_id in enumerate(pred_ids):
                if iou_matrix[i, j] > iou_threshold:
                    # Check if best match
                    if iou_matrix[i, j] == iou_matrix[i, :].max():
                        matches.append((gt_id, pred_id))
                        gt_matched.add(gt_id)
                        pred_matched.add(pred_id)
                        break

        # Second pass: detect splits (1 GT → N pred)
        for i, gt_id in enumerate(gt_ids):
            if gt_id in gt_matched:
                continue
            pred_matches = [pred_ids[j] for j in range(len(pred_ids))
                           if iou_matrix[i, j] > iou_threshold]
            if len(pred_matches) > 1:
                splits.append(gt_id)
                gt_matched.add(gt_id)
                pred_matched.update(pred_matches)

        # Third pass: detect merges (N GT → 1 pred)
        for j, pred_id in enumerate(pred_ids):
            if pred_id in pred_matched:
                continue
            gt_matches = [gt_ids[i] for i in range(len(gt_ids))
                         if iou_matrix[i, j] > iou_threshold]
            if len(gt_matches) > 1:
                merges.append(pred_id)
                pred_matched.add(pred_id)
                gt_matched.update(gt_matches)

        # FPs and FNs
        false_positives = [pid for pid in pred_ids if pid not in pred_matched]
        false_negatives = [gid for gid in gt_ids if gid not in gt_matched]

        return matches, splits, merges, false_positives, false_negatives

    def _compute_separation_rate(self, gt_labels, pred_labels):
        """
        Compute touching nuclei separation rate.

        Requires detecting which GT nuclei are touching, then checking
        if predictions separated them correctly.
        """

        # Find touching pairs in GT
        from scipy.ndimage import binary_dilation
        gt_ids = np.unique(gt_labels)[1:]

        touching_pairs = []
        for gt_id in gt_ids:
            mask = (gt_labels == gt_id)
            dilated = binary_dilation(mask, iterations=1)
            # Find neighbors
            neighbors = np.unique(gt_labels[dilated & (gt_labels != gt_id) & (gt_labels > 0)])
            for neighbor_id in neighbors:
                if gt_id < neighbor_id:  # Avoid duplicates
                    touching_pairs.append((gt_id, neighbor_id))

        if not touching_pairs:
            return 1.0  # No touching nuclei

        # Check how many were separated in predictions
        separated = 0
        for gt_id1, gt_id2 in touching_pairs:
            # Find predicted objects overlapping these GTs
            mask1 = (gt_labels == gt_id1)
            mask2 = (gt_labels == gt_id2)

            pred_ids1 = np.unique(pred_labels[mask1])[1:]
            pred_ids2 = np.unique(pred_labels[mask2])[1:]

            # If no overlap in predicted IDs, they were separated
            if not set(pred_ids1).intersection(set(pred_ids2)):
                separated += 1

        return separated / len(touching_pairs)

    def _load_masks(self, path: Path) -> np.ndarray:
        """Load binary masks from directory."""
        # BBBC038 specific: PNG files in masks/ subdirectory
        mask_files = sorted(path.glob("*.png"))
        masks = [imread(f) for f in mask_files]
        return np.stack(masks)

    def _load_labeled_masks(self, path: Path) -> np.ndarray:
        """Load instance segmentation masks (each nucleus has unique ID)."""
        from skimage.measure import label

        binary_masks = self._load_masks(path)
        # Convert to labeled instances
        labeled = label(binary_masks > 0)
        return labeled
```

### Tool Comparison Metrics (BBBC021)

From "Evaluation of cell segmentation methods without reference segmentations" (MBoC 2023):

```python
class ToolComparisonMetrics:
    """
    Compare tools WITHOUT ground truth segmentation.

    Based on Cimini et al., MBoC 2023 - evaluates consistency across tools
    rather than absolute correctness.
    """

    def __init__(self, reference_tool: str = "CellProfiler"):
        """
        Args:
            reference_tool: Which tool to use as baseline for comparison.
                           Default: CellProfiler (most established)
        """
        self.reference_tool = reference_tool

    def compute_consistency_score(
        self,
        tool_results: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """
        Compute consistency between tools.

        Args:
            tool_results: Dict mapping tool_name → results dict
                         Results should include:
                         - num_objects: int
                         - mean_intensity: float (per object)
                         - measurements: pd.DataFrame

        Returns:
            Dict of consistency metrics
        """

        ref_results = tool_results[self.reference_tool]

        scores = {}

        for tool_name, tool_result in tool_results.items():
            if tool_name == self.reference_tool:
                scores[tool_name] = 1.0  # Perfect self-consistency
                continue

            # Object count agreement
            count_agreement = min(
                tool_result['num_objects'],
                ref_results['num_objects']
            ) / max(
                tool_result['num_objects'],
                ref_results['num_objects']
            )

            # Feature correlation (for shared measurements)
            if 'measurements' in tool_result and 'measurements' in ref_results:
                # Compare distributions of features
                feature_corr = self._compute_feature_correlation(
                    ref_results['measurements'],
                    tool_result['measurements']
                )
            else:
                feature_corr = count_agreement  # Fallback

            # Combined score
            scores[tool_name] = (count_agreement + feature_corr) / 2

        return scores

    def _compute_feature_correlation(
        self,
        ref_features: pd.DataFrame,
        tool_features: pd.DataFrame
    ) -> float:
        """
        Compute correlation between feature distributions.

        Uses Earth Mover's Distance for robust comparison.
        """
        from scipy.stats import wasserstein_distance

        # Compare distributions of common features
        common_features = set(ref_features.columns).intersection(tool_features.columns)

        if not common_features:
            return 0.0

        correlations = []
        for feature in common_features:
            # Wasserstein distance (lower = more similar)
            dist = wasserstein_distance(
                ref_features[feature],
                tool_features[feature]
            )
            # Normalize to [0, 1] similarity score
            # (assumes features are normalized to similar scales)
            similarity = 1.0 / (1.0 + dist)
            correlations.append(similarity)

        return np.mean(correlations)
```

### Integration with Benchmark System

```python
# In benchmark/metrics/correctness.py

class CorrectnessMetric:
    """
    Unified correctness evaluation supporting multiple strategies.
    """

    def __init__(
        self,
        ground_truth_path: Optional[Path] = None,
        strategy: str = "auto"  # "ground_truth", "tool_comparison", "auto"
    ):
        self.gt_path = ground_truth_path
        self.strategy = strategy

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_result(self, tool_results: dict[str, Any]) -> dict[str, float]:
        """
        Compute correctness score(s).

        Args:
            tool_results: Results from tool execution, including:
                         - output_path: Path to segmentation masks
                         - measurements: Optional DataFrame of measurements

        Returns:
            Dict of correctness metrics
        """

        # Auto-select strategy
        if self.strategy == "auto":
            if self.gt_path and self.gt_path.exists():
                strategy = "ground_truth"
            else:
                strategy = "tool_comparison"
        else:
            strategy = self.strategy

        # Apply appropriate evaluator
        if strategy == "ground_truth":
            evaluator = CorrectnessMetricBBBC038(self.gt_path)
            return evaluator.evaluate(tool_results['output_path'])

        elif strategy == "tool_comparison":
            # Requires results from multiple tools
            if len(tool_results) < 2:
                return {'consistency_score': -1.0}  # Not enough tools

            evaluator = ToolComparisonMetrics(reference_tool="CellProfiler")
            return evaluator.compute_consistency_score(tool_results)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")
```

### Tolerance Envelopes (from Plan 03 Revisions)

```python
class CorrectnessTolerances:
    """
    Tolerance envelopes for pipeline equivalence.

    Based on typical variance in BBBC benchmarking papers.
    """

    NUCLEI_SEGMENTATION = {
        'object_count_delta_pct': 2.0,  # ±2% object count
        'iou_min': 0.90,  # IoU ≥ 0.9
        'feature_pearson_r_min': 0.98,  # r ≥ 0.98 for measurements
    }

    CELL_PAINTING = {
        'object_count_delta_pct': 5.0,  # ±5% (more complex)
        'iou_min': 0.85,  # Slightly relaxed
        'feature_pearson_r_min': 0.95,
    }

    @staticmethod
    def check_equivalence(
        ref_results: dict,
        tool_results: dict,
        pipeline_type: str = "nuclei_segmentation"
    ) -> bool:
        """
        Check if tool results are equivalent within tolerances.
        """

        tolerances = getattr(CorrectnessTolerances, pipeline_type.upper())

        # Object count check
        count_delta_pct = abs(
            tool_results['num_objects'] - ref_results['num_objects']
        ) / ref_results['num_objects'] * 100

        if count_delta_pct > tolerances['object_count_delta_pct']:
            return False

        # IoU check (if masks available)
        if 'iou' in tool_results and tool_results['iou'] < tolerances['iou_min']:
            return False

        # Feature correlation check
        if 'feature_correlation' in tool_results:
            if tool_results['feature_correlation'] < tolerances['feature_pearson_r_min']:
                return False

        return True
```

## Summary

### Available Ground Truth

| Dataset | Type | Coverage | Metrics |
|---------|------|----------|---------|
| BBBC021 | MoA labels | 103 compounds | Classification accuracy |
| BBBC022 | Segmentation masks | 200 images (via BBBC039) | IoU, F1, object-level |
| BBBC038 | Segmentation masks | All training images | Full pixel + object metrics |

### Recommendation

**Use BBBC038 for segmentation correctness** (full ground truth)
**Use BBBC021/022 for tool consistency comparison** (no/limited ground truth)

This matches how publications actually benchmark on these datasets.
