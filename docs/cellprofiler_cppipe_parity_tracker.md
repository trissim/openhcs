# Historical CellProfiler `.cppipe` Parity Snapshot

> **Frozen historical record — 2026-05-03.** This file preserves the focused
> parity investigation that preceded the portable Official30 acceptance route.
> It is not the current parity authority. Every `/tmp` path below was transient
> local evidence and may no longer exist; none is a durable run receipt. Current
> acceptance is defined by `benchmark/manifests/official30_portable_axis1.json`,
> `tests/integration/test_cellprofiler_official30_zmq.py`, and the receipt
> requirements in
> `docs/source/architecture/measurement_equivalence_system.rst`.

A pipeline was marked green in this snapshot only when the then-current semantic
equivalence run reported `differences=0` against its native CellProfiler
reference.

## Historical Status Snapshot

Last updated: 2026-05-03 23:51:11 EDT

| Pipeline | Dataset | Status | Proven at | Differences | Evidence |
| --- | --- | --- | --- | ---: | --- |
| ExampleColocalization | ExampleColocalization | Green | 2026-05-03 | 0 | `/tmp/openhcs_cppipe_parity_focus_colocalization_20260503_colocalized_tolerance` |
| ExampleCometAssay | ExampleCometAssay | Green | 2026-05-03 18:55:49 EDT | 0 | `/tmp/openhcs_cppipe_parity_comet_diag_binary_20260503_185527` |
| ExampleFly | ExampleFly | Green | 2026-05-03 | 0 | `/tmp/openhcs_cppipe_parity_focus_fly_20260503_slope_reverse_field` |
| ExampleFlyURL | ExampleFlyURL | Green | 2026-05-03 18:02:08 EDT | 0 | `/tmp/openhcs_cppipe_parity_full_20260503_tight_basin_restore` |
| ExampleHuman | ExampleHuman | Green | 2026-05-03 21:21:49 EDT | 0 | `/tmp/openhcs_cppipe_parity_focus_human_threshold_constant_20260503_211934` |
| ExampleIlluminationCorrection_Example1_AllMethod | ExampleIlluminationCorrection | Green | 2026-05-03 15:48:37 EDT | 0 | `/tmp/openhcs_cppipe_parity_focus_illum1_all_20260503` |
| ExampleIlluminationCorrection_Example1_EachMethod | ExampleIlluminationCorrection | Green | 2026-05-03 15:49:41 EDT | 0 | `/tmp/openhcs_cppipe_parity_focus_illum1_each_20260503` |
| ExampleIlluminationCorrection_Example2 | ExampleIlluminationCorrection | Green | 2026-05-03 15:58:34 EDT | 0 | `/tmp/openhcs_cppipe_parity_focus_illum2_rankmedian_numba2_20260503` |
| ExampleIlluminationCorrection_Example3 | ExampleIlluminationCorrection | Green | 2026-05-03 15:53:18 EDT | 0 | `/tmp/openhcs_cppipe_parity_focus_illum3_20260503` |
| ExampleImagingFlowCytometryObjectsInGrid | ExampleImagingFlowCytometryObjectsInGrid | Green | 2026-05-03 20:50:21 EDT | 0 | `/tmp/openhcs_cppipe_parity_focus_objects_in_grid_smooth_constant_20260503_204822` |
| ExampleNeighbors | ExampleNeighbors | Green | 2026-05-03 | 0 | `/tmp/openhcs_cppipe_parity_focus_neighbors_20260503` |
| ExamplePercentPositive | ExamplePercentPositive | Green | 2026-05-03 16:41:08 EDT | 0 | `/tmp/openhcs_cppipe_parity_focus_percent_positive_suppression_regression_20260503` |
| ExampleSpeckles | ExampleSpeckles | Green | 2026-05-03 21:05:49 EDT | 0 | `/tmp/openhcs_cppipe_parity_focus_speckles_declump_strel_20260503_210524` |
| ExampleTumor | ExampleTumor | Green | 2026-05-03 | 0 | `/tmp/openhcs_cppipe_parity_focus_tumor_20260503_after_thresholds` |
| ExampleTrackObjects | ExampleTrackObjects | Green | 2026-05-03 18:01:25 EDT | 0 | `/tmp/openhcs_cppipe_parity_focus_trackobjects_tracking_numba_tight_basin_20260503` |
| ExampleUntangleAndStraightenWorms | ExampleStraightenWorms | Green | 2026-05-03 16:08:02 EDT | 0 | `/tmp/openhcs_cppipe_parity_focus_untangle_straighten_20260503` |
| ExampleUntangleWorms | ExampleUntangleWorms | Green | 2026-04-30 | 0 | `/tmp/openhcs_cppipe_parity_batch_20260430` |
| ExampleUntangleWormsBrightField | ExampleUntangleWormsBrightField | Green | 2026-04-30 | 0 | `/tmp/openhcs_cppipe_parity_batch_20260430` |
| ExampleWoundHealing | ExampleWoundHealing | Green | 2026-05-03 | 0 | `/tmp/openhcs_cppipe_parity_focus_wound_20260503_after_thresholds` |
| ExampleYeastColonies | ExampleYeastColonies | Green | 2026-05-03 23:51:11 EDT | 0 | `/tmp/openhcs_cppipe_parity_yeast_numba_orientation_dot_20260503_235046` |
| ExampleYeastPatches | ExampleYeastPatches | Green | 2026-05-03 19:25:28 EDT | 0 | `/tmp/openhcs_cppipe_parity_focus_yeast_patches_current_20260503_192503` |
| ExampleVitra | ExampleVitraImages | Green | 2026-05-03 16:52:36 EDT | 0 | `/tmp/openhcs_cppipe_parity_focus_vitra_seed_remap_20260503` |

## Historical Full-Refresh Notes

- Then-latest full refresh evidence:
  `/tmp/openhcs_cppipe_parity_full_refresh_20260503_185928`
- Pass in that full refresh: `ExampleColocalization`,
  `ExampleCometAssay`, `ExampleFly`, `ExampleFlyURL`,
  `ExampleIlluminationCorrection_Example1_AllMethod`,
  `ExampleIlluminationCorrection_Example1_EachMethod`,
  `ExamplePercentPositive`, `ExampleTrackObjects`, `ExampleTumor`,
  `ExampleUntangleAndStraightenWorms`, `ExampleUntangleWorms`,
  `ExampleUntangleWormsBrightField`, `ExampleVitra`,
  `ExampleWoundHealing`.
- Quota-contaminated failures from that refresh were rerun after `/tmp`
  cleanup:
  `ExampleIlluminationCorrection_Example2` green at
  `/tmp/openhcs_cppipe_parity_quota_rerun_20260503_1912/ExampleIlluminationCorrection_Example2`;
  `ExampleIlluminationCorrection_Example3` green at
  `/tmp/openhcs_cppipe_parity_quota_rerun_20260503_1912/ExampleIlluminationCorrection_Example3`;
  `ExampleNeighbors` green after SaveImages-aware value-only pruning at
  `/tmp/openhcs_cppipe_parity_neighbors_prune_fix_20260503_1918`.
- Remaining non-green cases in the final focused reruns of this snapshot: none.

## Historical Focus Notes

- `ExampleYeastPatches` was confirmed green in this snapshot at
  `/tmp/openhcs_cppipe_parity_focus_yeast_patches_current_20260503_192503`
  with `differences=0`.
- `ExampleYeastColonies` previously failed in the focused rerun
  `/tmp/openhcs_cppipe_parity_focus_yeast_colonies_current_20260503_192136`
  with `differences=96`. The first divergence is
  `FinalThreshold_Colonies`/`OrigThreshold_Colonies`
  (`0.11586319655179977` candidate vs `0.11453950577557473` saved native
  reference), cascading to object count and downstream measurements.
- Native CellProfiler is available via
  `/home/ts/code/projects/openhcs/.venv-cellprofiler39/bin/cellprofiler`; use
  `PATH=/home/ts/code/projects/openhcs/.venv-cellprofiler39/bin:$PATH` when
  regenerating references from the OpenHCS benchmark venv.
- `ExampleYeastColonies` was rerun with a fresh native reference at
  `/tmp/openhcs_cppipe_parity_focus_yeast_colonies_fresh_native_20260503_1940`
  and still failed with `differences=96`, proving the mismatch was not stale
  reference drift. It was resolved by the automatic illumination smoothing-size
  fix documented below.
- `ExampleImagingFlowCytometryObjectsInGrid` was confirmed green in this snapshot at
  `/tmp/openhcs_cppipe_parity_focus_objects_in_grid_smooth_constant_20260503_204822`
  with `differences=0`. The last semantic blocker was upstream of
  `IdentifyPrimaryObjects`: CellProfiler `Smooth` Gaussian mode uses
  constant-zero boundary handling through
  `centrosome.filter.smooth_with_function_and_mask`, while the OpenHCS
  Gaussian smoothing backend was using reflect boundaries. After switching the
  explicit smoothing backends to CP's constant-zero semantics, direct
  `SmoothedBF`/`EdgedImage`/`MorphBf` comparisons were within float roundoff
  (`~1.9e-09` max) and the full semantic comparison completed with
  `differences=0`.
- `ExampleSpeckles` was confirmed green in this snapshot at
  `/tmp/openhcs_cppipe_parity_focus_speckles_declump_strel_20260503_210524`
  with `differences=0`. The final mismatch was in
  `IdentifyPrimaryObjects` declumping maxima geometry: native CellProfiler
  uses `centrosome.cpmorphology.strel_disk(max(1, maxima_suppression_size -
  0.5))`, whose extent is `int(radius)`. The OpenHCS morphology backend had
  an unbacked intensity/min-diameter expansion and a context-specific sigma
  divisor. Removing those special cases made `raw_maxima`,
  `shrunk_maxima`, and separated labels byte-identical to native CP for the
  Speckles `h2ax` IPO module, and the focused semantic comparison completed
  with `differences=0`.
- `ExampleHuman` was confirmed green in this snapshot at
  `/tmp/openhcs_cppipe_parity_focus_human_threshold_constant_20260503_211934`
  with `differences=0` and uncached OpenHCS runtime `7.142s`. The final
  mismatch was a 13-pixel `IdentifyPrimaryObjects` binary-mask divergence in
  the `Nuclei` module. Native CellProfiler applies
  `threshold_smoothing_scale` with
  `scipy.ndimage.gaussian_filter(..., mode="constant", cval=0)` through
  `centrosome.smooth.smooth_with_function_and_mask`; the OpenHCS threshold
  application path was using reflect boundaries. Switching threshold
  application smoothing to CP's constant-zero boundary semantics made the
  direct module-5 `binary_before`, initial labels, separated labels,
  border-filtered labels, and relabeled output byte-identical to native CP,
  removing the object/measurement cascade.
- `ExampleYeastColonies` was confirmed green in this snapshot at
  `/tmp/openhcs_cppipe_parity_yeast_numba_orientation_dot_20260503_235046`
  with `differences=0`. The cached-native parity run proved the shared
  Numba region-property backend matches native CP aggregate orientation after
  replacing the temporary current-skimage orientation leaf with a skimage-0.18
  crop-moment reduction order. Current hot OpenHCS steps in that run were
  `CorrectIlluminationCalculate` step 1 (`2.540s` cold-ish),
  `Align` (`1.697s`), `MeasureObjectSizeShape` (`1.918s`),
  `IdentifyPrimaryObjects` (`0.759s`), `MeasureObjectIntensity` (`0.391s`),
  and `ClassifyObjects` (`0.347s`). The generic runtime measurement lookup
  optimization was separately proven in
  `/tmp/openhcs_cppipe_runtime_yeast_measurement_lookup_fast_20260503_231149`,
  where Yeast runtime-only execution completed in `9.819s` and
  `ClassifyObjects` was `0.340s` total.
  The earlier mismatch was upstream of
  `IdentifyPrimaryObjects`: native CP 3.9 `CorrectIlluminationCalculate`
  automatic smoothing size is `min(30, max(image_shape) / 40.0)`, while the
  OpenHCS strategy was using `object_width * 2.0`. For the 1116x1112 Yeast
  image this changed the Gaussian filter size from `20` to `27.9`, made the
  illumination/corrected-red image match closely enough for the masked
  log-Multi-Otsu threshold to match, and removed the object/measurement
  cascade.
