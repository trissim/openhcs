# Preset Pipeline Spec Authority Refactor - 2026-05-18

## Advisor Evidence

Full-repo scan flagged cross-module spec axis duplication in preset pipelines:

- `processing/presets/pipelines/10x_mfd_crop_analyze.py`
- `processing/presets/pipelines/10x_mfd_crop_analyze_dapi-fitc-cy5.py`
- `processing/presets/pipelines/10x_mfd_stitch_ashlar_cpu.py`
- `processing/presets/pipelines/10x_mfd_stitch_gpu.py`

Advisor summary: multiple files encode the same step families and `name -> func`
pairs with small variant differences.

## Current Problem

Preset pipeline files are editable generated-looking Python modules containing
large literal `FunctionStep` declarations. Related variants copy most of the
same structure:

- same imports;
- same step names;
- same compartment functions;
- same crop sizes;
- small differences in channel-specific analysis or CPU/GPU backend choice.

This creates stale preset risk and makes GUI/editor integration harder.

## Target Shape

Introduce a typed preset spec layer:

```python
@dataclass(frozen=True, slots=True)
class PresetPipelineSpec:
    name: str
    imports: tuple[PresetImportSpec, ...]
    steps: tuple[PresetStepSpec, ...]

@dataclass(frozen=True, slots=True)
class PresetVariantOverlay:
    base_name: str
    replacements: tuple[PresetStepReplacement, ...]
```

Then materialize normal `FunctionStep` objects from specs.

## Phase 1: Characterize Preset Consumers

Find how `processing/presets/pipelines/*.py` files are loaded:

- direct Python import;
- file picker;
- GUI editor;
- text serialization;
- tests.

Do not change file format until the loading path is known.

## Phase 2: Add Spec Builder Without Removing Files

Create a new authority module, for example:

- `openhcs/processing/presets/pipeline_specs.py`
- `openhcs/processing/presets/mfd_specs.py`

Implement builders that produce the same `pipeline_steps` list for at least:

- `10x_mfd_crop_analyze`
- `10x_mfd_crop_analyze_dapi_fitc_cy5`

Keep existing files as thin materialization wrappers initially.

## Phase 3: Variant Overlay

Represent differences explicitly:

- channel 4 analysis behavior;
- CPU/GPU backend choice;
- stitch backend variants;
- template path constants.

Do not hide variant differences inside arbitrary lambdas. Use named overlay
records.

## Phase 4: Regeneration / Round-Trip

If users edit these files directly, add a generator command or editor path that
can write the current Python form from the spec authority.

If users only import them, replace files with wrappers:

```python
from openhcs.processing.presets.mfd_specs import build_10x_mfd_crop_analyze
pipeline_steps = build_10x_mfd_crop_analyze()
```

## Phase 5: Tests

Add tests that compare old and new pipeline materialization:

- same number of steps;
- same step names;
- same function identities;
- same parameter dictionaries for unchanged variants;
- expected variant differences are explicit.

## Risks

- These files may be edited by users and comments may matter. Preserve a
  human-readable generated output path.
- Absolute template paths are currently embedded. Decide whether they remain
  literal defaults or become environment/config parameters.
- Function identity comparisons must handle tuple/list/dict function patterns.

## Verification Gates

```bash
.venv/bin/python -m pytest tests/unit -q
python -m nominal_refactor_advisor openhcs/processing/presets/pipelines
```

## Completion Criteria

- Shared preset structure has one authoritative spec.
- Variant differences are declared as overlays.
- Existing import path compatibility remains intact.
- Cross-module preset spec-axis findings are removed or reduced to wrappers.

## Implementation Notes

Implemented in `openhcs/processing/presets/mfd_specs.py`.

- `MfdPresetKey` names the import-compatible preset variants.
- `MfdPresetDefinition` records variant deltas: crop/analyze channel-4 behavior
  and CPU/GPU stitch backend choice.
- `PresetStepBinding`, `PresetStepSpec`, and `PresetStepTemplate` centralize
  step naming, source binding, variable components, and fresh `FunctionStep`
  materialization.
- `MfdPresetMaterializer` owns the family split between crop/analyze and stitch
  presets through `AutoRegisterMeta`; the four preset files are now wrappers
  that expose `pipeline_steps = build_mfd_preset(...)`.

## Verification Record

Focused gates passed on 2026-05-18:

```bash
.venv/bin/python -m pytest tests/unit/test_mfd_preset_specs.py -q
timeout 120 .venv/bin/python -m nominal_refactor_advisor \
  openhcs/processing/presets/mfd_specs.py \
  openhcs/processing/presets/pipelines/10x_mfd_crop_analyze.py \
  openhcs/processing/presets/pipelines/10x_mfd_crop_analyze_dapi-fitc-cy5.py \
  openhcs/processing/presets/pipelines/10x_mfd_stitch_ashlar_cpu.py \
  openhcs/processing/presets/pipelines/10x_mfd_stitch_gpu.py
```

Results:

- `4 passed`
- `No refactoring findings`

Broader checkpoint:

- `git diff --check` passed.
- `.venv/bin/python -m pytest tests/unit -q`: `1522 passed, 10 warnings`.
- `timeout 180 .venv/bin/python -m nominal_refactor_advisor openhcs`:
  1,142 findings, 60.135s.
