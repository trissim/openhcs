# CellProfiler Runtime Declaration Semantics Inventory

Date: 2026-06-26

Purpose: identify module-specific semantics currently encoded in CellProfiler runtime/module-execution/adapter paths so those facts can move onto AutoRegisterMeta-backed `CellProfilerModule` declarations and inherited nominal policy families.

## Architectural Target

The CellProfiler runtime should be a generic executor. It should not decide that a specific module has a special object-label ABI, source-provenance rule, measurement-row shape, execution cardinality, or main-flow replacement rule by matching module names or maintaining parallel policy registries.

Replacement direction:

- `CellProfilerModule` declarations own module-specific facts: execution mode, primary domain, object input roles, special input roles, measurement cardinality, row materialization, output provenance, and main-flow behavior;
- reusable shared behavior should be factored into declaration mixins or parent classes, then inherited through MRO;
- runtime code should query the declaration for facts and run one generic lowering/execution path;
- temporary policy classes can remain as adapters during migration, but they should delegate to declaration facts rather than duplicate semantics.

## Live Code Recheck, 2026-06-26

The current runtime split is uneven:

- `runtime/adapter.py` does not currently appear to dispatch on CellProfiler
  module name. The surfaces below are generic runtime/source-provenance mechanics
  that consume compiled artifact input/output plans. They still matter because
  declaration-owned object-label provenance requirements must flow into those
  plans, but the adapter itself should not become a module policy registry.
- `runtime/module_execution.py` is still the main runtime mirror. The
  `CellProfilerModuleRuntimePlan.build(...)` path selects multiple
  `*.for_module(canonical_module_name)` policy families for special inputs,
  object inputs, execution mode, main-flow replacement, measurement recording,
  and dual-scope measurement. That centralized selection is cleaner than
  scattered branches, but the module-specific policy leaves and `module_name`
  class attributes still live in runtime.
- Good migration shape is to keep the runtime request/value/projection mechanics
  where they are, move or declare the module-specific selection facts on
  `CellProfilerModule` declarations or inherited declaration mixins, and leave
  the runtime policy classes as executable mechanics only where they still pay
  rent.

## Raw Semantics Inventory

### Adapter And Runtime Adapter Surfaces

| File:line | Branch / policy | Module-specific semantic encoded there | Declaration-owned fact to query |
|---|---|---|---|
| `openhcs/interop/cellprofiler/runtime/adapter.py:610` | `requires_declared_source_image_domain` | Object labels inherit or require source coordinate metadata when `source_image_name` is a declared image input. | Object-label output requires source-image-domain provenance from a declared image input. |
| `openhcs/interop/cellprofiler/runtime/adapter.py:728` | `resolve_source_objects` | Source-bound object labels are forced through image-payload semantics with `ImageType=Objects` before becoming `ObjectLabelSet`. | Source object inputs are object-label payloads with CellProfiler `Objects` image-type semantics. |
| `openhcs/interop/cellprofiler/runtime/adapter.py:872` | `add_objects` source-provenance branch | Raw label arrays opportunistically inherit provenance/spatial domain from `source_image_name`; declared source images require complete source coordinate metadata. | Object-label outputs may declare source-image provenance requirement and coordinate completeness. |
| `openhcs/interop/cellprofiler/runtime/main_flow.py:74` | `CorrectIlluminationApplyMainFlowReplacementPolicy` | Any corrected image output replaces main flow. | Image output owns downstream main flow. |
| `openhcs/interop/cellprofiler/runtime/main_flow.py:88` | `CorrectIlluminationCalculateMainFlowReplacementPolicy` | Illumination function image outputs are recorded but never replace main flow. | Image output does not replace downstream main flow. |
| `openhcs/interop/cellprofiler/runtime/output_contexts.py:119` | `CorrectIlluminationApplyImageOutputSourcePayloadPolicy` | Corrected outputs inherit provenance from matching `Orig*` original image input or positional original input. | Output provenance maps corrected output to original image input. |
| `openhcs/interop/cellprofiler/runtime/output_contexts.py:218` | `CorrectIlluminationApplyImageOutputValuePolicy` | Duplicate grouped-plane corrected outputs collapse to one source plane. | Output value collapse policy for duplicate source-plane stacks. |

### Object Input Policies

| File:line | Branch / policy | Module-specific semantic encoded there | Declaration-owned fact to query |
|---|---|---|---|
| `openhcs/interop/cellprofiler/runtime/object_input_policies.py:161` | `IdentifySecondaryObjectsInputPolicy` | Single object input binds as `primary_labels`. | Object input role: primary labels kwarg. |
| `openhcs/interop/cellprofiler/runtime/object_input_policies.py:168` | `IdentifyTertiaryObjectInputPolicy` | Two object inputs are reordered into smaller/primary and larger/secondary label kwargs. | Object input roles and ordering: smaller/larger labels. |
| `openhcs/interop/cellprofiler/runtime/object_input_policies.py:201` | `CropInputPolicy` | Object input binds as `cropping_labels`. | Object input role: crop mask labels. |
| `openhcs/interop/cellprofiler/runtime/object_input_policies.py:208` | `MeasureObject*InputPolicy` family | Object measurement modules bind their single label input as `labels`. | Object input role: measurement labels. |
| `openhcs/interop/cellprofiler/runtime/object_input_policies.py:252` | `MeasureObjectNeighborsInputPolicy` | One or two object inputs become measured/neighbor labels plus same-object flag and small-removed variants. | Neighbor-measurement object roles: measured, neighbor, same-object behavior, variant needs. |

### Execution Mode And Primary Domain Policies

| File:line | Branch / policy | Module-specific semantic encoded there | Declaration-owned fact to query |
|---|---|---|---|
| `openhcs/interop/cellprofiler/runtime/module_execution.py:2276` | `CorrectIlluminationCalculateExecutionModePolicy` | `calculation_scope=all images` forces full-stack execution. | Invocation execution mode depends on illumination calculation scope. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:2297` | `ColorToGrayExecutionModePolicy` | Always consumes channel composite as full stack. | Module consumes composed channel payload, not independent slices. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:2314` | `DefineGridManualExecutionModePolicy` | Grid execution scope follows CellProfiler once/per-cycle setting. | Module has grid cycle scope execution policy. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:2528` | `ThresholdExecutionModePolicy` | Volumetric input forces full-stack execution. | Module supports volumetric full-stack execution. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:2534` | `WatershedExecutionModePolicy` | Volumetric input forces full-stack execution. | Module supports volumetric full-stack execution. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:2540` | `RemoveHolesExecutionModePolicy` | Volumetric input forces full-stack execution. | Module supports volumetric full-stack execution. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:2546` | `ClosingExecutionModePolicy` | Full-stack only when structuring element rank covers volume rank. | Module volumetric execution depends on structuring-element dimensionality. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:2552` | `OpeningExecutionModePolicy` | Full-stack only when structuring element rank covers volume rank. | Module volumetric execution depends on structuring-element dimensionality. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:2558` | `ErodeImageExecutionModePolicy` | Full-stack only when structuring element rank covers volume rank. | Module volumetric execution depends on structuring-element dimensionality. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:2564` | `ErodeObjectsExecutionModePolicy` | Full-stack only when structuring element rank covers volume rank. | Module volumetric execution depends on structuring-element dimensionality. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:2570` | `DilateImageExecutionModePolicy` | Full-stack only when structuring element rank covers volume rank. | Module volumetric execution depends on structuring-element dimensionality. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:2743` | `MaskObjectsPrimaryImageInputPolicy` | Declared images are carriers; object labels define execution domain. | Primary execution domain is object-label driven. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:2749` | `TrackObjectsPrimaryImageInputPolicy` | Object labels drive domain across frame/site order. | Primary execution domain is object-label driven and temporal. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:2754` | `TrackObjectsPrimaryImageInputPolicy.invocation_runtime_kwargs` | Runtime binds `image_number_start` from source paths. | Tracking requires source-order image-number start binding. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:2782` | `CombineObjectsInputPolicy` | Requires exactly two object-label inputs and passes a paired label stack as invocation image with `ALIGNED_MULTI_IMAGE_STACK`. | Module consumes object-label pair as execution image/domain override. |

### Special Input And Runtime Binding Policies

| File:line | Branch / policy | Module-specific semantic encoded there | Declaration-owned fact to query |
|---|---|---|---|
| `openhcs/interop/cellprofiler/runtime/module_execution.py:3115` | `FilterObjectsInputPolicy` | Supports additional object count, optional enclosing object, measurement features, and relationship inputs. | Ordered primary/additional/enclosing object roles plus relationship/measurement dependencies. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:3173` | `CalculateMathInputPolicy` | Can bind without declared inputs; supports measurement inputs; operand kwargs are `operand1_value` and `operand2_value`. | Measurement/object/image operand binding ABI. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:3586` | `MaskImageSpecialInputPolicy` | Object masks align to the image being masked; runtime image masks avoid current-image projection. | Special input roles: image mask and object mask aligned to primary image. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:3662` | `RelateObjectsSpecialInputPolicy` | Parent/child object labels bind in current runtime plane; optional `slice_index` is injected. | Special input roles: parent/child labels plus slice-index support. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:3728` | `CropSpecialInputPolicy` | Trailing image/object inputs are crop masks; at most one image mask and one object mask; kwargs are `mask_plane` and `cropping_labels`. | Crop-mask input roles and cardinality. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:3773` | `ImageMathSpecialInputPolicy` | Trailing images become ordered `image_operands`. | Variadic image-operand ABI. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:3790` | `WatershedSpecialInputBindingStrategy` | Marker mode consumes marker labels then optional mask; other modes consume mask. | Method-specific special input roles and dense-label marker semantics. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:3902` | `StraightenWormsSpecialInputPolicy` | Requires one worm object input; optional one producer measurement input becomes `control_points`; `num_control_points` defaults to 21. | Worm-label/control-point input ABI and producer-measurement cardinality. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:3971` | `ConvertObjectsToImageSpecialInputPolicy` | Requires one object input and binds it as `labels` payload so rendered image inherits label provenance. | Single object-label payload input and rendered-image provenance. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:3992` | `DisplayDataOnImageSpecialInputPolicy` | Requires one object input and `measurement_feature`; binds `labels` plus aligned measurement vector. | Display annotation label/measurement-vector ABI. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:4032` | `ClassifyObjectsMeasurementInputPolicy` | Requires one object input; supports classification rules or one/two measurement feature kwargs; binds measurement vectors. | Classification measurement-vector ABI and rule schema. |

### Measurement Execution And Row Policies

| File:line | Branch / policy | Module-specific semantic encoded there | Declaration-owned fact to query |
|---|---|---|---|
| `openhcs/interop/cellprofiler/runtime/measurement_image_resolver.py:67` | `CellProfilerPerObjectMeasurementPolicy.measures_images_independently` | Per-object measurement modules either use independent per-source measurement images or require a composed measurement image. | Per-object measurement source cardinality/composition mode. |
| `openhcs/interop/cellprofiler/runtime/object_measurement_execution.py:45` | `CellProfilerPerObjectMeasurementPolicy.module_names` | Specific modules are treated as per-object measurement modules. | Module runs per object label set. |
| `openhcs/interop/cellprofiler/runtime/object_measurement_execution.py:56` | `composed_image_modules` | `MeasureColocalization` is the per-object exception that consumes composed image payloads. | Module requires composed multi-image measurement payload. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:4250` | `CellProfilerPerImageMeasurementPolicy` | Image measurement execution is inferred from no object inputs, image inputs, no special inputs, measurements-only outputs, and non-composed payload consumption. | Per-image measurement cardinality and output kind. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:4302` | `MeasureTextureDualScopeMeasurementPolicy` | Object-scope callable pairs with `measure_texture` image-scope callable. | Dual image/object measurement scope and paired image function. |
| `openhcs/interop/cellprofiler/runtime/module_execution.py:4309` | `MeasureColocalizationDualScopeMeasurementPolicy` | Object-scope callable pairs with `measure_colocalization` image-scope callable. | Dual image/object measurement scope and paired image function. |
| `openhcs/interop/cellprofiler/runtime/object_measurement_row_policies.py:1078` | `MeasureObjectSizeShapeObjectMeasurementRowPolicy` | Compact measured rows anchored by `Area`, `Center_X`, `Center_Y`; no table source image owner. | Compact row identity and measuredness anchor features. |
| `openhcs/interop/cellprofiler/runtime/object_measurement_row_policies.py:1107` | generated `MeasureObjectIntensityDistributionObjectMeasurementRowPolicy` | Uses compact measured row policy but treats emitted rows as dense/complete. | Compact dense emitted object-row domain. |
| `openhcs/interop/cellprofiler/runtime/object_measurement_row_policies.py:1116` | generated `MeasureGranularityObjectMeasurementRowPolicy` | Treats emitted rows as dense/complete. | Complete emitted object-row domain. |
| `openhcs/interop/cellprofiler/runtime/object_measurement_row_policies.py:1129` | `MeasureTextureObjectMeasurementRowPolicy` | Row identity changes between row sequence and row ordinal depending on multi-source plane domain; axes are limited to scale, direction, and gray-levels. | Texture row identity, feature-axis identity fields, and multi-source missing-value behavior. |
| `openhcs/interop/cellprofiler/runtime/object_measurement_row_policies.py:1584` | `MeasureColocalizationObjectMeasurementRowPolicy` | Expands composed source stacks into source-pair invocations; projects feature names per channel pair. | Source-pair measurement expansion and channel kwarg ABI. |
| `openhcs/interop/cellprofiler/runtime/object_measurement_row_policies.py:1659` | `TrackObjectsObjectMeasurementRowPolicy` | Emits object rows and image-level tracking counts together; requires explicit row ownership. | Mixed object/image measurement rows and explicit ownership requirement. |
| `openhcs/interop/cellprofiler/runtime/object_measurement_row_policies.py:1677` | generated `MeasureObjectIntensityObjectMeasurementRowPolicy` | Dense columnar rows; missing values use zero within positive extent. | Dense columnar object-row domain and missing-value policy. |

## Replacement Work Order

1. Add declaration-owned query methods or typed attributes to `CellProfilerModule` for the fact families above, using ABC where a family contract must be enforced.
2. Move repeated families into declaration mixins: volumetric morphology, per-object measurement, dual-scope measurement, object-label primary domain, object-input role mapping, special input role mapping, and output provenance/main-flow policy.
3. Rewire runtime policy lookups to ask the module declaration instead of owning module-name registries or module-specific branches.
4. Collapse policy classes that become pure delegation once callers use declaration facts directly.
5. Delete remaining module-name lists and per-module policy registries only after generator/compiler/runtime tests are green against declaration-derived behavior.

## Scan Commands Used

```bash
rg -n "ExecutionModePolicy|PrimaryImageInputPolicy|SpecialInputPolicy|InputPolicy|ObjectMeasurementRowPolicy|PerObjectMeasurementPolicy|DualScopeMeasurementPolicy|MainFlowReplacementPolicy|ImageOutput.*Policy|requires_declared_source_image_domain|resolve_source_objects|add_objects" openhcs/interop/cellprofiler/runtime -g '*.py'
rg -n "module_name =|module_names|composed_image_modules|MeasureColocalization|MeasureTexture|CorrectIllumination|TrackObjects|MaskObjects|Crop|Watershed|StraightenWorms|ClassifyObjects|DisplayDataOnImage|ConvertObjectsToImage" openhcs/interop/cellprofiler/runtime -g '*.py'
```
