# CellProfiler Semantic Refactor Audit

Date: 2026-05-07

Scope: current dirty checkout of `openhcs-benchmark-platform`, focused on the
CellProfiler converter, benchmark corpus, runtime adapter, and OpenHCS interop
surface. This is an audit queue, not an implementation log.

## Refactor Standard

- Use typed domain objects, enums, contracts, ABCs, and `AutoRegisterMeta` when
  a closed semantic family exists.
- Prefer deriving converter behavior from runtime strategy declarations or
  module/domain contracts.
- Avoid local string maps that restate CellProfiler semantics away from their
  owner.
- Avoid registry slop: do not replace a short table with many classes unless
  those classes become the load-bearing semantic authority.
- Keep array/backend handling generic through OpenHCS and arraybridge semantics.
  Do not bake NumPy-only or file-extension heuristics into CellProfiler-specific
  plumbing when an OpenHCS abstraction exists.

## Immediate Refactor Candidates

### 1. ImageMath operation semantics

Evidence:

- `benchmark/cellprofiler_library/functions/imagemath.py:22`
- `benchmark/cellprofiler_library/functions/imagemath.py:43`
- `benchmark/cellprofiler_library/functions/imagemath.py:47`
- `benchmark/cellprofiler_library/functions/imagemath.py:168`
- `benchmark/cellprofiler_library/functions/imagemath.py:186`

Smell:

`MathOperation` is nominal, but its related behavior is split across
`BINARY_OUTPUT_OPS`, `SINGLE_IMAGE_OPS`, `_CELLPROFILER_IMAGE_MATH_OPERATIONS`,
and a large branch chain in `image_math`.

Preferred direction:

Create a load-bearing `ImageMathOperationStrategy` family keyed by
`MathOperation`, not a separate alias registry. Each strategy should own
CellProfiler literals, operand arity policy, binary/logical output policy, and
the operation implementation hook.

### 2. Threshold setting semantics

Evidence:

- `benchmark/converter/module_settings_binding.py:279`
- `benchmark/converter/module_settings_binding.py:297`
- `benchmark/converter/module_settings_binding.py:305`
- `benchmark/converter/module_settings_binding.py:309`
- `benchmark/converter/module_settings_binding.py:331`
- `benchmark/converter/module_settings_binding.py:449`

Smell:

Threshold setting identity, parser selection, ignored legacy fields, legacy
method-name upgrades, active global/adaptive selection, and CP revision upgrade
rules all live as local constants/functions inside the general module binding
file.

Preferred direction:

Create a threshold block schema owner, not one generated class per setting row.
The abstraction should own revision-aware variants, typed setting declarations,
parser selection, ignored fields, active-row policy, legacy defaults, and the
consumed setting set.

### 3. Declarative module binder class families

Evidence:

- `benchmark/converter/module_settings_binding.py:1208`
- `benchmark/converter/module_settings_binding.py:1848`
- `benchmark/converter/module_settings_binding.py:1917`
- `benchmark/converter/module_settings_binding.py:1924`
- `benchmark/converter/module_settings_binding.py:1949`
- `benchmark/converter/module_settings_binding.py:1972`
- `benchmark/converter/module_settings_binding.py:1978`
- `benchmark/converter/module_settings_binding.py:1984`
- `benchmark/converter/module_settings_binding.py:1990`

Smell:

Several registered binding strategy leaves differ only by classvars such as
`module_name` and `setting_bindings`. The nominal refactor advisor flagged this
as metaprogrammable class-family boilerplate.

Preferred direction:

Keep `DeclarativeModuleSettingsBindingStrategy` as the behavior owner, but add a
typed declaration helper that materializes registered subclasses into
`globals()`. The declaration type must preserve setting names, kwarg targets,
parser functions, ignored settings, and any module-specific post-bind hooks.

### 4. Processing component/category semantics

Evidence:

- `benchmark/converter/pipeline_generator.py:140`
- `benchmark/converter/pipeline_generator.py:151`
- `benchmark/converter/pipeline_generator.py:177`
- `benchmark/converter/pipeline_generator.py:551`

Smell:

The pipeline generator still contains local component category defaults and an
inline subset guard for `CorrectIlluminationCalculate` scopes.

Preferred direction:

Move category-to-components and scope-to-grouping semantics into typed module
processing component strategies. The generator should ask semantic objects for
components instead of branching on enum subsets.

### 5. Runtime artifact contract builder registration

Evidence:

- `benchmark/converter/symbol_table.py:2112`
- `benchmark/converter/symbol_table.py:2154`

Smell:

`_FUNCTION_BACKED_MODULE_BUILDER_SPECS` is a central tuple mapping module names
to builder functions. It is load-bearing closed-family dispatch over module
semantics, but it sits as one monolithic list.

Preferred direction:

Introduce a `ModuleArtifactContractBuilder` registered family keyed by canonical
module name. Existing builder functions can become class methods or hooks on
small subclasses. Shared patterns should stay as ABC/template-method bases.

### 6. SaveImages export bit-depth semantics

Evidence:

- `benchmark/converter/execution_validation.py:34`
- `benchmark/converter/execution_validation.py:41`
- `benchmark/converter/execution_validation.py:45`

Smell:

`SAVE_IMAGES_BIT_DEPTHS` is a string map from CellProfiler UI literals to
OpenHCS export bit-depth semantics.

Preferred direction:

Add a typed `CellProfilerSaveImagesBitDepth` enum or `SaveImagesExportStrategy`
that derives the runtime export bit depth.

### 7. GrayToColor revision and channel schema semantics

Evidence:

- `benchmark/converter/gray_to_color_settings.py:33`
- `benchmark/converter/gray_to_color_settings.py:43`
- `benchmark/converter/gray_to_color_settings.py:55`
- `benchmark/converter/gray_to_color_settings.py:80`
- `benchmark/converter/gray_to_color_settings.py:160`
- `benchmark/converter/gray_to_color_settings.py:200`

Smell:

GrayToColor has partial nominal structure, but fixed RGB/CMYK channel setting
tuples, repeated Stack/Composite row parsing, blank-source literals, and
revision-specific rescale defaults still sit as local constants/functions.

Preferred direction:

Make the scheme strategy own fixed channel slots, repeated-channel parsing,
blank-source policy or delegation to a shared CellProfiler null-literal policy,
and revision-specific rescale defaults.

### 8. Dataset/cppipe source layout and image detection

Evidence:

- `benchmark/converter/cppipe_corpus.py:13`
- `benchmark/converter/cppipe_corpus.py:73`
- `benchmark/datasets/acquire.py:25`
- `benchmark/datasets/acquire.py:322`
- `openhcs/core/source_matching.py:14`
- `openhcs/core/source_matching.py:76`

Smell:

Dataset acquisition has its own `IMAGE_EXTENSIONS` set while OpenHCS already has
`LOADABLE_IMAGE_EXTENSIONS` and source matching helpers. Official CellProfiler
example layout is recognized by hardcoded path fragments such as
`CellProfiler3Pipelines`.

Preferred direction:

Use OpenHCS file/source matching for image detection. Put official CellProfiler
example layout into a typed dataset source/layout declaration. Keep the VFS
boundary explicit.

### 9. CellProfiler null/blank literal semantics

Evidence:

- `benchmark/converter/settings_binder.py:21`
- `benchmark/converter/settings_binder.py:254`
- `benchmark/converter/calculate_math_settings.py:202`
- `benchmark/converter/filter_objects_settings.py:483`
- `benchmark/converter/gray_to_color_settings.py:200`
- `benchmark/converter/module_function_resolution.py:319`
- `benchmark/converter/overlay_outlines_settings.py:290`

Smell:

Many modules spell their own variants of `"none"`, `"do not use"`,
`"leave this black"`, and related inactive-source literals.

Preferred direction:

Create a typed literal policy in `openhcs.interop.cellprofiler` that
distinguishes generic false/disabled literals, inactive source/image/object
literals, color-specific black channel literals, and optional measurement/object
literals. Do not use one broad "anything blank means no" helper everywhere.

### 10. Generic enum coercion heuristics

Evidence:

- `openhcs/interop/cellprofiler/settings_binder.py:43`
- `openhcs/interop/cellprofiler/settings_binder.py:125`
- `openhcs/interop/cellprofiler/settings_binder.py:139`
- `openhcs/interop/cellprofiler/settings_binder.py:148`

Smell:

`coerce_cellprofiler_enum` relies on normalized name/value matching, prefix
matching, negated-literal expansion, and enum-name suffix stripping. This is
useful but heuristic.

Preferred direction:

Let enum members or registered semantic strategies declare accepted
CellProfiler literals explicitly when parity depends on exact UI language. Keep
the generic coercer as a convenience fallback for low-risk cases.

### 11. Runtime execution measurement projection modules

Evidence:

- `openhcs/core/runtime_equivalence.py:146`

Smell:

The set of modules that need measurement-row projection appears as a local
tuple. That is likely module semantic policy.

Preferred direction:

Derive measurement projection policy from module artifact contracts or
CellProfiler module semantics rather than a standalone module-name tuple.

### 12. Viewer component abbreviation duplication

Evidence:

- `openhcs/runtime/napari_viewer_server.py:968`
- `openhcs/runtime/fiji_viewer_server.py:1415`
- `openhcs/runtime/fiji_viewer_server.py:1562`
- `openhcs/runtime/napari_stream_visualizer.py:978`

Smell:

Component abbreviations are duplicated across visualizer/server implementations.

Preferred direction:

Move display abbreviation/name policy to the component enum or a shared
component display descriptor.

## Defer Or Avoid

### Threshold row class generation without a schema owner

Do not generate one class per threshold setting just to avoid a dict. The
refactor has to create a threshold-block semantic owner that handles revisioning,
active rows, ignored fields, parser selection, and legacy defaults together.

### SourceLocator conversion helper cleanup

Evidence:

- `benchmark/converter/source_locator.py:52`
- `benchmark/converter/source_locator.py:97`
- `benchmark/converter/source_locator.py:130`
- `benchmark/converter/source_locator.py:149`

This file is old conversion support. It has hardcoded source tree layout and
string search, but it is not obviously on the benchmark runtime critical path.
Refactor only if it is still used by active conversion flows.

### GUI/action registries

Search finds action maps in PyQt/Textual UI files. They may deserve their own UI
refactor pass, but they are out of scope for the CellProfiler benchmark semantic
cleanup.

### Color/style constants in graphing

`benchmark/reports/cppipe_figures.py` contains plotting constants. These are
presentation policy, not CellProfiler runtime semantics.

## Advisor Run Notes

The nominal refactor advisor was run successfully on `benchmark/converter` and
reported 59 findings. The useful findings for this audit were closed string
dispatch in `execution_validation.py`, classvar-only registered strategy leaves
in `module_settings_binding.py`, the inline enum subset guard in
`pipeline_generator.py`, and the GrayToColor revision selector.

Running it on all of `benchmark` failed because vendored CellProfiler source
contains files with a UTF-8 BOM/non-printable character. That is an advisor input
hygiene issue, not evidence that the benchmark code is invalid.

## Suggested Order

1. Refactor ImageMath operation semantics into a real operation strategy family.
2. Design the threshold block schema owner before touching threshold code again.
3. Collapse declarative binder class-only leaves with typed class generation.
4. Move processing-component/category decisions out of the generator and into
   semantic strategies.
5. Convert artifact contract builder registration to an `AutoRegisterMeta`
   family.
6. Clean up dataset/source image detection through OpenHCS source matching and
   VFS-aware abstractions.

## Verification Expectations

- Run focused unit tests for the touched semantic family.
- Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../openhcs/.venv/bin/python -m pytest`
  on the relevant converter/runtime tests before claiming readiness.
- Run `git diff --check`.
- For performance-sensitive runtime paths, rerun the affected cppipe benchmark
  before assuming parity or speed was preserved.
