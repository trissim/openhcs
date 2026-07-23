# CellProfiler ↔ OpenHCS Architecture Mapping

**Date:** 2026-02-16
**Status:** Design Document
**Goal:** Leak-free abstraction for CellProfiler pipeline support in OpenHCS

---

## Executive Summary

This document maps CellProfiler's architecture to OpenHCS to identify:
1. **Direct mappings** - Concepts that translate cleanly
2. **Semantic gaps** - Missing concepts in OpenHCS
3. **Adapter layers** - Required translation mechanisms
4. **Abstraction leaks** - Where CellProfiler assumptions break OpenHCS patterns

---

## 1. Core Concept Mapping

### 1.1 Pipeline Execution Model

| CellProfiler | OpenHCS | Mapping |
|--------------|---------|---------|
| Pipeline (list of Modules) | Pipeline (list of FunctionSteps) | ✅ Direct |
| Module.run(workspace) | FunctionStep.process(context, step_index) | ✅ Direct |
| Sequential module execution | Sequential step execution | ✅ Direct |
| Image set iteration | Well/site iteration | ⚠️ Different granularity |
| Workspace (per-cycle state) | ProcessingContext (per-axis state) | ✅ Direct |

**Key Difference:**
- CellProfiler: One workspace per **image set** (single field of view)
- OpenHCS: One context per **axis** (well, potentially multiple sites)

### 1.2 Data Container Mapping

| CellProfiler | OpenHCS | Mapping |
|--------------|---------|---------|
| `workspace.image_set` | `context.filemanager` + step_plans | ⚠️ Requires adapter |
| `workspace.object_set` | **MISSING** | ❌ New concept needed |
| `workspace.measurements` | `@special_outputs` + MaterializationSpec | ⚠️ Different model |
| `workspace.display_data` | Not applicable (headless) | ✅ Skip |
| `workspace.pipeline` | `context.global_config` | ✅ Direct |

### 1.3 Object/Image Model

| CellProfiler | OpenHCS | Mapping |
|--------------|---------|---------|
| `Image.pixel_data` (named) | 3D numpy array (positional) | ⚠️ Channel naming needed |
| `Image.mask` | Not directly supported | ⚠️ Could use alpha channel |
| `Objects.segmented` | 3D label array (step output) | ⚠️ No object registry |
| `Objects.parent_image` | Not tracked | ❌ Missing |
| `Objects.relate_children()` | Not supported | ❌ Missing |

---

## 2. Semantic Gaps (Missing in OpenHCS)

### 2.1 Object Registry (CRITICAL)

**CellProfiler has:**
```python
# Named objects that persist across modules
workspace.object_set.add_objects(nuclei, "Nuclei")
cells = workspace.object_set.get_objects("Nuclei")  # Later module
```

**OpenHCS lacks:**
- No concept of named, referenceable objects
- Step outputs are anonymous 3D arrays
- No parent-child relationship tracking

**Required for CellProfiler:**
```python
# Proposed: ObjectRegistry in ProcessingContext
class ObjectRegistry:
    def register(self, name: str, labels: np.ndarray, metadata: dict)
    def get(self, name: str) -> ObjectEntry
    def relate(self, parent: str, child: str, mapping: np.ndarray)
    def list_objects() -> List[str]
```

### 2.2 Named Image Registry

**CellProfiler has:**
```python
# Named images from NamesAndTypes module
dapi = workspace.image_set.get_image("DNA")
gfp = workspace.image_set.get_image("GFP")
```

**OpenHCS has:**
- Channel dimension in arrays (positional: channel 0, 1, 2)
- No semantic naming of channels

**Required for CellProfiler:**
```python
# Proposed: NamedImageRegistry in ProcessingContext
class NamedImageRegistry:
    def register(self, name: str, channel_index: int)
    def get(self, name: str) -> np.ndarray
    def list_images() -> List[str]
```

### 2.3 Measurement Aggregation

**CellProfiler has:**
```python
# Per-object measurements with naming convention
workspace.measurements.add_measurement(
    "Nuclei",  # Object name
    "AreaShape_Area",  # Feature name
    areas  # np.array of per-object values
)
```

**OpenHCS has:**
- `@special_outputs` returns single value per step
- No per-object measurement aggregation
- No naming convention

**Required for CellProfiler:**
```python
# Proposed: MeasurementCollector in ProcessingContext
class MeasurementCollector:
    def add(self, object_name: str, feature: str, values: np.ndarray)
    def get(self, object_name: str, feature: str) -> np.ndarray
    def get_columns(self) -> List[Tuple[str, str]]  # (object, feature)
    def to_dataframe(self) -> pd.DataFrame
```

### 2.4 Object Relationships

**CellProfiler has:**
```python
# Primary → Secondary → Tertiary pattern
children_per_parent, parents_of_children = nuclei.relate_children(cells)
# children_per_parent[i] = number of cells from nucleus i
# parents_of_children[j] = parent nucleus of cell j
```

**OpenHCS lacks:**
- No object relationship tracking
- No parent-child semantics

**Required for CellProfiler:**
```python
# Proposed: RelationshipTracker
class RelationshipTracker:
    def record(self, parent: str, child: str, mapping: np.ndarray)
    def get_children_of(self, parent_name: str, parent_id: int) -> List[int]
    def get_parent_of(self, child_name: str, child_id: int) -> int
```

---

## 3. Adapter Layer Design

### 3.1 CellProfilerContextAdapter

Wraps OpenHCS ProcessingContext to provide CellProfiler-compatible workspace:

```python
class CellProfilerContextAdapter:
    """
    Adapts OpenHCS ProcessingContext to CellProfiler Workspace interface.
    
    Allows CellProfiler modules to run with minimal modification.
    """
    
    def __init__(self, context: ProcessingContext, step_index: int):
        self._context = context
        self._step_index = step_index
        
        # Registries (new concepts)
        self._object_registry = ObjectRegistry()
        self._image_registry = NamedImageRegistry()
        self._measurements = MeasurementCollector()
        self._relationships = RelationshipTracker()
        
        # Display data (for compatibility, not used in headless)
        self.display_data = SimpleNamespace()
    
    # CellProfiler Workspace interface
    @property
    def image_set(self) -> 'ImageSetAdapter':
        return ImageSetAdapter(self._context, self._image_registry)
    
    @property
    def object_set(self) -> 'ObjectSetAdapter':
        return ObjectSetAdapter(self._object_registry)
    
    @property
    def measurements(self) -> 'MeasurementsAdapter':
        return MeasurementsAdapter(self._measurements)
    
    @property
    def pipeline(self) -> 'PipelineAdapter':
        return PipelineAdapter(self._context.global_config)
    
    def add_measurement(self, object_name: str, feature: str, value):
        """Convenience method for single measurement."""
        self._measurements.add(object_name, feature, np.array([value]))
```

### 3.2 ImageSetAdapter

```python
class ImageSetAdapter:
    """Provides CellProfiler's image_set interface."""
    
    def __init__(self, context: ProcessingContext, registry: NamedImageRegistry):
        self._context = context
        self._registry = registry
    
    def get_image(self, name: str, must_be_grayscale: bool = True) -> ImageAdapter:
        # Get channel index from registry
        channel_idx = self._registry.get_channel_index(name)
        
        # Load from context's step plan
        step_plan = self._context.step_plans[self._step_index]
        # ... load image stack ...
        
        return ImageAdapter(image_stack, channel_idx, name)
```

### 3.3 ObjectSetAdapter

```python
class ObjectSetAdapter:
    """Provides CellProfiler's object_set interface."""
    
    def __init__(self, registry: ObjectRegistry):
        self._registry = registry
    
    def get_objects(self, name: str) -> ObjectsAdapter:
        entry = self._registry.get(name)
        return ObjectsAdapter(entry)
    
    def add_objects(self, objects: 'ObjectsAdapter', name: str):
        self._registry.register(name, objects.segmented, objects.metadata)
```

### 3.4 ObjectsAdapter

```python
class ObjectsAdapter:
    """
    Wraps OpenHCS label array to provide CellProfiler Objects interface.
    """
    
    def __init__(self, labels: np.ndarray, metadata: dict = None):
        self._labels = labels
        self._metadata = metadata or {}
        
        # CellProfiler properties
        self.segmented = labels
        self.unedited_segmented = labels.copy()
        self.small_removed_segmented = None
        self.parent_image = None
    
    @property
    def count(self) -> int:
        return int(self._labels.max())
    
    @property
    def indices(self) -> np.ndarray:
        return np.arange(1, self.count + 1)
    
    @property
    def areas(self) -> np.ndarray:
        from scipy import ndimage
        return ndimage.sum(
            np.ones_like(self._labels),
            self._labels,
            self.indices
        )
    
    def relate_children(self, child_objects: 'ObjectsAdapter') -> Tuple[np.ndarray, np.ndarray]:
        """Map parent objects to child objects based on overlap."""
        parent_labels = self._labels
        child_labels = child_objects._labels
        
        n_parents = self.count
        n_children = child_objects.count
        
        # For each child, find most overlapping parent
        parents_of_children = np.zeros(n_children + 1, dtype=int)
        children_per_parent = np.zeros(n_parents + 1, dtype=int)
        
        # Flatten and compare
        for child_id in range(1, n_children + 1):
            child_mask = child_labels == child_id
            parent_values = parent_labels[child_mask]
            
            if len(parent_values) > 0:
                # Most common parent
                parent_id = np.bincount(parent_values)[1:].argmax() + 1
                parents_of_children[child_id] = parent_id
                children_per_parent[parent_id] += 1
        
        return children_per_parent, parents_of_children
```

---

## 4. ProcessingContract Mapping

### 4.1 CellProfiler volumetric() → OpenHCS Contract

| CellProfiler | OpenHCS Contract | Notes |
|--------------|------------------|-------|
| `volumetric() = False` | `PURE_2D` | Process slices, restack |
| `volumetric() = True` | `PURE_3D` or `FLEXIBLE` | Full 3D processing |
| No volumetric method | `PURE_2D` | Default assumption |

### 4.2 Contract Inference Logic

```python
def infer_contract(module_class) -> ProcessingContract:
    """Infer OpenHCS contract from CellProfiler module."""
    
    # Check if module has volumetric() method
    if hasattr(module_class, 'volumetric'):
        instance = module_class()
        if instance.volumetric():
            # Check for slice_by_slice parameter
            sig = inspect.signature(instance.run)
            if 'slice_by_slice' in sig.parameters:
                return ProcessingContract.FLEXIBLE
            return ProcessingContract.PURE_3D
    
    # Default: 2D processing
    return ProcessingContract.PURE_2D
```

---

## 5. Measurement Naming Convention Mapping

### 5.1 CellProfiler → OpenHCS Path Mapping

| CellProfiler Measurement | OpenHCS Special Output Path |
|--------------------------|----------------------------|
| `Image_Count_Nuclei` | `{output_dir}_results/{filename}_image_count.csv` |
| `Nuclei_Location_Center_X` | `{output_dir}_results/{filename}_nuclei_location.csv` |
| `Nuclei_AreaShape_Area` | `{output_dir}_results/{filename}_nuclei_areashape.csv` |
| `Nuclei_Intensity_MeanIntensity_DAPI` | `{output_dir}_results/{filename}_nuclei_intensity_dapi.csv` |

### 5.2 MaterializationSpec for CellProfiler

```python
from openhcs.processing.materialization import MaterializationSpec, CsvOptions

# CellProfiler measurements → CSV
CELLPROFILER_MEASUREMENT_SPEC = MaterializationSpec(
    format=CsvOptions(
        index_col="ObjectNumber",
        include_header=True,
        float_format="%.6f"
    )
)

# Usage in absorbed function
@special_outputs(
    ("nuclei_measurements", CELLPROFILER_MEASUREMENT_SPEC),
    ("cell_measurements", CELLPROFILER_MEASUREMENT_SPEC),
)
def measure_objects(image, nuclei_labels, cell_labels):
    # ... compute measurements ...
    return image, nuclei_df, cells_df
```

---

## 6. Settings System Mapping

### 6.1 CellProfiler Settings → OpenHCS Parameters

| CellProfiler Setting | OpenHCS Parameter | Type Mapping |
|---------------------|-------------------|--------------|
| `Binary(text, value)` | `param: bool = value` | ✅ Direct |
| `Choice(text, choices)` | `param: Literal[*choices]` | ✅ Direct |
| `Float(text, value)` | `param: float = value` | ✅ Direct |
| `Integer(text, value)` | `param: int = value` | ✅ Direct |
| `IntegerRange(text, (min,max))` | `min_val: int, max_val: int` | ⚠️ Split to two params |
| `ImageSubscriber(text)` | Not a parameter | ⚠️ Resolved at compile time |
| `LabelSubscriber(text)` | Not a parameter | ⚠️ Resolved at compile time |
| `LabelName(text)` | Not a parameter | ⚠️ Output name, not input |

### 6.2 Settings Extraction Example

```python
# CellProfiler module settings
class IdentifyPrimaryObjects:
    def create_settings(self):
        self.x_name = ImageSubscriber("Select input image", "None")
        self.y_name = LabelName("Name primary objects", "Nuclei")
        self.size_range = IntegerRange("Typical diameter", (10, 40))
        self.exclude_size = Binary("Discard objects outside range?", True)
        self.unclump_method = Choice("Declumping method", ["Intensity", "Shape", "None"])

# OpenHCS absorbed function parameters
def identify_primary_objects(
    image: np.ndarray,  # x_name → resolved at compile time
    min_diameter: int = 10,  # size_range.min
    max_diameter: int = 40,  # size_range.max
    exclude_size: bool = True,  # exclude_size
    unclump_method: Literal["Intensity", "Shape", "None"] = "Intensity",
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    # y_name → output registered in ObjectRegistry
    ...
```

---

## 7. Execution Flow Comparison

### 7.1 CellProfiler Flow

```
Pipeline.run():
    prepare_run() → Create ImageSets from input
    
    for grouping in groupings:
        prepare_group()
        
        for image_number in grouping:
            workspace = Workspace(image_set, object_set, measurements)
            
            for module in modules:
                module.run(workspace)
                
        post_group()
    
    post_run()
    
    ExportToSpreadsheet: measurements → CSV
```

### 7.2 OpenHCS Flow

```
Orchestrator.execute_compiled_plate():
    
    for well in wells:
        context = ProcessingContext(well, ...)
        context.freeze()
        
        for step in pipeline:
            step.process(context, step_index)
            
            # Special outputs materialized at end
```

### 7.3 Integrated Flow (Proposed)

```
Orchestrator.execute_cellprofiler_pipeline():
    
    for well in wells:
        context = ProcessingContext(well, ...)
        cp_context = CellProfilerContextAdapter(context)
        
        # NamesAndTypes equivalent
        cp_context._image_registry.register("DNA", 0)
        cp_context._image_registry.register("GFP", 1)
        
        for step in pipeline:
            if step.is_cellprofiler_module:
                # CellProfiler-style execution
                step.module.run(cp_context)
            else:
                # Native OpenHCS execution
                step.process(context, step_index)
        
        # ExportToSpreadsheet equivalent
        measurements = cp_context._measurements.to_dataframe()
        context.filemanager.save(
            measurements,
            f"{well}_measurements.csv",
            Backend.DISK
        )
```

---

## 8. Abstraction Leak Analysis

### 8.1 Identified Leaks

| Leak | Severity | Cause | Mitigation |
|------|----------|-------|------------|
| **Object naming** | HIGH | CP modules reference objects by string name | ObjectRegistry adapter |
| **Image naming** | MEDIUM | CP modules reference images by semantic name | NamedImageRegistry adapter |
| **Measurement naming** | MEDIUM | CP has strict naming convention | MeasurementCollector with convention |
| **Parent-child relationships** | HIGH | CP tracks object genealogy | RelationshipTracker adapter |
| **Workspace mutation** | LOW | CP modules modify workspace in place | Adapter wraps immutable context |
| **Display data** | LOW | CP modules set display_data | Adapter provides dummy namespace |

### 8.2 Leak-Free Principle

**Goal:** CellProfiler modules should run without knowing they're in OpenHCS.

**Test:**
```python
def test_abstraction_leak():
    """Verify CellProfiler module runs identically in both environments."""
    
    # Create test data
    image = np.random.rand(100, 100)
    
    # Run in CellProfiler
    cp_workspace = create_cellprofiler_workspace(image)
    cp_module = IdentifyPrimaryObjects()
    cp_module.run(cp_workspace)
    cp_result = cp_workspace.object_set.get_objects("Nuclei").segmented
    
    # Run in OpenHCS
    context = create_openhcs_context(image)
    adapter = CellProfilerContextAdapter(context, step_index=0)
    oh_module = IdentifyPrimaryObjects()
    oh_module.run(adapter)
    oh_result = adapter.object_set.get_objects("Nuclei").segmented
    
    # Results should be identical
    np.testing.assert_array_equal(cp_result, oh_result)
```

---

## 9. Implementation Roadmap

### Phase 1: Core Adapters (Week 1-2)

1. **ObjectRegistry** - Named object storage and retrieval
2. **NamedImageRegistry** - Semantic channel naming
3. **CellProfilerContextAdapter** - Workspace-compatible wrapper

### Phase 2: Measurement System (Week 3)

1. **MeasurementCollector** - Per-object measurement aggregation
2. **RelationshipTracker** - Parent-child object tracking
3. **MaterializationSpec** - CellProfiler CSV format

### Phase 3: Module Absorption (Week 4-5)

1. Update absorbed functions to use adapters
2. Add `@cellprofiler_module` decorator for metadata
3. Generate pipeline from `.cppipe` files

### Phase 4: Integration Testing (Week 6)

1. Test with real CellProfiler pipelines
2. Verify measurement output matches CellProfiler
3. Performance benchmarking

---

## 10. API Design Summary

### 10.1 New Decorator for CellProfiler Modules

```python
from openhcs.core.pipeline.cellprofiler_contracts import cellprofiler_module

@cellprofiler_module(
    module_name="IdentifyPrimaryObjects",
    input_images={"image": "DNA"},  # Name → registry key
    output_objects={"nuclei": "Nuclei"},  # Output name → registry key
    volumetric=False,
)
def identify_primary_objects(
    image: np.ndarray,
    min_diameter: int = 10,
    max_diameter: int = 40,
    ...
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    ...
```

### 10.2 Context Extension

```python
# ProcessingContext extensions
class ProcessingContext:
    # Existing attributes...
    
    # CellProfiler support (optional, only if needed)
    _cp_adapter: Optional[CellProfilerContextAdapter] = None
    
    @property
    def cellprofiler(self) -> CellProfilerContextAdapter:
        if self._cp_adapter is None:
            self._cp_adapter = CellProfilerContextAdapter(self)
        return self._cp_adapter
```

### 10.3 Pipeline Generation

```python
from openhcs.benchmark.converter.cppipe_to_pipeline import CPPipeToPipeline

generator = CPPipeToPipeline()
pipeline = generator.convert("my_pipeline.cppipe")

# Result: List[FunctionStep] with CellProfiler modules wrapped
```

---

## 11. Conclusion

The mapping reveals that OpenHCS can support CellProfiler pipelines with three key additions:

1. **ObjectRegistry** - For named object references
2. **NamedImageRegistry** - For semantic channel names
3. **MeasurementCollector** - For per-object measurements

The adapter pattern allows CellProfiler modules to run unmodified while integrating cleanly with OpenHCS's execution model.

**Critical Insight:** The current absorbed functions (88 modules) are "leaky" because they:
- Don't track object names
- Don't aggregate measurements properly
- Use `PURE_2D` instead of `PURE_3D`

The refactor plan in `plans/cellprofiler_refactor_plan.md` should be updated to include these architectural changes.
