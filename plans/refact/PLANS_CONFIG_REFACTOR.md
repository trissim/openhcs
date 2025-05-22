# Configuration System Refactoring Plan for OpenHCS

**Date:** 2025-05-20
**Version:** 1.2 (Removed specific default backend types and force_microscope_parser from global config)

## 1. Overarching Goal

To modernize the OpenHCS configuration system ([`openhcs/core/config.py`](openhcs/core/config.py:1)) to:
1.  Fully align with the VFS-centric pipeline model (as per [`PLANS_CORE_PIPELINE_VFS.md`](plans/PLANS_CORE_PIPELINE_VFS.md:1)).
2.  Provide clear, type-safe, and easily loadable/serializable configuration for all aspects of the system (pipeline behavior, VFS, backends, TUI).
3.  Eliminate outdated or "legacy" configuration options that contradict the new architectural principles.
4.  Ensure configuration is primarily loaded at application startup and treated as immutable where possible, or managed through a well-defined lifecycle.

## 2. Current State Analysis (`openhcs/core/config.py`)

*   Uses dataclasses (`MISTConfig`, `BackendConfig`, `StitcherConfig`, `PipelineConfig`).
*   Relies on defaults from `constants.py`.
*   `PipelineConfig.storage_mode` ("legacy", "memory", "zarr") and `storage_root` are present and relate to VFS backends.
*   Path suffixes (`out_dir_suffix`, etc.) are defined, relevant to `PipelinePathPlanner`.
*   Backend-specific configurations exist.

## 3. Identified Issues & "Rot"

*   **"Legacy" Storage Mode:** The `storage_mode = "legacy"` option in `PipelineConfig` refers to an old in-memory dictionary model. This is superseded by VFS-exclusive inter-step communication.
*   **Implicit Configuration Loading/Usage:** Clarity needed on how `PipelineConfig` objects are instantiated, loaded, and consistently accessed.
*   **VFS Backend Configuration:** While `storage_mode` hints at VFS, explicit configuration for default VFS backends (used by planners) is needed.
*   **Step-Level Configuration:** Current focus is pipeline/backend level. Step-specific overrides might be a future consideration.
*   **Over-Specific Defaults:** Previous drafts included global defaults for specific processing step *types* (e.g., default position generator backend type), which is better handled by explicit user selection of functions in the pipeline definition. Similarly, forcing a microscope parser globally can conflict with per-dataset auto-detection or selection.

## 4. Proposed Refactoring Actions

### 4.1. Eliminate "Legacy" Configuration
*   **Action:** Remove `storage_mode = "legacy"` from `PipelineConfig`. `storage_mode` could simplify to name a default VFS backend for intermediates (e.g., "memory", "zarr", "disk") if not overridden by planners.
*   **Impact:** Simplifies `PipelineConfig`, aligns with VFS exclusivity.

### 4.2. Centralized Configuration Loading and Access
*   **Action:**
    *   Define a clear mechanism for loading a global configuration object (e.g., `GlobalPipelineConfig`) at application startup (e.g., from YAML/JSON or environment variables).
    *   The main application entry point loads this configuration.
    *   The loaded configuration object is injected into components like `PipelineOrchestrator` and then into `ProcessingContext`.
*   **Impact:** Single source of truth for configuration, explicit dependencies.

### 4.3. VFS Configuration Integration
*   **Action:**
    *   Global configuration should define default VFS settings (e.g., `VFSConfig.default_intermediate_backend: str`, `VFSConfig.default_materialization_backend: str = "disk"`) for planners to use as fallbacks.
    *   `VFSConfig.persistent_storage_root_path` for disk/Zarr backends should be configurable and validated.
    *   Path-related configurations (e.g., `PathPlanningConfig.output_dir_suffix`) consumed by `PipelinePathPlanner` from global config.
*   **Impact:** Makes VFS behavior more configurable and predictable.

### 4.4. Step-Specific Function Choice vs. Global Defaults
*   **Clarification:** Choices for specific processing functions (e.g., which Ashlar or MIST variant to use for position generation, or which assembler function) are made by the user explicitly when defining a pipeline (e.g., by selecting the specific function in the TUI's Function Pattern Editor). The global configuration should not attempt to set default "types" of these functions, as the functions themselves (available in `FUNC_REGISTRY`) represent these choices.
*   **Impact:** Reinforces user control and explicitness in pipeline definition, reduces unnecessary global config.

### 4.5. Configuration Immutability
*   **Action:** Loaded configuration objects should be treated as immutable. Use `copy()` methods for rare cases needing local modification.
*   **Impact:** Enhances predictability.

## 5. New Dataclasses / Structure (Illustrative)

```python
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any
# from pathlib import Path # Import Path if used for type hints of path values

@dataclass(frozen=True)
class VFSConfig:
    default_intermediate_backend: Literal["memory", "disk", "zarr"] = "memory"
    default_materialization_backend: Literal["disk"] = "disk" 
    persistent_storage_root_path: Optional[str] = None 

@dataclass(frozen=True)
class PathPlanningConfig:
    output_dir_suffix: str = "_outputs" 
    positions_dir_suffix: str = "_positions"
    stitched_dir_suffix: str = "_stitched"
    # ... other suffixes ...

@dataclass(frozen=True)
class GlobalPipelineConfig:
    path_planning: PathPlanningConfig = field(default_factory=PathPlanningConfig)
    num_workers: int = 1 
    vfs: VFSConfig = field(default_factory=VFSConfig)
    # Removed: default_position_generator_backend_type
    # Removed: default_image_assembler_backend_type
    # Removed: force_microscope_parser 

    @classmethod
    def load_from_yaml(cls, file_path: str) -> 'GlobalPipelineConfig':
        # Placeholder for loading from YAML.
        # import yaml
        # from dacite import from_dict
        # with open(file_path, 'r') as f:
        #     data = yaml.safe_load(f)
        # return from_dict(data_class=cls, data=data)
        pass
```

## 6. Integration with Other Components

*   **Application Entry Point (e.g., TUI main, script):** Responsible for loading the `GlobalPipelineConfig`.
*   **`PipelineOrchestrator`:** Initialized with `GlobalPipelineConfig`, passes it to `ProcessingContext`.
*   **`ProcessingContext`:** Holds `GlobalPipelineConfig` (or relevant sub-configs) for use by planners and steps.
*   **`PipelineCompiler`, `PipelinePathPlanner`, `MaterializationFlagPlanner`:**
    *   These components consume settings from `GlobalPipelineConfig` (via `ProcessingContext`) to guide their planning.
    *   For example, `PipelinePathPlanner` uses `PathPlanningConfig` for suffixes and `VFSConfig.persistent_storage_root_path` as a base for constructing VFS paths.
    *   `MaterializationFlagPlanner` uses `VFSConfig.default_intermediate_backend` and `VFSConfig.default_materialization_backend` as defaults when determining the `backend` and `materialize` flags for entries in `step_plan['special_input_vfs_info']` / `step_plan['special_output_vfs_info']`, and for `step_plan['read_backend']` / `step_plan['write_backend']` / `step_plan['force_disk_output']` for primary I/O.
    *   The `PipelineCompiler` then assembles all this information into the flat `step_plan` structure detailed in [`PLANS_CORE_PIPELINE_VFS.md`](plans/PLANS_CORE_PIPELINE_VFS.md:1) (v3.0).

## 7. Open Questions
*   Detailed schema for configuration files (e.g., YAML structure).
*   Strategy for handling sensitive configuration values (if any).
*   Mechanism for potential overrides (e.g., pipeline-specific config overriding global defaults). For initial refactor, a single global config is simpler.