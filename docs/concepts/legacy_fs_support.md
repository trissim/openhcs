# Legacy Filesystem Support

## Overview

EZStitcher is transitioning from a filesystem-based architecture to a more flexible storage-based architecture. During this transition, some components may still require access to the filesystem. The `requires_legacy_fs` flag is used to indicate that a step requires access to the filesystem, even when using a non-legacy storage mode.

## Usage

The `requires_legacy_fs` flag can be set in two ways:

1. As a class attribute:
   ```python
   class MyStep(Step):
       requires_legacy_fs = True
   ```

2. As a constructor parameter:
   ```python
   step = Step(func=my_func, requires_legacy_fs=True)
   ```

## How It Works

When a step with `requires_legacy_fs=True` is executed in a non-legacy storage mode (e.g., "memory" or "zarr"), the `PipelineOrchestrator` will automatically materialize the necessary files to the filesystem before executing the step, and clean up after the step is complete.

This is done through the `LegacyOverlayHelper` class, which manages the materialization of files from the storage backend to the filesystem.

## Example

```python
from ezstitcher.core.steps import Step

# Create a step that requires filesystem access
step = Step(
    func=my_func,
    requires_legacy_fs=True,
    name="My Legacy Step"
)

# Add the step to a pipeline
pipeline = Pipeline(
    steps=[step],
    name="My Pipeline"
)

# Run the pipeline with a non-legacy storage mode
orchestrator = PipelineOrchestrator(
    storage_mode="memory",
    overlay_mode="auto"
)
orchestrator.run_pipeline(pipeline)
```

In this example, the `PipelineOrchestrator` will automatically materialize the necessary files to the filesystem before executing the step, and clean up after the step is complete.

## Migration Path

The `requires_legacy_fs` flag is intended to be a temporary solution during the transition to a fully storage-based architecture. Eventually, all components should be updated to work with the storage backend directly, and the `requires_legacy_fs` flag will be deprecated.

For new components, it is recommended to use the storage backend directly rather than relying on the filesystem. This can be done by accessing the storage adapter through the context:

```python
def my_func(images, context):
    # Access the storage adapter
    storage_adapter = context.orchestrator.storage_adapter
    
    # Read from the storage adapter
    data = storage_adapter.read("my_key")
    
    # Write to the storage adapter
    storage_adapter.write("my_key", data)
    
    # Return the processed images
    return images
```
