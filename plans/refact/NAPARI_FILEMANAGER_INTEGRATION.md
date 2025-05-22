# Napari FileManager Integration

## Overview

This document outlines the approach for integrating the Napari visualization framework with the new VFS-exclusive, StepResult-free architecture. Instead of receiving tensor data directly, Napari will use a FileManager instance to read from the same VFS paths used by processing steps.

## Current State

Currently, NapariStreamVisualizer:
1. Receives tensor data directly through `push_tensor(step_id, tensor, well_id)`
2. Has no FileManager or access to the VFS
3. Depends on StepResult.data which will be eliminated

## Implementation Approach

### 1. FileManager-Based Napari Visualizer

```python
class NapariStreamVisualizer:
    """
    Manages a Napari viewer instance with VFS integration.
    """
    
    def __init__(
        self, 
        filemanager: 'FileManager',
        viewer_title: str = "OpenHCS Real-Time Visualization"
    ):
        """
        Initialize with a FileManager instance.
        
        Args:
            filemanager: FileManager instance with the same backends as the pipeline
            viewer_title: Title for the Napari window
        """
        self.filemanager = filemanager
        self.viewer_title = viewer_title
        self.viewer = None
        self.layers = {}
        self.data_queue = queue.Queue()
        self.viewer_thread = None
        self.is_running = False
        self._lock = threading.Lock()
```

### 2. Path-Based Visualization

Replace the current tensor-based `push_tensor` with a path-based method:

```python
def visualize_path(
    self, 
    step_id: str, 
    path: str, 
    backend: str,
    well_id: Optional[str] = None
):
    """
    Visualize data from a VFS path.
    
    Args:
        step_id: The step ID for display
        path: VFS path to the data
        backend: Backend name (e.g. 'memory')
        well_id: Optional well ID for grouping
    """
    if not self.is_running and self.viewer_thread is None:
        logger.info(f"Starting Napari viewer for step '{step_id}'.")
        self.start_viewer()
    
    if not self.is_running:
        logger.warning(f"Visualizer not running. Cannot visualize path for step '{step_id}'.")
        return
    
    try:
        # Queue the path for visualization by the viewer thread
        self.data_queue.put({
            'step_id': step_id,
            'path': path,
            'backend': backend,
            'well_id': well_id
        })
    except Exception as e:
        logger.error(f"Error queueing path for visualization: {e}")
```

### 3. Viewer Thread Modification

Update the viewer thread to load data from paths:

```python
def _process_queue(self):
    """Process data queue and update viewer."""
    while self.is_running:
        try:
            # Get next item from queue with timeout
            item = self.data_queue.get(timeout=0.1)
            
            if item is SHUTDOWN_SENTINEL:
                break
                
            # Extract visualization info
            step_id = item['step_id']
            path = item['path']
            backend = item['backend']
            well_id = item.get('well_id')
            
            # Load data using FileManager
            data = self.filemanager.load(path, backend)
            
            # Convert to format Napari can display
            display_data = self._prepare_data_for_display(data)
            
            # Update the viewer
            self._update_viewer(step_id, display_data, well_id)
            
        except queue.Empty:
            # No data in queue, continue loop
            continue
        except Exception as e:
            logger.error(f"Error in viewer thread: {e}")
```

### 4. Data Preparation

Maintain existing data preparation logic for converting various array types:

```python
def _prepare_data_for_display(self, data):
    """Convert data to format Napari can display."""
    # Handle different data types appropriately
    if hasattr(data, 'is_cuda') and data.is_cuda:  # PyTorch
        cpu_data = data.cpu().numpy()
    elif hasattr(data, 'get'):  # CuPy
        cpu_data = data.get()
    else:
        cpu_data = data
        
    # Additional processing...
    return cpu_data
```

## PipelineExecutor Integration

Update the PipelineExecutor to use the new visualization approach:

```python
def _visualize_step_output(
    step_id: str,
    context: ProcessingContext,
    visualizer: Any
) -> None:
    """
    Visualize step output using VFS paths.
    
    Args:
        step_id: ID of the step whose output to visualize
        context: The processing context
        visualizer: The visualizer to use
    """
    if not visualizer:
        return
        
    step_plan = context.step_plans.get(step_id, {})
    if not step_plan.get('visualize', False):
        return
        
    # Get output paths from step plan
    output_dir = step_plan.get('output_dir')
    if not output_dir:
        return
        
    # Get backend from step plan
    backend = step_plan.get('write_backend', 'disk')
    
    # Visualize using path and backend
    visualizer.visualize_path(
        step_id=step_id,
        path=output_dir,
        backend=backend,
        well_id=context.well_id
    )
```

## Implementation Sequence

1. Update `NapariStreamVisualizer` to accept a FileManager in `__init__`
2. Update visualization queue and processing to use paths instead of direct tensors
3. Create path resolution helper methods as needed
4. Update `PipelineExecutor._visualize_step_output` to pass paths instead of tensors
5. Update any code that instantiates the visualizer to pass a FileManager instance

## Benefits

1. Maintains VFS exclusivity (Clause 17)
2. Properly separates visualization from processing logic
3. Allows visualization to work even with the elimination of StepResult
4. Enables more flexible visualization scenarios (e.g., visualizing intermediate steps)
5. Consistent with TUI components that already use FileManager

## Testing Strategy

1. Unit test the FileManager loading in isolation
2. Integration test the path-based visualization flow
3. End-to-end test with pipeline execution