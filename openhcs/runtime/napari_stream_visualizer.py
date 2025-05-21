"""
Napari-based real-time visualization module for OpenHCS.

This module provides the NapariStreamVisualizer class for real-time
visualization of tensors during pipeline execution.

Doctrinal Clauses:
- Clause 65 — No Fallback Logic
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 368 — Visualization Must Be Observer-Only
"""

import logging
import queue
import threading
from typing import Any, Dict, Optional

import napari
import numpy as np

logger = logging.getLogger(__name__)

# Sentinel object to signal the viewer thread to shut down
SHUTDOWN_SENTINEL = object()

class NapariStreamVisualizer:
    """
    Manages a Napari viewer instance for real-time visualization of tensors
    streamed from the OpenHCS pipeline. Runs in a separate thread.
    """

    def __init__(self, viewer_title: str = "OpenHCS Real-Time Visualization"):
        self.viewer_title = viewer_title
        self.viewer: Optional[napari.Viewer] = None
        self.layers: Dict[str, napari.layers.Image] = {}
        self.data_queue = queue.Queue() # Thread-safe queue for tensor data
        self.viewer_thread: Optional[threading.Thread] = None
        self.is_running = False
        self._lock = threading.Lock()

        # Clause 368: Visualization must be observer-only.
        # This class will only read data and display it.

    def _initialize_viewer_in_thread(self):
        """
        Initializes and runs the Napari viewer event loop.
        This method is intended to be run in a separate thread.
        """
        try:
            logger.info("Napari viewer thread started.")
            # napari.gui_qt() ensures the Qt event loop is running if not already.
            # It's crucial for running napari in a non-blocking way from a script.
            with napari.gui_qt():
                self.viewer = napari.Viewer(title=self.viewer_title, show=True)
                self.is_running = True
                logger.info("Napari viewer initialized and shown.")

                while self.is_running:
                    try:
                        # Wait for data with a timeout to allow checking self.is_running
                        item = self.data_queue.get(timeout=0.1)
                        if item is SHUTDOWN_SENTINEL:
                            logger.info("Shutdown sentinel received. Exiting viewer loop.")
                            break
                        
                        layer_name, tensor_slice, metadata = item
                        self._update_layer_in_thread(layer_name, tensor_slice, metadata)
                        self.data_queue.task_done()
                    except queue.Empty:
                        continue # Timeout, check self.is_running again
                    except Exception as e:
                        logger.error(f"Error processing item in viewer thread: {e}", exc_info=True)
                        # Clause 65: Fail loudly (within the visualizer context) but don't crash pipeline.
            logger.info("Napari viewer event loop exited.")
        except Exception as e:
            logger.error(f"Fatal error in Napari viewer thread: {e}", exc_info=True)
        finally:
            self.is_running = False # Ensure flag is cleared
            if self.viewer:
                self.viewer.close()
            logger.info("Napari viewer thread finished.")


    def _update_layer_in_thread(self, layer_name: str, data: np.ndarray, metadata: Optional[Dict] = None):
        """
        Updates or creates a layer in the Napari viewer. Must be called from the viewer thread.
        """
        if not self.viewer:
            logger.warning("Viewer not initialized, cannot update layer.")
            return

        try:
            if layer_name in self.layers:
                self.layers[layer_name].data = data
                logger.debug(f"Updated layer: {layer_name} with shape {data.shape}")
            else:
                self.layers[layer_name] = self.viewer.add_image(data, name=layer_name)
                logger.info(f"Added new layer: {layer_name} with shape {data.shape}")
            
            if metadata and 'colormap' in metadata and self.layers[layer_name]:
                self.layers[layer_name].colormap = metadata['colormap']
            if metadata and 'contrast_limits' in metadata and self.layers[layer_name]:
                self.layers[layer_name].contrast_limits = metadata['contrast_limits']

        except Exception as e:
            logger.error(f"Error updating Napari layer '{layer_name}': {e}", exc_info=True)

    def start_viewer(self):
        """Starts the Napari viewer in a separate thread."""
        with self._lock:
            if self.is_running or self.viewer_thread is not None:
                logger.warning("Napari viewer is already running or starting.")
                return

            self.viewer_thread = threading.Thread(target=self._initialize_viewer_in_thread, daemon=True)
            self.viewer_thread.start()
            logger.info("NapariStreamVisualizer viewer thread initiated.")

    def push_tensor(self, step_id: str, tensor: Any, well_id: Optional[str] = None):
        """
        Receives a tensor from the pipeline executor, prepares it, and queues it for display.
        """
        if not self.is_running and self.viewer_thread is None:
            logger.info(f"First tensor received for step '{step_id}'. Starting Napari viewer.")
            self.start_viewer()
        
        if not self.is_running:
            logger.warning(f"Visualizer not running. Cannot push tensor for step '{step_id}'.")
            return

        try:
            # Explicit GPU to CPU conversion and slicing/projection.
            if hasattr(tensor, 'is_cuda') and tensor.is_cuda: # PyTorch
                cpu_tensor = tensor.cpu().numpy()
            elif hasattr(tensor, 'device') and 'cuda' in str(tensor.device).lower():
                if hasattr(tensor, 'get'): # CuPy
                    cpu_tensor = tensor.get()
                elif hasattr(tensor, 'numpy'): # JAX
                    cpu_tensor = np.asarray(tensor)
                else:
                    logger.warning(f"Unknown GPU tensor type for step '{step_id}'. Cannot convert.")
                    return
            elif isinstance(tensor, np.ndarray):
                cpu_tensor = tensor
            else:
                logger.warning(f"Unsupported tensor type for step '{step_id}': {type(tensor)}.")
                return

            if cpu_tensor.ndim == 3: # ZYX
                display_slice = cpu_tensor[cpu_tensor.shape[0] // 2, :, :]
            elif cpu_tensor.ndim == 2: # YX
                display_slice = cpu_tensor
            else:
                logger.warning(f"Tensor for step '{step_id}' has unsupported ndim: {cpu_tensor.ndim}.")
                return

            layer_name = f"{well_id}_{step_id}" if well_id else step_id
            metadata = {'colormap': 'gray'} 
            self.data_queue.put((layer_name, display_slice.copy(), metadata))
            logger.debug(f"Queued tensor slice for step '{step_id}' (layer: '{layer_name}').")
        except Exception as e:
            logger.error(f"Error preparing tensor from step '{step_id}' for visualization: {e}", exc_info=True)

    def stop_viewer(self):
        """Signals the viewer thread to shut down and waits for it to join."""
        logger.info("Attempting to stop Napari viewer...")
        with self._lock:
            if not self.is_running and self.viewer_thread is None:
                logger.info("Napari viewer was not running.")
                return
            if self.is_running:
                self.is_running = False
                self.data_queue.put(SHUTDOWN_SENTINEL)

        if self.viewer_thread and self.viewer_thread.is_alive():
            logger.info("Waiting for Napari viewer thread to join...")
            self.viewer_thread.join(timeout=5.0)
            if self.viewer_thread.is_alive():
                logger.warning("Napari viewer thread did not join in time.")
            else:
                logger.info("Napari viewer thread joined successfully.")
        self.viewer_thread = None
        logger.info("NapariStreamVisualizer stopped.")