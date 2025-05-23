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

from openhcs.io.filemanager import FileManager

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

    def __init__(self, filemanager: FileManager, viewer_title: str = "OpenHCS Real-Time Visualization"):
        self.filemanager = filemanager # Added
        self.viewer_title = viewer_title
        self.viewer: Optional[napari.Viewer] = None
        self.layers: Dict[str, napari.layers.Image] = {} # Consider if layer type should be more generic
        self.data_queue = queue.Queue()
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

                        # New logic for path-based items:
                        if isinstance(item, dict) and item.get('type') == 'data_path':
                            step_id = item['step_id']
                            path = item['path']
                            backend = item['backend']
                            well_id = item.get('well_id') # Can be None

                            logger.debug(f"Processing path '{path}' for step '{step_id}' from queue.")
                            try:
                                # Load data using FileManager
                                loaded_data = self.filemanager.load(path, backend)
                                if loaded_data is not None:
                                    # Prepare data for display (includes GPU->CPU, slicing)
                                    display_data = self._prepare_data_for_display(loaded_data, step_id)

                                    if display_data is not None:
                                        layer_name = f"{well_id}_{step_id}" if well_id else step_id
                                        # Metadata might come from step_plan or be fixed for now
                                        metadata = {'colormap': 'gray'}
                                        self._update_layer_in_thread(layer_name, display_data, metadata)
                                    # else: (logging already in _prepare_data_for_display)
                                else:
                                    logger.warning(f"FileManager returned None for path '{path}', backend '{backend}' (step '{step_id}').")
                            except Exception as e_load:
                                logger.error(f"Error loading or preparing data for step '{step_id}', path '{path}': {e_load}", exc_info=True)
                        else:
                            logger.warning(f"Unknown item type in data queue: {type(item)}. Item: {item}")

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

    def _prepare_data_for_display(self, data: Any, step_id_for_log: str) -> Optional[np.ndarray]:
        """Converts loaded data to a displayable NumPy array (e.g., 2D slice)."""
        cpu_tensor: Optional[np.ndarray] = None
        try:
            # GPU to CPU conversion logic
            if hasattr(data, 'is_cuda') and data.is_cuda: # PyTorch
                cpu_tensor = data.cpu().numpy()
            elif hasattr(data, 'device') and 'cuda' in str(data.device).lower(): # Check for device attribute
                if hasattr(data, 'get'): # CuPy
                    cpu_tensor = data.get()
                elif hasattr(data, 'numpy'): # JAX on GPU might have .numpy() after host transfer
                    cpu_tensor = np.asarray(data) # JAX arrays might need explicit conversion
                else: # Fallback for other GPU array types if possible
                    logger.warning(f"Unknown GPU array type for step '{step_id_for_log}'. Attempting .numpy().")
                    if hasattr(data, 'numpy'):
                        cpu_tensor = data.numpy()
                    else:
                        logger.error(f"Cannot convert GPU tensor of type {type(data)} for step '{step_id_for_log}'.")
                        return None
            elif isinstance(data, np.ndarray):
                cpu_tensor = data
            else:
                # Attempt to convert to numpy array if it's some other array-like structure
                try:
                    cpu_tensor = np.asarray(data)
                    logger.debug(f"Converted data of type {type(data)} to numpy array for step '{step_id_for_log}'.")
                except Exception as e_conv:
                    logger.warning(f"Unsupported data type for step '{step_id_for_log}': {type(data)}. Error: {e_conv}")
                    return None

            if cpu_tensor is None: # Should not happen if logic above is correct
                return None

            # Slicing logic
            display_slice: Optional[np.ndarray] = None
            if cpu_tensor.ndim == 3: # ZYX
                display_slice = cpu_tensor[cpu_tensor.shape[0] // 2, :, :]
            elif cpu_tensor.ndim == 2: # YX
                display_slice = cpu_tensor
            elif cpu_tensor.ndim > 3: # e.g. CZYX or TZYX
                logger.warning(f"Tensor for step '{step_id_for_log}' has ndim > 3 ({cpu_tensor.ndim}). Taking a default slice.")
                slicer = [0] * (cpu_tensor.ndim - 2) # Slice first channels/times
                slicer[-1] = cpu_tensor.shape[-3] // 2 # Middle Z
                try:
                    display_slice = cpu_tensor[tuple(slicer)]
                except IndexError: # Handle cases where slicing might fail (e.g. very small dimensions)
                    logger.error(f"Slicing failed for tensor with shape {cpu_tensor.shape} for step '{step_id_for_log}'.", exc_info=True)
                    display_slice = None
            else:
                logger.warning(f"Tensor for step '{step_id_for_log}' has unsupported ndim for display: {cpu_tensor.ndim}.")
                return None

            return display_slice.copy() if display_slice is not None else None

        except Exception as e:
            logger.error(f"Error preparing data from step '{step_id_for_log}' for display: {e}", exc_info=True)
            return None

    def visualize_path(self, step_id: str, path: str, backend: str, well_id: Optional[str] = None):
        """
        Receives a VFS path, backend, and associated info, and queues it for display.
        """
        if not self.is_running and self.viewer_thread is None:
            logger.info(f"Visualizer not running for step '{step_id}'. Starting Napari viewer.")
            self.start_viewer()

        if not self.viewer_thread: # Check if thread actually started
            logger.warning(f"Visualizer thread not available. Cannot visualize path for step '{step_id}'.")
            return

        try:
            item_to_queue = {
                'type': 'data_path', # To distinguish from other potential queue items
                'step_id': step_id,
                'path': path,
                'backend': backend,
                'well_id': well_id
            }
            self.data_queue.put(item_to_queue)
            logger.debug(f"Queued path '{path}' for step '{step_id}' (well: {well_id}).")
        except Exception as e:
            logger.error(f"Error queueing path for visualization: {e}", exc_info=True)

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