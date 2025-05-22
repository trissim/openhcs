# Module Integration and TUI Plan for OpenHCS Refactoring

**Date:** 2025-05-20

**Version:** 2.8 (Modules & TUI Align with `stack_utils` Handling Image (De)serialization - Full Content)

## I. Identified Areas for Refactoring ("Rot") - Specific Modules

### 1. Deep Learning Function Integration and Refinements (`openhcs/processing/backends/`)
*   **General Requirement:** Ensure all core DL processing functions are usable within `FunctionStep`.
    *   Functions are decorated appropriately (e.g., `@torch_func`, `@special_input(SpecialKey.MY_DATA)`, `@special_output(SpecialKey.MY_RESULT)`). These decorators take *only* the `SpecialKey` and imply mandatory VFS-linked raw data transfer.
    *   The `step_plan` provides all execution parameters.
    *   `FunctionStep.process` (via its helper `_process_single_pattern_vfs` as detailed in [`PLANS_CORE_PIPELINE_VFS.md`](plans/PLANS_CORE_PIPELINE_VFS.md:1) v3.0):
        *   Uses `FileManager.open(path, backend)` to retrieve data. When using disk backend for first read or final write, serialization/deserialization is handled by the backend.
        *   For primary input: FileManager retrieves data which `FunctionStep` passes to `stack_slices`. `stack_slices` performs type conversion and stacking of array-like objects (no serialization involved).
        *   For `@special_input(SpecialKey.KEY)`, it passes the data obtained from `FileManager.open()` directly as a *positional argument* to the user function.
        *   Passes regular `kwargs` from the `step_plan` as keyword arguments to the user function.
        *   If the user function has `@special_output(SpecialKey.KEY)`, its direct return value is passed to `FileManager.save(path, output_data, backend)`.
        *   If no `@special_output`, the user function returns a 3D typed array. `_process_single_pattern_vfs` calls `unstack_slices` which performs type conversion and unstacking to produce array-like objects (not bytes). These are then passed to `FileManager.save()` for storage.
    *   Decorator metadata simply lists `SpecialKey` names. The format/content of the raw special data is a contract between producing/consuming functions.
*   **Specific File Actions & How-To Details:**
    *   **[`self_supervised_3d_deconvolution.py`](openhcs/processing/backends/enhance/self_supervised_3d_deconvolution.py:1):**
        *   **Action:** Add `@torch_func`. Refactor its `**kwargs` into clear, named keyword arguments. If it needs VFS-linked data beyond the primary stack (e.g., a pre-trained model), it should be decorated with `@special_input(SpecialKey.YOUR_KEY_NAME)` and its function signature modified to accept raw bytes for that positional argument, deserializing these bytes internally if needed.
    *   **[`n2v2_processor_torch.py`](openhcs/processing/backends/enhance/n2v2_processor_torch.py:1):**
        *   **Action & How-To:**
            1.  **Decorators & Signature Example:**
                ```python
                # from openhcs.core.pipeline.function_contracts import special_input, special_output
                # from openhcs.constants.constants import SpecialKey 
                # import torch 
                # import io # For BytesIO
                # from typing import Optional, Dict, Any # For type hinting
                #
                # @torch_func
                # @special_input(SpecialKey.PRETRAINED_MODEL) 
                # @special_output(SpecialKey.TRAINED_MODEL)  
                # def n2v2_denoise_torch(
                #     image_stack: torch.Tensor, # Primary 3D input (deserialized and typed by stack_slices)
                #     pretrained_model_bytes: Optional[bytes] = None, # Positional arg, receives raw bytes.
                #     epochs: int = 10, 
                #     learning_rate: float = 0.001
                # ) -> bytes: # Returns raw bytes for SpecialKey.TRAINED_MODEL
                #     model = YourN2V2ModelClass() 
                #     device = image_stack.device 
                #     model.to(device)
                #
                #     if pretrained_model_bytes:
                #         state_dict = torch.load(io.BytesIO(pretrained_model_bytes), map_location=device)
                #         model.load_state_dict(state_dict)
                #     
                #     # ... training logic ...
                #
                #     buffer = io.BytesIO()
                #     torch.save(model.state_dict(), buffer)
                #     return buffer.getvalue() # Return raw serialized bytes
                ```
            2.  **`FunctionStep` Interaction:**
                *   `_process_single_pattern_vfs`:
                    *   Gets `list_of_raw_image_file_bytes` from `FileManager.open()`.
                    *   `stacked_3d_array = stack_slices(list_of_raw_image_file_bytes, ...)` (stack_slices deserializes + stacks).
                    *   Loads `raw_pretrained_model_bytes` for `PRETRAINED_MODEL` from VFS.
                    *   Calls: `n2v2_denoise_torch(stacked_3d_array, raw_pretrained_model_bytes, epochs=10, ...)`.
                    *   Takes the returned `raw_trained_model_bytes` and saves them to VFS via `FileManager.save()`.
    *   **[`dxf_mask_pipeline.py`](openhcs/processing/backends/analysis/dxf_mask_pipeline.py:1):**
        *   **Action & How-To:**
            1.  **Decorators & Signature Example:**
                ```python
                # from openhcs.core.pipeline.function_contracts import special_input
                # from openhcs.constants.constants import SpecialKey
                # import torch
                # import io 
                # from typing import Optional, Dict
                #
                # @torch_func 
                # @special_input(SpecialKey.DXF_FILE) 
                # @special_input(SpecialKey.CNN_MODEL_WEIGHTS) 
                # def dxf_mask_pipeline(
                #     image_stack: torch.Tensor, 
                #     dxf_file_raw_bytes: bytes, 
                #     cnn_model_weights_raw_bytes: Optional[bytes] = None,
                #     threshold_value: float = 0.5 
                # ) -> torch.Tensor: # Returns primary 3D mask (a typed array)
                #   dxf_file_content = dxf_file_raw_bytes.decode('utf-8') 
                #   
                #   registration_cnn = YourCNNModelClass() 
                #   if cnn_model_weights_raw_bytes: 
                #     cnn_model_state_dict = torch.load(io.BytesIO(cnn_model_weights_raw_bytes)) 
                #     registration_cnn.load_state_dict(cnn_model_state_dict)
                #   
                #   dxf_polygons = parse_dxf_content_to_polygons(dxf_file_content) 
                #   # ...
                #   return output_mask_stack 
                ```
            2.  **`FunctionStep` Interaction (Output):** `_process_single_pattern_vfs` receives `output_mask_stack` (typed array). It calls `list_of_raw_bytes = unstack_slices(output_mask_stack, ...)`. `unstack_slices` serializes each 2D slice to bytes. Then `_process_single_pattern_vfs` saves each item in `list_of_raw_bytes` using `FileManager.save()`.
    *   **[`self_supervised_stitcher.py`](openhcs/processing/backends/self_supervised_stitcher.py:1):**
        *   Adjustments: `@special_input` for models means the function receives raw bytes positionally. If it has an `@special_output` (e.g., for a positions dictionary), it must return raw, serialized bytes.
            ```python
            # import pickle 
            # from openhcs.core.pipeline.function_contracts import special_output
            # from openhcs.constants.constants import SpecialKey
            # from typing import Dict
            #
            # @special_output(SpecialKey.STITCHED_POSITIONS)
            # def self_supervised_stitcher_func(...) -> bytes: 
            #    # ...
            #    positions_dict: Dict = {...}
            #    return pickle.dumps(positions_dict) # User function serializes
            ```

### 2. `OperaPhenixXmlParser` VFS Compliance (`openhcs/microscopes/opera_phenix.py`)
*   **Action & How-To (Option 1 - Preferred):**
    ```python
    # class OperaPhenixXmlParser:
    #   def __init__(self, xml_content_str: str): self.xml_content_str = xml_content_str 
    #   def get_field_id_mapping(self) -> Dict[str, int]:
    #       import xml.etree.ElementTree as ET 
    #       root = ET.fromstring(self.xml_content_str); # ... parse ...
    
    # In OperaPhenixMetadataHandler:
    #   xml_path_str = str(self.find_metadata_file(plate_path)) 
    #   raw_xml_data = self.filemanager.open(xml_path_str, Backend.DISK.value) 
    #   xml_content_str = raw_xml_data.decode('utf-8') if isinstance(raw_xml_data, bytes) else raw_xml_data
    #   xml_parser = OperaPhenixXmlParser(xml_content_str=xml_content_str)
    ```

### 3. `pattern_resolver.py` VFS Compliance (`openhcs/formats/pattern/pattern_resolver.py`)
*   **Action & How-To:**
    *   `convert_filename_pattern` is pure string manipulation.
    *   **Revised Signature:**
        ```python
        # def convert_filename_pattern(filename_pattern: str, **kwargs_for_placeholders) -> str:
        #   # ... logic ...
        #   return resolved_pattern_string 
        ```

### 4. `PatternPath.is_fully_instantiated()` Validation in `PatternDiscoveryEngine` (`openhcs/formats/pattern/pattern_discovery.py`)
*   Remove premature validation as per previous versions of this plan.

### 5. Function Linkage and `ImageProcessorInterface` Adherence
*   Ensure processing functions align with the updated data flow (typed arrays in, typed arrays or raw special bytes out).

### 6. `ezstitcher` Naming Discrepancy
*   Consider renaming for consistency if it's a core module.

### 7. Duplicate `create_microscope_handler` Factory
*   Consolidate as per previous versions of this plan.

## II. Well-Structured ("Solved") Areas (Positive Controls) - Modules
*   `FileManager` is a pure raw byte I/O layer.
*   `stack_slices` and `unstack_slices` are now responsible for image format (de)serialization for primary image data paths, in addition to their stacking/type conversion roles.

## III. Architectural Insights and Considerations - Modules
*   This model centralizes image format (de)serialization within `stack_utils.py` for the primary pipeline flow, keeping `FileManager` generic. User functions handle (de)serialization for their specific `SpecialKey` raw data blobs.

## IV. Next Steps in Analysis - Modules & TUI

1.  **Microscope Handler and Pattern Detection VFS Compliance Verification:**
    *   Ensure all metadata file reading uses `FileManager.open()` and subsequent decoding by the caller.

2.  **Remaining `openhcs/processing/` Audit:**
    *   Review all functions for alignment with the raw data model for `SpecialKey` I/O and the (de)serialization roles of `stack_utils`.
    *   **Consideration for `trace_neurites_rrs_vectorized`:**
        *   **How-To:** If `@special_output(SpecialKey.NEURITE_TRACES)`, its return must be raw bytes (e.g., `json.dumps(dict_to_save).encode('utf-8')`). `_process_single_pattern_vfs` saves these bytes.

3.  **TUI Code Audit (`openhcs/tui/`):**
    *   **`FunctionPatternEditor.__init__` and `_edit_in_vim`:**
        *   `_edit_in_vim` `FileManager` usage for temp files (saving/loading text) is consistent.
    *   **`MenuBar` Loading Logic:** Consistent for its `menu.yaml` loading.

## V. TUI Design Considerations and Alignment with Core Refactoring (Focus on `openhcs/tui/function_pattern_editor.py`)
*   The TUI implementation in [`openhcs/tui/function_pattern_editor.py`](openhcs/tui/function_pattern_editor.py:1) (specifically methods like `_create_parameter_editor`) must be updated to:
    *   Use `inspect.signature()` and check `func.__special_inputs__` / `func.__special_output__` (attributes set by decorators).
    *   For parameters identified as `SpecialKey` inputs: Render UI for VFS path linking (for mandatory raw data input).
    *   For `SpecialKey` outputs: Render UI for VFS path definition (for raw data output).
    *   For all other parameters: Render UI for direct value input (these are `kwargs`).
*   This ensures the TUI accurately reflects that `SpecialKey` I/O is about VFS paths for raw data transfer, and `kwargs` are for direct behavioral settings.

---
*This is a working document and will be updated as the audit and planning progresses.*