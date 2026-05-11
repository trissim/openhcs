"""
Converted from CellProfiler: RunImageJMacro
Original: RunImageJMacro.run

Note: This module executes external ImageJ macros which is fundamentally incompatible
with OpenHCS's pure functional approach. This conversion provides a best-effort
implementation that:
1. Saves input images to a temporary directory
2. Executes the ImageJ macro via subprocess
3. Loads output images back

This breaks the pure functional paradigm but maintains compatibility with existing
ImageJ macro workflows.
"""

import numpy as np
import os
import subprocess
import tempfile
import random
from typing import Tuple, List, Optional
from dataclasses import dataclass
from openhcs.core.memory.decorators import numpy
import skimage.io


@numpy
def run_imagej_macro(
    image: np.ndarray,
    executable_path: str = "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",
    macro_path: str = "macro.ijm",
    input_filenames: Optional[List[str]] = None,
    output_filenames: Optional[List[str]] = None,
    directory_variable: str = "Directory",
    macro_variables: Optional[dict] = None,
    debug_mode: bool = False,
) -> np.ndarray:
    """
    Execute an ImageJ macro on input images and return the results.
    
    This function exports images to a temporary folder, executes an ImageJ macro,
    and loads the resulting images back.
    
    Args:
        image: Input image(s) stacked along dimension 0. Shape (N, H, W) where N
               is the number of input images to send to the macro.
        executable_path: Full path to ImageJ/Fiji executable.
        macro_path: Full path to the macro file to execute.
        input_filenames: List of filenames to save input images as. Length must
                        match dimension 0 of input image. Defaults to ["input_0.tiff", ...].
        output_filenames: List of filenames to load as output. Defaults to ["output_0.tiff"].
        directory_variable: Variable name in macro that specifies the working directory.
        macro_variables: Dictionary of additional variables to pass to the macro.
        debug_mode: If True, temporary files are not deleted (for debugging).
    
    Returns:
        Output image(s) stacked along dimension 0. Shape (M, H, W) where M is
        the number of output images specified.
    """
    # Handle defaults
    if input_filenames is None:
        input_filenames = [f"input_{i}.tiff" for i in range(image.shape[0])]
    if output_filenames is None:
        output_filenames = ["output_0.tiff"]
    if macro_variables is None:
        macro_variables = {}
    
    # Validate input
    if len(input_filenames) != image.shape[0]:
        raise ValueError(
            f"Number of input filenames ({len(input_filenames)}) must match "
            f"number of input images ({image.shape[0]})"
        )
    
    # Create temporary directory
    tag = f"runimagejmacro_{random.randint(100000, 999999)}"
    tempdir = tempfile.mkdtemp(prefix=tag)
    
    try:
        # Save input images to temporary directory
        for i, filename in enumerate(input_filenames):
            img_slice = image[i]
            # Ensure proper dtype for saving
            if img_slice.dtype == np.float64 or img_slice.dtype == np.float32:
                # Normalize to 0-1 range if needed
                if img_slice.max() > 1.0:
                    img_slice = img_slice / img_slice.max()
            skimage.io.imsave(
                os.path.join(tempdir, filename),
                img_slice,
                check_contrast=False
            )
        
        # Build command
        cmd = [
            executable_path,
            "--headless",
            "console",
            "--run",
            macro_path
        ]
        
        # Build variable string for macro
        var_parts = [f"{directory_variable}='{tempdir}'"]
        for var_name, var_value in macro_variables.items():
            var_parts.append(f"{var_name}='{var_value}'")
        var_string = ", ".join(var_parts)
        cmd.append(var_string)
        
        # Execute macro
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Load output images
        output_images = []
        for filename in output_filenames:
            output_path = os.path.join(tempdir, filename)
            if not os.path.exists(output_path):
                # Parse error from ImageJ output
                reject = ('console:', 'Java Hot', 'at org', 'at java', '[WARNING]', '\t')
                err_lines = []
                for line in result.stdout.splitlines():
                    if len(line.strip()) > 0 and not line.startswith(reject):
                        if line not in err_lines:
                            err_lines.append(line)
                err_msg = "\n".join(err_lines)
                raise FileNotFoundError(
                    f"ImageJ macro did not produce expected output file: {filename}\n"
                    f"ImageJ output: {err_msg}"
                )
            
            output_img = skimage.io.imread(output_path)
            output_images.append(output_img.astype(np.float32))
        
        # Stack output images along dimension 0
        if len(output_images) == 1:
            result_array = output_images[0][np.newaxis, ...]
        else:
            result_array = np.stack(output_images, axis=0)
        
        return result_array
        
    finally:
        # Cleanup temporary directory unless debug mode
        if not debug_mode:
            try:
                for filename in os.listdir(tempdir):
                    os.remove(os.path.join(tempdir, filename))
                os.rmdir(tempdir)
            except Exception:
                pass  # Best effort cleanup