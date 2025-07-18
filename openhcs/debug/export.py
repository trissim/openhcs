"""
Debug export functionality for OpenHCS.

Extracted from TUI to make debug export reusable from CLI, tests, etc.
"""

import stat
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Import the pickle to python converter
try:
    from openhcs.debug.pickle_to_python import convert_pickle_to_python
except ImportError:
    # Fallback if import fails
    def convert_pickle_to_python(pickle_path, output_path):
        raise ImportError("pickle_to_python module not available")


def export_debug_data(subprocess_data: Dict[str, Any], output_path: Path,
                     data_file_path: str = None, log_file_path: str = None) -> Dict[str, Path]:
    """
    Export debug data to multiple file formats for manual debugging.

    Args:
        subprocess_data: The subprocess data dictionary to export
        output_path: Path where to save the debug files (should end with .pkl)
        data_file_path: Original data file path (optional)
        log_file_path: Original log file path (optional)

    Returns:
        Dict mapping file types to their paths: {'pkl': Path, 'sh': Path, 'cmd': Path, 'info': Path, 'py': Path}
    """
    
    # Ensure output_path is a Path object
    output_path = Path(output_path)
    
    # Save the subprocess data to pickle file
    import dill as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(subprocess_data, f)
    
    # Create companion files
    base_path = output_path.with_suffix('')
    
    # Find subprocess_runner.py (relative to this module)
    subprocess_script = Path(__file__).parent.parent / "textual_tui" / "subprocess_runner.py"
    
    # Save executable shell script
    shell_script = base_path.with_suffix('.sh')
    with open(shell_script, 'w') as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"# OpenHCS Subprocess Debug Script\n")
        f.write(f"# Generated: {datetime.now()}\n")
        f.write(f"# Plates: {subprocess_data['plate_paths']}\n\n")

        f.write(f"echo \"🔥 Starting OpenHCS subprocess debugging...\"\n")
        f.write(f"echo \"🔥 Pickle file: {output_path.name}\"\n")
        f.write(f"echo \"🔥 Press Ctrl+C to stop\"\n")
        f.write(f"echo \"\"\n\n")

        # Change to the directory containing the pickle file
        f.write(f"cd \"{output_path.parent}\"\n\n")

        # Run the subprocess with the exact filenames
        f.write(f"python \"{subprocess_script}\" \\\n")
        f.write(f"    \"{output_path.name}\" \\\n")
        f.write(f"    \"debug_status.json\" \\\n")
        f.write(f"    \"debug_result.json\" \\\n")
        f.write(f"    \"debug.log\"\n\n")

        f.write(f"echo \"\"\n")
        f.write(f"echo \"🔥 Subprocess finished. Check the files:\"\n")
        f.write(f"echo \"  - debug_status.json (progress/death markers)\"\n")
        f.write(f"echo \"  - debug_result.json (final results)\"\n")
        f.write(f"echo \"  - debug.log (detailed logs)\"\n")

    # Make shell script executable
    shell_script.chmod(shell_script.stat().st_mode | stat.S_IEXEC)

    # Save command file for reference
    command_file = base_path.with_suffix('.cmd')
    command = f"python {subprocess_script} {output_path.name} debug_status.json debug_result.json debug.log"

    with open(command_file, 'w') as f:
        f.write(f"# Manual subprocess debugging command\n")
        f.write(f"# Run this command to execute the subprocess manually:\n\n")
        f.write(f"cd \"{output_path.parent}\"\n")
        f.write(f"{command}\n\n")
        f.write(f"# Original files from TUI execution:\n")
        if data_file_path:
            f.write(f"# Data file: {data_file_path}\n")
        if log_file_path:
            f.write(f"# Log file: {log_file_path}\n")

    # Save info file
    info_file = base_path.with_suffix('.info')
    with open(info_file, 'w') as f:
        f.write(f"Debug Subprocess Data\n")
        f.write(f"====================\n\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Plates: {len(subprocess_data['plate_paths'])}\n")
        f.write(f"Plate paths: {subprocess_data['plate_paths']}\n\n")
        f.write(f"Pipeline data keys: {list(subprocess_data['pipeline_data'].keys())}\n\n")
        f.write(f"Global config: {subprocess_data['global_config_dict']}\n\n")
        f.write(f"To debug manually:\n")
        f.write(f"1. Run: ./{shell_script.name} (executable shell script)\n")
        f.write(f"2. Or run: {command}\n")
        f.write(f"3. Check debug_status.json for progress/death markers\n")
        f.write(f"4. Check debug_result.json for results\n")
        f.write(f"5. Check debug.log for detailed logs\n")

    # Generate Python script from pickle
    python_script = base_path.with_suffix('.py')
    try:
        convert_pickle_to_python(str(output_path), str(python_script))
    except Exception as e:
        # If conversion fails, create a simple placeholder
        with open(python_script, 'w') as f:
            f.write(f"# Failed to convert pickle to Python: {e}\n")
            f.write(f"# Use the pickle file directly instead\n")

    return {
        'pkl': output_path,
        'sh': shell_script,
        'cmd': command_file,
        'info': info_file,
        'py': python_script
    }
