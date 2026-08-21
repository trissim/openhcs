#!/usr/bin/env python3
"""Pickle to Python Converter - OpenHCS pycodify integration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import dill as pickle

# Register OpenHCS-specific pycodify formatters.
import openhcs.serialization.pycodify_formatters  # noqa: F401

from pycodify import Assignment, BlankLine, CodeBlock, generate_python_source


def convert_pickle_to_python(
    pickle_path: str | Path,
    output_path: str | Path | None = None,
    clean_mode: bool = False,
) -> None:
    """Convert an OpenHCS debug pickle file to a runnable Python script."""
    pickle_file = Path(pickle_path)
    if not pickle_file.exists():
        print(f"Error: Pickle file not found: {pickle_path}")
        return

    if output_path is None:
        output_path = pickle_file.with_suffix(".py")

    print(f"Converting {pickle_file} to {output_path} (Clean Mode: {clean_mode})")

    try:
        with open(pickle_file, "rb") as handle:
            data = pickle.load(handle)

        with open(output_path, "w") as handle:
            handle.write("#!/usr/bin/env python3\n")
            handle.write('"""\n')
            handle.write(f"OpenHCS Pipeline Script - Generated from {pickle_file.name}\n")
            handle.write(f"Generated: {datetime.now()}\n")
            handle.write('"""\n\n')

            handle.write("import sys\n")
            handle.write("import os\n")
            handle.write("from pathlib import Path\n\n")
            handle.write("# Add OpenHCS to path\n")
            handle.write("sys.path.insert(0, \"/home/ts/code/projects/openhcs\")\n\n")

            handle.write("from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator\n")
            handle.write("from openhcs.core.steps.function_step import FunctionStep\n")
            handle.write(
                "from openhcs.core.config import (GlobalPipelineConfig, PathPlanningConfig, VFSConfig, ZarrConfig, \n"
                "                         MaterializationBackend, ZarrCompressor, ZarrChunkStrategy)\n"
            )
            handle.write("from openhcs.constants.constants import VariableComponents, Backend, Microscope\n\n")

            orchestrator_block = CodeBlock.from_items([
                Assignment("plate_paths", data.get("plate_paths", [])),
                BlankLine(),
                Assignment("global_config", data.get("global_config")),
                BlankLine(),
            ])

            if data.get("per_plate_configs"):
                orchestrator_block = CodeBlock.from_items([
                    *orchestrator_block.items,
                    Assignment("per_plate_configs", data.get("per_plate_configs")),
                    BlankLine(),
                ])
            elif data.get("pipeline_config") is not None:
                orchestrator_block = CodeBlock.from_items([
                    *orchestrator_block.items,
                    Assignment("pipeline_config", data.get("pipeline_config")),
                    BlankLine(),
                ])

            orchestrator_block = CodeBlock.from_items([
                *orchestrator_block.items,
                Assignment("pipeline_data", data.get("pipeline_data", {})),
            ])

            orchestrator_code = generate_python_source(
                orchestrator_block,
                header="# Edit this orchestrator configuration and save to apply changes",
                clean_mode=clean_mode,
            )

            handle.write(orchestrator_code)
            handle.write("\n\n")

            handle.write("def setup_signal_handlers():\n")
            handle.write('    """Setup signal handlers to kill all child processes and threads on Ctrl+C."""\n')
            handle.write("    import signal\n")
            handle.write("    import os\n")
            handle.write("    import sys\n\n")
            handle.write("    def cleanup_and_exit(signum, frame):\n")
            handle.write("        print(f\"\\n🔥 Signal {signum} received! Cleaning up all processes and threads...\")\n\n")
            handle.write("        os._exit(1)\n\n")
            handle.write("    signal.signal(signal.SIGINT, cleanup_and_exit)\n")
            handle.write("    signal.signal(signal.SIGTERM, cleanup_and_exit)\n\n")

            handle.write("def run_pipeline():\n")
            handle.write("    os.environ[\"OPENHCS_SUBPROCESS_MODE\"] = \"1\"\n")
            handle.write("    plate_paths, pipeline_data, global_config = create_pipeline()\n")
            handle.write("    for plate_path in plate_paths:\n")
            handle.write("        orchestrator = PipelineOrchestrator(plate_path)\n")
            handle.write("        orchestrator.initialize()\n")
            handle.write("        compilation_result = orchestrator.compile_pipelines(pipeline_data[plate_path])\n")
            handle.write("        execution_bundle = compilation_result[\"execution_bundle\"]\n")
            handle.write("        progress_context = {\n")
            handle.write("            \"execution_id\": f\"debug::{plate_path}\",\n")
            handle.write("            \"plate_id\": str(plate_path),\n")
            handle.write("            \"axis_id\": \"\",\n")
            handle.write("        }\n")
            handle.write("        orchestrator.execute_compiled_plate(\n")
            handle.write("            execution_bundle=execution_bundle,\n")
            handle.write("            max_workers=global_config.num_workers,\n")
            handle.write("            progress_queue=execution_bundle.runtime_environment.worker_start.multiprocessing_context().Queue(),\n")
            handle.write("            progress_context=progress_context,\n")
            handle.write("        )\n\n")

            handle.write("if __name__ == \"__main__\":\n")
            handle.write("    setup_signal_handlers()\n")
            handle.write("    run_pipeline()\n")

        print(f"✅ Successfully converted to {output_path}")
        print(f"You can now run: python {output_path}")

    except Exception as exc:
        print(f"Error converting pickle file: {exc}")
        import traceback

        traceback.print_exc()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert OpenHCS debug pickle files to runnable Python scripts."
    )
    parser.add_argument("pickle_file", help="Path to the input pickle file.")
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help="Path to the output Python script file (optional).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Generate a clean script with only non-default parameters.",
    )

    args = parser.parse_args()
    convert_pickle_to_python(args.pickle_file, args.output_file, clean_mode=args.clean)


if __name__ == "__main__":
    main()
