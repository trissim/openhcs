#!/usr/bin/env python3
"""Convert one CellProfiler pipeline to canonical public OpenHCS source."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.interop.cellprofiler.pipeline_import import import_cellprofiler_pipeline


logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Translate one ``.cppipe`` and write its canonical FunctionStep source."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Convert a .cppipe to public OpenHCS FunctionStep source."
    )
    parser.add_argument("cppipe_file", type=Path, help="CellProfiler pipeline path")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path (default: <name>_openhcs.py)",
    )
    args = parser.parse_args(argv)
    output_path = args.output or (
        args.cppipe_file.parent / f"{args.cppipe_file.stem}_openhcs.py"
    )

    pipeline_steps, _pipeline_config = import_cellprofiler_pipeline(args.cppipe_file)
    source = FunctionStepTransportAuthority.source_from_pipeline(pipeline_steps)
    output_path.write_text(source, encoding="utf-8")
    logger.info(
        "Converted %s to %s public FunctionSteps at %s",
        args.cppipe_file,
        len(pipeline_steps),
        output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
