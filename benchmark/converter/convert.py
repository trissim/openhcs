#!/usr/bin/env python3
"""
CellProfiler → OpenHCS Converter

Converts .cppipe files to OpenHCS pipelines using absorbed library.
Requires library to be absorbed first via:
    python -m benchmark.converter.absorb

Usage:
    python -m benchmark.converter.convert <cppipe_file>

If a module is not absorbed, conversion FAILS. No fallback. Absorb first.
"""

import argparse
import logging
import sys
from pathlib import Path

from .runtime_pipeline import CPPipePipelineGenerationRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert .cppipe to OpenHCS pipeline using absorbed library"
    )
    parser.add_argument(
        "cppipe_file",
        type=Path,
        help="Path to .cppipe file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path (default: <name>_openhcs.py)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.cppipe_file.exists():
        logger.error(f"File not found: {args.cppipe_file}")
        sys.exit(1)

    # Default output path
    if args.output is None:
        args.output = args.cppipe_file.parent / f"{args.cppipe_file.stem}_openhcs.py"

    logger.info(f"Converting: {args.cppipe_file}")

    conversion = CPPipePipelineGenerationRequest(
        cppipe_path=args.cppipe_file,
    ).generate()
    logger.info(f"Parsed {len(conversion.modules)} modules")

    for m in conversion.modules:
        logger.info(f"  - {m.name}")

    if conversion.infrastructure_modules:
        logger.info(
            "Skipping %d infrastructure modules:",
            len(conversion.infrastructure_modules),
        )
        for m in conversion.infrastructure_modules:
            logger.info(f"  - {m.name} (handled by OpenHCS infrastructure)")

    conversion.generated_pipeline.save(args.output)

    # Summary
    logger.info("=" * 50)
    logger.info(f"Pipeline: {conversion.generated_pipeline.name}")
    logger.info(f"Modules: {len(conversion.generated_pipeline.converted_modules)}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
