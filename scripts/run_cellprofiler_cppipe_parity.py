#!/usr/bin/env python
"""Run CellProfiler reference versus converted OpenHCS parity for one .cppipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.adapters.openhcs import OpenHCSAdapter
from benchmark.metrics.time import TimeMetric
from benchmark.runner import run_cellprofiler_cppipe_parity


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run or reuse native CellProfiler output, then run or reuse the "
            "same local .cppipe via OpenHCS and require semantic output parity."
        )
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--cppipe-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("benchmark_outputs"))
    parser.add_argument("--dataset-id")
    parser.add_argument("--pipeline-name")
    parser.add_argument("--microscope-type")
    parser.add_argument("--cellprofiler-timeout-seconds", type=float)
    parser.add_argument(
        "--equivalence-reference-output-dir",
        type=Path,
        help=(
            "Reuse an existing native CellProfiler output directory instead of "
            "running CellProfiler."
        ),
    )
    parser.add_argument(
        "--force-openhcs-run",
        action="store_true",
        help="Ignore any valid cached OpenHCS output and execute OpenHCS again.",
    )
    parser.add_argument(
        "--value-only",
        action="store_true",
        help="Compare semantic measurement/table parity only; skip image pixels.",
    )
    parser.add_argument(
        "--openhcs-runtime-only",
        action="store_true",
        help=(
            "Run only the converted OpenHCS pipeline. Skip native CellProfiler, "
            "semantic snapshot materialization, equivalence comparison, and "
            "benchmark runtime-execution cache writes."
        ),
    )
    args = parser.parse_args()

    pipeline_params = {}
    if args.cellprofiler_timeout_seconds is not None:
        pipeline_params["cellprofiler_timeout_seconds"] = (
            args.cellprofiler_timeout_seconds
        )
    if args.value_only:
        pipeline_params["compare_image_outputs"] = False
    if args.openhcs_runtime_only:
        if args.equivalence_reference_output_dir is not None:
            parser.error(
                "--openhcs-runtime-only cannot be combined with "
                "--equivalence-reference-output-dir"
            )
        resolved_dataset_id = args.dataset_id or args.dataset_path.name
        resolved_pipeline_name = args.pipeline_name or args.cppipe_path.stem
        run_slug = "".join(
            char if char.isalnum() or char in "._-" else "_"
            for char in f"{resolved_dataset_id}_{resolved_pipeline_name}"
        )
        result = OpenHCSAdapter().run(
            dataset_path=args.dataset_path,
            pipeline_name=resolved_pipeline_name,
            pipeline_params={
                **pipeline_params,
                "dataset_id": resolved_dataset_id,
                "cppipe_path": str(args.cppipe_path),
                **(
                    {"microscope_type": args.microscope_type}
                    if args.microscope_type is not None
                    else {}
                ),
            },
            metrics=[TimeMetric()],
            output_dir=args.output_root / f"OpenHCS_{run_slug}",
        )
        print(f"success={result.success}")
        print(f"openhcs_output={result.output_path}")
        print(f"metrics={result.metrics}")
        print(
            "equivalence_difference_count="
            f"{(result.provenance or {}).get('equivalence_difference_count')}"
        )
        return 0 if result.success else 1

    result = run_cellprofiler_cppipe_parity(
        args.dataset_path,
        args.cppipe_path,
        metrics=[TimeMetric()],
        dataset_id=args.dataset_id,
        pipeline_name=args.pipeline_name,
        microscope_type=args.microscope_type,
        pipeline_params=pipeline_params,
        output_root=args.output_root,
        equivalence_reference_output_dir=args.equivalence_reference_output_dir,
        reuse_openhcs_cache=not args.force_openhcs_run,
    )

    print(f"equivalent={result.is_equivalent}")
    print(f"native_output={result.native_cellprofiler.output_path}")
    print(f"openhcs_output={result.openhcs_converted.output_path}")
    print(
        "native_cached="
        f"{bool((result.native_cellprofiler.provenance or {}).get('reused_reference_output'))}"
    )
    print(
        "openhcs_cached="
        f"{bool((result.openhcs_converted.provenance or {}).get('reused_cached_output'))}"
    )
    return 0 if result.is_equivalent else 1


if __name__ == "__main__":
    raise SystemExit(main())
