"""Nominal CLI command family for CellProfiler/OpenHCS benchmarks."""

from __future__ import annotations

import argparse
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from openhcs.core.source_matching import is_image_path
from openhcs.core.source_schema_workspace import SourceSchemaImageSetSelection

from metaclass_registry import AutoRegisterMeta

from benchmark.datasets.cppipe_case_catalog import official_cp3_case_category
from benchmark.datasets.cache import (
    BenchmarkPathRootKind,
    CELLPROFILER_EXAMPLES_ROOT_ENV,
    resolve_benchmark_path_root,
)
from benchmark.runtime_env import configure_headless_cpu_benchmark_runtime


class BenchmarkCliCommand(ABC, metaclass=AutoRegisterMeta):
    """Registered benchmark CLI subcommand."""

    __registry_key__ = "command_name"
    __skip_if_no_key__ = True
    __registry__: ClassVar[dict[str, type["BenchmarkCliCommand"]]] = {}

    command_name: ClassVar[str | None] = None
    help_text: ClassVar[str]
    sort_order: ClassVar[int]

    @classmethod
    def registered_commands(cls) -> tuple["BenchmarkCliCommand", ...]:
        return tuple(
            command_type()
            for command_type in sorted(
                cls.__registry__.values(),
                key=lambda registered_type: registered_type.sort_order,
            )
        )

    @abstractmethod
    def configure(self, subparsers: argparse._SubParsersAction) -> None:
        """Register command-specific argparse schema."""

    @abstractmethod
    def run(self, args: argparse.Namespace) -> int:
        """Execute the parsed command."""

    def _parser(
        self, subparsers: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        if self.command_name is None:
            raise ValueError(f"{type(self).__name__} must declare command_name.")
        parser = subparsers.add_parser(self.command_name, help=self.help_text)
        parser.add_argument(
            "--log-level",
            default=os.environ.get("OPENHCS_BENCHMARK_LOG_LEVEL", "WARNING"),
            help="Python logging level for benchmark harness and OpenHCS runtime logs.",
        )
        parser.set_defaults(cli_command=self)
        return parser


class RunBenchmarkCommand(BenchmarkCliCommand):
    """Run benchmark cases and write complete benchmark artifacts."""

    command_name = "run"
    help_text = "Run benchmark cases."
    sort_order = 10

    def configure(self, subparsers: argparse._SubParsersAction) -> None:
        parser = self._parser(subparsers)
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument(
            "--case",
            action="append",
            dest="case_names",
            metavar="CASE_NAME",
            help=(
                "Run one exact benchmark case name from the manifest. "
                "Repeat for multiple cases. Defaults to all cases."
            ),
        )
        parser.add_argument("--native-reference-root", type=Path)
        parser.add_argument(
            "--require-native-reference",
            action="store_true",
            help=(
                "Fail or record a failed observation instead of running native "
                "CellProfiler when a case is missing from --native-reference-root."
            ),
        )
        parser.add_argument("--discard-openhcs-outputs", action="store_true")
        parser.add_argument("--continue-on-error", action="store_true")
        parser.add_argument(
            "--no-memory-metric",
            action="store_true",
            help=(
                "Collect execution-time metrics only. Use this for speed target "
                "runs where background RSS sampling would perturb Python runtime."
            ),
        )
        parser.add_argument("--suite-id")
        parser.add_argument("--repeats", type=int, default=1)
        parser.add_argument(
            "--speedup-target",
            type=float,
            default=5.0,
            help="Minimum acceptable OpenHCS speedup recorded in summary artifacts.",
        )
        parser.add_argument(
            "--force-openhcs-run",
            action="store_true",
            help="Disable OpenHCS benchmark/runtime execution cache reuse.",
        )
        parser.add_argument(
            "--source-schema-well",
            action="append",
            dest="source_schema_wells",
            default=None,
            help=(
                "Source-schema well/sample to execute. Repeat for multiple "
                "samples. Defaults to all samples."
            ),
        )
        parser.add_argument(
            "--source-schema-max-image-set-count",
            type=int,
            help="Execute only the first N source-schema samples.",
        )
        parser.add_argument(
            "--figures",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Generate benchmark figures from summary.csv after the run.",
        )
        parser.add_argument(
            "--figures-output-dir",
            type=Path,
            help="Directory for generated figures. Defaults to OUTPUT_DIR/figures.",
        )

    def run(self, args: argparse.Namespace) -> int:
        configure_headless_cpu_benchmark_runtime(args.log_level)
        from benchmark.cellprofiler_comparison import (
            ComparisonMetricPolicy,
            load_comparison_cases,
            run_comparison_suite,
        )

        suite_id = args.suite_id or datetime.now().strftime(
            "cp_vs_openhcs_%Y%m%d_%H%M%S"
        )
        cases = _filter_cases_by_name(
            load_comparison_cases(args.manifest),
            tuple(args.case_names or ()),
        )
        source_schema_image_set_selection = None
        if args.source_schema_wells is not None or (
            args.source_schema_max_image_set_count is not None
        ):
            source_schema_image_set_selection = SourceSchemaImageSetSelection(
                well_filter=tuple(args.source_schema_wells or ()),
                max_image_set_count=args.source_schema_max_image_set_count,
            )
        observations = run_comparison_suite(
            cases,
            output_root=args.output_dir,
            suite_id=suite_id,
            repeats=args.repeats,
            reuse_openhcs_cache=not args.force_openhcs_run,
            speedup_target=args.speedup_target,
            native_reference_root=args.native_reference_root,
            require_native_reference=args.require_native_reference,
            discard_openhcs_outputs=args.discard_openhcs_outputs,
            continue_on_error=args.continue_on_error,
            source_schema_image_set_selection=source_schema_image_set_selection,
            metric_policy=ComparisonMetricPolicy(
                collect_memory=not args.no_memory_metric,
            ),
            coverage_manifest_path=args.manifest,
        )
        print(f"suite_id={suite_id}")
        print(f"observations={len(observations)}")
        print(f"summary_csv={args.output_dir / 'summary.csv'}")
        print(
            "module_coverage_summary_json="
            f"{args.output_dir / 'module_coverage_summary.json'}"
        )
        if args.figures:
            figures_output_dir = args.figures_output_dir or args.output_dir / "figures"
            plot_summary(args.output_dir / "summary.csv", figures_output_dir)
            print(f"figures={figures_output_dir}")
        return 0


def _filter_cases_by_name(cases, requested_names: tuple[str, ...]):
    """Return manifest cases selected by exact name, failing on absent names."""
    if not requested_names:
        return cases

    available_names = tuple(case.name for case in cases)
    available_name_set = set(available_names)
    unknown_names = tuple(
        name
        for name in dict.fromkeys(requested_names)
        if name not in available_name_set
    )
    if unknown_names:
        raise ValueError(
            "Unknown benchmark case name(s): "
            f"{', '.join(unknown_names)}. Available case name(s): "
            f"{', '.join(available_names)}"
        )

    requested_name_set = set(requested_names)
    return tuple(case for case in cases if case.name in requested_name_set)


class OfficialCp3ManifestCommand(BenchmarkCliCommand):
    """Build a comparison manifest from local official CellProfiler examples."""

    command_name = "official-cp3-manifest"
    help_text = (
        "Build a comparison manifest from a local CellProfiler examples checkout."
    )
    sort_order = 20

    def configure(self, subparsers: argparse._SubParsersAction) -> None:
        parser = self._parser(subparsers)
        parser.add_argument(
            "--examples-root",
            type=Path,
            default=resolve_benchmark_path_root(
                BenchmarkPathRootKind.CELLPROFILER_EXAMPLES,
                env_name=CELLPROFILER_EXAMPLES_ROOT_ENV,
            ),
        )
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--value-only", action="store_true")
        parser.add_argument("--microscope-type")
        parser.add_argument("--cellprofiler-timeout-seconds", type=float)
        parser.add_argument(
            "--dataset-cache-root",
            type=Path,
            help=(
                "Benchmark dataset cache root used to materialize registry-backed "
                "cppipe cases. Defaults to the benchmark dataset cache root."
            ),
        )
        parser.add_argument(
            "--include-dataset-registry-cases",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Append case-bearing DatasetSpec entries from DATASET_REGISTRY to "
                "the official CP3 examples manifest."
            ),
        )

    def run(self, args: argparse.Namespace) -> int:
        configure_headless_cpu_benchmark_runtime(args.log_level)
        from benchmark.datasets.manifest import (
            cached_case_bearing_datasets,
            comparison_manifest_payload,
        )
        from benchmark.datasets.registry import DATASET_REGISTRY

        cppipe_dir = args.examples_root / "CellProfiler3Pipelines"
        if not cppipe_dir.is_dir():
            raise FileNotFoundError(
                f"CellProfiler3Pipelines directory not found: {cppipe_dir}"
            )
        cases = []
        for cppipe_path in sorted(cppipe_dir.glob("*.cppipe")):
            dataset_name = _official_cellprofiler3_source_name_for_pipeline(
                args.examples_root,
                cppipe_path.stem,
            )
            dataset_wrapper_path = args.examples_root / dataset_name
            dataset_path = _official_cellprofiler3_source_root(dataset_wrapper_path)
            resolved_cppipe_path = _official_cellprofiler3_cppipe_path(
                cppipe_path,
                dataset_wrapper_path,
            )
            case: dict[str, object] = {
                "name": cppipe_path.stem,
                "dataset_path": str(dataset_path),
                "cppipe_path": str(resolved_cppipe_path),
                "dataset_id": dataset_name,
                "value_only": args.value_only,
            }
            category = official_cp3_case_category(cppipe_path.stem)
            case["assay_category"] = category.assay
            case["module_category"] = category.module
            if args.microscope_type is not None:
                case["microscope_type"] = args.microscope_type
            if args.cellprofiler_timeout_seconds is not None:
                case["cellprofiler_timeout_seconds"] = args.cellprofiler_timeout_seconds
            cases.append(case)
        if args.include_dataset_registry_cases:
            registry_manifest = comparison_manifest_payload(
                cached_case_bearing_datasets(
                    DATASET_REGISTRY.values(),
                    cache_base=args.dataset_cache_root,
                )
            )
            cases.extend(registry_manifest["cases"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"cases": cases}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(str(args.output))
        print(f"cases={len(cases)}")
        return 0


class PlotBenchmarkCommand(BenchmarkCliCommand):
    """Plot benchmark CSV output."""

    command_name = "plot"
    help_text = "Plot benchmark CSV output."
    sort_order = 30

    def configure(self, subparsers: argparse._SubParsersAction) -> None:
        parser = self._parser(subparsers)
        parser.add_argument("--summary-csv", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)

    def run(self, args: argparse.Namespace) -> int:
        configure_headless_cpu_benchmark_runtime(args.log_level)
        plot_summary(args.summary_csv, args.output_dir)
        print(f"figures={args.output_dir}")
        return 0


class PlotWellThroughputPresentationCommand(BenchmarkCliCommand):
    """Plot the official30 parity/core/wells-per-core presentation pack."""

    command_name = "plot-well-throughput-presentation"
    help_text = "Plot the reusable well-throughput presentation figure pack."
    sort_order = 35

    def configure(self, subparsers: argparse._SubParsersAction) -> None:
        parser = self._parser(subparsers)
        parser.add_argument("--single-process-summary-csv", type=Path, required=True)
        parser.add_argument("--core-scaling-csv", type=Path, required=True)
        parser.add_argument("--wells-per-core-csv", type=Path, required=True)
        parser.add_argument(
            "--additional-wells-per-core-csv",
            type=Path,
            action="append",
            default=[],
        )
        parser.add_argument(
            "--module-coverage-semantic-families-csv",
            type=Path,
            help=(
                "Existing module_coverage_semantic_families.csv to include in "
                "the presentation pack."
            ),
        )
        parser.add_argument(
            "--module-coverage-manifest",
            type=Path,
            default=Path("benchmark/manifests/official30_portable_axis1.json"),
            help=(
                "Manifest used to generate module coverage artifacts when "
                "--module-coverage-semantic-families-csv is absent."
            ),
        )
        parser.add_argument("--output-dir", type=Path, required=True)

    def run(self, args: argparse.Namespace) -> int:
        configure_headless_cpu_benchmark_runtime(args.log_level)
        semantic_families_csv = args.module_coverage_semantic_families_csv
        if semantic_families_csv is None and args.module_coverage_manifest.exists():
            from benchmark.cellprofiler_comparison import (
                MODULE_COVERAGE_SEMANTIC_FAMILIES_CSV,
                write_module_coverage_artifacts,
            )

            write_module_coverage_artifacts(
                args.output_dir,
                manifest_path=args.module_coverage_manifest,
            )
            semantic_families_csv = (
                args.output_dir / MODULE_COVERAGE_SEMANTIC_FAMILIES_CSV
            )
        from benchmark.well_throughput_scaling import (
            WellThroughputPresentationReport,
            WellThroughputPresentationSources,
        )

        report = WellThroughputPresentationReport(
            sources=WellThroughputPresentationSources(
                single_process_summary_csv=args.single_process_summary_csv,
                core_scaling_csv=args.core_scaling_csv,
                wells_per_core_csv=args.wells_per_core_csv,
                additional_wells_per_core_csvs=tuple(
                    args.additional_wells_per_core_csv
                ),
                module_coverage_semantic_families_csv=semantic_families_csv,
            ),
            output_dir=args.output_dir,
        )
        outputs = report.generate()
        print(f"figures={args.output_dir}")
        print(f"outputs={len(outputs)}")
        return 0


class BioFormatsHcsValidationCommand(BenchmarkCliCommand):
    """Validate public Bio-Formats HCS sample datasets."""

    command_name = "bioformats-hcs-validate"
    help_text = "Download and validate public Bio-Formats HCS sample datasets."
    sort_order = 40

    def configure(self, subparsers: argparse._SubParsersAction) -> None:
        from benchmark.bioformats_hcs_validation import (
            bioformats_hcs_validation_specs,
        )

        parser = self._parser(subparsers)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument(
            "--dataset-cache-root",
            type=Path,
            help=(
                "Benchmark dataset cache root. Defaults to "
                "the benchmark dataset cache root."
            ),
        )
        parser.add_argument(
            "--dataset",
            action="append",
            choices=tuple(
                sorted(spec.id for spec in bioformats_hcs_validation_specs())
            ),
            help="Dataset id to validate. Repeat to select multiple datasets.",
        )
        parser.add_argument(
            "--max-size-mb",
            type=float,
            default=256.0,
            help="Skip catalog rows larger than this expected size.",
        )
        parser.add_argument(
            "--load-sample-count",
            type=int,
            default=1,
            help="Number of projected virtual planes to load per dataset.",
        )
        parser.add_argument("--continue-on-error", action="store_true")

    def run(self, args: argparse.Namespace) -> int:
        configure_headless_cpu_benchmark_runtime(args.log_level)
        from benchmark.bioformats_hcs_validation import (
            bioformats_hcs_validation_specs,
            validate_bioformats_hcs_catalog,
        )
        from benchmark.datasets.registry import DATASET_REGISTRY

        specs = (
            tuple(DATASET_REGISTRY[dataset_id] for dataset_id in args.dataset)
            if args.dataset
            else bioformats_hcs_validation_specs()
        )
        max_size_bytes = (
            None if args.max_size_mb is None else int(args.max_size_mb * 1024 * 1024)
        )
        outputs = validate_bioformats_hcs_catalog(
            specs,
            cache_base=args.dataset_cache_root,
            output_dir=args.output_dir,
            load_sample_count=args.load_sample_count,
            max_size_bytes=max_size_bytes,
            continue_on_error=args.continue_on_error,
        )
        print(f"datasets={len(outputs.results)}")
        print(f"passed={sum(result.status == 'passed' for result in outputs.results)}")
        print(f"summary_csv={outputs.summary_csv}")
        print(f"summary_json={outputs.summary_json}")
        return 0


def _official_cellprofiler3_cppipe_path(
    central_cppipe_path: Path,
    dataset_wrapper_path: Path,
) -> Path:
    """Prefer source-colocated pipelines when the example ships one."""
    colocated_cppipe_path = dataset_wrapper_path / central_cppipe_path.name
    if colocated_cppipe_path.exists():
        return colocated_cppipe_path
    return central_cppipe_path


def _official_cellprofiler3_source_root(dataset_wrapper_path: Path) -> Path:
    """Return the acquisition/source payload root for an official example."""
    images_path = dataset_wrapper_path / "images"
    if images_path.is_dir() and not _has_image_payloads(dataset_wrapper_path):
        return images_path
    return dataset_wrapper_path


def _has_image_payloads(path: Path) -> bool:
    return any(
        candidate.is_file() and is_image_path(str(candidate))
        for candidate in path.iterdir()
    )


def _official_cellprofiler3_source_name_for_pipeline(
    examples_root: Path,
    pipeline_name: str,
) -> str:
    candidate_names = (
        pipeline_name,
        pipeline_name.removesuffix("URL"),
        pipeline_name.split("_", maxsplit=1)[0],
        f"{pipeline_name}Images",
        pipeline_name.replace("ExampleUntangleAnd", "Example"),
    )
    for candidate_name in candidate_names:
        if candidate_name and (examples_root / candidate_name).exists():
            return candidate_name
    raise FileNotFoundError(
        f"No source directory found for official pipeline {pipeline_name!r} "
        f"under {examples_root}."
    )


def plot_summary(summary_csv: Path, output_dir: Path) -> None:
    """Create lab-meeting-ready runtime, speedup, parity, and memory figures."""
    from benchmark.reports.cppipe_figures import SummarySource
    from benchmark.reports.cppipe_figures import generate_cppipe_benchmark_figures

    generate_cppipe_benchmark_figures(
        (SummarySource("OH1", summary_csv),),
        output_dir=output_dir,
    )
