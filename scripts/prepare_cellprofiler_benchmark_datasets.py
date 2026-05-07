"""Acquire benchmark datasets and generate CellProfiler comparison manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.datasets.acquire import acquire_dataset
from benchmark.datasets.manifest import comparison_manifest_payload, write_comparison_manifest
from benchmark.datasets.registry import DATASET_REGISTRY, get_dataset_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List known benchmark datasets.")
    list_parser.add_argument(
        "--with-cases",
        action="store_true",
        help="Show dataset-relative benchmark cases.",
    )

    acquire_parser = subparsers.add_parser("acquire", help="Download/setup datasets.")
    _add_dataset_filters(acquire_parser)
    acquire_parser.add_argument("--cache-root", type=Path, default=None)

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Download/setup datasets and write a comparison manifest.",
    )
    _add_dataset_filters(manifest_parser)
    manifest_parser.add_argument("--output", required=True, type=Path)
    manifest_parser.add_argument("--cache-root", type=Path, default=None)
    manifest_parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Include only this benchmark case name. May be repeated.",
    )

    args = parser.parse_args()
    if args.command == "list":
        _list_datasets(with_cases=args.with_cases)
        return

    specs = _selected_specs(args.dataset_id, args.max_size_gb, args.only_with_cases)
    if args.command == "acquire":
        for spec in specs:
            acquired = acquire_dataset(spec, cache_base=args.cache_root)
            print(f"{spec.id}\t{acquired.path}\t{acquired.image_count} images")
        return

    acquired_pairs = [
        (spec, acquire_dataset(spec, cache_base=args.cache_root)) for spec in specs
    ]
    payload = comparison_manifest_payload(
        acquired_pairs,
        case_names=set(args.case) if args.case else None,
    )
    write_comparison_manifest(payload, args.output)
    print(f"Wrote {len(payload['cases'])} cases to {args.output}")


def _add_dataset_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=None,
        help="Dataset id to acquire. May be repeated. Defaults to case-bearing datasets.",
    )
    parser.add_argument(
        "--max-size-gb",
        type=float,
        default=None,
        help="Skip datasets whose declared size exceeds this limit.",
    )
    parser.add_argument(
        "--only-with-cases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict default selection to datasets with benchmark cases.",
    )


def _selected_specs(
    dataset_ids: list[str] | None,
    max_size_gb: float | None,
    only_with_cases: bool,
):
    specs = (
        [get_dataset_spec(dataset_id) for dataset_id in dataset_ids]
        if dataset_ids
        else list(DATASET_REGISTRY.values())
    )
    if only_with_cases:
        specs = [spec for spec in specs if spec.benchmark_cases]
    if max_size_gb is not None:
        max_size_bytes = int(max_size_gb * 1_000_000_000)
        specs = [spec for spec in specs if spec.size_bytes <= max_size_bytes]
    return specs


def _list_datasets(*, with_cases: bool) -> None:
    for spec in DATASET_REGISTRY.values():
        case_count = len(spec.benchmark_cases)
        source = spec.acquisition_source()
        print(
            f"{spec.id}\t{spec.size_bytes / 1_000_000_000:.3f} GB\t"
            f"{source.kind.value}\t{case_count} cases"
        )
        if with_cases:
            for case in spec.benchmark_cases:
                print(f"  {case.name}\t{case.cppipe_path}\t{case.dataset_path}")


if __name__ == "__main__":
    main()
