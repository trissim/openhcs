#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import importlib

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    core = importlib.import_module("experiments.embedding_collisions.core")
else:
    from . import core

audit_collision_geometry = core.audit_collision_geometry
budget_curve = core.budget_curve
build_fiber_summaries = core.build_fiber_summaries
collision_audit_payload = core.collision_audit_payload
export_lean_certificate = core.export_lean_certificate
fiber_sizes_from_summaries = core.fiber_sizes_from_summaries
fiber_weight_lists = core.fiber_weight_lists
scaled_sorted_fiber_weight_lists = core.scaled_sorted_fiber_weight_lists
hash_embed_texts = core.hash_embed_texts
histogram_from_summaries = core.histogram_from_summaries
load_records = core.load_records
maybe_make_plots = core.maybe_make_plots
quantize_embeddings = core.quantize_embeddings
quantized_keys = core.quantized_keys
save_json = core.save_json
sentence_transformer_embed = core.sentence_transformer_embed
summaries_to_frame = core.summaries_to_frame
synthetic_records_from_fiber_sizes = core.synthetic_records_from_fiber_sizes
transformers_mean_pool_embed = core.transformers_mean_pool_embed
write_curve_csv = core.write_curve_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit collision fibers in a deployed representation and generate finite "
            "rate-distortion style summaries that mirror the Lean theorems."
        )
    )
    parser.add_argument(
        "--input", type=Path, help="CSV, JSONL, or TXT file containing texts"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for outputs"
    )
    parser.add_argument(
        "--backend",
        choices=["hash", "sentence-transformers", "transformers"],
        default="hash",
        help="Embedding backend",
    )
    parser.add_argument("--model", default=None, help="Model name for learned backends")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--weight-column", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--hash-seed", type=int, default=0)
    parser.add_argument(
        "--quantization",
        choices=["exact", "round", "int8", "sign"],
        default="round",
    )
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--scale", type=float, default=64.0)
    parser.add_argument("--max-bits", type=int, default=12)
    parser.add_argument(
        "--synthetic-fibers",
        default=None,
        help="Comma-separated fiber sizes for an internal synthetic run when --input is omitted",
    )
    parser.add_argument(
        "--lean-certificate-path",
        type=Path,
        default=None,
        help="Optional path for a generated Lean certificate module",
    )
    parser.add_argument(
        "--lean-certificate-module",
        default="Paper1IT.Generated.EmbeddingCollisionAudit",
        help="Lean module name used in the generated certificate",
    )
    parser.add_argument(
        "--lean-weight-scale",
        type=int,
        default=1000000,
        help="Scale factor used when exporting weighted empirical masses to Lean",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input is not None:
        records = load_records(
            args.input,
            text_column=args.text_column,
            id_column=args.id_column,
            label_column=args.label_column,
            weight_column=args.weight_column,
        )
    else:
        fiber_sizes = [
            int(part)
            for part in (args.synthetic_fibers or "8,8,4,2,1,1").split(",")
            if part.strip()
        ]
        records = synthetic_records_from_fiber_sizes(fiber_sizes)

    embeddings = embed_records(records, args)
    quantized, quantization_label = quantize_embeddings(
        embeddings,
        mode=args.quantization,
        decimals=args.decimals,
        scale=args.scale,
    )
    keys = quantized_keys(quantized)

    fiber_summaries = build_fiber_summaries(records, keys)
    audit = audit_collision_geometry(fiber_summaries)
    curve = budget_curve(
        records=records,
        observation_count=audit.distinct_fibers,
        max_fiber_card=audit.max_fiber_card,
        fiber_weight_lists_value=fiber_weight_lists(records, keys),
        max_bits=args.max_bits,
    )
    histogram = histogram_from_summaries(fiber_summaries)

    payload = collision_audit_payload(
        audit=audit,
        embedding_backend=args.backend,
        quantization_label=quantization_label,
        model_name=args.model,
        input_path=None if args.input is None else str(args.input),
    )
    save_json(args.output_dir / "audit_summary.json", payload)
    write_curve_csv(args.output_dir / "budget_curve.csv", curve)
    write_curve_csv(args.output_dir / "fiber_histogram.csv", histogram)
    summaries_to_frame(fiber_summaries).to_csv(
        args.output_dir / "fiber_details.csv", index=False
    )
    plots_written = maybe_make_plots(args.output_dir, histogram, curve)
    lean_certificate_written = False
    if args.lean_certificate_path is not None:
        export_lean_certificate(
            args.lean_certificate_path,
            module_name=args.lean_certificate_module,
            fiber_sizes=fiber_sizes_from_summaries(fiber_summaries),
            scaled_fiber_weights_desc=scaled_sorted_fiber_weight_lists(
                records, keys, weight_scale=args.lean_weight_scale
            ),
            weight_scale=args.lean_weight_scale,
            audit=audit,
            curve=curve,
        )
        lean_certificate_written = True

    print("Collision audit complete")
    print(f"  items: {audit.item_count}")
    print(f"  fibers: {audit.distinct_fibers}")
    print(f"  max fiber A_pi: {audit.max_fiber_card}")
    print(f"  zero-error threshold bits: {audit.zero_error_threshold_bits}")
    print(f"  collided items: {audit.collided_items} ({audit.collision_rate:.2%})")
    print(f"  backend: {args.backend}")
    print(f"  quantization: {quantization_label}")
    print(f"  plots written: {'yes' if plots_written else 'no'}")
    print(f"  lean certificate written: {'yes' if lean_certificate_written else 'no'}")
    print(f"  output directory: {args.output_dir}")


def embed_records(records, args: argparse.Namespace):
    if args.backend == "hash":
        return hash_embed_texts(records, dim=args.hash_dim, seed=args.hash_seed)
    if args.backend == "sentence-transformers":
        model_name = args.model or "sentence-transformers/all-MiniLM-L6-v2"
        return sentence_transformer_embed(
            records, model_name=model_name, batch_size=args.batch_size
        )
    if args.backend == "transformers":
        model_name = args.model or "bert-base-uncased"
        return transformers_mean_pool_embed(
            records, model_name=model_name, batch_size=args.batch_size
        )
    raise ValueError(f"Unknown backend: {args.backend}")


if __name__ == "__main__":
    main()
