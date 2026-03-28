#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    core = importlib.import_module("experiments.embedding_collisions.core")
else:
    from . import core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate experiment-derived assets for paper1_jsait."
    )
    parser.add_argument("--paper-id", default="paper1_jsait")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    paper_dir = repo_root / "docs" / "papers" / "paper1_typing_discipline"
    latex_dir = paper_dir / "latex_jsait"
    content_dir = latex_dir / "content"
    out_dir = paper_dir / "experiments" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    reviewer_records = core.load_records(
        repo_root
        / "experiments"
        / "embedding_collisions"
        / "data"
        / "reviewer_demo_texts.csv",
        text_column="text",
        id_column="id",
        label_column="label",
        weight_column="weight",
    )
    semantic_records = build_semantic_grid_records()

    reviewer_embeddings = core.sentence_transformer_embed(
        reviewer_records, model_name=args.model, batch_size=args.batch_size
    )
    semantic_embeddings = core.sentence_transformer_embed(
        semantic_records, model_name=args.model, batch_size=args.batch_size
    )

    reviewer = run_regime(
        name="no_collapse",
        records=reviewer_records,
        embeddings=reviewer_embeddings,
        quantization_mode="round",
        decimals=2,
        scale=64.0,
        max_bits=8,
        certificate_path=paper_dir
        / "proofs"
        / "Paper1IT"
        / "Generated"
        / "ReviewerDemoCertificate.lean",
        certificate_module="Paper1IT.Generated.ReviewerDemoCertificate",
        output_dir=out_dir / "reviewer_demo",
    )
    partial = run_regime(
        name="partial_collapse",
        records=semantic_records,
        embeddings=semantic_embeddings,
        quantization_mode="int8",
        decimals=2,
        scale=3.0,
        max_bits=8,
        certificate_path=paper_dir
        / "proofs"
        / "Paper1IT"
        / "Generated"
        / "SemanticGridInt8Scale3Certificate.lean",
        certificate_module="Paper1IT.Generated.SemanticGridInt8Scale3Certificate",
        output_dir=out_dir / "semantic_grid_int8_scale3",
    )
    total = run_regime(
        name="total_collapse",
        records=semantic_records,
        embeddings=semantic_embeddings,
        quantization_mode="int8",
        decimals=2,
        scale=2.0,
        max_bits=8,
        certificate_path=paper_dir
        / "proofs"
        / "Paper1IT"
        / "Generated"
        / "SemanticGridTotalCollapseCertificate.lean",
        certificate_module="Paper1IT.Generated.SemanticGridTotalCollapseCertificate",
        output_dir=out_dir / "semantic_grid_int8_scale2",
    )

    summary = {
        "paper_id": args.paper_id,
        "model": args.model,
        "regimes": [reviewer, partial, total],
    }
    summary_path = out_dir / "paper1_jsait_experiment_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    tex_path = content_dir / "experiment_results_auto.tex"
    tex_path.write_text(render_tex_table(reviewer, partial, total), encoding="utf-8")
    print(f"[experiments] wrote {summary_path.relative_to(repo_root)}")
    print(f"[experiments] wrote {tex_path.relative_to(repo_root)}")


def build_semantic_grid_records() -> list:
    entities = [
        "invoice",
        "contract",
        "report",
        "dataset",
        "registry",
        "model",
        "cache",
        "artifact",
        "record",
        "checkpoint",
    ]
    actions = ["retrieval", "verification", "compression", "auditing"]
    modifiers = [
        ("semantic", "needs a stable identifier to recover exact identity"),
        ("quantized", "can collapse distinct instances under coarse precision"),
        ("collision", "creates residual ambiguity unless extra tag bits are stored"),
    ]
    records = []
    idx = 1
    for entity in entities:
        for action in actions:
            for label, ending in modifiers:
                records.append(
                    core.ItemRecord(
                        item_id=f"doc{idx:03d}",
                        text=f"The {entity} {action} pipeline uses embeddings and {ending}.",
                        label=f"{action}_{label}",
                        weight=float((idx % 3) + 1),
                    )
                )
                idx += 1
    return records


def run_regime(
    *,
    name: str,
    records,
    embeddings,
    quantization_mode: str,
    decimals: int,
    scale: float,
    max_bits: int,
    certificate_path: Path,
    certificate_module: str,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    quantized, quantization_label = core.quantize_embeddings(
        embeddings,
        mode=quantization_mode,
        decimals=decimals,
        scale=scale,
    )
    keys = core.quantized_keys(quantized)
    fiber_summaries = core.build_fiber_summaries(records, keys)
    audit = core.audit_collision_geometry(fiber_summaries)
    curve = core.budget_curve(
        records=records,
        observation_count=audit.distinct_fibers,
        max_fiber_card=audit.max_fiber_card,
        fiber_weight_lists_value=core.fiber_weight_lists(records, keys),
        max_bits=max_bits,
    )
    histogram = core.histogram_from_summaries(fiber_summaries)
    core.save_json(
        output_dir / "audit_summary.json",
        core.collision_audit_payload(
            audit=audit,
            embedding_backend="sentence-transformers",
            quantization_label=quantization_label,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            input_path=name,
        ),
    )
    core.write_curve_csv(output_dir / "budget_curve.csv", curve)
    core.write_curve_csv(output_dir / "fiber_histogram.csv", histogram)
    core.summaries_to_frame(fiber_summaries).to_csv(
        output_dir / "fiber_details.csv", index=False
    )
    core.maybe_make_plots(output_dir, histogram, curve)
    core.export_lean_certificate(
        certificate_path,
        module_name=certificate_module,
        fiber_sizes=core.fiber_sizes_from_summaries(fiber_summaries),
        scaled_fiber_weights_desc=core.scaled_sorted_fiber_weight_lists(
            records, keys, weight_scale=1000000
        ),
        weight_scale=1000000,
        audit=audit,
        curve=curve,
    )
    return {
        "name": name,
        "audit": asdict(audit),
        "curve": curve.to_dict(orient="records"),
        "quantization": quantization_label,
    }


def render_tex_table(no_collapse: dict, partial: dict, total: dict) -> str:
    rows = [no_collapse, partial, total]
    row_labels = {
        "no_collapse": "No",
        "partial_collapse": "Partial",
        "total_collapse": "Yes",
    }
    corpus_labels = {
        "no_collapse": "reviewer demo",
        "partial_collapse": "semantic grid",
        "total_collapse": "semantic grid",
    }

    def threshold_bits(regime: dict) -> int:
        return int(regime["audit"]["zero_error_threshold_bits"])

    def a_pi(regime: dict) -> int:
        return int(regime["audit"]["max_fiber_card"])

    partial_curve = {int(row["bits"]): row for row in partial["curve"]}
    total_curve = {int(row["bits"]): row for row in total["curve"]}

    lines = [
        "% Auto-generated by experiments/embedding_collisions/generate_paper1_jsait_assets.py.",
        r"We implemented an embedding-audit pipeline for quantized representations using a MiniLM sentence-transformer. Table~\ref{tab:embedding-audit-regimes} records three regimes. On a reviewer corpus with two-decimal quantization, the map remains injective and no side information is needed. On a synthetic semantic grid with int8 quantization at scale $3$, the encoder exhibits partial collapse: %d of %d items reside in collided fibers with max size $A_\pi=%d$, requiring a threshold of $\lceil \log_2 %d \rceil = %d$ bits. Under coarser scale-$2$ quantization, the same corpus collapses into a single fiber, where the zero-error identity task requires a full budget of $\lceil \log_2 %d \rceil = %d$ bits."
        % (
            int(partial["audit"]["collided_items"]),
            int(partial["audit"]["item_count"]),
            a_pi(partial),
            a_pi(partial),
            threshold_bits(partial),
            int(total["audit"]["item_count"]),
            threshold_bits(total),
        ),
        "",
        r"These audit results provide a machine-checked bridge to the formal theory. For the partial and total collapse regimes, the exported fiber statistics and budget curves are certified in Lean by proofs that the reported $A_\pi$, thresholds, and recoverable masses align with the paper's finite formulas. This certifies that the empirical summaries are consistent with the same rate-distortion laws proved in the formal development.",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l@{\,}c@{\,}c@{\,}c@{\,}c@{\,}c@{\,}c}",
        r"\hline",
        r"Collapse & Corpus & Quant. & Fibers & $A_\pi$ & Thresh. & Coll. \\",
        r"\hline",
    ]
    for regime in rows:
        audit = regime["audit"]
        lines.append(
            "%s & %s & %s & %d & %d & %d & %.2f\\%% \\\\"
            % (
                row_labels[regime["name"]],
                corpus_labels[regime["name"]],
                regime["quantization"].replace("_", r"\_"),
                int(audit["distinct_fibers"]),
                int(audit["max_fiber_card"]),
                int(audit["zero_error_threshold_bits"]),
                100.0 * float(audit["collision_rate"]),
            )
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Illustrative embedding-audit regimes for a MiniLM sentence-transformer after fixing a deployed quantization rule. The partial-collapse and total-collapse summaries are exported as Lean certificates and checked against the finite formulas used in the paper.}",
            r"\label{tab:embedding-audit-regimes}",
            r"\end{table}",
            "",
            r"In the partial-collapse regime, the worst-fiber floor is %.4f at $L=0$ and %.4f at $L=6$, while empirical distortion is %.4f and %.4f respectively; both vanish at the $L=7$ threshold. In the total-collapse regime, the data behaves as a single collision block, with distortion decreasing from %.4f at $L=0$ to %.4f at $L=6$ before vanishing."
            % (
                float(partial_curve[0]["worst_fiber_distortion_floor"]),
                float(partial_curve[0]["exact_empirical_distortion"]),
                float(partial_curve[6]["worst_fiber_distortion_floor"]),
                float(partial_curve[6]["exact_empirical_distortion"]),
                float(total_curve[0]["exact_empirical_distortion"]),
                float(total_curve[6]["exact_empirical_distortion"]),
            ),
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
