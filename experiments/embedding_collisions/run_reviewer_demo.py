#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reviewer-facing embedding collision audit workflow."
    )
    parser.add_argument(
        "--backend",
        choices=["sentence-transformers", "transformers", "hash"],
        default="sentence-transformers",
        help="Embedding backend to use for the reviewer demo",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name for learned backends",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/embedding_collisions/out/reviewer_demo"),
    )
    parser.add_argument(
        "--lean-certificate-path",
        type=Path,
        default=Path(
            "docs/papers/paper1_typing_discipline/proofs/Paper1IT/Generated/ReviewerDemoCertificate.lean"
        ),
    )
    parser.add_argument(
        "--lean-certificate-module",
        default="Paper1IT.Generated.ReviewerDemoCertificate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path("experiments/embedding_collisions/data/reviewer_demo_texts.csv")
    quantization = "sign" if args.backend == "hash" else "round"
    command = [
        sys.executable,
        "experiments/embedding_collisions/audit_embeddings.py",
        "--input",
        str(input_path),
        "--text-column",
        "text",
        "--id-column",
        "id",
        "--label-column",
        "label",
        "--weight-column",
        "weight",
        "--output-dir",
        str(args.output_dir),
        "--backend",
        args.backend,
        "--quantization",
        quantization,
        "--max-bits",
        "8",
        "--lean-certificate-path",
        str(args.lean_certificate_path),
        "--lean-certificate-module",
        args.lean_certificate_module,
    ]
    if args.backend != "hash":
        command.extend(["--model", args.model, "--decimals", "2"])

    print("Running reviewer demo audit:")
    print("  " + " ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print("Reviewer demo audit failed.")
        print(
            "If you selected a learned backend, install the optional experiment requirements first:"
        )
        print("  pip install -r experiments/embedding_collisions/requirements.txt")
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
