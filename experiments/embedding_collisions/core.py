from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ItemRecord:
    item_id: str
    text: str
    weight: float = 1.0
    label: str | None = None


@dataclass(frozen=True)
class FiberSummary:
    key: str
    size: int
    total_weight: float
    example_ids: tuple[str, ...]
    example_labels: tuple[str, ...]


@dataclass(frozen=True)
class CollisionAudit:
    item_count: int
    total_weight: float
    distinct_fibers: int
    collided_fibers: int
    collided_items: int
    collision_rate: float
    max_fiber_card: int
    zero_error_threshold_bits: int
    single_symbol_feasible: bool


def load_records(
    input_path: Path,
    *,
    text_column: str = "text",
    id_column: str = "id",
    label_column: str | None = None,
    weight_column: str | None = None,
) -> list[ItemRecord]:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        rows = pd.read_csv(input_path).to_dict(orient="records")
    elif suffix == ".jsonl":
        rows = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif suffix == ".txt":
        rows = []
        with input_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                text = line.rstrip("\n")
                if text:
                    rows.append({id_column: str(index), text_column: text})
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")

    records: list[ItemRecord] = []
    for index, row in enumerate(rows, start=1):
        text = str(row[text_column])
        item_id = str(row.get(id_column, index))
        label = (
            None if label_column is None else _safe_optional_str(row.get(label_column))
        )
        weight = 1.0 if weight_column is None else float(row.get(weight_column, 1.0))
        records.append(
            ItemRecord(item_id=item_id, text=text, label=label, weight=weight)
        )
    return records


def synthetic_records_from_fiber_sizes(fiber_sizes: Sequence[int]) -> list[ItemRecord]:
    records: list[ItemRecord] = []
    for fiber_index, size in enumerate(fiber_sizes):
        for local_index in range(size):
            records.append(
                ItemRecord(
                    item_id=f"f{fiber_index}_item{local_index}",
                    text=f"fiber {fiber_index} item {local_index}",
                    label=f"fiber_{fiber_index}",
                    weight=1.0,
                )
            )
    return records


def hash_embed_texts(
    records: Sequence[ItemRecord], *, dim: int = 256, seed: int = 0
) -> np.ndarray:
    if dim <= 0:
        raise ValueError("dim must be positive")
    vectors = np.zeros((len(records), dim), dtype=np.float32)
    salt = str(seed).encode("ascii")
    for row_index, record in enumerate(records):
        tokens = record.text.lower().split()
        if not tokens:
            tokens = [""]
        for token in tokens:
            digest = hashlib.sha256(salt + token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % dim
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vectors[row_index, index] += sign
        norm = np.linalg.norm(vectors[row_index])
        if norm > 0:
            vectors[row_index] /= norm
    return vectors


def sentence_transformer_embed(
    records: Sequence[ItemRecord],
    *,
    model_name: str,
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install the experiment requirements first."
        ) from exc

    model = SentenceTransformer(model_name)
    vectors = model.encode(
        [record.text for record in records],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def transformers_mean_pool_embed(
    records: Sequence[ItemRecord],
    *,
    model_name: str,
    batch_size: int = 16,
    normalize: bool = True,
) -> np.ndarray:
    try:
        import torch  # pyright: ignore[reportMissingImports]
        from transformers import AutoModel, AutoTokenizer  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "transformers and torch are not installed. Install the experiment requirements first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    all_rows: list[np.ndarray] = []
    texts = [record.text for record in records]
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        with torch.no_grad():
            encoded = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            output = model(**encoded)
            last_hidden = output.last_hidden_state
            attention = encoded["attention_mask"].unsqueeze(-1)
            masked = last_hidden * attention
            summed = masked.sum(dim=1)
            counts = attention.sum(dim=1).clamp(min=1)
            pooled = summed / counts
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_rows.append(pooled.cpu().numpy())
    return np.concatenate(all_rows, axis=0).astype(np.float32)


def quantize_embeddings(
    embeddings: np.ndarray,
    *,
    mode: str,
    decimals: int = 2,
    scale: float = 64.0,
) -> tuple[np.ndarray, str]:
    array = np.asarray(embeddings)
    if array.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    if mode == "exact":
        quantized = array.astype(np.float32, copy=False)
        return quantized, "float32"
    if mode == "round":
        quantized = np.round(array.astype(np.float32), decimals=decimals)
        return quantized, f"round_{decimals}dp"
    if mode == "int8":
        quantized = np.clip(np.rint(array * scale), -127, 127).astype(np.int16)
        return quantized, f"int8_scale_{_clean_number(scale)}"
    if mode == "sign":
        quantized = np.where(array >= 0, 1, -1).astype(np.int8)
        return quantized, "sign"
    raise ValueError(f"Unknown quantization mode: {mode}")


def quantized_keys(quantized_embeddings: np.ndarray) -> list[tuple[Any, ...]]:
    return [tuple(row.tolist()) for row in np.asarray(quantized_embeddings)]


def build_fiber_summaries(
    records: Sequence[ItemRecord],
    keys: Sequence[tuple[Any, ...]],
    *,
    example_limit: int = 5,
) -> list[FiberSummary]:
    grouped_ids: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    grouped_labels: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    grouped_weights: dict[tuple[Any, ...], float] = defaultdict(float)
    for record, key in zip(records, keys, strict=True):
        grouped_ids[key].append(record.item_id)
        if record.label is not None:
            grouped_labels[key].append(record.label)
        grouped_weights[key] += record.weight

    summaries: list[FiberSummary] = []
    for key, ids in grouped_ids.items():
        labels = grouped_labels.get(key, [])
        summaries.append(
            FiberSummary(
                key=_fingerprint_key(key),
                size=len(ids),
                total_weight=grouped_weights[key],
                example_ids=tuple(ids[:example_limit]),
                example_labels=tuple(labels[:example_limit]),
            )
        )
    summaries.sort(key=lambda row: (-row.size, -row.total_weight, row.key))
    return summaries


def audit_collision_geometry(fiber_summaries: Sequence[FiberSummary]) -> CollisionAudit:
    item_count = sum(summary.size for summary in fiber_summaries)
    total_weight = sum(summary.total_weight for summary in fiber_summaries)
    collided_fibers = sum(1 for summary in fiber_summaries if summary.size > 1)
    collided_items = sum(
        summary.size for summary in fiber_summaries if summary.size > 1
    )
    max_fiber_card = max((summary.size for summary in fiber_summaries), default=0)
    collision_rate = 0.0 if item_count == 0 else collided_items / item_count
    return CollisionAudit(
        item_count=item_count,
        total_weight=total_weight,
        distinct_fibers=len(fiber_summaries),
        collided_fibers=collided_fibers,
        collided_items=collided_items,
        collision_rate=collision_rate,
        max_fiber_card=max_fiber_card,
        zero_error_threshold_bits=zero_error_threshold_bits(max_fiber_card),
        single_symbol_feasible=max_fiber_card <= 1,
    )


def zero_error_threshold_bits(a: int) -> int:
    if a <= 1:
        return 0
    return math.ceil(math.log2(a))


def feasible_zero_error(a: int, bits: int) -> bool:
    return (2**bits) >= a


def uniform_worst_fiber_distortion_floor(a: int, bits: int) -> float:
    if a <= 0:
        return 0.0
    return max(0.0, 1.0 - (2**bits) / a)


def exact_recoverable_mass(
    fiber_weight_lists: Sequence[Sequence[float]], tag_alphabet: int
) -> float:
    if tag_alphabet < 0:
        raise ValueError("tag_alphabet must be nonnegative")
    total = 0.0
    for weights in fiber_weight_lists:
        total += sum(sorted(weights, reverse=True)[:tag_alphabet])
    return total


def exact_distortion(
    fiber_weight_lists: Sequence[Sequence[float]], tag_alphabet: int
) -> float:
    total_mass = sum(sum(weights) for weights in fiber_weight_lists)
    if total_mass <= 0:
        return 0.0
    recoverable = exact_recoverable_mass(fiber_weight_lists, tag_alphabet)
    return max(0.0, 1.0 - recoverable / total_mass)


def empirical_source_entropy(records: Sequence[ItemRecord]) -> float:
    weights = np.asarray([record.weight for record in records], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    probs = weights / total
    positive = probs[probs > 0]
    return float(-(positive * np.log(positive)).sum())


def binary_entropy_nats(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def fano_rhs_nats(
    distortion: float, *, class_count: int, observation_count: int, tag_alphabet: int
) -> float:
    if class_count <= 1:
        return 0.0
    success = 1.0 - distortion
    joint_budget = max(1, observation_count * tag_alphabet)
    return (
        binary_entropy_nats(distortion)
        + success * math.log(joint_budget)
        + distortion * math.log(class_count - 1)
    )


def fano_error_lower_bound(
    records: Sequence[ItemRecord],
    *,
    observation_count: int,
    tag_alphabet: int,
    grid_size: int = 4000,
) -> float:
    class_count = len(records)
    if class_count <= 1:
        return 0.0
    entropy = empirical_source_entropy(records)
    upper = 1.0 - (1.0 / class_count)
    candidate_values = np.linspace(0.0, upper, grid_size + 1)
    for distortion in candidate_values:
        if entropy <= fano_rhs_nats(
            float(distortion),
            class_count=class_count,
            observation_count=observation_count,
            tag_alphabet=tag_alphabet,
        ):
            return float(distortion)
    return float(upper)


def fiber_weight_lists(
    records: Sequence[ItemRecord], keys: Sequence[tuple[Any, ...]]
) -> list[list[float]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for record, key in zip(records, keys, strict=True):
        grouped[key].append(record.weight)
    return list(grouped.values())


def scaled_sorted_fiber_weight_lists(
    records: Sequence[ItemRecord],
    keys: Sequence[tuple[Any, ...]],
    *,
    weight_scale: int,
) -> list[list[int]]:
    if weight_scale <= 0:
        raise ValueError("weight_scale must be positive")
    grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for record, key in zip(records, keys, strict=True):
        scaled = int(round(record.weight * weight_scale))
        if scaled < 0:
            raise ValueError("weights must be nonnegative")
        grouped[key].append(scaled)
    ordered_groups = []
    for key, weights in grouped.items():
        sorted_weights = sorted(weights, reverse=True)
        ordered_groups.append((key, sorted_weights))
    ordered_groups.sort(
        key=lambda item: (-len(item[1]), -sum(item[1]), _fingerprint_key(item[0]))
    )
    return [weights for _, weights in ordered_groups]


def budget_curve(
    *,
    records: Sequence[ItemRecord],
    observation_count: int,
    max_fiber_card: int,
    fiber_weight_lists_value: Sequence[Sequence[float]],
    max_bits: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_entropy = empirical_source_entropy(records)
    class_count = len(records)
    for bits in range(max_bits + 1):
        tag_alphabet = 2**bits
        rows.append(
            {
                "bits": bits,
                "tag_alphabet": tag_alphabet,
                "class_count": class_count,
                "observation_count": observation_count,
                "source_entropy_nats": source_entropy,
                "worst_fiber_distortion_floor": uniform_worst_fiber_distortion_floor(
                    max_fiber_card, bits
                ),
                "exact_empirical_distortion": exact_distortion(
                    fiber_weight_lists_value, tag_alphabet
                ),
                "fano_lower_bound": fano_error_lower_bound(
                    records,
                    observation_count=observation_count,
                    tag_alphabet=tag_alphabet,
                ),
                "zero_error_feasible_on_worst_fiber": feasible_zero_error(
                    max_fiber_card, bits
                ),
            }
        )
    return pd.DataFrame(rows)


def histogram_from_summaries(fiber_summaries: Sequence[FiberSummary]) -> pd.DataFrame:
    histogram = Counter(summary.size for summary in fiber_summaries)
    rows = [
        {"fiber_size": fiber_size, "fiber_count": count}
        for fiber_size, count in sorted(histogram.items())
    ]
    return pd.DataFrame(rows)


def fiber_sizes_from_summaries(fiber_summaries: Sequence[FiberSummary]) -> list[int]:
    return [summary.size for summary in fiber_summaries]


def summaries_to_frame(fiber_summaries: Sequence[FiberSummary]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fiber_key": summary.key,
                "fiber_size": summary.size,
                "total_weight": summary.total_weight,
                "example_ids": "|".join(summary.example_ids),
                "example_labels": "|".join(summary.example_labels),
            }
            for summary in fiber_summaries
        ]
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_curve_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def maybe_make_plots(
    output_dir: Path,
    histogram: pd.DataFrame,
    curve: pd.DataFrame,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(histogram["fiber_size"], histogram["fiber_count"], color="#315c8a")
    ax.set_xlabel("Fiber size")
    ax.set_ylabel("Count")
    ax.set_title("Collision fiber histogram")
    fig.tight_layout()
    fig.savefig(output_dir / "fiber_histogram.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        curve["bits"],
        curve["worst_fiber_distortion_floor"],
        marker="o",
        label="Worst-fiber floor",
    )
    ax.plot(
        curve["bits"],
        curve["exact_empirical_distortion"],
        marker="s",
        label="Exact empirical distortion",
    )
    ax.set_xlabel("Auxiliary bits L")
    ax.set_ylabel("Distortion / error")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Rate-distortion style curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "distortion_curve.png", dpi=180)
    plt.close(fig)
    return True


def collision_audit_payload(
    *,
    audit: CollisionAudit,
    embedding_backend: str,
    quantization_label: str,
    model_name: str | None,
    input_path: str | None,
) -> dict[str, Any]:
    payload = asdict(audit)
    payload.update(
        {
            "embedding_backend": embedding_backend,
            "model_name": model_name,
            "quantization": quantization_label,
            "input_path": input_path,
        }
    )
    return payload


def export_lean_certificate(
    output_path: Path,
    *,
    module_name: str,
    fiber_sizes: Sequence[int],
    scaled_fiber_weights_desc: Sequence[Sequence[int]],
    weight_scale: int,
    audit: CollisionAudit,
    curve: pd.DataFrame,
) -> None:
    budget_rows = []
    total_items = int(sum(fiber_sizes))
    max_fiber = int(max(fiber_sizes, default=0))
    total_scaled_weight = int(
        sum(sum(weights) for weights in scaled_fiber_weights_desc)
    )
    for row in curve.to_dict(orient="records"):
        bits = int(row["bits"])
        tag_alphabet = int(row["tag_alphabet"])
        worst_num = max(0, max_fiber - tag_alphabet)
        exact_num = total_items - sum(min(size, tag_alphabet) for size in fiber_sizes)
        weighted_exact_num = total_scaled_weight - sum(
            sum(weights[:tag_alphabet]) for weights in scaled_fiber_weights_desc
        )
        zero_error = (
            "true" if bool(row["zero_error_feasible_on_worst_fiber"]) else "false"
        )
        budget_rows.append(
            "    { bits := %(bits)d, reportedTagAlphabet := %(tag_alphabet)d, "
            "reportedWorstFiberFloorNumerator := %(worst_num)d, "
            "reportedWorstFiberFloorDenominator := %(max_fiber)d, "
            "reportedExactUniformErrorNumerator := %(exact_num)d, "
            "reportedExactUniformErrorDenominator := %(total_items)d, "
            "reportedExactWeightedErrorNumeratorScaled := %(weighted_exact_num)d, "
            "reportedExactWeightedErrorDenominatorScaled := %(total_scaled_weight)d, "
            "reportedZeroErrorFeasible := %(zero_error)s }"
            % {
                "bits": bits,
                "tag_alphabet": tag_alphabet,
                "worst_num": worst_num,
                "max_fiber": max_fiber,
                "exact_num": exact_num,
                "total_items": total_items,
                "weighted_exact_num": weighted_exact_num,
                "total_scaled_weight": total_scaled_weight,
                "zero_error": zero_error,
            }
        )

    certificate_name = _safe_lean_ident(module_name.rsplit(".", 1)[-1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contents = f"""import Paper1IT.EmpiricalAuditBridge

namespace {module_name.rsplit(".", 1)[0]}

open Ssot.Paper1IT

def {certificate_name} : EmpiricalCertificate :=
  {{ fiberSizes := [{", ".join(str(int(size)) for size in fiber_sizes)}]
  , fiberWeightsDescending := [{", ".join("[" + ", ".join(str(int(w)) for w in weights) + "]" for weights in scaled_fiber_weights_desc)}]
  , weightScale := {int(weight_scale)}
  , reportedMaxFiberCard := {int(audit.max_fiber_card)}
  , reportedTotalItems := {int(audit.item_count)}
  , reportedThresholdBits := {int(audit.zero_error_threshold_bits)}
  , budgetRows := [
{",\n".join(budget_rows)}
    ]
  }}

theorem {certificate_name}_valid : ValidEmpiricalCertificate {certificate_name} := by
  native_decide

end {module_name.rsplit(".", 1)[0]}
"""
    output_path.write_text(contents, encoding="utf-8")


def write_text_table(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_optional_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _clean_number(value: float) -> str:
    return str(value).replace(".", "p")


def _fingerprint_key(key: tuple[Any, ...]) -> str:
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()
    return digest[:16]


def _safe_lean_ident(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not text:
        return "certificate"
    if text[0].isdigit():
        return f"cert_{text}"
    return text
