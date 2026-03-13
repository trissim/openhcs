# Embedding Collision Experiments

These experiments mirror the finite, deployed-precision representation geometry formalized in the Lean development.

The code tracks the same core objects:

- `A_pi`: worst collision fiber size (`maxFiberCard` in Lean)
- zero-error threshold: `2^L >= A_pi`
- worst-fiber distortion floor: `max(0, 1 - 2^L / a)`
- exact empirical distortion under a uniform per-fiber tag budget by taking the top `2^L` weighted states inside each fiber
- Fano-style entropy-sensitive lower bounds using the empirical source law and the observed budget alphabet

This gives a reproducible audit path for real embeddings after fixing a deployed representation and quantization rule.

## Files

- `core.py` - finite collision algebra and budget-curve logic
- `audit_embeddings.py` - CLI for real or synthetic audits
- `synthetic_demo.py` - small verification script that checks the finite formulas on a toy example
- `requirements.txt` - optional Python packages for learned backends and plotting
- `run_reviewer_demo.py` - reviewer-facing helper for text-corpus audits and Lean certificate generation

## Quick start

Run the synthetic verification first:

```bash
python experiments/embedding_collisions/synthetic_demo.py
```

Run a lightweight collision audit without external models:

```bash
python experiments/embedding_collisions/audit_embeddings.py \
  --output-dir experiments/embedding_collisions/out/hash_demo \
  --backend hash \
  --synthetic-fibers 16,8,8,4,2,1 \
  --quantization sign \
  --max-bits 6
```

Run the reviewer demo workflow on a small text corpus:

```bash
python experiments/embedding_collisions/run_reviewer_demo.py --backend hash
```

Or, once the optional ML dependencies are installed, switch to a real sentence-transformer backend:

```bash
python experiments/embedding_collisions/run_reviewer_demo.py \
  --backend sentence-transformers \
  --model sentence-transformers/all-MiniLM-L6-v2
```

Audit a real text file with a sentence-transformer backend:

```bash
python experiments/embedding_collisions/audit_embeddings.py \
  --input path/to/texts.csv \
  --text-column text \
  --id-column id \
  --output-dir experiments/embedding_collisions/out/minilm_round2 \
  --backend sentence-transformers \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --quantization round \
  --decimals 2 \
  --max-bits 10
```

## Input format

Supported inputs:

- `.csv`
- `.jsonl`
- `.txt` (one text per line)

Optional columns:

- `id`
- `text`
- `label`
- `weight`

If `weight` is omitted, the source is uniform over rows.

## Output files

- `audit_summary.json` - top-level collision statistics
- `fiber_histogram.csv` - histogram of fiber sizes
- `fiber_details.csv` - one row per quantized fiber
- `budget_curve.csv` - finite rate-distortion style curve over auxiliary bit budgets
- `budget_curve.csv` also includes an entropy-sensitive `fano_lower_bound` column for comparison with the exact finite curve
- `fiber_histogram.png` and `distortion_curve.png` when `matplotlib` is installed

## Interpreting the outputs

- `max_fiber_card` is the empirical `A_pi` after the chosen quantization.
- `zero_error_threshold_bits = ceil(log2 A_pi)` is the exact fixed-length threshold for worst-case zero-error recovery.
- `worst_fiber_distortion_floor` is the direct finite analogue of the paper's deterministic floor.
- `exact_empirical_distortion` is stronger on a full dataset because it aggregates across all fibers under the same per-fiber tag budget.
- `fano_lower_bound` mirrors the entropy-sensitive Lean converse layer and is usually looser than the exact finite fiber law.

Emit a Lean certificate from an audit:

```bash
python experiments/embedding_collisions/audit_embeddings.py \
  --output-dir experiments/embedding_collisions/out/hash_demo \
  --backend hash \
  --synthetic-fibers 16,8,8,4,2,1 \
  --quantization sign \
  --max-bits 6 \
  --lean-certificate-path docs/papers/paper1_typing_discipline/proofs/Paper1IT/Generated/HashDemoCertificate.lean \
  --lean-certificate-module Paper1IT.Generated.HashDemoCertificate
```

Then build the generated certificate with Lake:

```bash
cd docs/papers/paper1_typing_discipline/proofs
lake build Paper1IT.Generated.HashDemoCertificate
```

## Important caveat

For continuous embeddings, exact float equality is usually not the right deployed representation. The theory applies to the representation actually used in the system, so the audit should fix a precision or quantization rule first.

## Lean bridge

The bridge module is `docs/papers/paper1_typing_discipline/proofs/Paper1IT/EmpiricalAuditBridge.lean`.

It machine-checks, from the exported audited fiber sizes alone:

- reported `A_pi`
- reported threshold bits via `requiredTagBits`
- reported worst-fiber floor numerators
- reported exact unweighted empirical error numerators under each bit budget
- reported exact weighted empirical error numerators in scaled integer mass units
- reported zero-error feasibility flags

This is a certificate that the summarized empirical finite quantities are consistent with the same finite formulas used in the Lean development.
