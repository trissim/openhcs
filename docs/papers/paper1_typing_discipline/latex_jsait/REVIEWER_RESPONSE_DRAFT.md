# Reviewer Response Draft

## Rate-distortion-perception tie-in

We now make the connection explicit in the discussion section. In the zero-error limit, our deterministic distortion floor recovers the perception-constrained rate-distortion tradeoff when the perception loss is the 0-1 identity loss. We cite Blau-Michaeli to situate the result relative to rate-distortion-perception, while also stating clearly that extending the theory to continuous perceptual metrics is future work rather than a claim of the present paper.

On mechanization: the discrete finite core is already mechanized. The zero-error threshold and worst-fiber distortion laws are covered by the existing Lean development, and we now make explicit that the empirical post-quantization summaries can also be checked against the same finite formulas. What is not mechanized is the full continuous perception-aware theory; that would require a separate formalization of perceptual metrics and stochastic perceptual constraints.

## Embedding experiments and empirical rate-distortion behavior

We added an embedding-audit pipeline for deployed quantized representations and used it to study a MiniLM sentence-transformer under several quantization rules. The relevant point for the paper is that once the deployed representation is fixed, the theoremic quantities become directly measurable: the number of fibers, worst collision size $A_\pi$, the exact threshold budget $\lceil \log_2 A_\pi \rceil$, and the resulting finite budget-distortion curve.

The revision now highlights three regimes:

- No collapse: reviewer demo corpus, 20 items, two-decimal quantization, 20 fibers, $A_\pi=1$, threshold 0 bits, collision rate 0\%.
- Partial collapse: semantic grid corpus, 120 items, int8 quantization at scale 3, 13 fibers, $A_\pi=77$, threshold 7 bits, collision rate 98.33\%.
- Total collapse: same 120-item corpus, int8 quantization at scale 2, 1 fiber, $A_\pi=120$, threshold 7 bits, collision rate 100\%.

The partial-collapse case is especially informative because it exhibits a nontrivial finite curve rather than either injectivity or total degeneration. The total-collapse case is the clean extreme in which the observation channel carries no identity information at all, and the task reduces entirely to a side-information budget law.

## Machine-checked empirical bridge

We also added a machine-checked bridge from the empirical audits back to Lean. The exported partial-collapse and total-collapse summaries are turned into generated Lean certificates that verify the reported fiber histogram, threshold bits, exact empirical error numerators, weighted empirical masses, and zero-error feasibility flags against the finite formulas used in the paper.

This does not claim a proof of correctness for the neural encoder itself. It does certify that the empirical summaries reported in the revision are internally consistent with the same finite zero-error rate-distortion theorems proved in Lean.

## Files added in this revision

- Paper discussion update: `docs/papers/paper1_typing_discipline/latex_jsait/content/06b_extensions.tex`
- Experiment code: `experiments/embedding_collisions/`
- Lean bridge: `docs/papers/paper1_typing_discipline/proofs/Paper1IT/EmpiricalAuditBridge.lean`
- Certified partial-collapse audit: `docs/papers/paper1_typing_discipline/proofs/Paper1IT/Generated/SemanticGridInt8Scale3Certificate.lean`
- Certified total-collapse audit: `docs/papers/paper1_typing_discipline/proofs/Paper1IT/Generated/SemanticGridTotalCollapseCertificate.lean`
