# arXiv abstract (paper1_jsait)

Title: Semantic Identity Compression: Exact Zero-Error Laws, Rate-Distortion, and Neurosymbolic Necessity

## Abstract (MathJax, arXiv-ready)

```text
What happens when classical compression theory meets learned representations? We answer this question exactly for zero-error identity recovery when the decoder observes a fixed embedding: the central object is the collision-fiber geometry of the map $\pi$, and it alone controls all operational costs.

Concretely, let $A_{\pi}=\max_u |\pi^{-1}(u)|$ be the largest collision fiber. We prove a tight fixed-length converse $L \ge \log_2 A_{\pi}$, give the exact finite-block scaling law and a pointwise adaptive budget $\lceil \log_2 |\pi^{-1}(u)|\rceil$, and derive the rate‑distortion tradeoff including an explicit deterministic distortion floor when identity bits are withheld. These statements characterize precisely the residual ambiguity left by any non‑injective learned embedding.

The applied payoff follows: because the residual ambiguity is structural (fiber‑based) rather than an artifact of representation format, symbolic identity mechanisms (handles, keys, pointers, nominal tags) are the canonical system-level way to pay that identity debt. All main results are machine‑checked in Lean 4; the formalization verifies correctness and demonstrates that the fiber‑geometry framework is sufficiently clean for full mechanization. Keywords: semantics-aware compression, zero-error coding, neurosymbolic systems, learned representations, side information
```

## Abstract (Unicode, Zenodo-ready)

```text
What happens when classical compression theory meets learned representations? We answer this question exactly for zero-error identity recovery when the decoder observes a fixed embedding: the central object is the collision-fiber geometry of the map π, and it alone controls all operational costs.

Concretely, let A_π = maxᵤ|π⁻¹(u)| be the largest collision fiber. We prove a tight fixed-length converse L ≥ log₂A_π, give the exact finite-block scaling law and a pointwise adaptive budget ⌈log₂|π⁻¹(u)|⌉, and derive the rate‑distortion tradeoff including an explicit deterministic distortion floor when identity bits are withheld. These statements characterize precisely the residual ambiguity left by any non‑injective learned embedding.

The applied payoff follows: because the residual ambiguity is structural (fiber‑based) rather than an artifact of representation format, symbolic identity mechanisms (handles, keys, pointers, nominal tags) are the canonical system-level way to pay that identity debt. All main results are machine‑checked in Lean 4; the formalization verifies correctness and demonstrates that the fiber‑geometry framework is sufficiently clean for full mechanization. Keywords: semantics-aware compression, zero-error coding, neurosymbolic systems, learned representations, side information
```
