# arXiv abstract (paper1_jsait)

Title: Semantic Representation Compression: Exact Zero-Error Laws and Neurosymbolic Necessity

## Abstract (MathJax, arXiv-ready)

```text
We prove that learned semantic representations alone cannot achieve zero-error identity recovery. When distinct entities map to the same representation, no decoder, regardless of architecture, training, or compute, can distinguish them without additional bits. If $A_\pi$ is the maximum representation-collision multiplicity, then the minimum worst-case auxiliary description is exactly $\log_2 A_\pi$ bits, and this bound is tight.

The coding laws are exact. For repeated independent uses of the same representation map, the minimum block budget is $L_t^\star=t\log_2 A_\pi$. In the adaptive regime, the optimal fiber budget is $\lceil \log_2 |\pi^{-1}(u)| \rceil$, any feasible adaptive scheme has expected length at least the expectation of that quantity and hence at least $H(C\mid U)/\log 2$, and standard fiberwise prefix coding achieves expected length at most $H(C\mid U)/\log 2 + 1$. Under a uniform source on a collision block of size $a$, subcritical fixed-length budgets satisfy the explicit error floor $D \ge 1-2^L/a$.

The same ambiguity-fiber geometry yields exact task-sufficiency, helper-view, and factorized-representation consequences. We instantiate these generic theorems in a concrete formal identity domain, proving that symbolic handles are information-theoretically necessary, not optional, for zero-error open-world identity recovery (NSL2, NSL3). The instantiated domain is a witness, not a restriction: the necessity claim applies to any learned representation with collisions. The pair consisting of a neural encoder and an injective symbolic handle achieves zero-error identity recovery if and only if the handle distinguishes entities within each semantic fiber (NSL1--NSL6). The identity bits must be paid somewhere, either in the encoder or in the symbolic handle. No free lunch.

Keywords: semantics-aware compression, zero-error coding, neurosymbolic systems, learned representations, side information
```

## Abstract (Unicode, Zenodo-ready)

```text
We prove that learned semantic representations alone cannot achieve zero-error identity recovery. When distinct entities map to the same representation, no decoder, regardless of architecture, training, or compute, can distinguish them without additional bits. If A_π is the maximum representation-collision multiplicity, then the minimum worst-case auxiliary description is exactly log₂A_π bits, and this bound is tight.

The coding laws are exact. For repeated independent uses of the same representation map, the minimum block budget is Lₜ^⋆ = tlog₂A_π. In the adaptive regime, the optimal fiber budget is ⌈log₂|π⁻¹(u)|⌉, any feasible adaptive scheme has expected length at least the expectation of that quantity and hence at least H(C ∣ U)/log 2, and standard fiberwise prefix coding achieves expected length at most H(C ∣ U)/log 2 + 1. Under a uniform source on a collision block of size a, subcritical fixed-length budgets satisfy the explicit error floor D ≥ 1 − 2ᴸ/a.

The same ambiguity-fiber geometry yields exact task-sufficiency, helper-view, and factorized-representation consequences. We instantiate these generic theorems in a concrete formal identity domain, proving that symbolic handles are information-theoretically necessary, not optional, for zero-error open-world identity recovery (NSL2, NSL3). The instantiated domain is a witness, not a restriction: the necessity claim applies to any learned representation with collisions. The pair consisting of a neural encoder and an injective symbolic handle achieves zero-error identity recovery if and only if the handle distinguishes entities within each semantic fiber (NSL1–NSL6). The identity bits must be paid somewhere, either in the encoder or in the symbolic handle. No free lunch.

Keywords: semantics-aware compression, zero-error coding, neurosymbolic systems, learned representations, side information
```
