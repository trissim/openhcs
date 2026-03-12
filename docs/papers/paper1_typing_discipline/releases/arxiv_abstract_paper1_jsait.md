# arXiv abstract (paper1_jsait)

Title: Semantic Identity Compression: Exact Zero-Error Laws, Rate-Distortion, and Neurosymbolic Necessity

## Abstract (MathJax, arXiv-ready)

```text
The question of whether semantic representations can capture identity without auxiliary identifiers is often framed as a design question. This paper shows that it is governed by exact coding constraints. When distinct entities map to the same representation, no decoder can recover their identities without additional distinguishing information. The minimum worst-case auxiliary description is exactly $\log_2 A_\pi$ bits, where $A_\pi$ is the maximum collision multiplicity. If this cost is left unpaid, the distortion floor on a worst collision block is $D \ge 1 - 1/A_\pi$.

The paper isolates the deterministic corner of this problem. On a uniform collision block of size $a$ with auxiliary budget $L$ bits, the optimal distortion is exactly $D^\star(L)=\max(0,1-2^L/a)$. For repeated independent uses of the same representation map, the minimum zero-error block budget is $L_t^\star=t\log_2 A_\pi$. In the adaptive regime, the optimal fiber budget is $\lceil \log_2 |\pi^{-1}(u)| \rceil$, any feasible adaptive scheme has expected length at least the expectation of that quantity and hence at least $H(C\mid U)/\log 2$, and standard fiberwise prefix coding achieves expected length at most $H(C\mid U)/\log 2 + 1$. The point is that a question often discussed at the level of architecture can be stated and answered as an exact coding problem.

The same ambiguity-fiber geometry yields exact task-sufficiency, helper-view, and factorized-representation consequences. We instantiate these generic laws in a concrete formal identity domain and show that a neural encoder paired with a symbolic handle achieves zero-error identity recovery if and only if the handle resolves ambiguity inside each semantic fiber (NSL1--NSL6). Auxiliary distinguishing information must be supplied somewhere; symbolic layers are a standard computational mechanism for supplying it.

Keywords: semantics-aware compression, zero-error coding, neurosymbolic systems, learned representations, side information
```

## Abstract (Unicode, Zenodo-ready)

```text
The question of whether semantic representations can capture identity without auxiliary identifiers is often framed as a design question. This paper shows that it is governed by exact coding constraints. When distinct entities map to the same representation, no decoder can recover their identities without additional distinguishing information. The minimum worst-case auxiliary description is exactly log₂A_π bits, where A_π is the maximum collision multiplicity. If this cost is left unpaid, the distortion floor on a worst collision block is D ≥ 1 − 1/A_π.

The paper isolates the deterministic corner of this problem. On a uniform collision block of size a with auxiliary budget L bits, the optimal distortion is exactly D^⋆(L) = max (0, 1 − 2ᴸ/a). For repeated independent uses of the same representation map, the minimum zero-error block budget is Lₜ^⋆ = tlog₂A_π. In the adaptive regime, the optimal fiber budget is ⌈log₂|π⁻¹(u)|⌉, any feasible adaptive scheme has expected length at least the expectation of that quantity and hence at least H(C ∣ U)/log 2, and standard fiberwise prefix coding achieves expected length at most H(C ∣ U)/log 2 + 1. The point is that a question often discussed at the level of architecture can be stated and answered as an exact coding problem.

The same ambiguity-fiber geometry yields exact task-sufficiency, helper-view, and factorized-representation consequences. We instantiate these generic laws in a concrete formal identity domain and show that a neural encoder paired with a symbolic handle achieves zero-error identity recovery if and only if the handle resolves ambiguity inside each semantic fiber (NSL1–NSL6). Auxiliary distinguishing information must be supplied somewhere; symbolic layers are a standard computational mechanism for supplying it.

Keywords: semantics-aware compression, zero-error coding, neurosymbolic systems, learned representations, side information
```
