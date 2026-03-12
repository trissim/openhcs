# Paper 1 Generic-First Restructure Plan

## Goal

Restructure the paper so a JSAIT/TIT reader can read it as an information-theory paper first, a semantics-aware compression paper second, and a formal PL/Lean instantiation third.

The specific problem to fix is structural: the current draft still exposes readers to entity-attribute / PL vocabulary before they reach the generic theorem arc. That weakens fit for ML- and IT-oriented readers, even though the underlying results are now much better framed.

This plan therefore goes beyond local prose cleanup. It reorganizes the paper so the main coding laws are presented in generic representation-map form before any formal witness machinery appears.

---

## Desired Reader Experience

### A non-PL / ML / IT reader should be able to read:

1. the abstract,
2. the generic framework,
3. the main theorems,
4. the ML / learned-representation instantiation,

without ever needing to care where the representation map `pi` came from.

### A formal methods reader should then encounter:

- the entity-attribute model,
- the ACS / NSL theorems,
- the Lean mechanization,

as a rigorous machine-checked instantiation of the generic coding framework.

---

## Structural Diagnosis

### Current reading order still feels like

1. abstract coding framework,
2. formal entity-attribute instantiation,
3. representation-induced information barrier with `attribute-computable`,
4. main theorems,
5. applications / instantiations.

### Desired reading order

1. abstract coding framework,
2. main generic theorems,
3. generic consequences,
4. instantiations:
   - learned representations first,
   - formal entity-attribute / Lean witness second,
5. extensions and limitations.

### Central rule

If a theorem depends only on:

- a finite class space,
- a representation map,
- fiber cardinalities,
- and auxiliary description size,

then it should be stated in generic representation-map language.

If a statement genuinely depends on the formal entity-attribute model, it should appear later as an instantiation or corollary.

---

## Target Top-Level Narrative Arc

### 1. Abstract coding framework
- finite class space `C`
- representation map `pi : C -> U`
- decoder observes `U = pi(C)`
- encoder may send auxiliary description `M`
- fibers `pi^{-1}(u)`
- collision multiplicity `A_pi`

### 2. Main theorem arc in generic form
- zero auxiliary description iff injective representation
- fixed-length zero-error converse
- exact one-shot feasibility threshold
- exact finite-block scaling law
- adaptive fiberwise law
- expected-length and entropy comparisons
- deterministic-corner semantic identity rate-distortion theorem

### 3. Generic consequences
- task sufficiency
- helper-view sufficiency
- factorization law
- query-side interpretation as another currency for unresolved ambiguity

### 4. Instantiations

#### 4.1 Learned representations
- embeddings
- retrieval systems
- concept bottlenecks
- semantic caches
- open-world identity systems

#### 4.2 Formal entity-attribute model
- explicit entity domain
- attribute family and induced profile map
- information barrier in instantiated form
- ACS8, ACS9, EMB1
- NSL1-NSL6
- Lean note

### 5. Scope, limitations, and extensions

---

## File-by-File Implementation Plan With Drafted Edits

## 1. `latex_jsait/content/abstract.tex`

### Goal
Keep the current strong framing, but ensure the abstract reads as generic and theorem-first, not witness-first.

### Status after recent revisions
Already much improved. No major structural rewrite needed here.

### Keep / preserve
- design-question vs coding-constraint opening
- deterministic-corner rate-distortion formula
- adaptive fiberwise budget
- final instantiated neurosymbolic payoff

### Optional draft tightening

```latex
\begin{abstract}
Semantic representations are often expected to support identity recovery without auxiliary identifiers. This paper shows that the issue is governed by exact coding constraints. When distinct entities map to the same representation, no decoder can recover their identities without additional distinguishing information. The minimum worst-case auxiliary description is exactly $\log_2 A_\pi$ bits, where $A_\pi$ is the maximum collision multiplicity. If this cost is left unpaid, the distortion floor on a worst collision block is $D \ge 1 - 1/A_\pi$.

The paper isolates a deterministic corner of this problem. On a uniform collision block of size $a$ with auxiliary budget $L$ bits, the optimal distortion is exactly $D^\star(L)=\max(0,1-2^L/a)$. For repeated independent uses of the same representation map, the minimum zero-error block budget is $L_t^\star=t\log_2 A_\pi$. In the adaptive regime, the optimal fiber budget is $\lceil \log_2 |\pi^{-1}(u)| \rceil$, and the expected auxiliary rate is bounded around $H(C\mid U)/\log 2$ up to one bit.

The same ambiguity-fiber geometry yields exact task-sufficiency, helper-view, and factorized-representation consequences. We then instantiate these generic laws in a concrete formal identity domain and show that a neural encoder paired with a symbolic handle achieves zero-error identity recovery if and only if the handle resolves ambiguity inside each semantic fiber (\LH{NSL1}--\LH{NSL6}). Auxiliary distinguishing information must be supplied somewhere; symbolic layers are a standard computational mechanism for supplying it.
\end{abstract}
```

### Why
- keeps the abstract generic-first
- no local path references
- no PL vocabulary before the theoremic core lands

---

## 2. `latex_jsait/content/01_introduction.tex`

### Goal
Make the introduction explicitly generic-first and give readers a route through the paper that bypasses formal witness material until later.

### Required edits

#### 2.1 Keep the current `Why the Formulation Matters` subsection
This section is now better than `Why This Is Not Obvious` and should stay.

#### 2.2 Add an explicit generic-first sentence near the start

Suggested insertion after the formal problem statement:

```latex
The theoremic development is generic in the representation map $\pi$: the same laws apply whether $\pi$ arises from a learned encoder, a schema projection, a feature extractor, or an explicit attribute family.
```

#### 2.3 Add an ML-reader forward pointer in `Paper Organization`

Suggested replacement for the organization paragraph:

```latex
Section~\ref{sec:model} introduces the generic representation-side-information framework. Sections~\ref{sec:zero-error} and~\ref{sec:lwd} develop the main coding theorems: zero-error laws, adaptive budgets, and the deterministic-corner semantic identity rate-distortion result. Section~\ref{sec:applications} then presents two instantiations of those laws, beginning with learned representations and then turning to a formal entity-attribute model. Readers primarily interested in learned representations may read directly from the generic theorem arc to Section~\ref{sec:applications}. Section~\ref{sec:matroid} studies query costs as an operational alternative to transmitting side information, and Section~\ref{sec:robustness} studies the stability cost of residual ambiguity under system growth. \ifdefined\JSAITREVIEW Section~\ref{sec:conclusion} closes with extensions and open directions. \else Section~\ref{sec:extensions} sketches open directions, and Section~\ref{sec:conclusion} concludes. \fi
```

### Why
- it tells ML readers where they appear in the paper
- it reduces the feeling that the formal instantiation is mandatory reading

---

## 3. `latex_jsait/content/02_compression_framework.tex`

### Goal
This is the most important structural edit. The file must stop reading like a PL setup section and instead become the genuinely abstract front door of the paper.

### Current problem
The section now starts generically, but it still continues immediately into:
- attribute family
- `q_I`
- admissibility contract
- attribute-only observer
- attribute-computable

before the main theorem arc begins.

### Required restructuring

#### 3.1 Keep only the generic framework in this file's front section
Preserve:
- coding model induced by a representation
- representation fiber
- where `pi` may come from
- notation bridge between `C` and `V`

#### 3.2 Remove the following material from the early framework section and move it later
- `Entity space and attribute family`
- `Attribute observation family`
- `Attribute profile`
- `Attribute indistinguishability`
- admissibility contract discussion
- classification scheme
- attribute-only observer
- representation-induced information barrier in attribute-computable language
- fixed-axis sufficiency / insufficiency

These should move to the later formal-instantiation section in `06_applications.tex`, or to a new formal appendix-facing subsection if needed.

### Draft replacement body for the top of the file

```latex
\subsection{The Abstract Coding Framework}

The paper studies a finite class space together with a fixed representation map. The decoder observes the representation value, and the encoder may additionally send auxiliary description to resolve whatever ambiguity the representation leaves behind. The coding laws depend only on this representation map and its fibers, not on the mechanism that produced it.

\begin{definition}[Coding model induced by a representation]
Let $C$ be a finite class variable, let $U=\pi(C)$ be the representation available to the decoder, and let $M$ be any auxiliary description transmitted by the encoder. The main theorem arc studies the minimum length of $M$ needed for exact recovery of $C$ from $(U,M)$, in both fixed-length and representation-adaptive regimes.
\end{definition}

\begin{definition}[Representation fiber]
For a representation value $u$, define the corresponding fiber by
\[
\pi^{-1}(u) = \{c : \pi(c)=u\}.
\]
The paper's central object is the geometry of these fibers.
\end{definition}

\begin{remark}[Where the representation may come from]
No further structure is assumed. The map $\pi$ may arise from a learned encoder, a database schema projection, a feature-extraction pipeline, or a hand-designed attribute family. The theoremic contribution is generic in the representation map once its fibers are fixed.
\end{remark}
```

### Why
- this lets the theorem arc begin without PL interference
- it aligns the paper with TIT/JSAIT expectations immediately

---

## 4. `latex_jsait/content/04_kolmogorov_witness.tex`

### Goal
Make the main theorem arc fully generic wherever the proofs permit.

### Status
The opening and several definitions have already been genericized. This was the right move and should be extended as needed.

### Remaining principle
Every theorem in this file should read generically unless the proof genuinely needs the entity/attribute witness model.

### Draft generic template to follow

```latex
Let $\mathcal C$ be a finite class space and let $\pi : \mathcal C \to \mathcal U$ be a representation map.
```

### Specific review checklist
- theorem statements use `C`, `pi`, `U`, and fibers
- proofs do not refer to attribute profiles unless necessary
- examples and interpretive remarks may mention embeddings as well as formal domains

### Why
- this is where the generic-first restructuring becomes real rather than cosmetic

---

## 5. `latex_jsait/content/05_rate_distortion.tex`

### Goal
Keep the current strong deterministic-corner framing, but tighten it into a clean IT voice.

### Already good
- theorem title
- exact formula
- zero-error threshold corollary
- zero-budget floor corollary
- scope remark
- steepness remark

### Small draft cleanup

Current wording is acceptable, but if desired the subsection title can be simplified to:

```latex
\subsection{Semantic Identity Rate-Distortion on a Collision Block}
```

if the phrase `The Deterministic Corner` feels slightly essayistic in final form.

### Keep
- explicit scope: uniform source, single collision block
- explicit reason for scope: isolates geometry-only effect

---

## 6. `latex_jsait/content/06_applications.tex`

### Goal
Turn this into a true instantiations section with learned representations first and formal PL/Lean material second.

### Current problem
The section is better than before, but it still opens with instantiated neurosymbolic theorems before the ML reader has fully seen themselves.

### Required reordering

#### 6.1 Lead with learned-representation consequences
The order should be:

1. high-level ML / semantics reading
2. `What this means for ML systems`
3. `Systems-level reading`
4. attribute-only vs tag-augmented design
5. formal instantiation
6. instantiated neurosymbolic theorems

#### 6.2 Move the formal material later inside the section
Move later:
- `Instantiated neurosymbolic linking theorems`
- `Formal instantiation`

These should appear after the ML-native paragraphs.

### Draft replacement for the top of the section

```latex
\subsection{Instantiations: Learned Representations and Formal Instances}

The abstract coding framework applies to any representation map. We begin with the learned-representation setting that motivates many modern systems, and then turn to a formal entity-attribute model that gives a machine-checked instance of the same laws.

The same fiber picture clarifies the distinction between semantic description and exact identity. Semantic information is what the representation exposes. Identity information is the exact class label or entity name that may still need to be carried explicitly. Zero-error recovery from semantic information alone is possible exactly when the semantic representation is injective on identity classes. When it is not, the collision fibers quantify the irreducible identity residue that must be restored by explicit metadata.
```

### Draft ML-first local order

```latex
\paragraph{What this means for ML systems.}
...

\paragraph{Systems-level reading.}
...

\paragraph{Attribute-only versus tag-augmented design.}
...

\paragraph{Formal instantiation.}
...

\paragraph{Instantiated neurosymbolic linking theorems.}
...
```

### Draft formal-instantiation paragraph

```latex
\paragraph{Formal instantiation.}
The paper also develops a concrete formal identity domain in Lean. In that model, the representation map is induced by an explicit entity-attribute structure, and the resulting theorems (\LH{ACS8}, \LH{ACS9}, \LH{EMB1}, \LH{NSL1}--\LH{NSL6}) specialize the generic coding laws to a machine-checked setting. Readers interested only in the information-theoretic content may treat that development as a rigorous formal instantiation rather than as a prerequisite for understanding the main theorem arc.
```

### Why
- ML readers see themselves before formal witness material
- the PL/Lean side keeps full value, but later and more appropriately positioned

---

## 7. `latex_jsait/content/06b_extensions.tex`

### Goal
Keep the new `Scope and Limits` section. It is good and TIT-appropriate.

### No major rewrite needed
This section now does the right job.

### One optional draft polish

If desired, shorten the opening sentence to:

```latex
This section records directions beyond the deterministic finite side-information model studied in the main theorem arc.
```

to reduce self-consciousness.

---

## 8. `latex_jsait/content/07_conclusion.tex`

### Goal
Keep the current accounting-spine close, but make the ending more neutral and less self-advertising.

### Current status
Already much improved.

### Optional draft tightening

```latex
The central systems-level contribution is a formal justification for symbolic mechanisms in semantic systems. The coding laws show that a non-injective representation requires auxiliary information to resolve its collision fibers, with worst-case cost $\log_2 A_\pi$ bits and fiberwise cost $\lceil \log_2 |\pi^{-1}(u)| \rceil$. Symbolic handles, identifiers, pointers, keys, and index entries are standard computational ways to supply that information. In the instantiated formal domain, a neural encoder paired with an injective symbolic handle achieves zero-error identity recovery if and only if the handle distinguishes entities within each semantic fiber (\LH{NSL1}--\LH{NSL6}).
```

### Why
- slightly more TIT-like
- less manifesto-like
- still preserves the systems payoff

---

## 9. Lean / formalization mention

### Problem
Local filesystem paths should not appear in submission-facing prose.

### Rule

- If there is a public repository, cite that.
- If not, say only `formalized in Lean 4` or mention supplemental material.
- Do not mention author-local paths in the manuscript.

### Draft sentence

```latex
All formal witness theorems cited in the paper have been mechanized in Lean 4; the mechanization is supplementary to the prose theorem arc rather than required for reading it.
```

---

## Execution Order

1. Add ML-reader forward pointer in `01_introduction.tex`
2. Trim `02_compression_framework.tex` down to the genuinely generic framework only
3. Move formal entity-attribute / information-barrier material into the later formal-instantiation section in `06_applications.tex`
4. Check `04_kolmogorov_witness.tex` for any remaining PL-first theorem statements
5. Reorder `06_applications.tex` so ML instantiation visibly comes first
6. Do a final tone pass on abstract/conclusion/formalization mentions
7. Rebuild and verify

---

## Success Criterion

After this restructuring, a JSAIT reviewer should be able to summarize the paper as:

> The paper studies a generic representation map, derives exact zero-error, adaptive, and deterministic-corner distortion laws from its collision fibers, and then illustrates those laws in learned-representation systems and in a machine-checked formal entity-attribute model.

That is the target reading experience.
