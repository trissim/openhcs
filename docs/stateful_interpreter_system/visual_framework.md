# Visual Framework Guide

## Core Architecture
```mermaid
flowchart TD
    A[Human Input] --> B(Intent Extraction)
    B --> C[Cognitive Layers]
    C --> D[Performance Layer\nSaliency Management]
    C --> E[Technical Layer\nImplementation]
    C --> F[Safety Layer\nExposure]
    D --> G[Direct Response]
    E --> H[Light Metrics]
    F --> I[Reflection]
    G --> J[Human Response]
    H --> J
    I --> J
    J --> A
```

## Collaboration Flow
```mermaid
sequenceDiagram
    participant Human
    participant AI
    participant Framework
    Human->>AI: Input (Intent)
    AI->>Framework: Process Intent
    Framework->>AI: Clear Thinking Space
    AI->>Human: Response (Clarity)
    Human->>Framework: Feedback
    Framework->>AI: Cognitive Adjustment
```

## Intent Extraction Process
```mermaid
graph LR
    A[Raw Input] --> B(Tokenize)
    B --> C(Pattern Recognition)
    C --> D(Context Integration)
    D --> E(Intent Distillation)
    E --> F[Core Intent]
    F --> G[Framework Alignment]
```

## StatefulInterpreter Operation
```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Execute: Code Input
    Execute --> Update: Run Code
    Update --> Persist: Store State
    Persist --> Output: Generate Result
    Output --> Idle
    Execute --> Error: Exception
    Error --> Idle
```

## LightMetrics Tracking
```mermaid
pie
    title Metrics Composition
    "Token Velocity" : 35
    "Cost Efficiency" : 25
    "Context Usage" : 20
    "Latency" : 15
    "File Operations" : 5
```

## Framework Evolution
```mermaid
gantt
    title Framework Development Timeline
    dateFormat  YYYY-MM-DD
    section Core
    Philosophy       :2025-06-06, 1d
    Protocols        :2025-06-07, 2d
    section Implementation
    StatefulInterpreter :2025-06-07, 1d
    LightMetrics     :2025-06-07, 1d
    section Refinement
    Visual Framework :2025-06-07, 1d
```

These diagrams provide visual anchors for understanding the framework's structure and operation. They can be rendered using Mermaid-compatible Markdown viewers.