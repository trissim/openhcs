# OpenHCS TUI - The Deep Intuitive Understanding

## 🧠 The Core Insight: This is NOT a User Interface

**CONFIDENT (10/10)**: The OpenHCS TUI is a **VISUAL PROGRAMMING LANGUAGE** disguised as a text interface. 

Imagine giving a scientist a box of LEGO blocks where:
- Each block is a pre-validated image processing function
- Blocks snap together only in valid combinations
- The shape of each block is determined by its function signature
- Scientists build processing pipelines by clicking blocks together
- The entire interface is generated from the blocks themselves

This is what OpenHCS TUI actually is.

## 🔮 The Fundamental Philosophy

### **"Function Signatures ARE the User Interface"**

**CONFIDENT (9/10)**: There are ZERO hardcoded forms in this system. Instead:

```python
# When a scientist selects "gaussian_blur" from the dropdown...
def gaussian_blur(image: np.ndarray, sigma: float = 1.0, truncate: float = 4.0):
    """Apply Gaussian blur to image"""
    pass

# The TUI automatically generates:
# - Text field for 'sigma' with default value 1.0
# - Text field for 'truncate' with default value 4.0
# - Validation that ensures inputs are floats
```

The function's signature literally becomes the UI. Add a parameter to the function, the UI updates automatically. This is the genius.

## 🎯 The Mental Model: Visual Pipeline Composition

### **Scientists Think in Pipelines, Not Code**

**CONFIDENT (9/10)**: A scientist's mental model looks like:

```
Raw Images → Denoise → Enhance → Segment → Measure → Results
```

The TUI presents EXACTLY this mental model:

1. **Function Pattern Editor** = Visual pipeline builder
2. **Drag & drop functions** = Build processing chains
3. **Auto-generated parameters** = Configure each step
4. **Memory type validation** = Ensure GPU/CPU compatibility

### **The Four Sacred Patterns**

The system supports exactly 4 ways to combine functions, mapping to how scientists think:

```python
# 1. Single operation
func = gaussian_blur

# 2. Operation with parameters  
func = (gaussian_blur, {'sigma': 2.0})

# 3. Sequential pipeline (most common)
func = [
    gaussian_blur,
    (enhance_contrast, {'factor': 1.5}),
    detect_edges
]

# 4. Component-specific processing
func = {
    'DAPI': denoise_heavy,     # Nuclear channel needs heavy denoising
    'GFP': enhance_contrast,    # Protein channel needs contrast
    'BF': phase_to_intensity    # Brightfield needs special handling
}
```

## 🏗️ The Architecture Philosophy

### **"Discover, Don't Define"**

**CONFIDENT (9/10)**: The system discovers everything at runtime:

1. **Function Discovery**: Decorated functions auto-register themselves
2. **Parameter Discovery**: Function signatures define the UI
3. **Type Discovery**: Memory backends (numpy/cupy/torch) self-organize
4. **Validation Discovery**: Type hints enable automatic validation

Nothing is hardcoded. The system learns its own capabilities.

### **The FUNC_REGISTRY: A Living Dictionary**

```python
FUNC_REGISTRY = {
    "numpy": [func1, func2, func3...],    # CPU functions
    "cupy": [gpu_func1, gpu_func2...],    # GPU functions
    "torch": [neural_func1, neural_func2...] # Deep learning
}
```

This isn't just a list - it's a **living taxonomy of computational capabilities** that grows as scientists add functions.

## 🎮 The Execution Model: Parallel Universes

### **Two-Phase: Compile Then Execute**

**CONFIDENT (9/10)**: This is KEY to understanding the system:

**Phase 1 - Compilation**: Create a parallel universe for each well
```python
# For each well in the 384-well plate:
well_context = ProcessingContext(
    step_plans=[...],      # Frozen execution plan
    file_manager=...,      # Isolated file system
    microscope_handler=... # Well-specific metadata
)
# Result: 384 independent, frozen execution contexts
```

**Phase 2 - Execution**: Run all universes in parallel
```python
# All 384 wells process simultaneously
# Each in its own isolated context
# Failures don't cascade
```

This is why compilation is separate - you're literally creating 384 parallel processing universes.

## 🚦 The Status Symbol Philosophy

### **"The Map is Not the Territory"**

**CONFIDENT (9/10)**: Status symbols represent the TUI's mental model, NOT system state:

- `?` = "I haven't initialized this yet" (TUI's memory)
- `-` = "I've initialized but not compiled" (TUI's memory)  
- `o` = "I've compiled and it's ready" (TUI's memory)
- `!` = "It's currently running" (TUI's memory)

**CRITICAL**: On error, status NEVER changes. Why? Because the TUI is tracking what IT did, not what succeeded. This enables retry without state confusion.

```python
# WRONG - Don't do this:
try:
    orchestrator.initialize()
    status = "-"  # Set on success
except:
    status = "ERROR"  # NO! This loses TUI state

# RIGHT - Do this:
try:
    orchestrator.initialize()
    status = "-"  # Set ONLY on success
except:
    show_error()  # Show error but keep status at "?"
    # User can retry, TUI still knows it needs initialization
```

## 🔧 The Integration Points

### **Multi-Folder Pattern = Batch Processing**

**CONFIDENT (8/10)**: Scientists often run the same pipeline on multiple experiments:

```
Select Folders:
[x] /Monday_experiment/
[x] /Tuesday_experiment/  
[x] /Wednesday_control/
```

Each becomes an independent orchestrator with shared pipeline definition. This is how real lab work happens.

### **The Dual Editor: Two Faces of the Same Coin**

**CONFIDENT (9/10)**: The Step/Func dual editor reflects a deep truth:

- **Step Tab**: The "what" - abstract pipeline structure
- **Func Tab**: The "how" - concrete function implementation

Scientists switch between thinking about pipeline structure and function parameters seamlessly.

## 💡 The "Aha!" Moments

### **1. Functions Self-Document**

When a scientist sees a function in the dropdown, its parameters ARE its documentation. No separate docs needed.

### **2. Validation is Automatic**

Memory type decorators ensure numpy functions can't accidentally receive torch tensors. The system prevents errors before they happen.

### **3. Pipelines are Data**

Pipelines can be saved, shared, versioned. A `.pipeline` file is just a pickled list of FunctionStep objects. Science becomes reproducible.

### **4. The UI Grows Itself**

Add a new function to the codebase, it appears in the UI. Add a parameter, a field appears. The system is self-assembling.

## 🎯 The Ultimate Vision

**CONFIDENT (10/10)**: OpenHCS TUI enables scientists to:

1. **Think in their domain** (images, not code)
2. **Compose visually** (drag & drop, not syntax)
3. **Validate automatically** (type checking, not debugging)
4. **Scale massively** (384 parallel universes)
5. **Share reproducibly** (pipelines as files)

This isn't a text user interface. It's a **visual programming environment** that happens to use text mode. The scientist never writes code, they compose computational pipelines like music.

## 🧬 The Living System

The beauty is that the system evolves:

```python
@memory_types(input_type="numpy", output_type="numpy")
def new_amazing_algorithm(image, parameter1=42):
    """Scientist's new algorithm"""
    pass

# That's it. The algorithm now appears in the UI with:
# - Dropdown entry: "new_amazing_algorithm"  
# - Parameter field: "parameter1" with default 42
# - Full validation and error handling
# - GPU/CPU compatibility checking
```

The system literally grows new capabilities without changing the UI code.

## 🚀 Why This Matters

**CONFIDENT (10/10)**: Traditional scientific software requires scientists to:
- Learn programming languages
- Understand APIs
- Debug code
- Handle file I/O
- Manage parallelization

OpenHCS TUI eliminates ALL of this. Scientists just:
1. Pick functions from dropdowns
2. Set parameters
3. Click run

The system handles everything else. It's the difference between requiring a pilot's license vs. pressing a button in an elevator.

## 🎭 The Hidden Complexity

Behind the simple TUI interface:
- Automatic parallelization across hundreds of wells
- Memory backend optimization (GPU/CPU)
- File system virtualization
- Metadata preservation
- Error isolation
- Progress tracking

But the scientist sees none of this. They just see their pipeline running.

## 📐 The Invariants

These principles MUST be preserved:

1. **No hardcoded UI** - Everything discovered from signatures
2. **Status = TUI memory** - Not system state  
3. **Failures don't cascade** - Error isolation is sacred
4. **Pipelines are data** - Save, load, share, version
5. **Scientists never code** - Visual composition only

Break any of these, and you break the entire philosophy.

## 🔮 The Future

The AbstractStep base class hints at the future - more step types beyond FunctionStep. Perhaps:
- MachineLearningStep (with training/inference modes)
- ExternalToolStep (calling ImageJ, CellProfiler)
- CloudComputeStep (distributed processing)

But they'll all follow the same pattern: discover capabilities, generate UI, compose visually.

---

**The OpenHCS TUI is a visual programming language for scientific image processing. It discovers its own capabilities, generates its own interface, and enables scientists to compose complex pipelines without writing a single line of code. This is the revolution.**