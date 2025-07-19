# OpenHCS Documentation Evolution Analysis
**Leveraging Existing Sphinx Structure with OpenHCS Updates**
*Generated: 2025-07-18*
*Source: TUI-generated script + existing Sphinx docs analysis*

---

## Executive Summary

**🎯 KEY DISCOVERY**: Three excellent documentation sources need integration: Sphinx structure + architecture docs + TUI script patterns.

**📊 THREE-PRONGED DOCUMENTATION STRATEGY**:
- **SPHINX BASE**: Excellent structure (concepts/, api/, user_guide/) as foundation
- **ARCHITECTURE CONTENT**: Rich `docs/architecture/` content with OpenHCS specifics
- **TUI SCRIPT TODO**: Real working patterns as comprehensive requirements list
- **INTEGRATION**: Combine all three sources into updated Sphinx documentation

---

## Existing Sphinx Documentation Analysis

### **1. Three Documentation Sources Analysis**

**✅ SPHINX STRUCTURE** (`docs/source/`):
- Excellent organization: concepts/ → api/ → user_guide/
- Good cross-reference system and progressive complexity
- Needs content updates for OpenHCS evolution

**✅ ARCHITECTURE DOCS** (`docs/architecture/`):
- `function-registry-system.md` - Rich OpenHCS-specific content
- `memory-type-system.md` - Detailed technical documentation
- `tui-system.md` - Modern TUI architecture details
- `pipeline-compilation-system.md` - Advanced OpenHCS concepts
- **Status**: Mix of excellent content + some outdated parts

**✅ TUI SCRIPT PATTERNS** (generated script):
- Real working imports and execution patterns
- Actual configuration structures (GlobalPipelineConfig)
- Production-ready function patterns
- **Status**: Source of truth for what must be documented

### **2. Three-Pronged Integration Strategy**

**🔄 STEP 1: ARCHITECTURE CONTENT AUDIT**
- Extract excellent content from `docs/architecture/`
- Fact-check against current codebase using context engine
- Remove outdated/hallucinated content
- Identify content ready for Sphinx integration

**🔄 STEP 2: SPHINX STRUCTURE MAPPING**
- Map architecture content to appropriate Sphinx sections
- Update Sphinx examples with TUI script patterns
- Integrate architecture deep-dives into concepts/
- Add new API documentation for missing concepts

**🔄 STEP 3: TUI SCRIPT PATTERN MATCHING**
- Use TUI script as comprehensive todo list
- Ensure every concept in script is documented
- Validate all examples follow TUI script patterns
- Verify documentation patterns match codebase reality (no execution)

### **3. TUI Script as Documentation Todo List**

**📋 EVERY CONCEPT IN TUI SCRIPT MUST BE DOCUMENTED**:
- ✅ PipelineOrchestrator (exists, needs updating)
- ✅ FunctionStep (exists, needs real patterns)
- ❌ GlobalPipelineConfig (missing, needs new docs)
- ❌ Two-phase execution (needs updating)
- ❌ Function patterns (single, chain, dict)
- ❌ Processing backends (specific imports)
- ❌ VFS/Zarr configuration (missing)
- ❌ TUI workflow (completely new)

---

## Documentation Update Strategy

### **1. Systematic Content Updates**

**🔄 PHASE 1: Module Path Updates**
- Replace ALL `ezstitcher` → `openhcs` references
- Update import statements in ALL examples
- Verify module paths match current codebase structure

**🔄 PHASE 2: Example Modernization**
- Replace ALL examples with TUI-generated script patterns
- Use real function objects instead of legacy patterns
- Show actual working imports and configurations

**🔄 PHASE 3: New Concept Addition**
- Add GlobalPipelineConfig documentation
- Add function registry system docs
- Add memory management system docs
- Add TUI workflow documentation

### **2. Existing Documentation Strengths to Preserve**

**✅ EXCELLENT CONCEPTUAL FLOW**:
- `concepts/architecture_overview.rst` - Keep overall structure
- `concepts/pipeline_orchestrator.rst` - Update examples, keep concepts
- `concepts/step.rst` - Update to FunctionStep patterns
- `user_guide/` progression - Keep basic → intermediate → advanced

**✅ GOOD CROSS-REFERENCE SYSTEM**:
- Links between concepts and API docs
- Progressive disclosure of complexity
- Related classes sections

### **3. Complete Working Example**

**💻 FULL PIPELINE FROM TUI-GENERATED SCRIPT**:
```python
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add OpenHCS to path
sys.path.insert(0, "/home/ts/code/projects/openhcs")

from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.config import GlobalPipelineConfig
from openhcs.constants.constants import VariableComponents

def create_pipeline():
    plate_paths = ['/path/to/plate']
    global_config = GlobalPipelineConfig(num_workers=5, ...)

    steps = []
    step_1 = FunctionStep(
        func=[(tophat, {'selem_radius': 50})],
        name="preprocess",
        variable_components=[VariableComponents.SITE],
        force_disk_output=False
    )
    steps.append(step_1)

    return plate_paths, {plate_paths[0]: steps}, global_config

def run_pipeline():
    plate_paths, pipeline_data, global_config = create_pipeline()

    for plate_path in plate_paths:
        steps = pipeline_data[plate_path]
        orchestrator = PipelineOrchestrator(plate_path, global_config=global_config)
        orchestrator.initialize()

        compiled_contexts = orchestrator.compile_pipelines(steps)
        results = orchestrator.execute_compiled_plate(steps, compiled_contexts, max_workers=5)

        return results

if __name__ == "__main__":
    run_pipeline()
```

---

## Documentation Requirements

### **Primary Interface: TUI**
- OpenHCS is primarily used through `openhcs-tui`
- TUI generates executable Python scripts like the example above
- Scripts are self-contained and production-ready

### **API Documentation Focus**
- Document PipelineOrchestrator as main execution engine
- Document FunctionStep patterns (single, chain, dict)
- Document GlobalPipelineConfig structure
- Document specific function imports from processing backends

### **Real Usage Patterns**
- Users primarily interact through TUI
- Advanced users can modify generated scripts
- Scripts show the real API patterns that work
- Two-phase execution model is the core architecture
