# OpenHCS Sphinx Documentation Inventory Matrix

## Overview
This document provides a comprehensive inventory of the legacy Sphinx documentation for fact-checking against the current OpenHCS implementation.

**Documentation Root**: `docs/source/`
**Project Name**: EZStitcher (Legacy name)
**Current Project**: OpenHCS
**Documentation Version**: 0.1.0
**Last Updated**: 2024

## Configuration Files

| File | Purpose | Status | Notes |
|------|---------|--------|-------|
| `docs/source/conf.py` | Sphinx configuration | ⚠️ LEGACY | Project name "EZStitcher", needs update |
| `docs/requirements.txt` | Documentation dependencies | ⚠️ LEGACY | May need OpenHCS-specific deps |

## Main Documentation Structure

### 1. Getting Started (1 file)
| File | Purpose | Priority | Fact-Check Status |
|------|---------|----------|-------------------|
| `getting_started/getting_started.rst` | Installation & quick start | HIGH | 🔍 PENDING |

### 2. User Guide (6 files)
| File | Purpose | Priority | Fact-Check Status |
|------|---------|----------|-------------------|
| `user_guide/introduction.rst` | Introduction to EZStitcher | HIGH | 🔍 PENDING |
| `user_guide/basic_usage.rst` | EZ module usage | HIGH | 🔍 PENDING |
| `user_guide/intermediate_usage.rst` | Custom pipelines | HIGH | 🔍 PENDING |
| `user_guide/advanced_usage.rst` | Advanced customization | MEDIUM | 🔍 PENDING |
| `user_guide/best_practices.rst` | Best practices | MEDIUM | 🔍 PENDING |
| `user_guide/integration.rst` | Integration guide | LOW | 🔍 PENDING |

### 3. Core Concepts (9 files)
| File | Purpose | Priority | Fact-Check Status |
|------|---------|----------|-------------------|
| `concepts/architecture_overview.rst` | System architecture | HIGH | 🔍 PENDING |
| `concepts/pipeline_orchestrator.rst` | Orchestrator concepts | HIGH | 🔍 PENDING |
| `concepts/pipeline.rst` | Pipeline concepts | HIGH | 🔍 PENDING |
| `concepts/step.rst` | Step concepts | HIGH | 🔍 PENDING |
| `concepts/pipeline_factory.rst` | Factory patterns | MEDIUM | 🔍 PENDING |
| `concepts/function_handling.rst` | Function handling | MEDIUM | 🔍 PENDING |
| `concepts/processing_context.rst` | Context management | MEDIUM | 🔍 PENDING |
| `concepts/directory_structure.rst` | Directory organization | MEDIUM | 🔍 PENDING |
| `concepts/basic_microscopy.rst` | Microscopy basics | LOW | 🔍 PENDING |
| `concepts/module_structure.rst` | Module organization | LOW | 🔍 PENDING |
| `concepts/storage_adapter.rst` | Storage concepts | LOW | 🔍 PENDING |

### 4. API Reference (13 files)
| File | Purpose | Priority | Fact-Check Status |
|------|---------|----------|-------------------|
| `api/ez.rst` | EZ module API | HIGH | 🔍 PENDING |
| `api/pipeline_orchestrator.rst` | Orchestrator API | HIGH | 🔍 PENDING |
| `api/pipeline.rst` | Pipeline API | HIGH | 🔍 PENDING |
| `api/steps.rst` | Steps API | HIGH | 🔍 PENDING |
| `api/pipeline_factory.rst` | Factory API | MEDIUM | 🔍 PENDING |
| `api/stitcher.rst` | Stitcher API | MEDIUM | 🔍 PENDING |
| `api/image_processor.rst` | Image processing API | MEDIUM | 🔍 PENDING |
| `api/focus_analyzer.rst` | Focus analysis API | MEDIUM | 🔍 PENDING |
| `api/file_system_manager.rst` | File system API | MEDIUM | 🔍 PENDING |
| `api/microscope_interfaces.rst` | Microscope interfaces | MEDIUM | 🔍 PENDING |
| `api/microscopes.rst` | Microscope implementations | MEDIUM | 🔍 PENDING |
| `api/config.rst` | Configuration API | LOW | 🔍 PENDING |

### 5. Development (5 files)
| File | Purpose | Priority | Fact-Check Status |
|------|---------|----------|-------------------|
| `development/architecture.rst` | Development architecture | MEDIUM | 🔍 PENDING |
| `development/contributing.rst` | Contribution guide | LOW | 🔍 PENDING |
| `development/extending.rst` | Extension guide | LOW | 🔍 PENDING |
| `development/testing.rst` | Testing guide | LOW | 🔍 PENDING |

### 6. Appendices (4 files)
| File | Purpose | Priority | Fact-Check Status |
|------|---------|----------|-------------------|
| `appendices/file_formats.rst` | Supported formats | MEDIUM | 🔍 PENDING |
| `appendices/microscope_formats.rst` | Microscope formats | MEDIUM | 🔍 PENDING |
| `appendices/glossary.rst` | Terminology | LOW | 🔍 PENDING |

## Static Assets
| File | Purpose | Status |
|------|---------|--------|
| `_static/ezstitcher_logo.png` | Project logo | ⚠️ LEGACY NAME |

## Critical Fact-Check Areas

### High Priority Issues to Verify:
1. **Project Name**: Documentation uses "EZStitcher" but current project is "OpenHCS"
2. **Module Structure**: API references may not match current OpenHCS structure
3. **Core Architecture**: Pipeline/orchestrator concepts may have evolved
4. **Function Names**: API function signatures may have changed
5. **Configuration**: Config system may have been redesigned

### Key Questions for Fact-Checking:
1. Does the `ezstitcher` module still exist in OpenHCS?
2. Are `PipelineOrchestrator`, `Pipeline`, `Step` still the core abstractions?
3. Do the documented APIs match current implementations?
4. Are the architectural concepts still accurate?
5. Do the usage examples still work?

## Fact-Check Methodology
1. **Phase 2**: Analyze current OpenHCS implementation
2. **Phase 3**: Systematic comparison of each documentation file
3. **Phase 4**: Identify gaps and outdated information
4. **Phase 5**: Generate comprehensive update recommendations

## Total Documentation Scope
- **Total RST files**: 42
- **High Priority**: 15 files
- **Medium Priority**: 18 files  
- **Low Priority**: 9 files
- **Configuration files**: 2
- **Static assets**: 1

**Estimated fact-check effort**: 500+ tool calls (as anticipated)

## Cross-Reference Analysis

### Documentation Structure Hierarchy
```
index.rst (main)
├── Getting Started (1 file)
│   └── getting_started.rst
├── User Guide (6 files)
│   ├── introduction.rst
│   ├── basic_usage.rst → references concepts/architecture_overview
│   ├── intermediate_usage.rst → references concepts/step, concepts/pipeline
│   ├── advanced_usage.rst → references concepts/function_handling
│   ├── best_practices.rst → referenced by many other files
│   └── integration.rst
├── Core Concepts (9 files)
│   ├── architecture_overview.rst → central hub, referenced by many
│   ├── pipeline_orchestrator.rst → references architecture_overview, directory_structure
│   ├── pipeline.rst → references architecture_overview, directory_structure, best_practices
│   ├── step.rst → referenced by function_handling, pipeline
│   ├── function_handling.rst → references step, advanced_usage, best_practices
│   ├── processing_context.rst
│   ├── directory_structure.rst → references pipeline, step, api docs
│   └── others...
├── API Reference (13 files)
│   └── Heavily cross-referenced from concepts
└── Development & Appendices
```

### Key Cross-Reference Patterns
1. **Central Hub**: `concepts/architecture_overview.rst` is referenced by most other files
2. **Best Practices**: Referenced by multiple concept files for detailed guidance
3. **API Links**: Concept files link to corresponding API documentation
4. **Bidirectional**: Some files reference each other (pipeline ↔ step)

### Critical Reference Points for Fact-Checking
1. **Module imports**: `from ezstitcher import stitch_plate`
2. **Class references**: `:class:`~ezstitcher.core.steps.Step``
3. **Cross-doc links**: `:doc:`../api/pipeline``
4. **Internal refs**: `:ref:`best-practices-pipeline``

## User Journey & Navigation Flow

### Intended Learning Path
1. **Quick Start**: `index.rst` → `getting_started.rst` → immediate usage
2. **Basic Usage**: `user_guide/basic_usage.rst` → EZ module for non-coders
3. **Progressive Complexity**:
   - Basic → `user_guide/intermediate_usage.rst` (custom pipelines)
   - Intermediate → `user_guide/advanced_usage.rst` (implementation details)
4. **Deep Understanding**: `concepts/architecture_overview.rst` → detailed concepts
5. **Reference**: `api/` section for specific implementation details

### Documentation Dependencies & Reading Order

#### Prerequisites (Must read first):
- `getting_started.rst` - Foundation for all usage
- `concepts/basic_microscopy.rst` - Domain knowledge foundation

#### Core Learning Sequence:
1. `user_guide/basic_usage.rst`
2. `concepts/architecture_overview.rst` (central hub)
3. `user_guide/intermediate_usage.rst` → depends on concepts/step, concepts/pipeline
4. `concepts/pipeline_orchestrator.rst` → depends on architecture_overview
5. `concepts/pipeline.rst` → depends on architecture_overview, step
6. `concepts/step.rst` → foundational for pipeline concepts

#### Advanced Topics (Order flexible):
- `user_guide/advanced_usage.rst` → depends on function_handling
- `concepts/function_handling.rst` → depends on step concepts
- `concepts/processing_context.rst`
- `concepts/directory_structure.rst`

#### Reference Materials (As needed):
- `api/*` files → correspond to concept files
- `user_guide/best_practices.rst` → referenced throughout
- `appendices/*` → supplementary information

## Major Documentation Sections Analysis

### 1. Getting Started Section
**Purpose**: Onboarding new users
**Files**: 1 (`getting_started.rst`)
**Target Audience**: All users
**Critical for fact-check**: Installation instructions, first example

### 2. User Guide Section
**Purpose**: Progressive learning from basic to advanced usage
**Files**: 6 (introduction → basic → intermediate → advanced → best_practices → integration)
**Target Audience**: End users at different skill levels
**Critical for fact-check**: API examples, usage patterns, code samples

### 3. Core Concepts Section
**Purpose**: Deep understanding of architecture and design
**Files**: 9 (architecture_overview is central hub)
**Target Audience**: Users who need to understand internals
**Critical for fact-check**: Class relationships, architectural patterns

### 4. API Reference Section
**Purpose**: Detailed technical reference
**Files**: 13 (comprehensive API coverage)
**Target Audience**: Developers and advanced users
**Critical for fact-check**: Function signatures, class definitions, module structure

### 5. Development Section
**Purpose**: Contributor guidance
**Files**: 4 (architecture → extending → testing → contributing)
**Target Audience**: Contributors and maintainers
**Critical for fact-check**: Development setup, extension patterns

### 6. Appendices Section
**Purpose**: Reference materials and supplementary information
**Files**: 3 (glossary, file_formats, microscope_formats)
**Target Audience**: All users (reference)
**Critical for fact-check**: Supported formats, terminology consistency

## Documentation Build System Metadata

### Sphinx Configuration (`docs/source/conf.py`)
- **Project Name**: "EZStitcher" ⚠️ (Should be "OpenHCS")
- **Version**: "0.1.0" ⚠️ (May be outdated)
- **Author**: "trissim"
- **Copyright**: "2024, trissim"
- **Theme**: sphinx_rtd_theme (Read the Docs theme)
- **Extensions**: autodoc, viewcode, napoleon, intersphinx, autosummary, mathjax

### Dependencies (`docs/requirements.txt`)
- **Sphinx**: >=4.0.0
- **Theme**: sphinx-rtd-theme>=1.0.0
- **Scientific**: numpy>=1.20.0, scipy>=1.7.0, matplotlib>=3.4.0
- **Image Processing**: opencv-python>=4.5.0, pillow>=8.0.0, tifffile>=2021.7.2
- **Data**: pandas>=1.3.0, PyYAML>=6.0

### Build Configuration
- **Autodoc**: Enabled with comprehensive member documentation
- **Napoleon**: Google & NumPy docstring support
- **Intersphinx**: Links to external docs (numpy, scipy, matplotlib, pandas, scikit-image)
- **Mock System**: Mocks system libraries for Read the Docs builds

### Static Assets
- **Logo**: `_static/ezstitcher_logo.png` ⚠️ (Legacy branding)
- **Custom CSS/JS**: None detected

### Documentation Standards
- **Docstring Format**: Google & NumPy style
- **Cross-references**: Extensive use of :doc:, :ref:, :class: directives
- **Code Examples**: Python code blocks with syntax highlighting
- **Navigation Depth**: 4 levels maximum

## Comprehensive Documentation Map

### File Organization Matrix
```
docs/source/
├── index.rst (MAIN ENTRY POINT)
├── conf.py (BUILD CONFIG)
├── _static/
│   └── ezstitcher_logo.png
├── getting_started/ (1 file)
│   └── getting_started.rst
├── user_guide/ (6 files)
│   ├── introduction.rst
│   ├── basic_usage.rst ★ HIGH PRIORITY
│   ├── intermediate_usage.rst ★ HIGH PRIORITY
│   ├── advanced_usage.rst
│   ├── best_practices.rst
│   └── integration.rst
├── concepts/ (9 files)
│   ├── index.rst
│   ├── architecture_overview.rst ★ CENTRAL HUB
│   ├── pipeline_orchestrator.rst ★ HIGH PRIORITY
│   ├── pipeline.rst ★ HIGH PRIORITY
│   ├── step.rst ★ HIGH PRIORITY
│   ├── pipeline_factory.rst
│   ├── function_handling.rst
│   ├── processing_context.rst
│   ├── directory_structure.rst
│   ├── basic_microscopy.rst
│   ├── module_structure.rst
│   └── storage_adapter.rst
├── api/ (13 files)
│   ├── index.rst
│   ├── ez.rst ★ HIGH PRIORITY
│   ├── pipeline_orchestrator.rst ★ HIGH PRIORITY
│   ├── pipeline.rst ★ HIGH PRIORITY
│   ├── steps.rst ★ HIGH PRIORITY
│   ├── pipeline_factory.rst
│   ├── stitcher.rst
│   ├── image_processor.rst
│   ├── focus_analyzer.rst
│   ├── file_system_manager.rst
│   ├── microscope_interfaces.rst
│   ├── microscopes.rst
│   └── config.rst
├── development/ (4 files)
│   ├── index.rst
│   ├── architecture.rst
│   ├── extending.rst
│   ├── testing.rst
│   └── contributing.rst
└── appendices/ (3 files)
    ├── index.rst
    ├── glossary.rst
    ├── file_formats.rst
    └── microscope_formats.rst
```

### Priority Classification for Fact-Checking
- ★ **HIGH PRIORITY** (15 files): Core functionality, main APIs, central concepts
- **MEDIUM PRIORITY** (18 files): Supporting features, specialized topics
- **LOW PRIORITY** (9 files): Reference materials, supplementary content

### Phase 1 Discovery Complete ✅
**Total Documentation Scope Confirmed**:
- 42 RST files requiring fact-checking
- 2 configuration files needing updates
- 1 static asset requiring rebranding
- Comprehensive cross-reference network mapped
- User journey and dependencies identified
- Build system metadata extracted

**Ready for Phase 2**: Current System Analysis
