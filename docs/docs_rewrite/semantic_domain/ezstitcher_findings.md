## Process Instructions
**User directive received at 2025-06-07 07:13:**
1. Read and document all legacy EzStitcher docs in `docs/source/` to understand the semantic domain
2. Analyze OpenHCS backend source (focus on `textual_tui` directory LAST)
3. Create new documentation upgrade plan based on findings
4. Use CLI persistence for tracking and materialize all insights to files
5. Systematically track and fill gaps in understanding
# EzStitcher Semantic Domain Review

## Core Concepts
- **Processing Steps**:
  - Fundamental units of processing operations
  - Stateless design with StepResult objects
  - Predefined types: ZFlatStep, FocusStep, CompositeStep, PositionGenerationStep, ImageStitchingStep
  - Variable components define grouping logic
  - Group_by maps function dictionaries to component values
- **Processing Context**:
  - Maintains state during pipeline execution
  - Holds input/output directories, well filter, configuration
  - Provides access to orchestrator services
  - Stores results from previous steps
- **Storage Adapter**:
  - Abstract storage layer for pipeline artifacts
  - Supports memory and zarr storage modes
  - Key-value interface for numpy arrays
  - Context helper methods for storage operations
- **Module Structure**:
  - Interface-implementation separation
  - Schema-first design for data validation
  - Explicit component registration
  - Unidirectional dependencies to prevent cycles
  - Initialization discipline to avoid import-side effects
- **Directory Management**:
  - Workspace path protects original data
  - Automatic directory resolution for pipeline steps
  - Configurable directory suffixes
  - Best practices for step configuration
- **Function Handling**:
  - Supports single functions, with args, lists, dictionaries
  - stack() utility adapts single-image functions
  - Predefined steps for common operations

## Patterns & Metaphors
- **Pipeline Metaphor**: Processing as sequence of steps
- **Microscope Agnosticism**: Works with multiple microscope types
- **Automatic Detection**: Inferring microscope/image organization
- **Visual Explanation**: Diagrams for plate, channel, z-stack concepts
- **Progressive Disclosure**: Simple → complex options in parameters

## Documentation Structure
- **Getting Started First**: Quick start guide before deep concepts
- **Feature-Centric Organization**: Grouped by functionality
- **Progressive Complexity**: Basic → Intermediate → Advanced usage
- **Modular Sections**: Clear separation between concepts, usage, and API
- **Visual Anchors**: Logo image at top for branding
- **Code-Centric Guidance**: Abundant code examples for immediate application

## Open Questions
1. How are StepResults propagated between steps?
2. What specific services does orchestrator provide via context?
3. How does StorageAdapter handle large datasets?
4. What are the exact module dependency rules?
5. How are custom directory suffixes configured?
6. What are performance implications of different storage modes?

## Insights for OpenHCS
1. **Documentation Structure**:
   - Separate sections for core concepts: steps, context, storage, modules, directories
   - Include diagrams for module dependencies and directory flows
   - Provide function handling examples for common scenarios
   
2. **Architectural Patterns**:
   - Adopt stateless step design with explicit result objects
   - Implement context-based service access
   - Use storage abstraction for efficient data handling
   - Enforce module organization principles
   
3. **Best Practices**:
   - Prefer predefined steps for common operations
   - Follow directory resolution best practices
   - Use schemas for data validation
   - Explicitly initialize components