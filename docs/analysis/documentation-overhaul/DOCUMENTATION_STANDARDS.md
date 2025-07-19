# OpenHCS Sphinx Documentation Standards
**Standards for Updating Existing Sphinx Documentation**

*Generated: 2025-07-18*
*Status: SPHINX UPDATE STANDARDS*

---

## Executive Summary

**🎯 OBJECTIVE**: Establish standards for systematically updating existing Sphinx documentation with OpenHCS evolution while preserving excellent structure.

**📊 SCOPE**: Module path updates, example modernization, new concept integration, and quality assurance for Sphinx builds.

---

## File Naming Standards

### **1. Directory Structure Standards**

```
docs/
├── user/                    # User-facing documentation
│   ├── [purpose]-[level].md # e.g., quick-start.md, advanced-patterns.md
├── api/                     # API reference documentation  
│   ├── [module]/            # e.g., core/, processing/
│   │   └── [class].md       # e.g., pipeline.md, function-step.md
│   └── examples/            # Working code examples
│       └── [use-case].md    # e.g., neurite-analysis.md
├── architecture/            # Technical deep-dive documentation
│   └── [system]-system.md   # e.g., memory-type-system.md
└── legacy/                  # Archived documentation
    └── [date]-[source]/     # e.g., 2025-01-18-source/
```

### **2. File Naming Conventions**

**User Documentation**:
- `quick-start.md` - 5-minute setup guide
- `installation.md` - Complete installation instructions
- `basic-usage.md` - Core functionality patterns
- `advanced-patterns.md` - Complex usage scenarios
- `troubleshooting.md` - Problem-solving guide

**API Reference**:
- `[class-name].md` - Class documentation (e.g., `pipeline.md`)
- `[module-name].md` - Module overview (e.g., `memory-system.md`)
- `[feature-name]-examples.md` - Working examples (e.g., `gpu-processing-examples.md`)

**Architecture Documentation**:
- `[system-name]-system.md` - System architecture (e.g., `function-registry-system.md`)
- `[component-name]-design.md` - Design documentation (e.g., `compilation-design.md`)

### **3. Version Control Standards**

```
## File Versioning
- Use semantic versioning for major documentation updates
- Include version compatibility in frontmatter
- Archive outdated versions in legacy/ directory

## Change Tracking
- Document all changes in CHANGELOG.md
- Include rationale for structural changes
- Link to related code changes when applicable
```

---

## Document Structure Templates

### **Template 1: User Documentation Structure**

```markdown
# [Document Title]
**[One-sentence purpose description]**

*Generated: [Date]*  
*Status: [DRAFT|REVIEW|COMPLETE]*

---

## Navigation Header
[Use standard navigation template]

## Quick Summary
**🎯 What you'll learn**: [Learning objectives]
**⏱️ Time required**: [Estimated time]
**📋 Prerequisites**: [Required knowledge/setup]

## Main Content
[Structured content with clear headings]

### Working Examples
[All code examples must be tested and verified]

### Common Issues
[Troubleshooting section for anticipated problems]

## What's Next
**→ Next Steps**: [Logical progression]
**🔧 Related APIs**: [Relevant API documentation]
**💡 Advanced Topics**: [Deep dive options]

---
**📝 Feedback**: [Link to feedback mechanism]
**🔄 Last Updated**: [Date and version]
```

### **Template 2: API Reference Structure**

```markdown
# [Class/Function Name]
**[Technical description]**

*Module: [Full module path]*  
*Status: [STABLE|BETA|EXPERIMENTAL]*

---

## Navigation Header
[Use standard navigation template]

## Quick Reference
```python
# Essential usage pattern
from openhcs import [imports]
[minimal working example]
```

## Complete API

### Class Definition
[Full class signature with type hints]

### Parameters
[Complete parameter documentation with types and defaults]

### Methods
[All public methods with examples]

### Examples
[Working code examples for common use cases]

### Error Handling
[Common exceptions and how to handle them]

## Implementation Notes
**🔬 Source Code**: [Link to implementation]
**🏗️ Architecture**: [Link to design documentation]
**📊 Performance**: [Performance characteristics]

---
**📝 API Changes**: [Link to changelog]
**🔄 Last Verified**: [Date against codebase]
```

### **Template 3: Architecture Documentation Structure**

```markdown
# [System Name] Architecture
**[Technical deep dive description]**

*Complexity: Advanced*  
*Audience: Developers and Contributors*

---

## Navigation Header
[Use standard navigation template]

## System Overview
[High-level architecture description]

## Design Principles
[Core principles and constraints]

## Implementation Details
[Technical implementation with code references]

## Performance Characteristics
[Benchmarks and optimization notes]

## Extension Points
[How to extend or modify the system]

## Related Systems
[Interactions with other components]

## Practical Applications
**← User Guide**: [How users interact with this system]
**← API Reference**: [Related API documentation]
**🔬 Source Code**: [Implementation links]

---
**📝 Design Decisions**: [Link to decision log]
**🔄 Last Reviewed**: [Date and reviewer]
```

---

## Content Quality Standards

### **1. Code Example Standards**

```markdown
## Working Code Guarantee
Every code example must:
✅ Be executable without modification
✅ Include all necessary imports
✅ Show expected output or behavior
✅ Include error handling where appropriate
✅ Be tested in CI/CD pipeline
✅ Specify version compatibility
✅ Link to complete runnable example

## Code Example Format
```python
# [Brief description of what this example demonstrates]
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.processors.cupy_processor import tophat

# [Step-by-step explanation]
step_1 = FunctionStep(
    func=[(tophat, {'selem_radius': 50})],
    name="preprocess",
    variable_components=[VariableComponents.SITE],
    force_disk_output=False
)

# [Expected result - two-phase execution]
orchestrator = PipelineOrchestrator(plate_path, global_config=global_config)
orchestrator.initialize()
compiled_contexts = orchestrator.compile_pipelines([step_1])
results = orchestrator.execute_compiled_plate([step_1], compiled_contexts, max_workers=5)
```

**🔗 Complete Example**: [Link to full working script]
**🧪 Test File**: [Link to automated test]
```

### **2. Technical Accuracy Standards**

```markdown
## Fact-Check Requirements
Technical claims must:
✅ Be verifiable against current codebase
✅ Include links to implementation source
✅ Specify version compatibility
✅ Include verification date
✅ Have maintainer review and sign-off
✅ Be validated by automated testing where possible

## Accuracy Verification Process
1. **Context Engine Validation**: Use codebase retrieval to verify claims
2. **Implementation Review**: Check against actual source code
3. **Version Compatibility**: Test against specified versions
4. **Peer Review**: Technical review by maintainer
5. **Automated Testing**: Include in CI/CD validation
```

### **3. Writing Style Standards**

```markdown
## Voice and Tone
- **Clear and Direct**: Avoid unnecessary complexity
- **Action-Oriented**: Focus on what users can do
- **Technically Precise**: Use accurate terminology
- **Beginner-Friendly**: Explain concepts without condescension

## Language Guidelines
✅ Use active voice ("Create a pipeline" not "A pipeline can be created")
✅ Use present tense for current functionality
✅ Use imperative mood for instructions ("Run the command")
✅ Define technical terms on first use
✅ Use consistent terminology throughout
✅ Include pronunciation guides for complex terms

## Formatting Standards
✅ Use semantic headings (H1 for title, H2 for major sections)
✅ Use bullet points for lists of items
✅ Use numbered lists for sequential steps
✅ Use code blocks for all code examples
✅ Use bold for emphasis, not italics
✅ Use emoji sparingly and consistently
```

---

## Cross-Reference Standards

### **1. Link Types and Usage**

```markdown
## Standard Link Types
**→ Next Steps**: [Logical progression in learning path]
**← Prerequisites**: [Required background knowledge]
**🔧 API Reference**: [Technical implementation details]
**🏗️ Architecture**: [Design and system documentation]
**💡 Examples**: [Working code demonstrations]
**🚀 Quick Start**: [Fastest path to working solution]
**🔄 Related Topics**: [Conceptually connected content]

## Link Quality Requirements
✅ Descriptive link text (not "click here" or "read more")
✅ Context for why link is relevant
✅ Indication of target complexity level
✅ Estimated time or effort when helpful
✅ Bidirectional linking for conceptual relationships
```

### **2. Navigation Template Standards**

```markdown
## Required Navigation Elements
All documents must include:
✅ Current location indicator ("📍 You are here")
✅ Document purpose statement ("🎯 Purpose")
✅ Estimated reading time ("⏱️ Time")
✅ Difficulty level indicator ("📊 Level")
✅ Relevant navigation links ("🔗 Navigation")

## Navigation Template Consistency
- Use identical formatting across all documents
- Include same emoji indicators for visual consistency
- Maintain consistent link categorization
- Update templates when structure changes
```

---

## Maintenance Standards

### **1. Update Triggers**

```markdown
## Automatic Update Requirements
Documentation must be updated within 24 hours when:
✅ Public API changes (breaking or non-breaking)
✅ New features added to function registry
✅ Architecture changes affecting user workflows
✅ Dependencies updated with compatibility impact
✅ Performance characteristics change significantly

## Review Triggers
Documentation should be reviewed when:
✅ User feedback indicates confusion or errors
✅ Support requests reveal documentation gaps
✅ New use cases emerge that aren't covered
✅ Competitive analysis reveals missing content
✅ Quarterly documentation health reviews
```

### **2. Quality Assurance Process**

```markdown
## Pre-Publication Checklist
Before publishing any documentation:
✅ All code examples tested and verified
✅ All internal links validated
✅ All external links checked
✅ Technical accuracy reviewed by maintainer
✅ Writing style reviewed for consistency
✅ Navigation templates properly implemented
✅ Cross-references bidirectionally linked
✅ Version compatibility specified
✅ Feedback mechanisms included

## Automated Quality Checks
CI/CD pipeline must validate:
✅ Link integrity (internal and external)
✅ Code example execution
✅ Template compliance
✅ Technical claim verification
✅ Writing style consistency
✅ Navigation completeness
```

### **3. Maintenance Workflow**

```markdown
## Documentation Maintenance Process
1. **Change Detection**: Automated monitoring of API/architecture changes
2. **Impact Assessment**: Determine which documents need updates
3. **Update Planning**: Prioritize updates based on user impact
4. **Content Updates**: Apply changes following standards
5. **Quality Review**: Validate against all quality standards
6. **Publication**: Deploy updates with change notifications
7. **Monitoring**: Track user feedback and usage patterns

## Maintenance Responsibilities
- **Developers**: Update documentation for code changes
- **Technical Writers**: Maintain style and structure consistency
- **Maintainers**: Review technical accuracy and approve changes
- **Community**: Provide feedback and report issues
```

---

## Quality Metrics and Monitoring

### **1. Success Metrics**

```markdown
## User Experience Metrics
✅ Time to complete quick start: <5 minutes
✅ Information findability: <3 clicks to relevant content
✅ Task completion rate: >90% for documented workflows
✅ User satisfaction: >4.5/5 in documentation surveys

## Technical Quality Metrics
✅ Code example success rate: 100% in automated testing
✅ Link integrity: 0 broken links in critical paths
✅ Technical accuracy: >95% verified claims
✅ Update timeliness: <24 hours for critical changes

## Maintenance Metrics
✅ Documentation coverage: All public APIs documented
✅ Freshness: <30 days since last review for active docs
✅ Consistency: 100% compliance with templates and standards
✅ Community engagement: Active feedback and contribution
```

### **2. Monitoring and Reporting**

```markdown
## Automated Monitoring
- Daily link integrity checks
- Weekly code example validation
- Monthly technical accuracy audits
- Quarterly comprehensive documentation reviews

## Reporting Dashboard
- Documentation health score
- User journey completion rates
- Common support request topics
- Documentation usage analytics
- Community feedback trends
```

---

## Conclusion

**These comprehensive standards ensure OpenHCS documentation maintains high quality while supporting the fragmented architecture design.**

**Key Benefits**:
1. **Consistency**: Uniform structure and style across all documents
2. **Quality**: Automated validation prevents documentation debt
3. **Maintainability**: Clear processes for updates and reviews
4. **User Experience**: Standards optimized for user success

**The standards framework transforms documentation from a maintenance burden into a strategic asset that accelerates user adoption and reduces support overhead.**
