---
title: 'pyqt-reactor: A Reactive Application Framework for PyQt6 Desktop Software'
tags:
  - Python
  - PyQt6
  - reactive
  - GUI
  - desktop applications
  - state management
authors:
  - name: Tristan Simas
    orcid: 0000-0002-6526-3149
    affiliation: 1
affiliations:
  - name: McGill University
    index: 1
date: 13 January 2026
bibliography: paper.bib
---

# Summary

Every PyQt application with configuration forms eventually reimplements the same patterns: list managers with add/edit/delete, cross-window state synchronization, hierarchical settings inheritance, visual feedback for unsaved changes. These patterns are tedious to build correctly and painful to maintain.

`pyqt-reactor` eliminates this work. Declare what your UI should do; the framework handles how:

```python
class PipelineEditor(AbstractManagerWidget):
    TITLE = "Pipeline Editor"
    BUTTON_CONFIGS = [("Add", "add_step"), ("Del", "del_step"), ("Edit", "edit_step")]
    ITEM_HOOKS = {'backing_attr': 'steps', 'selection_signal': 'step_selected'}
    PREVIEW_FIELD_CONFIGS = ['streaming_config', 'output_config']
```

This configuration inherits complete CRUD infrastructure, cross-window reactivity, flash animations, dirty tracking, undo/redo, and live preview formatting. When a value changes in one window, every related window updates. When a user clears a field, it inherits from the parent scope. When an item is modified, it flashes. The framework composes these behaviors; applications declare their structure.

# Statement of Need

Desktop applications with complex state share a common fate: teams rebuild the same infrastructure repeatedly. Cross-window synchronization. Hierarchical configuration. CRUD list managers. Visual feedback systems. Each project reimplements these patterns from scratch because no framework provides them.

| Capability | Qt Designer | magicgui | Streamlit | React | pyqt-reactor |
|------------|:-----------:|:--------:|:---------:|:-----:|:------------:|
| Cross-window sync | — | — | — | ✓ | ✓ |
| Hierarchical config | — | — | — | ✓¹ | ✓ |
| CRUD abstractions | — | — | — | — | ✓ |
| Desktop native | ✓ | ✓ | — | — | ✓ |
| Type-driven widgets | — | ✓ | ✓ | — | ✓ |
| O(1) animations | — | — | — | — | ✓ |

¹ React Context provides hierarchy but requires manual inheritance implementation.

**Qt Designer** handles layout, not behavior. **magicgui** [@magicgui] generates widgets from signatures but stops there—no synchronization, no CRUD patterns. **Streamlit** [@streamlit] is reactive but web-only. **React** [@react] pioneered declarative UI but JavaScript cannot introspect types at runtime the way Python can.

`pyqt-reactor` fills the gap: the framework layer that PyQt6 lacks.

# Software Design

## Cross-Window Reactivity

Change a value in Window A; Window B updates. No save button. No reload. No explicit synchronization code.

The mechanism: `FieldChangeDispatcher` routes changes with reentrancy guards. `ObjectStateRegistry` notifies listeners via `contextvars` isolation. `CrossWindowPreviewMixin` debounces updates to prevent storms during rapid typing. Windows refresh only affected fields based on type-hierarchy matching.

**Example**: A user opens two windows—one editing a Pipeline's global settings, another editing a Step's settings. The Step inherits from the Pipeline. When the user changes `num_workers=8` in the Pipeline window, the Step window immediately shows the new inherited value in its placeholder text. No explicit synchronization code. The framework detects that both windows reference the same configuration hierarchy and updates both.

This is what React does for web components. `pyqt-reactor` brings it to PyQt6 desktop applications.

## CRUD Abstractions

`AbstractManagerWidget` is a template-method base class for list managers. Declare your structure:

- **BUTTON_CONFIGS**: Toolbar buttons mapping to actions
- **ITEM_HOOKS**: Selection tracking, backing storage, signal emission
- **PREVIEW_FIELD_CONFIGS**: Fields shown in list item previews
- **LIST_ITEM_FORMAT**: Multiline display with formatters

Implement the domain hooks. The base class provides everything else: list widget creation, selection with dirty-check prevention, drag-and-drop with undo, cross-window preview updates, flash animations, dirty tracking. Build a complete list manager by declaring what it manages.

**Example**: A pipeline editor needs to manage a list of processing steps. Without `AbstractManagerWidget`, the developer would write:
- A `QListWidget` subclass with selection handling
- Add/delete/edit button handlers
- Dirty tracking (mark as modified when items change)
- Undo/redo integration
- Cross-window updates when steps are modified elsewhere
- Flash animations when items are added/removed

With `AbstractManagerWidget`, the developer declares:
```python
class StepListManager(AbstractManagerWidget):
    ITEM_HOOKS = {'backing_attr': 'steps', 'selection_signal': 'step_selected'}
    BUTTON_CONFIGS = [("Add", "add_step"), ("Delete", "del_step"), ("Edit", "edit_step")]
    PREVIEW_FIELD_CONFIGS = ['function_name', 'parameters']
```

And implements three methods: `add_step()`, `del_step()`, `edit_step()`. The base class handles everything else.

## Hierarchical Configuration

Forms integrate with ObjectState [@objectstate] for dual-axis inheritance. Values resolve through both context hierarchy (Step → Pipeline → Global) and class hierarchy (StepConfig → PipelineConfig → BaseConfig).

The key insight: `None` means "inherit." Placeholder text shows the inherited value in real-time. Users see what they'll get. Clear a field to restore inheritance. Type a value to override. The UI model and data model are unified—no synchronization bugs, no hidden state.

## Protocol-Based Extensibility

The framework knows nothing about your domain. Protocol classes (`FunctionRegistryProtocol`, `LLMServiceProtocol`, `CodegenProvider`, `PreviewFormatterRegistry`) define integration points. Register implementations at startup; the framework calls them without knowing concrete types. Swap AI providers, function registries, or code generators without touching framework code.

## Flash Animation Architecture

Visual feedback matters. Modified items flash. Dirty fields highlight. But naive implementations scale O(n) with widget count—unacceptable for complex UIs.

Game-engine solution: `GlobalFlashCoordinator` runs a single timer, pre-computes all interpolated colors. `WindowFlashOverlay` renders every flash rectangle in one `paintEvent`. Cost scales with animating elements, not total widgets.

## Universal Callable Introspection

Forms generate from any callable: functions, dataclasses, classes (via `__init__`), or objects with `__call__`. The `python-introspect` [@pythonintrospect] dependency provides unified parameter analysis—one interface for all callable types. Pass a function, get a form. Pass a dataclass, get a form. Pass a callable object, get a form. The framework doesn't care which; the introspection layer normalizes them.

## Type Dispatch

Widget creation uses discriminated unions. `ParameterInfo` subclasses define `matches()` predicates; the factory selects the first match. Services dispatch by class name (`_reset_OptionalDataclassInfo`). No dispatch tables. Exhaustive handling. Type-safe throughout.

# Research Application

`pyqt-reactor` powers OpenHCS, an open-source high-content screening platform for automated microscopy. Multiple synchronized windows. Deeply nested configuration hierarchies. Multi-level scopes (Global → Plate → Pipeline → Step). Real-time inherited value preview. Git-style undo/redo with branching timelines.

Step-level settings inherit from pipeline defaults, which inherit from global configuration. Function editors generate forms from any callable signature—arbitrary Python functions become pipeline steps. Responsive updates across all windows during active editing. No perceptible lag.

## Broader Applicability

The patterns generalize. Video editors need timeline sync and effect parameter inheritance. Game engines need entity inspectors with prefab hierarchies. CAD software needs assembly parameters flowing to child components. Audio DAWs need track configs inheriting from master settings.

These are the same patterns. `pyqt-reactor` provides them once.

# AI Usage Disclosure

Generative AI (Claude claude-sonnet-4-5) assisted with code generation and documentation. All content was reviewed, tested, and integrated by human developers. Core architectural decisions—CRUD abstractions, cross-window reactivity, game-engine animation, ObjectState integration—were human-designed based on production requirements from OpenHCS development.

# Acknowledgements

This work was supported in part by the Fournier lab at the Montreal Neurological Institute, McGill University.

# References
