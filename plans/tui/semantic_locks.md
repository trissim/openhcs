# TUI-Specific Semantic Locks

## Overview

This document defines TUI-specific semantic locks that must be enforced in the OpenHCS TUI implementation. These locks are designed to prevent common architectural violations and ensure the TUI adheres to the core architectural principles of OpenHCS.

## Semantic Locks

| ID | Clause | Lock Name | Description | Violation Pattern | Fix Pattern |
|----|--------|-----------|-------------|-------------------|-------------|
| TUI-01 | Clause 315 | TUI_FILEMANAGER_INJECTION | All TUI managers must receive context.file_manager (never create a new one) and pass it, plus positional backend, into every call that might touch I/O. | `FileManager()` constructor call in TUI code; Missing backend parameter in I/O calls | Inject file_manager from context; Add positional backend parameter to all I/O calls |
| TUI-02 | Clause 316 | TUI_BACKEND_SELECT_WIDGET | Any UI element that lets the user pick a backend must emit an explicit string that is forwarded unchanged into step metadata (read_backend/write_backend). No implicit "current backend" drop-down defaults. | Default backend selection; Implicit backend inference | Explicit backend selection widget; Direct mapping to read_backend/write_backend |
| TUI-03 | Clause 317 | TUI_STATUS_THREADSAFETY | All updates to state.operation_status must be serialized through a lock or queue to avoid race conditions when the TUI runs in asyncio + curses threads. | Direct state.operation_status updates; Missing locks or queues | Use thread-safe queue or lock for status updates |
| TUI-04 | Clause 318 | TUI_PLAN_IMMUTABILITY_AFTER_COMPILE | Once the user compiles a pipeline, the TUI must treat that plan as read-only until the user resets; prevents accidental step mutation after validation. | Mutable plan after compilation; Step parameter changes post-validation | Freeze plan after compilation; Require explicit reset before changes |
| TUI-05 | Clause 319 | TUI_NO_VIRTUALPATH_EXPOSURE | TUI rendering/browsing widgets must work only with plain strings. If a backend returns a VirtualPath, it must be converted to str before it reaches the UI. | VirtualPath objects in UI code; VirtualPath imports in TUI modules | Convert VirtualPath to str at I/O boundary; Use only string paths in UI |

## Detection Rules

### TUI_FILEMANAGER_INJECTION (Clause 315)

```python
def detect_filemanager_injection_violations(file_path):
    """Detect violations of TUI_FILEMANAGER_INJECTION."""
    violations = []
    
    # Parse the file
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    # Define a visitor to find violations
    class FileManagerVisitor(ast.NodeVisitor):
        def __init__(self):
            self.violations = []
            self.in_tui_class = False
            
        def visit_ClassDef(self, node):
            # Check if this is a TUI class
            is_tui_class = any(base.id.endswith('Widget') or base.id.endswith('Manager') 
                              for base in node.bases if isinstance(base, ast.Name))
            old_in_tui_class = self.in_tui_class
            self.in_tui_class = is_tui_class
            self.generic_visit(node)
            self.in_tui_class = old_in_tui_class
            
        def visit_Call(self, node):
            # Check for FileManager constructor calls
            if self.in_tui_class and isinstance(node.func, ast.Name) and node.func.id == 'FileManager':
                self.violations.append({
                    'line': node.lineno,
                    'message': 'Clause 315 Violation: TUI classes must receive file_manager from context, not create new instances'
                })
            
            # Check for I/O calls without backend parameter
            if self.in_tui_class and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'file_manager':
                    # Check if backend is passed
                    has_backend = False
                    for keyword in node.keywords:
                        if keyword.arg == 'backend':
                            self.violations.append({
                                'line': node.lineno,
                                'message': 'Clause 315 Violation: Backend must be passed positionally, not as keyword'
                            })
                    
                    # Check if there are enough positional args for backend
                    method_name = node.func.attr
                    if method_name in ['read_image', 'write_image', 'list_files', 'is_dir'] and len(node.args) < 2:
                        self.violations.append({
                            'line': node.lineno,
                            'message': f'Clause 315 Violation: Missing positional backend parameter in call to file_manager.{method_name}'
                        })
            
            self.generic_visit(node)
    
    visitor = FileManagerVisitor()
    visitor.visit(tree)
    violations.extend(visitor.violations)
    
    return violations
```

### TUI_BACKEND_SELECT_WIDGET (Clause 316)

```python
def detect_backend_select_widget_violations(file_path):
    """Detect violations of TUI_BACKEND_SELECT_WIDGET."""
    violations = []
    
    # Parse the file
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    # Define a visitor to find violations
    class BackendSelectVisitor(ast.NodeVisitor):
        def __init__(self):
            self.violations = []
            self.in_backend_widget = False
            
        def visit_ClassDef(self, node):
            # Check if this is a backend selection widget
            is_backend_widget = 'Backend' in node.name and ('Select' in node.name or 'Dropdown' in node.name)
            old_in_backend_widget = self.in_backend_widget
            self.in_backend_widget = is_backend_widget
            self.generic_visit(node)
            self.in_backend_widget = old_in_backend_widget
            
        def visit_Assign(self, node):
            # Check for default backend assignments
            if self.in_backend_widget:
                for target in node.targets:
                    if isinstance(target, ast.Name) and 'default' in target.id.lower() and 'backend' in target.id.lower():
                        self.violations.append({
                            'line': node.lineno,
                            'message': 'Clause 316 Violation: Backend selection widgets must not use implicit defaults'
                        })
            
            self.generic_visit(node)
            
        def visit_FunctionDef(self, node):
            # Check for value getter methods
            if self.in_backend_widget and node.name in ['get_value', 'get_selected', 'on_change']:
                # Look for direct mapping to read_backend/write_backend
                has_direct_mapping = False
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name) and target.id in ['read_backend', 'write_backend']:
                                has_direct_mapping = True
                
                if not has_direct_mapping:
                    self.violations.append({
                        'line': node.lineno,
                        'message': 'Clause 316 Violation: Backend selection must map directly to read_backend/write_backend'
                    })
            
            self.generic_visit(node)
    
    visitor = BackendSelectVisitor()
    visitor.visit(tree)
    violations.extend(visitor.violations)
    
    return violations
```

### TUI_STATUS_THREADSAFETY (Clause 317)

```python
def detect_status_threadsafety_violations(file_path):
    """Detect violations of TUI_STATUS_THREADSAFETY."""
    violations = []
    
    # Parse the file
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    # Define a visitor to find violations
    class StatusThreadsafetyVisitor(ast.NodeVisitor):
        def __init__(self):
            self.violations = []
            self.has_lock = False
            self.has_queue = False
            
        def visit_Module(self, node):
            # Check for lock or queue imports
            for child in node.body:
                if isinstance(child, ast.Import) or isinstance(child, ast.ImportFrom):
                    for name in child.names:
                        if 'lock' in name.name.lower() or 'queue' in name.name.lower():
                            if 'lock' in name.name.lower():
                                self.has_lock = True
                            if 'queue' in name.name.lower():
                                self.has_queue = True
            
            self.generic_visit(node)
            
        def visit_Assign(self, node):
            # Check for direct operation_status assignments
            if isinstance(node.targets[0], ast.Attribute):
                attr = node.targets[0]
                if isinstance(attr.value, ast.Name) and attr.value.id == 'state' and attr.attr == 'operation_status':
                    # Check if this is inside a lock or queue context
                    in_safe_context = False
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.With) and any('lock' in w.optional_vars.id.lower() for w in parent.items if hasattr(w.optional_vars, 'id')):
                            in_safe_context = True
                    
                    if not in_safe_context and not (self.has_lock or self.has_queue):
                        self.violations.append({
                            'line': node.lineno,
                            'message': 'Clause 317 Violation: Updates to state.operation_status must be serialized through a lock or queue'
                        })
            
            self.generic_visit(node)
    
    visitor = StatusThreadsafetyVisitor()
    visitor.visit(tree)
    violations.extend(visitor.violations)
    
    return violations
```

### TUI_PLAN_IMMUTABILITY_AFTER_COMPILE (Clause 318)

```python
def detect_plan_immutability_violations(file_path):
    """Detect violations of TUI_PLAN_IMMUTABILITY_AFTER_COMPILE."""
    violations = []
    
    # Parse the file
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    # Define a visitor to find violations
    class PlanImmutabilityVisitor(ast.NodeVisitor):
        def __init__(self):
            self.violations = []
            self.in_compile_method = False
            self.in_reset_method = False
            self.plan_frozen = False
            
        def visit_FunctionDef(self, node):
            # Track if we're in compile or reset methods
            old_in_compile = self.in_compile_method
            old_in_reset = self.in_reset_method
            
            if node.name in ['compile', 'compile_pipeline', 'on_compile']:
                self.in_compile_method = True
                
                # Check if plan is frozen after compilation
                has_freeze = False
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name) and 'frozen' in target.id.lower():
                                has_freeze = True
                
                if not has_freeze:
                    self.violations.append({
                        'line': node.lineno,
                        'message': 'Clause 318 Violation: Plan must be frozen after compilation'
                    })
            
            if node.name in ['reset', 'reset_pipeline', 'on_reset']:
                self.in_reset_method = True
                
                # Check if plan is unfrozen during reset
                has_unfreeze = False
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name) and 'frozen' in target.id.lower():
                                has_unfreeze = True
                
                if not has_unfreeze:
                    self.violations.append({
                        'line': node.lineno,
                        'message': 'Clause 318 Violation: Plan must be explicitly unfrozen during reset'
                    })
            
            self.generic_visit(node)
            self.in_compile_method = old_in_compile
            self.in_reset_method = old_in_reset
            
        def visit_Assign(self, node):
            # Check for plan modifications outside of reset
            if not self.in_reset_method and isinstance(node.targets[0], ast.Attribute):
                attr = node.targets[0]
                if isinstance(attr.value, ast.Name) and attr.value.id in ['plan', 'pipeline', 'pipeline_plan']:
                    self.violations.append({
                        'line': node.lineno,
                        'message': 'Clause 318 Violation: Plan must not be modified after compilation without explicit reset'
                    })
            
            self.generic_visit(node)
    
    visitor = PlanImmutabilityVisitor()
    visitor.visit(tree)
    violations.extend(visitor.violations)
    
    return violations
```

### TUI_NO_VIRTUALPATH_EXPOSURE (Clause 319)

```python
def detect_virtualpath_exposure_violations(file_path):
    """Detect violations of TUI_NO_VIRTUALPATH_EXPOSURE."""
    violations = []
    
    # Parse the file
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    # Define a visitor to find violations
    class VirtualPathVisitor(ast.NodeVisitor):
        def __init__(self):
            self.violations = []
            self.imports_virtualpath = False
            
        def visit_Import(self, node):
            # Check for VirtualPath imports
            for name in node.names:
                if 'virtualpath' in name.name.lower():
                    self.imports_virtualpath = True
                    self.violations.append({
                        'line': node.lineno,
                        'message': 'Clause 319 Violation: TUI modules must not import VirtualPath'
                    })
            
            self.generic_visit(node)
            
        def visit_ImportFrom(self, node):
            # Check for VirtualPath imports
            if 'virtualpath' in node.module.lower():
                self.imports_virtualpath = True
                self.violations.append({
                    'line': node.lineno,
                    'message': 'Clause 319 Violation: TUI modules must not import VirtualPath'
                })
            
            for name in node.names:
                if 'virtualpath' in name.name.lower():
                    self.imports_virtualpath = True
                    self.violations.append({
                        'line': node.lineno,
                        'message': 'Clause 319 Violation: TUI modules must not import VirtualPath'
                    })
            
            self.generic_visit(node)
            
        def visit_Call(self, node):
            # Check for VirtualPath constructor calls
            if isinstance(node.func, ast.Name) and node.func.id == 'VirtualPath':
                self.violations.append({
                    'line': node.lineno,
                    'message': 'Clause 319 Violation: TUI code must not create VirtualPath objects'
                })
            
            # Check for missing str() conversion when getting paths from file_manager
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'file_manager':
                method_name = node.func.attr
                if method_name in ['get_path', 'list_files', 'list_directories']:
                    # Check if result is wrapped in str()
                    parent = self._find_parent(tree, node)
                    if not (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == 'str'):
                        self.violations.append({
                            'line': node.lineno,
                            'message': f'Clause 319 Violation: Result of file_manager.{method_name} must be converted to str before reaching UI'
                        })
            
            self.generic_visit(node)
            
        def _find_parent(self, tree, node):
            """Find the parent node of the given node."""
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    if child == node:
                        return parent
            return None
    
    visitor = VirtualPathVisitor()
    visitor.visit(tree)
    violations.extend(visitor.violations)
    
    return violations
```

## Integration with Existing Matrix

These TUI-specific semantic locks should be integrated with the existing semantic contract matrix in `plan_11_precise_backend_violation_matrix.md`. They provide additional constraints specific to the TUI implementation that complement the core backend-related contracts.

When implementing the TUI, all of these locks must be enforced in addition to the core contracts to ensure a robust, thread-safe, and architecturally sound user interface that adheres to OpenHCS principles.
