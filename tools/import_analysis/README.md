# Import Analysis Tools

This directory contains tools for analyzing and fixing import issues in Python code.

## Available Tools

### 1. Import Validator (`import_validator.py`)

Analyzes imports in Python code to identify issues.

**Features:**
- Detects missing imports (symbols used but not imported)
- Detects unused imports (symbols imported but not used)
- Identifies module structure issues (duplicate imports, etc.)

**Usage:**
```bash
./import_validator.py <path_to_analyze> -o <output_file.md>
```

### 2. Import Fixer (`fix_imports.py`)

Automatically fixes common import issues in Python code.

**Features:**
- Removes unused imports
- Consolidates duplicate imports
- Fixes import order

**Usage:**
```bash
./fix_imports.py <path_to_analyze>
```

## Integration with Code Analysis Tools

These tools are integrated with the code analysis tools in the `../code_analysis/` directory. You can run the import analysis as part of a comprehensive code analysis using the meta-analyzer:

```bash
../code_analysis/meta_analyzer.py imports <path_to_analyze> --fix
```

## Output

The import validator generates a Markdown report with details about import issues found in the codebase. The report includes:

- Missing imports (symbols used but not imported)
- Unused imports (symbols imported but not used)
- Module structure issues (duplicate imports, etc.)

The import fixer modifies Python files directly to fix import issues. It prints a summary of changes made to each file.

## Requirements

- Python 3.6+
- Standard library only (ast, os, sys, re)

## Examples

### Validate Imports

```bash
./import_validator.py openhcs/tui -o reports/import_analysis.md
```

This will analyze imports in the `openhcs/tui` directory and generate a report in the `reports/import_analysis.md` file.

### Fix Imports

```bash
./fix_imports.py openhcs/tui
```

This will fix import issues in the `openhcs/tui` directory.
