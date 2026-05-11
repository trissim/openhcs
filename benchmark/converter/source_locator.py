"""
SourceLocator - Locate CellProfiler source code for modules.

Maps module names from .cppipe files to their source implementations in
benchmark/cellprofiler_source/. Provides source code strings for LLM conversion.

Source layout:
    benchmark/cellprofiler_source/
    ├── modules/                    # Module classes (UI + settings)
    │   └── identifyprimaryobjects.py
    ├── library/
    │   ├── modules/               # Pure algorithm implementations
    │   │   ├── _threshold.py
    │   │   └── _gaussianfilter.py
    │   ├── functions/             # Core library functions
    │   │   ├── image_processing.py
    │   │   └── segmentation.py
    │   └── opts/                  # Enums and options
    │       └── threshold.py
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from openhcs.interop.cellprofiler.parser import ModuleBlock

logger = logging.getLogger(__name__)


@dataclass
class SourceLocation:
    """Located source code for a CellProfiler module."""
    
    module_name: str  # Original module name (e.g., "IdentifyPrimaryObjects")
    library_module_path: Optional[Path] = None  # library/modules/_*.py
    module_class_path: Optional[Path] = None  # modules/*.py
    source_code: str = ""  # Actual source code content
    dependencies: List[str] = None  # Required imports/dependencies
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
    
    @property
    def has_library_implementation(self) -> bool:
        """Check if pure algorithm implementation exists."""
        return self.library_module_path is not None and self.library_module_path.exists()


class SourceLocator:
    """
    Locate CellProfiler source code for conversion to OpenHCS.
    
    Searches benchmark/cellprofiler_source/ for:
    1. library/modules/_<name>.py - Pure algorithm implementations (preferred)
    2. modules/<name>.py - Module class implementations
    3. library/functions/*.py - Shared utility functions
    """
    
    def __init__(self, source_root: Optional[Path] = None):
        """
        Initialize source locator.
        
        Args:
            source_root: Root of CellProfiler source (default: benchmark/cellprofiler_source/)
        """
        if source_root is None:
            # Default to benchmark/cellprofiler_source relative to this file
            source_root = Path(__file__).parent.parent / "cellprofiler_source"
        
        self.source_root = Path(source_root)
        self.library_modules_dir = self.source_root / "library" / "modules"
        self.modules_dir = self.source_root / "modules"
        self.library_functions_dir = self.source_root / "library" / "functions"
        self.library_opts_dir = self.source_root / "library" / "opts"
        
        # Cache of located sources
        self._cache: Dict[str, SourceLocation] = {}
    
    def locate(self, module: ModuleBlock) -> SourceLocation:
        """
        Locate source code for a module.
        
        Args:
            module: ModuleBlock from parser
            
        Returns:
            SourceLocation with paths and source code
        """
        if module.name in self._cache:
            return self._cache[module.name]
        
        location = SourceLocation(module_name=module.name)
        
        # Try library/modules/_<name>.py first (pure algorithm)
        lib_module_name = f"_{module.name.lower()}.py"
        lib_module_path = self.library_modules_dir / lib_module_name
        
        if lib_module_path.exists():
            location.library_module_path = lib_module_path
            location.source_code = lib_module_path.read_text()
            logger.info(f"Found library module: {lib_module_path}")
        else:
            # Try modules/<name>.py (class implementation)
            module_path = self.modules_dir / f"{module.name.lower()}.py"
            if module_path.exists():
                location.module_class_path = module_path
                location.source_code = module_path.read_text()
                logger.info(f"Found module class: {module_path}")
            else:
                logger.warning(f"No source found for module: {module.name}")
        
        self._cache[module.name] = location
        return location
    
    def locate_all(self, modules: List[ModuleBlock]) -> Dict[str, SourceLocation]:
        """
        Locate source code for multiple modules.
        
        Args:
            modules: List of ModuleBlock from parser
            
        Returns:
            Dict mapping module name to SourceLocation
        """
        return {m.name: self.locate(m) for m in modules}
    
    def get_library_function(self, function_name: str) -> Optional[str]:
        """
        Get source code for a library function.
        
        Searches library/functions/*.py for the function.
        
        Args:
            function_name: Name of function to find
            
        Returns:
            Source code string if found, None otherwise
        """
        for py_file in self.library_functions_dir.glob("*.py"):
            content = py_file.read_text()
            if f"def {function_name}" in content:
                logger.info(f"Found function {function_name} in {py_file}")
                return content
        return None
    
    def get_opts_enum(self, enum_name: str) -> Optional[str]:
        """
        Get source code for an enum from library/opts/.
        
        Args:
            enum_name: Name of enum (e.g., "Scope", "Method")
            
        Returns:
            Source code string if found, None otherwise
        """
        for py_file in self.library_opts_dir.glob("*.py"):
            content = py_file.read_text()
            if f"class {enum_name}" in content:
                logger.info(f"Found enum {enum_name} in {py_file}")
                return content
        return None
    
    def list_available_modules(self) -> List[str]:
        """List all available library module implementations."""
        modules = []
        for py_file in self.library_modules_dir.glob("_*.py"):
            if py_file.name != "__init__.py":
                # _threshold.py -> Threshold
                name = py_file.stem[1:].title()
                modules.append(name)
        return sorted(modules)
