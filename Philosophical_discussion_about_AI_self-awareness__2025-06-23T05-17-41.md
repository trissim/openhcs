[ ] NAME:Current Task List DESCRIPTION:Root task for conversation __NEW_AGENT__
-[x] NAME:Fix circular import in OpenHCS microscope handler DESCRIPTION:Resolve the circular import between openhcs.py and microscope_base.py using TYPE_CHECKING pattern and runtime imports to prevent module loading failures
-[/] NAME:Implement fail-loud parser instantiation DESCRIPTION:Replace the problematic triple-fallback constructor pattern with explicit single constructor signature (filemanager=None, pattern_format=None) that fails immediately on errors instead of silently falling back
-[ ] NAME:Clean up semantic ambiguity in parser constructor kwargs DESCRIPTION:Address the interface compatibility issue where parsers accept filemanager and pattern_format parameters but don't actually use them - either make them functional or document the interface contract clearly
-[ ] NAME:Inline Interperter Test and validate OpenHCS handler fixes DESCRIPTION:Run Inline Interpreter based tests to ensure the circular import fix and fail-loud parser instantiation work correctly, and that auto-detection properly identifies OpenHCS format plates
-[ ] NAME:Create PR for OpenHCS microscope handler DESCRIPTION:Once all fixes are complete and tested, create a pull request to merge the improved OpenHCS microscope handler into the main branch
-[ ] NAME:Phase 1: Investigation and Analysis DESCRIPTION:Deep dive into current codebase to understand existing patterns, dependencies, and constraints before implementing changes
--[x] NAME:Phase 1 Complete: Investigation Summary DESCRIPTION:Comprehensive summary of all Phase 1 investigation findings for handler-based auto-detection pattern implementation

## 🎯 PHASE 1 INVESTIGATION COMPLETE - COMPREHENSIVE FINDINGS

### **1. Current Auto-Detection Logic Analysis**
**Location**: `openhcs/microscopes/microscope_base.py` lines 442-497

**Sequential Detection Order:**
1. **OpenHCS**: `openhcs_metadata.json` (✅ uses constant)
2. **Opera Phenix**: `Index.xml` (❌ hardcoded)  
3. **ImageXpress**: `{'.htd', '.HTD'}` extensions (❌ hardcoded)

**Key Issues:**
- ❌ Hardcoded filenames for Opera Phenix and ImageXpress
- ❌ Duplicated logic between auto-detection and handler `find_metadata_file()` methods
- ❌ All backend operations hardcoded to `Backend.DISK.value`
- ❌ Maintenance burden: changes require updating multiple locations

### **2. Registry Architecture Analysis**
**Current Pattern**: `MICROSCOPE_HANDLERS` registry in `microscope_base.py`
- ✅ Global registry populated by imports
- ✅ String keys match auto-detection returns
- ❌ Duplicate registration in two files (`__init__.py` and `microscope_interfaces.py`)
- ❌ No equivalent registry for MetadataHandler classes

**Recommended Solution**: Create `METADATA_HANDLERS` registry in `__init__.py` following same pattern

### **3. MetadataHandler Implementation Compatibility**
**Critical Inconsistencies Found:**
- **Return Types**: OpenHCS returns `Optional[Path]`, others return `Path`
- **Error Handling**: OpenHCS graceful (None), others fail-loud (exceptions)
- **Backend Usage**: OpenHCS uses default, others hardcode `Backend.DISK.value`
- **Method Signatures**: OpenHCS has optional `context` parameter

**Registry Requirements:**
- Need wrapper for exception handling (OperaPhenix/ImageXpress throw exceptions)
- Need standardized return types for auto-detection iteration
- All handlers have compatible constructors: `__init__(filemanager: FileManager)`

### **4. Circular Import Risk Analysis**
**🚨 CRITICAL FINDING**: Direct circular import risk if `microscope_base.py` imports handlers:
```
microscope_base.py → imagexpress.py → microscope_base.py (CIRCULAR!)
microscope_base.py → opera_phenix.py → microscope_base.py (CIRCULAR!)
```

**✅ SAFE SOLUTION**: Registry in `__init__.py`
- Already imports all handlers safely
- No new circular import risks
- Follows existing `MICROSCOPE_HANDLERS` pattern
- Clean separation of concerns

### **5. Test Coverage Analysis**
**Existing Tests:**
- ✅ OpenHCS auto-detection unit test with comprehensive mocking
- ✅ Integration tests for ImageXpress and OperaPhenix with real data
- ❌ Missing unit tests for Opera Phenix and ImageXpress auto-detection
- ❌ No error case testing for auto-detection failures

**Inline Python Interpreter Test Strategy:**
```python
# Registry verification
from openhcs.microscopes import METADATA_HANDLERS
assert all(key in METADATA_HANDLERS for key in ['openhcs', 'opera_phenix', 'imagexpress'])

# Handler instantiation
from openhcs.io.filemanager import FileManager
fm = FileManager()
for name, handler_class in METADATA_HANDLERS.items():
    handler = handler_class(fm)
    assert hasattr(handler, 'find_metadata_file')
```

### **6. Implementation Strategy**
**Phase 2 Design Requirements:**
1. Create `METADATA_HANDLERS` registry in `__init__.py`
2. Design exception wrapper for graceful auto-detection
3. Standardize return types and error handling
4. Maintain existing test interfaces
5. Plan iteration order and fallback logic

**Key Constraints:**
- ✅ Must maintain `create_microscope_handler(microscope_type='auto')` interface
- ✅ Must return same handler instances as current implementation
- ✅ Must provide equivalent error messages for unknown types
- ✅ Must work with existing FileManager and Backend patterns

### **7. Architecture Benefits**
**New Pattern Advantages:**
- 🔄 **Extensible**: Add handlers by registry registration only
- 🎯 **No Hardcoding**: Each handler owns its detection logic
- 🔒 **Consistent**: Auto-detection uses exact same logic as handlers
- 🧹 **Maintainable**: Single source of truth for metadata discovery
- 🛡️ **Safe**: No circular import risks

**Ready for Phase 2: Design Registry Architecture**
-[ ] NAME:Phase 2: Design Registry Architecture DESCRIPTION:Design the metadata handler registry system and integration points with existing MICROSCOPE_HANDLERS registry
-[ ] NAME:Phase 3: Implement Registry System DESCRIPTION:Create the metadata handler registry and registration mechanism
-[ ] NAME:Phase 4: Refactor Auto-Detection Logic DESCRIPTION:Replace hardcoded auto-detection with registry-based iteration pattern
-[ ] NAME:Phase 5: Testing and Validation DESCRIPTION:Comprehensive testing to ensure the new pattern works correctly and doesn't break existing functionality
-[ ] NAME:Phase 6: Cleanup and Documentation DESCRIPTION:Remove obsolete code, update documentation, and ensure code quality
-[x] NAME:1.1: Map Current Auto-Detection Logic DESCRIPTION:Document the exact current implementation of _auto_detect_microscope_type() including all hardcoded patterns, file paths, and detection logic
-[x] NAME:1.2: Analyze Existing Registry Patterns DESCRIPTION:Study how MICROSCOPE_HANDLERS registry works, where it's defined, how handlers are registered, and how it's used throughout the codebase
-[x] NAME:1.3: Inventory MetadataHandler Implementations DESCRIPTION:Catalog all existing MetadataHandler classes, their find_metadata_file() implementations, and any dependencies or special requirements
-[x] NAME:1.4: Identify Circular Import Risks DESCRIPTION:Map import dependencies to identify potential circular import issues when importing metadata handlers in microscope_base.py
-[x] NAME:1.5: Document Current Test Coverage DESCRIPTION:Find and analyze existing tests for auto-detection logic to understand what needs to be maintained/updated