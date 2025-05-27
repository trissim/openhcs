# Code Reference Analysis

Found 2 files with references:

## openhcs/tui/file_browser.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 51 | FileManagerBrowser.__init__ | `StorageBackendEnum` |
| 508 | main_async | `StorageBackendEnum` |

### String References
| Line | Reference |
| ---- | --------- |
| 30 | `from openhcs.io.base import StorageBackendEnum` |
| 51 | `backend: Optional[StorageBackendEnum] = None, # Backend for FileManager operatio...` |
| 484 | `from openhcs.io.base import StorageBackendEnum` |
| 508 | `backend=StorageBackendEnum.LOCAL, # Specify backend` |

## openhcs/tui/plate_manager_core.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 84 | PlateManagerPane.__init__ | `StorageBackendEnum` |

### String References
| Line | Reference |
| ---- | --------- |
| 47 | `from openhcs.io.base import storage_registry, StorageBackendEnum # Added Storage...` |
| 84 | `default_backend=StorageBackendEnum.LOCAL # Pass a default backend` |
