# Detailed Code Definition Matrix
Generated on: 2025-05-22 18:08:36

### Detailed Matrix for `openhcs/tui/services/plate_validation.py`

| Definition Type | Name | Parent Class | Parameters | Return Type | Lines |
| --- | --- | --- | --- | --- | --- |
| class | ValidationResultCallback |  |  |  | 13-15 |
| method | __call__ | ValidationResultCallback | self: Any, result: Dict[str, Any] | <complex_annotation> | 15-15 |
| class | ErrorCallback |  |  |  | 17-19 |
| method | __call__ | ErrorCallback | self: Any, message: str, details: Optional[str] | <complex_annotation> | 19-19 |
| class | PlateValidationService |  |  |  | 22-258 |
| method | __init__ | PlateValidationService | self: Any, context: ProcessingContext, on_validation_result: ValidationResultCallback, on_error: ErrorCallback, storage_registry: Any, io_executor: Optional[ThreadPoolExecutor] |  | 30-63 |
| method | close | PlateValidationService | self: Any |  | 65-72 |
| method | validate_plate | PlateValidationService |  | Dict[str, Any] | 74-145 |
| method | _validate_plate_directory | PlateValidationService |  | bool | 147-189 |
| method | generate_plate_id | PlateValidationService |  | str | 191-230 |
| method | close | PlateValidationService | self: Any |  | 232-246 |
| method | __del__ | PlateValidationService | self: Any |  | 248-258 |

