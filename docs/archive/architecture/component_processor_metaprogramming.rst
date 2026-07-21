Component Processor Metaprogramming
===================================

Overview
--------

Traditional component processing systems hardcode assumptions about component names and processing patterns. The DynamicInterfaceMeta system eliminates these assumptions by generating abstract method interfaces dynamically from component enums.

.. code-block:: python

   # Generate processing interface for any component enum
   ProcessorInterface = DynamicInterfaceMeta(
       "CustomProcessor",
       (ComponentProcessorInterface,),
       {},
       component_enum=MyComponents,
       method_patterns=['process', 'validate', 'summarize']
   )

   # Interface automatically gets abstract methods:
   # - process_well(), process_site(), process_channel()
   # - validate_well(), validate_site(), validate_channel()
   # - summarize_well(), summarize_site(), summarize_channel()

This enables the same processing framework to work with any component structure without manual interface definitions.

Dynamic Interface Generation
----------------------------

The metaclass generates abstract methods for each component × pattern combination.

.. code-block:: python

   class DynamicInterfaceMeta(type):
       """Metaclass that generates component processing interfaces."""

       def __new__(mcs, name, bases, namespace, component_enum=None, method_patterns=None, **kwargs):
           """Create interface class with generated abstract methods."""
           if component_enum and method_patterns:
               # Generate abstract methods for each component × pattern combination
               for component in component_enum:
                   for pattern in method_patterns:
                       method_name = f"{pattern}_{component.value}"
                       # Create abstract method dynamically
                       def create_abstract_method(method_name=method_name):
                           @abstractmethod
                           def abstract_method(self, context, **kwargs):
                               raise NotImplementedError(f"Method {method_name} must be implemented")
                           abstract_method.__name__ = method_name
                           return abstract_method
                       namespace[method_name] = create_abstract_method()

           return super().__new__(mcs, name, bases, namespace)

This creates abstract methods that must be implemented by concrete processor classes.

Concrete Implementation
-----------------------

Concrete processors inherit from the generated interface and implement the abstract methods.

.. code-block:: python

   # Generate interface for specific component enum
   MyProcessorInterface = DynamicInterfaceMeta(
       "MyProcessorInterface",
       (ComponentProcessorInterface,),
       {},
       component_enum=MyComponents,
       method_patterns=['process', 'validate']
   )

   # Implement concrete processor
   class MyProcessor(MyProcessorInterface):
       def process_well(self, context, **kwargs):
           # Process well data
           pass

       def process_site(self, context, **kwargs):
           # Process site data
           pass

       def validate_well(self, context, **kwargs):
           # Validate well configuration
           pass

The metaclass ensures all required methods are implemented at class creation time.

Factory Pattern
---------------

The InterfaceGenerator provides cached interface creation for performance.

.. code-block:: python

   class InterfaceGenerator:
       def __init__(self):
           self._interface_cache: Dict[str, Type] = {}

       def create_interface(self, component_enum: Type[T],
                           method_patterns: Optional[list] = None) -> Type:
           """Create component-specific interface class."""
           cache_key = f"{component_enum.__name__}_{id(component_enum)}"
           if cache_key in self._interface_cache:
               return self._interface_cache[cache_key]

           interface_class = DynamicInterfaceMeta(
               f"{component_enum.__name__}ProcessorInterface",
               (ComponentProcessorInterface,),
               {},
               component_enum=component_enum,
               method_patterns=method_patterns or ['process', 'validate']
           )

           self._interface_cache[cache_key] = interface_class
           return interface_class

This enables efficient creation of component-specific processing interfaces.

Usage Example
-------------

.. code-block:: python

   # Create interface for custom components
   class AnalysisComponents(Enum):
       SAMPLE = "sample"
       CONDITION = "condition"
       REPLICATE = "replicate"

   generator = InterfaceGenerator()
   AnalysisInterface = generator.create_interface(
       AnalysisComponents,
       method_patterns=['analyze', 'validate', 'export']
   )

   # Implement concrete processor
   class SampleAnalyzer(AnalysisInterface):
       def analyze_sample(self, context, **kwargs):
           # Process sample data
           pass

       def validate_condition(self, context, **kwargs):
           # Validate condition setup
           pass

       # Must implement all generated abstract methods

**Common Gotchas**:

- All generated abstract methods must be implemented - missing methods cause initialization errors
- Method patterns are fixed at interface creation time - can't be changed later
- Interface classes are cached by enum object ID - enum changes require new interfaces
