Component Configuration Framework
==================================

Overview
--------

Traditional microscopy processing systems hardcode assumptions about data dimensions and multiprocessing strategies. The ComponentConfiguration framework eliminates these assumptions by providing a generic configuration abstraction that can represent any enum-based component system.

.. code-block:: python

   @dataclass(frozen=True)
   class ComponentConfiguration(Generic[T]):
       all_components: Set[T]
       multiprocessing_axis: T
       default_variable: List[T]
       default_group_by: Optional[T]

This enables the same processing engine to work with different component structures - wells/sites/channels, timepoints/batches/conditions, or any other dimensional organization - without code changes.

Core Constraint
---------------

The framework enforces one fundamental processing constraint: the multiprocessing axis cannot be used as a variable component. This prevents ambiguous processing behavior where the same component would be used for both task partitioning and data grouping.

.. code-block:: python

   def validate_combination(self, variable_components: List[T], group_by: Optional[T]) -> None:
       """Validate that group_by is not in variable_components."""
       if group_by and group_by in variable_components:
           raise ValueError(f"group_by {group_by.value} cannot be in variable_components")

This constraint is enforced at configuration creation and during step validation.

ComponentConfiguration Usage
-----------------------------

Component configurations define the dimensional structure and processing behavior for a system.

Creating Configurations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Standard OpenHCS configuration
   config = ComponentConfigurationFactory.create_configuration(
       StandardComponents,
       multiprocessing_axis=StandardComponents.WELL,
       default_variable=[StandardComponents.SITE],
       default_group_by=StandardComponents.CHANNEL
   )

   # Custom temporal analysis configuration
   config = ComponentConfigurationFactory.create_configuration(
       TimeSeriesComponents,
       multiprocessing_axis=TimeSeriesComponents.TIMEPOINT,
       default_variable=[TimeSeriesComponents.WELL, TimeSeriesComponents.SITE],
       default_group_by=TimeSeriesComponents.CONDITION
   )

The factory automatically calculates remaining components available for user selection.

ComponentConfigurationFactory
-----------------------------

The factory creates configurations with automatic default resolution when defaults aren't specified.

.. code-block:: python

   # Explicit configuration
   config = ComponentConfigurationFactory.create_configuration(
       MyComponents,
       multiprocessing_axis=MyComponents.BATCH,
       default_variable=[MyComponents.SAMPLE],
       default_group_by=MyComponents.CONDITION
   )

   # Auto-resolved defaults
   config = ComponentConfigurationFactory.create_configuration(
       MyComponents,
       multiprocessing_axis=MyComponents.BATCH
       # default_variable and default_group_by auto-resolved from remaining components
   )

Auto-resolution uses the first remaining component as default_variable and the second as default_group_by.

Integration Examples
-------------------

Component configurations drive enum generation and validation across OpenHCS subsystems.

Dynamic Enum Creation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Configuration drives enum creation
   config = get_openhcs_config()
   remaining = config.get_remaining_components()

   # AllComponents: Complete dimensional space
   AllComponents = Enum('AllComponents', {c.name: c.value for c in config.all_components})

   # VariableComponents: User-selectable components
   VariableComponents = Enum('VariableComponents', {c.name: c.value for c in remaining})

Validation Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Configuration drives validation
   validator = GenericValidator(config)
   result = validator.validate_step(
       variable_components=[VariableComponents.SITE],
       group_by=GroupBy.CHANNEL
   )

Multiprocessing Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Configuration drives task coordination
   coordinator = MultiprocessingCoordinator(config)
   tasks = coordinator.create_tasks(orchestrator, pipeline_definition)

Extension Examples
------------------

Custom Component Systems
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class TimeSeriesComponents(Enum):
       WELL = "well"
       TIMEPOINT = "timepoint"
       CHANNEL = "channel"
       FIELD = "field"

   # Temporal parallelization strategy
   timeseries_config = ComponentConfigurationFactory.create_configuration(
       TimeSeriesComponents,
       multiprocessing_axis=TimeSeriesComponents.TIMEPOINT,
       default_variable=[TimeSeriesComponents.WELL, TimeSeriesComponents.FIELD],
       default_group_by=TimeSeriesComponents.CHANNEL
   )

**Common Gotchas**:

- Don't use the multiprocessing axis as a variable component - validation will fail
- Component keys are cached on initialization - call ``clear_component_cache()`` if input directory changes
- Dict pattern keys must match actual component values, not enum names
