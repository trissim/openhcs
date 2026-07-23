Parser Metaprogramming System
=============================

Overview
--------

Traditional filename parsers hardcode component names and validation logic. The metaprogramming system eliminates these assumptions by generating parser interfaces dynamically from component enums.

.. code-block:: python

   class CustomFilenameParser(GenericFilenameParser):
       # FILENAME_COMPONENTS automatically set from AllComponents
       pass

   # Interface automatically generated with component-specific methods:
   # - get_well_keys(), get_site_keys(), get_channel_keys()
   # - validate_well(), validate_site(), validate_channel()
   # - construct_filename(well=..., site=..., channel=...)

This enables the same parser framework to work with any component structure without manual method definitions.

Dynamic Method Generation
------------------------

The system generates component-specific methods at runtime using metaclass programming. Each component in the configuration automatically gets validation and extraction methods.

.. code-block:: python

   def _generate_dynamic_methods(self):
       """Generate component-specific methods at runtime."""
       for component in self.FILENAME_COMPONENTS:
           if component != 'extension':
               # Generate get_{component}_keys method
               setattr(self, f'get_{component}_keys',
                      lambda filenames, comp=component: self._extract_component_keys(filenames, comp))

               # Generate validate_{component} method
               setattr(self, f'validate_{component}',
                      lambda value, comp=component: self._validate_component_value(value, comp))

This eliminates the need to manually define methods for each component type.

GenericFilenameParser Implementation
-----------------------------------

The GenericFilenameParser provides the base implementation that works with any component configuration.

Centralized Component Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class GenericFilenameParser(ABC):
       def __init__(self, component_enum: Type[T]):
           """Initialize with component enum - FILENAME_COMPONENTS set automatically."""
           self.component_enum = component_enum
           self.FILENAME_COMPONENTS = [c.value for c in component_enum] + ['extension']
           self._generate_dynamic_methods()

Component-specific methods are generated automatically based on the component configuration.

Concrete Parser Implementation
-----------------------------

Concrete parsers inherit from GenericFilenameParser and implement format-specific parsing.

.. code-block:: python

   class ImageXpressFilenameParser(FilenameParser):
       """Parser for ImageXpress filename format."""

       # FILENAME_COMPONENTS automatically set to AllComponents + ['extension']
       # by the FilenameParser.__init__() → GenericFilenameParser.__init__() chain

       _pattern = re.compile(r'(?:.*?_)?([A-Z]\d+)(?:_s(\d+|\{[^\}]*\}))?(?:_w(\d+|\{[^\}]*\}))?(?:_z(\d+|\{[^\}]*\}))?(\.\w+)?$')

       def parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
           """Parse ImageXpress filename, handling placeholders like {iii}."""
           basename = Path(str(filename)).name
           match = self._pattern.match(basename)

           if match:
               well, site_str, channel_str, z_str, ext = match.groups()
               parse_comp = lambda s: None if not s or '{' in s else int(s)

               return {
                   'well': well,
                   'site': parse_comp(site_str),
                   'channel': parse_comp(channel_str),
                   'z_index': parse_comp(z_str),
                   'extension': ext if ext else '.tif'
               }
           return None

The parser automatically gets component-specific methods like ``get_well_keys()``, ``validate_site()``, etc.

Multiprocessing Compatibility
-----------------------------

Dynamic methods are handled specially for multiprocessing serialization.

.. code-block:: python

   def __getstate__(self):
       """Remove dynamic methods for pickling."""
       state = self.__dict__.copy()
       # Remove unpicklable dynamic methods
       for component in self.component_enum:
           method_name = f"validate_{component.value}"
           if method_name in state:
               del state[method_name]
       return state

   def __setstate__(self, state):
       """Regenerate dynamic methods after unpickling."""
       self.__dict__.update(state)
       self._generate_dynamic_methods()

This ensures parser objects work correctly in ProcessPoolExecutor environments.

**Common Gotchas**:

- Dynamic methods are regenerated after unpickling - don't rely on method identity
- Component enum must be importable in worker processes
- Parser state should be minimal for efficient serialization
