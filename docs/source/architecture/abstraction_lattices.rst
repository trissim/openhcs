Abstraction lattices: wrong vs right
====================================

OpenHCS attaches semantics to a single owning declaration. Agents often recreate
poly-authority by inventing a second catalog, a compatibility bridge, or a new
``AutoRegisterMeta`` family that only copies facts from an existing root.

This page is a worked catalog from the ``benchmark-platform`` branch: incorrect
lattices that appeared in history or in the unification cutover, and the correct
owners that replaced them. Use it before adding types, registries, or packages.

Evidence labels
---------------

``history:``
  A landed commit on ``benchmark-platform`` (short hash).

``wip:``
  Present in the unification working tree and listed for permanent deletion by
  ``tests/unit/test_cellprofiler_static_deletion_gates.py`` (and related plan
  inventories). Treat resurrection of these paths as an architecture failure.

Classify before you code
------------------------

Ask:

1. **Am I adding a root, or a mirror of a root?**
2. **Am I specializing via a leaf hook / mixin on that root, or via
   ``isinstance`` / name / ``elif`` in the caller?**

.. list-table:: Failure classes
   :header-rows: 1
   :widths: 22 38 40

   * - Class
     - Smell
     - Test
   * - Parallel lattice
     - Second package, catalog, or sidecar for the same facts
     - Two places answer the same question
   * - Unnecessary nominal
     - ``AutoRegisterMeta``, strategy family, or declaration with no unique semantics
     - Collapses to a map, dataclass, or direct call with no loss
   * - Nominal mirror
     - Registry or declaration that reads or copies another owner
     - Deleting it and querying the real root still works; keeping both requires sync
   * - Caller-side dispatch
     - ``isinstance`` on concrete subclasses, class-name strings, long ``elif``
       on enums, priority ints, or free helpers next to consumers
     - Moving the branch into a root method / leaf hook / shared strategy mixin
       deletes the consumer ladder with no second catalog

``AutoRegisterMeta`` plus ``__registry__`` is not proof of ownership. Ownership
means unique facts that do not exist elsewhere. If facts are copied from
declarations, you built a mirror.

Having the correct root is also not enough. Generic code must call methods on
the abstract owner (or a shared selection mixin). Leaves and mixins supply
subclass-specific hooks. Do not teach the caller the concrete subclass lattice.

Related pages: :doc:`nominal_ownership`, :doc:`../development/respecting_codebase_architecture`,
:doc:`cellprofiler_interop`, :doc:`source_model`,
:doc:`microscope_handler_integration`.

Per-example shape
-----------------

Ownership examples (parallel / unnecessary nominal / mirror) use: problem,
incorrect lattice, failure class, correct lattice, evidence, stop-rule.

The **caller-side dispatch** section is code-first: incorrect and correct
snippets simplified from real commits. Prefer that section when deciding how
to specialize an existing root.

Parallel lattices
-----------------

Absorbed CellProfiler function catalog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Make CellProfiler algorithms available inside OpenHCS quickly.

**Incorrect lattice.** The historical ``benchmark/cellprofiler_library/`` LLM
absorption tree, its later backend ``library.py`` catalog, and
``contracts.json`` as a contract catalog.

**Failure class.** Parallel lattice (and orphan sidecar when the library is
gone but JSON remains).

**Correct lattice.** ``CellProfilerModule`` subclasses in
``openhcs/interop/cellprofiler/module_declarations.py`` own module semantics.
Backend callables live under ``openhcs/processing/backends/cellprofiler/``.
Contracts come from declarations and ``CallableContract``, not a JSON catalog.

**Evidence.** ``history:`` ``ece9962b``, ``5a3c3e70`` introduced absorption;
``history:`` ``38ba601b`` moved absorbed registry into the backend;
``wip:`` deletes ``benchmark/cellprofiler_library`` and ``library.py``
(deletion gates). Residual orphan: tracked
``openhcs/processing/backends/cellprofiler/contracts.json`` with no Python
readers — do not revive consumers.

**Stop-rule.** If you need a function or contract catalog for CellProfiler
modules, extend ``CellProfilerModule`` / backend callables. Do not recreate
``cellprofiler_library``, ``library.py``, or ``contracts.json``.

Benchmark CellProfiler compat bridge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Keep benchmarks green while dual runtimes exist.

**Incorrect lattice.** ``benchmark/cellprofiler_compat/``
(``runtime_adapter.py``, ``module_execution.py``, measurement helpers).

**Failure class.** Parallel lattice (compat bridge).

**Correct lattice.** ``openhcs/interop/cellprofiler/runtime/`` plus the generic
``FunctionStep`` execution path. Benchmarks consume the public pipeline
boundary; they do not own a second runtime.

**Evidence.** ``history:`` bridge grew through commits such as ``380a50f9`` and
``aece40f1``; ``wip:`` deletes ``benchmark/cellprofiler_compat`` (deletion
gates).

**Stop-rule.** If a benchmark needs adapter logic, put it in interop or make
the public runtime correct. Do not recreate ``cellprofiler_compat``.

Per-module settings file trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Bind ``.cppipe`` settings to callables module by module.

**Incorrect lattice.** Dozens of ``*_settings.py`` files under converter then
interop (for example ``align_settings.py``, ``filter_objects_settings.py``).

**Failure class.** Parallel lattice.

**Correct lattice.** Settings and binding facts on ``CellProfilerModule``
declarations, executed through ``settings_binder`` / setting-to-keyword
bindings — not a parallel per-module settings package tree.

**Evidence.** ``history:`` ``63f0ede1`` deleted obsolete per-module settings
files and moved semantics onto declarations.

**Stop-rule.** If a module needs settings semantics, declare them on the
module class. Do not add a new ``*_settings.py`` catalog file.

LLM absorber / form-registry toolchain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Automate absorption and parameter mapping.

**Incorrect lattice.** Historical converter commands named ``absorb.py``,
``library_absorber.py``, ``llm_converter.py``, ``form_registry.py``,
``add_parameter_mappings.py``, ``backfill_parameter_mappings.py``,
``system_prompt.py``, ``source_locator.py``.

**Failure class.** Parallel lattice.

**Correct lattice.** Declaration-owned modules in
``openhcs/interop/cellprofiler/module_declarations.py``. The live
``benchmark/converter/compatibility_matrix.py`` may *report* over that registry;
it must not become a second writable catalog.

**Evidence.** ``history:`` introduced with the converter stack (``ece9962b``
era); ``wip:`` paths listed in ``REQUIRED_DELETED_PATHS``.

**Stop-rule.** Do not resurrect absorber CLIs or form registries as semantic
owners.

Duplicate Napari stream server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Stream plates into Napari from more than one call site.

**Incorrect lattice.** A second stream-server implementation inside
``napari_stream_visualizer.py``.

**Failure class.** Parallel lattice.

**Correct lattice.** One shared viewer streaming / server axis.

**Evidence.** ``history:`` ``a1cdbe0d`` collapsed the duplicate Napari stream
server.

**Stop-rule.** If streaming needs a new behavior, extend the shared viewer
streaming owner. Do not fork a second server loop.

Unnecessary nominal abstractions
--------------------------------

GrayToColor AutoRegisterMeta resolvers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Map GrayToColor input schemes to resolvers.

**Incorrect lattice.** ``_GrayToColorInputNameResolver`` with
``AutoRegisterMeta`` plus ``GrayToColorInputNameResolverDeclaration.materialize()``
dynamic subclasses.

**Failure class.** Unnecessary nominal.

**Correct lattice.** Plain dataclass plus a scheme→resolver map (no registry
family for a static table).

**Evidence.** ``history:`` ``5540aaa9`` declared resolver families;
``history:`` ``96ffa98b`` collapsed them the next day.

**Stop-rule.** If the registry has one meaningful leaf pattern or only encodes
a fixed scheme table, use a map or dataclass. Do not AutoRegister it.

CellProfilerFunctionContractMetadata wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Expose processing-contract metadata for absorbed functions.

**Incorrect lattice.** ``CellProfilerFunctionContractMetadata`` wrapping an
existing contract authority.

**Failure class.** Unnecessary nominal.

**Correct lattice.** ``CallableContract.from_callable`` /
``ProcessingContract`` directly.

**Evidence.** ``history:`` ``f39e132b`` collapsed the wrapper.

**Stop-rule.** Do not add a CellProfiler-named wrap around a generic contract
type that already owns the fact.

module_semantics / semantic-default AutoRegisterMeta
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Share semantic defaults across modules.

**Incorrect lattice.** The deleted ``module_semantics.py`` table and unused
semantic-default ``AutoRegisterMeta`` facades.

**Failure class.** Unnecessary nominal (facade without unique ownership).

**Correct lattice.** Fields and methods on ``CellProfilerModule`` subclasses in
``openhcs/interop/cellprofiler/module_declarations.py``.

**Evidence.** ``wip:`` deletes ``module_semantics.py`` (deletion gates).

**Stop-rule.** Defaults that belong to a module are declared on that module
class, not on a second AutoRegister root.

Thin image-execution / plane strategy shells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Dispatch CellProfiler image execution modes.

**Incorrect lattice.** One-leaf or thin strategy families such as
``runtime/image_execution_strategies.py`` and plane-projection requirement
registries that rename generic runtime behavior.

**Failure class.** Unnecessary nominal.

**Correct lattice.** Generic runtime plane projection and module execution
without a CellProfiler-only strategy lattice for the same dispatch.

**Evidence.** ``history:`` ``2de80350``, ``85e36d1f`` collapse thin subclasses;
``wip:`` deletes ``image_execution_strategies.py`` and related shells.

**Stop-rule.** If the strategy family only forwards to generic runtime
projection, delete the family and call the generic owner.

Backend authority re-export wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Give backends a local “authority” import surface.

**Incorrect lattice.** Wrapper modules that only re-export another owner.

**Failure class.** Unnecessary nominal.

**Correct lattice.** Call the real owner directly.

**Evidence.** ``history:`` ``ddbe41e4`` inlined CP backend authority wrappers.

**Stop-rule.** Do not add an authority module whose body is only imports and
aliases.

Nominal mirrors (fake SSOT)
---------------------------

Highest agent risk: these look like correct OpenHCS style and still violate
ownership.

Symbol table as compile / artifact authority
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Track CellProfiler workspace names and artifact bindings at
compile time.

**Incorrect lattice.** The deleted ``CellProfilerSymbolKey`` / workspace-binding
symbol table plus its dedicated symbol-table test suite.

**Failure class.** Nominal mirror.

**Correct lattice.** ``CellProfilerModuleArtifactContracts`` in
``openhcs/interop/cellprofiler/module_artifact_contracts.py`` and nominal module
leaves in ``openhcs/interop/cellprofiler/module_artifact_declarations.py`` own
artifact facts. ``openhcs/interop/cellprofiler/pipeline_import.py`` projects
those declarations to public ``list[FunctionStep]`` + ``PipelineConfig``.

**Evidence.** Long-lived at HEAD; ``wip:`` deletes ``symbol_table.py`` and its
dedicated tests (deletion gates).

**Stop-rule.** Do not recreate a symbol table that mirrors declaration artifact
facts. Extend module artifact contracts.

Compiler registry / dialect compiler singleton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Select a CellProfiler dialect compiler.

**Incorrect lattice.** ``compiler_registry.py`` plus ``pipeline_compiler.py``
(``CellProfilerDialectCompiler`` singleton).

**Failure class.** Nominal mirror (fake registry).

**Correct lattice.** Pure import (``pipeline_import``) to public steps. There
is no replacement one-leaf compiler registry.

**Evidence.** ``wip:`` deletes both paths (deletion gates).

**Stop-rule.** Do not add a process-global compiler registry for a single
import function.

module_roles role table
~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Decide which modules emit function steps vs setup-only.

**Incorrect lattice.** The deleted interop and benchmark ``module_roles.py``
tables.

**Failure class.** Nominal mirror.

**Correct lattice.** ``CellProfilerModule.emits_function_step`` in
``openhcs/interop/cellprofiler/module_declarations.py`` and
``SourceSetupCellProfilerModule`` in
``openhcs/interop/cellprofiler/module_artifact_declarations.py``.

**Evidence.** ``wip:`` deletes both role tables (deletion gates).

**Stop-rule.** Role is declaration inheritance, not a parallel role registry.

Pipeline generator and generated-pipeline sidecars
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Lower ``.cppipe`` into executable OpenHCS pipelines.

**Incorrect lattice.** ``pipeline_generator.py``,
``runtime/generated_pipeline.py``, ``runtime_pipeline.py``, and generated
artifact-contract sidecars transported as hidden authority.

**Failure class.** Nominal mirror.

**Correct lattice.** Public ``PipelineConfig`` + ``FunctionStep`` declarations.
The compiler derives invocation contracts from declarations; transport carries
public source, not hidden sidecars.

**Evidence.** ``history:`` ``95e6f245``, ``a5f9d99d``, ``f9af176b`` move to the
public compile path; ``wip:`` deletes generator / generated-pipeline shells
(deletion gates).

**Stop-rule.** If generated code needs a fact at runtime, put it on the public
step or derive it at compile time. Do not ship a second pipeline type or
sidecar catalog.

Runtime policy_registry copying declaration attrs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Expose module policies to the runtime adapter.

**Incorrect lattice.** The deleted runtime ``policy_registry.py`` building
policy-view classes from declaration attributes.

**Failure class.** Nominal mirror.

**Correct lattice.** Query ``CellProfilerModule`` through
``openhcs/interop/cellprofiler/module_declarations.py`` and compiled
``CallableContract`` through ``openhcs/core/callable_contract.py`` directly.

**Evidence.** ``wip:`` deletes ``policy_registry.py`` (deletion gates).

**Stop-rule.** Do not materialize a second registry whose values are copies of
declaration fields.

Invocation-contract shells as owners
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Carry CellProfiler callable contracts across UI / ZMQ.

**Incorrect lattice.** ``CellProfilerRuntimeCallable`` metadata fields,
``function_step_invocation_contracts.py``, and related invocation shells as
contract authorities.

**Failure class.** Nominal mirror.

**Correct lattice.** Raw public callable kwargs on ``FunctionStep``; compile
produces ``InvocationContractPlan`` / ``CallableContract``. Runtime executes
the compiled plan.

**Evidence.** ``history:`` ``cae5aa60`` … ``a5f9d99d`` derive contracts from
declarations; ``wip:`` deletes ``function_step_invocation_contracts.py`` and
related shells (deletion gates).

**Stop-rule.** Do not store a second contract object on the step as authority
when the compiler can derive it from declarations.

Source-schema workspace parallel language
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Describe plate sources for CellProfiler-shaped inputs.

**Incorrect lattice.** ``pipeline_image_schema.py`` and
``source_schema_workspace.py`` (and one-leaf source policy parsers) as a second
source language.

**Failure class.** Nominal mirror / parallel source lattice.

**Correct lattice.** ``SourceBindingsConfig`` plus virtual-workspace projection
(``source_binding_workspace``, ``VirtualWorkspaceSourceProjection``).
Microscope handlers project physical layout; bindings name sources.

**Evidence.** ``history:`` ``de884b8b`` centralizes candidate authority;
``wip:`` deletes ``pipeline_image_schema.py`` and
``source_schema_workspace.py`` (deletion gates).

**Stop-rule.** Do not invent a source-schema layer beside source bindings.

ZMQ pipeline transport shim
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Normalize pipelines for ZMQ submission.

**Incorrect lattice.** The deleted ZMQ-only pipeline transport shim.

**Failure class.** Nominal mirror.

**Correct lattice.** ``FunctionStepTransportAuthority``
(``openhcs/core/function_step_transport.py``) on the public FunctionStep
boundary.

**Evidence.** ``wip:`` deletes ``zmq_pipeline_transport.py`` (deletion gates).

**Stop-rule.** Transport normalization belongs with FunctionStep transport, not
a ZMQ-only parallel owner.

Module function-resolution / processing-component catalogs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Resolve callables and processing components by module name.

**Incorrect lattice.** ``module_function_resolution.py`` and
``module_processing_components.py`` as module-name catalogs.

**Failure class.** Nominal mirror.

**Correct lattice.** Declaration hooks on ``CellProfilerModule`` (resolve
function, processing facts) queried generically.

**Evidence.** Direction set in ``history:`` ``63f0ede1``; ``wip:`` deletes both
catalog modules (deletion gates).

**Stop-rule.** Module-name dispatch tables that restate declaration facts are
mirrors. Extend the module class.

Special-output declaration parallel vocabulary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem.** Name non-image outputs.

**Incorrect lattice.** ``special_outputs.py`` /
``special_output_declarations.py`` as a parallel I/O vocabulary.

**Failure class.** Nominal mirror.

**Correct lattice.** ``ArtifactType`` and callable artifact contracts.

**Evidence.** ``wip:`` deletes these core modules in the cutover tree; prefer
artifact contracts going forward.

**Stop-rule.** Non-image I/O is an artifact type, not a second special-output
registry.

Caller-side dispatch vs owner-side hooks
----------------------------------------

This is the compositional half of ownership. Even with one registry root,
agents recreate antipatterns by putting specialization in the *caller*
instead of on the *owner*. Snippets below are simplified from real
``benchmark-platform`` commits (hashes cited). Match the *shape*, not every
line still present on ``HEAD``.

**Wrong.** The consumer knows concrete subclasses or enum cases and branches.

**Right.** The consumer calls one method on the abstract root (or a shared
selection mixin). Leaves implement hooks; mixins share behavior.

Leaf hook instead of injected callable / sidecar helper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``history:`` ``fd2cc077`` — robust-background center.

**Incorrect** — root stores a callable; mode uses a parallel helper type:

.. code-block:: python

   class RobustBackgroundCenterStrategy(...):
       center_helper: ClassVar[Callable[[np.ndarray], float]]

       def center(self, values: np.ndarray) -> float:
           return float(type(self).center_helper(values))


   class BinnedModeCenterHelper:
       def __call__(self, values: np.ndarray) -> float:
           return float(threshold_primitives().binned_mode(values))


   class MeanRobustBackgroundCenterStrategy(RobustBackgroundCenterStrategy):
       center_helper = staticmethod(np.mean)


   class ModeRobustBackgroundCenterStrategy(RobustBackgroundCenterStrategy):
       center_helper = BinnedModeCenterHelper()

**Correct** — abstract hook on the root; leaves override:

.. code-block:: python

   class RobustBackgroundCenterStrategy(...):
       def center(self, values: np.ndarray) -> float:
           return float(type(self)._center(values))

       @staticmethod
       @abstractmethod
       def _center(values: np.ndarray) -> float:
           """Return the strategy-specific center estimate."""


   class MeanRobustBackgroundCenterStrategy(RobustBackgroundCenterStrategy):
       @staticmethod
       def _center(values: np.ndarray) -> float:
           return float(np.mean(values))


   class ModeRobustBackgroundCenterStrategy(RobustBackgroundCenterStrategy):
       @staticmethod
       def _center(values: np.ndarray) -> float:
           return float(threshold_primitives().binned_mode(values))

**Stop-rule.** Variation among registered leaves is a method override, not a
``ClassVar[Callable]`` or sidecar helper type.

Template method on the root
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``history:`` ``290f9d1e`` — illumination smoothing.

**Incorrect** — every leaf reimplements the public entrypoint:

.. code-block:: python

   class FitPolynomialSmoothingPlaneStrategy(SmoothingPlaneStrategy):
       def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
           return fit_polynomial_surface(request.pixel_data, request.mask)


   class GaussianFilterSmoothingPlaneStrategy(SmoothingPlaneStrategy):
       def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
           return IlluminationGaussianFilter(...).apply()

**Correct** — root owns the template; leaves own only the helper call:

.. code-block:: python

   class HelperBackedSmoothingPlaneStrategy(SmoothingPlaneStrategy):
       def smooth(self, request: SmoothingPlaneRequest) -> np.ndarray:
           return self._smooth_with_helper(request)

       @abstractmethod
       def _smooth_with_helper(self, request: SmoothingPlaneRequest) -> np.ndarray:
           """Delegate to the concrete helper without exposing child identity."""


   class FitPolynomialSmoothingPlaneStrategy(HelperBackedSmoothingPlaneStrategy):
       def _smooth_with_helper(self, request: SmoothingPlaneRequest) -> np.ndarray:
           return fit_polynomial_surface(request.pixel_data, request.mask)

Callers keep using ``strategy.smooth(request)``. They never name the leaf.

``elif`` on enum → EnumKeyed strategy leaves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``history:`` ``85013d59`` — global threshold methods.

**Incorrect** — consumer ladder:

.. code-block:: python

   if method is CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY:
       threshold = primitives.minimum_cross_entropy_threshold(...)
   elif method is CellProfilerThresholdMethod.LI:
       threshold = primitives.li_threshold(values)
   elif method is CellProfilerThresholdMethod.OTSU:
       threshold = primitives.otsu_threshold(values)
   elif method is CellProfilerThresholdMethod.TRIANGLE:
       threshold = primitives.triangle_threshold(values)
   # ... more elif arms ...
   else:
       raise NotImplementedError(f"Threshold method {method} not supported.")

**Correct** — one call through the strategy root; leaves register on the enum:

.. code-block:: python

   class GlobalThresholdMethodStrategy(
       EnumKeyedStrategyMixin[CellProfilerThresholdMethod],
       ABC,
       metaclass=AutoRegisterMeta,
   ):
       @classmethod
       def for_method(cls, method: CellProfilerThresholdMethod):
           return cls.for_enum_member(method)

       @abstractmethod
       def compute(self, request: GlobalThresholdRequest) -> float: ...


   # caller:
   threshold = GlobalThresholdMethodStrategy.for_method(method).compute(
       GlobalThresholdRequest(...)
   )

Same shape in ``history:`` ``1a453026`` — behavior moved *off* the enum:

.. code-block:: python

   # incorrect — methods on the enum itself
   def resolve(self, func: Callable[..., Any]) -> Callable[..., Any]:
       if self is RuntimeCallableView.DECORATED:
           return func
       if self is RuntimeCallableView.RAW:
           return inspect.unwrap(func)
       raise ValueError(...)

   # correct — EnumKeyedStrategyMixin leaves; enum stays a key
   class RuntimeCallableViewStrategy(
       EnumKeyedStrategyMixin[RuntimeCallableView],
       ABC,
       metaclass=AutoRegisterMeta,
   ):
       ...

**Stop-rule.** Closed enum variation → register a leaf. Do not grow ``elif``
in the caller. Do not put algorithm methods on the enum when a strategy root
exists.

MostDerived / nearest-MRO instead of priority ``if`` chains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``history:`` ``9434d847`` — ``openhcs/core/registry_strategies.py``.

**Incorrect** — local priority numbers, first ``isinstance`` wins, or
``sorted(..., key=priority)``.

**Correct** — inheritance expresses precedence; callers select once:

.. code-block:: python

   class MostDerivedContextStrategyMixin(Generic[_ContextT], ABC):
       """Selection returns the single most-derived matching implementation,
       so callers do not need local if/elif chains, priority numbers, or
       repeated registry scans."""

       @classmethod
       def for_context(cls, context: _ContextT, *, required: bool = True):
           ...

       @abstractmethod
       def matches(self, context: _ContextT) -> bool: ...

   # NominalTypeKeyed: instantiate the *most specific* owning type, not the
   # first registry hit that isinstance-matches.

**Stop-rule.** If two strategies could match, fix the hierarchy / ``matches``
— do not add a priority integer in the consumer.

Collapse thin AutoRegister shells into owner payload
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``history:`` ``99989d86`` — plane-axis projection.

**Incorrect** — one-method-per-leaf strategy family that only forwards:

.. code-block:: python

   class RuntimePlaneAxisProjectionStrategy(
       EnumKeyedStrategyMixin[RuntimePlaneAxis],
       ABC,
       metaclass=AutoRegisterMeta,
   ):
       axis: ClassVar[RuntimePlaneAxis]

       @abstractmethod
       def plane_index(self, projector, *, source_aliases) -> int | None: ...


   class RuntimeSlicePlaneAxisProjectionStrategy(RuntimePlaneAxisProjectionStrategy):
       axis = RuntimePlaneAxis.RUNTIME_SLICE

       def plane_index(self, projector, *, source_aliases) -> int | None:
           return projector.runtime_slice_plane_index()

**Correct** — hook lives on the enum member; strategy family deleted:

.. code-block:: python

   class RuntimePlaneAxis(str, Enum):
       def __new__(cls, value, plane_index_resolver):
           return str_enum_member_with_payload(
               cls, value,
               payload_attribute="_plane_index_resolver",
               payload=plane_index_resolver,
           )

       RUNTIME_SLICE = ("runtime_slice", runtime_slice_axis_plane_index)
       SOURCE_BINDING = ("source_binding", source_binding_axis_plane_index)

       def plane_index(self, projector, *, source_aliases) -> int | None:
           return self.plane_index_resolver(projector, source_aliases)

``history:`` ``96ffa98b`` — GrayToColor input discovery (unnecessary nominal).

**Incorrect** — AutoRegisterMeta + ``materialize()`` dynamic subclasses:

.. code-block:: python

   class _GrayToColorInputNameResolver(ABC, metaclass=AutoRegisterMeta):
       __registry_key__ = "scheme_literal"
       scheme_literal: ClassVar[str | None] = None

       @classmethod
       def for_module(cls, module):
           resolver_type = cls.__registry__.get(scheme.value)
           return resolver_type()


   class GrayToColorInputNameResolverDeclaration:
       def materialize(self) -> type[_GrayToColorInputNameResolver]:
           return type(self.class_name, (self.base,), {...})

**Correct** — dataclass + scheme map (no registry family):

.. code-block:: python

   @dataclass(frozen=True, slots=True)
   class _GrayToColorInputNameResolver:
       scheme: GrayToColorScheme
       image_settings: tuple[str, ...] = ()
       repeated_channels: bool = False

       @classmethod
       def for_module(cls, module):
           resolver = _GRAY_TO_COLOR_INPUT_NAME_RESOLVERS.get(scheme)
           if resolver is None:
               raise ValueError(...)
           return resolver


   _GRAY_TO_COLOR_INPUT_NAME_RESOLVERS = {
       GrayToColorScheme.RGB: _GrayToColorInputNameResolver(
           GrayToColorScheme.RGB,
           image_settings=GRAY_TO_COLOR_RGB_IMAGE_SETTINGS,
       ),
       # ... additional scheme declarations ...
   }

**Stop-rule.** If every leaf is a one-liner forward, put the hook/field on the
existing owner and delete the strategy family.

Lift leaf ``isinstance`` into root template + ClassVars
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``history:`` ``eaeb13d0`` — measurement query cache invalidation.

**Incorrect** — each leaf reimplements the same methods with its own
``isinstance``:

.. code-block:: python

   class ObjectFeatureValueCacheInvalidationPolicy(...):
       def entry_object_name(self, entry: object) -> str | None:
           if not isinstance(entry, RuntimeObjectMeasurementQuery):
               return None
           return entry.object_name

       def entry_feature_name(self, entry: object) -> str | None:
           if not isinstance(entry, RuntimeObjectMeasurementQuery):
               return None
           return entry.feature_name


   class ObjectLabelMeasurementValuesCacheInvalidationPolicy(...):
       def entry_object_name(self, entry: object) -> str | None:
           if not isinstance(entry, RuntimeObjectLabelMeasurementQuery):
               return None
           return entry.object_name
       # ... same pattern again ...

**Correct** — root owns the algorithm; leaves declare types/flags:

.. code-block:: python

   class MeasurementQueryCacheInvalidationPolicy(...):
       cache_accessor: ClassVar[Callable[..., MutableMapping[object, object]]]
       entry_type: ClassVar[type[object]]
       feature_scoped: ClassVar[bool] = False

       def entry_object_name(self, entry: object) -> str | None:
           if not isinstance(entry, type(self).entry_type):
               return None
           return entry.object_name

       def entry_feature_name(self, entry: object) -> str | None:
           if not type(self).feature_scoped:
               return None
           if not isinstance(entry, type(self).entry_type):
               return None
           return entry.feature_name


   class ObjectFeatureValueCacheInvalidationPolicy(
       MeasurementQueryCacheInvalidationPolicy
   ):
       cache_accessor = staticmethod(lambda adapter: adapter.object_feature_value_cache())
       entry_type = RuntimeObjectMeasurementQuery
       feature_scoped = True

Note the remaining ``isinstance`` checks the leaf's *declared* ``entry_type``,
not a hard-coded concrete class ladder in generic code. That is structural
validation against the owner ClassVar, not subclass dispatch.

**Stop-rule.** If you are about to write ``isinstance(x, ConcreteLeafA)`` /
``ConcreteLeafB`` in a shared policy, stop. Put ``entry_type`` (or a hook) on
the abstract owner and implement leaves as data + overrides.

Module declaration hooks (polymorphic owner)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generic CellProfiler code must not do:

.. code-block:: python

   if module_name == "IdentifyPrimaryObjects":
       ...
   elif isinstance(module, IdentifyPrimaryObjects):
       ...

It must call methods on ``CellProfilerModule`` (for example
``resolve_function``, ``contribute_source_bindings``) and let MRO / mixins
supply leaf behavior. ``history:`` ``63f0ede1`` and later cutovers deleted
name-keyed catalogs (``module_function_resolution``, ``module_roles``, …) in
favor of that shape.

**Stop-rule.** Naming a concrete module/strategy subclass in generic code is a
dispatch lattice. Extend the abstract owner instead.

Correct factorizations
----------------------

Not every large change is a lattice violation. These splits keep one owner.

Typed runtime value modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Move.** Monolithic ``runtime_values.py`` factored into typed modules such as
``runtime_image_values``, ``runtime_measurements``, ``runtime_object_labels``,
``runtime_relationships``, ``runtime_tabular_values``.

**Why correct.** Split by value kind / artifact type. Consumers still query
typed owners and strategies — not a second catalog of the same facts.

Microscope handler vs source bindings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Move.** Format handlers project a virtual workspace; source bindings name
and filter sources. Non-empty ``SourceBindingsConfig`` selects
``SourceBindingsHandler``.

**Why correct.** Different host boundaries (projection vs naming). See
:doc:`microscope_handler_integration` and :doc:`source_model`.

Fat module_declarations root
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Move.** Module semantics concentrate in ``module_declarations.py``.

**Why correct.** One declaration root may be large. Size is not a failure;
dual ownership is.

Collapse helper registries
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Move.** Microscope and materialization helper registries collapsed into
handlers and materialization core.

**Evidence.** ``history:`` ``4b3e0ce9``.

**Why correct.** Fewer roots, same semantics — the opposite of inventing
mirrors.

Meta rules for agents
---------------------

Checkpoint and compat commits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Commits titled as checkpoints or that grow compatibility bridges can be valid
*sequencing* while a cutover is incomplete. They are not an allowed steady
state. When the real owner exists, delete the bridge in the same migration that
moves the last consumer.

Do not resurrect gated paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests/unit/test_cellprofiler_static_deletion_gates.py`` is the machine-checked
forbidden list for the CellProfiler unification. If your change needs a symbol
from ``REQUIRED_DELETED_PATHS``, you are recreating a deleted lattice — stop
and extend the surviving owner instead.

Orphan sidecars
~~~~~~~~~~~~~~~

A file that still exists after its owning library is deleted (for example
``contracts.json`` after ``library.py``) is an orphan mirror. Do not attach new
readers. Delete or regenerate only under the declaration-owned path.

SSOT at a boundary
~~~~~~~~~~~~~~~~~~

Informally: zero-delay synchronization is the “single” in single source of
truth; native source evidence is the “source.” A value that is correct while
its owner is erased or duplicated is still an architecture failure.

Agent landing checklist
-----------------------

Before landing a new type, registry, package, strategy family, or dispatch
branch:

1. Name the question it answers.
2. ``rg`` for an existing owner (``__registry__``, declaration roots, strategy
   mixins, domain terms).
3. Classify the addition as parallel lattice, unnecessary nominal, nominal
   mirror, caller-side dispatch, or true owner with leaf hooks.
4. If specialization varies by subclass, put a hook on the abstract root (or
   a mixin) and implement it in the leaf — do not ``isinstance`` the leaf in
   the caller.
5. If the key is an enum, runtime type, or context, use the matching shared
   strategy mixin instead of ``elif`` or priority tables.
6. If a new strategy family only forwards, put the hook on the existing owner
   and delete the shell.
7. If you migrate callers off a mirror or dispatch ladder, delete it in the
   same change.
8. Run focused tests for the owner and, for CellProfiler unification surfaces,
   the static deletion gates.
