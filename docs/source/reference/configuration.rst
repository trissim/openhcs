Configuration fields
====================

Use this page to look up the exact authoring path, type, default, accepted
values, inheritance behavior, and declaring owner of an OpenHCS configuration
field. For the mental model behind scopes and resolution, read
:doc:`../guide_for_biologists/configuration_reference`.

This reference is generated from the same typed declarations and introspection
used by the desktop help controls and ``openhcs_describe_config_schema``. It is
not a separately maintained field catalogue.

Global configuration
--------------------

``GlobalPipelineConfig`` provides concrete application and execution defaults.

.. openhcs-config-reference:: global

Pipeline configuration
----------------------

``PipelineConfig`` contains inheritable overrides for one pipeline.

.. openhcs-config-reference:: pipeline

Step configuration
------------------

``FunctionStep`` exposes the nested configuration families that one step may
override.

.. openhcs-config-reference:: step

Desktop UI configuration
------------------------

``UIConfig`` owns process-level desktop behavior such as logging, shortcuts,
progress updates, execution transport, and the local agent bridge. These fields
are edited through the UI configuration ObjectState, not ``ConfigPatch``.

.. openhcs-config-reference:: ui
