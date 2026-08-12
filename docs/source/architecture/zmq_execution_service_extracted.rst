ZMQ execution transition
========================

ZMQRuntime owns generic transport; OpenHCS owns its compiled-execution adapter.
The generic connection retains the readiness ``PongResponse``; OpenHCS compares
the application identity in that handshake with its own version declaration.
The UI observes the resulting compatibility value and may request a
state-preserving endpoint and desktop restart. No second probe, copied version
field, or UI-owned connection flag participates in the decision.

See :doc:`external_foundations`.
