# B1 full-scale runtime provenance diagnostic

The exact frozen command was launched once through a standalone Windows wrapper.
The wrapper wrote telemetry every two seconds, but was externally terminated
before post-child collection. The last durable sample was 49.65 seconds.
There was no Python traceback or native LightGBM error in stderr.

Root cause: **EXTERNAL_TERMINATION** (medium confidence). This is not called
OOM or timeout: CIM memory telemetry was denied, no Windows event was captured,
and no timeout ceiling was exposed. No scientific metric or B1 result was made.
