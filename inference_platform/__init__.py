"""Shared runtime for the AIM inference microservices.

Two kinds of process use this package:

* the **gateway** (:mod:`gateway.app`), a stateless HTTP API that accepts
  region-processing jobs and reports on them, and
* the **model workers** (:mod:`worker.main`), one container image per model,
  which take tasks off a per-model queue and run inference on a GPU.

Everything they share -- job and task records, the Redis Streams task queue,
overlay storage, the model registry and metrics -- lives here so that the two
sides cannot drift apart.
"""

__version__ = "1.0.0"
