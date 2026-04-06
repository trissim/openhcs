---
title: 'zmqruntime: Distributed Execution Framework with Dual-Channel ZMQ Transport'
tags:
  - Python
  - distributed computing
  - ZMQ
  - microscopy
  - scientific computing
authors:
  - name: Tristan Simas
    orcid: 0000-0002-6526-3149
    affiliation: 1
affiliations:
  - name: McGill University
    index: 1
date: 15 January 2026
bibliography: paper.bib
---

# Summary

`zmqruntime` provides a distributed execution framework for Python using ZMQ (ZeroMQ) as the transport layer. The key innovation is the **dual-channel pattern**: a control channel for commands (execute, status, cancel) and a data channel for results and streaming output. This separation enables:

- **Non-blocking execution**: Submit a job, get an execution ID, poll for results asynchronously
- **Streaming results**: Large arrays stream directly to disk/memory without buffering in memory
- **Graceful cancellation**: Cancel running jobs without killing the server
- **Auto-spawning servers**: If no server is running, the client spawns one automatically

```python
from zmqruntime import ExecutionClient

client = ExecutionClient(port=5555)
exec_id = client.execute(pipeline_code, config_code)  # Non-blocking
status = client.get_status(exec_id)  # Poll for results
results = client.get_results(exec_id)  # Retrieve when ready
```

# Statement of Need

Scientific pipelines often require distributed execution: process thousands of images across multiple machines, stream results to visualization tools, and handle failures gracefully. Traditional approaches either use heavyweight frameworks (Celery, Dask) or hand-written socket code.

`zmqruntime` provides a lightweight middle ground: ZMQ's simplicity with structured message types, dual-channel transport for non-blocking execution, and automatic server spawning for development convenience.

# Software Design

**Dual-Channel Transport**: Control channel (REQ/REP) for commands; data channel (PUSH/PULL) for results. This prevents head-of-line blocking where a slow result transfer blocks incoming commands.

```python
# Control channel: fast, synchronous
client.execute(pipeline_code, config)  # Returns execution_id immediately

# Data channel: asynchronous, streaming
results = client.get_results(execution_id)  # Retrieves streamed results
```

**Message Type Dispatch**: Enum-based message types with automatic handler dispatch. Adding a new command requires only defining the enum and implementing the handler method:

```python
class ControlMessageType(Enum):
    EXECUTE = "execute"
    STATUS = "status"
    CANCEL = "cancel"

    def dispatch(self, server, message):
        return getattr(server, self.get_handler_method())(message)
```

**Execution Tracking**: Each execution gets a unique ID. Clients poll for status without blocking. Results are streamed to disk or memory as they arrive.

**Auto-Spawning Servers**: If no server is running on the specified port, the client automatically spawns one. This enables development workflows where users don't need to manually start servers.

# Research Application

`zmqruntime` powers distributed execution in OpenHCS, enabling high-throughput screening where thousands of images are processed across multiple worker nodes. The dual-channel design allows the orchestrator to submit new jobs while previous jobs are still streaming results.

# References

