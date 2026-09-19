Lightweight LLM Serving & Inference Optimization

An experimental LLM inference stack focused on improving response time and serving efficiency under concurrent workloads.

The project implements a small inference server around Qwen2.5 and explores several techniques commonly used to improve autoregressive generation: request scheduling, KV-cache reuse, and speculative decoding. The different approaches are benchmarked against a simple baseline server to measure their impact on throughput and latency.

Project Overview

The goal of this project is to understand what happens inside an LLM serving system rather than relying entirely on high-level inference frameworks.

The implementation includes:

A custom autoregressive generation loop with explicit KV-cache handling

Round-robin scheduling for multiple simultaneous requests

Shared-prefix detection and automatic KV-cache reuse

Speculative generation using a smaller draft model

A load-testing setup for evaluating concurrent requests

Latency and throughput measurements across different concurrency levels

A note about batching

Some parts of the project use the term continuous batching. Here, that terminology refers to the scheduler's iteration-level admission of requests rather than true GPU-side batched execution.

The implementation does not combine multiple independent requests into a single batched model forward pass as a production inference engine such as vLLM would.

Performance Evaluation

The optimized pipeline was compared with a straightforward single-request baseline under increasing concurrent load.

Concurrent Requests	Baseline Throughput	Optimized Throughput	Baseline Latency	Optimized Latency
5	12.1 tok/s	17.6 tok/s	8.3 s	5.7 s
10	9.8 tok/s	17.9 tok/s	20.4 s	9.0 s
20	10.3 tok/s	18.2 tok/s	38.8 s	18.0 s

Across these experiments, combining the serving optimizations resulted in approximately 1.8× higher throughput and substantially lower latency under concurrent workloads.

These numbers are environment-dependent and should be treated as measurements from the experiment rather than general production benchmarks.

Speculative Decoding

The repository also evaluates how the size of the speculative draft window affects generation efficiency.

With a draft window of k=8, the experiment reaches approximately 4.55 generated tokens per target-model forward pass, compared with one token per target forward pass for conventional autoregressive decoding.

The experiment illustrates the main idea behind speculative decoding: use a smaller model to propose several tokens and let the larger target model verify those proposals together.

Components

The implementation is organized into separate modules so that each optimization can be examined independently.

.
├── inference (6).ipynb
├── all_in_one.py
├── src/
│   ├── model_setup.py
│   ├── decode.py
│   ├── scheduler.py
│   ├── prefix_caching.py
│   ├── speculative_decoding.py
│   ├── combined_pipeline.py
│   ├── servers.py
│   ├── load_testing.py
│   └── plotting.py
└── assets/
    └── draft_k_sweep.png

Source modules

model_setup.py
Handles model and tokenizer initialization.

decode.py
Contains the manually implemented autoregressive decoding logic and KV-cache management.

scheduler.py
Implements the round-robin request scheduler and capacity-controlled request admission.

prefix_caching.py
Finds reusable prompt prefixes and avoids recomputing their KV representations.

speculative_decoding.py
Implements the draft-and-verify generation procedure.

combined_pipeline.py
Combines prefix caching with speculative decoding.

servers.py
Contains the FastAPI server implementations used for the different serving configurations.

load_testing.py
Provides the concurrent workload generator and collects serving metrics.

plotting.py
Generates plots used for analyzing the benchmark results.

Running the Experiments

The main experiment is available as a Jupyter notebook:

inference (6).ipynb


The notebook was originally executed in a GPU environment using two NVIDIA T4 GPUs.

For users who prefer a single Python file, the complete implementation is also available in:

all_in_one.py

What This Project Demonstrates

Rather than treating LLM inference as a single model call, this project breaks serving into several independent optimization problems:

Scheduling — deciding which request should make progress next.

Caching — avoiding repeated computation for identical prompt prefixes.

Generation efficiency — reducing the number of expensive target-model forward passes.

Concurrency — understanding how serving behavior changes as the number of simultaneous requests increases.

Measurement — evaluating changes using throughput and latency rather than relying only on theoretical improvements.

This makes the repository useful as a small-scale exploration of the mechanisms behind modern LLM inference serving.

Limitations

This is an educational/experimental inference implementation rather than a production serving framework.

In particular:

GPU execution is not optimized to the level of mature inference engines.

The scheduler does not provide true production-grade continuous batching.

Benchmark results depend heavily on the hardware and model configuration.

Memory management and fault tolerance are intentionally simplified.

The implementation is designed primarily for understanding and experimentation.

Experiments to Try

Some useful directions for extending the project include:

Testing additional concurrency levels

Comparing different draft models

Sweeping speculative decoding window sizes

Measuring TTFT separately from end-to-end latency

Testing different prompt-prefix sharing patterns

Comparing against established inference engines

Investigating actual batched GPU execution

Profiling GPU utilization and memory consumption

License & Attribution

If portions of this implementation were adapted from another repository, notebook, or codebase, retain the original project's license and attribution requirements here.

Add the original source and any required attribution before publishing a derivative version.
