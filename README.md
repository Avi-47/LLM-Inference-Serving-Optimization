Mini LLM Serving Lab

A from-scratch exploration of LLM inference serving using Qwen2.5, with an emphasis on understanding what happens between a user request and token generation.

Instead of relying completely on a high-level generation API, this project implements the decoding loop, KV-cache management, request scheduling, prefix reuse, and speculative decoding explicitly. The resulting components are exposed through lightweight FastAPI servers and evaluated under concurrent workloads.

The project is primarily an inference-engine learning and experimentation project rather than a production-ready serving framework.

What this project explores

The implementation builds the serving stack incrementally:

Client Requests
FastAPI Server
Serving Strategy
Baseline Generator
Round-Robin Scheduler
Optimized Pipeline
Request Queue
Capacity-Limited Active Requests
One Decode Step per Request
Prefix Cache
Speculative Decoder
Reuse Shared KV Cache
Draft Model
Candidate Tokens
Target Model Verification
Token Stream
Client

The main ideas implemented in the repository are:

Manual autoregressive decoding with past_key_values

Explicit KV-cache management

Round-robin scheduling across active requests

Capacity-limited admission of requests

Reuse of shared prompt prefixes

Speculative decoding with a smaller draft model

Streaming token responses using Server-Sent Events

Concurrent load testing and latency/throughput measurement

Models

The experiments use two Qwen2.5 instruction-tuned models:

Role	Model
Target model	Qwen/Qwen2.5-1.5B-Instruct
Draft model	Qwen/Qwen2.5-0.5B-Instruct

The target model is responsible for producing the final generation. The smaller model is used only for speculative token proposals.

Both models are loaded in FP16 and placed on CUDA:

target_model = AutoModelForCausalLM.from_pretrained(
    target_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)

draft_model = AutoModelForCausalLM.from_pretrained(
    draft_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)

1. Manual KV-Cache Decoding

The first stage replaces the normal model.generate() path with an explicit decoding loop.

During the initial prompt pass, the model produces a KV cache:

out = model(input_ids, use_cache=True)
past_key_values = out.past_key_values


For subsequent tokens, only the newly generated token is passed back to the model:

out = model(
    input_ids=next_token,
    past_key_values=past_key_values,
    use_cache=True
)


This avoids recomputing the complete prompt and generated sequence on every decoding step.

The basic flow is:

Prompt
Tokenize
Initial Forward Pass
KV Cache
Next Token
Feed One Token
Updated KV Cache
Next Token

The implementation uses greedy decoding, selecting the token with the highest logit at each step.

2. Streaming Inference Server

The manual generator is exposed through FastAPI.

Each generated token is decoded and returned through a streaming response:

Client
  │
  │ POST /generate
  ▼
FastAPI
  │
  ▼
Generator
  │
  ├── token 1 ──► client
  ├── token 2 ──► client
  ├── token 3 ──► client
  └── ...


The response uses:

text/event-stream


so that the client can observe tokens as they are generated rather than waiting for the entire response.

3. Round-Robin Request Scheduling

The baseline server generates each request independently.

The next version introduces a request manager that maintains:

a waiting queue

a collection of active requests

a maximum number of simultaneously active requests

flowchart TD
    A[Incoming Requests] --> B[Waiting Queue]

    B --> C{Capacity Available?}

    C -- Yes --> D[Admit Request]
    C -- No --> B

    D --> E[Active Requests]

    E --> F[Request 1]
    E --> G[Request 2]
    E --> H[Request 3]
    E --> I[Request N]

    F --> J[One Token]
    G --> K[One Token]
    H --> L[One Token]
    I --> M[One Token]

    J --> E
    K --> E
    L --> E
    M --> E


The scheduler repeatedly calls next() on each active generator.

Conceptually:

Request A → token
Request B → token
Request C → token
Request A → token
Request B → token
Request C → token
...


This prevents one long generation from occupying the server's entire execution loop while other requests wait.

Important terminology

This project uses the phrase iteration-level scheduling rather than claiming production-grade continuous batching.

The scheduler interleaves independent generation streams. It does not construct a GPU batch containing multiple unrelated user requests for every decoding step.

That distinction matters when comparing this implementation with production inference engines.

4. Prefix KV-Cache Reuse

Many real workloads contain prompts with a common beginning.

For example:

"You are a helpful assistant. Answer concisely: What is the capital of France?"

"You are a helpful assistant. Answer concisely: What is the capital of Japan?"

"You are a helpful assistant. Answer concisely: What is the capital of Germany?"


The initial portion is identical.

Instead of recomputing that shared prefix for every request, this project searches for a matching token prefix and stores its KV cache.

flowchart TD
    A[Incoming Prompt] --> B[Tokenize Prompt]

    B --> C{Cached Prefix?}

    C -- Yes --> D[Retrieve KV Cache]
    C -- No --> E[Compare with Other Waiting Prompts]

    E --> F{Shared Prefix Found?}

    F -- Yes --> G[Compute Shared Prefix Once]
    G --> H[Store KV Cache]
    H --> D

    F -- No --> I[Normal Prefill]

    D --> J[Process Remaining Suffix]
    I --> J

    J --> K[Autoregressive Generation]


The implementation searches for the longest cached prefix that matches the beginning of the current token sequence.

The cached state contains the model's past_key_values, allowing the suffix to be processed without recomputing the cached portion.

The code also checks that the optimized generation produces the same token sequence as the ordinary decoding path for the test prompts.

5. Speculative Decoding

The project also implements speculative decoding using two models.

Draft model
    │
    ├── proposes token 1
    ├── proposes token 2
    ├── proposes token 3
    └── proposes token 4
             │
             ▼
       Target model
             │
             ├── verifies proposal 1
             ├── verifies proposal 2
             ├── verifies proposal 3
             └── verifies proposal 4


The smaller Qwen2.5-0.5B-Instruct model proposes several tokens.

The larger Qwen2.5-1.5B-Instruct model then evaluates the proposed sequence in a single forward call and determines which proposed tokens can be accepted.

If a proposed token disagrees with the target model's greedy prediction, the target prediction is used as the correction.

The target KV cache is then truncated to the appropriate accepted prefix before generation continues.

Why speculative decoding can help

Ordinary decoding generally requires a target-model forward pass for each newly generated token.

Speculative decoding attempts to generate several tokens from the inexpensive draft model and have the target model verify them together.

If multiple draft tokens are accepted, more than one output token can be produced from a target-model verification step.

The important metric here is therefore not simply the draft model's speed, but:

accepted output tokens
───────────────────────
target-model calls


The experiment records this as tokens per target forward pass.

6. Combining the Optimizations

The final server combines:

Request scheduling

Prefix KV-cache reuse

Speculative decoding

The high-level execution path becomes:

flowchart TD
    A[Incoming Request] --> B[Request Queue]

    B --> C[Round-Robin Scheduler]

    C --> D[Tokenize Prompt]

    D --> E{Shared Prefix Available?}

    E -- Yes --> F[Reuse Cached Target + Draft KV]
    E -- No --> G[Normal Prompt Prefill]

    F --> H[Current Target/Draft State]
    G --> H

    H --> I[Draft Model]

    I --> J[Propose K Tokens]

    J --> K[Target Model Verification]

    K --> L{Draft Tokens Accepted?}

    L -- Yes --> M[Emit Accepted Tokens]
    L -- Partial --> N[Use Target Correction]
    L -- No --> N

    N --> M
    M --> O{Generation Complete?}

    O -- No --> C
    O -- Yes --> P[Finish Request]


This allows the project to study how multiple inference optimizations interact rather than evaluating each technique completely in isolation.

Benchmarking

A concurrent load-testing client is included using httpx and asyncio.

The test client records:

TTFT — time to first token

Total latency — request start to completion

Per-request token rate

Aggregate throughput

The code also calculates percentile measurements such as p50, p95, and p99 for TTFT and latency.

Metrics

TTFT

time of first generated token
──────────────────────────────
request start time


Total latency

request completion time - request start time


Aggregate throughput

total generated tokens
──────────────────────
wall-clock test duration


These measurements are useful because optimizing average token generation speed alone does not necessarily improve the experience of concurrent users.

Experimental Results

The following measurements were obtained from the project's benchmark runs:

Concurrent Requests	Baseline Throughput	Optimized Throughput	Baseline Latency	Optimized Latency
5	12.1 tok/s	17.6 tok/s	8.3 s	5.7 s
10	9.8 tok/s	17.9 tok/s	20.4 s	9.0 s
20	10.3 tok/s	18.2 tok/s	38.8 s	18.0 s

Across these recorded runs, the combined configuration achieved approximately:

1.8× the measured aggregate throughput

2.2× lower measured latency

These are experiment-specific measurements, not universal performance guarantees. Results can change with GPU model, CUDA/PyTorch versions, prompt distribution, sequence length, concurrency, and model configuration.

Speculative Decoding Experiment

The repository also contains a draft-window sweep.

The measured values used for the experiment are:

Draft window k	Tokens / target forward pass	Aggregate throughput
1	1.79	17.5 tok/s
2	2.62	19.0 tok/s
4	3.42	18.3 tok/s
8	4.55	15.8 tok/s

This demonstrates an important trade-off.

Increasing k can increase the number of accepted tokens per target verification call, but a larger speculative window does not automatically mean higher end-to-end throughput.

For this experiment, the highest measured tokens-per-target-forward-pass value occurs at k=8, while the highest measured aggregate throughput occurs at a smaller draft window.

That distinction is useful when discussing speculative decoding: verification efficiency and overall serving throughput are related, but they are not the same metric.

Speculative Decoding Plot

The plot compares the speculative window against both:

tokens generated per target-model forward pass

aggregate throughput

The horizontal reference at 1.0 represents the conventional one-token-per-target-step baseline.

Repository Layout
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

inference (6).ipynb

The notebook containing the end-to-end experiments.

The original experiment environment used Kaggle with 2× NVIDIA T4 GPUs.

all_in_one.py

A consolidated version of the implementation.

src/model_setup.py

Model and tokenizer initialization.

src/decode.py

Manual autoregressive decoding and KV-cache handling.

src/scheduler.py

Request management, waiting queues, active requests, and round-robin stepping.

src/prefix_caching.py

Shared-prefix detection and KV-cache reuse.

src/speculative_decoding.py

Draft-and-verify speculative generation.

src/combined_pipeline.py

Combines prefix caching and speculative decoding.

src/servers.py

FastAPI server implementations.

src/load_testing.py

Concurrent request generation and benchmark metric collection.

src/plotting.py

Benchmark visualization.

Running the Project

Install the main dependencies:

pip install torch transformers accelerate


For the server and load-testing components:

pip install fastapi uvicorn httpx


Additional analysis/plotting dependencies:

pip install pandas matplotlib


The notebook can then be executed in a CUDA-enabled environment with sufficient GPU memory.

Limitations

This implementation intentionally keeps the serving stack small enough to inspect and understand.

It should not be interpreted as a replacement for production inference engines.

Some important limitations are:

Request scheduling is implemented in Python and does not provide true multi-request GPU batching.

The prefix cache is a simple in-memory structure rather than a production cache manager.

KV-cache copying and management are intentionally straightforward.

The speculative decoder uses greedy decoding and a fixed draft-model strategy.

The benchmark uses relatively short generations.

Hardware utilization is not optimized to the level of specialized inference systems.

The benchmark values depend on the specific execution environment.

The project is therefore best viewed as an educational implementation for studying inference-serving techniques.

Key Takeaways

This project demonstrates several layers of an LLM serving system:

                 LLM Serving
                      │
       ┌──────────────┼──────────────┐
       │              │              │
   Decoding       Scheduling      Caching
       │              │              │
   KV Cache       Round Robin    Prefix Reuse
       │
       ▼
 Speculative Decoding
       │
       ▼
 Draft Model → Target Verification
       │
       ▼
 Concurrent Serving
       │
       ▼
 TTFT / Latency / Throughput


The main objective is not simply to make a model generate text, but to understand how model execution, cache reuse, scheduling, and concurrent request handling interact inside an inference server.

Attribution

If this repository contains code, experiments, diagrams, or other material adapted from another project, retain the original project's license and attribution requirements.

This README describes the implementation and experiments represented in the current repository.
