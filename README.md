# Mini LLM Inference Engine

A mini LLM inference server built from scratch on top of Qwen2.5, implementing round-robin request scheduling, automatic prefix caching, and speculative decoding — benchmarked against a naive single-request server under concurrent load.

Note: "continuous batching" is used elsewhere in this project (notebook, commit history) to refer to the round-robin request scheduler implemented here — iteration-level, capacity-bounded admission across requests. It does not include batched GPU execution (stacking multiple requests into a single forward pass).

## What's inside

- A hand-written, manually-managed KV-cache autoregressive decode loop
- A round-robin request scheduler with capacity-bounded admission
- Automatic prefix caching — detects the longest shared prefix across in-flight requests and reuses its KV cache
- Speculative decoding — a smaller draft model proposes multiple tokens per round, verified in a single batched forward pass by the target model
- A concurrency-controlled load-testing harness (TTFT / latency / throughput at p50 / p95 / p99)

## Results

Round-robin scheduling + prefix caching + speculative decoding vs a naive single-request-per-thread baseline, under concurrent load:

| concurrency | baseline throughput | our throughput | baseline latency | our latency |
|---|---|---|---|---|
| 5  | 12.1 tok/s | 17.6 tok/s | 8.3s  | 5.7s  |
| 10 | 9.8 tok/s  | 17.9 tok/s | 20.4s | 9.0s  |
| 20 | 10.3 tok/s | 18.2 tok/s | 38.8s | 18.0s |

**~1.8x higher throughput and ~2.2x lower latency under concurrent load.**

## Speculative decoding efficiency

![draft k sweep](assets/draft_k_sweep.png)

Speculative decoding delivers up to **4.55 tokens per target-model forward pass** at a draft window of k=8, compared to 1 token per forward pass with standard decoding.

## Repo structure

- `inference (6).ipynb` — the original notebook, run end-to-end on Kaggle (2x T4 GPUs)
- `src/` — the same code broken into individual files by component, for readability:
  - `model_setup.py` — model and tokenizer loading
  - `decode.py` — manual KV-cache decode loop
  - `scheduler.py` — request scheduler (round-robin, capacity-bounded)
  - `prefix_caching.py` — shared-prefix detection and KV cache reuse
  - `speculative_decoding.py` — draft/verify speculative decoding
  - `combined_pipeline.py` — prefix caching + speculative decoding combined
  - `servers.py` — the three FastAPI servers (baseline, round-robin, combined)
  - `load_testing.py` — concurrency load-test harness
  - `plotting.py` — benchmark plots
- `all_in_one.py` — the full notebook code in a single file
