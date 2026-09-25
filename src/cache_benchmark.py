"""Small benchmark harness for real generation functions.

Pass a callable that returns generated token IDs. The harness does not estimate
or manufacture speedups; measurements are from the supplied implementation.
"""

import time


def benchmark_modes(prompts, generate_without_cache, generate_with_cache,
                    generate_with_validated_cache, iterations=1,
                    validated_cache_metrics=None):
    results = {}
    for name, generator in (
            ("no_prefix_cache", generate_without_cache),
            ("prefix_cache", generate_with_cache),
            ("prefix_cache_lru_version_validation", generate_with_validated_cache)):
        latencies = []
        first_token_latencies = []
        token_count = 0
        started = time.perf_counter()
        for _ in range(iterations):
            for prompt in prompts:
                request_started = time.perf_counter()
                token_iterator = iter(generator(prompt))
                try:
                    next(token_iterator)
                except StopIteration:
                    tokens = []
                else:
                    first_token_latencies.append(time.perf_counter() - request_started)
                    tokens = [None] + list(token_iterator)
                if not tokens:
                    first_token_latencies.append(time.perf_counter() - request_started)
                token_count += len(tokens)
                latencies.append(time.perf_counter() - request_started)
        elapsed = time.perf_counter() - started
        results[name] = {
            "ttft_seconds": sum(first_token_latencies) / len(first_token_latencies),
            "total_latency_seconds": sum(latencies) / len(latencies),
            "aggregate_throughput_tokens_per_second": token_count / elapsed if elapsed else 0.0,
        }
    if validated_cache_metrics is not None:
        results["prefix_cache_lru_version_validation"].update(
            validated_cache_metrics()
        )
    return results