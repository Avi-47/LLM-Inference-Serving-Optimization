import asyncio
import time

import httpx

results_log = []


async def single_request(client, prompt, max_tokens=20, port=8000):
    start = time.perf_counter()
    first_token_time = None
    tokens = []

    async with client.stream(
        "POST", f"http://localhost:{port}/generate",
        json={"prompt": prompt, "max_tokens": max_tokens},
        timeout=60
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                tokens.append(line[6:])

    end = time.perf_counter()
    decode_time = end - first_token_time
    decode_tokens = len(tokens) - 1
    tps = decode_tokens / decode_time if decode_time > 0 and decode_tokens > 0 else 0.0

    return {
        "ttft": (first_token_time - start) * 1000,
        "latency": (end - start) * 1000,
        "tokens": len(tokens),
        "tps": tps
    }


async def run_load_test(concurrency, prompt="The capital of France is", prompts=None, max_tokens=20, port=8000):
    if prompts is None:
        prompts = [prompt]
    async with httpx.AsyncClient() as client:
        batch_start = time.perf_counter()
        tasks = [single_request(client, prompts[i % len(prompts)], max_tokens, port) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)
        batch_end = time.perf_counter()
    return results, batch_end - batch_start


def summarize(results, concurrency, wall_time, phase="dumb_baseline"):
    ttfts = sorted(r["ttft"] for r in results)
    latencies = sorted(r["latency"] for r in results)
    tps_values = sorted(r["tps"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)
    aggregate_tps = total_tokens / wall_time

    def pct(values, p):
        idx = min(int(len(values) * p), len(values) - 1)
        return values[idx]

    row = {
        "phase": phase,
        "concurrency": concurrency,
        "ttft_p50": pct(ttfts, 0.5), "ttft_p95": pct(ttfts, 0.95), "ttft_p99": pct(ttfts, 0.99),
        "latency_p50": pct(latencies, 0.5), "latency_p95": pct(latencies, 0.95),
        "tps_p50": pct(tps_values, 0.5),
        "aggregate_tps": aggregate_tps
    }
    results_log.append(row)

    print(f"concurrency={concurrency}")
    print(f"  TTFT        p50={row['ttft_p50']:.1f}ms  p95={row['ttft_p95']:.1f}ms  p99={row['ttft_p99']:.1f}ms")
    print(f"  latency     p50={row['latency_p50']:.1f}ms  p95={row['latency_p95']:.1f}ms")
    print(f"  per-req TPS p50={row['tps_p50']:.1f}")
    print(f"  aggregate throughput = {row['aggregate_tps']:.1f} tokens/sec")
