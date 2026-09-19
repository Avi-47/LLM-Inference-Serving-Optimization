import pandas as pd
import matplotlib.pyplot as plt

from load_testing import results_log


def plot_phase_comparison():
    df = pd.DataFrame(results_log)
    phases = ["dumb_baseline", "continuous_batching_v1", "combined_prefix_spec"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for phase in phases:
        sub = df[df["phase"] == phase]
        axes[0].plot(sub["concurrency"], sub["aggregate_tps"], marker="o", label=phase)
        axes[1].plot(sub["concurrency"], sub["ttft_p50"], marker="o", label=phase)
        axes[2].plot(sub["concurrency"], sub["latency_p50"], marker="o", label=phase)

    axes[0].set_title("Aggregate throughput")
    axes[0].set_xlabel("Concurrent requests")
    axes[0].set_ylabel("tokens/sec")

    axes[1].set_title("Time to first token (p50)")
    axes[1].set_xlabel("Concurrent requests")
    axes[1].set_ylabel("ms")
    axes[1].set_yscale("log")

    axes[2].set_title("Total request latency (p50)")
    axes[2].set_xlabel("Concurrent requests")
    axes[2].set_ylabel("ms")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig("phase_comparison.png", dpi=150)
    plt.show()


def plot_batch_size_sweep():
    df = pd.DataFrame(results_log)
    batch_df = df[df["phase"].str.startswith("batch_size_")].copy()
    batch_df["batch_size"] = batch_df["phase"].str.replace("batch_size_", "").astype(int)
    batch_df = batch_df.sort_values("batch_size")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(batch_df["batch_size"], batch_df["aggregate_tps"], marker="o", color="tab:blue", label="aggregate throughput")
    ax1.set_xlabel("max_batch_size")
    ax1.set_ylabel("Aggregate throughput (tokens/sec)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(batch_df["batch_size"], batch_df["ttft_p95"], marker="s", color="tab:red", label="TTFT p95")
    ax2.set_ylabel("TTFT p95 (ms)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Batch size tradeoff at concurrency=20")
    fig.tight_layout()
    plt.savefig("batch_size_sweep.png", dpi=150)
    plt.show()


def plot_draft_k_sweep():
    draft_k_stats = pd.DataFrame([
        {"k": 1, "tokens_per_target_call": 1.79, "aggregate_tps": 17.5},
        {"k": 2, "tokens_per_target_call": 2.62, "aggregate_tps": 19.0},
        {"k": 4, "tokens_per_target_call": 3.42, "aggregate_tps": 18.3},
        {"k": 8, "tokens_per_target_call": 4.55, "aggregate_tps": 15.8},
    ])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(draft_k_stats["k"], draft_k_stats["tokens_per_target_call"], marker="o", color="tab:green", label="tokens per target forward pass")
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("draft tokens per round (k)")
    ax1.set_ylabel("tokens per target forward pass", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:green")

    ax2 = ax1.twinx()
    ax2.plot(draft_k_stats["k"], draft_k_stats["aggregate_tps"], marker="s", color="tab:purple", label="aggregate throughput")
    ax2.set_ylabel("aggregate throughput (tokens/sec)", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")

    plt.title("Speculative decoding: draft window k vs efficiency and throughput")
    fig.tight_layout()
    plt.savefig("draft_k_sweep.png", dpi=150)
    plt.show()
