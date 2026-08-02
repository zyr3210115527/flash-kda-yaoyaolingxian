"""SUPERSEDED -- kept for the record, do not trust its numbers.

This harness calls the `flash_kda.fwd` wrapper, which at the time allocated the
workspace on every call: 302 MB of host-side zeros copied to the device per
invocation at T=1024 H=8. That was 88-98% of everything it measured, and it is
where the "790x slower than CUDA" figure came from. The wrapper now caches the
workspace, so the allocation is gone, but this file is left as the artifact that
made the mistake visible.

Use benchmarks/ab_compare.py (interleaved A/B, medians, resolves ~1%) or
benchmarks/bench_cuda_shapes.py (the reference shapes) instead.
"""

import torch
import torch_npu
import math
import time
import argparse
from flash_kda import fwd, get_workspace_size
from torch_ref import torch_ref


def benchmark_fwd(B, T, H, D=128, lower_bound=-5.0, num_iters=50, warmup=5):
    """Benchmark flash_kda.fwd() on NPU."""
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(D)

    # Create inputs
    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    g = torch.randn(B, T, H, D, device=device, dtype=dtype)
    beta = torch.randn(B, T, H, device=device, dtype=dtype)
    A_log = torch.randn(H, device=device, dtype=torch.float32)
    dt_bias = torch.randn(H, D, device=device, dtype=torch.float32)
    out = torch.empty_like(q)

    # Warmup
    for _ in range(warmup):
        fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound)
    torch_npu.synchronize()

    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound)
    torch_npu.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / num_iters * 1000
    total_tokens = B * T * H
    throughput = total_tokens / (elapsed / num_iters)  # tokens/sec

    print(f"flash_kda | B={B} T={T} H={H} D={D} | avg={avg_ms:.2f}ms | throughput={throughput:.0f} tok/s")
    return avg_ms, throughput


def benchmark_ref(B, T, H, D=128, lower_bound=-5.0, num_iters=5, warmup=1):
    """Benchmark torch_ref on NPU (much slower, fewer iterations)."""
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    g = torch.randn(B, T, H, D, device=device, dtype=dtype)
    beta = torch.randn(B, T, H, device=device, dtype=dtype)
    A_log = torch.randn(H, device=device, dtype=torch.float32)
    dt_bias = torch.randn(H, D, device=device, dtype=torch.float32)
    out = torch.empty_like(q)

    # Warmup
    for _ in range(warmup):
        torch_ref(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound)
    torch_npu.synchronize()

    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        torch_ref(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound)
    torch_npu.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / num_iters * 1000
    print(f"torch_ref | B={B} T={T} H={H} D={D} | avg={avg_ms:.2f}ms")
    return avg_ms


def main():
    parser = argparse.ArgumentParser(description="FlashKDA-Ascend benchmark")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compare", action="store_true", help="Also benchmark torch_ref")
    args = parser.parse_args()

    print(f"=== FlashKDA-Ascend Benchmark ===")
    print(f"Config: B={args.B} T={args.T} H={args.H} D={args.D}")

    kernel_ms, throughput = benchmark_fwd(
        args.B, args.T, args.H, args.D,
        num_iters=args.iters, warmup=args.warmup,
    )

    if args.compare:
        ref_ms = benchmark_ref(args.B, args.T, args.H, args.D)
        speedup = ref_ms / kernel_ms
        print(f"Speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()
