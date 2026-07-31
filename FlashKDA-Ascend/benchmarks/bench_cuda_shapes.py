"""How far from the CUDA reference are we, at the CUDA reference's own shapes?

Every previous comparison used our shapes against the CUDA implementation's
T=8192 H=64, which is not like-for-like: our microseconds-per-token-head still
improves with size, so comparing our small shape to their large one overstates
the gap.

Reaching T=8192 H=64 was previously impossible because the wrapper built the
workspace as host-side zeros and copied it -- ~20 GB through the host. Now that
it is allocated on device with torch.empty and cached, the only limit is HBM,
and the card has 64 GB.

Runs through the public wrapper, so this is what a caller actually gets.
"""
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tests'))

import torch_npu  # noqa: F401

DEV = torch.device("npu:0")
D = 128

# The CUDA implementation's own benchmark, on an H20: ms and us/token/head.
# H=96 has its own per-token figure; using H=64's for both understates it.
CUDA_REF = {(8192, 64): (1.6217, 0.0031), (8192, 96): (2.6220, 0.0033)}
CUDA_PER_DEFAULT = 0.0031


def main():
    from flash_kda import fwd, _C, clear_workspace_cache

    hdr = (f"{'T':>6} {'H':>4} {'workspace':>10} {'ms':>9} "
           f"{'us/tok/head':>12} {'vs CUDA':>9}")
    print(hdr)

    for T, H in [(2048, 16), (2048, 64), (4096, 64), (8192, 64), (8192, 96)]:
        ws_bytes = _C.get_workspace_size(T, H, 1)
        try:
            torch.manual_seed(0)
            mk = lambda *s, dt=torch.bfloat16: torch.randn(*s, dtype=dt).to(DEV)
            q, k, v, g = (mk(1, T, H, D) for _ in range(4))
            beta = mk(1, T, H)
            a_log = mk(H, dt=torch.float32)
            dt_bias = mk(H, D, dt=torch.float32)
            out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=DEV)
            scale = 1.0 / math.sqrt(D)

            def call():
                fwd(q, k, v, g, beta, scale, out, a_log, dt_bias, -5.0)

            for _ in range(2):
                call()
            torch_npu.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                call()
            torch_npu.npu.synchronize()
            ms = (time.perf_counter() - t0) / 5 * 1e3

            per = ms * 1e3 / (T * H)
            ref = CUDA_REF.get((T, H))
            if ref:
                # Direct wall-clock ratio at the same shape -- the only
                # like-for-like number here.
                tag = f"{ms / ref[0]:>8.0f}x"
                note = f"   (CUDA/H20: {ref[0]:.4f} ms, same shape)"
            else:
                tag = f"{per / CUDA_PER_DEFAULT:>8.0f}x"
                note = "   (vs CUDA at T=8192 H=64, different shape)"
            print(f"{T:>6} {H:>4} {ws_bytes / 2**30:>9.1f}G {ms:>9.2f} "
                  f"{per:>12.4f} {tag}{note}")
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).splitlines()[0][:45]
            print(f"{T:>6} {H:>4} {ws_bytes / 2**30:>9.1f}G   "
                  f"FAILED: {type(exc).__name__}: {msg}")
        finally:
            for name in ('q', 'k', 'v', 'g', 'beta', 'out'):
                if name in locals():
                    del locals()[name]
            clear_workspace_cache()
            torch_npu.npu.empty_cache()

    print()
    print("Caveats: different silicon (910B3 vs H20), and this is one card.")


if __name__ == "__main__":
    main()
