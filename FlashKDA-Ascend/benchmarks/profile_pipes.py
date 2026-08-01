"""Hardware pipe counters for both kernels, via torch_npu.profiler.

Everything up to now was inferred: wall clock plus stubbing phases plus
arithmetic on transfer sizes. That found the big structural wins, but it cannot
see which functional unit is actually busy. AiCMetrics.PipeUtilization reports
per-pipe busy ratios directly.

What it showed at T=4096 H=64, and what none of the earlier method could:

    kernel1 (PrepareFused)      kernel2 (K2FusedAll)
      aic_mac_ratio   0.03        0.02     <- cube MACs almost idle
      aic_mte2_ratio  0.24        0.244    <- cube's largest pipe is memory-in
      aiv_vec_ratio   0.405       0.218
      aiv_scalar_ratio 0.326      0.306    <- a third of vector-core time is
                                              spent in the SCALAR unit

The scalar ratio is the surprise. In kernel2 the scalar unit is busier than the
vector unit. The cube, meanwhile, is at 2-3% MAC utilization -- it is not
computing, it is waiting on operands, which is the m=16 problem.

Note that a low aic_mac_ratio alongside a high cube_utilization(%) is not a
contradiction: the latter is occupancy (a block is resident), the former is
whether the MACs are doing anything.

Requires the CANN profiler; it works on this image even though torch's
elementwise operators do not, because it reads hardware counters rather than
dispatching aclnn kernels.

Usage:  python3 benchmarks/profile_pipes.py [T] [H]
"""
import csv
import glob
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tests'))

import torch_npu  # noqa: F401

DEV = torch.device("npu:0")
D = 128
OUT = "/tmp/flash_kda_prof"

KEYS = [
    ("Duration(us)", "wall"),
    ("aic_mac_ratio", "cube MAC"),
    ("aic_mte2_ratio", "cube mem-in"),
    ("aic_fixpipe_ratio", "cube fixpipe"),
    ("aiv_vec_ratio", "vec"),
    ("aiv_scalar_ratio", "scalar"),
    ("aiv_mte2_ratio", "vec mem-in"),
    ("aiv_mte3_ratio", "vec mem-out"),
]


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    from flash_kda import fwd, clear_workspace_cache

    torch.manual_seed(0)
    mk = lambda *s, dt=torch.bfloat16: torch.randn(*s, dtype=dt).to(DEV)
    q, k, v, g = (mk(1, T, H, D) for _ in range(4))
    beta = mk(1, T, H)
    a_log = mk(H, dt=torch.float32)
    dt_bias = mk(H, D, dt=torch.float32)
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=DEV)

    def call():
        fwd(q, k, v, g, beta, 1.0 / math.sqrt(D), out, a_log, dt_bias, -5.0)

    for _ in range(3):
        call()
    torch_npu.npu.synchronize()

    exp = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=exp,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(OUT)):
        for _ in range(3):
            call()
    torch_npu.npu.synchronize()

    files = glob.glob(os.path.join(OUT, "*", "ASCEND_PROFILER_OUTPUT",
                                   "kernel_details.csv"))
    if not files:
        print("no profiler output under", OUT)
        return
    rows = list(csv.DictReader(open(sorted(files)[-1])))

    print(f"T={T} H={H}")
    print(f"{'kernel':<24}" + "".join(f"{lbl:>13}" for _, lbl in KEYS))
    seen = set()
    for r in rows:
        n = r.get("Name", "")
        if "FlashKda" not in n or n in seen:
            continue
        seen.add(n)
        cells = []
        for key, _ in KEYS:
            val = r.get(key, "")
            try:
                val = f"{float(val):.3f}" if "ratio" in key else f"{float(val):.0f}"
            except ValueError:
                pass
            cells.append(f"{val:>13}")
        print(f"{n:<24}" + "".join(cells))

    clear_workspace_cache()


if __name__ == "__main__":
    main()
