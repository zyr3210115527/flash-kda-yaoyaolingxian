"""Does the bf16 state update accumulate error along the sequence?

Writing the state update (k_res^T @ u) as bf16 rather than fp32 halves
kernel2s largest transfer: a [128,128] tile written by PostGemms and read by
DecayState, 2.0 GB and ~2.7 ms at T=4096 H=64.

test_shapes says it costs almost nothing, but every case there is at most 128
tokens -- 8 chunks. The state is the recurrent carrier, decayed and added to
once per chunk, so a rounding introduced in its update compounds along the
sequence in a way 8 chunks cannot reveal. At T=8192 there are 512 chunks.

So the question is not whether it passes at the test shapes, it is whether the
gap against the fp32 update GROWS with T. Flat means the rounding stays local
and the speed is free. Growing means it accumulates, and then it is not worth
taking however fast it is.

The reference is the same kernel with the fp32 update, which isolates this one
change instead of re-measuring the whole bf16 pipelines error budget.
"""
import math, os, sys, torch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "tests"))
import torch_npu  # noqa

DEV = torch.device("npu:0"); D = 128; CHUNK = 16

def run(T, H, seed=0):
    from flash_kda import fwd, clear_workspace_cache
    torch.manual_seed(seed)
    mk = lambda *s, dt=torch.bfloat16: torch.randn(*s, dtype=dt).to(DEV)
    q, k, v, g = (mk(1, T, H, D) for _ in range(4))
    beta = mk(1, T, H)
    a_log = mk(H, dt=torch.float32); dt_bias = mk(H, D, dt=torch.float32)
    out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=DEV)
    fs = torch.empty(1, H, D, D, dtype=torch.float32, device=DEV)
    fwd(q, k, v, g, beta, 1.0 / math.sqrt(D), out, a_log, dt_bias, -5.0, final_state=fs)
    torch_npu.npu.synchronize()
    o = out.cpu().float(); f = fs.cpu().float()
    clear_workspace_cache(); torch_npu.npu.empty_cache()
    return o, f

mode = os.environ.get("FLASH_KDA_BF16_UPDATE", "0")
for T in (256, 512, 1024, 2048, 4096):
    o, f = run(T, 8)
    torch.save({"out": o, "state": f}, "/tmp/drift_%s_%d.pt" % (mode, T))
print("done mode=%s" % mode)
