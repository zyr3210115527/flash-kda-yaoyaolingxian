"""Which kernel1 field first goes bad at CHUNK=32?

Uses the offsets exported from the extension rather than the byte constants the
older probes hardcode, since those move with CHUNK.
"""
import math, os, sys, torch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "tests"))
import torch_npu  # noqa
from flash_kda import _C

DEV = torch.device("npu:0"); D = 128
CH = _C.CHUNK; PT = _C.WS_PER_TILE
T, H = CH, 1                      # exactly one tile, simplest case
torch.manual_seed(0)
mk = lambda *s, dt=torch.bfloat16: torch.randn(*s, dtype=dt).to(DEV)
q, k, v, g = (mk(1, T, H, D) for _ in range(4))
beta = mk(1, T, H); a = mk(H, dt=torch.float32); db = mk(H, D, dt=torch.float32)
out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=DEV)
LB = float(os.environ.get("LB","-5.0"))
ws = torch.zeros(_C.get_workspace_size(T, H, 1), dtype=torch.uint8).to(DEV)
_C.fwd(q, k, v, g, beta, 1.0/math.sqrt(D), out, ws, a, db, LB)
torch_npu.npu.synchronize()
h = ws.cpu()

def bf16(off, n):
    return h[off:off+n*2].contiguous().view(torch.bfloat16).float()
def f32(off, n):
    return h[off:off+n*4].contiguous().view(torch.float32)

print(f"CHUNK={CH} per_tile={PT}")
for name, off, n, w in [
        ("k_decayed", 0, CH*D, 2), ("q_decayed", CH*D*2, CH*D, 2),
        ("k_inv", CH*D*2*2, CH*D, 2), ("k_restored", CH*D*2*3, CH*D, 2),
        ("g_total", _C.WS_OFF_KGTOTAL, D, 4),
        ("INV", _C.WS_OFF_KINV, CH*CH, 2),
        ("Mqk", _C.WS_OFF_KMQK, CH*CH, 2),
        ("identity", _C.WS_OFF_IDENTITY, CH*CH, 2)]:
    t = f32(off, n) if w == 4 else bf16(off, n)
    print(f"  {name:11} nan={int(t.isnan().sum()):>6}/{t.numel():<6} "
          f"absmax={t.abs().max().item():.4e}")
print("  out nan=", int(out.cpu().float().isnan().sum()))
