# Bring-up on Cybertron (Ascend 910B)

HiDevLab is not needed. Cybertron's Wulanchabu cluster has Ascend 910B — Atlas
A2, `CATLASS_ARCH=2201` — which is exactly what this port targets. Guiyang has
910C (Atlas A3) if more capacity is needed.

Everything below has been executed and verified, not copied from documentation.

## Getting a card

Cybertron's job API authenticates with your Teleport certificate, so a valid
`tsh` login is all you need.

```bash
TSH="$HOME/Library/Application Support/Cursor/User/globalStorage/yangsuiyun.cybertron/bin/tsh"
"$TSH" status          # must show a non-expired cluster login
CERT="$HOME/.tsh/keys/teleport.cybertron.modelbest.co/zhouyiran.crt"
B64=$(base64 < "$CERT" | tr -d '\n')
```

Every request carries two headers:

```
User-Agent: Cybertron VSCode Extension
X-Teleport-Cert: <base64 of the .crt above>
```

Create a preemptable devspace with `POST /api/job?project_id=<id>`:

```json
{
  "cluster": "wulan",
  "namespace": "training",
  "training_type": "devspace",
  "priority": "preemptable",
  "requeue_on_preemption": false,
  "code_type": "image",
  "entry": "sleep infinity",
  "image": "cybertron/autoloop-harness:ascend910b-vllm0.12.0-cann8.5.0-ubuntu22.04",
  "resources": {"gpu_series": "910b", "gpu_num": 1, "cpu_num": 24, "memory_size": 180},
  "resource_pool_id": 225,
  "filesystem_id": 268,
  "scheduler_backend": "volcano",
  "duration": 240,
  "billing_account_id": "N00069",
  "description": "..."
}
```

`priority: "preemptable"` is what keeps this off other people's quota — the
Ascend pools are otherwise fully subscribed, and preemptable slots do not
displace running work.

Useful read-only endpoints while setting this up:

| Endpoint | Purpose |
|---|---|
| `GET /api/cluster/` | cluster ids and their GPU models |
| `GET /api/resource_pool/` | full pool tree with per-cluster quotas |
| `GET /api/resources/spec?cluster_id=&resource_pool=` | per-machine cpu/mem/gpu ceilings |
| `GET /api/file_system/?cluster_id=` | filesystem ids |
| `GET /api/image/?cluster_id=&limit=2000` | available images |
| `GET /api/job/?status=Running&own=false&limit=500` | what images actually run on a cluster |
| `GET /api/job?id=<id>` | status, plus `killed_reason` / `failure_evidence` on failure |
| `DELETE /api/job?id=<id>&reason=<text>` | release the job |

### Two traps worth knowing

**Pick an image that is actually synced to the target cluster.** The first
attempt used `sglang:v0.5.16-cann9.0.0-910b`, which resolves to the Guiyang
registry (`swr.cn-north-9`) and cannot be pulled on Wulanchabu. The job died
with `ImagePullBackOff`. The reliable way to choose is to list *running* jobs on
that cluster and reuse an image someone else is already running.

**A devspace inherits `entry` from its project.** Creating one under an existing
project silently copies that project's training launch command. Set `entry`
explicitly.

## Connecting

```bash
"$TSH" ls                                     # find the node name
"$TSH" ssh root@devspace-zhouyiran-rl-zyr-<jobid>
```

The login is **`root`**, not your username — `zhouyiran@` fails with
"unknown user".

## Environment on the pod

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # sets ASCEND_HOME_PATH
export PATH=/usr/local/python3.11.14/bin:$PATH       # python is not on PATH by default
```

Verified present:

| Component | Version / path |
|---|---|
| NPU | Ascend 910B3, 65536 MB HBM |
| CANN | 8.5.0 at `/usr/local/Ascend/cann-8.5.0` |
| Compilers | `bisheng`, `ccec` in `$ASCEND_HOME_PATH/bin` |
| CMake ASC module | `$ASCEND_HOME_PATH/aarch64-linux/lib64/cmake/ASCConfig.cmake` |
| cmake | 3.22.1 |
| Python | 3.11.14 at `/usr/local/python3.11.14` |
| PyTorch | 2.8.0 + torch_npu 2.8.0, `torch.npu.is_available() == True` |
| Arch | aarch64 |

`npu-smi` and `torch_npu` both fail until `set_env.sh` is sourced —
`libc_sec.so` not found, and a torch backend-extension load error respectively.

### Filesystem

- `/user/zhouyiran` — writable, persistent, shared. Put source here.
- `/root`, `/tmp`, `/workspace` — writable, container-local (500 GB overlay).
- `/wulan` — the JuiceFS mount, **read-only at the root**. Do not try to
  `mkdir /wulan/<name>`, and never run an unbounded `find /` — it will walk a
  petabyte.

### Network

atomgit.com is reachable, so catlass can be cloned directly on the pod.
github.com is not, so pushing to GitHub has to happen from your laptop.

## Verifying the toolchain

Do this before touching kernel code. It is the step whose absence produced the
inherited draft.

```bash
cd /user/zhouyiran/flashkda
git clone --depth 1 https://atomgit.com/cann/catlass.git    # lands on b71539e
# wait for checkout to finish -- `ls tools` must succeed, not just `ls examples`
cd catlass
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh 00_basic_matmul
./output/bin/00_basic_matmul 256 512 1024 0
```

Expected, and confirmed on 2026-07-30:

```
Detected NPU_MODEL: Ascend910B3
CATLASS_ARCH is not defined, use default value "2201"
[INFO]Target '00_basic_matmul' built successfully
Compare success.
```

`Compare success` means the kernel ran on the NPU and matched a CPU reference.
The `find_package(ASC)` → bisheng → link → execute path is proven end to end.

If you kick off the build before `git clone` finishes checking out, CMake fails
with `add_subdirectory given source "tools" which is not an existing directory`.
That is a race, not a real error.
