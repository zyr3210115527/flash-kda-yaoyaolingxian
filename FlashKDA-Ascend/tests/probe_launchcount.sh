#!/bin/bash
# Runs ON the card. Is kernel2's hang a function of launch count?
#
# Every kernel2 phase stubbed to return immediately still hangs, so the fault is
# the launch structure, not the bodies. Kernel1 issues 4 launches and works;
# kernel2 issues 1 + 5*chunks + 1 (22 for T=64). This caps the chunk loop to
# find the threshold.
set -u
cd /user/zhouyiran/flashkda/FlashKDA-Ascend || exit 1
source /usr/local/Ascend/cann-8.5.0/set_env.sh 2>/dev/null || true
export PATH=/usr/local/python3.11.14/bin:$PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

A=src/fwd_kernel2.asc
cp "$A" /tmp/k2asc_cap.bak

for CAP in 0 1 2 4; do
  cp /tmp/k2asc_cap.bak "$A"
  perl -0pi -e "s/(for \(int c = 0; c < )maxChunks(; \+\+c\) \{)/\${1}($CAP < maxChunks ? $CAP : maxChunks)\$2/s" "$A"
  ( cd build && make -j8 > /tmp/bc.log 2>&1 )
  if [ $? -ne 0 ]; then echo "cap=$CAP -> BUILD FAIL"; grep -E " error" /tmp/bc.log|head -2; continue; fi
  cp build/_C*.so flash_kda/
  out=$(PYTHONPATH=$PWD:$PWD/tests timeout 120 python3 -u test_npu_nocompute.py 2>&1); rc=$?
  n=$((1 + CAP*5 + 1))
  if [ $rc -eq 124 ]; then
    echo "cap=$CAP (${n} k2 launches) -> HANG"
  elif echo "$out" | grep -qa 'aicore exception'; then
    echo "cap=$CAP (${n} k2 launches) -> AICORE EXCEPTION"
  elif echo "$out" | grep -qa 'err_ratio'; then
    echo "cap=$CAP (${n} k2 launches) -> RAN  $(echo "$out"|grep -ao 'err_ratio=[0-9.e+-]*'|head -1)"
  else
    echo "cap=$CAP (${n} k2 launches) -> other rc=$rc"
  fi
done

cp /tmp/k2asc_cap.bak "$A"
