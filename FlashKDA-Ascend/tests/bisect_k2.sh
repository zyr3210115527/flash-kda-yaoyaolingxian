#!/bin/bash
# Runs ON the card. Bisects kernel2's phases cumulatively to find the faulting one.
#
# Kernel1 is known good. Each variant stubs out a suffix of kernel2's phase
# methods, so the first variant that faults names the culprit.
set -u
cd /user/zhouyiran/flashkda/FlashKDA-Ascend || exit 1

source /usr/local/Ascend/cann-8.5.0/set_env.sh 2>/dev/null || true
export PATH=/usr/local/python3.11.14/bin:$PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

BASE=/tmp/k2_bisect_base.bak
K=include/flash_kda/fwd_kernel2.hpp
cp "$K" "$BASE"

stub () {   # stub <method> <arity>   1 = (params), 2 = (params, chunk)
  if [ "$2" = "2" ]; then
    perl -0pi -e 's/(CATLASS_DEVICE void '"$1"'\(Params const& params, int chunk\)\n    \{\n)/$1        return;\n/s' "$K"
  else
    perl -0pi -e 's/(CATLASS_DEVICE void '"$1"'\(Params const& params\)\n    \{\n)/$1        return;\n/s' "$K"
  fi
}

run_variant () {
  local name="$1"
  ( cd build && make -j8 > /tmp/bv.log 2>&1 )
  if [ $? -ne 0 ]; then
    echo "$name -> BUILD FAIL"; grep -E " error" /tmp/bv.log | head -3; return
  fi
  cp build/_C*.so flash_kda/
  local out rc
  out=$(PYTHONPATH=$PWD:$PWD/tests timeout 150 python3 -u test_npu_nocompute.py 2>&1); rc=$?
  if [ $rc -eq 124 ]; then
    echo "$name -> HANG"
  elif echo "$out" | grep -qa 'aicore exception'; then
    echo "$name -> AICORE EXCEPTION"
  elif echo "$out" | grep -qa 'err_ratio'; then
    echo "$name -> RAN   $(echo "$out" | grep -ao 'err_ratio=[0-9.e+-]*  *finite=[A-Za-z]*  *nonzero=[A-Za-z]*' | head -1)"
  else
    echo "$name -> other (rc=$rc)"; echo "$out" | tail -2 | sed 's/^/        /'
  fi
}

ALL2="RunBuildU RunPreGemms RunFinishOut RunPostGemms RunFinishChunk"

# v0: everything stubbed
cp "$BASE" "$K"; stub RunInitState 1; stub RunStoreFinalState 1
for m in $ALL2; do stub "$m" 2; done
run_variant "v0 all stubbed        "

# v1: + InitState
cp "$BASE" "$K"; stub RunStoreFinalState 1
for m in $ALL2; do stub "$m" 2; done
run_variant "v1 +InitState         "

# v2: + BuildU
cp "$BASE" "$K"; stub RunStoreFinalState 1
for m in RunPreGemms RunFinishOut RunPostGemms RunFinishChunk; do stub "$m" 2; done
run_variant "v2 +BuildU            "

# v3: + PreGemms   (first cube phase)
cp "$BASE" "$K"; stub RunStoreFinalState 1
for m in RunFinishOut RunPostGemms RunFinishChunk; do stub "$m" 2; done
run_variant "v3 +PreGemms  [cube]  "

# v4: + FinishOut
cp "$BASE" "$K"; stub RunStoreFinalState 1
for m in RunPostGemms RunFinishChunk; do stub "$m" 2; done
run_variant "v4 +FinishOut         "

# v5: + PostGemms  (second cube phase)
cp "$BASE" "$K"; stub RunStoreFinalState 1
stub RunFinishChunk 2
run_variant "v5 +PostGemms [cube]  "

# v6: everything
cp "$BASE" "$K"
run_variant "v6 all active         "

cp "$BASE" "$K"
