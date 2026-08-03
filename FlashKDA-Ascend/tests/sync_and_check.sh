#!/bin/bash
# Bring a fresh devspace up to the current tree and run the ordered checks.
#
# The workspace under /user/zhouyiran/flashkda is on JuiceFS and survives pods,
# but its sources are from the previous session. This syncs them from the
# scratchpad clone (which tracks GitHub) and rebuilds.
#
# Usage: sync_and_check.sh <node-name>
set -u

NODE="${1:?usage: sync_and_check.sh <node-name>}"
TSH="$HOME/Library/Application Support/Cursor/User/globalStorage/yangsuiyun.cybertron/bin/tsh"
LOCAL="$HOME/Documents/可选/bio-pipeline-kg-matcher/scratchpad/repo/FlashKDA-Ascend"
REMOTE=/user/zhouyiran/flashkda/FlashKDA-Ascend

echo "=== packing local sources ==="
cd "$LOCAL" || exit 1
# COPYFILE_DISABLE stops macOS tar emitting AppleDouble "._foo.py" resource
# forks, which land on the device as null-byte files that look like source.
COPYFILE_DISABLE=1 tar czf /tmp/sync.tgz include src tests flash_kda CMakeLists.txt || exit 1
ls -la /tmp/sync.tgz

echo "=== uploading ==="
"$TSH" scp /tmp/sync.tgz "root@${NODE}:/tmp/" || exit 1

echo "=== unpacking and rebuilding ==="
"$TSH" ssh "root@${NODE}" 'bash -lc "
set -e
source /usr/local/Ascend/cann-8.5.0/set_env.sh 2>/dev/null || true
export PATH=/usr/local/python3.11.14/bin:\$PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

cd '"$REMOTE"'
tar xzf /tmp/sync.tgz
echo \"--- sources synced ---\"

# catlass is a sibling checkout, symlinked in
[ -e catlass ] || ln -sfn /user/zhouyiran/flashkda/catlass catlass
ls -d catlass/include >/dev/null && echo \"catlass present\"

rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=\$(which python3) > /tmp/cfg.log 2>&1
echo \"cmake=\$?\"
make -j8 > /tmp/build.log 2>&1
echo \"make=\$?\"
grep -E \" error\" /tmp/build.log | head -10
ls -la _C*.so 2>/dev/null | awk \"{print \\\$5, \\\$9}\"
# The tests import flash_kda._C from the package dir, not from build/.
cp _C*.so ../flash_kda/ && echo \"installed .so into flash_kda/\"
"'
