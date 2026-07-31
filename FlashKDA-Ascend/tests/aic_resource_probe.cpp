/**
 * Does constructing Catlass::Arch::Resource on an AIC core hang?
 *
 * This is the experiment that confirms or kills the current hypothesis, and it
 * takes about two minutes on a card.
 *
 * What is already known, all measured on a 910B3:
 *   - An AIC kernel that writes a scalar to GM (no Resource, no L1) completes.
 *   - An AIV kernel with its own TPipe + TBuf<VECCALC> doing DataCopy/Exp/DataCopy
 *     completes and is numerically exact.
 *   - Kernel1's AIV phases, which use Resource's ubBuf, complete.
 *   - Kernel1's AIC phases, which use Resource's l1Buf/l0*, hang -- and still
 *     hang when reduced to a single Nd2Nz GM->L1 DataCopy whose descriptor
 *     fields match catlass's own derivation exactly.
 *
 * Hypothesis: Catlass::Arch::Resource constructs *every* buffer unconditionally,
 * including a 192 KB UB allocation via GetTPipePtr()->InitBuffer, and an AIC
 * core has no UB. The vector phases survive because asking a vector core for L1
 * is harmless; asking the cube for UB is not.
 *
 * Stage A: AIC constructs Resource and immediately returns.
 * Stage B: AIC constructs only its own TPipe + TBuf<A1> and returns.
 * Stage C: Stage B plus one Nd2Nz GM->L1 DataCopy.
 *
 * If A hangs and B completes, the hypothesis holds and the fix is the explicit
 * per-core-type buffers in scratchpad/patch_bufs.py.
 * If A completes and C hangs, the fault is in the DataCopy itself and the next
 * suspect is the L1 destination extent rather than Resource.
 *
 * Build (adjust the catlass path):
 *   source /usr/local/Ascend/cann-8.5.0/set_env.sh
 *   CANN=/usr/local/Ascend/cann-8.5.0
 *   bisheng -std=c++17 -x cce --cce-aicore-arch=dav-c220 -DCATLASS_ARCH=2201 -O2 \
 *     -I <catlass>/include -I $CANN/include \
 *     -I $CANN/compiler/tikcpp/tikcfw -I $CANN/compiler/tikcpp/tikcfw/impl \
 *     -I $CANN/compiler/tikcpp/tikcfw/interface \
 *     aic_resource_probe.cpp -o aic_probe -L $CANN/lib64 -lascendcl -lruntime
 *
 * Run: ./aic_probe A   (then B, then C)
 *
 * Note this is a standalone binary. If all three stages pass here, rerun the
 * same three as kernels inside the Python extension -- the standalone/extension
 * split is itself a known variable in this project.
 */

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "kernel_operator.h"

#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>

using ArchTag = Catlass::Arch::AtlasA2;

constexpr int ROWS = 16;
constexpr int COLS = 128;

// Stage A: construct Resource on the cube, do nothing else.
extern "C" CATLASS_GLOBAL void ProbeResource(GM_ADDR mark)
{
    if constexpr (g_coreType == AscendC::AIC) {
        Catlass::Arch::Resource<ArchTag> resource;
        AscendC::GlobalTensor<int32_t> g;
        g.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mark));
        g.SetValue(0, 1);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(g);
    }
}

// Stage B: own TPipe with only an L1 buffer.
extern "C" CATLASS_GLOBAL void ProbeOwnL1(GM_ADDR mark)
{
    if constexpr (g_coreType == AscendC::AIC) {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::A1> l1;
        pipe.InitBuffer(l1, ROWS * COLS * sizeof(bfloat16_t));

        AscendC::GlobalTensor<int32_t> g;
        g.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mark));
        g.SetValue(0, 2);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(g);
    }
}

// Stage C: Stage B plus the Nd2Nz load that hangs in the real kernel.
extern "C" CATLASS_GLOBAL void ProbeNd2Nz(GM_ADDR src, GM_ADDR mark)
{
    if constexpr (g_coreType == AscendC::AIC) {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::A1> l1;
        pipe.InitBuffer(l1, ROWS * COLS * sizeof(bfloat16_t));
        AscendC::LocalTensor<bfloat16_t> dst = l1.Get<bfloat16_t>();

        AscendC::GlobalTensor<bfloat16_t> gsrc;
        gsrc.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(src));

        // Field values follow CopyGmToL1's own derivation for RowMajor -> zN:
        //   dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0
        //                 = (roundUp16(ROWS) * 16) / 16 = ROWS
        //   dstNzNStride  = layoutDst.stride(0) / ELE_NUM_PER_C0 = 1
        //   srcDValue     = layoutSrc.stride(0) = COLS
        AscendC::Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = ROWS;
        p.dValue = COLS;
        p.srcNdMatrixStride = 0;
        p.srcDValue = COLS;
        p.dstNzC0Stride = ROWS;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;
        AscendC::DataCopy(dst, gsrc, p);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((event_t)0);

        AscendC::GlobalTensor<int32_t> g;
        g.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mark));
        g.SetValue(0, 3);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(g);
    }
}

#define CHECK(x)                                                   \
    do {                                                           \
        aclError _e = (x);                                         \
        if (_e != ACL_SUCCESS) {                                   \
            printf("FAILED %s -> %d\n", #x, static_cast<int>(_e)); \
            return 1;                                              \
        }                                                          \
    } while (0)

int main(int argc, char **argv)
{
    const char stage = (argc > 1) ? argv[1][0] : 'A';

    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream));

    void *mark = nullptr;
    void *src = nullptr;
    CHECK(aclrtMalloc(&mark, 32, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemset(mark, 32, 0, 32));
    CHECK(aclrtMalloc(&src, ROWS * COLS * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemset(src, ROWS * COLS * 2, 0, ROWS * COLS * 2));

    printf("stage %c: launching\n", stage);
    switch (stage) {
        case 'A': ProbeResource<<<1, nullptr, stream>>>(reinterpret_cast<GM_ADDR>(mark)); break;
        case 'B': ProbeOwnL1<<<1, nullptr, stream>>>(reinterpret_cast<GM_ADDR>(mark)); break;
        default:  ProbeNd2Nz<<<1, nullptr, stream>>>(reinterpret_cast<GM_ADDR>(src),
                                                     reinterpret_cast<GM_ADDR>(mark)); break;
    }

    aclError sync = aclrtSynchronizeStreamWithTimeout(stream, 15000);
    int32_t host = 0;
    aclrtMemcpy(&host, sizeof(host), mark, sizeof(host), ACL_MEMCPY_DEVICE_TO_HOST);
    printf("stage %c: sync=%d %s, marker=%d\n", stage, static_cast<int>(sync),
           sync == ACL_SUCCESS ? "(completed)" : "(TIMED OUT)", host);

    aclrtFree(src);
    aclrtFree(mark);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
