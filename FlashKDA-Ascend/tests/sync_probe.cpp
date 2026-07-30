/**
 * Minimal AIC/AIV cross-core handshake probe.
 *
 * The full kernel1 hangs with "aicore execution times out", which points at the
 * CrossCore handshake rather than at any of the compute. On Atlas A2 one AIC is
 * paired with GetSubBlockNum() AIVs, so the set/wait counts on the two sides do
 * not match one-for-one, and the exact semantics decide whether this pattern
 * deadlocks.
 *
 * This program runs the same two-round handshake kernel1 uses, with all compute
 * removed, and writes a per-core progress marker to GM after each step. Reading
 * the markers back tells us exactly which wait blocked.
 *
 * Build/run standalone (no torch), so the iteration is seconds:
 *   bisheng -std=c++17 -x cce --cce-aicore-arch=dav-c220 -O2 \
 *     -I<catlass>/include -I$CANN/include -I$CANN/compiler/tikcpp/tikcfw{,/impl,/interface} \
 *     sync_probe.cpp -o sync_probe -L$CANN/lib64 -lascendcl -lruntime
 */

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "kernel_operator.h"

#include <acl/acl.h>
#include <cstdio>

using ArchTag = Catlass::Arch::AtlasA2;

constexpr Catlass::Arch::FlagID ELEM_READY = 1;
constexpr Catlass::Arch::FlagID MMA_READY = 2;

// Progress markers, one uint32 slot per stage per core.
constexpr int kSlotsPerCore = 8;

class SyncProbe {
public:
    struct Params {
        GM_ADDR marks;
        int cores;
    };

    CATLASS_DEVICE
    SyncProbe() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        const uint32_t core = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        const uint32_t sub = AscendC::GetSubBlockIdx();
        if (static_cast<int>(core) >= params.cores) {
            return;
        }

        Mark(params, core, 0, 100 + sub);
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(elemReady_);
        Mark(params, core, 1, 110 + sub);

        Catlass::Arch::CrossCoreWaitFlag(mmaReady_);
        Mark(params, core, 2, 120 + sub);

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(elemReady_);
        Mark(params, core, 3, 130 + sub);

        Catlass::Arch::CrossCoreWaitFlag(mmaReady_);
        Mark(params, core, 4, 140 + sub);
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        const uint32_t core = AscendC::GetBlockIdx();
        if (static_cast<int>(core) >= params.cores) {
            return;
        }

        Mark(params, core, 5, 200);
        Catlass::Arch::CrossCoreWaitFlag(elemReady_);
        Mark(params, core, 6, 210);

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(mmaReady_);
        Catlass::Arch::CrossCoreWaitFlag(elemReady_);
        Mark(params, core, 7, 220);

        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(mmaReady_);
    }

private:
    // Scalar GM store: no vector unit needed, works from both AIC and AIV.
    CATLASS_DEVICE
    void Mark(Params const &params, uint32_t core, int slot, uint32_t value)
    {
        AscendC::GlobalTensor<uint32_t> g;
        g.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(params.marks));
        g.SetValue(core * kSlotsPerCore + slot, value);
        AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(g);
    }

    Catlass::Arch::CrossCoreFlag elemReady_{ELEM_READY};
    Catlass::Arch::CrossCoreFlag mmaReady_{MMA_READY};
};

extern "C" CATLASS_GLOBAL void SyncProbeKernel(uint64_t syncAddr, SyncProbe::Params params)
{
    AscendC::SetSyncBaseAddr(syncAddr);
    SyncProbe op;
    op(params);
}

#define CHECK(x)                                                                  \
    do {                                                                          \
        aclError _e = (x);                                                        \
        if (_e != ACL_SUCCESS) {                                                  \
            printf("FAILED %s -> %d\n", #x, static_cast<int>(_e));                \
            return 1;                                                             \
        }                                                                         \
    } while (0)

int main(int argc, char **argv)
{
    const int cores = (argc > 1) ? atoi(argv[1]) : 2;

    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream));

    const size_t bytes = static_cast<size_t>(cores) * kSlotsPerCore * sizeof(uint32_t);
    void *dev = nullptr;
    CHECK(aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemset(dev, bytes, 0, bytes));

    uint64_t syncAddr = 0;
    CHECK(aclrtGetHardwareSyncAddr(reinterpret_cast<void **>(&syncAddr)));

    SyncProbe::Params params{reinterpret_cast<GM_ADDR>(dev), cores};
    printf("launching %d cores...\n", cores);
    SyncProbeKernel<<<cores, nullptr, stream>>>(syncAddr, params);

    aclError sync = aclrtSynchronizeStreamWithTimeout(stream, 10000);
    printf("sync result: %d %s\n", static_cast<int>(sync),
           sync == ACL_SUCCESS ? "(completed)" : "(TIMED OUT / failed)");

    uint32_t *host = new uint32_t[cores * kSlotsPerCore]();
    CHECK(aclrtMemcpy(host, bytes, dev, bytes, ACL_MEMCPY_DEVICE_TO_HOST));

    const char *names[kSlotsPerCore] = {
        "aiv:entry", "aiv:after set#1", "aiv:after wait#1",
        "aiv:after set#2", "aiv:after wait#2",
        "aic:entry", "aic:after wait#1", "aic:after wait#2"};
    for (int c = 0; c < cores; ++c) {
        printf("core %d:\n", c);
        for (int s = 0; s < kSlotsPerCore; ++s) {
            uint32_t v = host[c * kSlotsPerCore + s];
            printf("  %-18s %s (%u)\n", names[s], v ? "reached" : "NOT REACHED", v);
        }
    }

    delete[] host;
    aclrtFree(dev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
