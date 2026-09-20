#pragma once

#include <cuda_runtime.h>

namespace ninfer::ops {

// Number of SMs on the current device, queried once and cached. Launch tuning
// follows the actual GPU instead of the reference RTX 5090 (170 SMs); falls
// back to 170 when the query fails so reference behavior is preserved.
inline int device_sm_count() {
    static const int sm_count = [] {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) { return 170; }
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, device) != cudaSuccess) { return 170; }
        return props.multiProcessorCount > 0 ? props.multiProcessorCount : 170;
    }();
    return sm_count;
}

} // namespace ninfer::ops
