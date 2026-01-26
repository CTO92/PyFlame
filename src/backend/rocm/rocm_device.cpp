// ROCm Device Management Implementation
// Phase 1: Infrastructure Foundation

#ifdef PYFLAME_HAS_ROCM

#include "pyflame/backend/rocm/rocm_backend.hpp"

namespace pyflame::backend::rocm {

int get_device_count() {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess) {
        // Don't throw - return 0 to indicate no devices
        return 0;
    }
    return count;
}

bool is_available() {
    return get_device_count() > 0;
}

DeviceInfo get_device_info(int device_id) {
    DeviceInfo info;
    info.device_id = device_id;

    // Validate device ID
    int count = get_device_count();
    if (device_id < 0 || device_id >= count) {
        throw std::runtime_error(
            "Invalid device ID: " + std::to_string(device_id) +
            " (available: 0-" + std::to_string(count - 1) + ")");
    }

    // Get device properties
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));

    info.name = props.name;
    info.architecture = "gfx" + std::to_string(props.gcnArch);
    info.total_memory = props.totalGlobalMem;
    info.compute_units = props.multiProcessorCount;
    info.max_threads_per_block = props.maxThreadsPerBlock;
    info.warp_size = props.warpSize;

    // Get free memory (requires setting device temporarily)
    int current_device;
    HIP_CHECK(hipGetDevice(&current_device));

    if (current_device != device_id) {
        HIP_CHECK(hipSetDevice(device_id));
    }

    size_t free_mem, total_mem;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    info.free_memory = free_mem;

    // Restore previous device if we changed it
    if (current_device != device_id) {
        HIP_CHECK(hipSetDevice(current_device));
    }

    return info;
}

void set_device(int device_id) {
    int count = get_device_count();
    if (device_id < 0 || device_id >= count) {
        throw std::runtime_error(
            "Invalid device ID: " + std::to_string(device_id) +
            " (available: 0-" + std::to_string(count - 1) + ")");
    }
    HIP_CHECK(hipSetDevice(device_id));
}

int get_device() {
    int device_id;
    HIP_CHECK(hipGetDevice(&device_id));
    return device_id;
}

void synchronize() {
    HIP_CHECK(hipDeviceSynchronize());
}

}  // namespace pyflame::backend::rocm

#endif  // PYFLAME_HAS_ROCM
