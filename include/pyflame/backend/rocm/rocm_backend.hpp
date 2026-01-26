#pragma once

// Only compile if ROCm is enabled
#ifdef PYFLAME_HAS_ROCM

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <miopen/miopen.h>

#include <memory>
#include <string>
#include <stdexcept>

namespace pyflame::backend::rocm {

// ============================================================================
// Error Checking Macros
// ============================================================================

/// Error checking macro for HIP calls
/// Usage: HIP_CHECK(hipMalloc(&ptr, size));
/// Note: In release builds (NDEBUG), detailed location info is omitted
#ifndef NDEBUG
#define HIP_CHECK(call)                                                      \
    do {                                                                     \
        hipError_t err = (call);                                             \
        if (err != hipSuccess) {                                             \
            throw std::runtime_error(                                        \
                std::string("HIP error at ") + __FILE__ + ":" +              \
                std::to_string(__LINE__) + " in " + __func__ + ": " +        \
                hipGetErrorString(err) + " (" + std::to_string(err) + ")");  \
        }                                                                    \
    } while (0)
#else
#define HIP_CHECK(call)                                                      \
    do {                                                                     \
        hipError_t err = (call);                                             \
        if (err != hipSuccess) {                                             \
            throw std::runtime_error(                                        \
                std::string("HIP error: ") + hipGetErrorString(err));        \
        }                                                                    \
    } while (0)
#endif

/// Error checking macro for rocBLAS calls
/// Note: In release builds (NDEBUG), detailed location info is omitted
#ifndef NDEBUG
#define ROCBLAS_CHECK(call)                                                  \
    do {                                                                     \
        rocblas_status status = (call);                                      \
        if (status != rocblas_status_success) {                              \
            throw std::runtime_error(                                        \
                std::string("rocBLAS error at ") + __FILE__ + ":" +          \
                std::to_string(__LINE__) + " in " + __func__ + ": " +        \
                rocblas_status_to_string(status));                           \
        }                                                                    \
    } while (0)
#else
#define ROCBLAS_CHECK(call)                                                  \
    do {                                                                     \
        rocblas_status status = (call);                                      \
        if (status != rocblas_status_success) {                              \
            throw std::runtime_error(                                        \
                std::string("rocBLAS error: ") +                             \
                rocblas_status_to_string(status));                           \
        }                                                                    \
    } while (0)
#endif

/// Error checking macro for MIOpen calls
/// Note: In release builds (NDEBUG), detailed location info is omitted
#ifndef NDEBUG
#define MIOPEN_CHECK(call)                                                   \
    do {                                                                     \
        miopenStatus_t status = (call);                                      \
        if (status != miopenStatusSuccess) {                                 \
            throw std::runtime_error(                                        \
                std::string("MIOpen error at ") + __FILE__ + ":" +           \
                std::to_string(__LINE__) + " in " + __func__ +               \
                ": status code " + std::to_string(static_cast<int>(status)));\
        }                                                                    \
    } while (0)
#else
#define MIOPEN_CHECK(call)                                                   \
    do {                                                                     \
        miopenStatus_t status = (call);                                      \
        if (status != miopenStatusSuccess) {                                 \
            throw std::runtime_error(                                        \
                std::string("MIOpen error: status code ") +                  \
                std::to_string(static_cast<int>(status)));                   \
        }                                                                    \
    } while (0)
#endif

// ============================================================================
// Device Information
// ============================================================================

/// Information about a GPU device
struct DeviceInfo {
    int device_id;              ///< Device index (0, 1, 2, ...)
    std::string name;           ///< Device name (e.g., "AMD Instinct MI100")
    std::string architecture;   ///< GFX architecture (e.g., "gfx908")
    size_t total_memory;        ///< Total VRAM in bytes
    size_t free_memory;         ///< Currently available VRAM in bytes
    int compute_units;          ///< Number of compute units
    int max_threads_per_block;  ///< Maximum threads per block
    int warp_size;              ///< Wavefront size (typically 64)
};

// ============================================================================
// Device Management Functions
// ============================================================================

/// Get information about a GPU device
/// @param device_id Device index (default: 0)
/// @return DeviceInfo struct with device properties
/// @throws std::runtime_error if device_id is invalid
DeviceInfo get_device_info(int device_id = 0);

/// Get the number of available GPU devices
/// @return Number of AMD GPUs, or 0 if none available
int get_device_count();

/// Set the current GPU device for subsequent operations
/// @param device_id Device index to use
/// @throws std::runtime_error if device_id is invalid
void set_device(int device_id);

/// Get the currently active GPU device
/// @return Current device index
int get_device();

/// Synchronize the current device (wait for all operations to complete)
void synchronize();

/// Check if ROCm is available at runtime
/// @return true if at least one AMD GPU is detected
bool is_available();

}  // namespace pyflame::backend::rocm

#endif  // PYFLAME_HAS_ROCM
