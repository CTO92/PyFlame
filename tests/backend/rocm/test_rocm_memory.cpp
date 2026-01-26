/// @file test_rocm_memory.cpp
/// @brief Tests for ROCm memory management

#include <gtest/gtest.h>

#ifdef PYFLAME_HAS_ROCM

#include "pyflame/backend/rocm/rocm_memory.hpp"
#include "pyflame/backend/rocm/rocm_backend.hpp"
#include <vector>
#include <cstring>

using namespace pyflame::backend::rocm;

class ROCmMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!is_available()) {
            GTEST_SKIP() << "ROCm not available";
        }
        set_device(0);
        memory_manager_ = std::make_unique<ROCmMemoryManager>();
    }

    void TearDown() override {
        memory_manager_.reset();
    }

    std::unique_ptr<ROCmMemoryManager> memory_manager_;
};

TEST_F(ROCmMemoryTest, BasicAllocation) {
    size_t size = 1024 * sizeof(float);
    void* ptr = memory_manager_->allocate(size);

    EXPECT_NE(ptr, nullptr) << "Allocation should succeed";

    memory_manager_->deallocate(ptr);
}

TEST_F(ROCmMemoryTest, MultipleAllocations) {
    std::vector<void*> ptrs;
    size_t size = 1024 * sizeof(float);

    for (int i = 0; i < 100; ++i) {
        void* ptr = memory_manager_->allocate(size);
        EXPECT_NE(ptr, nullptr) << "Allocation " << i << " should succeed";
        ptrs.push_back(ptr);
    }

    for (auto ptr : ptrs) {
        memory_manager_->deallocate(ptr);
    }
}

TEST_F(ROCmMemoryTest, LargeAllocation) {
    // 1GB allocation
    size_t size = 1024ULL * 1024 * 1024;

    auto info = get_device_info(0);
    if (info.free_memory < size) {
        GTEST_SKIP() << "Not enough free memory for 1GB allocation test";
    }

    void* ptr = memory_manager_->allocate(size);
    EXPECT_NE(ptr, nullptr) << "1GB allocation should succeed";

    memory_manager_->deallocate(ptr);
}

TEST_F(ROCmMemoryTest, ZeroAllocation) {
    // Zero-size allocation should not crash
    void* ptr = memory_manager_->allocate(0);
    // Behavior is implementation-defined, but should not crash
    if (ptr != nullptr) {
        memory_manager_->deallocate(ptr);
    }
}

TEST_F(ROCmMemoryTest, CopyHostToDevice) {
    const size_t count = 1024;
    const size_t size = count * sizeof(float);

    // Create host data
    std::vector<float> host_data(count);
    for (size_t i = 0; i < count; ++i) {
        host_data[i] = static_cast<float>(i);
    }

    // Allocate device memory
    void* device_ptr = memory_manager_->allocate(size);
    ASSERT_NE(device_ptr, nullptr);

    // Copy to device
    memory_manager_->copy_host_to_device(device_ptr, host_data.data(), size);

    // Allocate second buffer and copy back
    std::vector<float> result(count);
    memory_manager_->copy_device_to_host(result.data(), device_ptr, size);

    // Verify
    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(result[i], host_data[i]) << "Mismatch at index " << i;
    }

    memory_manager_->deallocate(device_ptr);
}

TEST_F(ROCmMemoryTest, CopyDeviceToDevice) {
    const size_t count = 1024;
    const size_t size = count * sizeof(float);

    // Create host data
    std::vector<float> host_data(count);
    for (size_t i = 0; i < count; ++i) {
        host_data[i] = static_cast<float>(i * 2.5f);
    }

    // Allocate two device buffers
    void* src = memory_manager_->allocate(size);
    void* dst = memory_manager_->allocate(size);
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);

    // Copy host -> src
    memory_manager_->copy_host_to_device(src, host_data.data(), size);

    // Copy src -> dst on device
    memory_manager_->copy_device_to_device(dst, src, size);

    // Copy dst -> host to verify
    std::vector<float> result(count);
    memory_manager_->copy_device_to_host(result.data(), dst, size);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(result[i], host_data[i]) << "Mismatch at index " << i;
    }

    memory_manager_->deallocate(src);
    memory_manager_->deallocate(dst);
}

TEST_F(ROCmMemoryTest, MemsetZero) {
    const size_t count = 1024;
    const size_t size = count * sizeof(float);

    void* ptr = memory_manager_->allocate(size);
    ASSERT_NE(ptr, nullptr);

    // Memset to zero
    memory_manager_->memset(ptr, 0, size);

    // Copy back and verify
    std::vector<float> result(count);
    memory_manager_->copy_device_to_host(result.data(), ptr, size);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(result[i], 0.0f) << "Element " << i << " should be zero";
    }

    memory_manager_->deallocate(ptr);
}

TEST_F(ROCmMemoryTest, MemoryStats) {
    auto stats_before = memory_manager_->get_stats();

    // Allocate some memory
    const size_t size = 1024 * 1024;  // 1MB
    void* ptr = memory_manager_->allocate(size);
    ASSERT_NE(ptr, nullptr);

    auto stats_after = memory_manager_->get_stats();

    // Allocated bytes should increase
    EXPECT_GE(stats_after.current_allocated, stats_before.current_allocated + size);

    memory_manager_->deallocate(ptr);
}

TEST_F(ROCmMemoryTest, AllocationAlignment) {
    // Allocations should be aligned for efficient GPU access
    // Typically 256-byte alignment is required for AMD GPUs

    void* ptr = memory_manager_->allocate(100);
    ASSERT_NE(ptr, nullptr);

    // Check alignment (at least 256 bytes for GPU)
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % 256, 0u) << "Allocation should be 256-byte aligned";

    memory_manager_->deallocate(ptr);
}

TEST_F(ROCmMemoryTest, RapidAllocDealloc) {
    // Stress test with rapid allocation/deallocation
    const size_t iterations = 1000;
    const size_t size = 4096;

    for (size_t i = 0; i < iterations; ++i) {
        void* ptr = memory_manager_->allocate(size);
        ASSERT_NE(ptr, nullptr) << "Allocation failed at iteration " << i;
        memory_manager_->deallocate(ptr);
    }
}

#endif  // PYFLAME_HAS_ROCM
