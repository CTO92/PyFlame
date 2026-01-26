/// @file test_rocm_device.cpp
/// @brief Tests for ROCm device management

#include <gtest/gtest.h>

#ifdef PYFLAME_HAS_ROCM

#include "pyflame/backend/rocm/rocm_backend.hpp"

using namespace pyflame::backend::rocm;

class ROCmDeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!is_available()) {
            GTEST_SKIP() << "ROCm not available";
        }
    }
};

TEST_F(ROCmDeviceTest, IsAvailable) {
    EXPECT_TRUE(is_available());
}

TEST_F(ROCmDeviceTest, DeviceCount) {
    int count = get_device_count();
    EXPECT_GT(count, 0) << "Expected at least one ROCm device";
}

TEST_F(ROCmDeviceTest, GetDeviceInfo) {
    auto info = get_device_info(0);

    // Name should not be empty
    EXPECT_FALSE(info.name.empty()) << "Device name should not be empty";

    // Total memory should be positive
    EXPECT_GT(info.total_memory, 0u) << "Total memory should be positive";

    // Compute capability should be reasonable
    EXPECT_GT(info.compute_units, 0) << "Should have at least 1 compute unit";

    // Free memory should not exceed total memory
    EXPECT_LE(info.free_memory, info.total_memory)
        << "Free memory should not exceed total memory";
}

TEST_F(ROCmDeviceTest, GetDeviceInfo_InvalidDevice) {
    int count = get_device_count();

    // Accessing invalid device should throw
    EXPECT_THROW(get_device_info(count), std::runtime_error);
    EXPECT_THROW(get_device_info(-1), std::runtime_error);
}

TEST_F(ROCmDeviceTest, SetDevice) {
    int count = get_device_count();

    // Should be able to set device to any valid index
    for (int i = 0; i < count; ++i) {
        EXPECT_NO_THROW(set_device(i));
    }
}

TEST_F(ROCmDeviceTest, SetDevice_InvalidDevice) {
    int count = get_device_count();

    EXPECT_THROW(set_device(count), std::runtime_error);
    EXPECT_THROW(set_device(-1), std::runtime_error);
}

TEST_F(ROCmDeviceTest, GetCurrentDevice) {
    set_device(0);
    EXPECT_EQ(get_current_device(), 0);

    if (get_device_count() > 1) {
        set_device(1);
        EXPECT_EQ(get_current_device(), 1);
    }
}

TEST_F(ROCmDeviceTest, Synchronize) {
    // Synchronize should not throw
    EXPECT_NO_THROW(synchronize());
}

TEST_F(ROCmDeviceTest, AllDevicesHaveValidInfo) {
    int count = get_device_count();

    for (int i = 0; i < count; ++i) {
        auto info = get_device_info(i);

        // Each device should have valid properties
        EXPECT_FALSE(info.name.empty())
            << "Device " << i << " should have a name";
        EXPECT_GT(info.total_memory, 0u)
            << "Device " << i << " should have memory";
        EXPECT_GT(info.compute_units, 0)
            << "Device " << i << " should have compute units";
    }
}

#endif  // PYFLAME_HAS_ROCM
