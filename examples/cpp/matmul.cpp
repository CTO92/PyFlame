/**
 * PyFlame Matrix Multiplication Example
 *
 * Demonstrates matrix multiplication with different layouts.
 */

#include <iostream>
#include <chrono>
#include "pyflame/pyflame.hpp"

using namespace pyflame;

int main() {
    std::cout << "PyFlame Matrix Multiplication Example\n";
    std::cout << "======================================\n\n";

    // Small example with known values
    std::cout << "1. Small matrix multiply:\n";
    {
        float a_data[] = {1, 2, 3, 4, 5, 6};  // 2x3
        float b_data[] = {7, 8, 9, 10, 11, 12};  // 3x2

        auto A = Tensor::from_data(a_data, {2, 3});
        auto B = Tensor::from_data(b_data, {3, 2});
        auto C = matmul(A, B);

        std::cout << "   A shape: [" << A.shape()[0] << ", " << A.shape()[1] << "]\n";
        std::cout << "   B shape: [" << B.shape()[0] << ", " << B.shape()[1] << "]\n";
        std::cout << "   C shape: [" << C.shape()[0] << ", " << C.shape()[1] << "]\n";

        C.eval();
        const float* c_data = C.data<float>();
        std::cout << "   C = [[" << c_data[0] << ", " << c_data[1] << "],\n";
        std::cout << "        [" << c_data[2] << ", " << c_data[3] << "]]\n";
    }

    // Larger benchmark
    std::cout << "\n2. Performance benchmark:\n";
    {
        const int M = 256, K = 256, N = 256;

        auto A = Tensor::randn({M, K});
        auto B = Tensor::randn({K, N});

        auto start = std::chrono::high_resolution_clock::now();

        auto C = matmul(A, B);
        C.eval();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "   Matrix size: " << M << "x" << K << " @ " << K << "x" << N << "\n";
        std::cout << "   Time: " << duration.count() / 1000.0 << " ms\n";

        double flops = 2.0 * M * K * N;
        double gflops = flops / (duration.count() * 1000.0);
        std::cout << "   Estimated: " << gflops << " GFLOP/s (CPU reference)\n";
    }

    // With different layouts (for future WSE execution)
    std::cout << "\n3. Layouts for distributed execution:\n";
    {
        // SinglePE layout (default)
        auto layout_single = MeshLayout::SinglePE();
        std::cout << "   SinglePE: " << layout_single << "\n";

        // Grid layout for larger matrices
        auto layout_grid = MeshLayout::Grid(4, 4);
        std::cout << "   Grid(4,4): " << layout_grid << "\n";
        std::cout << "     Total PEs: " << layout_grid.total_pes() << "\n";

        // Memory per PE calculation
        std::vector<int64_t> large_shape = {1024, 1024};
        size_t mem_single = layout_single.memory_per_pe(large_shape, sizeof(float));
        size_t mem_grid = layout_grid.memory_per_pe(large_shape, sizeof(float));

        std::cout << "\n   For 1024x1024 float32 tensor:\n";
        std::cout << "     SinglePE memory: " << mem_single / 1024 << " KB\n";
        std::cout << "     Grid(4,4) memory per PE: " << mem_grid / 1024 << " KB\n";
    }

    // Chained operations
    std::cout << "\n4. Fused operations:\n";
    {
        auto A = Tensor::randn({128, 64});
        auto B = Tensor::randn({64, 32});
        auto bias = Tensor::randn({32});

        // Linear layer: relu(A @ B + bias)
        auto out = relu(matmul(A, B) + bias);

        std::cout << "   Input: [128, 64]\n";
        std::cout << "   Weight: [64, 32]\n";
        std::cout << "   Output: [" << out.shape()[0] << ", " << out.shape()[1] << "]\n";

        out.eval();
        std::cout << "   First output value: " << out.data<float>()[0] << "\n";
    }

    std::cout << "\nDone!\n";
    return 0;
}
