// Matrix multiplication CPU reference implementation

#include "pyflame/core/tensor.hpp"
#include <vector>
#include <algorithm>

namespace pyflame {
namespace ops {

// Basic matrix multiplication: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
void matmul_cpu(
    const float* A,
    const float* B,
    float* C,
    int64_t M,
    int64_t K,
    int64_t N
) {
    // Simple O(M*N*K) implementation
    // Could be optimized with blocking, SIMD, etc.
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Blocked matrix multiplication for better cache utilization
void matmul_blocked_cpu(
    const float* A,
    const float* B,
    float* C,
    int64_t M,
    int64_t K,
    int64_t N,
    int64_t block_size = 32
) {
    // Initialize C to zero
    for (int64_t i = 0; i < M * N; ++i) {
        C[i] = 0.0f;
    }

    // Blocked multiplication
    for (int64_t ii = 0; ii < M; ii += block_size) {
        for (int64_t jj = 0; jj < N; jj += block_size) {
            for (int64_t kk = 0; kk < K; kk += block_size) {
                // Block boundaries
                int64_t i_end = std::min(ii + block_size, M);
                int64_t j_end = std::min(jj + block_size, N);
                int64_t k_end = std::min(kk + block_size, K);

                // Multiply within block
                for (int64_t i = ii; i < i_end; ++i) {
                    for (int64_t j = jj; j < j_end; ++j) {
                        float sum = C[i * N + j];
                        for (int64_t k = kk; k < k_end; ++k) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

// Matrix-vector multiplication: y = A @ x
// A: [M, K], x: [K], y: [M]
void matvec_cpu(
    const float* A,
    const float* x,
    float* y,
    int64_t M,
    int64_t K
) {
    for (int64_t i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            sum += A[i * K + k] * x[k];
        }
        y[i] = sum;
    }
}

// Vector-matrix multiplication: y = x @ A
// x: [K], A: [K, N], y: [N]
void vecmat_cpu(
    const float* x,
    const float* A,
    float* y,
    int64_t K,
    int64_t N
) {
    for (int64_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            sum += x[k] * A[k * N + j];
        }
        y[j] = sum;
    }
}

// Dot product: result = a . b
// a: [K], b: [K], result: scalar
float dot_cpu(
    const float* a,
    const float* b,
    int64_t K
) {
    float result = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        result += a[k] * b[k];
    }
    return result;
}

// Batched matrix multiplication: C[b] = A[b] @ B[b]
// A: [batch, M, K], B: [batch, K, N], C: [batch, M, N]
void batch_matmul_cpu(
    const float* A,
    const float* B,
    float* C,
    int64_t batch,
    int64_t M,
    int64_t K,
    int64_t N
) {
    int64_t A_batch_stride = M * K;
    int64_t B_batch_stride = K * N;
    int64_t C_batch_stride = M * N;

    for (int64_t b = 0; b < batch; ++b) {
        matmul_cpu(
            A + b * A_batch_stride,
            B + b * B_batch_stride,
            C + b * C_batch_stride,
            M, K, N
        );
    }
}

// Transposed matrix multiplication variants

// C = A.T @ B (A is transposed)
// A: [K, M] (stored), A.T: [M, K], B: [K, N], C: [M, N]
void matmul_at_b_cpu(
    const float* A,  // [K, M] in memory
    const float* B,  // [K, N]
    float* C,        // [M, N]
    int64_t M,
    int64_t K,
    int64_t N
) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                // A.T[i, k] = A[k, i]
                sum += A[k * M + i] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// C = A @ B.T (B is transposed)
// A: [M, K], B: [N, K] (stored), B.T: [K, N], C: [M, N]
void matmul_a_bt_cpu(
    const float* A,  // [M, K]
    const float* B,  // [N, K] in memory
    float* C,        // [M, N]
    int64_t M,
    int64_t K,
    int64_t N
) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                // B.T[k, j] = B[j, k]
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

// Matrix transpose
void transpose_2d_cpu(
    const float* A,
    float* B,
    int64_t M,
    int64_t N
) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            B[j * M + i] = A[i * N + j];
        }
    }
}

// In-place transpose for square matrices
void transpose_inplace_cpu(
    float* A,
    int64_t N
) {
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = i + 1; j < N; ++j) {
            std::swap(A[i * N + j], A[j * N + i]);
        }
    }
}

}  // namespace ops
}  // namespace pyflame
