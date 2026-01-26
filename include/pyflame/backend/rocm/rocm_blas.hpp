#pragma once

#ifdef PYFLAME_HAS_ROCM

#include "pyflame/backend/rocm/rocm_backend.hpp"
#include "pyflame/core/dtype.hpp"
#include <memory>

namespace pyflame::backend::rocm {

/// Transpose operation type for BLAS operations
enum class TransposeOp {
    None,       ///< No transpose (op(A) = A)
    Transpose,  ///< Transpose (op(A) = A^T)
    ConjTrans   ///< Conjugate transpose (op(A) = A^H, for complex types)
};

/// rocBLAS wrapper class
///
/// Provides high-level interface to rocBLAS operations with:
/// - Automatic data type handling (FP32, FP16, BF16)
/// - Stream support for asynchronous execution
/// - Error checking with descriptive messages
///
/// Example:
/// @code
/// ROCmBLAS blas;
/// blas.set_stream(my_stream);
/// blas.gemm(DType::Float32, TransposeOp::None, TransposeOp::None,
///           M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
/// @endcode
class ROCmBLAS {
public:
    ROCmBLAS();
    ~ROCmBLAS();

    // Disable copy
    ROCmBLAS(const ROCmBLAS&) = delete;
    ROCmBLAS& operator=(const ROCmBLAS&) = delete;

    /// Set the HIP stream for operations
    /// @param stream HIP stream to use for BLAS operations
    void set_stream(hipStream_t stream);

    /// Get the rocBLAS handle
    /// @return The underlying rocBLAS handle
    rocblas_handle handle() const { return handle_; }

    // ========================================================================
    // GEMM Operations (General Matrix Multiply)
    // ========================================================================

    /// General matrix multiply: C = alpha * op(A) * op(B) + beta * C
    ///
    /// Computes the matrix product of op(A) and op(B), scaled by alpha,
    /// and adds the result to C scaled by beta.
    ///
    /// @param dtype Data type of matrices (Float32, Float16, BFloat16)
    /// @param trans_a Transpose operation for matrix A
    /// @param trans_b Transpose operation for matrix B
    /// @param M Number of rows in op(A) and C
    /// @param N Number of columns in op(B) and C
    /// @param K Number of columns in op(A) / rows in op(B)
    /// @param alpha Scalar multiplier for A*B
    /// @param A Pointer to matrix A (device memory)
    /// @param lda Leading dimension of A
    /// @param B Pointer to matrix B (device memory)
    /// @param ldb Leading dimension of B
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to matrix C (device memory, input/output)
    /// @param ldc Leading dimension of C
    void gemm(
        DType dtype,
        TransposeOp trans_a,
        TransposeOp trans_b,
        int64_t M, int64_t N, int64_t K,
        float alpha,
        const void* A, int64_t lda,
        const void* B, int64_t ldb,
        float beta,
        void* C, int64_t ldc
    );

    /// Strided batched GEMM: C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
    ///
    /// Performs GEMM on multiple matrices stored with fixed strides.
    /// Efficient for batched operations like multi-head attention.
    ///
    /// @param dtype Data type of matrices
    /// @param trans_a Transpose operation for A matrices
    /// @param trans_b Transpose operation for B matrices
    /// @param M Number of rows in op(A[i]) and C[i]
    /// @param N Number of columns in op(B[i]) and C[i]
    /// @param K Number of columns in op(A[i]) / rows in op(B[i])
    /// @param alpha Scalar multiplier for A*B
    /// @param A Pointer to first matrix A (device memory)
    /// @param lda Leading dimension of A matrices
    /// @param stride_a Stride between consecutive A matrices (in elements)
    /// @param B Pointer to first matrix B (device memory)
    /// @param ldb Leading dimension of B matrices
    /// @param stride_b Stride between consecutive B matrices (in elements)
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to first matrix C (device memory)
    /// @param ldc Leading dimension of C matrices
    /// @param stride_c Stride between consecutive C matrices (in elements)
    /// @param batch_count Number of matrices in the batch
    void gemm_strided_batched(
        DType dtype,
        TransposeOp trans_a,
        TransposeOp trans_b,
        int64_t M, int64_t N, int64_t K,
        float alpha,
        const void* A, int64_t lda, int64_t stride_a,
        const void* B, int64_t ldb, int64_t stride_b,
        float beta,
        void* C, int64_t ldc, int64_t stride_c,
        int64_t batch_count
    );

    // ========================================================================
    // Matrix Operations
    // ========================================================================

    /// Matrix transpose/scale: C = alpha * op(A) + beta * op(B)
    ///
    /// Performs matrix addition with optional transposition.
    /// Commonly used for matrix transpose (set B to zeros, alpha=1).
    ///
    /// @param dtype Data type of matrices (currently only Float32 supported)
    /// @param trans_a Transpose operation for matrix A
    /// @param trans_b Transpose operation for matrix B
    /// @param M Number of rows in C
    /// @param N Number of columns in C
    /// @param alpha Scalar multiplier for op(A)
    /// @param A Pointer to matrix A (device memory)
    /// @param lda Leading dimension of A
    /// @param beta Scalar multiplier for op(B)
    /// @param B Pointer to matrix B (device memory, can be nullptr if beta=0)
    /// @param ldb Leading dimension of B
    /// @param C Pointer to matrix C (device memory, output)
    /// @param ldc Leading dimension of C
    void geam(
        DType dtype,
        TransposeOp trans_a,
        TransposeOp trans_b,
        int64_t M, int64_t N,
        float alpha,
        const void* A, int64_t lda,
        float beta,
        const void* B, int64_t ldb,
        void* C, int64_t ldc
    );

    // ========================================================================
    // Vector Operations (BLAS Level 1)
    // ========================================================================

    /// Vector dot product: result = sum(x[i] * y[i])
    /// @param dtype Data type of vectors
    /// @param n Number of elements
    /// @param x Pointer to first vector (device memory)
    /// @param incx Stride between elements of x
    /// @param y Pointer to second vector (device memory)
    /// @param incy Stride between elements of y
    /// @param result Pointer to result (device memory)
    void dot(
        DType dtype,
        int64_t n,
        const void* x, int64_t incx,
        const void* y, int64_t incy,
        void* result
    );

    /// Vector scaling: x = alpha * x
    /// @param dtype Data type of vector
    /// @param n Number of elements
    /// @param alpha Scalar multiplier
    /// @param x Pointer to vector (device memory, in/out)
    /// @param incx Stride between elements
    void scal(
        DType dtype,
        int64_t n,
        float alpha,
        void* x, int64_t incx
    );

    /// Vector addition: y = alpha * x + y (AXPY)
    /// @param dtype Data type of vectors
    /// @param n Number of elements
    /// @param alpha Scalar multiplier for x
    /// @param x Pointer to source vector (device memory)
    /// @param incx Stride between elements of x
    /// @param y Pointer to destination vector (device memory, in/out)
    /// @param incy Stride between elements of y
    void axpy(
        DType dtype,
        int64_t n,
        float alpha,
        const void* x, int64_t incx,
        void* y, int64_t incy
    );

private:
    rocblas_handle handle_;

    /// Convert PyFlame dtype to rocBLAS datatype
    rocblas_datatype to_rocblas_type(DType dtype) const;

    /// Convert transpose op to rocBLAS operation
    rocblas_operation to_rocblas_op(TransposeOp op) const;
};

/// Get the global BLAS instance (thread-safe singleton)
/// @return Reference to the global ROCmBLAS instance
ROCmBLAS& get_blas();

}  // namespace pyflame::backend::rocm

#endif  // PYFLAME_HAS_ROCM
