// ROCm BLAS Operations Implementation
// Phase 3: Core Operations - BLAS

#ifdef PYFLAME_HAS_ROCM

#include "pyflame/backend/rocm/rocm_blas.hpp"
#include <limits>

namespace pyflame::backend::rocm {

// SECURITY: Helper to safely cast int64_t to rocblas_int with overflow check
namespace {

constexpr int64_t ROCBLAS_INT_MAX = std::numeric_limits<rocblas_int>::max();

inline rocblas_int safe_cast_to_rocblas_int(int64_t value, const char* param_name) {
    if (value > ROCBLAS_INT_MAX || value < std::numeric_limits<rocblas_int>::min()) {
        throw std::overflow_error(
            std::string("BLAS dimension overflow: ") + param_name +
            "=" + std::to_string(value) +
            " exceeds rocblas_int max=" + std::to_string(ROCBLAS_INT_MAX));
    }
    return static_cast<rocblas_int>(value);
}

// Validate all GEMM dimensions at once
inline void validate_gemm_dims(int64_t M, int64_t N, int64_t K,
                               int64_t lda, int64_t ldb, int64_t ldc) {
    safe_cast_to_rocblas_int(M, "M");
    safe_cast_to_rocblas_int(N, "N");
    safe_cast_to_rocblas_int(K, "K");
    safe_cast_to_rocblas_int(lda, "lda");
    safe_cast_to_rocblas_int(ldb, "ldb");
    safe_cast_to_rocblas_int(ldc, "ldc");
}

}  // anonymous namespace

ROCmBLAS::ROCmBLAS() {
    ROCBLAS_CHECK(rocblas_create_handle(&handle_));
}

ROCmBLAS::~ROCmBLAS() {
    if (handle_) {
        rocblas_destroy_handle(handle_);
    }
}

void ROCmBLAS::set_stream(hipStream_t stream) {
    ROCBLAS_CHECK(rocblas_set_stream(handle_, stream));
}

rocblas_datatype ROCmBLAS::to_rocblas_type(DType dtype) const {
    switch (dtype) {
        case DType::Float32:  return rocblas_datatype_f32_r;
        case DType::Float16:  return rocblas_datatype_f16_r;
        case DType::BFloat16: return rocblas_datatype_bf16_r;
        case DType::Int32:    return rocblas_datatype_i32_r;
        case DType::Int8:     return rocblas_datatype_i8_r;
        default:
            throw std::runtime_error("Unsupported dtype for rocBLAS: " +
                                     dtype_name(dtype));
    }
}

rocblas_operation ROCmBLAS::to_rocblas_op(TransposeOp op) const {
    switch (op) {
        case TransposeOp::None:      return rocblas_operation_none;
        case TransposeOp::Transpose: return rocblas_operation_transpose;
        case TransposeOp::ConjTrans: return rocblas_operation_conjugate_transpose;
        default:                     return rocblas_operation_none;
    }
}

void ROCmBLAS::gemm(
    DType dtype,
    TransposeOp trans_a,
    TransposeOp trans_b,
    int64_t M, int64_t N, int64_t K,
    float alpha,
    const void* A, int64_t lda,
    const void* B, int64_t ldb,
    float beta,
    void* C, int64_t ldc
) {
    // SECURITY: Validate dimensions before casting to prevent overflow
    validate_gemm_dims(M, N, K, lda, ldb, ldc);

    rocblas_datatype type = to_rocblas_type(dtype);
    rocblas_operation op_a = to_rocblas_op(trans_a);
    rocblas_operation op_b = to_rocblas_op(trans_b);

    // Use gemm_ex for flexibility with different data types
    // Note: rocBLAS uses column-major ordering, but our interface expects row-major
    // We swap A and B and transpose operations to handle this
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle_,
        op_b, op_a,  // Swapped for row-major
        safe_cast_to_rocblas_int(N, "N"),  // Swapped M and N
        safe_cast_to_rocblas_int(M, "M"),
        safe_cast_to_rocblas_int(K, "K"),
        &alpha,
        B, type, safe_cast_to_rocblas_int(ldb, "ldb"),  // Swapped A and B
        A, type, safe_cast_to_rocblas_int(lda, "lda"),
        &beta,
        C, type, safe_cast_to_rocblas_int(ldc, "ldc"),
        C, type, safe_cast_to_rocblas_int(ldc, "ldc"),
        rocblas_datatype_f32_r,  // compute type always f32 for precision
        rocblas_gemm_algo_standard,
        0,  // solution index
        0   // flags
    ));
}

void ROCmBLAS::gemm_strided_batched(
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
) {
    // SECURITY: Validate dimensions before casting to prevent overflow
    validate_gemm_dims(M, N, K, lda, ldb, ldc);
    safe_cast_to_rocblas_int(batch_count, "batch_count");

    rocblas_datatype type = to_rocblas_type(dtype);
    rocblas_operation op_a = to_rocblas_op(trans_a);
    rocblas_operation op_b = to_rocblas_op(trans_b);

    // Similar row-major to column-major handling as gemm
    ROCBLAS_CHECK(rocblas_gemm_strided_batched_ex(
        handle_,
        op_b, op_a,  // Swapped for row-major
        safe_cast_to_rocblas_int(N, "N"),
        safe_cast_to_rocblas_int(M, "M"),
        safe_cast_to_rocblas_int(K, "K"),
        &alpha,
        B, type, safe_cast_to_rocblas_int(ldb, "ldb"), stride_b,
        A, type, safe_cast_to_rocblas_int(lda, "lda"), stride_a,
        &beta,
        C, type, safe_cast_to_rocblas_int(ldc, "ldc"), stride_c,
        C, type, safe_cast_to_rocblas_int(ldc, "ldc"), stride_c,
        safe_cast_to_rocblas_int(batch_count, "batch_count"),
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard,
        0, 0
    ));
}

void ROCmBLAS::geam(
    DType dtype,
    TransposeOp trans_a,
    TransposeOp trans_b,
    int64_t M, int64_t N,
    float alpha,
    const void* A, int64_t lda,
    float beta,
    const void* B, int64_t ldb,
    void* C, int64_t ldc
) {
    // SECURITY: Validate dimensions before casting
    safe_cast_to_rocblas_int(M, "M");
    safe_cast_to_rocblas_int(N, "N");
    safe_cast_to_rocblas_int(lda, "lda");
    safe_cast_to_rocblas_int(ldb, "ldb");
    safe_cast_to_rocblas_int(ldc, "ldc");

    rocblas_operation op_a = to_rocblas_op(trans_a);
    rocblas_operation op_b = to_rocblas_op(trans_b);

    // geam only supports single and double precision directly
    if (dtype == DType::Float32) {
        ROCBLAS_CHECK(rocblas_sgeam(
            handle_,
            op_a, op_b,
            safe_cast_to_rocblas_int(M, "M"),
            safe_cast_to_rocblas_int(N, "N"),
            &alpha,
            static_cast<const float*>(A), safe_cast_to_rocblas_int(lda, "lda"),
            &beta,
            static_cast<const float*>(B), safe_cast_to_rocblas_int(ldb, "ldb"),
            static_cast<float*>(C), safe_cast_to_rocblas_int(ldc, "ldc")
        ));
    } else {
        throw std::runtime_error("geam only supports Float32 directly. "
                                 "Use custom kernel for: " + dtype_name(dtype));
    }
}

void ROCmBLAS::dot(
    DType dtype,
    int64_t n,
    const void* x, int64_t incx,
    const void* y, int64_t incy,
    void* result
) {
    // SECURITY: Validate dimensions before casting
    safe_cast_to_rocblas_int(n, "n");
    safe_cast_to_rocblas_int(incx, "incx");
    safe_cast_to_rocblas_int(incy, "incy");

    if (dtype == DType::Float32) {
        ROCBLAS_CHECK(rocblas_sdot(
            handle_,
            safe_cast_to_rocblas_int(n, "n"),
            static_cast<const float*>(x), safe_cast_to_rocblas_int(incx, "incx"),
            static_cast<const float*>(y), safe_cast_to_rocblas_int(incy, "incy"),
            static_cast<float*>(result)
        ));
    } else if (dtype == DType::Float16) {
        ROCBLAS_CHECK(rocblas_hdot(
            handle_,
            safe_cast_to_rocblas_int(n, "n"),
            static_cast<const rocblas_half*>(x), safe_cast_to_rocblas_int(incx, "incx"),
            static_cast<const rocblas_half*>(y), safe_cast_to_rocblas_int(incy, "incy"),
            static_cast<rocblas_half*>(result)
        ));
    } else {
        throw std::runtime_error("dot only supports Float32 and Float16. "
                                 "Unsupported dtype: " + dtype_name(dtype));
    }
}

void ROCmBLAS::scal(
    DType dtype,
    int64_t n,
    float alpha,
    void* x, int64_t incx
) {
    // SECURITY: Validate dimensions before casting
    safe_cast_to_rocblas_int(n, "n");
    safe_cast_to_rocblas_int(incx, "incx");

    if (dtype == DType::Float32) {
        ROCBLAS_CHECK(rocblas_sscal(
            handle_,
            safe_cast_to_rocblas_int(n, "n"),
            &alpha,
            static_cast<float*>(x), safe_cast_to_rocblas_int(incx, "incx")
        ));
    } else if (dtype == DType::Float16) {
        rocblas_half alpha_half;
        // Convert float to half
        alpha_half = static_cast<rocblas_half>(alpha);
        ROCBLAS_CHECK(rocblas_hscal(
            handle_,
            safe_cast_to_rocblas_int(n, "n"),
            &alpha_half,
            static_cast<rocblas_half*>(x), safe_cast_to_rocblas_int(incx, "incx")
        ));
    } else {
        throw std::runtime_error("scal only supports Float32 and Float16. "
                                 "Unsupported dtype: " + dtype_name(dtype));
    }
}

void ROCmBLAS::axpy(
    DType dtype,
    int64_t n,
    float alpha,
    const void* x, int64_t incx,
    void* y, int64_t incy
) {
    // SECURITY: Validate dimensions before casting
    safe_cast_to_rocblas_int(n, "n");
    safe_cast_to_rocblas_int(incx, "incx");
    safe_cast_to_rocblas_int(incy, "incy");

    if (dtype == DType::Float32) {
        ROCBLAS_CHECK(rocblas_saxpy(
            handle_,
            safe_cast_to_rocblas_int(n, "n"),
            &alpha,
            static_cast<const float*>(x), safe_cast_to_rocblas_int(incx, "incx"),
            static_cast<float*>(y), safe_cast_to_rocblas_int(incy, "incy")
        ));
    } else if (dtype == DType::Float16) {
        rocblas_half alpha_half;
        alpha_half = static_cast<rocblas_half>(alpha);
        ROCBLAS_CHECK(rocblas_haxpy(
            handle_,
            safe_cast_to_rocblas_int(n, "n"),
            &alpha_half,
            static_cast<const rocblas_half*>(x), safe_cast_to_rocblas_int(incx, "incx"),
            static_cast<rocblas_half*>(y), safe_cast_to_rocblas_int(incy, "incy")
        ));
    } else {
        throw std::runtime_error("axpy only supports Float32 and Float16. "
                                 "Unsupported dtype: " + dtype_name(dtype));
    }
}

// Global singleton
ROCmBLAS& get_blas() {
    static ROCmBLAS instance;
    return instance;
}

}  // namespace pyflame::backend::rocm

#endif  // PYFLAME_HAS_ROCM
