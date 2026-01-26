#pragma once

#ifdef PYFLAME_HAS_ROCM

#include "pyflame/backend/rocm/rocm_backend.hpp"
#include "pyflame/core/dtype.hpp"
#include "pyflame/ir/op_type.hpp"
#include <vector>

namespace pyflame::backend::rocm {

// ============================================================================
// Parameter Structures
// ============================================================================

/// Convolution parameters
struct ConvParams {
    std::vector<int> padding;     ///< Padding for each dimension [H, W]
    std::vector<int> stride;      ///< Stride for each dimension [H, W]
    std::vector<int> dilation;    ///< Dilation for each dimension [H, W]
    int groups = 1;               ///< Number of groups (for grouped/depthwise conv)

    /// Create default Conv2D params
    static ConvParams Conv2D(int pad_h = 0, int pad_w = 0,
                              int stride_h = 1, int stride_w = 1,
                              int dilation_h = 1, int dilation_w = 1,
                              int groups = 1) {
        ConvParams p;
        p.padding = {pad_h, pad_w};
        p.stride = {stride_h, stride_w};
        p.dilation = {dilation_h, dilation_w};
        p.groups = groups;
        return p;
    }
};

/// Pooling parameters
struct PoolParams {
    std::vector<int> kernel_size;  ///< Kernel size for each dimension [H, W]
    std::vector<int> padding;      ///< Padding for each dimension [H, W]
    std::vector<int> stride;       ///< Stride for each dimension [H, W]
    bool ceil_mode = false;        ///< Use ceiling for output size calculation

    /// Create default Pool2D params
    static PoolParams Pool2D(int kernel_h, int kernel_w,
                              int pad_h = 0, int pad_w = 0,
                              int stride_h = 1, int stride_w = 1,
                              bool ceil_mode = false) {
        PoolParams p;
        p.kernel_size = {kernel_h, kernel_w};
        p.padding = {pad_h, pad_w};
        p.stride = {stride_h, stride_w};
        p.ceil_mode = ceil_mode;
        return p;
    }
};

/// Batch normalization parameters
struct BatchNormParams {
    double epsilon = 1e-5;         ///< Epsilon for numerical stability
    double momentum = 0.1;         ///< Momentum for running stats
    bool training = true;          ///< Training or inference mode
};

// ============================================================================
// MIOpen Wrapper Class
// ============================================================================

/// MIOpen wrapper class for DNN operations
///
/// Provides high-level interface to MIOpen operations with:
/// - Convolution (2D)
/// - Pooling (Max, Avg)
/// - Batch Normalization
/// - Activations (ReLU, Sigmoid, Tanh)
/// - Softmax
/// - Tensor operations and reductions
///
/// Example:
/// @code
/// ROCmDNN dnn;
/// dnn.set_stream(my_stream);
/// dnn.conv2d_forward(dtype, input_shape, input, weight_shape, weight,
///                    output_shape, output, conv_params, workspace, ws_size);
/// @endcode
class ROCmDNN {
public:
    ROCmDNN();
    ~ROCmDNN();

    // Disable copy
    ROCmDNN(const ROCmDNN&) = delete;
    ROCmDNN& operator=(const ROCmDNN&) = delete;

    /// Set the HIP stream for operations
    /// @param stream HIP stream to use for DNN operations
    void set_stream(hipStream_t stream);

    /// Get the MIOpen handle
    /// @return The underlying MIOpen handle
    miopenHandle_t handle() const { return handle_; }

    // ========================================================================
    // Convolution Operations
    // ========================================================================

    /// 2D Convolution forward
    ///
    /// @param dtype Data type of tensors
    /// @param input_shape Input tensor shape [N, C, H, W]
    /// @param input Input tensor data (device memory)
    /// @param weight_shape Weight tensor shape [O, I/groups, kH, kW]
    /// @param weight Weight tensor data (device memory)
    /// @param output_shape Output tensor shape [N, O, H', W']
    /// @param output Output tensor data (device memory)
    /// @param params Convolution parameters
    /// @param workspace Workspace buffer (device memory)
    /// @param workspace_size Size of workspace in bytes
    void conv2d_forward(
        DType dtype,
        const std::vector<int64_t>& input_shape,
        const void* input,
        const std::vector<int64_t>& weight_shape,
        const void* weight,
        const std::vector<int64_t>& output_shape,
        void* output,
        const ConvParams& params,
        void* workspace,
        size_t workspace_size
    );

    /// Get workspace size for conv2d forward
    ///
    /// @param dtype Data type of tensors
    /// @param input_shape Input tensor shape
    /// @param weight_shape Weight tensor shape
    /// @param output_shape Output tensor shape
    /// @param params Convolution parameters
    /// @return Required workspace size in bytes
    size_t conv2d_forward_workspace_size(
        DType dtype,
        const std::vector<int64_t>& input_shape,
        const std::vector<int64_t>& weight_shape,
        const std::vector<int64_t>& output_shape,
        const ConvParams& params
    );

    // ========================================================================
    // Pooling Operations
    // ========================================================================

    /// 2D Max/Avg Pooling forward
    ///
    /// @param pool_type Pooling type (MAX_POOL2D or AVG_POOL2D)
    /// @param dtype Data type of tensors
    /// @param input_shape Input tensor shape [N, C, H, W]
    /// @param input Input tensor data (device memory)
    /// @param output_shape Output tensor shape [N, C, H', W']
    /// @param output Output tensor data (device memory)
    /// @param params Pooling parameters
    void pooling2d_forward(
        ir::OpType pool_type,
        DType dtype,
        const std::vector<int64_t>& input_shape,
        const void* input,
        const std::vector<int64_t>& output_shape,
        void* output,
        const PoolParams& params
    );

    // ========================================================================
    // Normalization Operations
    // ========================================================================

    /// Batch normalization forward (training mode)
    ///
    /// @param dtype Data type of tensors
    /// @param input_shape Input tensor shape [N, C, H, W]
    /// @param input Input tensor data (device memory)
    /// @param output Output tensor data (device memory)
    /// @param scale Gamma parameter [C] (device memory)
    /// @param bias Beta parameter [C] (device memory)
    /// @param running_mean Running mean [C] (device memory, updated in place)
    /// @param running_var Running variance [C] (device memory, updated in place)
    /// @param saved_mean Saved mean for backward [C] (device memory)
    /// @param saved_inv_var Saved inverse variance for backward [C] (device memory)
    /// @param params Batch normalization parameters
    void batchnorm_forward_training(
        DType dtype,
        const std::vector<int64_t>& input_shape,
        const void* input,
        void* output,
        const void* scale,
        const void* bias,
        void* running_mean,
        void* running_var,
        void* saved_mean,
        void* saved_inv_var,
        const BatchNormParams& params
    );

    /// Batch normalization forward (inference mode)
    ///
    /// @param dtype Data type of tensors
    /// @param input_shape Input tensor shape [N, C, H, W]
    /// @param input Input tensor data (device memory)
    /// @param output Output tensor data (device memory)
    /// @param scale Gamma parameter [C] (device memory)
    /// @param bias Beta parameter [C] (device memory)
    /// @param running_mean Running mean [C] (device memory)
    /// @param running_var Running variance [C] (device memory)
    /// @param params Batch normalization parameters
    void batchnorm_forward_inference(
        DType dtype,
        const std::vector<int64_t>& input_shape,
        const void* input,
        void* output,
        const void* scale,
        const void* bias,
        const void* running_mean,
        const void* running_var,
        const BatchNormParams& params
    );

    // ========================================================================
    // Activation Operations
    // ========================================================================

    /// Activation forward (ReLU, Sigmoid, Tanh)
    ///
    /// Note: For GELU and SiLU, use custom kernels instead.
    ///
    /// @param activation_type Activation type (RELU, SIGMOID, TANH)
    /// @param dtype Data type of tensors
    /// @param shape Tensor shape
    /// @param input Input tensor data (device memory)
    /// @param output Output tensor data (device memory)
    /// @param alpha Alpha parameter (for leaky ReLU, etc.)
    void activation_forward(
        ir::OpType activation_type,
        DType dtype,
        const std::vector<int64_t>& shape,
        const void* input,
        void* output,
        float alpha = 0.0f
    );

    // ========================================================================
    // Softmax Operations
    // ========================================================================

    /// Softmax forward
    ///
    /// @param dtype Data type of tensors
    /// @param shape Tensor shape
    /// @param input Input tensor data (device memory)
    /// @param output Output tensor data (device memory)
    /// @param axis Axis along which to compute softmax (-1 for last)
    /// @param log_softmax If true, compute log(softmax(x))
    void softmax_forward(
        DType dtype,
        const std::vector<int64_t>& shape,
        const void* input,
        void* output,
        int axis = -1,
        bool log_softmax = false
    );

    // ========================================================================
    // Tensor Operations
    // ========================================================================

    /// Element-wise tensor operation
    ///
    /// Computes C = alpha1 * A op alpha2 * B + beta * C
    ///
    /// @param op_type Operation type (ADD, MUL, MIN, MAX)
    /// @param dtype Data type of tensors
    /// @param shape Tensor shape
    /// @param A First input tensor (device memory)
    /// @param B Second input tensor (device memory)
    /// @param C Output tensor (device memory)
    /// @param alpha1 Scalar for A
    /// @param alpha2 Scalar for B
    /// @param beta Scalar for C
    void tensor_op(
        ir::OpType op_type,
        DType dtype,
        const std::vector<int64_t>& shape,
        const void* A,
        const void* B,
        void* C,
        float alpha1 = 1.0f,
        float alpha2 = 1.0f,
        float beta = 0.0f
    );

    // ========================================================================
    // Reduction Operations
    // ========================================================================

    /// Tensor reduction (SUM, MEAN, MAX, MIN)
    ///
    /// @param reduce_type Reduction type (SUM, MEAN, MAX, MIN)
    /// @param dtype Data type of tensors
    /// @param input_shape Input tensor shape
    /// @param input Input tensor data (device memory)
    /// @param output_shape Output tensor shape
    /// @param output Output tensor data (device memory)
    /// @param reduce_dims Dimensions to reduce
    /// @param workspace Workspace buffer (device memory)
    /// @param workspace_size Size of workspace in bytes
    void reduce(
        ir::OpType reduce_type,
        DType dtype,
        const std::vector<int64_t>& input_shape,
        const void* input,
        const std::vector<int64_t>& output_shape,
        void* output,
        const std::vector<int>& reduce_dims,
        void* workspace,
        size_t workspace_size
    );

    /// Get workspace size for reduction
    ///
    /// @param reduce_type Reduction type
    /// @param dtype Data type of tensors
    /// @param input_shape Input tensor shape
    /// @param output_shape Output tensor shape
    /// @return Required workspace size in bytes
    size_t reduce_workspace_size(
        ir::OpType reduce_type,
        DType dtype,
        const std::vector<int64_t>& input_shape,
        const std::vector<int64_t>& output_shape
    );

private:
    miopenHandle_t handle_;

    /// Convert PyFlame dtype to MIOpen datatype
    miopenDataType_t to_miopen_type(DType dtype) const;

    /// Convert activation type to MIOpen activation mode
    miopenActivationMode_t to_miopen_activation(ir::OpType op) const;

    /// Convert pooling type to MIOpen pooling mode
    miopenPoolingMode_t to_miopen_pooling(ir::OpType op) const;

    /// Convert reduction type to MIOpen reduce op
    miopenReduceTensorOp_t to_miopen_reduce(ir::OpType op) const;

    /// Create tensor descriptor from shape
    miopenTensorDescriptor_t create_tensor_desc(
        DType dtype,
        const std::vector<int64_t>& shape
    ) const;
};

/// Get the global DNN instance (thread-safe singleton)
/// @return Reference to the global ROCmDNN instance
ROCmDNN& get_dnn();

}  // namespace pyflame::backend::rocm

#endif  // PYFLAME_HAS_ROCM
