// ROCm DNN Operations Implementation (MIOpen)
// Phase 4: Core Operations - DNN

#ifdef PYFLAME_HAS_ROCM

#include "pyflame/backend/rocm/rocm_dnn.hpp"

namespace pyflame::backend::rocm {

ROCmDNN::ROCmDNN() {
    MIOPEN_CHECK(miopenCreate(&handle_));
}

ROCmDNN::~ROCmDNN() {
    if (handle_) {
        miopenDestroy(handle_);
    }
}

void ROCmDNN::set_stream(hipStream_t stream) {
    MIOPEN_CHECK(miopenSetStream(handle_, stream));
}

miopenDataType_t ROCmDNN::to_miopen_type(DType dtype) const {
    switch (dtype) {
        case DType::Float32:  return miopenFloat;
        case DType::Float16:  return miopenHalf;
        case DType::BFloat16: return miopenBFloat16;
        case DType::Int32:    return miopenInt32;
        case DType::Int8:     return miopenInt8;
        default:
            throw std::runtime_error("Unsupported dtype for MIOpen: " +
                                     dtype_name(dtype));
    }
}

miopenTensorDescriptor_t ROCmDNN::create_tensor_desc(
    DType dtype,
    const std::vector<int64_t>& shape
) const {
    miopenTensorDescriptor_t desc;
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&desc));

    std::vector<int> dims(shape.begin(), shape.end());
    std::vector<int> strides(shape.size());

    // Calculate strides (row-major / NCHW)
    int stride = 1;
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= dims[i];
    }

    MIOPEN_CHECK(miopenSetTensorDescriptor(
        desc,
        to_miopen_type(dtype),
        static_cast<int>(dims.size()),
        dims.data(),
        strides.data()
    ));

    return desc;
}

// ============================================================================
// Convolution Operations
// ============================================================================

void ROCmDNN::conv2d_forward(
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
) {
    // Create descriptors
    auto input_desc = create_tensor_desc(dtype, input_shape);
    auto weight_desc = create_tensor_desc(dtype, weight_shape);
    auto output_desc = create_tensor_desc(dtype, output_shape);

    // Create convolution descriptor
    miopenConvolutionDescriptor_t conv_desc;
    MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&conv_desc));

    MIOPEN_CHECK(miopenInitConvolutionDescriptor(
        conv_desc,
        miopenConvolution,
        params.padding[0], params.padding[1],
        params.stride[0], params.stride[1],
        params.dilation[0], params.dilation[1]
    ));

    if (params.groups > 1) {
        MIOPEN_CHECK(miopenSetConvolutionGroupCount(conv_desc, params.groups));
    }

    // Find best algorithm
    miopenConvAlgoPerf_t perf;
    int returned_algo_count;

    MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
        handle_,
        input_desc, input,
        weight_desc, weight,
        conv_desc,
        output_desc, output,
        1,  // request count
        &returned_algo_count,
        &perf,
        workspace,
        workspace_size,
        false  // exhaustive search
    ));

    // Execute convolution
    float alpha = 1.0f, beta = 0.0f;
    MIOPEN_CHECK(miopenConvolutionForward(
        handle_,
        &alpha,
        input_desc, input,
        weight_desc, weight,
        conv_desc,
        perf.fwd_algo,
        &beta,
        output_desc, output,
        workspace,
        workspace_size
    ));

    // Cleanup
    miopenDestroyTensorDescriptor(input_desc);
    miopenDestroyTensorDescriptor(weight_desc);
    miopenDestroyTensorDescriptor(output_desc);
    miopenDestroyConvolutionDescriptor(conv_desc);
}

size_t ROCmDNN::conv2d_forward_workspace_size(
    DType dtype,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& output_shape,
    const ConvParams& params
) {
    auto input_desc = create_tensor_desc(dtype, input_shape);
    auto weight_desc = create_tensor_desc(dtype, weight_shape);
    auto output_desc = create_tensor_desc(dtype, output_shape);

    miopenConvolutionDescriptor_t conv_desc;
    MIOPEN_CHECK(miopenCreateConvolutionDescriptor(&conv_desc));

    MIOPEN_CHECK(miopenInitConvolutionDescriptor(
        conv_desc,
        miopenConvolution,
        params.padding[0], params.padding[1],
        params.stride[0], params.stride[1],
        params.dilation[0], params.dilation[1]
    ));

    if (params.groups > 1) {
        MIOPEN_CHECK(miopenSetConvolutionGroupCount(conv_desc, params.groups));
    }

    size_t workspace_size = 0;
    MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(
        handle_,
        weight_desc,
        input_desc,
        conv_desc,
        output_desc,
        &workspace_size
    ));

    miopenDestroyTensorDescriptor(input_desc);
    miopenDestroyTensorDescriptor(weight_desc);
    miopenDestroyTensorDescriptor(output_desc);
    miopenDestroyConvolutionDescriptor(conv_desc);

    return workspace_size;
}

// ============================================================================
// Pooling Operations
// ============================================================================

void ROCmDNN::pooling2d_forward(
    ir::OpType pool_type,
    DType dtype,
    const std::vector<int64_t>& input_shape,
    const void* input,
    const std::vector<int64_t>& output_shape,
    void* output,
    const PoolParams& params
) {
    auto input_desc = create_tensor_desc(dtype, input_shape);
    auto output_desc = create_tensor_desc(dtype, output_shape);

    miopenPoolingDescriptor_t pool_desc;
    MIOPEN_CHECK(miopenCreatePoolingDescriptor(&pool_desc));

    miopenPoolingMode_t mode = to_miopen_pooling(pool_type);

    MIOPEN_CHECK(miopenSet2dPoolingDescriptor(
        pool_desc,
        mode,
        params.kernel_size[0], params.kernel_size[1],
        params.padding[0], params.padding[1],
        params.stride[0], params.stride[1]
    ));

    float alpha = 1.0f, beta = 0.0f;

    MIOPEN_CHECK(miopenPoolingForward(
        handle_,
        pool_desc,
        &alpha,
        input_desc, input,
        &beta,
        output_desc, output,
        false,    // do_backward (save indices)
        nullptr,  // workspace
        0         // workspace size
    ));

    miopenDestroyPoolingDescriptor(pool_desc);
    miopenDestroyTensorDescriptor(output_desc);
    miopenDestroyTensorDescriptor(input_desc);
}

miopenPoolingMode_t ROCmDNN::to_miopen_pooling(ir::OpType op) const {
    switch (op) {
        case ir::OpType::MAX_POOL1D:
        case ir::OpType::MAX_POOL2D:
        case ir::OpType::MAX_POOL3D:
            return miopenPoolingMax;
        case ir::OpType::AVG_POOL1D:
        case ir::OpType::AVG_POOL2D:
        case ir::OpType::AVG_POOL3D:
            return miopenPoolingAverage;
        default:
            throw std::runtime_error("Unsupported pooling type for MIOpen: " +
                                     ir::op_type_name(op));
    }
}

// ============================================================================
// Normalization Operations
// ============================================================================

void ROCmDNN::batchnorm_forward_training(
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
) {
    auto input_desc = create_tensor_desc(dtype, input_shape);

    // Derive batch norm tensor descriptor
    miopenTensorDescriptor_t bn_desc;
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&bn_desc));
    MIOPEN_CHECK(miopenDeriveBNTensorDescriptor(
        bn_desc, input_desc, miopenBNSpatial));

    float alpha = 1.0f, beta = 0.0f;

    // SECURITY NOTE: const_cast is required because MIOpen API does not properly
    // declare read-only parameters as const. According to MIOpen documentation,
    // scale and bias are only read during forward pass, not modified.
    // This const_cast is safe for these specific parameters.
    MIOPEN_CHECK(miopenBatchNormalizationForwardTraining(
        handle_,
        miopenBNSpatial,
        &alpha, &beta,
        input_desc, input,
        input_desc, output,
        bn_desc,
        const_cast<void*>(scale),   // Read-only: safe const_cast per MIOpen docs
        const_cast<void*>(bias),    // Read-only: safe const_cast per MIOpen docs
        params.momentum,
        running_mean,
        running_var,
        params.epsilon,
        saved_mean,
        saved_inv_var
    ));

    miopenDestroyTensorDescriptor(bn_desc);
    miopenDestroyTensorDescriptor(input_desc);
}

void ROCmDNN::batchnorm_forward_inference(
    DType dtype,
    const std::vector<int64_t>& input_shape,
    const void* input,
    void* output,
    const void* scale,
    const void* bias,
    const void* running_mean,
    const void* running_var,
    const BatchNormParams& params
) {
    auto input_desc = create_tensor_desc(dtype, input_shape);

    miopenTensorDescriptor_t bn_desc;
    MIOPEN_CHECK(miopenCreateTensorDescriptor(&bn_desc));
    MIOPEN_CHECK(miopenDeriveBNTensorDescriptor(
        bn_desc, input_desc, miopenBNSpatial));

    float alpha = 1.0f, beta = 0.0f;

    // SECURITY NOTE: const_cast is required because MIOpen API does not properly
    // declare read-only parameters as const. According to MIOpen documentation,
    // all parameters below are only read during inference, not modified.
    // This const_cast is safe for these specific parameters.
    MIOPEN_CHECK(miopenBatchNormalizationForwardInference(
        handle_,
        miopenBNSpatial,
        &alpha, &beta,
        input_desc, input,
        input_desc, output,
        bn_desc,
        const_cast<void*>(scale),        // Read-only: safe const_cast per MIOpen docs
        const_cast<void*>(bias),         // Read-only: safe const_cast per MIOpen docs
        const_cast<void*>(running_mean), // Read-only: safe const_cast per MIOpen docs
        const_cast<void*>(running_var),  // Read-only: safe const_cast per MIOpen docs
        params.epsilon
    ));

    miopenDestroyTensorDescriptor(bn_desc);
    miopenDestroyTensorDescriptor(input_desc);
}

// ============================================================================
// Activation Operations
// ============================================================================

void ROCmDNN::activation_forward(
    ir::OpType activation_type,
    DType dtype,
    const std::vector<int64_t>& shape,
    const void* input,
    void* output,
    float alpha_param
) {
    auto desc = create_tensor_desc(dtype, shape);

    miopenActivationDescriptor_t act_desc;
    MIOPEN_CHECK(miopenCreateActivationDescriptor(&act_desc));

    miopenActivationMode_t mode = to_miopen_activation(activation_type);
    MIOPEN_CHECK(miopenSetActivationDescriptor(
        act_desc,
        mode,
        alpha_param,  // alpha (for leaky ReLU)
        0.0,          // beta
        1.0           // gamma
    ));

    float alpha = 1.0f, beta = 0.0f;
    MIOPEN_CHECK(miopenActivationForward(
        handle_,
        act_desc,
        &alpha,
        desc, input,
        &beta,
        desc, output
    ));

    miopenDestroyActivationDescriptor(act_desc);
    miopenDestroyTensorDescriptor(desc);
}

miopenActivationMode_t ROCmDNN::to_miopen_activation(ir::OpType op) const {
    switch (op) {
        case ir::OpType::RELU:    return miopenActivationRELU;
        case ir::OpType::SIGMOID: return miopenActivationLOGISTIC;
        case ir::OpType::TANH:    return miopenActivationTANH;
        // Note: GELU and SILU are not directly supported by MIOpen
        // Use custom kernels for these
        default:
            throw std::runtime_error("Unsupported activation for MIOpen: " +
                                     ir::op_type_name(op) +
                                     ". Use custom kernel instead.");
    }
}

// ============================================================================
// Softmax Operations
// ============================================================================

void ROCmDNN::softmax_forward(
    DType dtype,
    const std::vector<int64_t>& shape,
    const void* input,
    void* output,
    int axis,
    bool log_softmax
) {
    (void)axis;  // MIOpen uses fixed axis behavior

    auto desc = create_tensor_desc(dtype, shape);

    float alpha = 1.0f, beta = 0.0f;

    miopenSoftmaxAlgorithm_t algo = log_softmax ?
        miopenSoftmaxLog : miopenSoftmaxAccurate;

    // MIOpen softmax operates on channel dimension for NCHW
    MIOPEN_CHECK(miopenSoftmaxForward_V2(
        handle_,
        &alpha,
        desc, input,
        &beta,
        desc, output,
        algo,
        miopenSoftmaxModeChannel
    ));

    miopenDestroyTensorDescriptor(desc);
}

// ============================================================================
// Tensor Operations
// ============================================================================

void ROCmDNN::tensor_op(
    ir::OpType op_type,
    DType dtype,
    const std::vector<int64_t>& shape,
    const void* A,
    const void* B,
    void* C,
    float alpha1,
    float alpha2,
    float beta
) {
    auto desc = create_tensor_desc(dtype, shape);

    miopenTensorOp_t op;
    switch (op_type) {
        case ir::OpType::ADD: op = miopenTensorOpAdd; break;
        case ir::OpType::MUL: op = miopenTensorOpMul; break;
        case ir::OpType::MIN: op = miopenTensorOpMin; break;
        case ir::OpType::MAX: op = miopenTensorOpMax; break;
        default:
            miopenDestroyTensorDescriptor(desc);
            throw std::runtime_error("Unsupported tensor op for MIOpen: " +
                                     ir::op_type_name(op_type));
    }

    MIOPEN_CHECK(miopenOpTensor(
        handle_,
        op,
        &alpha1, desc, A,
        &alpha2, desc, B,
        &beta, desc, C
    ));

    miopenDestroyTensorDescriptor(desc);
}

// ============================================================================
// Reduction Operations
// ============================================================================

miopenReduceTensorOp_t ROCmDNN::to_miopen_reduce(ir::OpType op) const {
    switch (op) {
        case ir::OpType::SUM:  return MIOPEN_REDUCE_TENSOR_ADD;
        case ir::OpType::MEAN: return MIOPEN_REDUCE_TENSOR_AVG;
        case ir::OpType::MAX:  return MIOPEN_REDUCE_TENSOR_MAX;
        case ir::OpType::MIN:  return MIOPEN_REDUCE_TENSOR_MIN;
        default:
            throw std::runtime_error("Unsupported reduction for MIOpen: " +
                                     ir::op_type_name(op));
    }
}

void ROCmDNN::reduce(
    ir::OpType reduce_type,
    DType dtype,
    const std::vector<int64_t>& input_shape,
    const void* input,
    const std::vector<int64_t>& output_shape,
    void* output,
    const std::vector<int>& reduce_dims,
    void* workspace,
    size_t workspace_size
) {
    (void)reduce_dims;  // MIOpen infers from tensor shapes

    auto input_desc = create_tensor_desc(dtype, input_shape);
    auto output_desc = create_tensor_desc(dtype, output_shape);

    miopenReduceTensorDescriptor_t reduce_desc;
    MIOPEN_CHECK(miopenCreateReduceTensorDescriptor(&reduce_desc));

    MIOPEN_CHECK(miopenSetReduceTensorDescriptor(
        reduce_desc,
        to_miopen_reduce(reduce_type),
        to_miopen_type(dtype),
        MIOPEN_PROPAGATE_NAN,
        MIOPEN_REDUCE_TENSOR_NO_INDICES,
        MIOPEN_32BIT_INDICES
    ));

    float alpha = 1.0f, beta = 0.0f;

    MIOPEN_CHECK(miopenReduceTensor(
        handle_,
        reduce_desc,
        nullptr, 0,  // indices
        workspace, workspace_size,
        &alpha,
        input_desc, input,
        &beta,
        output_desc, output
    ));

    miopenDestroyReduceTensorDescriptor(reduce_desc);
    miopenDestroyTensorDescriptor(output_desc);
    miopenDestroyTensorDescriptor(input_desc);
}

size_t ROCmDNN::reduce_workspace_size(
    ir::OpType reduce_type,
    DType dtype,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& output_shape
) {
    auto input_desc = create_tensor_desc(dtype, input_shape);
    auto output_desc = create_tensor_desc(dtype, output_shape);

    miopenReduceTensorDescriptor_t reduce_desc;
    MIOPEN_CHECK(miopenCreateReduceTensorDescriptor(&reduce_desc));

    MIOPEN_CHECK(miopenSetReduceTensorDescriptor(
        reduce_desc,
        to_miopen_reduce(reduce_type),
        to_miopen_type(dtype),
        MIOPEN_PROPAGATE_NAN,
        MIOPEN_REDUCE_TENSOR_NO_INDICES,
        MIOPEN_32BIT_INDICES
    ));

    size_t workspace_size = 0;
    MIOPEN_CHECK(miopenGetReductionWorkspaceSize(
        handle_,
        reduce_desc,
        input_desc,
        output_desc,
        &workspace_size
    ));

    miopenDestroyReduceTensorDescriptor(reduce_desc);
    miopenDestroyTensorDescriptor(output_desc);
    miopenDestroyTensorDescriptor(input_desc);

    return workspace_size;
}

// Global singleton
ROCmDNN& get_dnn() {
    static ROCmDNN instance;
    return instance;
}

}  // namespace pyflame::backend::rocm

#endif  // PYFLAME_HAS_ROCM
