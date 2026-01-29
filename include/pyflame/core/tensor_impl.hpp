#pragma once

#include <memory>
#include <vector>
#include <random>
#include <cstring>
#include <mutex>

#include "pyflame/core/dtype.hpp"
#include "pyflame/core/layout.hpp"
#include "pyflame/core/allocator.hpp"
#include "pyflame/core/safe_math.hpp"
#include "pyflame/ir/graph.hpp"
#include "pyflame/ir/node.hpp"
#include "pyflame/ir/shape_inference.hpp"

namespace pyflame {

class Tensor;

/// Internal implementation of Tensor
/// Each Tensor holds a shared_ptr to a TensorImpl, which in turn references
/// a node in a computation graph. This enables lazy evaluation.
class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
public:
    /// Create from an existing graph node
    static std::shared_ptr<TensorImpl> from_node(
        std::shared_ptr<ir::Graph> graph,
        std::shared_ptr<ir::Node> node
    ) {
        auto impl = std::make_shared<TensorImpl>();
        impl->graph_ = graph;
        impl->node_ = node;
        return impl;
    }

    /// Create from data (creates a constant node)
    /// @param data Pointer to source data buffer
    /// @param shape Shape of the tensor (validated for overflow)
    /// @param dtype Data type
    /// @param layout Mesh layout
    /// @param data_size Size of data buffer (REQUIRED for security validation)
    ///
    /// SECURITY: This function requires the caller to specify the actual data
    /// buffer size to prevent buffer overread vulnerabilities. The data_size
    /// parameter is validated against the expected size computed from shape.
    static std::shared_ptr<TensorImpl> from_data(
        const void* data,
        const std::vector<int64_t>& shape,
        DType dtype,
        MeshLayout layout,
        size_t data_size
    ) {
        if (!data) {
            throw std::invalid_argument("Data pointer cannot be null");
        }

        // Security: Require explicit buffer size validation
        // data_size=0 is no longer accepted (was legacy behavior that could
        // lead to buffer overreads if caller miscalculates shape)
        if (data_size == 0) {
            throw std::invalid_argument(
                "data_size must be provided for buffer validation. "
                "This is required for security to prevent buffer overread."
            );
        }

        // Validate shape to prevent overflow attacks
        validate_shape(shape);

        auto impl = std::make_shared<TensorImpl>();

        // Get or create graph
        impl->graph_ = get_current_graph();

        // Create tensor spec (numel() and size_bytes() now have overflow checking)
        ir::TensorSpec spec(shape, dtype, layout);

        // Security: Validate buffer size matches expected
        size_t expected_bytes = spec.size_bytes();
        if (data_size < expected_bytes) {
            throw std::invalid_argument(
                "Data buffer too small: provided " + std::to_string(data_size) +
                " bytes, need " + std::to_string(expected_bytes) + " bytes. "
                "This could indicate a buffer overread vulnerability.");
        }

        // Create constant node
        impl->node_ = impl->graph_->create_constant(spec, data);

        // Store data immediately (constants are always materialized)
        // Security: Only copy expected_bytes (not data_size) to prevent
        // reading beyond shape-defined bounds
        impl->data_ = std::shared_ptr<uint8_t>(
            static_cast<uint8_t*>(Allocator::allocate(expected_bytes)),
            [](uint8_t* p) { Allocator::deallocate(p); }
        );
        std::memcpy(impl->data_.get(), data, expected_bytes);

        return impl;
    }

    /// Create zeros tensor
    static std::shared_ptr<TensorImpl> zeros(
        const std::vector<int64_t>& shape,
        DType dtype,
        MeshLayout layout
    ) {
        // Validate shape to prevent overflow
        validate_shape(shape);

        ir::TensorSpec spec(shape, dtype, layout);
        size_t bytes = spec.size_bytes();

        // Allocate and zero-fill
        std::shared_ptr<uint8_t> data(
            static_cast<uint8_t*>(Allocator::allocate_zeroed(bytes)),
            [](uint8_t* p) { Allocator::deallocate(p); }
        );

        return from_data(data.get(), shape, dtype, layout, bytes);
    }

    /// Create ones tensor
    static std::shared_ptr<TensorImpl> ones(
        const std::vector<int64_t>& shape,
        DType dtype,
        MeshLayout layout
    ) {
        // Validate shape to prevent overflow
        validate_shape(shape);

        ir::TensorSpec spec(shape, dtype, layout);
        size_t bytes = spec.size_bytes();
        int64_t numel = spec.numel();

        std::shared_ptr<uint8_t> data(
            static_cast<uint8_t*>(Allocator::allocate(bytes)),
            [](uint8_t* p) { Allocator::deallocate(p); }
        );

        // Fill with ones based on dtype
        switch (dtype) {
            case DType::Float32: {
                float* ptr = reinterpret_cast<float*>(data.get());
                std::fill(ptr, ptr + numel, 1.0f);
                break;
            }
            case DType::Int32: {
                int32_t* ptr = reinterpret_cast<int32_t*>(data.get());
                std::fill(ptr, ptr + numel, 1);
                break;
            }
            default:
                throw std::runtime_error("Unsupported dtype for ones");
        }

        return from_data(data.get(), shape, dtype, layout, bytes);
    }

    /// Create tensor filled with a value
    static std::shared_ptr<TensorImpl> full(
        const std::vector<int64_t>& shape,
        float value,
        DType dtype,
        MeshLayout layout
    ) {
        // Validate shape to prevent overflow
        validate_shape(shape);

        ir::TensorSpec spec(shape, dtype, layout);
        size_t bytes = spec.size_bytes();
        int64_t numel = spec.numel();

        std::shared_ptr<uint8_t> data(
            static_cast<uint8_t*>(Allocator::allocate(bytes)),
            [](uint8_t* p) { Allocator::deallocate(p); }
        );

        switch (dtype) {
            case DType::Float32: {
                float* ptr = reinterpret_cast<float*>(data.get());
                std::fill(ptr, ptr + numel, value);
                break;
            }
            case DType::Int32: {
                int32_t* ptr = reinterpret_cast<int32_t*>(data.get());
                std::fill(ptr, ptr + numel, static_cast<int32_t>(value));
                break;
            }
            default:
                throw std::runtime_error("Unsupported dtype for full");
        }

        return from_data(data.get(), shape, dtype, layout, bytes);
    }

    /// Get thread-safe random generator
    /// Uses lazy initialization with unique_ptr for controlled destruction
    static std::mt19937& get_random_generator() {
        // Use unique_ptr for controlled destruction order
        static thread_local std::unique_ptr<std::mt19937> gen_ptr;
        static thread_local std::once_flag init_flag;

        // Lazy initialization
        if (!gen_ptr) {
            std::random_device rd;
            gen_ptr = std::make_unique<std::mt19937>(rd());
        }
        return *gen_ptr;
    }

    /// Create tensor with random normal values
    static std::shared_ptr<TensorImpl> randn(
        const std::vector<int64_t>& shape,
        DType dtype,
        MeshLayout layout
    ) {
        // Validate shape to prevent overflow
        validate_shape(shape);

        ir::TensorSpec spec(shape, dtype, layout);
        size_t bytes = spec.size_bytes();
        int64_t numel = spec.numel();

        std::shared_ptr<uint8_t> data(
            static_cast<uint8_t*>(Allocator::allocate(bytes)),
            [](uint8_t* p) { Allocator::deallocate(p); }
        );

        auto& gen = get_random_generator();
        std::normal_distribution<float> dist(0.0f, 1.0f);

        switch (dtype) {
            case DType::Float32: {
                float* ptr = reinterpret_cast<float*>(data.get());
                for (int64_t i = 0; i < numel; ++i) {
                    ptr[i] = dist(gen);
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported dtype for randn");
        }

        return from_data(data.get(), shape, dtype, layout, bytes);
    }

    /// Create tensor with uniform random values in [0, 1)
    static std::shared_ptr<TensorImpl> rand(
        const std::vector<int64_t>& shape,
        DType dtype,
        MeshLayout layout
    ) {
        // Validate shape to prevent overflow
        validate_shape(shape);

        ir::TensorSpec spec(shape, dtype, layout);
        size_t bytes = spec.size_bytes();
        int64_t numel = spec.numel();

        std::shared_ptr<uint8_t> data(
            static_cast<uint8_t*>(Allocator::allocate(bytes)),
            [](uint8_t* p) { Allocator::deallocate(p); }
        );

        auto& gen = get_random_generator();
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        switch (dtype) {
            case DType::Float32: {
                float* ptr = reinterpret_cast<float*>(data.get());
                for (int64_t i = 0; i < numel; ++i) {
                    ptr[i] = dist(gen);
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported dtype for rand");
        }

        return from_data(data.get(), shape, dtype, layout, bytes);
    }

    /// Create tensor with values from start to end
    static std::shared_ptr<TensorImpl> arange(
        int64_t start,
        int64_t end,
        int64_t step,
        DType dtype
    ) {
        if (step == 0) {
            throw std::invalid_argument("arange step cannot be zero");
        }

        int64_t numel = (end - start + step - 1) / step;
        if (numel <= 0) numel = 0;

        // Validate the resulting shape
        std::vector<int64_t> shape = {numel};
        validate_shape(shape);

        ir::TensorSpec spec(shape, dtype, MeshLayout::SinglePE());
        size_t bytes = spec.size_bytes();

        std::shared_ptr<uint8_t> data(
            static_cast<uint8_t*>(Allocator::allocate(bytes)),
            [](uint8_t* p) { Allocator::deallocate(p); }
        );

        switch (dtype) {
            case DType::Float32: {
                float* ptr = reinterpret_cast<float*>(data.get());
                for (int64_t i = 0; i < numel; ++i) {
                    ptr[i] = static_cast<float>(start + i * step);
                }
                break;
            }
            case DType::Int32: {
                int32_t* ptr = reinterpret_cast<int32_t*>(data.get());
                for (int64_t i = 0; i < numel; ++i) {
                    ptr[i] = static_cast<int32_t>(start + i * step);
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported dtype for arange");
        }

        return from_data(data.get(), shape, dtype, MeshLayout::SinglePE(), bytes);
    }

    // Accessors
    std::shared_ptr<ir::Graph> graph() const { return graph_; }
    std::shared_ptr<ir::Node> node() const { return node_; }

    const std::vector<int64_t>& shape() const { return node_->shape(); }
    DType dtype() const { return node_->dtype(); }
    const MeshLayout& layout() const { return node_->layout(); }
    int64_t numel() const { return node_->numel(); }
    int ndim() const { return node_->ndim(); }

    bool is_evaluated() const { return data_ != nullptr || node_->is_evaluated(); }

    /// Force evaluation and return data pointer
    void* materialize();

    /// Apply a unary operation
    std::shared_ptr<TensorImpl> apply_unary(ir::OpType op) const;

    /// Apply a binary operation
    std::shared_ptr<TensorImpl> apply_binary(ir::OpType op, std::shared_ptr<TensorImpl> other) const;

    /// Apply a reduction operation
    std::shared_ptr<TensorImpl> apply_reduction(
        ir::OpType op,
        std::optional<int> dim,
        bool keepdim
    ) const;

    /// Apply reshape
    std::shared_ptr<TensorImpl> apply_reshape(const std::vector<int64_t>& new_shape) const;

    /// Apply transpose
    std::shared_ptr<TensorImpl> apply_transpose(int dim0, int dim1) const;

    /// Data access (only valid after materialization)
    void* data() { return data_.get(); }
    const void* data() const { return data_.get(); }

    /// Get or create the thread-local graph
    static std::shared_ptr<ir::Graph> get_current_graph();

    /// Set the current graph (for testing/advanced use)
    static void set_current_graph(std::shared_ptr<ir::Graph> graph);

private:
    /// CPU reference implementation for operations
    void execute_cpu();

    std::shared_ptr<ir::Graph> graph_;
    std::shared_ptr<ir::Node> node_;
    std::shared_ptr<uint8_t> data_;  // Materialized data (may be null)
};

}  // namespace pyflame
