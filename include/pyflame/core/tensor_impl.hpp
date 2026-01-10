#pragma once

#include <memory>
#include <vector>
#include <random>
#include <cstring>

#include "pyflame/core/dtype.hpp"
#include "pyflame/core/layout.hpp"
#include "pyflame/core/allocator.hpp"
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
    static std::shared_ptr<TensorImpl> from_data(
        const void* data,
        const std::vector<int64_t>& shape,
        DType dtype,
        MeshLayout layout
    ) {
        auto impl = std::make_shared<TensorImpl>();

        // Get or create graph
        impl->graph_ = get_current_graph();

        // Create tensor spec
        ir::TensorSpec spec(shape, dtype, layout);

        // Create constant node
        impl->node_ = impl->graph_->create_constant(spec, data);

        // Store data immediately (constants are always materialized)
        size_t bytes = spec.size_bytes();
        impl->data_ = std::shared_ptr<uint8_t>(
            static_cast<uint8_t*>(Allocator::allocate(bytes)),
            [](uint8_t* p) { Allocator::deallocate(p); }
        );
        std::memcpy(impl->data_.get(), data, bytes);

        return impl;
    }

    /// Create zeros tensor
    static std::shared_ptr<TensorImpl> zeros(
        const std::vector<int64_t>& shape,
        DType dtype,
        MeshLayout layout
    ) {
        ir::TensorSpec spec(shape, dtype, layout);
        size_t bytes = spec.size_bytes();

        // Allocate and zero-fill
        std::shared_ptr<uint8_t> data(
            static_cast<uint8_t*>(Allocator::allocate_zeroed(bytes)),
            [](uint8_t* p) { Allocator::deallocate(p); }
        );

        return from_data(data.get(), shape, dtype, layout);
    }

    /// Create ones tensor
    static std::shared_ptr<TensorImpl> ones(
        const std::vector<int64_t>& shape,
        DType dtype,
        MeshLayout layout
    ) {
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

        return from_data(data.get(), shape, dtype, layout);
    }

    /// Create tensor filled with a value
    static std::shared_ptr<TensorImpl> full(
        const std::vector<int64_t>& shape,
        float value,
        DType dtype,
        MeshLayout layout
    ) {
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

        return from_data(data.get(), shape, dtype, layout);
    }

    /// Create tensor with random normal values
    static std::shared_ptr<TensorImpl> randn(
        const std::vector<int64_t>& shape,
        DType dtype,
        MeshLayout layout
    ) {
        ir::TensorSpec spec(shape, dtype, layout);
        size_t bytes = spec.size_bytes();
        int64_t numel = spec.numel();

        std::shared_ptr<uint8_t> data(
            static_cast<uint8_t*>(Allocator::allocate(bytes)),
            [](uint8_t* p) { Allocator::deallocate(p); }
        );

        static thread_local std::mt19937 gen(std::random_device{}());
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

        return from_data(data.get(), shape, dtype, layout);
    }

    /// Create tensor with uniform random values in [0, 1)
    static std::shared_ptr<TensorImpl> rand(
        const std::vector<int64_t>& shape,
        DType dtype,
        MeshLayout layout
    ) {
        ir::TensorSpec spec(shape, dtype, layout);
        size_t bytes = spec.size_bytes();
        int64_t numel = spec.numel();

        std::shared_ptr<uint8_t> data(
            static_cast<uint8_t*>(Allocator::allocate(bytes)),
            [](uint8_t* p) { Allocator::deallocate(p); }
        );

        static thread_local std::mt19937 gen(std::random_device{}());
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

        return from_data(data.get(), shape, dtype, layout);
    }

    /// Create tensor with values from start to end
    static std::shared_ptr<TensorImpl> arange(
        int64_t start,
        int64_t end,
        int64_t step,
        DType dtype
    ) {
        int64_t numel = (end - start + step - 1) / step;
        if (numel <= 0) numel = 0;

        std::vector<int64_t> shape = {numel};
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

        return from_data(data.get(), shape, dtype, MeshLayout::SinglePE());
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
    std::shared_ptr<ir::Graph> graph_;
    std::shared_ptr<ir::Node> node_;
    std::shared_ptr<uint8_t> data_;  // Materialized data (may be null)
};

}  // namespace pyflame
